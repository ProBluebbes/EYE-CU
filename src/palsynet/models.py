"""Model registry.

Every model takes a batch of segments (B, T, C, H, W) and returns logits (B, num_classes).
Register a builder with @register("name") and select it with `model.name` in the config.
Builders receive `num_classes`, `data_cfg` (the `data:` section, for input shape) and every
other key under `model:` as keyword arguments.

Registered:
  r3d_18, mc3_18, r2plus1d_18   3D CNNs (torchvision, Kinetics-400 pretrained)
  vivit                         ViViT, joint spatio-temporal attention (Model 1), optional ViT-B/16 init
  frame_gru                     2D ResNet-18 per frame + GRU
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def build_model(cfg_model: dict, data_cfg: dict | None = None, num_classes: int = 2) -> nn.Module:
    name = cfg_model["name"]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_REGISTRY)}")
    kwargs = {k: v for k, v in cfg_model.items() if k != "name"}
    return _REGISTRY[name](num_classes=num_classes, data_cfg=data_cfg or {}, **kwargs)


def segment_frames(data_cfg: dict) -> int:
    return max(1, round(data_cfg["segment_seconds"] * data_cfg["sample_fps"]))


# ----------------------------------------------------------------------------- 3D CNNs
class Video3DCNN(nn.Module):
    """torchvision 3D ResNet; accepts (B, T, C, H, W) and permutes to (B, C, T, H, W)."""

    def __init__(self, arch: str, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3, freeze: list[str] | None = None):
        """freeze: blocks to keep fixed, from "stem", "layer1", "layer2", "layer3", "layer4".
        Frozen blocks get no gradients (cheaper backward) and keep their BatchNorm statistics."""
        super().__init__()
        from torchvision.models import video as tvv

        fns = {
            "r3d_18": (tvv.r3d_18, tvv.R3D_18_Weights.KINETICS400_V1),
            "mc3_18": (tvv.mc3_18, tvv.MC3_18_Weights.KINETICS400_V1),
            "r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights.KINETICS400_V1),
        }
        fn, weights = fns[arch]
        self.net = fn(weights=weights if pretrained else None)
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.net.fc.in_features, num_classes))
        self.frozen = [getattr(self.net, name) for name in (freeze or [])]
        for block in self.frozen:
            block.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for block in self.frozen:
            block.eval()  # keep pretrained BatchNorm running stats in frozen blocks
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1, 3, 4))


for _arch in ("r3d_18", "mc3_18", "r2plus1d_18"):
    register(_arch)(lambda num_classes=2, data_cfg=None, _a=_arch, **kw: Video3DCNN(_a, num_classes, **kw))


# ----------------------------------------------------------------------------- ViViT
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        # Memory-efficient fused attention: every token attends to every token (no factorisation).
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(b, n, d)))


class Block(nn.Module):
    """Pre-norm transformer block (as in ViT / ViViT)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))
        self.drop_path = drop_path

    def _dp(self, x: torch.Tensor) -> torch.Tensor:  # stochastic depth
        if not self.training or self.drop_path == 0.0:
            return x
        keep = 1 - self.drop_path
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._dp(self.attn(self.norm1(x)))
        return x + self._dp(self.mlp(self.norm2(x)))


class ViViT(nn.Module):
    """ViViT Model 1 (Arnab et al., 2021): tubelet embedding + one transformer with joint
    spatio-temporal self-attention over all nt*nh*nw tokens, CLS token for classification.

    Positional embedding = learned spatial (nh*nw) + learned temporal (nt), so ViT-B/16 spatial
    embeddings can be reused; attention itself is full, not factorised.
    Token count = (T/tubelet_t) * (H/patch)^2 + 1; attention cost grows with its square.
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_frames: int = 32,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_t: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        if num_frames % tubelet_t or img_size % patch_size:
            raise ValueError(f"num_frames ({num_frames}) must divide by tubelet_t ({tubelet_t}) and img_size ({img_size}) by patch_size ({patch_size})")
        self.nt, self.ns = num_frames // tubelet_t, (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv3d(3, embed_dim, (tubelet_t, patch_size, patch_size), (tubelet_t, patch_size, patch_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_spatial = nn.Parameter(torch.zeros(1, self.ns, embed_dim))
        self.pos_temporal = nn.Parameter(torch.zeros(1, self.nt, embed_dim))
        self.pos_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(Block(embed_dim, heads, mlp_ratio, dropout, attn_dropout, dpr[i]) for i in range(depth))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(embed_dim, num_classes))
        self._init_weights()

    def _init_weights(self) -> None:
        for p in (self.cls_token, self.pos_spatial, self.pos_temporal, self.pos_cls):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C, H, W)
        x = self.patch_embed(x.permute(0, 2, 1, 3, 4))  # B, D, nt, nh, nw
        b, d = x.shape[:2]
        x = x.flatten(3).permute(0, 2, 3, 1)  # B, nt, ns, D
        x = x + self.pos_spatial.unsqueeze(1) + self.pos_temporal.unsqueeze(2)
        x = x.reshape(b, -1, d)  # B, nt*ns, D  -- all space-time tokens in one sequence
        x = torch.cat([self.cls_token.expand(b, -1, -1) + self.pos_cls, x], dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))

    @torch.no_grad()
    def load_vit_b16(self, tubelet_init: str = "central") -> None:
        """Initialise from torchvision's ImageNet ViT-B/16 (paper sec. 3.4).

        tubelet_init: "central" puts the 2D patch filter on the middle frame of the tubelet
        (zeros elsewhere, so it starts as a per-frame ViT); "inflate" repeats it over time / t.
        """
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if self.patch_embed.out_channels != 768 or len(self.blocks) > len(vit.encoder.layers):
            raise ValueError("ViT-B/16 init needs embed_dim=768, heads=12, depth<=12, patch_size=16")
        w2d = vit.conv_proj.weight  # 768, 3, 16, 16
        t = self.patch_embed.kernel_size[0]
        w3d = torch.zeros_like(self.patch_embed.weight)
        if tubelet_init == "central":
            w3d[:, :, t // 2] = w2d
        else:
            w3d[:] = w2d.unsqueeze(2) / t
        self.patch_embed.weight.copy_(w3d)
        self.patch_embed.bias.copy_(vit.conv_proj.bias)

        pos = vit.encoder.pos_embedding  # 1, 1 + 14*14, 768
        self.pos_cls.copy_(pos[:, :1])
        self.cls_token.copy_(vit.class_token)
        grid_src, grid_dst = int(math.sqrt(pos.shape[1] - 1)), int(math.sqrt(self.ns))
        sp = pos[:, 1:].reshape(1, grid_src, grid_src, -1).permute(0, 3, 1, 2)
        if grid_src != grid_dst:
            sp = F.interpolate(sp, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
        self.pos_spatial.copy_(sp.permute(0, 2, 3, 1).reshape(1, self.ns, -1))
        self.pos_temporal.zero_()

        for blk, layer in zip(self.blocks, vit.encoder.layers):
            blk.norm1.load_state_dict(layer.ln_1.state_dict())
            blk.norm2.load_state_dict(layer.ln_2.state_dict())
            blk.attn.qkv.weight.copy_(layer.self_attention.in_proj_weight)
            blk.attn.qkv.bias.copy_(layer.self_attention.in_proj_bias)
            blk.attn.proj.load_state_dict(layer.self_attention.out_proj.state_dict())
            blk.mlp[0].load_state_dict(layer.mlp[0].state_dict())
            blk.mlp[3].load_state_dict(layer.mlp[3].state_dict())
        if len(self.blocks) == len(vit.encoder.layers):
            self.norm.load_state_dict(vit.encoder.ln.state_dict())


@register("vivit")
def _vivit(num_classes=2, data_cfg=None, pretrained: str | bool | None = None, tubelet_init: str = "central", **kw) -> nn.Module:
    data_cfg = data_cfg or {}
    kw.setdefault("num_frames", segment_frames(data_cfg))
    kw.setdefault("img_size", data_cfg.get("size", 224))
    m = ViViT(num_classes=num_classes, **kw)
    if pretrained in ("vit_b_16", True):
        m.load_vit_b16(tubelet_init)
    return m


# ----------------------------------------------------------------------------- 2D CNN + GRU
class FrameGRU(nn.Module):
    """2D ResNet-18 per frame -> bidirectional GRU over time -> linear."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        enc.fc = nn.Identity()
        self.encoder = enc
        self.gru = nn.GRU(512, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C, H, W)
        b, t = x.shape[:2]
        out, _ = self.gru(self.encoder(x.flatten(0, 1)).view(b, t, -1))
        return self.head(out.mean(1))


@register("frame_gru")
def _frame_gru(num_classes=2, data_cfg=None, **kw) -> nn.Module:
    return FrameGRU(num_classes, **kw)


# ----------------------------------------------------------------------------- landmarks only
class LandmarkTCN(nn.Module):
    """Face-landmark sequence (B, T, N, 2) in eye units -> per-frame features -> dilated temporal
    convolutions -> mean+max pooling over time -> linear. No pixels, so no lighting/camera cues.

    Per-frame feature groups (choose with `features`):
      pos       coordinates of the N points (face shape; also identity, so easiest to overfit)
      asym      each point minus the mirror image of its partner: zero for a perfectly symmetric
                face, so it measures left/right asymmetry directly
      vel       frame-to-frame motion of every point
      asym_vel  frame-to-frame change of the asymmetry (one side moving, the other not)
    """

    GROUPS = ("pos", "asym", "vel", "asym_vel")

    def __init__(self, num_classes: int = 2, features=("asym", "vel", "asym_vel"), hidden: int = 128,
                 layers: int = 3, kernel: int = 5, dropout: float = 0.3, coord_dims: int = 2):
        super().__init__()
        from .landmarks import MIRROR, NUM_POINTS

        bad = set(features) - set(self.GROUPS)
        if bad:
            raise ValueError(f"unknown landmark features {bad}; choose from {self.GROUPS}")
        self.features = tuple(features)
        self.register_buffer("mirror", torch.tensor(MIRROR), persistent=False)
        in_dim = len(self.features) * NUM_POINTS * coord_dims  # 2 = (x, y); 3 = frontalised (x, y, z)
        self.norm = nn.BatchNorm1d(in_dim)  # the groups have very different scales
        self.proj = nn.Sequential(nn.Conv1d(in_dim, hidden, 1), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=(kernel // 2) * 2**i, dilation=2**i),
                nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            )
            for i in range(layers)
        )
        self.head = nn.Linear(2 * hidden, num_classes)

    def frame_features(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, N, 2) -> (B, T, F)
        reflected = x[:, :, self.mirror] * x.new_tensor([-1.0] + [1.0] * (x.shape[-1] - 1))
        asym = x - reflected
        vel = torch.diff(x, dim=1, prepend=x[:, :1])
        asym_vel = torch.diff(asym, dim=1, prepend=asym[:, :1])
        groups = {"pos": x, "asym": asym, "vel": vel, "asym_vel": asym_vel}
        return torch.cat([groups[k].flatten(2) for k in self.features], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm(self.frame_features(x).transpose(1, 2)))  # (B, hidden, T)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


@register("landmark_tcn")
def _landmark_tcn(num_classes=2, data_cfg=None, pretrained=None, **kw) -> nn.Module:
    return LandmarkTCN(num_classes, coord_dims=_coord_dims(data_cfg), **kw)


def _coord_dims(data_cfg) -> int:
    return 3 if (data_cfg or {}).get("landmark_mode", "2d") == "3d" else 2


class LandmarkCRNN(LandmarkTCN):
    """Landmark CNN + RNN: the same per-frame features and temporal convolutions as LandmarkTCN
    (short movements: a smile starting, a blink), then a bidirectional GRU over the whole segment
    (links events seconds apart), mean+max pooled over time -> linear."""

    def __init__(self, num_classes: int = 2, features=("asym", "vel", "asym_vel"), hidden: int = 128,
                 layers: int = 2, kernel: int = 5, rnn_hidden: int = 128, rnn_layers: int = 1, dropout: float = 0.3,
                 coord_dims: int = 2):
        super().__init__(num_classes, features, hidden, layers, kernel, dropout, coord_dims)  # layers = temporal conv layers
        self.rnn = nn.GRU(hidden, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        self.rnn_drop = nn.Dropout(dropout)
        self.head = nn.Linear(4 * rnn_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm(self.frame_features(x).transpose(1, 2)))  # (B, hidden, T)
        for blk in self.blocks:
            h = h + blk(h)
        out, _ = self.rnn(h.transpose(1, 2))  # (B, T, 2 * rnn_hidden)
        out = self.rnn_drop(out)
        return self.head(torch.cat([out.mean(1), out.amax(1)], dim=-1))


@register("landmark_crnn")
def _landmark_crnn(num_classes=2, data_cfg=None, pretrained=None, **kw) -> nn.Module:
    return LandmarkCRNN(num_classes, coord_dims=_coord_dims(data_cfg), **kw)


class LandmarkSeqGRU(LandmarkTCN):
    """Memory across windows: every 2 s window is encoded by the LandmarkTCN body into one vector, then a
    one-directional (causal) GRU runs over the last K window vectors, so the verdict for the current window
    can use what the face did in the previous seconds (e.g. the person's usual resting face).

    Input (B, K, T, N, D): K consecutive windows, oldest first. Sequences shorter than K are left-padded with
    NaN windows (start of a video / of a live session); those are skipped. A plain (B, T, N, D) window is
    treated as K = 1. Output: logits for the newest window, from [GRU state, newest window's own vector]."""

    def __init__(self, num_classes: int = 2, features=("asym", "vel", "asym_vel"), hidden: int = 128,
                 layers: int = 3, kernel: int = 5, dropout: float = 0.3, coord_dims: int = 3,
                 gru_hidden: int = 128, gru_layers: int = 1):
        super().__init__(num_classes, features, hidden, layers, kernel, dropout, coord_dims)
        self.gru = nn.GRU(2 * hidden, gru_hidden, num_layers=gru_layers, batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.seq_drop = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden + 2 * hidden, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, N, D) windows -> (B, 2 * hidden) window vectors."""
        h = self.proj(self.norm(self.frame_features(x).transpose(1, 2)))
        for blk in self.blocks:
            h = h + blk(h)
        return torch.cat([h.mean(-1), h.amax(-1)], dim=-1)

    def classify(self, emb: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """emb (B, K, E) right-padded window vectors (valid ones first), lengths (B,) -> logits of the last valid."""
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        last = emb[torch.arange(len(emb), device=emb.device), lengths - 1]
        return self.head(self.seq_drop(torch.cat([h_n[-1], last], dim=-1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, K = x.shape[:2]
        valid = ~torch.isnan(x[:, :, 0, 0, 0])  # (B, K); padding is on the left
        lengths = valid.sum(1)
        emb_valid = self.encode(x[valid])  # only real windows go through the encoder
        # move each sequence's valid windows to the front (right padding) for pack_padded_sequence
        order = torch.argsort((~valid).int(), dim=1, stable=True)  # valid first, original order kept
        emb = x.new_zeros(B, K, emb_valid.shape[-1])
        emb[valid] = emb_valid
        emb = emb.gather(1, order[..., None].expand(-1, -1, emb.shape[-1]))
        return self.classify(emb, lengths)


@register("landmark_seq_gru")
def _landmark_seq_gru(num_classes=2, data_cfg=None, pretrained=None, **kw) -> nn.Module:
    return LandmarkSeqGRU(num_classes, coord_dims=_coord_dims(data_cfg), **kw)


# ----------------------------------------------------------------------------- spatial landmark models
# Both use the fixed face topology from scripts/build_face_topology.py (src/palsynet/face_topology.npz):
# a symmetric mean-face layout of the 152 points and graph edges between them.

def _own_kwargs(cls, kw: dict) -> dict:
    """Keep only the settings `cls` accepts: configs inherit model keys of their base (e.g. the TCN's `hidden`)."""
    import inspect

    ok = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kw.items() if k in ok}


def _face_topology():
    import numpy as np
    from pathlib import Path

    z = np.load(Path(__file__).with_name("face_topology.npz"))
    return z["layout"], z["edges"]


def _point_features(x: torch.Tensor, mirror: torch.Tensor, features) -> torch.Tensor:
    """(B, T, N, D) frontalised landmarks -> (B, T, N, F) per-point features (same groups as LandmarkTCN)."""
    reflected = x[:, :, mirror] * x.new_tensor([-1.0] + [1.0] * (x.shape[-1] - 1))
    asym = x - reflected
    vel = torch.diff(x, dim=1, prepend=x[:, :1])
    asym_vel = torch.diff(asym, dim=1, prepend=asym[:, :1])
    groups = {"pos": x, "asym": asym, "vel": vel, "asym_vel": asym_vel}
    return torch.cat([groups[k] for k in features], dim=-1)


def _geometry_image_weights(layout, grid: int):
    """Linear (barycentric) interpolation from the N layout points to a grid x grid face-surface image:
    W (grid*grid, N) and a (grid, grid) mask of cells inside the face. Rows = top (forehead) to bottom (chin),
    columns = image-left to image-right, centred on the face midline, so a horizontal flip = left/right mirror."""
    import numpy as np
    from scipy.spatial import Delaunay

    half = float(np.abs(layout[:, 0]).max()) + 0.06
    xs = np.linspace(-half, half, grid)
    ys = np.linspace(layout[:, 1].min() - 0.06, layout[:, 1].max() + 0.06, grid)
    gx, gy = np.meshgrid(xs, ys)
    q = np.stack([gx.ravel(), gy.ravel()], 1)
    tri = Delaunay(layout[:, :2])
    s = tri.find_simplex(q)
    ok = s >= 0
    T = tri.transform[s[ok]]
    b = np.einsum("mij,mj->mi", T[:, :2], q[ok] - T[:, 2])
    bary = np.c_[b, 1 - b.sum(1)]
    verts = tri.simplices[s[ok]]
    W = np.zeros((len(q), len(layout)), np.float32)
    rows = np.flatnonzero(ok)
    for k in range(3):
        W[rows, verts[:, k]] = bary[:, k]
    return W, ok.reshape(grid, grid).astype(np.float32)


class _Res3d(nn.Module):
    def __init__(self, ci: int, co: int, stride, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(ci, co, 3, stride=stride, padding=1, bias=False), nn.BatchNorm3d(co), nn.GELU(),
            nn.Conv3d(co, co, 3, padding=1, bias=False), nn.BatchNorm3d(co), nn.Dropout3d(dropout * 0.5),
        )
        self.skip = nn.Sequential(nn.Conv3d(ci, co, 1, stride=stride, bias=False), nn.BatchNorm3d(co))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class LandmarkGeomCNN(nn.Module):
    """Face-surface 3D CNN: every frame's per-point features (asymmetry, motion, change of asymmetry) are
    interpolated onto a grid x grid "geometry image" of the face (a symmetric mean-face layout, so neighbouring
    cells are neighbouring face regions and a horizontal flip is the left/right mirror). A residual 3D CNN then
    convolves over time x face height x face width. Input (B, T, N, D) like the other landmark models."""

    def __init__(self, num_classes: int = 2, features=("asym", "vel", "asym_vel"), grid: int = 32, width: int = 32,
                 dropout: float = 0.3, coord_dims: int = 3):
        super().__init__()
        from .landmarks import MIRROR

        layout, _ = _face_topology()
        W, mask = _geometry_image_weights(layout, grid)
        self.features, self.grid = tuple(features), grid
        self.register_buffer("mirror", torch.tensor(MIRROR), persistent=False)
        self.register_buffer("W", torch.from_numpy(W), persistent=False)
        self.register_buffer("mask", torch.from_numpy(mask)[None, None, None], persistent=False)
        cin = len(self.features) * coord_dims + 1  # + inside-the-face mask channel
        self.bn_in = nn.BatchNorm3d(cin)
        self.stem = nn.Sequential(nn.Conv3d(cin, width, 3, padding=1, bias=False), nn.BatchNorm3d(width), nn.GELU())
        self.layers = nn.Sequential(
            _Res3d(width, width * 3 // 2, (1, 2, 2), dropout),      # T x 16 x 16
            _Res3d(width * 3 // 2, width * 2, (2, 2, 2), dropout),  # T/2 x 8 x 8
            _Res3d(width * 2, width * 4, (2, 2, 2), dropout),       # T/4 x 4 x 4
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * width * 4, num_classes)

    def images(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, N, D) -> (B, F + 1, T, grid, grid) geometry images (+ mask channel)."""
        f = _point_features(x, self.mirror, self.features)  # (B, T, N, F)
        img = torch.einsum("btnf,gn->bftg", f, self.W)
        B, F_, T, _ = img.shape
        img = img.reshape(B, F_, T, self.grid, self.grid)
        return torch.cat([img, self.mask.expand(B, 1, T, -1, -1)], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(self.stem(self.bn_in(self.images(x))))
        return self.head(self.drop(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], -1)))


@register("landmark_geom_cnn")
def _landmark_geom_cnn(num_classes=2, data_cfg=None, pretrained=None, **kw) -> nn.Module:
    return LandmarkGeomCNN(num_classes, coord_dims=_coord_dims(data_cfg), **_own_kwargs(LandmarkGeomCNN, kw))


class _STGCNBlock(nn.Module):
    """Spatial graph conv (one weight per adjacency partition: self / mesh neighbours / mirror partner, with
    learnable edge importance) followed by a temporal conv over each point's track; residual."""

    def __init__(self, ci: int, co: int, A: torch.Tensor, t_kernel: int, stride: int, dropout: float):
        super().__init__()
        self.register_buffer("A", A, persistent=False)
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        self.P = A.shape[0]
        self.gcn = nn.Conv2d(ci, co * self.P, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(co), nn.GELU(),
            nn.Conv2d(co, co, (t_kernel, 1), stride=(stride, 1), padding=(t_kernel // 2, 0)),
            nn.BatchNorm2d(co), nn.Dropout(dropout),
        )
        self.res = (nn.Identity() if ci == co and stride == 1 else
                    nn.Sequential(nn.Conv2d(ci, co, 1, stride=(stride, 1)), nn.BatchNorm2d(co)))
        self.act = nn.GELU()

    def forward(self, x):  # (B, C, T, N)
        B, _, T, N = x.shape
        y = self.gcn(x).view(B, self.P, -1, T, N)
        y = torch.einsum("bpctn,pnm->bctm", y, self.A * self.edge_importance)
        return self.act(self.tcn(y) + self.res(x))


class LandmarkSTGCN(nn.Module):
    """Spatio-temporal graph CNN over the 152 face points: graph convolutions along the face mesh (contour +
    nearest-neighbour edges) with a separate mirror-partner partition (left <-> right), interleaved with
    temporal convolutions. Same per-point features and input (B, T, N, D) as the other landmark models."""

    def __init__(self, num_classes: int = 2, features=("asym", "vel", "asym_vel"), channels=(64, 64, 128, 128),
                 strides=(1, 1, 2, 1), t_kernel: int = 5, dropout: float = 0.3, coord_dims: int = 3):
        super().__init__()
        from .landmarks import MIRROR

        _, edges = _face_topology()
        N = len(MIRROR)
        adj = torch.zeros(N, N)
        e = torch.from_numpy(edges)
        adj[e[:, 0], e[:, 1]] = adj[e[:, 1], e[:, 0]] = 1.0
        d = adj.sum(1).clamp(min=1).rsqrt()
        neigh = d[:, None] * adj * d[None, :]  # symmetric normalisation
        mirror = torch.zeros(N, N)
        mirror[torch.arange(N), torch.tensor(MIRROR)] = 1.0
        mirror.fill_diagonal_(0.0)  # midline points are their own partner: already covered by "self"
        A = torch.stack([torch.eye(N), neigh, mirror])  # (3, N, N)
        self.features = tuple(features)
        self.register_buffer("mirror_idx", torch.tensor(MIRROR), persistent=False)
        cin = len(self.features) * coord_dims
        self.data_bn = nn.BatchNorm1d(cin * N)
        blocks, ci = [], cin
        for co, s in zip(channels, strides):
            blocks.append(_STGCNBlock(ci, co, A, t_kernel, s, dropout))
            ci = co
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * ci, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = _point_features(x, self.mirror_idx, self.features)  # (B, T, N, F)
        B, T, N, F_ = f.shape
        f = self.data_bn(f.permute(0, 2, 3, 1).reshape(B, N * F_, T)).view(B, N, F_, T).permute(0, 2, 3, 1)
        h = self.blocks(f)  # (B, C, T', N)
        return self.head(self.drop(torch.cat([h.mean((2, 3)), h.amax((2, 3))], -1)))


@register("landmark_stgcn")
def _landmark_stgcn(num_classes=2, data_cfg=None, pretrained=None, **kw) -> nn.Module:
    return LandmarkSTGCN(num_classes, coord_dims=_coord_dims(data_cfg), **_own_kwargs(LandmarkSTGCN, kw))
