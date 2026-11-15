"""FastAPI server: ward dashboard at /, one page per bed at /bed/<id>, phone cameras at /phone.

    python -m strokewatch.server --camera 0 --tunnel            # webcam + phones via QR code
    python -m strokewatch.server --camera 0 --mock-beds 7       # webcam + 7 simulated patients
    python -m strokewatch.server --camera "Bed 12=0" --camera clip.mp4
    python -m strokewatch.server --mock                         # one simulated patient
    python -m strokewatch.server                                # webcam 0 only (as before)

--tunnel starts a Cloudflare quick tunnel (public HTTPS URL, needed for phone cameras) and the
dashboard's "Add phone" button shows QR codes for it. Remote requests need the per-run access
key the QR codes carry (see access.py); the browser on this machine needs nothing.

Endpoints
  GET  /                          dashboard (grid + alert center)
  WS   /api/dashboard/ws          JSON state (all beds + alerts + tunnel) and binary thumbnails
  GET  /bed/<id>                  bed page
  GET  /api/beds/<id>/video       MJPEG, full size
  WS   /api/beds/<id>/ws          JSON status for one bed; webcam beds also {"type": "say", ...}
                                  (spoken check prompts, played by the page)
  POST /api/beds/<id>/reset       rebuild baseline (closes any open alert as "reset")
  POST /api/beds/<id>/calibrate   same, as a guided calibration (look straight, turn and hold
                                  each side, straight again); prompts in the status
  POST /api/beds/<id>/say         {"text": ...} phone beds: speak through the phone
  POST /api/beds/<id>/exam        start a voice-guided BE-FAST check (eyes, arms, legs)
  POST /api/beds/<id>/exam/cancel stop a running check
  DELETE /api/beds/<id>           phone beds: remove
  POST /api/alerts/<n>/ack        acknowledge (silences the alarm, bed stays in ALERT)
  POST /api/alerts/<n>/resolve    {"resolution": "false_alarm"|"confirmed"} + rebuild baseline
  GET  /phone                     phone camera page (Safari)
  WS   /api/phone/ws?device=&name=  phone -> server: JPEG frames; server -> phone: JSON
  GET  /api/connect               tunnel status + links;  GET /api/qr.svg?target=phone|dashboard
  GET  /api/health                no key needed (tunnel readiness check)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import io
from contextlib import asynccontextmanager

import segno
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import ROOT, apply_overrides, load_config, palsy
from .access import AccessKeyMiddleware, new_key
from .befast import ServerVoice, exam_cfg
from .beds import BedManager, CameraBed, PhoneBed, SimBed, bed_cfg
from .mock import single_scenario, ward_scenarios
from .tunnel import Tunnel

WEB = ROOT / "web"
PHONE_DEFAULTS = {"send_fps": 10, "max_side": 640, "jpeg_quality": 0.7}
MAX_FRAME_BYTES = 2_000_000


def phone_cfg(cfg: dict) -> dict:
    return {**PHONE_DEFAULTS, **(cfg.get("phone") or {})}


def public_links(url: str | None, key: str) -> dict:
    """QR-code links for a tunnel URL; they carry the access key."""
    if url is None:
        return {"phone": None, "dashboard": None}
    return {"phone": f"{url}/phone?k={key}", "dashboard": f"{url}/?k={key}"}


def create_app(manager: BedManager, key: str | None = None, tunnel: Tunnel | None = None
               ) -> FastAPI:
    key = key or new_key()

    voice = ServerVoice(exam_cfg(manager.cfg)["server_voice"])
    bed_listeners: dict[str, set[asyncio.Queue]] = {}  # open bed pages, for spoken prompts

    def to_bed_pages(bed_id: str, msg: dict):  # event loop only
        for q in bed_listeners.get(bed_id, ()):
            q.put_nowait(msg)

    @asynccontextmanager
    async def lifespan(_app):
        loop = asyncio.get_running_loop()

        def say_hook(bed, text: str, step: str | None):
            """BE-FAST check prompts (called from bed threads). Phones speak and show the
            prompt on screen; webcam beds are spoken by their open bed pages (and this
            computer's speaker if exam.server_voice); SIM beds stay silent."""
            msg = {"type": "say", "text": text, "step": step}
            if isinstance(bed, PhoneBed):
                asyncio.run_coroutine_threadsafe(phone_send(bed, msg), loop)
            elif isinstance(bed, CameraBed):
                loop.call_soon_threadsafe(to_bed_pages, bed.bed_id, msg)
                voice.say(text)

        manager.say_hook = say_hook
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    def stopping() -> bool:
        # Set by main(); lets endless streams end so uvicorn's graceful shutdown doesn't hang.
        srv = getattr(app.state, "server", None)
        return bool(srv and srv.should_exit)

    def get_bed(bed_id: str):
        bed = manager.beds.get(bed_id)
        if bed is None:
            raise HTTPException(404, f"no bed {bed_id!r}")
        return bed

    def tunnel_info() -> dict:
        return tunnel.info() if tunnel else {"status": "off", "url": None, "error": None}

    app = FastAPI(title="StrokeWatch", lifespan=lifespan)
    app.add_middleware(AccessKeyMiddleware, key=key)
    thumb_period = 1.0 / bed_cfg(manager.cfg)["thumb_fps"]
    pc = phone_cfg(manager.cfg)

    # ------------------------------------------------------------ dashboard
    @app.get("/")
    def dashboard():
        return FileResponse(WEB / "dashboard.html")

    @app.websocket("/api/dashboard/ws")
    async def dashboard_ws(sock: WebSocket):
        """Text frames: JSON snapshot. Binary frames: [len(bed_id)][bed_id utf-8][JPEG]."""
        await sock.accept()
        sent: dict[str, int] = {}
        try:
            while not stopping():
                snap = manager.snapshot()
                snap["tunnel"] = tunnel_info()
                await sock.send_json(snap)
                for bed in manager.bed_list():
                    tid, thumb = bed.latest_thumb()
                    if thumb is not None and sent.get(bed.bed_id) != tid:
                        sent[bed.bed_id] = tid
                        k = bed.bed_id.encode()
                        await sock.send_bytes(bytes([len(k)]) + k + thumb)
                await asyncio.sleep(thumb_period)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/connect")
    def connect_info():
        return {"tunnel": tunnel_info(), "links": public_links(tunnel_info()["url"], key)}

    @app.get("/api/qr.svg")
    def qr(target: str = "phone"):
        url = public_links(tunnel_info()["url"], key).get(target)
        if url is None:
            raise HTTPException(404, "no tunnel URL yet (start the server with --tunnel)")
        buf = io.BytesIO()  # full SVG (with xmlns): svg_inline() output won't load as an <img>
        segno.make(url, error="m").save(buf, kind="svg", scale=6, border=2, dark="#000",
                                        light="#fff", xmldecl=False)
        return Response(buf.getvalue(), media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------ bed page
    @app.get("/bed/{bed_id}")
    def bed_page(bed_id: str):
        get_bed(bed_id)
        return FileResponse(WEB / "bed.html")



    @app.get("/api/beds/{bed_id}/video")
    async def video(bed_id: str):
        bed = get_bed(bed_id)

        async def gen():
            last = -1
            while not stopping() and manager.beds.get(bed_id) is bed:
                bed.want_full_frames()
                fid, jpeg, _ = bed.latest()
                if jpeg is not None and fid != last:
                    last = fid
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.02)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/api/beds/{bed_id}/ws")
    async def bed_ws(sock: WebSocket, bed_id: str):
        await sock.accept()
        bed = manager.beds.get(bed_id)
        if bed is None:
            await sock.close(code=4404)
            return
        last = -1
        says: asyncio.Queue = asyncio.Queue()
        bed_listeners.setdefault(bed_id, set()).add(says)
        try:
            while not stopping():
                if manager.beds.get(bed_id) is not bed:  # removed
                    await sock.close(code=4404)
                    return
                while not says.empty():  # spoken prompts: the page plays them
                    await sock.send_json(says.get_nowait())
                fid, _, _ = bed.latest()
                st = manager.bed_status(bed, detail=True)
                # every new frame, plus at least 1/s so "offline" shows up when frames stop
                if fid != last or st["state"] == "offline":
                    last = fid
                    await sock.send_json(st)
                await asyncio.sleep(0.05 if st["state"] != "offline" else 1.0)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bed_listeners.get(bed_id, set()).discard(says)

    @app.post("/api/beds/{bed_id}/reset")
    def reset_bed(bed_id: str):
        get_bed(bed_id)
        manager.reset_bed(bed_id)
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/calibrate")
    def calibrate_bed(bed_id: str):
        get_bed(bed_id)
        manager.reset_bed(bed_id, calibrate=True)
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/say")
    async def say(bed_id: str, text: str = Body(..., embed=True)):
        bed = get_bed(bed_id)
        if not isinstance(bed, PhoneBed):
            raise HTTPException(400, "only phone beds have a speaker")
        if not await phone_send(bed, {"type": "say", "text": text[:300]}):
            raise HTTPException(409, "phone not connected")
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/exam")
    def start_exam(bed_id: str):
        get_bed(bed_id)
        return {"ok": manager.request_exam(bed_id)}

    @app.post("/api/beds/{bed_id}/exam/cancel")
    def cancel_exam(bed_id: str):
        get_bed(bed_id)
        return {"ok": manager.cancel_exam(bed_id)}

    @app.delete("/api/beds/{bed_id}")
    async def remove_bed(bed_id: str):
        bed = get_bed(bed_id)
        if not isinstance(bed, PhoneBed):
            raise HTTPException(400, "only phone beds can be removed while running")
        sock = bed.sock
        if sock is not None:
            try:
                await sock.close(code=4004)  # phone page: "removed from the dashboard"
            except RuntimeError:
                pass
        await asyncio.to_thread(manager.remove, bed_id)
        return {"ok": True}

    # ------------------------------------------------------------ alerts
    @app.post("/api/alerts/{alert_id}/ack")
    def ack(alert_id: int):
        return {"ok": manager.acknowledge(alert_id)}

    @app.post("/api/alerts/{alert_id}/resolve")
    def resolve(alert_id: int, resolution: str = Body(..., embed=True)):
        if resolution not in ("false_alarm", "confirmed"):
            raise HTTPException(400, "resolution must be false_alarm or confirmed")
        return {"ok": manager.resolve(alert_id, resolution)}

    # ------------------------------------------------------------ phones
    @app.get("/phone")
    def phone_page():
        return FileResponse(WEB / "phone.html")

    send_locks: dict[str, asyncio.Lock] = {}

    async def phone_send(bed: PhoneBed, msg: dict) -> bool:
        sock = bed.sock
        if sock is None:
            return False
        lock = send_locks.setdefault(bed.bed_id, asyncio.Lock())
        try:
            async with lock:
                await sock.send_json(msg)
            return True
        except (WebSocketDisconnect, RuntimeError):
            return False

    def phone_status(bed: PhoneBed) -> dict:
        st = manager.bed_status(bed)
        ex = st.get("exam") or {}
        return {"type": "status", "bed_id": bed.bed_id, "bed_name": bed.bed_name,
                **{k: st.get(k) for k in ("state", "valid", "invalid_reason", "baseline_progress",
                                          "fps", "prompt", "calibrating", "calibrated_range")},
                "exam": {k: ex.get(k) for k in ("active", "step", "say", "label")}}

    @app.websocket("/api/phone/ws")
    async def phone_ws(sock: WebSocket):
        """One phone = one bed, keyed by a device id the page keeps in localStorage, so a
        reload or network blip reconnects to the same bed (and the same baseline)."""
        await sock.accept()
        device = sock.query_params.get("device", "")[:64]
        name = sock.query_params.get("name", "").strip()[:40]
        if not device:
            await sock.close(code=4400)
            return
        bed = manager.phone_bed(device)
        if bed is None:
            bed_id, default_name = manager.allocate()
            # FaceAnalyzer loads the model: keep that off the event loop
            bed = await asyncio.to_thread(PhoneBed, bed_id, name or default_name, manager.cfg,
                                          device)
            manager.add(bed)
            print(f"Phone joined as {bed.bed_name} ({bed.bed_id})")
        elif name:
            bed.bed_name = name
        old, bed.sock = bed.sock, sock
        if old is not None:  # same phone opened the page twice: newest wins
            try:
                await old.close(code=4000)
            except RuntimeError:
                pass
        await phone_send(bed, {"type": "hello", "bed_id": bed.bed_id, "bed_name": bed.bed_name,
                               **pc})

        async def status_loop():
            while True:
                await asyncio.sleep(1.0)
                await phone_send(bed, phone_status(bed))

        status_task = asyncio.create_task(status_loop())
        try:
            while not stopping():
                msg = await sock.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data and len(data) <= MAX_FRAME_BYTES:
                    bed.push(data)
                elif msg.get("text"):  # control messages from the phone page
                    try:
                        ctl = json.loads(msg["text"])
                    except ValueError:
                        continue
                    if ctl.get("type") == "calibrate":
                        manager.reset_bed(bed.bed_id, calibrate=True)
                        await phone_send(bed, phone_status(bed))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            status_task.cancel()
            if bed.sock is sock:
                bed.sock = None

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def build_manager(cfg: dict, cameras: list[str], mock_beds: int, scenario: str | None
                  ) -> BedManager:
    """cameras: ["0", "Bed 12=0", "clip.mp4", "rtsp://..."]. Beds are numbered in order."""
    manager = BedManager(cfg)
    for spec in cameras:
        bed_id, name = manager.allocate()
        # Optional "NAME=SRC". A bare URL/path may contain "=" itself, but then the part before
        # the first "=" contains ":" or "/", which a bed name doesn't.
        src = spec
        head, eq, tail = spec.partition("=")
        if eq and head.strip() and not any(c in head for c in ":/\\"):
            name, src = head.strip(), tail
        manager.add(CameraBed(bed_id, name, cfg, src))
    scenarios = [single_scenario(scenario)] if scenario else ward_scenarios(mock_beds)
    for sc in scenarios:
        bed_id, name = manager.allocate()
        manager.add(SimBed(bed_id, name, cfg, sc, seed=int(bed_id[3:])))
    return manager


def print_links(url: str, key: str):
    links = public_links(url, key)
    print("\nTunnel ready. Scan the QR code on the dashboard (Add phone), or:")
    print(f"  Phone camera:          {links['phone']}")
    print(f"  Dashboard (any device): {links['dashboard']}")
    segno.make(links["phone"], error="m").terminal(compact=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="append", default=[], metavar="[NAME=]SRC",
                    help="camera bed: webcam index, video file or RTSP URL; repeatable")
    ap.add_argument("--source", action="append", default=[], help="same as --camera (old flag)")
    ap.add_argument("--mock-beds", type=int, default=0, help="add N simulated patients")
    ap.add_argument("--mock", action="store_true", help="one simulated patient (old demo)")
    ap.add_argument("--scenario", default="normal", choices=["normal", "predroop"],
                    help="scenario for --mock")
    ap.add_argument("--tunnel", action="store_true",
                    help="start a Cloudflare quick tunnel so phones can join via QR code")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VAL",
                    help="config override, e.g. detector.threshold=3.5")
    ap.add_argument("--no-palsy", action="store_true",
                    help="droop alerts from StrokeWatch's detector alone (default: combined with the palsy model)")
    ap.add_argument("--palsy-model", metavar="RUN_DIR",
                    help="palsynet run folder for the combined detection (default: palsy.model_dir in config.yaml)")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    palsy.configure(cfg, enabled=not args.no_palsy, model_dir=args.palsy_model)

    cameras = args.camera + args.source
    if not (cameras or args.mock or args.mock_beds or args.tunnel):
        cameras = ["0"]
    manager = build_manager(cfg, cameras, args.mock_beds,
                            args.scenario if args.mock else None)

    import uvicorn
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]
    key = new_key()
    tunnel = None
    if args.tunnel:
        tunnel = Tunnel(port, (cfg.get("tunnel") or {}).get("protocol"),
                        on_ready=lambda url: print_links(url, key))
    app = create_app(manager, key, tunnel)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning",
                                           timeout_graceful_shutdown=3, ws_max_size=4 * 2**20))
    app.state.server = server
    print(f"StrokeWatch: {len(manager.beds)} bed(s) on http://{host}:{port}")
    if tunnel:
        print("Starting Cloudflare tunnel…")
        tunnel.start()
    try:
        server.run()
    finally:
        if tunnel:
            tunnel.stop()
        manager.stop()  # idempotent; covers startup failures before lifespan teardown
        print("StrokeWatch stopped, cameras released.")


if __name__ == "__main__":
    main()
