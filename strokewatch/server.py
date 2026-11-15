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
  GET  /api/beds/<id>/video       MJPEG, full size, with landmarks; ?overlay=0 without
  WS   /api/beds/<id>/ws?overlay=1|0  JSON status for one bed + binary JPEG video frames
                                  (overlay=0: no landmarks); webcam beds also {"type": "say", ...}
                                  (spoken check prompts, played by the page)
  POST /api/beds/<id>/reset       rebuild baseline (closes any open alert as "reset")
  POST /api/beds/<id>/calibrate   same, as a guided calibration (look straight, turn and hold
                                  each side, straight again); prompts in the status
  POST /api/beds/<id>/patient     {"name","mrn","age","sex","dx","attending","notes","lkw"}
  POST /api/beds/<id>/say         {"text": ...} phone beds: speak through the phone
  POST /api/beds/<id>/exam        start a voice-guided BE-FAST check (eyes, arms)
  POST /api/beds/<id>/exam?only=arms  hand test: arms out, palms up, 10 s (arm drift only)
  POST /api/beds/<id>/exam/cancel stop a running check
  DELETE /api/beds/<id>           phone and demo beds: remove
  GET  /api/demos                 demo videos in clips/demo/;  POST {"name": ...} adds one as a bed
  POST /api/alerts/<n>/ack        acknowledge (silences the alarm, bed stays in ALERT)
  POST /api/alerts/<n>/resolve    {"resolution": "false_alarm"|"confirmed"} + rebuild baseline
  POST /api/alerts/<n>/step       {"step": "team"|"glucose"|..., "done": bool} code-stroke checklist
  (POST bodies may carry "by": staff initials, recorded in the activity log)
  GET  /phone                     phone camera page (Safari)
  WS   /api/phone/ws?device=&name=  phone -> server: JPEG frames; server -> phone: JSON
  GET  /api/connect               tunnel status + links;  GET /api/qr.svg?target=phone|dashboard
  GET  /api/health                no key needed (tunnel readiness check)
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import io
from contextlib import asynccontextmanager
from pathlib import Path

import segno
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import ROOT, apply_overrides, load_config, palsy
from .access import AccessKeyMiddleware, new_key
from .befast import ServerVoice, exam_cfg
from .beds import CHECKLIST, BedManager, CameraBed, DemoBed, PhoneBed, SimBed, bed_cfg
from .mock import single_scenario, ward_scenarios
from .tunnel import Tunnel

WEB = ROOT / "web"
DEMO_DIR = ROOT / "clips" / "demo"  # demo videos offered by the dashboard's "Add demo" button
DEMO_EXTS = {".mp4", ".mov", ".m4v"}
PHONE_DEFAULTS = {"send_fps": 10, "max_side": 640, "jpeg_quality": 0.7}
MAX_FRAME_BYTES = 2_000_000


def demo_clips() -> dict[str, Path]:
    """Display name (file name without extension) -> path, sorted by name."""
    if not DEMO_DIR.is_dir():
        return {}
    files = sorted(p for p in DEMO_DIR.iterdir() if p.suffix.lower() in DEMO_EXTS)
    return {p.stem: p for p in files}


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
    esp32_speaker = bool(bed_cfg(manager.cfg).get("esp32_speaker", False))  # ESP32 beds talk on their own speaker
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
            elif isinstance(bed, CameraBed) and getattr(bed.src, "is_esp32", False) and esp32_speaker:
                # ESP32-S3 bed: its own buzzer/speaker talks; bed pages only show the prompt
                bed.src.cap.speaker.say(text)
                loop.call_soon_threadsafe(to_bed_pages, bed.bed_id, {**msg, "speaker": "esp32"})
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

    def staff(body: dict | None) -> str | None:
        """Staff initials sent with an action (shown in the activity log)."""
        by = str((body or {}).get("by") or "").strip()[:12]
        return by or None

    def tunnel_info() -> dict:
        return tunnel.info() if tunnel else {"status": "off", "url": None, "error": None}

    app = FastAPI(title="Eye-CU", lifespan=lifespan)
    app.add_middleware(AccessKeyMiddleware, key=key)

    @app.middleware("http")
    async def revalidate(request, call_next):
        """Pages and static files: browsers must check for a newer version on every load
        (cheap 304 via ETag), so an update never mixes old cached CSS/JS with new pages."""
        resp = await call_next(request)
        path = request.url.path
        if path in ("/", "/phone") or path.startswith(("/static/", "/bed/")):
            resp.headers["Cache-Control"] = "no-cache"
        return resp
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
    async def video(bed_id: str, overlay: int = 1):
        """overlay=1 (default): landmark lines and dots drawn on the face; overlay=0: plain image."""
        bed = get_bed(bed_id)
        marked = bool(overlay)

        async def gen():
            last = -1
            while not stopping() and manager.beds.get(bed_id) is bed:
                bed.want_full_frames(marked)
                fid, jpeg, _ = bed.latest(marked)
                if jpeg is not None and fid != last:
                    last = fid
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.02)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/api/beds/{bed_id}/ws")
    async def bed_ws(sock: WebSocket, bed_id: str, overlay: int = 1):
        marked = bool(overlay)
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
                # Video rides on this socket too (binary JPEG after each status): no extra
                # long-lived HTTP stream, so many open bed pages don't hit the browser's
                # ~6-connections-per-server limit. overlay=0: without landmark lines/dots.
                bed.want_full_frames(marked)
                fid, jpeg, _ = bed.latest(marked)
                st = manager.bed_status(bed, detail=True)
                # every new frame, plus at least 1/s so "offline" shows up when frames stop
                if fid != last or st["state"] == "offline":
                    last = fid
                    await sock.send_json(st)
                    if jpeg is not None:
                        await sock.send_bytes(jpeg)
                await asyncio.sleep(0.05 if st["state"] != "offline" else 1.0)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bed_listeners.get(bed_id, set()).discard(says)

    @app.post("/api/beds/{bed_id}/reset")
    def reset_bed(bed_id: str, body: dict | None = Body(None)):
        get_bed(bed_id)
        manager.reset_bed(bed_id, by=staff(body))
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/calibrate")
    def calibrate_bed(bed_id: str, body: dict | None = Body(None)):
        get_bed(bed_id)
        manager.reset_bed(bed_id, calibrate=True, by=staff(body))
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/patient")
    def set_patient(bed_id: str, body: dict = Body(...)):
        get_bed(bed_id)
        return {"ok": manager.set_patient(bed_id, body, staff(body))}

    @app.post("/api/beds/{bed_id}/say")
    async def say(bed_id: str, text: str = Body(..., embed=True)):
        bed = get_bed(bed_id)
        if isinstance(bed, CameraBed) and getattr(bed.src, "is_esp32", False) and esp32_speaker:
            bed.src.cap.speaker.say(text[:300])
            return {"ok": True}
        if not isinstance(bed, PhoneBed):
            raise HTTPException(400, "only phone and ESP32 beds have a speaker")
        if not await phone_send(bed, {"type": "say", "text": text[:300]}):
            raise HTTPException(409, "phone not connected")
        return {"ok": True}

    @app.post("/api/beds/{bed_id}/exam")
    def start_exam(bed_id: str, only: str | None = None):
        get_bed(bed_id)
        if only not in (None, "arms"):
            raise HTTPException(400, "only may be 'arms' (hand test)")
        return {"ok": manager.request_exam(bed_id, only)}

    @app.post("/api/beds/{bed_id}/exam/cancel")
    def cancel_exam(bed_id: str):
        get_bed(bed_id)
        return {"ok": manager.cancel_exam(bed_id)}

    @app.delete("/api/beds/{bed_id}")
    async def remove_bed(bed_id: str):
        bed = get_bed(bed_id)
        if not isinstance(bed, (PhoneBed, DemoBed)):
            raise HTTPException(400, "only phone and demo beds can be removed while running")
        sock = getattr(bed, "sock", None)
        if sock is not None:
            try:
                await sock.close(code=4004)  # phone page: "removed from the dashboard"
            except RuntimeError:
                pass
        await asyncio.to_thread(manager.remove, bed_id)
        return {"ok": True}

    # ------------------------------------------------------------ demo videos
    @app.get("/api/demos")
    def list_demos():
        return {"demos": list(demo_clips()), "folder": str(DEMO_DIR.relative_to(ROOT))}

    @app.post("/api/demos")
    async def add_demo(name: str = Body(..., embed=True)):
        """Add a bed that plays clips/demo/<name> in a loop (only names from the list)."""
        path = demo_clips().get(name)
        if path is None:
            raise HTTPException(404, f"no demo video {name!r}")
        bed_id, bed_name = manager.allocate()
        try:
            bed = await asyncio.to_thread(DemoBed, bed_id, bed_name, manager.cfg, str(path), name)
        except RuntimeError as e:
            raise HTTPException(500, str(e)) from e
        manager.add(bed)
        return {"ok": True, "bed_id": bed_id, "bed_name": bed_name}

    # ------------------------------------------------------------ alerts
    @app.post("/api/alerts/{alert_id}/ack")
    def ack(alert_id: int, body: dict | None = Body(None)):
        return {"ok": manager.acknowledge(alert_id, staff(body))}

    @app.post("/api/alerts/{alert_id}/resolve")
    def resolve(alert_id: int, body: dict = Body(...)):
        resolution = body.get("resolution")
        if resolution not in ("false_alarm", "confirmed"):
            raise HTTPException(400, "resolution must be false_alarm or confirmed")
        return {"ok": manager.resolve(alert_id, resolution, staff(body))}

    @app.post("/api/alerts/{alert_id}/step")
    def checklist_step(alert_id: int, body: dict = Body(...)):
        if body.get("step") not in CHECKLIST:
            raise HTTPException(400, f"step must be one of {', '.join(CHECKLIST)}")
        return {"ok": manager.checklist_step(alert_id, body["step"], bool(body.get("done", True)),
                                             staff(body))}

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
    videos = []  # permanent video beds (config beds.videos), with their fixed bed numbers
    for v in bed_cfg(cfg).get("videos") or []:
        path = Path(v["source"]).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            print(f"Video bed {v.get('bed') or v.get('label')!r} skipped: {path} not found")
            continue
        if v.get("bed"):
            manager.reserve(int(v["bed"]))
        videos.append((v, path))
    for spec in cameras:
        bed_id, name = manager.allocate()
        # Optional "NAME=SRC". A bare URL/path may contain "=" itself, but then the part before
        # the first "=" contains ":" or "/", which a bed name doesn't.
        src = spec
        head, eq, tail = spec.partition("=")
        if eq and head.strip() and not any(c in head for c in ":/\\"):
            name, src = head.strip(), tail
        bed = CameraBed(bed_id, name, cfg, src)
        if bed.src.is_esp32:
            bed.label = f"ESP32-S3 camera ({bed.src.cap.port})"
        manager.add(bed)
    # Permanent video beds: looped forever, always present.
    for v, path in videos:
        bed_id, name = (f"bed{int(v['bed'])}", f"Bed {int(v['bed'])}") if v.get("bed") \
            else manager.allocate()
        vcfg = apply_overrides(copy.deepcopy(cfg), v.get("set"))  # per-bed tweaks, as --set
        bed = CameraBed(bed_id, v.get("name") or name, vcfg, str(path))
        bed.kind, bed.label = "video", v.get("label")
        if v.get("patient"):  # default patient details (staff can still edit them)
            bed.patient = {k: str(x) for k, x in v["patient"].items() if x}
        manager.add(bed)
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
                    help="droop alerts from Eye-CU's own detector alone (default: combined with the palsy model)")
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
    print(f"Eye-CU: {len(manager.beds)} bed(s) on http://{host}:{port}")
    if tunnel:
        print("Starting Cloudflare tunnel…")
        tunnel.start()
    try:
        server.run()
    finally:
        if tunnel:
            tunnel.stop()
        manager.stop()  # idempotent; covers startup failures before lifespan teardown
        print("Eye-CU stopped, cameras released.")


if __name__ == "__main__":
    main()
