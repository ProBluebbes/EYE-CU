"""Render Eye-CU's spoken prompts once with ElevenLabs -> web/voice/*.mp3 + manifest.json.

The API key is read from the ELEVENLABS_API_KEY environment variable, or from a line
ELEVENLABS_API_KEY=... in .env at the repo root (.env is git-ignored). Only the fixed prompt
sentences are sent (strokewatch.voice.spoken_lines), no patient data.

    .venv/bin/python scripts/make_voice.py --list             # premade voices on the account
    .venv/bin/python scripts/make_voice.py --samples          # one sample per calm-sounding voice
    .venv/bin/python scripts/make_voice.py --voice <voice_id> # render every prompt

Re-running --voice only renders sentences that are new or changed (or all, with another voice
or settings). Restart the server afterwards: check timing is fitted to the clip lengths.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strokewatch import ROOT  # noqa: E402
from strokewatch.voice import MANIFEST, VOICE_DIR, clip_name, spoken_lines  # noqa: E402

API = "https://api.elevenlabs.io"
FORMAT, KBPS = "mp3_44100_128", 128
# Calm and unhurried: steadier delivery (stability), no added style, a bit slower than normal.
SETTINGS = {"stability": 0.65, "similarity_boost": 0.75, "style": 0.0, "speed": 0.9,
            "use_speaker_boost": True}
SAMPLE_TEXT = ("This is Eye-CU. Please follow a few quick instructions. "
               "Without moving your head, look to your left.")
CALM = re.compile(r"calm|soft|gentle|warm|sooth|relax|kind|friendly|reassur|soothing", re.I)


def api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ELEVENLABS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("No API key: set ELEVENLABS_API_KEY or put ELEVENLABS_API_KEY=... in .env")
    return key


def request(method: str, path: str, key: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"xi-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"ElevenLabs {method} {path.split('?')[0]}: HTTP {e.code} {e.read()[:300]!r}")


def voices(key: str) -> list[dict]:
    q = urllib.parse.urlencode({"category": "premade", "page_size": 100})
    return json.loads(request("GET", f"/v2/voices?{q}", key)).get("voices", [])


def describe(v: dict) -> str:
    labels = ", ".join(f"{k}: {x}" for k, x in (v.get("labels") or {}).items())
    return f"{v['voice_id']}  {v['name']:<12} {labels}  {v.get('description') or ''}".strip()


def tts(key: str, voice_id: str, model: str, text: str) -> bytes:
    q = urllib.parse.urlencode({"output_format": FORMAT})
    return request("POST", f"/v1/text-to-speech/{voice_id}?{q}", key,
                   {"text": text, "model_id": model, "voice_settings": SETTINGS})


def duration(path: Path) -> float:
    """Clip length in seconds, from the constant bitrate of FORMAT."""
    return round(path.stat().st_size * 8 / (KBPS * 1000), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list premade voices")
    ap.add_argument("--samples", action="store_true",
                    help="render a sample per calm-sounding voice (or per --voice) to tmp/voice_samples")
    ap.add_argument("--voice", help="voice_id (comma-separated with --samples)")
    ap.add_argument("--model", default="eleven_multilingual_v2")
    a = ap.parse_args()
    key = api_key()

    if a.list:
        for v in voices(key):
            print(describe(v))
        return

    if a.samples:
        vs = voices(key)
        if a.voice:
            want = set(a.voice.split(","))
            vs = [v for v in vs if v["voice_id"] in want]
        else:
            vs = [v for v in vs if CALM.search(json.dumps([v.get("labels"), v.get("description")]))][:6]
        out = ROOT / "tmp" / "voice_samples"
        out.mkdir(parents=True, exist_ok=True)
        for v in vs:
            f = out / f"{re.sub(r'[^A-Za-z0-9]+', '_', v['name'])}_{v['voice_id']}.mp3"
            f.write_bytes(tts(key, v["voice_id"], a.model, SAMPLE_TEXT))
            print(f"{f}\n    {describe(v)}")
        return

    if not a.voice:
        ap.error("--voice is required to render the prompts (see --list / --samples)")
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    same = (old.get("voice_id"), old.get("model"), old.get("settings")) == (a.voice, a.model, SETTINGS)
    prev = old.get("lines", {}) if same else {}
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    lines, keep = {}, set()
    for text in spoken_lines():
        name = clip_name(text)
        path = VOICE_DIR / name
        if text in prev and path.exists():
            lines[text] = prev[text]
        else:
            path.write_bytes(tts(key, a.voice, a.model, text))
            lines[text] = {"file": name, "seconds": duration(path)}
            print(f"{lines[text]['seconds']:5.2f}s  {text}")
        keep.add(name)
    for f in VOICE_DIR.glob("*.mp3"):
        if f.name not in keep:
            f.unlink()
    MANIFEST.write_text(json.dumps({"voice_id": a.voice, "model": a.model, "settings": SETTINGS,
                                    "lines": lines}, indent=1) + "\n")
    print(f"{len(lines)} prompts in {VOICE_DIR.relative_to(ROOT)}; restart the server to use them.")


if __name__ == "__main__":
    main()
