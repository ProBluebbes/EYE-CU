"""Pre-recorded voice prompts.

Everything StrokeWatch says to a patient is a fixed sentence, so a natural voice is rendered
once (scripts/make_voice.py, ElevenLabs) into web/voice/*.mp3. The pages (phone and bed page,
web/voice.js) play the clip for a sentence when there is one, otherwise the browser's own
speech. web/voice/manifest.json maps sentence -> {"file", "seconds"}.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from . import ROOT

VOICE_DIR = ROOT / "web" / "voice"
MANIFEST = VOICE_DIR / "manifest.json"

# Lines the web pages say on their own (keep in sync with web/phone.js and web/bed.js).
PAGE_LINES = (
    "StrokeWatch camera on.",
    "This is a voice test from StrokeWatch.",
    "Calibration starting.",
    "This is a test from the StrokeWatch dashboard.",
)


def spoken_lines() -> list[str]:
    """Every fixed sentence StrokeWatch can speak, in a stable order."""
    from .befast import DONE_FLAGGED, DONE_OK, RETRY_STEPS, STEPS, STOPPED
    from .detector import CALIB_DONE, CALIB_PROMPTS

    lines = [s.say for s in (*STEPS, *RETRY_STEPS)] + [DONE_OK, DONE_FLAGGED, STOPPED]
    lines += [*CALIB_PROMPTS, CALIB_DONE, *PAGE_LINES]
    return list(dict.fromkeys(lines))


def clip_name(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12] + ".mp3"


@lru_cache(maxsize=1)
def manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text()).get("lines", {})
    except (OSError, ValueError):
        return {}


def seconds(text: str) -> float | None:
    """Length of the recorded clip for this sentence, or None if there is none."""
    c = manifest().get(text)
    return float(c["seconds"]) if c else None


def clip_path(text: str):
    c = manifest().get(text)
    return VOICE_DIR / c["file"] if c else None
