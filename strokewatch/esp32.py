"""ESP32-S3 camera + speaker on one USB cable (firmware: reading_esp/esp32s3_cam_usb/esp32s3_cam_usb.ino).

The board sends JPEG frames over USB serial and plays voice prompts we send back on a buzzer / small speaker:
    ESP32 -> here:  FF AA BB CC | u32 LE length | JPEG
    here -> ESP32:  FF AA BB DD | 'P' | u32 LE length | 16 kHz mono s16le PCM     (queue audio)
                    FF AA BB DD | 'S' | 00 00 00 00                               (stop: drop the queue)

Camera source names (server --camera, or config):  "esp32"  (first ESP32 found)  or  "esp32:/dev/cu.usbmodem1101".
Esp32Capture looks like a cv2.VideoCapture (read / get / set / isOpened / release) so VideoSource can use it.
"""
from __future__ import annotations

import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

FRAME_MAGIC = b"\xff\xaa\xbb\xcc"
AUDIO_MAGIC = b"\xff\xaa\xbb\xdd"
MAX_FRAME = 500_000           # bigger = corrupt header
SAMPLE_RATE = 16000
ESPRESSIF_VID = 0x303A        # native USB (USB CDC on boot)
BRIDGE_VIDS = {0x1A86, 0x10C4, 0x0403}  # CH34x / CP210x / FTDI USB-serial chips (UART port)


def is_esp32_source(source: str) -> bool:
    return str(source).lower().startswith("esp32")


def find_port(source: str) -> str:
    """'esp32:/dev/cu.x' -> that port; 'esp32' -> the first Espressif (or USB-serial) port."""
    s = str(source)
    if ":" in s:
        return s.split(":", 1)[1]
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for want in ({ESPRESSIF_VID}, BRIDGE_VIDS):
        for p in ports:
            if p.vid in want:
                return p.device
    for p in ports:
        if "usbmodem" in p.device or "usbserial" in p.device:
            return p.device
    raise RuntimeError("no ESP32 found on USB (plug it in, or name the port: esp32:/dev/cu.usbmodemXXXX)")


class Esp32Capture:
    """cv2.VideoCapture-like reader for the ESP32's JPEG stream, plus a speaker."""

    def __init__(self, source: str, baud: int = 2_000_000):
        import serial
        self.port = find_port(source)
        try:
            self.ser = serial.Serial(self.port, baud, timeout=0.5, write_timeout=10)
        except OSError:  # port can't do 2 Mbaud (native USB ignores the rate anyway)
            self.ser = serial.Serial(self.port, 115200, timeout=0.5, write_timeout=10)
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.read_id = 0
        self.alive = True
        self.status: list[str] = []   # text lines the board printed (CAMERA_OK, SPEAKER_OK, ...)
        self.wlock = threading.Lock()
        self.reader = threading.Thread(target=self._read_loop, name="esp32-rx", daemon=True)
        self.reader.start()
        self.speaker = Esp32Speaker(self)

    # ---------------------------------------------------------------- frames in
    def _read_loop(self):
        buf = bytearray()
        while self.alive:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
            except Exception:  # unplugged
                break
            if not chunk:
                continue
            buf += chunk
            for frame in self._parse(buf):
                with self.cond:
                    self.frame, self.frame_id = frame, self.frame_id + 1
                    self.cond.notify_all()
        self.alive = False
        with self.cond:
            self.cond.notify_all()

    def _parse(self, buf: bytearray):
        """Yield decoded frames from buf (consumed in place); keep printed text lines in self.status."""
        while True:
            i = buf.find(FRAME_MAGIC)
            if i < 0:
                self._text(buf[:-3] if len(buf) > 3 else b"")
                del buf[:max(0, len(buf) - 3)]
                return
            if i:
                self._text(buf[:i])
                del buf[:i]
            if len(buf) < 8:
                return
            (n,) = struct.unpack("<I", buf[4:8])
            if not 0 < n < MAX_FRAME:
                del buf[:4]
                continue
            if len(buf) < 8 + n:
                return
            jpg = bytes(buf[8:8 + n])
            del buf[:8 + n]
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img

    def _text(self, b: bytes):
        for line in bytes(b).decode("ascii", "ignore").splitlines():
            line = line.strip()
            # only the board's own status lines (CAMERA_OK, SPEAKER_INIT_FAILED (...)), not stray JPEG bytes
            if re.match(r"^[A-Z]+_[A-Z_]+(\s.*)?$", line) and line.isprintable():
                self.status = (self.status + [line])[-10:]
                print(f"esp32 {self.port}: {line}")

    def read(self, timeout: float = 3.0):
        """Next new frame -> (ok, BGR image)."""
        end = time.monotonic() + timeout
        with self.cond:
            while self.alive and self.frame_id == self.read_id:
                left = end - time.monotonic()
                if left <= 0:
                    return False, None
                self.cond.wait(left)
            if self.frame_id == self.read_id:
                return False, None
            self.read_id = self.frame_id
            return True, self.frame.copy()

    def isOpened(self) -> bool:
        return self.alive

    def get(self, prop):
        return 0.0

    def set(self, prop, value):
        return False

    def release(self):
        self.alive = False
        self.speaker.close()
        try:
            self.ser.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- audio out
    def test_beep(self):
        """1 s, 2 kHz beep on the board's buzzer (checks wiring without any audio data)."""
        self.send(b"T")

    def send(self, kind: bytes, payload: bytes = b""):
        with self.wlock:
            self.ser.write(AUDIO_MAGIC + kind + struct.pack("<I", len(payload)) + payload)


class Esp32Speaker:
    """Turns a prompt into 16 kHz PCM (recorded clip if there is one, else macOS `say`) and sends it.
    A new prompt stops the previous one. Runs on its own thread so bed threads never wait on USB."""

    def __init__(self, cap: Esp32Capture, gain_db: float = 0.0):
        self.cap, self.gain_db = cap, gain_db
        self.cache: dict[str, bytes] = {}
        self.pending: str | None = None
        self.cond = threading.Condition()
        self.open = True
        threading.Thread(target=self._loop, name="esp32-tx", daemon=True).start()

    def say(self, text: str):
        with self.cond:
            self.pending = text
            self.cond.notify()

    def close(self):
        with self.cond:
            self.open = False
            self.cond.notify()

    def _loop(self):
        while True:
            with self.cond:
                while self.open and self.pending is None:
                    self.cond.wait()
                if not self.open:
                    return
                text, self.pending = self.pending, None
            try:
                pcm = self.pcm(text)
                self.cap.send(b"S")
                if pcm:
                    for i in range(0, len(pcm), 32768):  # chunks, so a newer prompt can cut in quickly
                        if self.pending is not None:
                            break
                        self.cap.send(b"P", pcm[i:i + 32768])
            except Exception as e:  # never take the bed down over audio
                print(f"esp32 speaker: {e}")

    def pcm(self, text: str) -> bytes:
        if text in self.cache:
            return self.cache[text]
        from .voice import clip_path
        src = clip_path(text)
        tmp = None
        if not (src and Path(src).exists()):
            if not shutil.which("say"):
                return b""
            tmp = Path(tempfile.mkstemp(suffix=".aiff")[1])
            subprocess.run(["say", "-o", str(tmp), text], check=True, capture_output=True)
            src = tmp
        try:
            out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
                                  # piezo range only (350-3500 Hz), then compress so quiet syllables come up to full level
                                  "-af", "highpass=f=350,highpass=f=350,lowpass=f=3500,lowpass=f=3500,"
                                         "acompressor=threshold=-30dB:ratio=8:attack=5:release=80:makeup=12,"
                                         f"dynaudnorm=f=150:g=9:p=0.95,volume={self.gain_db}dB,alimiter=limit=0.97",
                                  "-f", "s16le", "-"], check=True, capture_output=True).stdout
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)
        self.cache[text] = out
        return out
