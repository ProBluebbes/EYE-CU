# Read JPEG frames from the ESP32-S3 over USB serial and run MediaPipe on them.
# pip install pyserial opencv-python mediapipe numpy
#
# Find the port:  python -m serial.tools.list_ports
#   macOS: /dev/cu.usbmodemXXXX   Windows: COM5   Linux: /dev/ttyACM0
# Close the Arduino Serial Monitor first - only one program can hold the port.

import os
import struct
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import serial
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

PORT = "/dev/cu.usbmodem1101"   # <- change this
BAUD = 2000000
MAGIC = b"\xff\xaa\xbb\xcc"
MAX_FRAME = 500_000             # sanity limit; anything bigger is a corrupted header


def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None  # timeout
        buf.extend(chunk)
    return bytes(buf)


def frames(ser):
    window = b""
    while True:
        b = ser.read(1)
        if not b:
            continue
        window = (window + b)[-4:]
        if window != MAGIC:
            continue
        header = read_exact(ser, 4)
        if header is None:
            continue
        (length,) = struct.unpack("<I", header)
        if not 0 < length < MAX_FRAME:
            continue
        data = read_exact(ser, length)
        if data is None:
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            yield img
        window = b""


def make_landmarker():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model (~4 MB, one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def main():
    ser = serial.Serial(PORT, BAUD, timeout=2)
    landmarker = make_landmarker()

    t0, n, last_ts = time.time(), 0, -1
    for frame in frames(ser):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)  # must strictly increase
        last_ts = ts
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if res.face_landmarks:
            h, w = frame.shape[:2]
            lm = res.face_landmarks[0]
            for idx in (61, 291):  # left / right mouth corners
                cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 4, (0, 255, 0), -1)

        n += 1
        fps = n / (time.time() - t0)
        cv2.putText(frame, f"{fps:.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("ESP32-S3 (USB)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
