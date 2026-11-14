"""Cloudflare quick tunnel: a public https://<random>.trycloudflare.com URL for this server.

Phones need HTTPS for camera access, and eduroam-style networks block device-to-device
traffic; the tunnel solves both. No Cloudflare account needed. The URL changes every run.

status: off -> starting -> connecting (URL known) -> ready (cloudflared registered its
        connection to Cloudflare's edge), or error (cloudflared missing / exited).

Readiness comes from cloudflared's own "Registered tunnel connection" log line, not from
fetching the URL: looking up the brand-new hostname too early gets a "not found" answer
that this Mac then caches for about a minute. The URL check runs only after registration,
as a log-only confirmation.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
import urllib.request

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
HEALTH_PATH = "/api/health"


class Tunnel:
    def __init__(self, port: int, protocol: str | None = None, on_ready=None):
        self.port = port
        self.protocol = None if protocol in (None, "", "auto") else protocol
        self.on_ready = on_ready  # fn(url), called once the tunnel is registered
        self.status = "off"
        self.url: str | None = None
        self.error: str | None = None
        self.proc: subprocess.Popen | None = None
        self._log: list[str] = []
        self._stopping = False
        self._registered = False
        self._ready_lock = threading.Lock()
        self._t0 = 0.0

    def info(self) -> dict:
        return {"status": self.status, "url": self.url, "error": self.error}

    def start(self):
        exe = shutil.which("cloudflared")
        if exe is None:
            self.status, self.error = "error", "cloudflared not installed (brew install cloudflared)"
            print(f"Tunnel: {self.error}")
            return
        # 127.0.0.1, not localhost: "localhost" may resolve to ::1, where uvicorn isn't listening.
        cmd = [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"]
        if self.protocol:
            cmd += ["--protocol", self.protocol]
        self.status = "starting"
        self._t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read_output, daemon=True, name="tunnel-log").start()

    def _read_output(self):
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self._log = (self._log + [line.rstrip()])[-30:]
            m = URL_RE.search(line)
            if m and self.url is None:
                self.url = m.group(0)
                self.status = "connecting"
            if "Registered tunnel connection" in line:
                self._registered = True
            if self.url and self._registered:
                self._mark_ready()
        if not self._stopping:
            last = next((ln for ln in reversed(self._log) if ln.strip()), "no output")
            self.status, self.error = "error", f"cloudflared exited: {last[-200:]}"
            print(f"Tunnel: {self.error}")

    def _mark_ready(self):
        with self._ready_lock:
            if self.status != "connecting":
                return
            self.status = "ready"
        print(f"Tunnel registered after {time.time() - self._t0:.0f} s")
        if self.on_ready:
            self.on_ready(self.url)
        threading.Thread(target=self._confirm_reachable, daemon=True, name="tunnel-check").start()

    def _confirm_reachable(self, timeout: float = 120.0):
        """Log when the public URL answers from here (DNS for a new hostname takes a while)."""
        time.sleep(3)
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stopping:
            try:
                with urllib.request.urlopen(self.url + HEALTH_PATH, timeout=5) as r:
                    if r.status == 200:
                        print(f"Tunnel URL answering from this machine after "
                              f"{time.time() - self._t0:.0f} s")
                        return
            except Exception:
                pass
            time.sleep(3)

    def stop(self):
        self._stopping = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.status != "error":
            self.status = "off"
