"""Access key for anything not coming from this machine (i.e. through the tunnel).

The tunnel URL is public, so remote requests need the random per-run key that the dashboard's
QR codes carry (?k=KEY). The first request with a valid ?k= gets an HttpOnly cookie and a
redirect to the same URL without the key; later requests (and WebSockets) use the cookie.
Local requests (browser on this Mac) need no key.

Remote = has Cloudflare's cf-connecting-ip header (cloudflared itself connects from localhost),
or comes from a non-loopback address (if the server is bound to 0.0.0.0).
"""
from __future__ import annotations

import hmac
import secrets
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, urlencode

COOKIE = "sw_key"
OPEN_PATHS = {"/api/health"}  # used by the tunnel readiness check
LOOPBACK = {"127.0.0.1", "::1", "localhost"}

DENIED_HTML = b"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>StrokeWatch</title><body style="font:18px system-ui;background:#111;color:#eee;padding:24px">
<h2>Access key needed</h2><p>Scan a QR code on the StrokeWatch dashboard
(<b>Add phone</b> button). Links change every time the server restarts.</p></body>"""


def new_key() -> str:
    return secrets.token_urlsafe(12)


class AccessKeyMiddleware:
    def __init__(self, app, key: str):
        self.app, self.key = app, key

    def _ok(self, value: str | None) -> bool:
        return bool(value) and hmac.compare_digest(value.encode(), self.key.encode())

    @staticmethod
    def _remote(scope) -> bool:
        headers = dict(scope.get("headers") or [])
        if b"cf-connecting-ip" in headers:
            return True
        client = scope.get("client")
        return bool(client) and client[0] not in LOOPBACK

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope["path"] in OPEN_PATHS \
                or not self._remote(scope):
            return await self.app(scope, receive, send)

        query = parse_qsl(scope.get("query_string", b"").decode(), keep_blank_values=True)
        given = next((v for k, v in query if k == "k"), None)
        cookie = SimpleCookie()
        for name, value in scope.get("headers") or []:
            if name == b"cookie":
                cookie.load(value.decode("latin-1"))
        has_cookie = COOKIE in cookie and self._ok(cookie[COOKIE].value)

        if scope["type"] == "websocket":
            if has_cookie or self._ok(given):
                return await self.app(scope, receive, send)
            # accept then close, so the page sees code 4401 instead of a generic failure
            await receive()
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": 4401})
            return

        if self._ok(given):  # set the cookie and drop the key from the address bar
            rest = urlencode([(k, v) for k, v in query if k != "k"])
            location = scope["path"] + (f"?{rest}" if rest else "")
            await send({"type": "http.response.start", "status": 303, "headers": [
                (b"location", location.encode()),
                (b"set-cookie", f"{COOKIE}={self.key}; Path=/; HttpOnly; Secure; "
                                f"SameSite=Lax; Max-Age=86400".encode()),
                (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": b""})
            return
        if has_cookie:
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"text/html; charset=utf-8")]})
        await send({"type": "http.response.body", "body": DENIED_HTML})
