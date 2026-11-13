"""usage_ticket.py — the signed usage pass for metered runs (AutoML / autogluon).

The monthly AutoML limit is counted by the app (POST /api/usage/consume), but
runs come straight here from the browser, so a script could skip the count.
The app now signs a short "usage pass" when it counts a run and the browser
sends it as X-Skari-Usage-Ticket. This module only checks it — the plan logic
stays in the app (statistica-frontend src/lib/usage-server.ts).

FORMAT — identical to statistica-frontend src/lib/usage-ticket.ts and
r-backend/usage_ticket.R (change all three together; the shared test vector is
in scripts/check_security.py):
    ticket  = b64url(payload_json) "." b64url(sig)
    payload = {"v":1,"u":uid,"m":meter,"x":expiry_unix_s,"j":random_id,"s":0|1}
    key     = HMAC-SHA256(key=SKARI_BACKEND_KEY, msg=b"skari-usage-ticket-v1")
    sig     = HMAC-SHA256(key=key, msg=the b64url payload, ASCII)

Mode, env SKARI_USAGE_TICKETS: off (default) | monitor (log, allow) | on (403).
Only signed-in browser requests are checked; service-key calls and requests
while authentication is off pass.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time

logger = logging.getLogger("usage_ticket")

HEADER = "x-skari-usage-ticket"
ROUTES = {"/api/analysis/autogluon": "automl"}
TTL = 600
MAX_USES = 3  # a cold-start retry may present the same pass again
CONTEXT = b"skari-usage-ticket-v1"

MESSAGE = ("No usage pass for this run (monthly limit check). Refresh the page and run it again. · "
           "이번 실행의 사용권이 없습니다(월 한도 확인). 페이지를 새로고침한 뒤 다시 실행하세요.")


def mode(raw=None) -> str:
    v = (os.environ.get("SKARI_USAGE_TICKETS", "") if raw is None else raw).strip().lower()
    return v if v in ("on", "monitor") else "off"


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _from_b64url(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def ticket_key(service_key: str) -> bytes:
    return hmac.new(service_key.encode("utf-8"), CONTEXT, hashlib.sha256).digest()


def sign(payload: dict, service_key: str) -> str:
    """Used by the tests; the app is what issues passes in production."""
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(ticket_key(service_key), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url(sig)}"


def verify(ticket, uid, meter, service_key, now=None):
    """(True, payload) or (False, reason)."""
    if not isinstance(ticket, str) or not ticket or len(ticket) > 1024:
        return False, "missing"
    parts = ticket.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False, "malformed"
    try:
        body_ascii = parts[0].encode("ascii")
    except UnicodeEncodeError:
        return False, "malformed"
    want = _b64url(hmac.new(ticket_key(service_key), body_ascii, hashlib.sha256).digest())
    if not hmac.compare_digest(want, parts[1]):
        return False, "signature"
    try:
        p = json.loads(_from_b64url(parts[0]))
    except Exception:
        return False, "malformed"
    if not isinstance(p, dict) or p.get("v") != 1:
        return False, "version"
    if p.get("u") != uid:
        return False, "account"
    if p.get("m") != meter:
        return False, "meter"
    t = int(time.time() if now is None else now)
    x = p.get("x")
    if not isinstance(x, (int, float)) or x < t:
        return False, "expired"
    if x > t + TTL + 60:
        return False, "expiry"
    return True, p


_uses = {}
_uses_lock = threading.Lock()


def reset_uses():
    with _uses_lock:
        _uses.clear()


def _note_use(jti, expiry, now) -> bool:
    with _uses_lock:
        for k in [k for k, (_, x) in _uses.items() if x < now]:
            del _uses[k]
        n = _uses.get(jti, (0, expiry))[0] + 1
        _uses[jti] = (n, expiry)
        return n <= MAX_USES


def check_request(path, method, auth_kind, uid, headers, service_key, current_mode, now=None):
    """(True, None) to let the request through, or (False, reason)."""
    if current_mode == "off":
        return True, None
    meter = ROUTES.get(path)
    if meter is None or method != "POST":
        return True, None
    if auth_kind != "user" or not uid:  # service key, or authentication off
        return True, None
    if not service_key:
        logger.warning("[usage-ticket] SKARI_BACKEND_KEY is empty -- cannot check usage passes; allowed")
        return True, None
    tk = headers.get(HEADER, "")
    if not tk:
        ok, info = False, "missing"
    else:
        ok, info = verify(tk, uid, meter, service_key, now)
    t = int(time.time() if now is None else now)
    if ok and not _note_use(str(info.get("j")), float(info.get("x")), t):
        ok, info = False, "reused"
    if ok:
        return True, None
    if current_mode == "monitor":
        logger.warning("[usage-ticket] monitor %s: %s pass -- allowed (SKARI_USAGE_TICKETS=monitor)", path, info)
        return True, None
    return False, info
