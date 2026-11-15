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
    key     = HMAC-SHA256(key=SKARI_USAGE_TICKET_KEY or SKARI_BACKEND_KEY,
                          msg=b"skari-usage-ticket-v1")
    sig     = HMAC-SHA256(key=key, msg=the b64url payload, ASCII)

Mode, env SKARI_USAGE_TICKETS: off (default) | monitor (log, allow) | on (403).
Only signed-in browser requests are checked; service-key calls and requests
while authentication is off pass. With mode `on` and no key to check passes
with, metered runs are refused (fail closed), not waved through.

Key: SKARI_USAGE_TICKET_KEY when set, else SKARI_BACKEND_KEY (the original and
still the default). The separate key only works once the app
(src/lib/usage-ticket.ts) and r-backend/usage_ticket.R read the same variable,
so set it on all three together or not at all.

Two places check a pass:
  * check_request — the metered script routes in ROUTES (middleware).
  * check_train   — POST /api/models/{id}/train with engine "autogluon", which
    runs the same AutoGluon fit (models_api.train_model). That save follows a
    run that may itself have used the pass's whole ten minutes, so the same
    `automl` pass is accepted there for SAVE_GRACE seconds past its expiry,
    at most MAX_SAVE_USES times.

Use counts are per process (per Cloud Run instance) and are forgotten on a
restart: one pass is worth up to MAX_USES runs on EACH instance it reaches.
A shared counter would need a store every instance can reach on each request
(Firestore/Redis); not done — bound the multiple with --max-instances.
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


def signing_secret() -> str:
    """The secret passes are signed with: SKARI_USAGE_TICKET_KEY, else
    SKARI_BACKEND_KEY. Empty when neither is set."""
    return os.environ.get("SKARI_USAGE_TICKET_KEY", "") or os.environ.get("SKARI_BACKEND_KEY", "")


def ticket_key(service_key: str) -> bytes:
    return hmac.new(service_key.encode("utf-8"), CONTEXT, hashlib.sha256).digest()


def sign(payload: dict, service_key: str) -> str:
    """Used by the tests; the app is what issues passes in production."""
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(ticket_key(service_key), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url(sig)}"


def verify(ticket, uid, meter, service_key, now=None, grace=0):
    """(True, payload) or (False, reason). `grace` seconds past the expiry are
    still accepted (check_train only)."""
    if not isinstance(ticket, str) or not ticket or len(ticket) > 1024:
        return False, "missing"
    parts = ticket.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False, "malformed"
    # Both halves as ASCII bytes: headers arrive latin-1 decoded, and
    # compare_digest on two str raises TypeError (a 500) for any non-ASCII one.
    try:
        body_ascii = parts[0].encode("ascii")
        sig_ascii = parts[1].encode("ascii")
    except UnicodeEncodeError:
        return False, "malformed"
    want = _b64url(hmac.new(ticket_key(service_key), body_ascii, hashlib.sha256).digest())
    if not hmac.compare_digest(want.encode("ascii"), sig_ascii):
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
    if not isinstance(x, (int, float)) or isinstance(x, bool) or x < t - grace:
        return False, "expired"
    if x > t + TTL + 60:
        return False, "expiry"
    return True, p


_uses = {}
_uses_lock = threading.Lock()


def reset_uses():
    with _uses_lock:
        _uses.clear()


def _note_use(jti, expiry, now, max_uses=MAX_USES) -> bool:
    with _uses_lock:
        for k in [k for k, (_, x) in _uses.items() if x < now]:
            del _uses[k]
        n = _uses.get(jti, (0, expiry))[0] + 1
        _uses[jti] = (n, expiry)
        return n <= max_uses


def check_request(path, method, auth_kind, uid, headers, service_key, current_mode, now=None):
    """(True, None) to let the request through, or (False, reason)."""
    if current_mode == "off":
        return True, None
    meter = ROUTES.get(path)
    if meter is None or method != "POST":
        return True, None
    if auth_kind != "user" or not uid:  # service key, or authentication off
        return True, None
    return _check(path, meter, uid, headers, service_key, current_mode, now, 0, "", MAX_USES)


# How long after a pass expires it still covers saving that run's model, and
# how many saves (each a full AutoGluon fit) one pass covers.
SAVE_GRACE = 3600
MAX_SAVE_USES = 3


def check_train(engine, auth_kind, uid, headers, service_key, current_mode, now=None):
    """The pass check for POST /api/models/{id}/train. Only engine "autogluon"
    is metered -- it is the same fit /api/analysis/autogluon runs; the
    fourteen-estimator path costs about a second and is not."""
    if current_mode == "off":
        return True, None
    if (engine or "").strip().lower() != "autogluon":
        return True, None
    if auth_kind != "user" or not uid:
        return True, None
    return _check("/api/models/{id}/train", "automl", uid, headers, service_key, current_mode,
                  now, SAVE_GRACE, "save:", MAX_SAVE_USES)


def _check(path, meter, uid, headers, service_key, current_mode, now, grace, use_prefix, max_uses):
    if not service_key:
        if current_mode == "on":
            logger.error("[usage-ticket] no SKARI_USAGE_TICKET_KEY / SKARI_BACKEND_KEY -- "
                         "cannot check usage passes; metered run refused (SKARI_USAGE_TICKETS=on)")
            return False, "unconfigured"
        logger.warning("[usage-ticket] no key -- cannot check usage passes; allowed (monitor)")
        return True, None
    tk = headers.get(HEADER, "")
    if not tk:
        ok, info = False, "missing"
    else:
        ok, info = verify(tk, uid, meter, service_key, now, grace)
    t = int(time.time() if now is None else now)
    if ok and not _note_use(use_prefix + str(info.get("j")), float(info.get("x")) + grace, t, max_uses):
        ok, info = False, "reused"
    if ok:
        return True, None
    if current_mode == "monitor":
        logger.warning("[usage-ticket] monitor %s: %s pass -- allowed (SKARI_USAGE_TICKETS=monitor)", path, info)
        return True, None
    return False, info
