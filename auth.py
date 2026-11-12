"""auth.py — request authentication for the analysis API.

Contract shared with the Next.js frontend and the R server:

  * Browser requests carry   Authorization: Bearer <Firebase ID token>
  * Next.js server requests  X-Skari-Service-Key: <secret>

Enforcement is ON when env SKARI_FIREBASE_PROJECT is set (the Firebase
project id, which is also the ID token's `aud`). When it is unset the
middleware logs one warning and lets every request through — local dev only.

SKARI_BACKEND_KEY holds the service key, compared in constant time. Empty or
unset means service-key auth is unavailable (the header is then ignored).

ID tokens are verified locally with google-auth against Google's securetoken
x509 certificates (fetched once and cached for the response's Cache-Control
max-age). Checks: RS256 signature by a current Google key, exp, aud ==
project, iss == https://securetoken.google.com/<project>, non-empty sub,
iat not in the future (300 s skew allowed).

On success the middleware stores, in the ASGI scope's `state`:
    auth_kind  "user" | "service" | "anonymous" (only when enforcement is off)
    uid        the Firebase uid for "user", else None
so route handlers read them as request.state.auth_kind / request.state.uid.
"""

import base64
import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.request

logger = logging.getLogger("skari.auth")

CERTS_URL = ("https://www.googleapis.com/robot/v1/metadata/x509/"
             "securetoken@system.gserviceaccount.com")
CLOCK_SKEW_SECS = 300
SERVICE_KEY_HEADER = "x-skari-service-key"

# Paths that answer without credentials: liveness/readiness probes.
EXEMPT_PATHS = frozenset({"/", "/health"})


class AuthError(Exception):
    pass


def firebase_project() -> str:
    return os.environ.get("SKARI_FIREBASE_PROJECT", "").strip()


def service_key() -> str:
    return os.environ.get("SKARI_BACKEND_KEY", "")


def check_service_key(presented: str) -> bool:
    expected = service_key()
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


# ---------------------------------------------------------------------------
# Google certificate cache
# ---------------------------------------------------------------------------

def _default_cert_fetcher():
    """Return ({kid: pem_cert}, max_age_seconds)."""
    req = urllib.request.Request(CERTS_URL, headers={"User-Agent": "skari-backend"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read()
        cache_control = resp.headers.get("Cache-Control", "") or ""
    m = re.search(r"max-age=(\d+)", cache_control)
    max_age = int(m.group(1)) if m else 3600
    return json.loads(body), max_age


class CertCache:
    def __init__(self, fetcher=None):
        self._fetcher = fetcher or _default_cert_fetcher
        self._lock = threading.Lock()
        self._certs = None
        self._expires_at = 0.0
        self._last_forced = 0.0

    def set_fetcher(self, fetcher):
        with self._lock:
            self._fetcher = fetcher
            self._certs = None
            self._expires_at = 0.0
            self._last_forced = 0.0

    def get(self, force=False):
        now = time.time()
        with self._lock:
            # A forced refresh (unknown kid — Google rotated keys) is allowed
            # at most once a minute so a stream of junk kids cannot turn this
            # service into a request amplifier against googleapis.com.
            if force and now - self._last_forced < 60:
                force = False
            if self._certs is not None and now < self._expires_at and not force:
                return self._certs
            certs, max_age = self._fetcher()
            if not isinstance(certs, dict) or not certs:
                raise AuthError("certificate endpoint returned no keys")
            self._certs = certs
            self._expires_at = now + max(0, int(max_age))
            if force:
                self._last_forced = now
            return certs


cert_cache = CertCache()


def _b64url_json(segment: str) -> dict:
    pad = "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(segment + pad))


def verify_id_token(token: str, project: str, now: float = None) -> dict:
    """Verify a Firebase ID token; return its claims or raise AuthError."""
    from google.auth import jwt as gjwt

    if not token or token.count(".") != 2:
        raise AuthError("malformed token")
    try:
        header = _b64url_json(token.split(".", 1)[0])
    except Exception:
        raise AuthError("malformed token header")
    if header.get("alg") != "RS256":
        raise AuthError("unexpected token algorithm")
    kid = header.get("kid")
    if not kid or not isinstance(kid, str):
        raise AuthError("token has no key id")

    try:
        certs = cert_cache.get()
        if kid not in certs:
            certs = cert_cache.get(force=True)
    except AuthError:
        raise
    except Exception as e:  # network error fetching Google's keys
        logger.error("could not fetch Firebase signing certificates: %s", e)
        raise AuthError("signing certificates unavailable")
    if kid not in certs:
        raise AuthError("token signed by an unknown key")

    try:
        claims = gjwt.decode(token, certs={kid: certs[kid]}, audience=project,
                             clock_skew_in_seconds=CLOCK_SKEW_SECS)
    except Exception as e:  # google.auth.exceptions.* / ValueError
        raise AuthError(f"invalid token: {e}")

    now = time.time() if now is None else now
    if claims.get("iss") != f"https://securetoken.google.com/{project}":
        raise AuthError("wrong issuer")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub or len(sub) > 128:
        raise AuthError("missing subject")
    iat = claims.get("iat")
    if not isinstance(iat, (int, float)) or iat > now + CLOCK_SKEW_SECS:
        raise AuthError("token issued in the future")
    auth_time = claims.get("auth_time")
    if auth_time is not None and (not isinstance(auth_time, (int, float))
                                  or auth_time > now + CLOCK_SKEW_SECS):
        raise AuthError("invalid auth_time")
    return claims


# ---------------------------------------------------------------------------
# Middleware (pure ASGI, so it composes with the others without buffering)
# ---------------------------------------------------------------------------

_warned_disabled = False


def _warn_disabled_once():
    global _warned_disabled
    if not _warned_disabled:
        _warned_disabled = True
        logger.warning(
            "SKARI_FIREBASE_PROJECT is not set: API authentication is DISABLED. "
            "This is only acceptable for local development.")


def authenticate(headers: dict):
    """(auth_kind, uid) for a request's lower-cased headers, or raise AuthError.

    Returns ("anonymous", None) when enforcement is off and no valid
    credential was presented."""
    project = firebase_project()

    presented_key = headers.get(SERVICE_KEY_HEADER, "")
    if presented_key and check_service_key(presented_key):
        return "service", None

    authz = headers.get("authorization", "")
    if project and authz[:7].lower() == "bearer ":
        claims = verify_id_token(authz[7:].strip(), project)
        return "user", claims["sub"]

    if not project:
        _warn_disabled_once()
        return "anonymous", None
    raise AuthError("missing or invalid credentials")


class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        if scope.get("method") == "OPTIONS" or scope.get("path") in EXEMPT_PATHS:
            state.setdefault("auth_kind", "exempt")
            state.setdefault("uid", None)
            return await self.app(scope, receive, send)

        headers = {}
        for k, v in scope.get("headers") or []:
            headers[k.decode("latin-1").lower()] = v.decode("latin-1")
        try:
            kind, uid = authenticate(headers)
        except AuthError as e:
            logger.info("auth rejected %s %s: %s", scope.get("method"), scope.get("path"), e)
            body = json.dumps({"detail": "Authentication required."}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": body})
            return
        state["auth_kind"] = kind
        state["uid"] = uid
        return await self.app(scope, receive, send)
