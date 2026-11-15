"""check_security.py — the security fixes, each against the attack it exists for.

    pip install httpx            # TestClient's dependency (dev only)
    python scripts/check_security.py

No network, no GCS, no Firebase: ID tokens are signed with a locally
generated RSA key whose self-signed cert is injected into auth.cert_cache,
and model_store's bucket is replaced by an in-memory fake.

Sections:
  1. _optexpr: the generator-frame RCE payload and its relatives are refused
     before anything runs; normal objectives / constraint LHS still evaluate.
  2. auth middleware: Firebase ID token checks (sig, aud, iss, exp, iat, sub,
     kid, alg), service key, exemptions, CORS preflight, uid on request.state.
  3. rate-limit key: uid / service bucket / LAST X-Forwarded-For entry.
  4. body size limit and subprocess timeout.
  5. model store: only signed models/{id}/<ts>/pipeline.joblib objects are
     ever deserialized; delete requires the owner.
  6. end to end through FastAPI: the optimisation analyses, and Model Lab
     train -> predict -> delete.
  7. 2026-11 audit fixes (area C, Python): fail-closed auth (M2), real path in
     the middlewares (L1), no tracebacks / API docs (L2), scrubbed script
     environment (L4), per-IP / per-day / in-flight ceilings (L5, M3), the
     tuned-parameter allowlist (H2), predict bound to the owner (H3), the
     AutoGluon save metered (M1), usage-pass edge cases (L6), bounded model
     cache and the owner purge (M3, E-06).
"""

import datetime
import io
import json
import os
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# Make sure nothing in the environment leaks into the checks.
for _k in ('SKARI_FIREBASE_PROJECT', 'SKARI_BACKEND_KEY', 'SKARI_MODEL_SIGNING_KEY',
           'SKARI_AUTH_DISABLED', 'SKARI_USAGE_TICKET_KEY', 'SKARI_USAGE_TICKETS', 'SKARI_API_DOCS'):
    os.environ.pop(_k, None)

_ok = 0
_failed = 0


def check(cond, msg, *extra):
    global _ok, _failed
    if cond:
        _ok += 1
        print(f'ok    {msg}')
    else:
        _failed += 1
        print(f'FAIL  {msg}')
        for e in extra:
            print(f'        {e}')


def raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:
        return e
    return None


MARKER = os.path.join(tempfile.gettempdir(), f'skari_pwned_{os.getpid()}')


# ---------------------------------------------------------------------------
# 1. expression evaluator
# ---------------------------------------------------------------------------
def section_expr():
    from _optexpr import make_func

    v = ['x1', 'x2']
    bad = [
        "x1+[*(g:=(g.gi_frame.f_back.f_back.f_builtins['_'+'_import'+'_'+'_']('os').system('id') for _ in [1]))][0]",
        f"[c for c in ().__class__.__base__.__subclasses__()]",
        "(lambda: 1)()",
        "(g:=1)",
        "x1.real",
        "x1.__class__",
        "[x1][0]",
        "{x1: 1}",
        "'abc'",
        "x1 if x2 else 1",
        "x1 < 2",
        "x1 and x2",
        "exp(x=1)",
        "exp(*[x1])",
        "__import__('os')",
        "open('/etc/passwd')",
        "getattr(x1, 'real')",
        "np.exp(x1)",
        "exp(x1, x1)",
        "unknown_var + 1",
        "9**9**9",
        "2**(10**4)",
        "x1 ** 1e6",
        "f'{x1}'",
        "x1 @ x2",
        "x1 << 2",
        "~x1",
        "True + x1",
        "1j * x1",
        "x" * 2001,
        "(" * 300 + "x1" + ")" * 300,
        "+".join(["x1"] * 400),
    ]
    for b in bad:
        e = raises(lambda: make_func(b, v)([1.0, 2.0]), ValueError)
        check(e is not None, f'refused: {b[:70]!r}')
    check(not os.path.exists(MARKER), 'no refused expression ran code')

    good = [
        ('(x1-3)**2 + exp(x2)', [1.0, 0.0], 5.0),
        ('(x1 - 3)**2 + (x2 - 2)**2', [3.0, 2.0], 0.0),
        ('x1**2 + x2**2', [1.0, 2.0], 5.0),
        ('x1^2 + x2^2', [1.0, 2.0], 5.0),
        ('x1 + x2', [1.0, 2.0], 3.0),
        ('exp(x1) + exp(x2) - x1 - x2', [0.0, 0.0], 2.0),
        ('3*x1 + 2*x2', [1.0, 1.0], 5.0),
        ('-x1 + +x2', [1.0, 3.0], 2.0),
        ('2*pi*x1 + e', [1.0, 0.0], 2 * 3.141592653589793 + 2.718281828459045),
        ('log(x1) + log10(x2) + sqrt(x2)', [1.0, 100.0], 12.0),
        ('abs(x1 - 5) + maximum(x1, x2) + minimum(x1, x2) + power(x2, 2)', [1.0, 2.0], 4 + 2 + 1 + 4),
        ('sin(x1)**2 + cos(x1)**2 + tanh(0) + sign(x2)', [0.7, -3.0], 0.0),
        ('x2 % 3 + x2 // 3 + x1 / 4', [2.0, 7.0], 1 + 2 + 0.5),
        ('x1 * x2 / (1 + x1**2)', [1.0, 4.0], 2.0),
        ('1e3 * x1 - 2.5e-1', [1.0, 0.0], 999.75),
    ]
    for expr, x, want in good:
        try:
            got = make_func(expr, v)(x)
            check(abs(got - want) < 1e-9, f'evaluates: {expr!r} = {want}', f'got {got}')
        except Exception as e:
            check(False, f'evaluates: {expr!r}', repr(e))
    # The same callable handles numpy input (SLSQP passes ndarrays).
    import numpy as np
    check(make_func('x1*x2', v)(np.array([2.0, 3.0])) == 6.0, 'accepts numpy vectors')


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
PROJECT = 'skari-test-project'


def make_keypair(cn='test'):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .sign(key, hashes.SHA256()))
    key_pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode()
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    return key_pem, cert_pem


def make_token(key_pem, kid, **over):
    from google.auth import crypt, jwt
    now = int(time.time())
    claims = {
        'iss': f'https://securetoken.google.com/{PROJECT}', 'aud': PROJECT,
        'sub': 'user-a', 'user_id': 'user-a', 'iat': now - 10, 'exp': now + 3600,
        'auth_time': now - 10,
    }
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    signer = crypt.RSASigner.from_string(key_pem, key_id=kid)
    return jwt.encode(signer, claims).decode()


def b64(obj):
    import base64
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b'=').decode()


# ---------------------------------------------------------------------------
# 2-4. auth, rate limit, body limit, timeout
# ---------------------------------------------------------------------------
def section_auth(client, main, auth, key_pem, cert_pem, other_key_pem):
    from fastapi import Request

    @main.app.get('/__whoami')
    def _whoami(request: Request):
        return {'kind': request.state.auth_kind, 'uid': request.state.uid}
    # Routes registered after startup still resolve (router is consulted live).

    fetches = []

    def fetcher():
        fetches.append(1)
        return {'kid-1': cert_pem}, 3600
    auth.cert_cache.set_fetcher(fetcher)

    # No project and no explicit switch: fail closed (M2).
    r = client.get('/__whoami')
    check(r.status_code == 503 and 'not configured' in r.json().get('detail', ''),
          'SKARI_FIREBASE_PROJECT unset: every route 503 (fail closed)', f'{r.status_code} {r.text}')
    check(client.get('/health').status_code == 200, 'SKARI_FIREBASE_PROJECT unset: /health still 200')
    os.environ['SKARI_BACKEND_KEY'] = 'svc-only'
    r = client.get('/__whoami', headers={'X-Skari-Service-Key': 'svc-only'})
    check(r.status_code == 200 and r.json()['kind'] == 'service',
          'SKARI_FIREBASE_PROJECT unset: the service key still works', r.text)
    os.environ.pop('SKARI_BACKEND_KEY', None)
    os.environ['SKARI_AUTH_DISABLED'] = '1'
    r = client.get('/__whoami')
    check(r.status_code == 200 and r.json()['kind'] == 'anonymous',
          'SKARI_AUTH_DISABLED=1 and no project: requests allowed (local dev)', r.text)
    os.environ['SKARI_AUTH_DISABLED'] = 'yes'
    check(client.get('/__whoami').status_code == 503, "SKARI_AUTH_DISABLED must be exactly '1'")
    os.environ['SKARI_AUTH_DISABLED'] = '1'

    os.environ['SKARI_FIREBASE_PROJECT'] = PROJECT
    try:
        r = client.get('/__whoami')
        check(r.status_code == 401 and r.json() == {'detail': 'Authentication required.'},
              'no credentials -> 401 JSON (SKARI_AUTH_DISABLED=1 is ignored once a project is set)',
              f'{r.status_code} {r.text}')
        check(client.get('/health').status_code == 200, '/health exempt')
        check(client.get('/').status_code == 200, '/ exempt')
        r = client.options('/api/analysis/nonlinear-programming', headers={
            'Origin': 'https://app.example', 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'authorization,x-skari-service-key,content-type'})
        allow = r.headers.get('access-control-allow-headers', '').lower()
        check(r.status_code == 200 and 'authorization' in allow and 'x-skari-service-key' in allow,
              'CORS preflight answered without auth, allows Authorization + X-Skari-Service-Key',
              f'{r.status_code} {dict(r.headers)}')
        check('access-control-allow-credentials' not in r.headers,
              'CORS credentials not enabled')
        r = client.get('/__whoami', headers={'Origin': 'https://app.example'})
        check(r.status_code == 401 and r.headers.get('access-control-allow-origin') == '*',
              '401 still carries CORS headers (browser can read the error)')

        tok = make_token(key_pem, 'kid-1')
        r = client.get('/__whoami', headers={'Authorization': f'Bearer {tok}'})
        check(r.status_code == 200 and r.json() == {'kind': 'user', 'uid': 'user-a'},
              'valid ID token -> 200, uid on request.state', r.text)
        for prov in ('google.com', 'password', 'custom'):
            t = make_token(key_pem, 'kid-1', firebase={'sign_in_provider': prov})
            r = client.get('/__whoami', headers={'Authorization': f'Bearer {t}'})
            check(r.status_code == 200, f'signed-in provider {prov} -> 200', f'{r.status_code} {r.text}')
        n = len(fetches)
        client.get('/__whoami', headers={'Authorization': f'Bearer {tok}'})
        check(len(fetches) == n, 'certs cached (no refetch within max-age)')

        now = int(time.time())
        bad_tokens = {
            'expired': make_token(key_pem, 'kid-1', iat=now - 7200, exp=now - 3600, auth_time=now - 7200),
            'wrong audience': make_token(key_pem, 'kid-1', aud='other-project'),
            'wrong issuer': make_token(key_pem, 'kid-1', iss='https://securetoken.google.com/other'),
            'iat in the future': make_token(key_pem, 'kid-1', iat=now + 900, exp=now + 4000),
            'empty sub': make_token(key_pem, 'kid-1', sub=''),
            'no sub': make_token(key_pem, 'kid-1', sub=None),
            'signed by another key': make_token(other_key_pem, 'kid-1'),
            'unknown kid': make_token(other_key_pem, 'kid-unknown'),
            'alg none': b64({'alg': 'none', 'kid': 'kid-1'}) + '.' + b64(
                {'iss': f'https://securetoken.google.com/{PROJECT}', 'aud': PROJECT,
                 'sub': 'x', 'iat': now, 'exp': now + 60}) + '.',
            'alg HS256': b64({'alg': 'HS256', 'kid': 'kid-1', 'typ': 'JWT'}) + '.' + b64(
                {'iss': f'https://securetoken.google.com/{PROJECT}', 'aud': PROJECT,
                 'sub': 'x', 'iat': now, 'exp': now + 60}) + '.c2ln',
            'garbage': 'not-a-token',
            'anonymous account': make_token(key_pem, 'kid-1', firebase={'sign_in_provider': 'anonymous'}),
        }
        for label, t in bad_tokens.items():
            r = client.get('/__whoami', headers={'Authorization': f'Bearer {t}'})
            check(r.status_code == 401, f'ID token rejected: {label}', f'{r.status_code} {r.text}')

        # Service key.
        r = client.get('/__whoami', headers={'X-Skari-Service-Key': 'anything'})
        check(r.status_code == 401, 'SKARI_BACKEND_KEY unset: service-key auth unavailable')
        os.environ['SKARI_BACKEND_KEY'] = 's3cret-service-key'
        r = client.get('/__whoami', headers={'X-Skari-Service-Key': 's3cret-service-key'})
        check(r.status_code == 200 and r.json() == {'kind': 'service', 'uid': None},
              'correct service key -> 200 (kind=service)', r.text)
        r = client.get('/__whoami', headers={'X-Skari-Service-Key': 's3cret-service-kez'})
        check(r.status_code == 401, 'wrong service key -> 401')

        # ---- 3. rate-limit key -------------------------------------------
        from starlette.requests import Request as SReq

        def req(headers, state=None, client_addr=('198.51.100.9', 1)):
            scope = {'type': 'http', 'method': 'POST', 'path': '/x', 'headers':
                     [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                     'client': client_addr, 'query_string': b''}
            if state is not None:
                scope['state'] = state
            return SReq(scope)
        k1 = main._rl_key(req({'X-Forwarded-For': '1.1.1.1, 203.0.113.5'}))[0]
        k2 = main._rl_key(req({'X-Forwarded-For': '9.9.9.9, 203.0.113.5'}))[0]
        check(k1 == k2 == '203.0.113.5',
              'rate-limit key = LAST X-Forwarded-For entry (spoofed first entry ignored)', f'{k1} {k2}')
        check(main._rl_key(req({}))[0] == '198.51.100.9', 'no XFF -> socket peer')
        check(main._rl_key(req({}, {'auth_kind': 'user', 'uid': 'u1'}))[0] == 'uid:u1',
              'authenticated -> keyed by uid')
        check(main._rl_key(req({}, {'auth_kind': 'service', 'uid': None}))[0] == 'service',
              'service-key callers share one bucket')
        # The limiter really keys on uid: rotate XFF, still the same bucket.
        main._rl_store.clear()
        statuses = []
        for i in range(main._RL_MAX_REQUESTS + 1):
            statuses.append(client.get('/__whoami', headers={
                'Authorization': f'Bearer {tok}', 'X-Forwarded-For': f'10.0.{i // 250}.{i % 250}'}).status_code)
        check(statuses[-2] == 200 and statuses[-1] == 429,
              'per-uid limit holds while X-Forwarded-For rotates', str(statuses[-3:]))
        main._rl_store.clear()
    finally:
        os.environ.pop('SKARI_FIREBASE_PROJECT', None)
        os.environ.pop('SKARI_BACKEND_KEY', None)

    # ---- 4. body limit + timeout ----------------------------------------
    # (local-dev mode from here: the body limit sits outside authentication)
    big = b'{"a": "' + b'x' * (main._MAX_BODY_BYTES + 10) + b'"}'
    r = client.post('/api/analysis/nonlinear-programming', content=big,
                    headers={'Content-Type': 'application/json'})
    check(r.status_code == 413, f'body over {main._MAX_BODY_BYTES // (1024 * 1024)} MB -> 413 (Content-Length)', r.status_code)

    def chunks():
        for _ in range(main._MAX_BODY_BYTES // (1024 * 1024) + 2):
            yield b'x' * (1024 * 1024)
    r = client.post('/api/analysis/nonlinear-programming', content=chunks(),
                    headers={'Content-Type': 'application/json'})
    check(r.status_code == 413, 'chunked body over the limit -> 413', r.status_code)

    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, 'sleepy.py'), 'w') as fh:
        fh.write('import time, subprocess, sys\n'
                 'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
                 'time.sleep(30)\n')
    real_dir, real_to = main._BACKEND_DIR, main._SCRIPT_TIMEOUT_SECS
    main._BACKEND_DIR, main._SCRIPT_TIMEOUT_SECS = tmp, 1
    t0 = time.time()
    try:
        e = raises(lambda: main.run_script('sleepy.py', {}))
    finally:
        main._BACKEND_DIR, main._SCRIPT_TIMEOUT_SECS = real_dir, real_to
    check(e is not None and getattr(e, 'status_code', 0) == 504 and time.time() - t0 < 10,
          'run_script: a script past the timeout is killed -> 504', repr(e))
    os.environ.pop('SKARI_AUTH_DISABLED', None)


# ---------------------------------------------------------------------------
# 5. model store
# ---------------------------------------------------------------------------
class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name
        self.metadata = None
        self.generation = None

    def upload_from_file(self, fh, content_type=None):
        self.store[self.name] = (fh.read(), dict(self.metadata or {}))

    def download_as_bytes(self):
        return self.store[self.name][0]

    def delete(self, if_generation_match=None):
        self.store.pop(self.name, None)


class FakeBucket:
    def __init__(self):
        self.store = {}
        self.downloads = []

    def blob(self, name):
        return FakeBlob(self.store, name)

    def list_blobs(self, prefix=''):
        out = []
        for name in list(self.store):
            if name.startswith(prefix):
                b = FakeBlob(self.store, name)
                b.metadata = dict(self.store[name][1])
                out.append(b)
        return out

    def get_blob(self, name):
        if name not in self.store:
            return None
        b = FakeBlob(self.store, name)
        b.metadata = dict(self.store[name][1])
        b.generation = 1
        orig = b.download_as_bytes

        def dl():
            self.downloads.append(name)
            return orig()
        b.download_as_bytes = dl
        return b


class Boom:
    """Unpickling this creates MARKER — the stand-in for arbitrary code."""
    def __reduce__(self):
        return (open, (MARKER, 'w'))


def section_model_store(model_store, fake):
    import joblib
    import pickle  # noqa: F401  (joblib.dump uses it)

    os.environ['SKARI_MODEL_SIGNING_KEY'] = 'model-signing-key-1'
    bucket = model_store._BUCKET_NAME
    uri = model_store.save_pipeline('m1', {'hello': 'world'}, 'user-a')
    path = uri[len(f'gs://{bucket}/'):]
    check(path.startswith('models/m1/') and path.endswith('/pipeline.joblib'),
          'save_pipeline writes models/{id}/<ts>/pipeline.joblib', uri)
    check(model_store.load_pipeline(uri, 'm1') == {'hello': 'world'},
          'a signed artifact loads for its own model id')
    model_store._pipeline_cache.clear()
    fake.downloads.clear()
    e = raises(lambda: model_store.load_pipeline(uri, 'm1', caller_uid='user-b', is_service=False),
               model_store.NotOwner)
    check(e is not None and not fake.downloads,
          'load for another account -> NotOwner, refused from metadata before any download')
    check(model_store.load_pipeline(uri, 'm1', caller_uid='user-a', is_service=False) == {'hello': 'world'},
          'load for the signed owner')
    e = raises(lambda: model_store.load_pipeline(uri, 'm1', caller_uid='user-b', is_service=False),
               model_store.NotOwner)
    check(e is not None, 'a cached artifact is still refused to another account')
    e = raises(lambda: model_store.load_pipeline(uri, 'm1', caller_uid=None, is_service=False,
                                                 auth_enforced=False), model_store.NotOwner)
    check(e is not None, 'local-dev anonymous caller cannot load an artifact that names an owner')
    model_store._pipeline_cache.clear()

    e = raises(lambda: model_store.load_pipeline(uri, 'm2'), ValueError)
    check(e is not None, "another model's id cannot load it (path bound to model id)")

    def plant(p, obj, meta=None):
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        fake.store[p] = (buf.getvalue(), meta or {})

    # A user-writable location (storage.rules allows skari/{uid}/**).
    plant('skari/user-b/evil.json', Boom())
    fake.downloads.clear()
    e = raises(lambda: model_store.load_pipeline(f'gs://{bucket}/skari/user-b/evil.json', 'm1'), ValueError)
    check(e is not None and not fake.downloads,
          'URI outside models/{id}/ refused before download')
    e = raises(lambda: model_store.load_pipeline(f'gs://{bucket}/models/m1/../../skari/user-b/evil.json', 'm1'), ValueError)
    check(e is not None, "'..' in the path refused")
    e = raises(lambda: model_store.load_pipeline(f'gs://other-bucket/models/m1/1/pipeline.joblib', 'm1'), ValueError)
    check(e is not None, 'other bucket refused')

    # Unsigned pickle planted at a valid-looking path.
    plant('models/m1/123/pipeline.joblib', Boom())
    e = raises(lambda: model_store.load_pipeline(f'gs://{bucket}/models/m1/123/pipeline.joblib', 'm1'), ValueError)
    check(e is not None and not os.path.exists(MARKER), 'unsigned artifact refused, never unpickled', repr(e))

    # Malicious bytes under a copied, valid signature.
    good_meta = fake.store[path][1]
    plant('models/m1/124/pipeline.joblib', Boom(), dict(good_meta))
    e = raises(lambda: model_store.load_pipeline(f'gs://{bucket}/models/m1/124/pipeline.joblib', 'm1'), ValueError)
    check(e is not None and not os.path.exists(MARKER), 'copied signature on other bytes refused', repr(e))

    # Signature made with a different key.
    os.environ['SKARI_MODEL_SIGNING_KEY'] = 'attacker-guess'
    # The artifact folder is the save time in ms; on a fast machine this save can land in
    # the same millisecond as the first one and overwrite it, leaving `forged` empty.
    time.sleep(0.02)
    model_store.save_pipeline('m1', Boom(), 'user-b')
    forged = [p for p in fake.store if p.startswith('models/m1/') and p not in
              (path, 'models/m1/123/pipeline.joblib', 'models/m1/124/pipeline.joblib')]
    os.environ['SKARI_MODEL_SIGNING_KEY'] = 'model-signing-key-1'
    e = raises(lambda: model_store.load_pipeline(f'gs://{bucket}/{forged[0]}', 'm1'), ValueError)
    check(e is not None and not os.path.exists(MARKER), 'signature under another key refused', repr(e))

    # Delete.
    e = raises(lambda: model_store.delete_artifact(uri, 'm1', caller_uid='user-b'), model_store.NotOwner)
    check(e is not None and path in fake.store, 'delete by a non-owner refused, object kept')
    e = raises(lambda: model_store.delete_artifact(
        f'gs://{bucket}/models/m1/123/pipeline.joblib', 'm1', caller_uid='user-a'), model_store.NotOwner)
    check(e is not None, 'delete of an unsigned object by a user refused')
    check(model_store.delete_artifact(f'gs://{bucket}/skari/user-b/evil.json', 'm1', is_service=True) is False
          and 'skari/user-b/evil.json' in fake.store,
          'delete outside models/{id}/ is a no-op, even for the service key')
    check(model_store.delete_artifact(uri, 'm1', caller_uid='user-a') is True and path not in fake.store,
          'owner deletes their own artifact')
    check(model_store.delete_artifact(f'gs://{bucket}/models/m1/123/pipeline.joblib', 'm1', is_service=True)
          and 'models/m1/123/pipeline.joblib' not in fake.store,
          'service key may delete a (legacy) artifact under models/{id}/')
    check(not os.path.exists(MARKER), 'no planted pickle ever executed')


# ---------------------------------------------------------------------------
# 6. end to end
# ---------------------------------------------------------------------------
def section_e2e(client, main, auth, model_store, fake, key_pem):
    os.environ['SKARI_FIREBASE_PROJECT'] = PROJECT
    os.environ['SKARI_BACKEND_KEY'] = 's3cret-service-key'
    svc = {'X-Skari-Service-Key': 's3cret-service-key'}
    try:
        # Nonlinear programming — the page's own default problem.
        nlp = {'objective': '(x1 - 3)**2 + (x2 - 2)**2', 'maximize': False,
               'var_names': ['x1', 'x2'], 'start': [0, 0],
               'con_expr': ['x1**2 + x2**2', 'x1 + x2'], 'con_ops': ['<=', '>='],
               'con_rhs': [9, 1], 'con_names': ['Capacity', 'Minimum']}
        r = client.post('/api/analysis/nonlinear-programming', json=nlp)
        check(r.status_code == 401, 'analysis route without credentials -> 401')
        r = client.post('/api/analysis/nonlinear-programming', json=nlp, headers=svc)
        ok = r.status_code == 200
        res = r.json().get('results', {}) if ok else {}
        check(ok and res.get('objective_value') is not None and abs(res['objective_value'] - (13 ** 0.5 - 3) ** 2) < 1e-3,
              'nonlinear-programming end to end (service key) -> optimum on the circle',
              f'{r.status_code} {str(r.text)[:300]}')

        r = client.post('/api/analysis/convex-optimization', headers=svc, json={
            'objective': 'exp(x1) + exp(x2) - x1 - x2', 'var_names': ['x1', 'x2'],
            'start': [1, 1], 'con_expr': ['x1 + x2'], 'con_ops': ['<='], 'con_rhs': [2]})
        check(r.status_code == 200 and abs(r.json()['results']['objective_value'] - 2.0) < 1e-3,
              'convex-optimization end to end', f'{r.status_code} {r.text[:300]}')

        r = client.post('/api/analysis/constrained-optimization', headers=svc, json={
            'objective': 'x1*x2', 'maximize': True, 'var_names': ['x1', 'x2'], 'start': [1, 1],
            'con_expr': ['x1 + x2'], 'con_types': ['eq'], 'con_ops': ['='], 'con_rhs': [10]})
        check(r.status_code == 200 and abs(r.json()['results']['objective_value'] - 25.0) < 1e-3,
              'constrained-optimization end to end', f'{r.status_code} {r.text[:300]}')

        r = client.post('/api/analysis/multi-objective-optimization', headers=svc, json={
            'var_names': ['x1', 'x2'], 'obj_expr': ['3*x1 + 2*x2', '-x1 + 4*x2'],
            'obj_maximize': [True, True], 'con_matrix': [[1, 1]], 'con_ops': ['<='],
            'con_rhs': [10], 'n_pareto_points': 5})
        check(r.status_code == 200 and r.json().get('results', {}).get('pareto_front'),
              'multi-objective-optimization end to end', f'{r.status_code} {r.text[:300]}')

        evil = dict(nlp, objective="x1+[*(g:=(g.gi_frame.f_back.f_back.f_builtins['_'+'_import'+'_'+'_']"
                                   f"('os').system('touch {MARKER}') for _ in [1]))][0]")
        r = client.post('/api/analysis/nonlinear-programming', json=evil, headers=svc)
        check(r.status_code == 400 and not os.path.exists(MARKER),
              'RCE payload through the HTTP route -> 400, nothing ran', f'{r.status_code} {r.text[:200]}')

        # Model Lab: train -> predict -> delete, as two different users.
        tok_a = make_token(key_pem, 'kid-1', sub='user-a')
        tok_b = make_token(key_pem, 'kid-1', sub='user-b')
        A = {'Authorization': f'Bearer {tok_a}'}
        B = {'Authorization': f'Bearer {tok_b}'}
        rows = [{'f1': i % 7, 'f2': (i * 3) % 5, 'y': 'yes' if i % 7 > 3 else 'no'} for i in range(60)]
        r = client.post('/api/models/abc123/train', headers=A, json={
            'data': rows, 'algorithm': 'Decision Tree', 'target': 'y',
            'features': ['f1', 'f2'], 'task': 'classification'})
        uri = r.json().get('artifactUri') if r.status_code == 200 else None
        check(bool(uri) and '/models/abc123/' in uri, 'train (user A) -> artifactUri, unchanged response shape',
              f'{r.status_code} {r.text[:300]}')
        if uri:
            model_store._pipeline_cache.clear()
            r = client.post('/api/models/abc123/predict', headers=A, json={
                'artifactUri': uri, 'rows': [{'f1': 6, 'f2': 1}]})
            check(r.status_code == 200 and r.json()['predictions'] == ['yes'],
                  'predict (owner) with the returned artifactUri -> 200, same shape', f'{r.status_code} {r.text[:300]}')
            r = client.post('/api/models/abc123/predict', headers=B, json={
                'artifactUri': uri, 'rows': [{'f1': 6, 'f2': 1}], 'explain': True})
            check(r.status_code == 403 and 'shapImportance' not in r.text,
                  'predict/explain by another account -> 403 (H3)', f'{r.status_code} {r.text[:200]}')
            r = client.post('/api/models/abc123/predict', headers=svc, json={
                'artifactUri': uri, 'rows': [{'f1': 6, 'f2': 1}]})
            check(r.status_code == 200 and r.json()['predictions'] == ['yes'],
                  'predict with the service key (public form route) -> 200', f'{r.status_code} {r.text[:200]}')
            r = client.post('/api/models/zzz999/predict', headers=A, json={
                'artifactUri': uri, 'rows': [{'f1': 6, 'f2': 1}]})
            check(r.status_code == 403, "predict naming another model's artifact -> 403", r.status_code)
            bucket = model_store._BUCKET_NAME
            r = client.post('/api/models/abc123/predict', headers=A, json={
                'artifactUri': f'gs://{bucket}/skari/user-b/evil.json', 'rows': [{'f1': 1, 'f2': 1}]})
            check(r.status_code == 403 and not os.path.exists(MARKER),
                  'predict on a user-uploaded object -> 403, not unpickled', r.status_code)
            path = uri.split(f'gs://{bucket}/', 1)[1]
            r = client.delete('/api/models/abc123', params={'artifactUri': uri}, headers=B)
            check(r.status_code == 403 and path in fake.store, 'DELETE by another user -> 403', r.status_code)
            r = client.delete('/api/models/abc123', params={'artifactUri': f'gs://{bucket}/skari/user-b/evil.json'},
                              headers=A)
            check(r.status_code == 200 and 'skari/user-b/evil.json' in fake.store,
                  'DELETE of an arbitrary object -> ignored', r.status_code)
            r = client.delete('/api/models/abc123', params={'artifactUri': uri}, headers=A)
            check(r.status_code == 200 and r.json() == {'ok': True} and path not in fake.store,
                  'DELETE by the owner -> 200 {ok: true}, object gone', r.status_code)
            r = client.delete('/api/models/abc123', params={'artifactUri': uri})
            check(r.status_code == 401, 'DELETE without credentials -> 401')
    finally:
        os.environ.pop('SKARI_FIREBASE_PROJECT', None)
        os.environ.pop('SKARI_BACKEND_KEY', None)


# ---------------------------------------------------------------------------
# usage pass (metered AutoML runs) — usage_ticket.py
# ---------------------------------------------------------------------------
# Same string the app signs in src/lib/__tests__/usage-ticket-check.ts and the
# R server checks in r-backend/tests/check-usage-ticket.R.
USAGE_VECTOR = ('eyJ2IjoxLCJ1IjoidXNlci0xMjMiLCJtIjoiZXJnbSIsIngiOjE5MDAwMDAwMDAsImoiOiJhYmNkZWYwMTIzNDU2Nzg5MDEiLCJzIjowfQ'
                '.oitflVXsqKiZk2bl2V8UpgdIwv7kRm8B0NNy3y-PACQ')


def section_usage_ticket(client, main, key_pem):
    import usage_ticket as ut
    K = 'test-service-key'
    AT = 1899999900
    check(ut.sign({'v': 1, 'u': 'user-123', 'm': 'ergm', 'x': 1900000000, 'j': 'abcdef012345678901', 's': 0}, K) == USAGE_VECTOR,
          'usage pass: Python signs the shared vector byte for byte')
    check(ut.verify(USAGE_VECTOR, 'user-123', 'ergm', K, AT)[0], 'usage pass: shared vector verifies')
    for label, args, want in [
        ('another account', ('user-9', 'ergm', K, AT), 'account'),
        ('another meter', ('user-123', 'automl', K, AT), 'meter'),
        ('another key', ('user-123', 'ergm', 'other', AT), 'signature'),
        ('expired', ('user-123', 'ergm', K, 1900000001), 'expired'),
    ]:
        r = ut.verify(USAGE_VECTOR, *args)
        check(r == (False, want), f'usage pass: {label} refused', r)
    far = ut.sign({'v': 1, 'u': 'user-123', 'm': 'ergm', 'x': AT + 86400, 'j': 'x', 's': 0}, K)
    check(ut.verify(far, 'user-123', 'ergm', K, AT) == (False, 'expiry'), 'usage pass: far-future expiry refused')
    body, sig = USAGE_VECTOR.split('.')
    forged = ut._b64url(b'{"v":1,"u":"user-123","m":"automl","x":1900000000,"j":"abcdef012345678901","s":0}')
    check(ut.verify(f'{forged}.{sig}', 'user-123', 'automl', K, AT) == (False, 'signature'),
          'usage pass: edited payload (meter swapped) refused')
    check(ut.verify('', 'u', 'automl', K, AT) == (False, 'missing') and ut.verify('abc', 'u', 'automl', K, AT)[1] == 'malformed',
          'usage pass: empty / garbage refused')
    check(ut.mode('') == 'off' and ut.mode(' ON') == 'on' and ut.mode('typo') == 'off', 'usage pass: mode default off')

    # Through the app, on a probe route standing in for /api/analysis/autogluon.
    @main.app.post('/__ticketprobe')
    def _probe():
        return {'ran': True}
    ut.ROUTES['/__ticketprobe'] = 'automl'
    os.environ['SKARI_FIREBASE_PROJECT'] = PROJECT
    os.environ['SKARI_BACKEND_KEY'] = K
    main._rl_store.clear()
    ut.reset_uses()
    try:
        tok = make_token(key_pem, 'kid-1', sub='user-a')
        H = {'Authorization': f'Bearer {tok}'}
        now = int(time.time())
        good = ut.sign({'v': 1, 'u': 'user-a', 'm': 'automl', 'x': now + 600, 'j': 'p1', 's': 0}, K)

        os.environ.pop('SKARI_USAGE_TICKETS', None)
        check(client.post('/__ticketprobe', headers=H).status_code == 200, 'usage pass: mode off -> run without a pass')
        os.environ['SKARI_USAGE_TICKETS'] = 'monitor'
        check(client.post('/__ticketprobe', headers=H).status_code == 200, 'usage pass: monitor -> no pass logged, allowed')
        os.environ['SKARI_USAGE_TICKETS'] = 'on'
        r = client.post('/__ticketprobe', headers=H)
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_missing' and 'Refresh' in r.json().get('detail', ''),
              'usage pass: on, no pass -> 403 usage_ticket_missing (bilingual detail)', f'{r.status_code} {r.text}')
        r = client.post('/__ticketprobe', headers={**H, 'X-Skari-Usage-Ticket': good})
        check(r.status_code == 200 and r.json() == {'ran': True}, 'usage pass: on, valid pass -> runs', r.text)
        codes = [client.post('/__ticketprobe', headers={**H, 'X-Skari-Usage-Ticket': good}).status_code for _ in range(3)]
        check(codes == [200, 200, 403], 'usage pass: one pass serves 3 uses (retries), the 4th is refused', codes)
        other = ut.sign({'v': 1, 'u': 'user-b', 'm': 'automl', 'x': now + 600, 'j': 'p2', 's': 0}, K)
        r = client.post('/__ticketprobe', headers={**H, 'X-Skari-Usage-Ticket': other})
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_account', "usage pass: someone else's pass -> 403")
        ergm = ut.sign({'v': 1, 'u': 'user-a', 'm': 'ergm', 'x': now + 600, 'j': 'p3', 's': 0}, K)
        r = client.post('/__ticketprobe', headers={**H, 'X-Skari-Usage-Ticket': ergm})
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_meter', 'usage pass: a network pass does not open AutoML')
        r = client.post('/__ticketprobe', headers={'X-Skari-Service-Key': K})
        check(r.status_code == 200, 'usage pass: service-key call needs no pass', f'{r.status_code} {r.text}')
        r = client.post('/api/analysis/nonlinear-programming', headers=H, json={})
        check(r.json().get('code', '') == '' if r.headers.get('content-type', '').startswith('application/json') else True,
              'usage pass: unmetered route is not checked', r.text[:120])
        r = client.options('/api/analysis/autogluon', headers={
            'Origin': 'https://app.example', 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'authorization,content-type,x-skari-usage-ticket'})
        check(r.status_code == 200 and 'x-skari-usage-ticket' in r.headers.get('access-control-allow-headers', '').lower(),
              'usage pass: CORS preflight allows X-Skari-Usage-Ticket', f'{r.status_code} {dict(r.headers)}')
    finally:
        os.environ.pop('SKARI_USAGE_TICKETS', None)
        os.environ.pop('SKARI_FIREBASE_PROJECT', None)
        os.environ.pop('SKARI_BACKEND_KEY', None)
        ut.ROUTES.pop('/__ticketprobe', None)
        main._rl_store.clear()


# ---------------------------------------------------------------------------
# 7. 2026-11 audit fixes (area C, Python)
# ---------------------------------------------------------------------------
def section_audit_2026_11(client, main, auth, model_store, fake, key_pem):
    import inspect
    import socket
    import threading
    import usage_ticket as ut
    import models_api
    import algorithm_registry as ar

    K = 's3cret-service-key'
    os.environ['SKARI_FIREBASE_PROJECT'] = PROJECT
    os.environ['SKARI_BACKEND_KEY'] = K
    os.environ['SKARI_MODEL_SIGNING_KEY'] = 'model-signing-key-1'
    svc = {'X-Skari-Service-Key': K}
    A = {'Authorization': 'Bearer ' + make_token(key_pem, 'kid-1', sub='user-a')}
    B = {'Authorization': 'Bearer ' + make_token(key_pem, 'kid-1', sub='user-b')}
    main._rl_store.clear(); main._rl_ip_store.clear(); main._rl_day_store.clear()
    rows = [{'f1': i % 7, 'f2': (i * 3) % 5, 'y': 'yes' if i % 7 > 3 else 'no'} for i in range(60)]
    base = {'data': rows, 'target': 'y', 'features': ['f1', 'f2'], 'task': 'classification'}
    try:
        # ---- H2: tuned-parameter allowlist --------------------------------
        def train(mid, headers=A, **over):
            return client.post(f'/api/models/{mid}/train', headers=headers, json={**base, **over})

        lst = socket.socket(); lst.bind(('127.0.0.1', 0)); lst.listen(1)
        port = lst.getsockname()[1]
        seen = []

        def acc():
            lst.settimeout(3)
            try:
                c, addr = lst.accept(); seen.append(addr); c.close()
            except Exception:
                pass
        t = threading.Thread(target=acc, daemon=True); t.start()
        t0 = time.time()
        r = train('h2lgb', algorithm='LightGBM', params={
            'n_estimators': 3, 'tree_learner': 'data', 'num_machines': 2, 'time_out': 1,
            'machines': f'127.0.0.1:{port},127.0.0.1:{port + 1}', 'local_listen_port': port + 1})
        t.join(4); lst.close()
        check(r.status_code == 400 and not seen and time.time() - t0 < 10,
              'H2: LightGBM network parameters -> 400 before any fit, no outbound connection',
              f'{r.status_code} {r.text[:200]} seen={seen}')
        out_dir = tempfile.mkdtemp(prefix='skari_h2_')
        target = os.path.join(out_dir, 'catboost_dir')
        r = train('h2cat', algorithm='CatBoost', params={
            'iterations': 5, 'train_dir': target, 'allow_writing_files': True,
            'save_snapshot': True, 'snapshot_file': os.path.join(out_dir, 'json.py')})
        check(r.status_code == 400 and not os.path.exists(target) and os.listdir(out_dir) == [],
              'H2: CatBoost train_dir / snapshot_file -> 400, nothing written', f'{r.status_code} {r.text[:200]}')
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        for label, algo, params in [
            ('thread_count', 'CatBoost', {'thread_count': 64}),
            ('n_jobs', 'Random Forest', {'n_jobs': -1}),
            ('n_estimators above the ceiling', 'Random Forest', {'n_estimators': 10 ** 7}),
            ('learning_rate 0', 'XGBoost', {'learning_rate': 0}),
            ('a bool for an int', 'GBM', {'max_depth': True}),
            ('a string for a float', 'LightGBM', {'subsample': '0.5'}),
            ('NaN', 'XGBoost', {'subsample': float('nan')}),
            ('max_features as a path', 'Random Forest', {'max_features': '/etc/passwd'}),
            ('params for an untuned model', 'Decision Tree', {'max_depth': 3}),
            ('nthread (xgboost)', 'XGBoost', {'nthread': 64}),
        ]:
            e = raises(lambda: ar.build_estimator(algo, 'classification', params), ValueError)
            check(e is not None, f'H2: refused -- {label}', repr(params))
        good = {
            'Random Forest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 5,
                              'min_samples_leaf': 2, 'max_features': 'sqrt'},
            'GBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.85},
            'XGBoost': {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.7,
                        'colsample_bytree': 1.0, 'min_child_weight': 3},
            'LightGBM': {'n_estimators': 100, 'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': -1,
                         'subsample': 1.0, 'colsample_bytree': 0.85},
            'CatBoost': {'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3},
        }
        for algo, params in good.items():
            try:
                est = ar.build_estimator(algo, 'classification', params)
                got = est.get_params()
                check(all(got.get(k) == v for k, v in params.items()),
                      f'H2: a tune_analysis.py best_params set still applies -- {algo}')
            except Exception as ex:
                check(False, f'H2: a tune_analysis.py best_params set still applies -- {algo}', repr(ex))
        check(ar.validate_params('LightGBM', {'n_estimators': 200.0}) == {'n_estimators': 200},
              'H2: 200.0 for an int parameter is taken as 200')
        r = train('h2ok', algorithm='Random Forest', params={'n_estimators': 20, 'max_depth': 4})
        check(r.status_code == 200 and r.json().get('artifactUri'),
              'H2: train with allowed tuned params -> 200', f'{r.status_code} {r.text[:200]}')

        # ---- H3: predict bound to the owner; service key never sees rows ---
        r = train('h3model', algorithm='Decision Tree')
        uri = r.json().get('artifactUri') if r.status_code == 200 else None
        check(bool(uri), 'H3: owner trains a model', r.text[:200])
        model_store._pipeline_cache.clear()
        fake.downloads.clear()
        req = {'artifactUri': uri, 'rows': [{'f1': 6, 'f2': 1}], 'explain': True}
        r = client.post('/api/models/h3model/predict', headers=B, json=req)
        check(r.status_code == 403 and not fake.downloads,
              'H3: another account -> 403, artifact never downloaded', f'{r.status_code} {fake.downloads}')
        r = client.post('/api/models/h3model/predict', headers=A, json=req)
        pts = [p for f in ((r.json().get('shapImportance') or {}).get('features') or []) for p in f['points']]
        check(r.status_code == 200 and pts and all('value' in p for p in pts),
              'H3: owner explain unchanged (beeswarm points carry their values)', f'{r.status_code} {r.text[:200]}')
        r = client.post('/api/models/h3model/predict', headers=B, json=req)
        check(r.status_code == 403, 'H3: another account still 403 once the artifact is cached', r.status_code)
        r = client.post('/api/models/h3model/predict', headers=svc, json=req)
        j = r.json() if r.status_code == 200 else {}
        pts = [p for f in ((j.get('shapImportance') or {}).get('features') or []) for p in f['points']]
        check(r.status_code == 200 and pts and not any('value' in p for p in pts)
              and all('meanAbsShap' in f for f in j['shapImportance']['features']),
              'H3: service-key explain -> SHAP kept, training-row values stripped', f'{r.status_code} {r.text[:200]}')
        r = client.post('/api/models/h3model/predict', headers=A, json={
            'artifactUri': uri, 'rows': [{'f1': 1, 'f2': 1}] * (models_api.MAX_EXPLAIN_ROWS + 1), 'explain': True})
        check(r.status_code == 400, f'M3: explain over {models_api.MAX_EXPLAIN_ROWS} rows -> 400', r.status_code)
        real = models_api.MAX_PREDICT_ROWS
        models_api.MAX_PREDICT_ROWS = 5
        try:
            r = client.post('/api/models/h3model/predict', headers=A, json={
                'artifactUri': uri, 'rows': [{'f1': 1, 'f2': 1}] * 6})
            check(r.status_code == 400, 'M3: predict over the row cap -> 400', r.status_code)
        finally:
            models_api.MAX_PREDICT_ROWS = real
        real = models_api.MAX_TRAIN_ROWS
        models_api.MAX_TRAIN_ROWS = 30
        try:
            r = train('m3rows', algorithm='Decision Tree')
            check(r.status_code == 400 and 'Too many rows' in r.text, 'M3: train over the row cap -> 400', r.text[:200])
        finally:
            models_api.MAX_TRAIN_ROWS = real
        r = train('m3preset', algorithm='x', engine='autogluon', preset='/tmp/evil')
        check(r.status_code == 400 and 'preset' in r.text, 'M3: unknown AutoGluon preset -> 400', r.text[:200])

        # ---- E-06: purge every artifact of one account (service key only) --
        train('e06a2', algorithm='Decision Tree')
        r = train('e06b1', headers=B, algorithm='Decision Tree')
        b_uri = r.json().get('artifactUri', '')
        def signed_for(name, meta):
            try:
                return model_store._verify(name, meta) == 'user-a'
            except ValueError:
                return False  # e.g. section 5's planted copy: not servable, not purged
        a_paths = [n for n, (_, m) in fake.store.items() if signed_for(n, m)]
        r = client.post('/api/models/owner-purge', headers=A, json={'uid': 'user-a'})
        check(r.status_code == 403 and all(p in fake.store for p in a_paths),
              'E-06: owner purge with a user token -> 403', r.status_code)
        r = client.post('/api/models/owner-purge', headers=svc, json={'uid': 'user-a'})
        check(r.status_code == 200 and r.json().get('deleted') == len(a_paths) >= 3
              and not any(p in fake.store for p in a_paths)
              and b_uri.split('/', 3)[-1] in fake.store,
              "E-06: service-key purge deletes all of that account's artifacts, nobody else's",
              f'{r.status_code} {r.text} {len(a_paths)}')
        r = client.post('/api/models/owner-purge', headers=svc, json={'uid': ''})
        check(r.status_code == 400, 'E-06: purge with an empty uid -> 400', r.status_code)

        # ---- M2: local-dev mode cannot delete someone's artifact -----------
        os.environ.pop('SKARI_FIREBASE_PROJECT')
        os.environ['SKARI_AUTH_DISABLED'] = '1'
        r = client.delete('/api/models/e06b1', params={'artifactUri': b_uri})
        check(r.status_code == 403 and b_uri.split('/', 3)[-1] in fake.store,
              'M2: with authentication off, an owned artifact still cannot be deleted', r.status_code)
        os.environ.pop('SKARI_AUTH_DISABLED')
        os.environ['SKARI_FIREBASE_PROJECT'] = PROJECT

        # ---- M1 + L6: usage pass on the AutoGluon save ----------------------
        ut.reset_uses()
        os.environ['SKARI_USAGE_TICKETS'] = 'on'
        now = int(time.time())
        ag = dict(algorithm='AutoGluon', engine='autogluon', target='nope', timeLimit=600)

        def save(headers):
            return client.post('/api/models/m1save/train', headers=headers, json={**base, **ag})

        r = save(A)
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_missing',
              'M1: /train engine=autogluon without a pass -> 403 usage_ticket_missing', f'{r.status_code} {r.text[:200]}')
        r = train('m1plain', algorithm='Decision Tree')
        check(r.status_code == 200, 'M1: /train without engine needs no pass', r.status_code)
        r = save(svc)
        check(r.json().get('code', '') == '' and r.status_code != 403,
              'M1: service-key /train engine=autogluon needs no pass', f'{r.status_code} {r.text[:120]}')
        p = ut.sign({'v': 1, 'u': 'user-a', 'm': 'automl', 'x': now - 1800, 'j': 'save1', 's': 0}, K)
        codes = [save({**A, 'X-Skari-Usage-Ticket': p}) for _ in range(ut.MAX_SAVE_USES + 1)]
        check(all(c.status_code != 403 for c in codes[:-1])
              and codes[-1].status_code == 403 and codes[-1].json().get('code') == 'usage_ticket_reused',
              f'M1: a run\'s pass (expired 30 min ago) covers {ut.MAX_SAVE_USES} saves, then 403 reused',
              [(c.status_code, c.text[:80]) for c in codes])
        old = ut.sign({'v': 1, 'u': 'user-a', 'm': 'automl', 'x': now - ut.SAVE_GRACE - 60, 'j': 'save2', 's': 0}, K)
        r = save({**A, 'X-Skari-Usage-Ticket': old})
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_expired',
              'M1: a pass older than the save grace -> 403 expired', r.text[:200])
        net = ut.sign({'v': 1, 'u': 'user-a', 'm': 'ergm', 'x': now + 600, 'j': 'save3', 's': 0}, K)
        r = save({**A, 'X-Skari-Usage-Ticket': net})
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_meter',
              'M1: a network-model pass does not open the AutoGluon save', r.text[:200])
        ok_, why = ut.check_request('/api/analysis/autogluon', 'POST', 'user', 'user-a', {}, '', 'on')
        check((ok_, why) == (False, 'unconfigured'),
              'L6: mode on with no key -> metered run refused, not allowed', (ok_, why))
        check(ut.verify('abc.déf', 'u', 'automl', K)[1] == 'malformed',
              'L6: non-ASCII signature -> malformed (no TypeError)')
        ut.ROUTES['/__l6probe'] = 'automl'

        @main.app.post('/__l6probe')
        def _l6():
            return {'ran': True}
        r = client.post('/__l6probe', headers={**A, 'X-Skari-Usage-Ticket': 'abc.déf'.encode('latin-1')})
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_malformed',
              'L6: non-ASCII pass through the app -> 403 malformed, not 500', f'{r.status_code} {r.text[:120]}')

        # ---- L1: the middlewares follow the router's path, not Host ---------
        r = client.post('/__l6probe', headers={**A, 'Host': 'x/health?'})
        check(r.status_code == 403 and r.json().get('code') == 'usage_ticket_missing',
              'L1: Host: x/health? no longer skips the usage pass', f'{r.status_code} {r.text[:120]}')
        main._rl_store.clear()
        codes = [client.post('/__l6probe', headers={**A, 'Host': 'x/health?'}).status_code
                 for _ in range(main._RL_MAX_REQUESTS + 2)]
        check(codes.count(429) >= 2, 'L1: Host: x/health? no longer skips the rate limit', codes[-3:])
        main._rl_store.clear()
        os.environ.pop('SKARI_USAGE_TICKETS', None)
        ut.ROUTES.pop('/__l6probe', None)

        # ---- L5: per-IP (counts 401s), per-uid per day ----------------------
        real_ip, real_day = main._RL_IP_MAX_REQUESTS, main._RL_DAY_MAX_REQUESTS
        main._RL_IP_MAX_REQUESTS, main._RL_DAY_MAX_REQUESTS = 5, 4
        try:
            main._rl_ip_store.clear()
            ip = {'X-Forwarded-For': '203.0.113.77'}
            codes = [client.get('/__whoami', headers=ip).status_code for _ in range(6)]
            check(codes == [401] * 5 + [429], 'L5: rejected (401) requests count toward the per-IP ceiling', codes)
            r = client.get('/__whoami', headers={**ip, **svc})
            check(r.status_code == 200, 'L5: the service key is exempt from the per-IP ceiling', r.status_code)
            check(client.get('/health', headers=ip).status_code == 200, 'L5: /health exempt from the per-IP ceiling')
            main._rl_ip_store.clear(); main._rl_store.clear(); main._rl_day_store.clear()
            codes = [client.get('/__whoami', headers={**A, 'X-Forwarded-For': f'198.51.100.{i}'}).status_code
                     for i in range(5)]
            r = client.get('/__whoami', headers={**A, 'X-Forwarded-For': '198.51.100.99'})
            check(codes == [200] * 4 + [429] and 'Daily' in r.text,
                  'L5: per-account daily ceiling (SKARI_RL_UID_PER_DAY)', codes)
        finally:
            main._RL_IP_MAX_REQUESTS, main._RL_DAY_MAX_REQUESTS = real_ip, real_day
            main._rl_ip_store.clear(); main._rl_store.clear(); main._rl_day_store.clear()

        # ---- M3: in-flight cap, conjoint routes off the event loop, LRU ----
        main._inflight['uid:user-a'] = main._INFLIGHT_MAX
        try:
            r = client.get('/__whoami', headers=A)
            check(r.status_code == 429 and 'running at once' in r.text,
                  f'M3: a {main._INFLIGHT_MAX + 1}th request in flight for one account -> 429', r.status_code)
            r = client.get('/__whoami', headers=B)
            check(r.status_code == 200, 'M3: another account is not affected', r.status_code)
        finally:
            main._inflight.pop('uid:user-a', None)
        check(client.get('/__whoami', headers=A).status_code == 200 and not main._inflight,
              'M3: in-flight slots are released after each response')
        sync = {r.path: not inspect.iscoroutinefunction(r.endpoint) for r in main.app.routes
                if getattr(r, 'path', '') in ('/api/analysis/rating-conjoint', '/api/analysis/conjoint',
                                              '/api/analysis/adaptive-conjoint', '/api/analysis/conjoint-hb',
                                              '/api/analysis/adaptive-cbc')}
        check(len(sync) == 5 and all(sync.values()),
              'M3: the five conjoint routes are plain def (thread pool, not the event loop)', sync)
        real_max = model_store._CACHE_MAX
        model_store._CACHE_MAX = 2
        try:
            model_store._pipeline_cache.clear()
            ag_dir = tempfile.mkdtemp(prefix='ag_serve_')
            uris = []
            for i in range(3):
                time.sleep(0.002)
                uris.append(model_store.save_pipeline('lru', {'i': i, '_ag_dir': ag_dir if i == 0 else None}, 'user-a'))
            for u in uris:
                model_store.load_pipeline(u, 'lru')
            check(len(model_store._pipeline_cache) == 2 and not os.path.exists(ag_dir),
                  'M3: model cache is LRU-bounded; eviction removes the ag_serve_* directory',
                  f'{len(model_store._pipeline_cache)} {os.path.exists(ag_dir)}')
        finally:
            model_store._CACHE_MAX = real_max
            model_store._pipeline_cache.clear()

        # ---- L2: no API docs, no tracebacks ---------------------------------
        for path in ('/openapi.json', '/docs', '/redoc'):
            r = client.get(path, headers=A)
            check(r.status_code == 404, f'L2: {path} not served (SKARI_API_DOCS unset)', r.status_code)
        r = client.post('/api/analysis/nps', headers={**A, 'Content-Type': 'application/json'}, content=b'{')
        check(r.status_code == 400, 'L2: malformed JSON on a script route -> 400, not 500', r.status_code)
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, 'crash.py'), 'w') as fh:
            fh.write('import os, sys, json\n'
                     'leaked = sorted(k for k in os.environ if k.startswith("SKARI_"))\n'
                     'cwd_is_backend = os.path.abspath(os.getcwd()) == os.path.abspath(sys.argv[0]).rsplit("/", 1)[0]\n'
                     'if json.load(sys.stdin).get("crash"):\n'
                     '    raise RuntimeError("boom at /app/secret_path.py")\n'
                     'print(json.dumps({"leaked": leaked, "cwd_is_script_dir": cwd_is_backend, "cwd": os.getcwd()}))\n')
        real_dir = main._BACKEND_DIR
        main._BACKEND_DIR = tmp
        os.environ['SKARI_USAGE_TICKET_KEY'] = 'k2'
        try:
            e = raises(lambda: main.run_script('crash.py', {'crash': True}))
            d = getattr(e, 'detail', '')
            check(getattr(e, 'status_code', 0) == 400 and 'Traceback' not in d and 'secret_path' not in d
                  and 'ref ' in d, 'L2: an uncaught script traceback -> generic message with a reference', d)
            out = main.run_script('crash.py', {})
            check(out['leaked'] == [], 'L4: analysis subprocesses get no SKARI_* secrets', out['leaked'])
            check(not out['cwd_is_script_dir'] and not os.path.exists(out['cwd']),
                  'L4: each run gets its own working directory, removed afterwards', out)
        finally:
            main._BACKEND_DIR = real_dir
            os.environ.pop('SKARI_USAGE_TICKET_KEY', None)
            shutil.rmtree(tmp, ignore_errors=True)

        # ---- L4: separate usage-pass key, with the old one as fallback ------
        check(ut.signing_secret() == K, 'L4: usage-pass key falls back to SKARI_BACKEND_KEY')
        os.environ['SKARI_USAGE_TICKET_KEY'] = 'separate'
        check(ut.signing_secret() == 'separate', 'L4: SKARI_USAGE_TICKET_KEY is used when set')
        os.environ.pop('SKARI_USAGE_TICKET_KEY', None)
    finally:
        for k in ('SKARI_FIREBASE_PROJECT', 'SKARI_BACKEND_KEY', 'SKARI_USAGE_TICKETS',
                  'SKARI_AUTH_DISABLED', 'SKARI_USAGE_TICKET_KEY'):
            os.environ.pop(k, None)
        main._rl_store.clear(); main._rl_ip_store.clear(); main._rl_day_store.clear()
        ut.reset_uses()


def main_():
    section_expr()

    from fastapi.testclient import TestClient
    import main
    import auth
    import model_store

    fake = FakeBucket()
    model_store._bucket = lambda: fake

    key_pem, cert_pem = make_keypair('kid-1')
    other_key_pem, _ = make_keypair('other')
    client = TestClient(main.app)

    section_auth(client, main, auth, key_pem, cert_pem, other_key_pem)
    main._rl_store.clear()
    section_model_store(model_store, fake)
    main._rl_store.clear()
    section_e2e(client, main, auth, model_store, fake, key_pem)
    main._rl_store.clear()
    section_usage_ticket(client, main, key_pem)
    main._rl_store.clear()
    section_audit_2026_11(client, main, auth, model_store, fake, key_pem)

    if os.path.exists(MARKER):
        os.remove(MARKER)
        check(False, 'MARKER file was created — some payload executed')
    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main_())
