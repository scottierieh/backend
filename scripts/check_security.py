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
for _k in ('SKARI_FIREBASE_PROJECT', 'SKARI_BACKEND_KEY', 'SKARI_MODEL_SIGNING_KEY'):
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

    # Enforcement off: allowed, anonymous.
    r = client.get('/__whoami')
    check(r.status_code == 200 and r.json()['kind'] == 'anonymous',
          'SKARI_FIREBASE_PROJECT unset: requests allowed (dev mode)', r.text)

    os.environ['SKARI_FIREBASE_PROJECT'] = PROJECT
    try:
        r = client.get('/__whoami')
        check(r.status_code == 401 and r.json() == {'detail': 'Authentication required.'},
              'no credentials -> 401 JSON', f'{r.status_code} {r.text}')
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
            r = client.post('/api/models/abc123/predict', headers=B, json={
                'artifactUri': uri, 'rows': [{'f1': 6, 'f2': 1}]})
            check(r.status_code == 200 and r.json()['predictions'] == ['yes'],
                  'predict with the returned artifactUri -> 200, same shape', f'{r.status_code} {r.text[:300]}')
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

    if os.path.exists(MARKER):
        os.remove(MARKER)
        check(False, 'MARKER file was created — some payload executed')
    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main_())
