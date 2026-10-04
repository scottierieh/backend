"""check_startup.py — the boot contract, asked of the app rather than of me.

    python scripts/check_startup.py          # a few seconds, no network

Everything in this service hangs off one module-level table in main.py:
SCRIPT_ROUTES maps 119 endpoint paths to 119 script files, and a loop hands
each pair to register_script_route(). That loop checks NOTHING. A script that
was renamed, deleted, or never added still registers a route perfectly
happily; the endpoint exists, answers, and returns HTTP 400 with a Python
"can't open file" message — but only for whoever clicks that one analysis,
and only in production. The same is true of a duplicate path (Starlette
matches in registration order, so the second one is unreachable code) and of
a script importing a package requirements.txt does not install.

None of those are boot failures, which is exactly why they need a boot check:
the container comes up healthy and reports nothing.

So this imports the real app and asks IT what it serves — main.app's own
OpenAPI schema, not a re-parse of main.py. A check that re-derives the route
table from the source it is checking agrees with itself by construction.

What it establishes:

  1. main imports at all — the actual Cloud Run boot path, routers included
  2. every SCRIPT_ROUTES path is served, and the table has no duplicate path
     or duplicate script
  3. every registered script exists on disk and compiles
  4. every top-level third-party import is pinned in requirements.txt
  5. the six mounted routers each contributed endpoints — include_router() on
     a router whose decorators moved is silent, and adds nothing
  6. no *_analysis.py on disk is unwired, and none listed has gone missing
  7. the shared error path: a script that fails gives the user ITS message,
     not a traceback
  8. the rate limiter, which sits in front of all 127 endpoints, counts and
     exempts /health

Sections 7 and 8 drive the ASGI app by hand rather than through
fastapi.testclient, which needs httpx — a dev dependency this image does not
carry, and a check that cannot run where the code runs is not a check.

A green run says nothing unless a red one is possible, and every check above
is an assertion about code it does not control. So:

    python scripts/check_startup.py --prove   # ~2 min

breaks each of those eleven things in a throwaway copy of the backend — a
deleted script, a duplicated path, an unpinned import, an unmounted router, a
limiter that forgot to count — and requires the matching line to turn red. The
copy is discarded; nothing here touches the working tree.
"""

import ast
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

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


# Modules that are NOT analysis scripts: the five conjoint/survey routers, the
# model registry, and the shared helpers. Anything else ending in .py that is
# not in SCRIPT_ROUTES is either dead or forgotten wiring, and section 6 says
# which.
_ROUTER_MODULES = {
    'rc_analysis.py': '/api/analysis/rating-conjoint',
    'conjoint_analysis.py': '/api/analysis/conjoint',
    'aca_analysis.py': '/api/analysis/adaptive-conjoint',
    'conjoint_hb.py': '/api/analysis/conjoint-hb',
    'acbc_analysis.py': '/api/analysis/adaptive-cbc',
}
_MODELS_PATHS = {
    '/api/models/{model_id}/train',
    '/api/models/{model_id}/predict',
    '/api/models/{model_id}',
}
_HELPER_MODULES = {
    'main.py', 'models_api.py', 'model_store.py', 'analysis_common.py',
    'cv_strategy.py', 'feature_pipeline.py', 'blend_registry.py',
    'algorithm_registry.py', 'guardrails.py', 'hb_mnl.py', '_optexpr.py',
}

# requirements.txt pins distributions; scripts import modules. Where the two
# names differ, the mapping is here.
_DIST_TO_IMPORT = {
    'scikit-learn': 'sklearn',
    'scikit-learn-extra': 'sklearn_extra',
    'google-cloud-storage': 'google',
    'autogluon.tabular': 'autogluon',
    'tensorflow-cpu': 'tensorflow',
    'umap-learn': 'umap',
    'pillow': 'PIL',
    'pyportfolioopt': 'pypfopt',
    'empyrical-reloaded': 'empyrical',
}

# Modules that arrive with a pin rather than as a pin of their own. Each entry
# names what brings it, so a NEW unpinned import cannot hide among them.
_SHIPS_WITH = {
    'mpl_toolkits': 'ships inside the matplotlib wheel',
    'pydantic': 'a hard dependency of fastapi',
    'starlette': 'a hard dependency of fastapi',
}


def script_routes():
    """SCRIPT_ROUTES as literal data, without executing main.py."""
    tree = ast.parse(open(os.path.join(_ROOT, 'main.py'), encoding='utf-8').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', '') == 'SCRIPT_ROUTES' for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def pinned_imports():
    names = set()
    for line in open(os.path.join(_ROOT, 'requirements.txt'), encoding='utf-8'):
        line = line.split('#')[0].strip()
        if not line:
            continue
        dist = line.split('==')[0].split('>=')[0].split('[')[0].strip()
        names.add(_DIST_TO_IMPORT.get(dist.lower(), dist.replace('-', '_')))
    return names


def top_level_imports(path):
    """Module names imported at MODULE level — the ones that run on import.

    An import inside a function is deferred and cannot break a boot, so it is
    deliberately not collected here.
    """
    tree = ast.parse(open(path, encoding='utf-8').read(), path)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split('.')[0])
    return found


# ---------------------------------------------------------------------------
# A minimal ASGI caller. Enough to put one request through the real
# middleware stack and read the status back.
# ---------------------------------------------------------------------------

def call(app, method, path, ip='10.0.0.1', body=b''):
    scope = {
        'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.1'},
        'http_version': '1.1', 'method': method, 'scheme': 'http',
        'path': path, 'raw_path': path.encode(), 'query_string': b'',
        'root_path': '', 'client': ('127.0.0.1', 5000),
        'server': ('testserver', 80),
        'headers': [(b'host', b'testserver'), (b'x-forwarded-for', ip.encode()),
                    (b'content-type', b'application/json'),
                    (b'content-length', str(len(body)).encode())],
    }
    sent = []

    async def receive():
        return {'type': 'http.request', 'body': body, 'more_body': False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(m['status'] for m in sent if m['type'] == 'http.response.start')
    payload = b''.join(m.get('body', b'') for m in sent
                       if m['type'] == 'http.response.body')
    return status, payload


def main():
    routes = script_routes()
    # The count is written down on purpose. Deriving it from the table would
    # make this line agree with itself; stated, a route added or lost shows up
    # here and in the doc rather than only in a diff.
    check(routes is not None and len(routes) == 119,
          f'SCRIPT_ROUTES is a literal table of 119 entries '
          f'({len(routes) if routes else 0} read)')
    if not routes:
        print(f'\n{_ok} ok, {_failed} failure(s)')
        return 1

    paths = [r[0] for r in routes]
    scripts = [r[1] for r in routes]

    # --- 1. the real boot path -------------------------------------------
    started = time.time()
    try:
        import main as app_module  # noqa: F401
        boot_error = None
    except Exception as exc:  # pragma: no cover - the thing being checked
        boot_error = f'{type(exc).__name__}: {exc}'
    check(boot_error is None,
          f'main imports — routers, registry and all {len(routes)} registrations '
          f'({time.time() - started:.1f}s)',
          boot_error or '')
    if boot_error:
        print('\n' + 'nothing below can be checked without an app.')
        print(f'{_ok} ok, {_failed} failure(s)')
        return 1

    served = app_module.app.openapi()['paths']

    # --- 2. the table is internally sound, and the app agrees ------------
    dup_paths = sorted({p for p in paths if paths.count(p) > 1})
    check(not dup_paths,
          'no path is registered twice — Starlette matches in registration '
          'order, so a second registration is unreachable',
          *dup_paths)

    dup_scripts = sorted({s for s in scripts if scripts.count(s) > 1})
    check(not dup_scripts,
          'no script is wired to two paths, which is normally a copy-paste '
          'slip that left one path on the wrong script',
          *dup_scripts)

    unserved = [p for p in paths if p not in served]
    check(not unserved,
          f'all {len(paths)} table paths are in the served schema',
          *unserved[:8])

    check('/health' in served and '/' in served,
          "/health and / are served — Cloud Run's readiness probe hits /health")

    # --- 3. the scripts exist, and parse --------------------------------
    missing = [s for s in scripts if not os.path.exists(os.path.join(_ROOT, s))]
    check(not missing,
          'every registered script exists on disk — the registration loop '
          'never looks, so a missing one is a 400 at click time',
          *missing)

    broken = []
    for s in scripts:
        full = os.path.join(_ROOT, s)
        if not os.path.exists(full):
            continue
        try:
            compile(open(full, encoding='utf-8').read(), full, 'exec')
        except SyntaxError as exc:
            broken.append(f'{s}:{exc.lineno} {exc.msg}')
    check(not broken,
          f'all {len(scripts) - len(missing)} scripts compile — each runs as '
          'its own subprocess, so a syntax error surfaces only when called',
          *broken)

    # --- 4. every top-level import is pinned -----------------------------
    pinned = pinned_imports()
    stdlib = set(sys.stdlib_module_names)
    first_party = {f[:-3] for f in os.listdir(_ROOT) if f.endswith('.py')}
    unpinned = {}
    for f in sorted(os.listdir(_ROOT)):
        if not f.endswith('.py'):
            continue
        for mod in top_level_imports(os.path.join(_ROOT, f)):
            if (mod in stdlib or mod in first_party or mod in pinned
                    or mod in _SHIPS_WITH):
                continue
            unpinned.setdefault(mod, []).append(f)
    check(not unpinned,
          'every module imported at import time is pinned in requirements.txt '
          '(or listed as shipping with one)',
          *[f'{m} — imported by {", ".join(fs[:3])}'
            for m, fs in sorted(unpinned.items())])

    # --- 5. the mounted routers contributed something --------------------
    for module, path in sorted(_ROUTER_MODULES.items()):
        check(path in served, f'{module} → {path}',
              'include_router() on a router whose decorators moved adds '
              'nothing, silently — this path is not served')
    check(_MODELS_PATHS <= set(served),
          f'models_api contributed all {len(_MODELS_PATHS)} registry '
          f'endpoints (train / predict / delete)',
          *sorted(_MODELS_PATHS - set(served)))

    expected = len(paths) + len(_ROUTER_MODULES) + len(_MODELS_PATHS) + 2
    check(len(served) == expected,
          f'{len(served)} paths served = {len(paths)} scripts + '
          f'{len(_ROUTER_MODULES)} conjoint routers + {len(_MODELS_PATHS)} registry '
          f'+ / + /health',
          f'expected {expected}')

    # --- 6. nothing on disk is unwired, nothing listed has vanished ------
    known = set(scripts) | set(_ROUTER_MODULES) | _HELPER_MODULES
    stray = sorted(f for f in os.listdir(_ROOT)
                   if f.endswith('.py') and f not in known)
    check(not stray,
          'no .py in the backend root is unaccounted for — an analysis script '
          'that was added but never wired is reachable by nobody',
          *stray)

    # --- 7. the error path all 118 routes share --------------------------
    # A failing script prints {"error": ...} to stderr and exits 1. The user
    # must get that sentence; a traceback tells them nothing they can act on.
    with tempfile.TemporaryDirectory() as tmp:
        saying = os.path.join(tmp, 'says.py')
        open(saying, 'w').write(
            'import json, sys\n'
            'sys.stderr.write(json.dumps({"error": "타깃 열을 고르세요"}))\n'
            'sys.exit(1)\n')
        junk = os.path.join(tmp, 'junk.py')
        open(junk, 'w').write('print("not json at all")\n')

        real_dir = app_module._BACKEND_DIR
        app_module._BACKEND_DIR = tmp
        try:
            try:
                app_module.run_script('says.py', {})
                detail = '<no exception raised>'
                status = 0
            except Exception as exc:
                detail = getattr(exc, 'detail', str(exc))
                status = getattr(exc, 'status_code', 0)
            check(status == 400 and detail == '타깃 열을 고르세요',
                  "a failing script's own message reaches the user as a 400, "
                  'not a traceback',
                  f'status={status} detail={detail!r}')

            try:
                app_module.run_script('junk.py', {})
                detail, status = '<no exception raised>', 0
            except Exception as exc:
                detail = getattr(exc, 'detail', str(exc))
                status = getattr(exc, 'status_code', 0)
            check(status == 500 and 'junk.py' in str(detail),
                  'a script printing non-JSON is a 500 that names the script',
                  f'status={status} detail={str(detail)[:80]!r}')
        finally:
            app_module._BACKEND_DIR = real_dir

    # --- 8. the rate limiter, which every endpoint sits behind -----------
    app = app_module.app
    cap = app_module._RL_MAX_REQUESTS
    ip = '203.0.113.7'                     # TEST-NET-3, used by nobody
    app_module._rl_store.pop(ip, None)
    first, last, over = None, None, None
    for i in range(cap + 1):
        status, _ = call(app, 'POST', '/api/analysis/__no_such_route__', ip=ip)
        if i == 0:
            first = status
        elif i == cap - 1:
            last = status
        elif i == cap:
            over = status
    check(first == 404 and last == 404,
          f'the first and the {cap}th request in a window are both let '
          f'through (404 — no such route, which is the point)',
          f'first={first} {cap}th={last}')
    check(over == 429,
          f'one more is a 429 — the cap is {cap} per '
          f'{app_module._RL_WINDOW_SECS}s and it is enforced',
          f'got {over}')
    status, body = call(app, 'GET', '/health', ip=ip)
    check(status == 200 and json.loads(body) == {'status': 'ok'},
          '/health still answers from an IP that is over the cap — a limiter '
          'that rate-limits the readiness probe takes the service down',
          f'status={status} body={body[:80]!r}')
    app_module._rl_store.pop(ip, None)

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


# ---------------------------------------------------------------------------
# --prove: each check, against the defect it exists for.
#
# (label, what to break in a copy of the backend, which line must go red)
# ---------------------------------------------------------------------------

def _sub(path, pattern, repl):
    def mutate(root):
        full = os.path.join(root, path)
        text = open(full, encoding='utf-8').read()
        new, n = re.subn(pattern, repl, text, count=1)
        assert n == 1, f'the mutation for {path} no longer applies: {pattern}'
        open(full, 'w', encoding='utf-8').write(new)
    return mutate


def _drop(path):
    return lambda root: os.remove(os.path.join(root, path))


def _copy(src, dst):
    return lambda root: shutil.copyfile(os.path.join(root, src),
                                        os.path.join(root, dst))


def _prepend(path, line):
    def mutate(root):
        full = os.path.join(root, path)
        text = open(full, encoding='utf-8').read()
        open(full, 'w', encoding='utf-8').write(line + '\n' + text)
    return mutate


def _append(path, text):
    def mutate(root):
        with open(os.path.join(root, path), 'a', encoding='utf-8') as fh:
            fh.write(text)
    return mutate


_FAULTS = [
    ('a registered script is deleted',
     _drop('knn_analysis.py'), 'exists on disk'),
    ('a path is registered twice',
     _sub('main.py', r'\("/api/analysis/knn",', '("/api/analysis/svm",'),
     'registered twice'),
    ('two paths share one script',
     _sub('main.py', r'"knn_analysis\.py", +True', '"svm_analysis.py", True'),
     'two paths'),
    ('a script imports an unpinned package',
     _prepend('knn_analysis.py', 'import definitely_not_pinned'),
     'pinned in requirements'),
    ('a script stops compiling',
     _append('knn_analysis.py', '\ndef (\n'), 'compile'),
    ('a router is no longer mounted',
     _sub('main.py', r'app\.include_router\(rc_router[^\n]*\n', ''),
     'rc_analysis'),
    ('a script is added but never wired',
     _copy('knn_analysis.py', 'brand_new_analysis.py'), 'unaccounted for'),
    ("a failing script's message is swallowed",
     _sub('main.py', r'detail = parsed\["error"\]', 'pass'),
     'reaches the user'),
    ('non-JSON output is not reported',
     _sub('main.py', r'status_code=500', 'status_code=200'),
     'names the script'),
    ('the limiter stops counting',
     _sub('main.py', r'\n    hits\.append\(now\)', '\n    pass'), '429'),
    ('/health loses its exemption',
     _sub('main.py', r'in \("/health", "/"\)', 'in ("/__none__",)'),
     'readiness probe'),
]


def prove():
    caught = missed = 0
    for label, mutate, expect in _FAULTS:
        tmp = tempfile.mkdtemp()
        root = os.path.join(tmp, 'backend')
        shutil.copytree(_ROOT, root,
                        ignore=shutil.ignore_patterns('__pycache__', '.git'))
        try:
            mutate(root)
            proc = subprocess.run(
                [sys.executable, os.path.join('scripts', 'check_startup.py')],
                cwd=root, capture_output=True, text=True)
            red = [ln for ln in proc.stdout.splitlines()
                   if ln.startswith('FAIL') and expect in ln]
            if red:
                caught += 1
                print(f'caught   {label}')
            else:
                missed += 1
                print(f'MISSED   {label}')
                for ln in proc.stdout.splitlines():
                    if ln.startswith('FAIL') or ln.endswith('failure(s)'):
                        print(f'           {ln}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{caught} of {len(_FAULTS)} faults caught, {missed} missed')
    return 1 if missed else 0


if __name__ == '__main__':
    sys.exit(prove() if '--prove' in sys.argv else main())
