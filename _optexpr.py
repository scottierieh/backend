"""Safe math-expression evaluator for the optimization backends.

Turns a user string like "(x1-3)**2 + exp(x2)" plus variable names into a
callable f(x_vector).

SECURITY: this used to be `eval()` with empty builtins, which is NOT a
sandbox — a generator expression reaches its frame, the frame reaches the
caller's builtins, and from there `__import__('os').system(...)`. A "__"
substring check does not help either ('_'+'_import'+'_'+'_'). So nothing
here calls eval/exec/compile any more. The string is parsed with
`ast.parse(mode='eval')`, every node is checked against a short whitelist,
and the validated tree is turned into plain Python closures that only ever
do float arithmetic and call the numpy functions in `_FUNCS`.

Accepted grammar (exactly what the optimisation pages document — objective
and constraint left-hand sides; the comparison operator and right-hand side
travel separately as con_ops / con_rhs, so comparisons are not needed):

    expr   := number | name | expr OP expr | (+|-) expr | func(expr, ...)
    OP     := + - * / ** % //       ('^' is accepted as a synonym for **)
    name   := one of the caller's var_names, or pi / e
    func   := exp log log10 sqrt sin cos tan arctan sinh cosh tanh abs
              power maximum minimum sign   (positional arguments only)

Everything else — attributes, subscripts, comprehensions, generators,
lambdas, walrus, starred args, keyword args, strings, comparisons, boolean
ops, conditionals, f-strings — is rejected before anything runs.

All arithmetic is done on numpy float64 (constants are converted up front),
so there is no arbitrary-precision integer path: `9**9**9` is inf, not a
CPU bomb. Literal exponents are additionally capped, and the expression
length and node count are bounded.
"""
import ast
import operator

import numpy as np

MAX_EXPR_LEN = 2000
MAX_NODES = 500
MAX_LITERAL_EXPONENT = 1000.0  # |constant exponent| cap

# name -> (callable, positional arity)
_FUNCS = {
    "exp": (np.exp, 1), "log": (np.log, 1), "log10": (np.log10, 1),
    "sqrt": (np.sqrt, 1), "sin": (np.sin, 1), "cos": (np.cos, 1),
    "tan": (np.tan, 1), "arctan": (np.arctan, 1), "sinh": (np.sinh, 1),
    "cosh": (np.cosh, 1), "tanh": (np.tanh, 1), "abs": (np.abs, 1),
    "sign": (np.sign, 1), "power": (np.power, 2), "maximum": (np.maximum, 2),
    "minimum": (np.minimum, 2),
}
_CONSTS = {"pi": np.float64(np.pi), "e": np.float64(np.e)}

# Kept for backwards compatibility with anything that imported the old table.
_ALLOWED = {**{k: v[0] for k, v in _FUNCS.items()}, **_CONSTS}


def _pow(a, b):
    return np.power(np.float64(a), np.float64(b))


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: _pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class ExpressionError(ValueError):
    pass


def _is_constant(node, var_index):
    """True when the subtree references no variable (only numbers, pi, e)."""
    return not any(isinstance(n, ast.Name) and n.id in var_index for n in ast.walk(node))


def _build(node, var_index):
    """Validate `node` and return a closure x -> float64."""
    if isinstance(node, ast.Expression):
        return _build(node.body, var_index)

    if isinstance(node, ast.Constant):
        # bool is a subclass of int; complex/str/bytes/None are not numbers here.
        if type(node.value) not in (int, float):
            raise ExpressionError("Only numeric constants are allowed.")
        try:
            val = np.float64(node.value)
        except OverflowError:
            raise ExpressionError("Numeric constant too large.") from None
        return lambda x: val

    if isinstance(node, ast.Name):
        nm = node.id
        if nm in var_index:
            i = var_index[nm]
            return lambda x: np.float64(x[i])
        if nm in _CONSTS:
            val = _CONSTS[nm]
            return lambda x: val
        raise ExpressionError(f"Unknown name '{nm}'.")

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError("Operator not allowed.")
        left = _build(node.left, var_index)
        right = _build(node.right, var_index)
        if isinstance(node.op, ast.Pow) and _is_constant(node.right, var_index):
            # A literal (or all-constant) exponent is checked up front, so
            # 9**9**9 is refused at parse time instead of evaluating to inf.
            with np.errstate(all="ignore"):
                ev = float(right(None))
            if not np.isfinite(ev) or abs(ev) > MAX_LITERAL_EXPONENT:
                raise ExpressionError("Exponent too large.")
        return lambda x: op(left(x), right(x))

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ExpressionError("Operator not allowed.")
        operand = _build(node.operand, var_index)
        return lambda x: op(operand(x))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ExpressionError("Only the documented math functions may be called.")
        if node.keywords:
            raise ExpressionError("Keyword arguments are not allowed.")
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise ExpressionError("Starred arguments are not allowed.")
        fn, arity = _FUNCS[node.func.id]
        if len(node.args) != arity:
            raise ExpressionError(f"{node.func.id}() takes {arity} argument(s).")
        args = [_build(a, var_index) for a in node.args]
        if arity == 1:
            a0 = args[0]
            return lambda x: fn(a0(x))
        a0, a1 = args
        return lambda x: fn(a0(x), a1(x))

    raise ExpressionError(f"Unsupported syntax: {type(node).__name__}.")


def compile_expr(expr, var_names):
    """Parse + validate `expr`; return a closure x -> float64.

    Raises ValueError (ExpressionError) for anything outside the grammar."""
    expr = str(expr)
    if len(expr) > MAX_EXPR_LEN:
        raise ExpressionError(f"Expression too long (max {MAX_EXPR_LEN} characters).")
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as ex:
        raise ExpressionError(f"Invalid expression '{expr}': {ex}") from None
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > MAX_NODES:
        raise ExpressionError(f"Expression too complex (max {MAX_NODES} nodes).")
    var_index = {}
    for i, nm in enumerate(var_names):
        var_index.setdefault(str(nm), i)
    try:
        return _build(tree, var_index)
    except RecursionError:
        raise ExpressionError("Expression too deeply nested.") from None


def make_func(expr: str, var_names):
    expr = str(expr)
    fn = compile_expr(expr, list(var_names))

    def f(x):
        try:
            return float(fn(x))
        except Exception as ex:
            raise ValueError(f"Cannot evaluate '{expr}': {ex}")
    return f


def num_grad(f, x, h=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


# ── Numeric curvature (convexity) sampling ───────────────────────────────
#
# The old check compared the RAW second difference f(x+hd) - 2f(x) + f(x-hd)
# with h = 1e-4 against -1e-6. That raw difference is about h**2 * curvature
# = 1e-8 * curvature, so any function whose curvature was above about -100 —
# e.g. the concave -(x1**2) - x2**2 (curvature -2) — "passed", and the convex
# page then called the minimum global. The second difference is now divided by
# h**2, so it estimates the directional second derivative d'H d itself, and
# it is compared with a tolerance relative to the function's size. It is
# sampled at several points (the solution plus seeded random points around it,
# kept inside a caller-supplied region such as the inequality-feasible set) and
# in several directions (every axis, the e_i ± e_j diagonals and seeded random
# directions). A pass means "no violation found at the sampled points" — a
# numerical sampling check, not a proof of convexity.

CURV_TOL = 1e-6        # relative: s must be >= -CURV_TOL * max(1, |f(p)|)
CURV_POINTS = 12       # the solution + 11 random points around it
CURV_RANDOM_DIRS = 4
CURV_MAX_DIAG_PAIRS = 15


def _curv_directions(n, rng):
    eye = np.eye(n)
    dirs = [eye[i] for i in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:CURV_MAX_DIAG_PAIRS]
    for i, j in pairs:
        for sgn in (1.0, -1.0):
            d = eye[i] + sgn * eye[j]
            dirs.append(d / np.linalg.norm(d))
    for _ in range(CURV_RANDOM_DIRS):
        d = rng.standard_normal(n)
        nd = np.linalg.norm(d)
        if nd > 0:
            dirs.append(d / nd)
    return dirs


def curvature_sample_points(x, region=None, n_points=CURV_POINTS, seed=0):
    """The solution x plus seeded random points in a box of half-width
    max(1, |x|_inf) around it, keeping only those `region(p)` accepts."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    scale = max(1.0, float(np.max(np.abs(x)))) if x.size else 1.0
    pts = [x.copy()]
    tries = 0
    while len(pts) < n_points and tries < 40 * n_points:
        tries += 1
        p = x + scale * rng.uniform(-1.0, 1.0, x.size)
        try:
            with np.errstate(all="ignore"):
                keep = region is None or bool(region(p))
        except Exception:
            keep = False
        if keep:
            pts.append(p)
    return pts


def curvature_check(f, points, require="convex", tol=CURV_TOL, seed=0):
    """Sample the directional second derivative s = (f(p+hd) - 2f(p) + f(p-hd)) / h**2.

    require: "convex"  -> violation when s < -tol*max(1,|f(p)|)
             "concave" -> violation when s >  tol*max(1,|f(p)|)
             "affine"  -> violation when |s| > tol*max(1,|f(p)|)
    Points or directions where f is undefined (nan/inf, domain errors) are
    skipped. Returns a dict; `ok` is None when nothing could be tested."""
    pts = [np.asarray(p, float) for p in points]
    if not pts:
        return {"ok": None, "require": require, "n_points": 0, "n_directions": 0,
                "n_tests": 0, "n_violations": 0, "worst": None}
    n = pts[0].size
    rng = np.random.default_rng(seed)
    dirs = _curv_directions(n, rng)
    n_tests = 0
    n_viol = 0
    used_pts = 0
    worst = None
    worst_excess = 0.0
    for p in pts:
        h = 1e-3 * max(1.0, float(np.max(np.abs(p))) if n else 1.0)
        tested_here = False
        for d in dirs:
            try:
                with np.errstate(all="ignore"):
                    f0 = float(f(p)); fp = float(f(p + h * d)); fm = float(f(p - h * d))
            except Exception:
                continue
            if not (np.isfinite(f0) and np.isfinite(fp) and np.isfinite(fm)):
                continue
            s = (fp - 2.0 * f0 + fm) / (h * h)
            thr = tol * max(1.0, abs(f0))
            if require == "convex":
                excess = -s - thr
            elif require == "concave":
                excess = s - thr
            else:
                excess = abs(s) - thr
            n_tests += 1
            tested_here = True
            if excess > 0:
                n_viol += 1
                if worst is None or excess > worst_excess:
                    worst_excess = excess
                    worst = {"point": [float(round(v, 6)) for v in p],
                             "direction": [float(round(v, 6)) for v in d],
                             "second_derivative": float(round(s, 6))}
        if tested_here:
            used_pts += 1
    return {"ok": (n_viol == 0) if n_tests else None, "require": require,
            "n_points": used_pts, "n_directions": len(dirs),
            "n_tests": n_tests, "n_violations": n_viol, "worst": worst}


def is_convex_numeric(f, x, n_dirs=8, h=None, region=None):
    """Backwards-compatible wrapper: (passed, (n_ok, n_tests)).

    `passed` is True only when the sampling check found no violation —
    "no violation found", not a proof. n_dirs/h are accepted for old callers
    and ignored (directions and step are chosen by curvature_check)."""
    rep = curvature_check(f, curvature_sample_points(x, region=region), require="convex")
    return bool(rep["ok"]), (rep["n_tests"] - rep["n_violations"], rep["n_tests"])
