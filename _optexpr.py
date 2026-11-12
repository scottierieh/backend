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


def is_convex_numeric(f, x, n_dirs=8, h=1e-4):
    """Rough convexity check: second directional differences >= 0 at x."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    ok = 0; tot = 0
    for _ in range(n_dirs):
        d = rng.standard_normal(len(x)); d /= (np.linalg.norm(d) or 1)
        try:
            s = f(x + h*d) - 2*f(x) + f(x - h*d)
        except Exception:
            continue
        tot += 1
        if s >= -1e-6:
            ok += 1
    return (tot > 0 and ok == tot), (ok, tot)
