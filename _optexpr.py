"""Safe math-expression evaluator for the optimization backends.

Turns a user string like "(x1-3)**2 + exp(x2)" plus variable names into a
callable f(x_vector). Evaluation runs with NO builtins and only a whitelist
of numpy math functions, so it cannot import modules or touch the system —
it only does arithmetic on the supplied variables.
"""
import numpy as np

_ALLOWED = {
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "abs": np.abs,
    "power": np.power, "pi": np.pi, "e": np.e, "maximum": np.maximum,
    "minimum": np.minimum, "sign": np.sign,
}


def make_func(expr: str, var_names):
    expr = str(expr)
    # normalise a couple of common human forms
    expr = expr.replace("^", "**")
    code = compile(expr, "<expr>", "eval")
    # basic guard: reject dunder access
    if "__" in expr:
        raise ValueError("Invalid expression.")

    def f(x):
        env = {"__builtins__": {}}
        env.update(_ALLOWED)
        for i, nm in enumerate(var_names):
            env[nm] = x[i]
        try:
            return float(eval(code, env))
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
