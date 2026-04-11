"""
Efficiency Analysis Router — FastAPI (DEA)
Modules: data_prep | efficiency | benchmarking | rts | insights
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

router = APIRouter()

# ─── Input / Output Models ────────────────────────────────────────────────────

class DMUInput(BaseModel):
    name: str
    inputs: Dict[str, float]   # e.g. {"labor": 10, "capital": 20}
    outputs: Dict[str, float]  # e.g. {"revenue": 100, "units": 500}

class DEARequest(BaseModel):
    dmus: List[DMUInput]
    analysis_type: str   # efficiency | benchmarking | rts | sensitivity | window
    model: str = "CCR"   # CCR | BCC | SBM
    orientation: str = "input"  # input | output
    params: Optional[Dict[str, Any]] = {}

class DMUResult(BaseModel):
    name: str
    score: float
    rank: int
    classification: str          # Efficient | Near-Efficient | Inefficient
    peers: Optional[List[str]] = []
    slacks: Optional[Dict[str, float]] = {}
    targets: Optional[Dict[str, float]] = {}
    metadata: Optional[Dict[str, Any]] = {}

class DEAResult(BaseModel):
    analysis_type: str
    model: str
    orientation: str
    summary: Dict[str, Any]
    dmus: List[DMUResult]
    insights: List[str]

# ─── Core DEA Solver ─────────────────────────────────────────────────────────

def _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """Wrapper with numerical stability fallback."""
    from scipy.optimize import linprog
    # Primary: HiGHS
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.success:
        return res
    # Fallback: revised simplex
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="revised simplex")
    return res


def solve_dea(
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    model: str = "CCR",
    orientation: str = "input",
) -> np.ndarray:
    """
    CCR  — Constant Returns to Scale  (Charnes, Cooper, Rhodes 1978)
    BCC  — Variable Returns to Scale  (Banker, Charnes, Cooper 1984)
    SBM  — Slack-Based Measure, non-radial, units-invariant (Tone 2001)
    NIRS — Non-Increasing RTS  (Σλ ≤ 1)
    NDRS — Non-Decreasing RTS  (Σλ ≥ 1)
    """
    n, m_in  = input_matrix.shape
    m_out    = output_matrix.shape[1]
    scores   = np.ones(n)

    if model == "SBM":
        # ── True SBM (input-oriented, Tone 2001) ──────────────────────────────
        # Minimize  ρ = (1 - (1/m) Σ s_i^- / x_i0) / (1 + (1/s) Σ s_r^+ / y_r0)
        # Charnes-Cooper transformation: t = 1 / denominator
        # Variables: [t, τ_1..n, σ_1..m_in (=t*s^-), ξ_1..m_out (=t*s^+)]
        # ── LP form (Tone 2001, eq. 11) ───────────────────────────────────────
        for j in range(n):
            x0 = input_matrix[j]
            y0 = output_matrix[j]

            # Skip degenerate DMUs
            if np.any(x0 <= 0) or np.any(y0 <= 0):
                scores[j] = 1.0
                continue

            # num_vars = 1 (t) + n (tau) + m_in (sigma, input slacks) + m_out (xi, output slacks)
            nv = 1 + n + m_in + m_out
            idx_t     = 0
            idx_tau   = slice(1, 1 + n)
            idx_sigma = slice(1 + n, 1 + n + m_in)
            idx_xi    = slice(1 + n + m_in, nv)

            # Objective: minimise t - (1/m_in) * Σ σ_i / x_i0
            c_obj = np.zeros(nv)
            c_obj[idx_t] = 1.0
            for i in range(m_in):
                c_obj[1 + n + i] = -1.0 / (m_in * x0[i])

            # Equality constraints
            n_eq = 1 + m_in + m_out
            A_eq = np.zeros((n_eq, nv))
            b_eq = np.zeros(n_eq)

            # (1) t + (1/m_out) Σ ξ_r / y_r0 = 1
            A_eq[0, idx_t] = 1.0
            for r in range(m_out):
                A_eq[0, 1 + n + m_in + r] = 1.0 / (m_out * y0[r])
            b_eq[0] = 1.0

            # (2) X τ + σ = t x0  →  X τ + σ - t x0 = 0
            for i in range(m_in):
                A_eq[1 + i, idx_t]         = -x0[i]
                A_eq[1 + i, idx_tau]       =  input_matrix[:, i]
                A_eq[1 + i, 1 + n + i]    =  1.0
            # b_eq[1..m_in] = 0

            # (3) Y τ - ξ = t y0  →  Y τ - ξ - t y0 = 0
            for r in range(m_out):
                A_eq[1 + m_in + r, idx_t]              = -y0[r]
                A_eq[1 + m_in + r, idx_tau]            =  output_matrix[:, r]
                A_eq[1 + m_in + r, 1 + n + m_in + r]  = -1.0
            # b_eq[1+m_in..] = 0

            bounds = [(0, None)] * nv
            bounds[idx_t] = (1e-6, None)

            res = _solve_lp(c_obj, None, None, A_eq, b_eq, bounds)
            if res.success and res.fun is not None:
                scores[j] = max(0.0, min(1.0, float(res.fun)))
            else:
                # Fallback to CCR score for this DMU
                scores[j] = float(_solve_ccr_single(j, input_matrix, output_matrix, "input"))

        return scores

    # ── CCR / BCC / NIRS / NDRS (radial) ─────────────────────────────────────
    for j in range(n):
        x0 = input_matrix[j]
        y0 = output_matrix[j]

        if orientation == "input":
            num_vars = 1 + n
            c = np.zeros(num_vars); c[0] = 1.0

            A_ub = np.zeros((m_out + m_in, num_vars))
            b_ub = np.zeros(m_out + m_in)
            for i in range(m_out):
                A_ub[i, 1:] = -output_matrix[:, i]; b_ub[i] = -y0[i]
            for i in range(m_in):
                A_ub[m_out + i, 0]  = -x0[i]
                A_ub[m_out + i, 1:] =  input_matrix[:, i]

            bounds = [(0, None)] * num_vars; bounds[0] = (1e-6, 1.0)

            A_eq = b_eq = None
            if model == "BCC":
                A_eq = np.zeros((1, num_vars)); A_eq[0, 1:] = 1.0; b_eq = np.array([1.0])
            elif model == "NIRS":
                A_ub_nirs = np.zeros((1, num_vars)); A_ub_nirs[0, 1:] = 1.0
                A_ub = np.vstack([A_ub, A_ub_nirs]); b_ub = np.append(b_ub, 1.0)
            elif model == "NDRS":
                A_eq = np.zeros((1, num_vars)); A_eq[0, 1:] = 1.0
                b_eq = np.array([1.0])
                # Σλ ≥ 1  →  -Σλ ≤ -1
                A_ub_ndrs = np.zeros((1, num_vars)); A_ub_ndrs[0, 1:] = -1.0
                A_ub = np.vstack([A_ub, A_ub_ndrs]); b_ub = np.append(b_ub, -1.0)
                A_eq = b_eq = None  # handled via inequality

            res = _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
            scores[j] = float(res.fun) if res.success and res.fun is not None else 1.0

        else:  # output
            num_vars = 1 + n
            c = np.zeros(num_vars); c[0] = -1.0

            A_ub = np.zeros((m_in + m_out, num_vars))
            b_ub = np.zeros(m_in + m_out)
            for i in range(m_in):
                A_ub[i, 1:] = input_matrix[:, i]; b_ub[i] = x0[i]
            for i in range(m_out):
                A_ub[m_in + i, 0]  =  y0[i]
                A_ub[m_in + i, 1:] = -output_matrix[:, i]

            bounds = [(1.0, None)] + [(0, None)] * n

            A_eq = b_eq = None
            if model == "BCC":
                A_eq = np.zeros((1, num_vars)); A_eq[0, 1:] = 1.0; b_eq = np.array([1.0])
            elif model == "NIRS":
                A_ub_nirs = np.zeros((1, num_vars)); A_ub_nirs[0, 1:] = 1.0
                A_ub = np.vstack([A_ub, A_ub_nirs]); b_ub = np.append(b_ub, 1.0)
            elif model == "NDRS":
                A_ub_ndrs = np.zeros((1, num_vars)); A_ub_ndrs[0, 1:] = -1.0
                A_ub = np.vstack([A_ub, A_ub_ndrs]); b_ub = np.append(b_ub, -1.0)

            res = _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
            if res.success and res.fun is not None:
                phi = float(-res.fun)
                scores[j] = 1.0 / phi if phi > 1e-9 else 1.0
            else:
                scores[j] = 1.0

    return np.clip(scores, 0.0, 1.0)


def _solve_ccr_single(j: int, X: np.ndarray, Y: np.ndarray, orientation: str) -> float:
    """CCR score for one DMU — used as SBM fallback."""
    n, m_in = X.shape; m_out = Y.shape[1]
    x0, y0  = X[j], Y[j]
    if orientation == "input":
        num_vars = 1 + n
        c = np.zeros(num_vars); c[0] = 1.0
        A_ub = np.zeros((m_out + m_in, num_vars))
        b_ub = np.zeros(m_out + m_in)
        for i in range(m_out): A_ub[i, 1:] = -Y[:, i]; b_ub[i] = -y0[i]
        for i in range(m_in):  A_ub[m_out+i, 0] = -x0[i]; A_ub[m_out+i, 1:] = X[:, i]
        bounds = [(0, None)] * num_vars; bounds[0] = (1e-6, 1.0)
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return float(res.fun) if res.success else 1.0
    return 1.0


def compute_peers(
    j: int,
    scores: np.ndarray,
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    names: List[str],
    model: str,
    orientation: str,
) -> List[str]:
    """Return efficient DMUs that form the reference set (peer group) for DMU j."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return []

    efficient_idx = [i for i, s in enumerate(scores) if s >= 0.999]
    if not efficient_idx:
        return []

    n, m_in  = input_matrix.shape
    m_out    = output_matrix.shape[1]
    x0 = input_matrix[j];  y0 = output_matrix[j]
    theta = scores[j]

    # Solve for λ of efficient DMUs only
    k = len(efficient_idx)
    X_eff = input_matrix[efficient_idx]
    Y_eff = output_matrix[efficient_idx]

    c   = np.zeros(k)
    A_ub = np.zeros((m_out + m_in, k))
    b_ub = np.zeros(m_out + m_in)

    for i in range(m_out):
        A_ub[i, :] = -Y_eff[:, i]; b_ub[i] = -y0[i]
    for i in range(m_in):
        A_ub[m_out + i, :] = X_eff[:, i]; b_ub[m_out + i] = theta * x0[i]

    bounds = [(0, None)] * k
    A_eq = b_eq = None
    if model == "BCC":
        A_eq = np.ones((1, k)); b_eq = np.array([1.0])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return []

    return [names[efficient_idx[i]] for i, lam in enumerate(res.x) if lam > 0.01]


def compute_slacks_and_targets(
    j: int,
    score: float,
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    input_names: List[str],
    output_names: List[str],
    orientation: str,
) -> tuple:
    """Returns (slacks dict, targets dict) for DMU j."""
    x0 = input_matrix[j]; y0 = output_matrix[j]
    slacks  = {}
    targets = {}

    if orientation == "input":
        for k, name in enumerate(input_names):
            projected = score * x0[k]
            slack = x0[k] - projected
            slacks[f"in_{name}"]  = round(float(slack), 4)
            targets[f"in_{name}"] = round(float(projected), 4)
        for k, name in enumerate(output_names):
            slacks[f"out_{name}"]  = 0.0
            targets[f"out_{name}"] = round(float(y0[k]), 4)
    else:
        phi = 1.0 / score if score > 0 else 1.0
        for k, name in enumerate(input_names):
            slacks[f"in_{name}"]  = 0.0
            targets[f"in_{name}"] = round(float(x0[k]), 4)
        for k, name in enumerate(output_names):
            projected = phi * y0[k]
            slack = projected - y0[k]
            slacks[f"out_{name}"]  = round(float(slack), 4)
            targets[f"out_{name}"] = round(float(projected), 4)

    return slacks, targets


# ─── Module 1: Efficiency Scores ─────────────────────────────────────────────

def analyze_efficiency(dmus: List[DMUInput], model: str, orientation: str) -> DEAResult:
    names       = [d.name for d in dmus]
    input_keys  = list(dmus[0].inputs.keys())
    output_keys = list(dmus[0].outputs.keys())

    input_matrix  = np.array([[d.inputs[k]  for k in input_keys]  for d in dmus], dtype=float)
    output_matrix = np.array([[d.outputs[k] for k in output_keys] for d in dmus], dtype=float)

    scores = solve_dea(input_matrix, output_matrix, model, orientation)

    ranked = np.argsort(-scores)
    rank_map = {idx: rank + 1 for rank, idx in enumerate(ranked)}

    def classify(s: float) -> str:
        if s >= 0.999: return "Efficient"
        if s >= 0.85:  return "Near-Efficient"
        return "Inefficient"

    dmu_results = []
    for j, d in enumerate(dmus):
        s = float(scores[j])
        peers = compute_peers(j, scores, input_matrix, output_matrix, names, model, orientation)
        slacks, targets = compute_slacks_and_targets(
            j, s, input_matrix, output_matrix, input_keys, output_keys, orientation,
        )
        dmu_results.append(DMUResult(
            name=d.name,
            score=round(s, 4),
            rank=rank_map[j],
            classification=classify(s),
            peers=peers,
            slacks=slacks,
            targets=targets,
            metadata={
                "inputs":  d.inputs,
                "outputs": d.outputs,
            },
        ))

    dmu_results.sort(key=lambda x: x.rank)

    n_efficient     = int(np.sum(scores >= 0.999))
    n_near          = int(np.sum((scores >= 0.85) & (scores < 0.999)))
    n_inefficient   = int(np.sum(scores < 0.85))
    avg_score       = float(np.mean(scores))
    min_score       = float(np.min(scores))
    worst_dmu       = names[int(np.argmin(scores))]
    best_dmu        = names[int(np.argmax(scores))]

    summary = {
        "n_dmus":           len(dmus),
        "model":            model,
        "orientation":      orientation,
        "n_inputs":         len(input_keys),
        "n_outputs":        len(output_keys),
        "input_vars":       input_keys,
        "output_vars":      output_keys,
        "n_efficient":      n_efficient,
        "n_near_efficient": n_near,
        "n_inefficient":    n_inefficient,
        "avg_score":        round(avg_score, 4),
        "min_score":        round(min_score, 4),
        "worst_dmu":        worst_dmu,
        "best_dmu":         best_dmu,
        "score_distribution": {
            "efficient":     n_efficient,
            "near_efficient": n_near,
            "inefficient":   n_inefficient,
        },
    }

    insights: List[str] = []

    eff_pct = n_efficient / len(dmus) * 100
    insights.append(
        f"{n_efficient} of {len(dmus)} DMUs ({eff_pct:.0f}%) are fully efficient under the {model} model. "
        f"Average efficiency score: {avg_score:.3f}."
    )

    if n_inefficient > 0:
        insights.append(
            f"Worst performer: {worst_dmu} (score {min_score:.3f}). "
            f"This DMU could improve outputs by {(1/min_score - 1)*100:.1f}% under output orientation, "
            f"or reduce inputs by {(1 - min_score)*100:.1f}% under input orientation."
        )

    if avg_score < 0.75:
        insights.append(
            f"Low average efficiency ({avg_score:.3f}) suggests significant room for improvement. "
            f"Consider structural changes rather than incremental optimisation."
        )
    elif avg_score > 0.9:
        insights.append(
            f"High average efficiency ({avg_score:.3f}) — the peer group is operating near the frontier. "
            f"Marginal gains may require strategic investment rather than operational changes."
        )

    if model == "CCR":
        insights.append(
            "CCR model assumes constant returns to scale. "
            "If DMUs operate at different scales, consider comparing with BCC results to isolate pure technical efficiency."
        )
    elif model == "BCC":
        insights.append(
            "BCC model allows variable returns to scale. "
            "Scale efficiency = CCR score / BCC score — compare both to diagnose scale vs. technical inefficiency."
        )

    if n_efficient == len(dmus):
        insights.append(
            "All DMUs are on the efficient frontier. "
            "This may indicate an under-specified model — consider adding more discriminating input/output variables."
        )

    return DEAResult(
        analysis_type="efficiency",
        model=model,
        orientation=orientation,
        summary=summary,
        dmus=dmu_results,
        insights=insights,
    )


# ─── Super-Efficiency Solver ──────────────────────────────────────────────────

def solve_super_efficiency(
    j: int,
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    model: str,
    orientation: str,
) -> float:
    """
    Andersen & Petersen (1993) super-efficiency.
    Exclude DMU j from its own reference set — allows score > 1 for efficient DMUs.
    """
    n, m_in = input_matrix.shape
    m_out   = output_matrix.shape[1]

    # Build reference set excluding DMU j
    idx = [i for i in range(n) if i != j]
    X_ref = input_matrix[idx]
    Y_ref = output_matrix[idx]
    x0, y0 = input_matrix[j], output_matrix[j]
    k = len(idx)

    if orientation == "input":
        num_vars = 1 + k
        c = np.zeros(num_vars); c[0] = 1.0
        A_ub = np.zeros((m_out + m_in, num_vars))
        b_ub = np.zeros(m_out + m_in)
        for i in range(m_out):
            A_ub[i, 1:] = -Y_ref[:, i]; b_ub[i] = -y0[i]
        for i in range(m_in):
            A_ub[m_out + i, 0]  = -x0[i]
            A_ub[m_out + i, 1:] =  X_ref[:, i]
        bounds = [(0, None)] * num_vars; bounds[0] = (0, None)  # no upper bound → can exceed 1

        A_eq = b_eq = None
        if model == "BCC":
            A_eq = np.zeros((1, num_vars)); A_eq[0, 1:] = 1.0; b_eq = np.array([1.0])

        res = _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
        return float(res.fun) if res.success and res.fun is not None else 1.0

    else:  # output
        num_vars = 1 + k
        c = np.zeros(num_vars); c[0] = -1.0
        A_ub = np.zeros((m_in + m_out, num_vars))
        b_ub = np.zeros(m_in + m_out)
        for i in range(m_in):
            A_ub[i, 1:] = X_ref[:, i]; b_ub[i] = x0[i]
        for i in range(m_out):
            A_ub[m_in + i, 0]  =  y0[i]
            A_ub[m_in + i, 1:] = -Y_ref[:, i]
        bounds = [(0, None)] * num_vars; bounds[0] = (0, None)

        A_eq = b_eq = None
        if model == "BCC":
            A_eq = np.zeros((1, num_vars)); A_eq[0, 1:] = 1.0; b_eq = np.array([1.0])

        res = _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
        if res.success and res.fun is not None:
            phi = float(-res.fun)
            return 1.0 / phi if phi > 1e-9 else 1.0
        return 1.0


# ─── Cross-Efficiency Solver ──────────────────────────────────────────────────

def solve_cross_efficiency(
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
) -> np.ndarray:
    """
    Sexton et al. (1986) cross-efficiency matrix.
    Each DMU d is evaluated using the optimal weights of every other DMU d'.
    cross_score[d][d'] = (u_d' · y_d) / (v_d' · x_d)
    Returns the average cross-efficiency score per DMU.

    Uses the dual multiplier form (CCR input-oriented).
    Variables: [u_1..m_out, v_1..m_in]  (output weights, input weights)
    """
    from scipy.optimize import linprog

    n, m_in = input_matrix.shape
    m_out   = output_matrix.shape[1]
    cross_matrix = np.ones((n, n))  # cross_matrix[d, d'] = score of d using weights of d'

    for d_prime in range(n):
        x0 = input_matrix[d_prime]
        y0 = output_matrix[d_prime]

        # CCR multiplier form:
        # Maximise u·y0  s.t.  v·x0 = 1,  u·Y_j - v·X_j ≤ 0 ∀j,  u≥ε, v≥ε
        # Variables: [u_1..m_out, v_1..m_in]
        nv = m_out + m_in
        eps = 1e-6

        # Minimise -u·y0 = -Σ u_r * y0_r
        c = np.zeros(nv)
        for r in range(m_out): c[r] = -y0[r]

        # Inequality: u·Y_j - v·X_j ≤ 0  for all j
        A_ub = np.zeros((n, nv))
        b_ub = np.zeros(n)
        for j in range(n):
            for r in range(m_out): A_ub[j, r]          =  output_matrix[j, r]
            for i in range(m_in):  A_ub[j, m_out + i]  = -input_matrix[j, i]

        # Equality: v·x0 = 1
        A_eq = np.zeros((1, nv))
        for i in range(m_in): A_eq[0, m_out + i] = x0[i]
        b_eq = np.array([1.0])

        bounds = [(eps, None)] * nv
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        if not res.success or res.x is None:
            continue

        u_opt = res.x[:m_out]
        v_opt = res.x[m_out:]

        for d in range(n):
            num = float(np.dot(u_opt, output_matrix[d]))
            den = float(np.dot(v_opt, input_matrix[d]))
            cross_matrix[d, d_prime] = num / den if den > 1e-9 else 1.0

    # Average cross-efficiency: mean over all d' ≠ d
    avg_cross = np.array([
        np.mean([cross_matrix[d, dp] for dp in range(n) if dp != d])
        for d in range(n)
    ])
    return np.clip(avg_cross, 0.0, None)


# ─── Module 2: Benchmarking ───────────────────────────────────────────────────

def analyze_benchmarking(dmus: List[DMUInput], model: str, orientation: str, params: dict) -> DEAResult:
    names       = [d.name for d in dmus]
    input_keys  = list(dmus[0].inputs.keys())
    output_keys = list(dmus[0].outputs.keys())

    input_matrix  = np.array([[d.inputs[k]  for k in input_keys]  for d in dmus], dtype=float)
    output_matrix = np.array([[d.outputs[k] for k in output_keys] for d in dmus], dtype=float)

    bm_type = params.get("type", "peer")  # peer | super | cross

    base_scores = solve_dea(input_matrix, output_matrix, model, orientation)

    # ── Peer frequency ──
    peer_map: Dict[int, List[str]] = {}
    for j in range(len(dmus)):
        peer_map[j] = compute_peers(j, base_scores, input_matrix, output_matrix, names, model, orientation)

    peer_count_map: Dict[int, int] = {j: 0 for j in range(len(dmus))}
    for j in range(len(dmus)):
        for i in range(len(dmus)):
            if i != j and names[j] in peer_map[i]:
                peer_count_map[j] += 1

    # ── Super-efficiency ──
    super_scores = np.array([
        solve_super_efficiency(j, input_matrix, output_matrix, model, orientation)
        for j in range(len(dmus))
    ])

    # ── Cross-efficiency (CCR input only — multiplier form) ──
    try:
        cross_scores = solve_cross_efficiency(input_matrix, output_matrix)
    except Exception:
        cross_scores = base_scores.copy()

    dmu_results = []
    for j, d in enumerate(dmus):
        base_s  = float(base_scores[j])
        super_s = float(super_scores[j])
        cross_s = float(cross_scores[j])

        # Reported score depends on requested type
        if bm_type == "super":
            display_score = super_s          # can exceed 1
        elif bm_type == "cross":
            display_score = cross_s
        else:
            display_score = base_s

        dmu_results.append(DMUResult(
            name=d.name,
            score=round(display_score, 4),
            rank=0,
            classification="Efficient" if base_s >= 0.999 else "Inefficient",
            peers=peer_map[j],
            metadata={
                "base_score":   round(base_s,  4),
                "super_score":  round(super_s, 4),
                "cross_score":  round(cross_s, 4),
                "peer_count":   peer_count_map[j],
                "is_benchmark": peer_count_map[j] > 0,
            },
        ))

    # Rank by requested score
    dmu_results.sort(key=lambda x: -x.score)
    for i, r in enumerate(dmu_results):
        r.rank = i + 1

    top_benchmarks = sorted(
        [r for r in dmu_results if r.metadata.get("peer_count", 0) > 0],
        key=lambda x: -x.metadata["peer_count"],
    )

    # Super-efficiency ranking (efficient DMUs only)
    super_ranked = sorted(
        [(names[j], float(super_scores[j])) for j in range(len(dmus)) if base_scores[j] >= 0.999],
        key=lambda x: -x[1],
    )

    summary = {
        "n_dmus":           len(dmus),
        "model":            model,
        "benchmark_type":   bm_type,
        "n_benchmarks":     len(top_benchmarks),
        "top_benchmarks":   [{"name": r.name, "peer_count": r.metadata["peer_count"]} for r in top_benchmarks[:5]],
        "super_ranking":    [{"name": n, "super_score": round(s, 4)} for n, s in super_ranked[:5]],
        "avg_cross":        round(float(np.mean(cross_scores)), 4),
    }

    insights: List[str] = []
    if top_benchmarks:
        top = top_benchmarks[0]
        insights.append(
            f"Most referenced benchmark: {top.name} "
            f"(peer for {top.metadata['peer_count']} DMUs). "
            f"Defines best practice most broadly."
        )

    if super_ranked:
        top_super = super_ranked[0]
        insights.append(
            f"Super-efficiency leader: {top_super[0]} (score {top_super[1]:.4f}). "
            f"This DMU dominates the frontier most — removing it would cause the largest regression."
        )

    non_benchmarks = [r.name for r in dmu_results if r.metadata.get("peer_count", 0) == 0 and r.metadata.get("base_score", 0) >= 0.999]
    if non_benchmarks:
        insights.append(
            f"Efficient but never referenced as peer: {', '.join(non_benchmarks[:3])}. "
            f"These lie on a niche part of the frontier — check for outlier inputs/outputs."
        )

    avg_cross = float(np.mean(cross_scores))
    insights.append(
        f"Average cross-efficiency score: {avg_cross:.3f}. "
        f"{'High peer-appraised efficiency — the frontier is robust.' if avg_cross > 0.8 else 'Moderate cross-efficiency — some efficient DMUs may be self-appraised outliers.'}"
    )

    return DEAResult(
        analysis_type="benchmarking",
        model=model,
        orientation=orientation,
        summary=summary,
        dmus=dmu_results,
        insights=insights,
    )


# ─── Module 3: Returns to Scale ───────────────────────────────────────────────

def analyze_rts(dmus: List[DMUInput], orientation: str) -> DEAResult:
    names       = [d.name for d in dmus]
    input_keys  = list(dmus[0].inputs.keys())
    output_keys = list(dmus[0].outputs.keys())

    input_matrix  = np.array([[d.inputs[k]  for k in input_keys]  for d in dmus], dtype=float)
    output_matrix = np.array([[d.outputs[k] for k in output_keys] for d in dmus], dtype=float)

    scores_ccr  = solve_dea(input_matrix, output_matrix, "CCR",  orientation)
    scores_bcc  = solve_dea(input_matrix, output_matrix, "BCC",  orientation)
    scores_nirs = solve_dea(input_matrix, output_matrix, "NIRS", orientation)
    scores_ndrs = solve_dea(input_matrix, output_matrix, "NDRS", orientation)

    # Scale efficiency = CCR / BCC
    scale_eff = np.where(scores_bcc > 0, scores_ccr / scores_bcc, 0.0)
    scale_eff = np.clip(scale_eff, 0.0, 1.0)

    def classify_rts(j: int) -> str:
        """
        Proper RTS classification via NIRS / NDRS models (Färe et al. 1994).
        - Solve BCC  (VRS):   θ_VRS
        - Solve NIRS (Σλ≤1):  θ_NIRS
        - Solve NDRS (Σλ≥1):  θ_NDRS

        Decision rules (input-oriented):
          If θ_NIRS ≈ θ_VRS  → DRS  (shadow price of convexity is on NIRS boundary)
          If θ_NDRS ≈ θ_VRS  → IRS
          If both ≈ θ_VRS    → CRS  (already at MPSS)
        """
        bcc_s  = float(scores_bcc[j])
        nirs_s = float(scores_nirs[j])
        ndrs_s = float(scores_ndrs[j])
        tol    = 1e-4

        if scores_ccr[j] >= 0.999:
            return "Constant (CRS)"
        if abs(nirs_s - bcc_s) < tol and abs(ndrs_s - bcc_s) < tol:
            return "Constant (CRS)"
        if abs(nirs_s - bcc_s) < tol:
            return "Decreasing (DRS)"
        if abs(ndrs_s - bcc_s) < tol:
            return "Increasing (IRS)"
        return "Constant (CRS)"

    dmu_results = []
    for j, d in enumerate(dmus):
        rts = classify_rts(j)
        dmu_results.append(DMUResult(
            name=d.name,
            score=round(float(scale_eff[j]), 4),
            rank=j + 1,
            classification=rts,
            metadata={
                "ccr_score":   round(float(scores_ccr[j]), 4),
                "bcc_score":   round(float(scores_bcc[j]), 4),
                "scale_eff":   round(float(scale_eff[j]), 4),
                "rts":         rts,
                "inputs":  d.inputs,
                "outputs": d.outputs,
            },
        ))

    dmu_results.sort(key=lambda x: -x.score)
    for i, r in enumerate(dmu_results):
        r.rank = i + 1

    n_crs = sum(1 for r in dmu_results if "Constant" in r.classification)
    n_irs = sum(1 for r in dmu_results if "Increasing" in r.classification)
    n_drs = sum(1 for r in dmu_results if "Decreasing" in r.classification)

    avg_scale = float(np.mean(scale_eff))
    mpss_candidates = [r.name for r in dmu_results if r.score >= 0.95 and "Constant" in r.classification]

    summary = {
        "n_dmus":        len(dmus),
        "avg_scale_eff": round(avg_scale, 4),
        "n_crs":         n_crs,
        "n_irs":         n_irs,
        "n_drs":         n_drs,
        "mpss_candidates": mpss_candidates[:5],
    }

    insights: List[str] = []
    insights.append(
        f"Average scale efficiency: {avg_scale:.3f}. "
        f"{n_crs} DMUs at CRS, {n_irs} at IRS, {n_drs} at DRS."
    )

    if n_irs > n_drs:
        insights.append(
            f"Majority of DMUs ({n_irs}) exhibit Increasing Returns to Scale. "
            f"These units are operating below their Most Productive Scale Size (MPSS) — "
            f"expanding scale could improve efficiency."
        )
    elif n_drs > n_irs:
        insights.append(
            f"Majority of DMUs ({n_drs}) exhibit Decreasing Returns to Scale. "
            f"These units are too large relative to their optimal scale — "
            f"downsizing or restructuring may improve efficiency."
        )

    if mpss_candidates:
        insights.append(
            f"MPSS candidates (high scale efficiency at CRS): {', '.join(mpss_candidates[:3])}. "
            f"These DMUs are operating at their optimal scale."
        )

    return DEAResult(
        analysis_type="rts",
        model="CCR+BCC",
        orientation=orientation,
        summary=summary,
        dmus=dmu_results,
        insights=insights,
    )


# ─── Module 4: Sensitivity Analysis ──────────────────────────────────────────

def analyze_sensitivity(dmus: List[DMUInput], model: str, orientation: str, params: dict) -> DEAResult:
    """
    Perturb each input/output by ±delta% and measure score change.
    Identifies which variables most influence the frontier.
    """
    names       = [d.name for d in dmus]
    input_keys  = list(dmus[0].inputs.keys())
    output_keys = list(dmus[0].outputs.keys())

    input_matrix  = np.array([[d.inputs[k]  for k in input_keys]  for d in dmus], dtype=float)
    output_matrix = np.array([[d.outputs[k] for k in output_keys] for d in dmus], dtype=float)

    base_scores = solve_dea(input_matrix, output_matrix, model, orientation)
    delta = float(params.get("delta", 0.1))  # default ±10%

    sensitivity: Dict[str, float] = {}

    for k, name in enumerate(input_keys):
        perturbed = input_matrix.copy(); perturbed[:, k] *= (1 + delta)
        s_up = solve_dea(perturbed, output_matrix, model, orientation)
        sensitivity[f"in_{name}_+{int(delta*100)}%"] = round(float(np.mean(base_scores - s_up)), 4)

        perturbed[:, k] = input_matrix[:, k] * (1 - delta)
        s_dn = solve_dea(perturbed, output_matrix, model, orientation)
        sensitivity[f"in_{name}_-{int(delta*100)}%"] = round(float(np.mean(s_dn - base_scores)), 4)

    for k, name in enumerate(output_keys):
        perturbed = output_matrix.copy(); perturbed[:, k] *= (1 + delta)
        s_up = solve_dea(input_matrix, perturbed, model, orientation)
        sensitivity[f"out_{name}_+{int(delta*100)}%"] = round(float(np.mean(s_up - base_scores)), 4)

        perturbed[:, k] = output_matrix[:, k] * (1 - delta)
        s_dn = solve_dea(input_matrix, perturbed, model, orientation)
        sensitivity[f"out_{name}_-{int(delta*100)}%"] = round(float(np.mean(base_scores - s_dn)), 4)

    # Rank variables by average absolute impact
    impact = {k: abs(v) for k, v in sensitivity.items()}
    ranked_vars = sorted(impact.items(), key=lambda x: -x[1])

    dmu_results = []
    for j, d in enumerate(dmus):
        dmu_results.append(DMUResult(
            name=d.name,
            score=round(float(base_scores[j]), 4),
            rank=j + 1,
            classification="Efficient" if base_scores[j] >= 0.999 else "Inefficient",
            metadata={"base_score": round(float(base_scores[j]), 4)},
        ))

    summary = {
        "n_dmus":          len(dmus),
        "delta":           delta,
        "sensitivity":     sensitivity,
        "ranked_impact":   ranked_vars[:10],
        "most_sensitive":  ranked_vars[0][0] if ranked_vars else None,
    }

    insights: List[str] = []
    if ranked_vars:
        top_var = ranked_vars[0]
        insights.append(
            f"Most sensitive variable: {top_var[0]} "
            f"(average score change = {top_var[1]:.4f} per {int(delta*100)}% perturbation). "
            f"This variable has the largest impact on the efficiency frontier."
        )

    high_impact = [k for k, v in ranked_vars if v > 0.05]
    if len(high_impact) > 1:
        insights.append(
            f"{len(high_impact)} variables show high sensitivity (impact > 0.05): "
            f"{', '.join(high_impact[:3])}. Measurement accuracy for these variables is critical."
        )

    low_impact = [k for k, v in ranked_vars if v < 0.01]
    if low_impact:
        insights.append(
            f"{len(low_impact)} variable(s) show minimal sensitivity — "
            f"consider dropping them from the model to reduce dimensionality."
        )

    return DEAResult(
        analysis_type="sensitivity",
        model=model,
        orientation=orientation,
        summary=summary,
        dmus=dmu_results,
        insights=insights,
    )


# ─── Module 5: Window Analysis ────────────────────────────────────────────────

def analyze_window(dmus: List[DMUInput], model: str, orientation: str, params: dict) -> DEAResult:
    """
    Window analysis: each DMU is treated as a different unit across time windows.
    Requires dmu.metadata['period'] (int) to be populated via params['periods'].
    Falls back to treating DMUs sequentially if no period data.
    """
    periods = params.get("periods", None)   # list of period labels aligned with dmus
    window  = int(params.get("window", 3))

    names       = [d.name for d in dmus]
    input_keys  = list(dmus[0].inputs.keys())
    output_keys = list(dmus[0].outputs.keys())

    input_matrix  = np.array([[d.inputs[k]  for k in input_keys]  for d in dmus], dtype=float)
    output_matrix = np.array([[d.outputs[k] for k in output_keys] for d in dmus], dtype=float)

    n = len(dmus)
    # If no period metadata, treat as single-panel sliding window
    period_labels = periods if periods else list(range(n))
    unique_periods = sorted(set(period_labels))

    if len(unique_periods) < window:
        # Not enough periods — fall back to base efficiency
        scores = solve_dea(input_matrix, output_matrix, model, orientation)
        dmu_results = [
            DMUResult(
                name=d.name,
                score=round(float(scores[j]), 4),
                rank=j + 1,
                classification="Efficient" if scores[j] >= 0.999 else "Inefficient",
                metadata={"note": "Insufficient periods for window analysis"},
            )
            for j, d in enumerate(dmus)
        ]
        return DEAResult(
            analysis_type="window",
            model=model,
            orientation=orientation,
            summary={"note": "Insufficient period data", "n_dmus": n},
            dmus=dmu_results,
            insights=["Not enough time periods detected. Window analysis requires at least 3 periods."],
        )

    window_scores: Dict[str, List[float]] = {d.name: [] for d in dmus}
    window_results = []

    for start in range(len(unique_periods) - window + 1):
        window_periods = unique_periods[start:start + window]
        idx = [i for i, p in enumerate(period_labels) if p in window_periods]
        if not idx:
            continue
        X_w = input_matrix[idx]; Y_w = output_matrix[idx]
        s_w = solve_dea(X_w, Y_w, model, orientation)
        for rel_i, abs_i in enumerate(idx):
            window_scores[names[abs_i]].append(float(s_w[rel_i]))
        window_results.append({
            "window": f"{window_periods[0]}–{window_periods[-1]}",
            "avg_score": round(float(np.mean(s_w)), 4),
            "n_efficient": int(np.sum(s_w >= 0.999)),
        })

    dmu_results = []
    for j, d in enumerate(dmus):
        ws = window_scores[d.name]
        avg_s = float(np.mean(ws)) if ws else 0.0
        trend = float(ws[-1] - ws[0]) if len(ws) >= 2 else 0.0
        dmu_results.append(DMUResult(
            name=d.name,
            score=round(avg_s, 4),
            rank=0,
            classification="Improving" if trend > 0.02 else "Declining" if trend < -0.02 else "Stable",
            metadata={
                "window_scores": [round(s, 4) for s in ws],
                "trend":         round(trend, 4),
                "avg_score":     round(avg_s, 4),
            },
        ))

    dmu_results.sort(key=lambda x: -x.score)
    for i, r in enumerate(dmu_results):
        r.rank = i + 1

    improving = [r.name for r in dmu_results if r.classification == "Improving"]
    declining = [r.name for r in dmu_results if r.classification == "Declining"]

    summary = {
        "n_dmus":        n,
        "n_periods":     len(unique_periods),
        "window_size":   window,
        "n_windows":     len(window_results),
        "window_results": window_results,
        "improving":     improving[:5],
        "declining":     declining[:5],
        "avg_score":     round(float(np.mean([r.score for r in dmu_results])), 4),
    }

    insights: List[str] = []
    if improving:
        insights.append(
            f"{len(improving)} DMU(s) show improving efficiency over time: "
            f"{', '.join(improving[:3])}. These units are catching up to the frontier."
        )
    if declining:
        insights.append(
            f"{len(declining)} DMU(s) show declining efficiency: "
            f"{', '.join(declining[:3])}. Investigate whether this is due to scale changes, "
            f"technological lag, or deteriorating operations."
        )
    if not improving and not declining:
        insights.append(
            "No significant efficiency trends detected. "
            "The relative performance of DMUs is stable across periods."
        )

    return DEAResult(
        analysis_type="window",
        model=model,
        orientation=orientation,
        summary=summary,
        dmus=dmu_results,
        insights=insights,
    )


# ─── API Endpoint ─────────────────────────────────────────────────────────────

@router.post("/dea", response_model=DEAResult)
async def run_dea(request: DEARequest):
    if not request.dmus:
        raise HTTPException(status_code=400, detail="No DMUs provided.")
    if len(request.dmus) < 3:
        raise HTTPException(status_code=400, detail="At least 3 DMUs are required for DEA.")

    # Validate consistent keys
    input_keys  = set(request.dmus[0].inputs.keys())
    output_keys = set(request.dmus[0].outputs.keys())
    for d in request.dmus[1:]:
        if set(d.inputs.keys()) != input_keys or set(d.outputs.keys()) != output_keys:
            raise HTTPException(status_code=400, detail="All DMUs must have identical input/output variable names.")

    try:
        t = request.analysis_type
        p = request.params or {}

        if t == "efficiency":
            return analyze_efficiency(request.dmus, request.model, request.orientation)
        elif t == "benchmarking":
            return analyze_benchmarking(request.dmus, request.model, request.orientation, p)
        elif t == "rts":
            return analyze_rts(request.dmus, request.orientation)
        elif t == "sensitivity":
            return analyze_sensitivity(request.dmus, request.model, request.orientation, p)
        elif t == "window":
            return analyze_window(request.dmus, request.model, request.orientation, p)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported analysis_type: '{t}'. "
                       f"Available: efficiency, benchmarking, rts, sensitivity, window",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/dea/health")
async def health():
    return {
        "status":  "ok",
        "version": "1.0",
        "modules": ["efficiency", "benchmarking", "rts", "sensitivity", "window"],
        "models":  ["CCR", "BCC", "SBM"],
        "endpoints": {
            "analysis": "POST /api/analysis/dea",
            "health":   "GET  /api/analysis/dea/health",
        },
    }
