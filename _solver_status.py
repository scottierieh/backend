"""Map SciPy solver result codes to the status strings the optimisation pages read.

Several *_analysis.py scripts used to report every unsuccessful solve as
status "infeasible", so an unbounded model, an iteration/time limit or a
numerical failure all read as "no plan satisfies the constraints". These
helpers turn the solver's own status code into a specific status and keep
the code and the solver's message alongside it:

    status              "infeasible" (unchanged value for real infeasibility,
                        so existing callers keep working), "unbounded",
                        "iteration_limit", "time_limit", "numerical",
                        "solver_error"
    unsolved            True
    solver_status_code  the raw res.status integer
    solver_message      the raw res.message string
    message             a readable sentence: what happened + solver message

scipy.optimize.linprog (method="highs") res.status:
    0 optimal, 1 iteration limit, 2 infeasible, 3 unbounded, 4 numerical difficulties
scipy.optimize.milp res.status:
    0 optimal, 1 iteration or time limit reached (a feasible x may exist),
    2 infeasible, 3 unbounded, 4 other
"""

_LINPROG = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "numerical"}

_DEFAULT_LEAD = {
    "infeasible": "No solution satisfies every constraint.",
    "unbounded": "The model is unbounded: the objective can improve without limit, "
                 "so there is no best solution (a limit or usage coefficient is probably missing).",
    "iteration_limit": "The solver stopped at its iteration limit before proving an optimum.",
    "time_limit": "The solver stopped at its time limit before proving an optimum.",
    "numerical": "The solver ran into numerical difficulties and could not certify a solution; "
                 "check for very large or very small coefficients.",
    "solver_error": "The solver could not solve the model.",
}


def _payload(status, code, raw_msg, leads):
    lead = (leads or {}).get(status) or _DEFAULT_LEAD.get(status, _DEFAULT_LEAD["solver_error"])
    raw = str(raw_msg or "").strip()
    return {
        "status": status,
        "unsolved": True,
        "message": f"{lead} {raw}".strip() if raw else lead,
        "solver_status_code": int(code) if code is not None else None,
        "solver_message": raw or None,
    }


def linprog_failure(res, leads=None):
    """results{} for an unsuccessful scipy.optimize.linprog call.

    leads: optional {status: sentence} to replace the default lead sentence
    for a status (e.g. a model-specific explanation of infeasibility).
    """
    code = getattr(res, "status", None)
    status = _LINPROG.get(code, "solver_error")
    if status == "optimal":          # success=False with status 0 should not happen
        status = "solver_error"
    return _payload(status, code, getattr(res, "message", ""), leads)


def milp_failure(res, leads=None):
    """results{} for an unsuccessful scipy.optimize.milp call.

    Status 1 covers both the time and the iteration (node) limit; the HiGHS
    message says which. When the solver stopped at a limit with a feasible
    (but unproven) point, feasible_solution_found is True and the message
    says so; the point itself is not reported as a plan.
    """
    code = getattr(res, "status", None)
    raw = str(getattr(res, "message", "") or "")
    if code == 0:
        status = "solver_error"
    elif code == 1:
        status = "time_limit" if "time" in raw.lower() else "iteration_limit"
    elif code == 2:
        status = "infeasible"
    elif code == 3:
        status = "unbounded"
    else:
        status = "solver_error"
    out = _payload(status, code, raw, leads)
    if code == 1:
        found = getattr(res, "x", None) is not None
        out["feasible_solution_found"] = bool(found)
        out["message"] = (out["message"] + (
            " A feasible solution was found but it is not proven optimal, so it is not reported."
            if found else " No feasible solution had been found yet."))
    return out


def solved_meta(res):
    """Extra fields for a successful solve (status code + solver message)."""
    return {"solver_status_code": int(getattr(res, "status", 0) or 0),
            "solver_message": str(getattr(res, "message", "") or "") or None}
