# Validates conjoint_analysis.py (POST /api/analysis/conjoint, choice-based conjoint,
# aggregate conditional logit) against statsmodels ConditionalLogit on an independently
# built dummy design, in the exact row format the app sends (CBCAnalysisTab: respondent_id,
# task_id, alternative 1..n | 'none', choice 0/1, one column per attribute, '' on none rows).
#
# Two data sets: (1) with a "none" alternative and a price attribute, (2) without "none".
# Checked: coefficients, SE, z, p (dummy coding, alphabetical-first reference level), none
# constant, log-likelihood, null log-likelihood, AIC, AICc, BIC, McFadden and adjusted R²,
# hit rate, importance, WTP (continuous-price refit, delta-method CI), share-of-preference
# simulation, data summary, segmentation bookkeeping, and recovery of the planted utilities.
import json
import numpy as np, pandas as pd
from scipy.stats import norm
from statsmodels.discrete.conditional_models import ConditionalLogit
from _pyharness import run_router, chk, report

LEVELS = {"Brand": ["Apple", "Banana", "Cherry"],
          "Price": ["₩1,000", "₩1,500", "₩2,000"],
          "Storage": ["128GB", "256GB", "64GB"]}            # sorted() order = backend reference order
AMOUNT = {"₩1,000": 1000.0, "₩1,500": 1500.0, "₩2,000": 2000.0}
# planted utilities relative to the alphabetical-first level (that level = 0)
TRUE = {"Brand": {"Banana": -0.6, "Cherry": 0.5},
        "Price": {"₩1,500": -0.7, "₩2,000": -1.5},
        "Storage": {"256GB": 0.6, "64GB": -0.8}}
ASC_NONE = -1.2


def simulate(seed, n_resp, n_tasks, n_alts, none):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_resp):
        for t in range(n_tasks):
            alts = []
            for a in range(n_alts):
                lv = {k: str(rng.choice(v)) for k, v in LEVELS.items()}
                u = sum(TRUE[k].get(lv[k], 0.0) for k in LEVELS)
                alts.append((lv, u))
            us = [u for _, u in alts] + ([ASC_NONE] if none else [])
            pick = int(np.argmax(np.array(us) + rng.gumbel(size=len(us))))
            for a, (lv, _) in enumerate(alts):
                rows.append({"respondent_id": f"R{r}", "task_id": f"T{t+1}", "alternative": a + 1,
                             "choice": int(pick == a), **lv})
            if none:
                rows.append({"respondent_id": f"R{r}", "task_id": f"T{t+1}", "alternative": "none",
                             "choice": int(pick == n_alts), **{k: "" for k in LEVELS}})
    return rows


def reference(rows, none, continuous_price=False):
    df = pd.DataFrame(rows)
    df["grp"] = df.respondent_id + "|" + df.task_id
    is_none = df.alternative.astype(str) == "none"
    cols = {}
    for attr, lv in LEVELS.items():
        if continuous_price and attr == "Price":
            cols["Price"] = np.where(is_none, 0.0, df[attr].map(AMOUNT).fillna(0.0)) / 1000.0  # fitted per 1,000 won
            continue
        for l in lv[1:]:
            cols[f"{attr}__{l}"] = np.where(is_none, 0.0, (df[attr] == l).astype(float))
    if none:
        cols["_ASC_none"] = is_none.astype(float).values
    X = pd.DataFrame(cols)
    m = ConditionalLogit(df.choice.values, X.values, groups=df.grp.values)
    f = m.fit(method="bfgs", maxiter=2000, disp=False, gtol=1e-10)
    return df, X, f


def check_dataset(tag, rows, none, sim_profiles):
    res = run_router("conjoint_analysis", "/conjoint", {
        "data": rows, "responseCol": "choice", "taskCol": "task_id", "respCol": "respondent_id",
        "attributeCols": list(LEVELS), "priceCol": "Price", "nSegments": 3,
        "noneOption": none, "simulationProfiles": sim_profiles})["results"]
    df, X, f = reference(rows, none)
    names = list(X.columns)
    b = dict(zip(names, f.params)); se = dict(zip(names, f.bse))
    coef = {(c["attribute"], c["level"]): c for c in res["charts"]["coefficient_table"]}

    chk(f"{tag}.ref_levels", json.dumps(res["ref_levels"], sort_keys=True), json.dumps({k: v[0] for k, v in LEVELS.items()}, sort_keys=True))
    chk(f"{tag}.converged", bool(res["model_info"]["converged"]), True)
    for attr, lv in LEVELS.items():
        for l in lv[1:]:
            n_ = f"{attr}__{l}"; c = coef[(attr, l)]
            z = b[n_] / se[n_]
            chk(f"{tag}.coef.{attr}.{l}", c["coef"], b[n_], tol=1e-3)
            chk(f"{tag}.se.{attr}.{l}", c["se"], se[n_], tol=5e-4)
            chk(f"{tag}.z.{attr}.{l}", c["z"], z, tol=2e-2)
            chk(f"{tag}.p.{attr}.{l}", c["p"], 2 * norm.sf(abs(z)), tol=1e-3)
            chk(f"{tag}.ci_low.{attr}.{l}", c["ci_lower"], b[n_] - 1.96 * se[n_], tol=3e-3)
            chk(f"{tag}.recovery.{attr}.{l}", c["coef"], TRUE[attr][l], tol=0.2)
        chk(f"{tag}.reference.{attr}.{lv[0]}", coef[(attr, lv[0])]["coef"], 0.0, tol=0)
    if none:
        # the none constant is not in the coefficient table; the WTP refit carries it, the
        # main model's value is checked through the log-likelihood and the simulation below
        chk(f"{tag}.recovery.ASC_none", b["_ASC_none"], ASC_NONE, tol=0.2)

    fs = res["fit_statistics"]
    k = len(names); n_sets = df.grp.nunique()
    ll0 = float(sum(np.log(1.0 / s) for s in df.groupby("grp").size()))
    chk(f"{tag}.loglik", fs["log_likelihood"], f.llf, tol=1e-3)
    chk(f"{tag}.loglik_null", fs["log_likelihood_null"], ll0, tol=1e-8)
    chk(f"{tag}.aic", fs["aic"], 2 * k - 2 * f.llf, tol=2e-3)
    chk(f"{tag}.bic(n=choice sets)", fs["bic"], k * np.log(n_sets) - 2 * f.llf, tol=2e-3)
    chk(f"{tag}.aicc", fs["aicc"], 2 * k - 2 * f.llf + 2 * k * (k + 1) / (n_sets - k - 1), tol=2e-3)
    chk(f"{tag}.mcfadden_r2", fs["mcfadden_r2"], 1 - f.llf / ll0, tol=1e-6)
    chk(f"{tag}.adj_mcfadden_r2", fs["adj_mcfadden_r2"], 1 - (f.llf - k) / ll0, tol=1e-6)
    V = X.values @ f.params
    hits = sum(int(np.argmax(V[g.index]) == np.argmax(df.choice.values[g.index])) for _, g in df.groupby("grp"))
    chk(f"{tag}.hit_rate", fs["hit_rate"], hits / n_sets, tol=1e-12)

    ds = res["data_summary"]
    chk(f"{tag}.n_respondents", ds["n_respondents"], df.respondent_id.nunique(), tol=0)
    chk(f"{tag}.n_tasks(choice sets)", ds["n_tasks"], n_sets, tol=0)
    chk(f"{tag}.n_parameters", ds["n_parameters"], k, tol=0)
    if none:
        nr = df[(df.alternative.astype(str) == "none")].choice.sum() / n_sets * 100
        chk(f"{tag}.none_rate_pct", ds["none_rate"], round(nr, 1), tol=1e-9)

    ranges = {a: max([0.0] + [b[f"{a}__{l}"] for l in lv[1:]]) - min([0.0] + [b[f"{a}__{l}"] for l in lv[1:]])
              for a, lv in LEVELS.items()}
    tot = sum(ranges.values())
    for a in LEVELS:
        chk(f"{tag}.importance.{a}", res["importance"][a], ranges[a] / tot * 100, tol=0.05)

    # WTP: refit with one linear price term (raw amount) + the same none constant.
    _, Xw, fw = reference(rows, none, continuous_price=True)
    bw = dict(zip(Xw.columns, fw.params)); cov = pd.DataFrame(fw.cov_params(), index=Xw.columns, columns=Xw.columns)
    S_ = np.ones(len(Xw.columns)); S_[list(Xw.columns).index("Price")] = 1 / 1000.0   # per-1,000 -> per-won
    bw = {k_: v_ * S_[i] for i, (k_, v_) in enumerate(bw.items())}
    cov = pd.DataFrame(fw.cov_params() * np.outer(S_, S_), index=Xw.columns, columns=Xw.columns)
    bp = bw["Price"]
    chk(f"{tag}.beta_price", res["beta_price"], bp, tol=2e-6)
    wtp = {(w["attribute"], w["level"]): w for w in res["wtp"]}
    for a in ("Brand", "Storage"):
        for l in LEVELS[a][1:]:
            n_ = f"{a}__{l}"; bk = bw[n_]
            w = -bk / bp
            # delta method with the full covariance (the backend's formula drops cov(b_k, b_price))
            g = np.zeros(len(Xw.columns)); g[list(Xw.columns).index(n_)] = -1 / bp
            g[list(Xw.columns).index("Price")] = bk / bp ** 2
            se_full = float(np.sqrt(g @ cov.values @ g))
            se_nocov = abs(1 / bp) * np.sqrt(cov.loc[n_, n_] + (bk / bp) ** 2 * cov.loc["Price", "Price"])
            chk(f"{tag}.wtp.{a}.{l}", wtp[(a, l)]["wtp"], w, tol=1.0)
            # The backend's delta-method SE omits cov(b_k, b_price): it matches that formula; the
            # gap to the full delta method is recorded as a finding (FINDINGS below), not a check.
            chk(f"{tag}.wtp_ci_low(backend formula, no cov term).{a}.{l}", wtp[(a, l)]["wtp_ci_lower"], w - 1.96 * se_nocov, tol=1.5)
            FINDINGS.append((f"{tag} WTP {a}={l}", w, 1.96 * se_nocov, 1.96 * se_full))
            chk(f"{tag}.wtp.recovery.{a}.{l}", w, -TRUE[a][l] / (-1.5 / 1000.0), tol=0.35 * abs(TRUE[a][l] / (1.5 / 1000.0)))

    # Share-of-preference simulation: logit shares of the profiles (no none alternative).
    sim = res["simulation"]
    util = [sum(b.get(f"{a}__{p[a]}", 0.0) for a in LEVELS) for p in sim_profiles]
    e = np.exp(np.array(util) - max(util)); sh = e / e.sum() * 100
    for i, p in enumerate(sim_profiles):
        chk(f"{tag}.sim.{p['label']}.utility", sim[i]["utility"], util[i], tol=2e-3)
        chk(f"{tag}.sim.{p['label']}.share_pct", sim[i]["share_pct"], sh[i], tol=0.05)

    seg = res["segments"]
    chk(f"{tag}.segments.count", len(seg), 3, tol=0)
    chk(f"{tag}.segments.respondents_sum", sum(s["n_respondents"] for s in seg), df.respondent_id.nunique(), tol=0)
    chk(f"{tag}.segments.pct_sum", sum(s["pct_respondents"] for s in seg), 100.0, tol=1e-9)
    for s in seg:
        chk(f"{tag}.segment{s['segment_id']}.importance_sum", sum(s["importance"].values()), 100.0, tol=1e-9)
    return res


SIM = [{"label": "Phone A", "Brand": "Apple", "Price": "₩2,000", "Storage": "256GB"},
       {"label": "Phone B", "Brand": "Cherry", "Price": "₩1,500", "Storage": "128GB"},
       {"label": "Phone C", "Brand": "Banana", "Price": "₩1,000", "Storage": "64GB"}]

FINDINGS = []
check_dataset("cbc_none", simulate(11, 250, 10, 3, True), True, SIM)
check_dataset("cbc_plain", simulate(12, 250, 10, 3, False), False, SIM)

print("\nFINDING — WTP 95% CI half-width: backend (delta method without the cov term) vs full delta method")
for name, w, h_nocov, h_full in FINDINGS:
    print(f"  {name}: WTP {w:,.1f}  half-width {h_nocov:,.1f} vs {h_full:,.1f}  ({(h_nocov - h_full) / h_full * 100:+.1f}%)")
report("CHOICE-BASED CONJOINT, AGGREGATE MNL (Python)")
