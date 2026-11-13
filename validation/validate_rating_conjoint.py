# Validates rc_analysis.py (POST /api/analysis/rating-conjoint, rating-based conjoint)
# against statsmodels OLS on an independently built effects-coded design (patsy Sum coding).
#
# Data: 12 estimation profiles + 2 holdout profiles, 80 respondents, 1–10 ratings simulated
# from known part-worths plus respondent noise; a few ratings left blank; a segment variable.
# Checked: every coefficient / part-worth (including the reference level, -Σβ) with SE, t, p
# and 95% CI; intercept; R², adjusted R², RMSE, log-likelihood, AIC, AICc, BIC, F and its p;
# importance; WTP; holdout MSE/MAE/R²; per-respondent part-worths; per-segment importance;
# the respondent bootstrap (same seed, refitted with statsmodels).
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from _pyharness import run_router, chk, report

rng = np.random.default_rng(7)
ATTR = [
    {"id": "brand", "name": "Brand", "levels": [{"value": "Alpha"}, {"value": "Beta"}, {"value": "Gamma"}]},
    {"id": "price", "name": "Price", "levels": [{"value": "1,000원"}, {"value": "1,500원"}, {"value": "2,000원"}]},
    {"id": "size", "name": "Size", "levels": [{"value": "S"}, {"value": "M"}, {"value": "L"}, {"value": "XL"}]},
]
TRUE = {"brand": {"Alpha": 0.8, "Beta": 0.1, "Gamma": -0.9},
        "price": {"1,000원": 1.2, "1,500원": 0.1, "2,000원": -1.3},
        "size": {"S": -0.7, "M": 0.0, "L": 0.4, "XL": 0.3}}
LV = {a["id"]: [l["value"] for l in a["levels"]] for a in ATTR}

# 12 estimation profiles (balanced-ish fraction of 3x3x4) + 2 holdouts.
combos = [(b, p, s) for b in LV["brand"] for p in LV["price"] for s in LV["size"]]
pick = rng.choice(len(combos), 14, replace=False)
profiles = []
for i, k in enumerate(pick):
    b, p, s = combos[k]
    profiles.append({"id": f"p{i+1}", "levels": {"brand": b, "price": p, "size": s}, "isHoldout": i >= 12})

responses = []
for r in range(80):
    shift = rng.normal(0, 0.3, size=3)  # respondent-level heterogeneity in the attribute weights
    ratings = {}
    for pr in profiles:
        lv = pr["levels"]
        u = 5.5 + (1 + shift[0]) * TRUE["brand"][lv["brand"]] + (1 + shift[1]) * TRUE["price"][lv["price"]] \
            + (1 + shift[2]) * TRUE["size"][lv["size"]] + rng.normal(0, 0.9)
        ratings[pr["id"]] = float(np.clip(np.round(u), 1, 10))
    if r % 17 == 3:  # a few blanks
        ratings[profiles[2]["id"]] = None
    responses.append({"respondentId": f"r{r}", "ratings": ratings, "segment": "A" if r % 3 else "B"})

res = run_router("rc_analysis", "/rating-conjoint",
                 {"profiles": profiles, "responses": responses, "attributes": ATTR, "priceAttributeId": "price"})["results"]

# ---- reference: long data + statsmodels OLS with Sum (effects) coding ----
est = [p for p in profiles if not p["isHoldout"]]
rows = []
for resp in responses:
    for p in est:
        v = resp["ratings"].get(p["id"])
        if v is None: continue
        rows.append({"y": v, "resp": resp["respondentId"], "seg": resp["segment"], **p["levels"]})
df = pd.DataFrame(rows)
def formula():
    return ("y ~ " + " + ".join(f"C({a}, Sum, levels={LV[a]!r})" for a in ["brand", "price", "size"]))
fit = smf.ols(formula(), df).fit()
names = list(fit.params.index)

def level_contrast(attr, j, K):
    """Contrast vector picking level j of attr (last level = minus the sum of the others)."""
    c = np.zeros(len(names))
    cols = [i for i, n in enumerate(names) if n.startswith(f"C({attr},")]
    if j < K - 1: c[cols[j]] = 1.0
    else: c[cols] = -1.0
    return c

pw = {d["attributeId"]: d for d in res["partWorths"]}
ref_util = {}
for a in ATTR:
    aid, K = a["id"], len(a["levels"])
    ref_util[aid] = {}
    for j, lvl in enumerate(LV[aid]):
        c = level_contrast(aid, j, K)
        tt = fit.t_test(c)
        est_v = float(tt.effect[0]); se = float(tt.sd[0]); t = float(tt.tvalue[0]); p = float(tt.pvalue)
        lo, hi = map(float, tt.conf_int(0.05)[0])
        ref_util[aid][lvl] = est_v
        got = pw[aid]["levels"][j]
        chk(f"rc.{aid}.{lvl}.utility", got["utility"], est_v, tol=1e-9)
        chk(f"rc.{aid}.{lvl}.se", got["se"], se, tol=1e-9)
        chk(f"rc.{aid}.{lvl}.t", got["t"], t, tol=1e-7)
        chk(f"rc.{aid}.{lvl}.p", got["pValue"], p, tol=1e-6)
        chk(f"rc.{aid}.{lvl}.ci_low", got["ciLow"], lo, tol=1e-6)
        chk(f"rc.{aid}.{lvl}.ci_high", got["ciHigh"], hi, tol=1e-6)

chk("rc.intercept", res["intercept"], float(fit.params["Intercept"]), tol=1e-9)
chk("rc.intercept.se", res["interceptStats"]["se"], float(fit.bse["Intercept"]), tol=1e-9)
chk("rc.intercept.p", res["interceptStats"]["pValue"], float(fit.pvalues["Intercept"]), tol=1e-6)
fs = res["fitStats"]
n = int(fit.nobs); k = len(fit.params)
chk("rc.r2", res["rSquared"], fit.rsquared, tol=1e-10)
chk("rc.adj_r2", fs["adjRSquared"], fit.rsquared_adj, tol=1e-10)
chk("rc.rmse(√(SSR/n))", fs["rmse"], np.sqrt(fit.ssr / n), tol=1e-10)
chk("rc.loglik", fs["logLikelihood"], fit.llf, tol=1e-8)
chk("rc.aic", fs["aic"], fit.aic, tol=1e-8)
chk("rc.bic", fs["bic"], fit.bic, tol=1e-8)
chk("rc.aicc", fs["aicc"], fit.aic + 2 * k * (k + 1) / (n - k - 1), tol=1e-8)
chk("rc.F", fs["fStat"], fit.fvalue, tol=1e-8)
chk("rc.F.p", fs["fPValue"], fit.f_pvalue, tol=1e-9)
chk("rc.df_resid", fs["dfResid"], fit.df_resid, tol=0)
chk("rc.n_obs", res["nObs"], n, tol=0)
chk("rc.n_respondents", res["n"], 80, tol=0)

ranges = {aid: max(u.values()) - min(u.values()) for aid, u in ref_util.items()}
tot = sum(ranges.values())
for aid in ranges:
    chk(f"rc.importance.{aid}", pw[aid]["importance"], ranges[aid] / tot * 100, tol=1e-9)

# WTP: least-squares slope of price utility on amount, WTP = -utility / slope.
amt = np.array([1000.0, 1500.0, 2000.0]); uu = np.array([ref_util["price"][v] for v in LV["price"]])
slope = np.polyfit(amt, uu, 1)[0]
for aid in ("brand", "size"):
    name = next(a["name"] for a in ATTR if a["id"] == aid)
    for lvl in LV[aid]:
        chk(f"rc.wtp.{aid}.{lvl}", res["wtp"][f"{name}_{lvl}"], -ref_util[aid][lvl] / slope, tol=1e-6)

# Holdout: predicted = intercept + Σ level utilities, against every holdout rating.
hold = [p for p in profiles if p["isHoldout"]]
pred, act = [], []
for resp in responses:
    for p in hold:
        v = resp["ratings"].get(p["id"])
        if v is None: continue
        act.append(v); pred.append(fit.params["Intercept"] + sum(ref_util[a][p["levels"][a]] for a in ref_util))
act, pred = np.array(act), np.array(pred)
hv = res["holdoutValidation"]
chk("rc.holdout.mse", hv["mse"], float(np.mean((act - pred) ** 2)), tol=1e-9)
chk("rc.holdout.mae", hv["mae"], float(np.mean(np.abs(act - pred))), tol=1e-9)
chk("rc.holdout.r2", hv["r2"], float(1 - np.sum((act - pred) ** 2) / np.sum((act - act.mean()) ** 2)), tol=1e-9)

# Recovery: the simulated utilities are centred, so the fitted ones should be close (sampling noise).
for aid, levels in TRUE.items():
    for lvl, tv in levels.items():
        chk(f"rc.recovery.{aid}.{lvl}", ref_util[aid][lvl], tv, tol=0.25)

# Per-respondent OLS (respondents with enough distinct ratings) — compare the first five.
ind = res["individualPartWorths"]
kept = []
for resp in responses:
    sub = df[df.resp == resp["respondentId"]]
    if len(sub) < k + 1 or sub.y.nunique() < 2: continue
    kept.append(resp["respondentId"])
chk("rc.individual.count", len(ind), len(kept), tol=0)
for i, rid in enumerate(kept[:5]):
    f1 = smf.ols(formula(), df[df.resp == rid]).fit()
    for aid in ("brand", "price", "size"):
        K = len(LV[aid])
        for j, lvl in enumerate(LV[aid]):
            v = float(level_contrast(aid, j, K) @ f1.params.values)
            chk(f"rc.individual[{rid}].{aid}.{lvl}", ind[i][aid][lvl], v, tol=1e-8)

# Per-segment importance.
segs = {d["segment"]: d for d in res["segmentImportance"]}
for seg in ("A", "B"):
    f2 = smf.ols(formula(), df[df.seg == seg]).fit()
    rr = {}
    for aid in ("brand", "price", "size"):
        K = len(LV[aid]); vals = [float(level_contrast(aid, j, K) @ f2.params.values) for j in range(K)]
        rr[aid] = max(vals) - min(vals)
    t2 = sum(rr.values())
    for a in ATTR:
        chk(f"rc.segment[{seg}].importance.{a['id']}", segs[seg]["importance"][a["name"]], rr[a["id"]] / t2 * 100, tol=1e-8)
    chk(f"rc.segment[{seg}].n", segs[seg]["n"], int(sum(1 for r in responses if r["segment"] == seg)), tol=0)

# Bootstrap: same generator and seed, resampling whole respondents, refit with statsmodels.
blocks = [df[df.resp == r["respondentId"]] for r in responses]
brng = np.random.default_rng(20240601)
imps = {a: [] for a in ("brand", "price", "size")}
for _ in range(500):
    idx = brng.integers(0, len(blocks), len(blocks))
    fb = smf.ols(formula(), pd.concat([blocks[i] for i in idx])).fit()
    rr = {}
    for aid in imps:
        K = len(LV[aid]); vals = [float(level_contrast(aid, j, K) @ fb.params.values) for j in range(K)]
        rr[aid] = max(vals) - min(vals)
    t3 = sum(rr.values())
    for aid in imps: imps[aid].append(rr[aid] / t3 * 100)
bci = {d["attributeId"]: d for d in res["bootstrap"]["importanceCI"]}
chk("rc.bootstrap.iterations", res["bootstrap"]["iterations"], 500, tol=0)
for aid, s in imps.items():
    chk(f"rc.bootstrap.{aid}.mean", bci[aid]["mean"], float(np.mean(s)), tol=1e-7)
    chk(f"rc.bootstrap.{aid}.ci_lo", bci[aid]["lo"], float(np.percentile(s, 2.5)), tol=1e-7)
    chk(f"rc.bootstrap.{aid}.ci_hi", bci[aid]["hi"], float(np.percentile(s, 97.5)), tol=1e-7)

report("RATING-BASED CONJOINT (Python)")
