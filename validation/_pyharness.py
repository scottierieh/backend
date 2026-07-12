"""Shared helpers for Python backend validation (scottierieh/backend).

Instrumented to mirror the R harness (r-backend/validation/_harness.R): every
chk() records its tolerance and the observed absolute difference, and report()
writes a machine-readable metadata sidecar to validation/_results/<script>.json
capturing check counts, pass/fail, tolerance range, maximum observed
difference, package versions, the Python version, the git commit and the run
date. Those measured facts feed the per-analysis validation documents.
"""
import json, subprocess, sys, os, platform, datetime

PASS=[0]; FAIL=[0]; LOG=[]; CHECKS=[]

def run_script(script, payload):
    """Run a backend CLI script with a JSON payload on stdin, return parsed JSON stdout."""
    root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p=subprocess.run([sys.executable, os.path.join(root,script)],
                     input=json.dumps(payload), capture_output=True, text=True)
    if p.returncode!=0:
        raise RuntimeError(f"{script} failed: {p.stderr[:500]}")
    return json.loads(p.stdout)

def chk(name, got, exp, tol=1e-6):
    is_num = not isinstance(exp,(str,bool))
    diff=None
    if is_num and got is not None and exp is not None:
        try: diff=abs(float(got)-float(exp))
        except (TypeError, ValueError): diff=None
    if is_num:
        ok = got is not None and exp is not None and diff is not None and diff<=tol
    else:
        ok = got==exp
    if ok: PASS[0]+=1; LOG.append(f"PASS | {name} = {got}")
    else:  FAIL[0]+=1; LOG.append(f"FAIL | {name} got={got} exp={exp}")
    CHECKS.append({"name":name, "tolerance":(tol if is_num else None),
                   "abs_difference":diff, "numeric":is_num,
                   "status":("PASS" if ok else "FAIL")})
    return ok

def _git_commit():
    try:
        root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out=subprocess.run(["git","-C",root,"rev-parse","--short","HEAD"],
                           capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:
        return None

# versions of the analysis-relevant packages actually imported by this run
def _pkg_versions():
    candidates=["numpy","pandas","scipy","sklearn","statsmodels","xgboost","lightgbm",
                "catboost","umap","hdbscan","shap","optuna"]
    out={}
    for name in candidates:
        mod=sys.modules.get(name)
        if mod is None: continue
        ver=getattr(mod,"__version__",None)
        if ver: out[name]=str(ver)
    return dict(sorted(out.items()))

def report(title):
    print("\n".join(LOG)); print(f"\n==== {title}: {PASS[0]} PASS, {FAIL[0]} FAIL ====")
    diffs=[c["abs_difference"] for c in CHECKS if c["numeric"] and c["abs_difference"] is not None]
    tols =[c["tolerance"] for c in CHECKS if c["numeric"] and c["tolerance"] is not None]
    n_num=sum(1 for c in CHECKS if c["numeric"])
    total=len(CHECKS)
    meta={
        "title": title,
        "total_checks": total,
        "numeric_checks": n_num,
        "exact_checks": total-n_num,
        "pass": PASS[0], "fail": FAIL[0],
        "pass_rate": (round(100*PASS[0]/total,2) if total else None),
        "tolerance_min": (min(tols) if tols else None),
        "tolerance_max": (max(tols) if tols else None),
        "max_abs_difference": (max(diffs) if diffs else None),
        "all_within_tolerance": FAIL[0]==0,
        "language": "Python",
        "runtime_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "git_commit": _git_commit(),
        "validation_date": datetime.date.today().isoformat(),
        "checks": [{"name":c["name"],"tolerance":c["tolerance"],
                    "abs_difference":c["abs_difference"],"status":c["status"]} for c in CHECKS],
    }
    try:
        here=os.path.dirname(os.path.abspath(__file__))
        outdir=os.path.join(here,"_results"); os.makedirs(outdir, exist_ok=True)
        script=os.path.splitext(os.path.basename(sys.argv[0]))[0] if sys.argv and sys.argv[0] else "unknown"
        with open(os.path.join(outdir, script+".json"),"w") as f:
            json.dump(meta,f,indent=2)
    except Exception as e:
        sys.stderr.write(f"pyharness: could not write metadata: {e}\n")
    return meta

def find_key(d, key):
    """Depth-first search for the first occurrence of `key` in nested dicts/lists."""
    if isinstance(d, dict):
        if key in d: return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None: return r
    elif isinstance(d, list):
        for v in d:
            r = find_key(v, key)
            if r is not None: return r
    return None
