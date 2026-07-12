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


def classifier_checks(prefix, result, model, X_train, y_train, X_test, y_test, tol=1e-9):
    """Cross-check every classification metric the handler reports against
    scikit-learn recomputed on the reproduced model's own predictions —
    test accuracy, train accuracy, macro precision/recall/F1, and the
    train/test split sizes. Metrics the handler does not expose are skipped
    (guarded on find_key), so a handler that reports a subset never triggers a
    false failure. y_train/y_test must be in the label space the model predicts."""
    import numpy as _np
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)
    g = lambda k: find_key(result, k)
    pred = _np.asarray(model.predict(X_test)).ravel()      # ravel handles CatBoost's (n,1) output
    chk(f"{prefix}.accuracy", g('accuracy'), accuracy_score(y_test, pred), tol=tol)
    if g('train_accuracy') is not None:
        pred_tr = _np.asarray(model.predict(X_train)).ravel()
        chk(f"{prefix}.train_accuracy", g('train_accuracy'), accuracy_score(y_train, pred_tr), tol=tol)

    # macro precision/recall/F1 — accept either flat keys (precision_macro) or a
    # nested sklearn classification_report ('macro avg' -> precision/recall/f1-score)
    cr = g('classification_report')
    macro = cr.get('macro avg') if isinstance(cr, dict) else None
    ref = {'precision': precision_score(y_test, pred, average='macro', zero_division=0),
           'recall':    recall_score(y_test, pred, average='macro', zero_division=0),
           'f1':        f1_score(y_test, pred, average='macro', zero_division=0)}
    for short, flat in (('precision', 'precision_macro'), ('recall', 'recall_macro'), ('f1', 'f1_macro')):
        if g(flat) is not None:
            chk(f"{prefix}.{flat}", g(flat), ref[short], tol=tol)
        elif isinstance(macro, dict):
            rep_key = 'f1-score' if short == 'f1' else short
            if macro.get(rep_key) is not None:
                chk(f"{prefix}.{short}_macro", macro[rep_key], ref[short], tol=tol)

    cm = g('confusion_matrix')
    if cm is not None:
        cm_ref = confusion_matrix(y_test, pred)
        cm_got = _np.asarray(cm)
        if cm_got.shape == cm_ref.shape:
            # verify every cell N_ij individually (value-by-value, not just aggregate)
            for i in range(cm_ref.shape[0]):
                for j in range(cm_ref.shape[1]):
                    chk(f"{prefix}.cm[{i},{j}]", int(cm_got[i, j]), int(cm_ref[i, j]))
        else:
            chk(f"{prefix}.confusion_matrix_shape", str(cm_got.shape), str(cm_ref.shape))
    if g('n_train') is not None:
        chk(f"{prefix}.n_train", g('n_train'), len(X_train))
    if g('n_test') is not None:
        chk(f"{prefix}.n_test", g('n_test'), len(X_test))
