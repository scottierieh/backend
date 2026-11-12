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

def _fmt(v):
    """Format a scalar value for the per-check detail table."""
    if v is None: return "None"
    if isinstance(v, bool): return str(v)
    if isinstance(v, (int,)): return str(v)
    try:
        f = float(v)
        if f == int(f) and abs(f) < 1e15: return str(int(f))
        return f"{f:.6g}"
    except (TypeError, ValueError):
        return str(v)

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
    CHECKS.append({"name":name, "got":_fmt(got), "expected":_fmt(exp),
                   "tolerance":(tol if is_num else None),
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
                "catboost","umap","hdbscan","shap","optuna","sklearn_extra","tensorflow","keras"]
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
        "checks": [{"name":c["name"],"got":c["got"],"expected":c["expected"],
                    "tolerance":c["tolerance"],"abs_difference":c["abs_difference"],
                    "status":c["status"]} for c in CHECKS],
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


# ---------------------------------------------------------------------------
# Exploration-simulation payload checks (threshold_curve, learning_curve_data,
# tree_structure). Each compares the handler's field against a reference the
# validation script computes itself from an independently reproduced model.
# ---------------------------------------------------------------------------

def binary_split(df, feat, target, test_size=0.2, random_state=42, scale=False):
    """The split every leak-safe handler makes on an all-numeric, complete
    frame: stratified train_test_split, labels encoded alphabetically, and
    (scale=True) a StandardScaler fit on the training rows only."""
    import numpy as _np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    X = df[feat].values.astype(float); y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if scale:
        sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    le = LabelEncoder().fit(ytr)
    return Xtr, Xte, le.transform(ytr), le.transform(yte), [str(c) for c in le.classes_]


def threshold_curve_checks(prefix, tc, y_true01, scores, labels=None, auc_reported=None,
                           set_name='test', score_kind='probability', tol=1e-9):
    """threshold_curve == roc_curve(drop_intermediate=False) on the same rows
    and scores; counts; tpr*n_pos and fpr*n_neg integer-valued; AUC of the
    curve == the AUC the handler reports."""
    import numpy as _np
    from sklearn.metrics import roc_curve, auc
    p = f"{prefix}.threshold_curve"
    if not isinstance(tc, dict):
        chk(f"{p}.present", False, True); return
    y = _np.asarray(y_true01).astype(int); s = _np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    thr = thr.astype(float).copy()
    if not _np.isfinite(thr[0]):
        thr[0] = 1.0 if score_kind == 'probability' else float(thr[1:].max()) + 1.0
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    chk(f"{p}.n_pos", tc.get('n_pos'), n_pos)
    chk(f"{p}.n_neg", tc.get('n_neg'), n_neg)
    chk(f"{p}.set", tc.get('set'), set_name)
    chk(f"{p}.score_kind", tc.get('score_kind'), score_kind)
    if labels is not None:
        chk(f"{p}.positive_label", tc.get('positive_label'), str(labels[1]))
        chk(f"{p}.negative_label", tc.get('negative_label'), str(labels[0]))
    gf, gt, gth = (_np.asarray(tc.get(k), dtype=float) for k in ('fpr', 'tpr', 'thresholds'))
    if tc.get('downsampled'):
        # every kept threshold must be one of roc_curve's, with its fpr/tpr
        pos = {float(t): i for i, t in enumerate(thr)}
        idx = [pos.get(float(t), -1) for t in gth]
        chk(f"{p}.downsampled_points_on_curve", int(min(idx) >= 0), 1)
        idx = _np.asarray(idx)
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
    chk(f"{p}.n_points", len(gth), len(thr))
    if len(gth) == len(thr):
        chk(f"{p}.max|thresholds-ref|", float(_np.max(_np.abs(gth - thr))), 0.0, tol=tol)
        chk(f"{p}.max|fpr-ref|", float(_np.max(_np.abs(gf - fpr))), 0.0, tol=tol)
        chk(f"{p}.max|tpr-ref|", float(_np.max(_np.abs(gt - tpr))), 0.0, tol=tol)
    chk(f"{p}.thresholds_descending", bool(_np.all(_np.diff(gth) <= 0)), True)
    chk(f"{p}.tpr*n_pos_integer", float(_np.max(_np.abs(gt * n_pos - _np.round(gt * n_pos)))), 0.0, tol=1e-9)
    chk(f"{p}.fpr*n_neg_integer", float(_np.max(_np.abs(gf * n_neg - _np.round(gf * n_neg)))), 0.0, tol=1e-9)
    if auc_reported is not None:
        chk(f"{p}.auc==reported_auc", float(auc(gf, gt)), float(auc_reported), tol=1e-9)


def learning_curve_checks(prefix, lc, train_ref, valid_ref, best_ref, metric=None,
                          lower_is_better=None, valid_set='test', tol=1e-9):
    """learning_curve_data against the library's own per-iteration record
    (reproduced independently): length, last train / valid, best iteration
    (1-based), and every round when nothing was subsampled."""
    import numpy as _np
    p = f"{prefix}.learning_curve_data"
    if not isinstance(lc, dict):
        chk(f"{p}.present", False, True); return
    tr = _np.asarray(train_ref, dtype=float)
    chk(f"{p}.n_iterations", lc.get('n_iterations'), len(tr))
    chk(f"{p}.last_iteration", lc['iterations'][-1], len(tr))
    chk(f"{p}.last_train", lc['train'][-1], float(tr[-1]), tol=tol)
    if valid_ref is not None:
        va = _np.asarray(valid_ref, dtype=float)
        chk(f"{p}.last_valid", lc['valid'][-1], float(va[-1]), tol=tol)
    chk(f"{p}.best_iteration", str(lc.get('best_iteration')), str(best_ref))
    if metric is not None:
        chk(f"{p}.metric", lc.get('metric'), metric)
    if lower_is_better is not None:
        chk(f"{p}.lower_is_better", lc.get('lower_is_better'), bool(lower_is_better))
    chk(f"{p}.valid_set", lc.get('valid_set'), valid_set)
    if not lc.get('downsampled'):
        chk(f"{p}.max|train-ref|", float(_np.max(_np.abs(_np.asarray(lc['train'], dtype=float) - tr))), 0.0, tol=tol)
        if valid_ref is not None:
            chk(f"{p}.max|valid-ref|", float(_np.max(_np.abs(_np.asarray(lc['valid'], dtype=float) - va))), 0.0, tol=tol)


def tree_traverse(ts, x, feature_index):
    """Walk tree_structure for one row; returns the node position reached.
    scikit-learn's trees compare the row cast to float32 against the float64
    threshold, so the row is cast the same way here."""
    import numpy as _np
    x = _np.asarray(x, dtype=_np.float32)
    k = 0
    while ts['left'][k] != -1:
        j = feature_index[ts['feature'][k]]
        k = ts['left'][k] if x[j] <= ts['threshold'][k] else ts['right'][k]
    return k


def tree_structure_checks(prefix, ts, model, X, feature_names, task, class_names=None):
    """Traverse tree_structure for every row of X and compare the leaf with
    the reproduced model: classification -> argmax(leaf value) == predict and
    leaf value == predict_proba; regression -> leaf mean == predict."""
    import numpy as _np
    p = f"{prefix}.tree_structure"
    if not isinstance(ts, dict):
        chk(f"{p}.present", False, True); return
    fidx = {f: i for i, f in enumerate(feature_names)}
    chk(f"{p}.task", ts.get('task'), task)
    chk(f"{p}.n_nodes", len(ts['left']), int(min(model.tree_.node_count, 255)) if not ts.get('truncated') else len(ts['left']))
    chk(f"{p}.truncated", bool(ts.get('truncated')), model.tree_.node_count > 255)
    if ts.get('truncated'):
        chk(f"{p}.n_nodes<=255", len(ts['left']) <= 255, True)
        return
    leaves = _np.asarray([tree_traverse(ts, x, fidx) for x in X])
    if task == 'classification':
        vals = _np.asarray([ts['value'][k] for k in leaves], dtype=float)
        pred = model.predict(X); proba = model.predict_proba(X)
        chk(f"{p}.class_names", str(ts.get("class_names")), str([str(c) for c in class_names]))
        chk(f"{p}.argmax_leaf==predict(all {len(X)} rows)", int(_np.sum(_np.argmax(vals, axis=1) != pred)), 0)
        chk(f"{p}.max|leaf_value-predict_proba|", float(_np.max(_np.abs(vals - proba))), 0.0, tol=1e-12)
    else:
        vals = _np.asarray([ts['value'][k][0] for k in leaves], dtype=float)
        chk(f"{p}.max|leaf_mean-predict|", float(_np.max(_np.abs(vals - model.predict(X)))), 0.0, tol=1e-9)
    # the tree the handler shows is the one predict uses: same node ids by position
    chk(f"{p}.n_samples_root", ts['n_samples'][0], int(model.tree_.n_node_samples[0]))
