# Validates lstm_forecasting_analysis.py (Keras LSTM, recursive multi-step forecast).
#
# An LSTM fit has no closed-form reference, so the checks target everything that
# is deterministic and recomputable outside the handler:
#   - reproducibility: two runs on identical input return identical test
#     predictions and forecasts (the handler fixes its seed with
#     keras.utils.set_random_seed(42));
#   - data cleaning / ordering: rows arrive shuffled, plus one row with an
#     unparseable date and one with a missing value; the handler must drop both
#     and sort by date;
#   - windowing and split, rebuilt independently: (n - window) supervised
#     windows, the first int((n - window) * (1 - test_size)) for training, the
#     rest for testing; the returned test actuals must equal the raw series at
#     positions window + split .. n-1, value-by-value;
#   - test RMSE / MAE / R^2 recomputed from their formulas with numpy, using
#     the independently built test actuals and the handler's returned test
#     predictions (results['_validation']['y_pred_test']);
#   - shape / finiteness of test predictions and forecasts, the forecast horizon
#     and its dates (the monthly calendar continued from the last observation);
#   - training bookkeeping: 1 <= epochs_run <= epochs, finite non-negative loss.
# Naive (last value) and seasonal-naive (t-12) baselines are computed and
# printed for context only: the handler does not report a baseline, so there
# is no handler value to compare them with.
# The train-set metrics are not checked: the handler does not return the
# train-set predictions they are computed from.
#
# Needs TensorFlow (requirements.txt: tensorflow-cpu==2.17.0, Python <= 3.12).
# Where it is not installed this skips with zero checks.
import numpy as np, pandas as pd
try:
    import tensorflow  # noqa: F401
    HAS_TF = True
except Exception:
    HAS_TF = False
from _pyharness import run_script, chk, report

TITLE = "LSTM forecasting (Python)"

if not HAS_TF:
    print("SKIP | tensorflow not installed — run in the pinned environment (requirements.txt)")
    report(TITLE)
else:
    # ---- data: deterministic trend + seasonality, monthly, n = 120 ----
    n, W, H, TEST = 120, 12, 6, 0.2
    EPOCHS = 30
    dates = pd.date_range('2015-01-01', periods=n, freq='MS')
    t = np.arange(n)
    vals = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12)
    clean = pd.DataFrame({'ds': dates.strftime('%Y-%m-%d'), 'y': vals})
    # shuffled order + two rows the handler must drop
    shuffled = clean.sample(frac=1.0, random_state=0).reset_index(drop=True)
    junk = pd.DataFrame({'ds': ['not-a-date', '2031-01-01'], 'y': [5.0, None]})
    rows = pd.concat([shuffled, junk], ignore_index=True)
    payload = {'data': rows.to_dict('records'), 'date_col': 'ds', 'value_col': 'y',
               'window_size': W, 'forecast_periods': H, 'lstm_units': 16,
               'epochs': EPOCHS, 'batch_size': 8, 'test_size': TEST}

    r = run_script('lstm_forecasting_analysis.py', payload)['results']
    r2 = run_script('lstm_forecasting_analysis.py', payload)['results']

    pred = np.asarray(r['_validation']['y_pred_test'], dtype=float)
    act = np.asarray(r['_validation']['y_test_actual'], dtype=float)
    fc = np.asarray([f['forecast_value'] for f in r['forecast']], dtype=float)

    # ---- reproducibility ----
    pred2 = np.asarray(r2['_validation']['y_pred_test'], dtype=float)
    fc2 = np.asarray([f['forecast_value'] for f in r2['forecast']], dtype=float)
    chk("lstm.reproducible.test_pred_maxdiff.run1_vs_run2",
        float(np.max(np.abs(pred - pred2))) if pred.shape == pred2.shape else None, 0.0, tol=0.0)
    chk("lstm.reproducible.forecast_maxdiff.run1_vs_run2",
        float(np.max(np.abs(fc - fc2))) if fc.shape == fc2.shape else None, 0.0, tol=0.0)
    chk("lstm.reproducible.test_rmse.run1_vs_run2", r['metrics']['test']['rmse'], r2['metrics']['test']['rmse'], tol=0.0)

    # ---- windowing and split, rebuilt independently ----
    s = vals                                   # date-sorted series after cleaning
    n_win = len(s) - W
    split = int(n_win * (1 - TEST))
    chk("lstm.window_size", r['window_size'], W)
    chk("lstm.n_train_samples", r['n_train_samples'], split)
    chk("lstm.n_test_samples", r['n_test_samples'], n_win - split)
    y_test_ref = s[W + split:]                 # targets of the test windows
    chk("lstm.test_actual_len", len(act), len(y_test_ref))
    # MinMax scale -> inverse round trip leaves only float64 rounding (~1e-15 x range)
    for i in range(len(y_test_ref)):
        chk(f"lstm.test_actual[{i}]", float(act[i]), float(y_test_ref[i]), tol=1e-9)

    # ---- test metrics from their formulas ----
    e = y_test_ref - pred
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    r2s = float(1 - np.sum(e ** 2) / np.sum((y_test_ref - y_test_ref.mean()) ** 2))
    tm = r['metrics']['test']
    chk("lstm.test_rmse", tm['rmse'], rmse, tol=1e-9)
    chk("lstm.test_mae", tm['mae'], mae, tol=1e-9)
    chk("lstm.test_r2", tm['r2_score'], r2s, tol=1e-9)

    # ---- shapes / finiteness ----
    chk("lstm.test_pred_len", len(pred), n_win - split)
    chk("lstm.test_pred_finite", int(np.all(np.isfinite(pred))), 1)
    for k in ('rmse', 'mae', 'r2_score'):
        chk(f"lstm.train_{k}_finite", int(np.isfinite(r['metrics']['train'][k])), 1)

    # ---- forecast: horizon, values, dates ----
    chk("lstm.forecast_length", len(r['forecast']), H)
    chk("lstm.forecast_finite", int(np.all(np.isfinite(fc))), 1)
    fdates = pd.date_range(dates[-1], periods=H + 1, freq='MS')[1:]
    for i, d in enumerate(fdates):
        chk(f"lstm.forecast_date[{i}]", str(pd.Timestamp(r['forecast'][i]['forecast_date']).date()), str(d.date()))

    # ---- training bookkeeping ----
    chk("lstm.epochs_run_in_range", int(1 <= r['epochs_run'] <= EPOCHS), 1)
    chk("lstm.final_loss_finite_nonneg",
        int(np.isfinite(r['final_training_loss']) and r['final_training_loss'] >= 0), 1)

    # ---- baselines, for context only (not reported by the handler) ----
    idx = np.arange(W + split, len(s))
    naive = s[idx - 1]; snaive = s[idx - 12]
    print(f"LOG | test RMSE: LSTM {rmse:.4f} · naive(t-1) {np.sqrt(np.mean((y_test_ref - naive) ** 2)):.4f}"
          f" · seasonal naive(t-12) {np.sqrt(np.mean((y_test_ref - snaive) ** 2)):.4f}")

    report(TITLE)
