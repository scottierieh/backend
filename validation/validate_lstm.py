# Validates lstm_forecasting_analysis.py. LSTM training needs TensorFlow, which
# is absent in the lightweight validation sandbox, so this skips cleanly there
# and runs in the deploy/CI image (same pattern as CRAN-only R packages). The
# reported test RMSE/MAE are verified independently with scikit-learn on the
# handler's own inverse-scaled actuals/predictions (exposed under
# results['_validation']), plus structural forecast/consistency checks.
import numpy as np, pandas as pd
try:
    import tensorflow  # noqa: F401
    HAS_TF = True
except Exception:
    HAS_TF = False
from sklearn.metrics import mean_squared_error, mean_absolute_error
from _pyharness import run_script, chk, report

if not HAS_TF:
    print("SKIP | tensorflow not installed — LSTM validation runs in the deploy/CI env")
    report("LSTM forecasting (Python)")
else:
    n = 120
    dates = pd.date_range('2015-01-01', periods=n, freq='MS')
    t = np.arange(n)
    vals = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12)   # deterministic trend + seasonality
    df = pd.DataFrame({'ds': dates.astype(str), 'y': vals})
    payload = {'data': df.to_dict('records'), 'date_col': 'ds', 'value_col': 'y',
               'window_size': 12, 'forecast_periods': 6, 'lstm_units': 16,
               'epochs': 20, 'batch_size': 8, 'test_size': 0.2}
    res = run_script('lstm_forecasting_analysis.py', payload)
    r = res['results'] if 'results' in res else res
    v = r['_validation']
    y_test = np.asarray(v['y_test_actual']); y_pred = np.asarray(v['y_pred_test'])
    tm = r['metrics']['test']

    # reported metrics must equal scikit-learn's on the handler's own arrays
    chk("lstm.test_rmse", tm['rmse'], float(np.sqrt(mean_squared_error(y_test, y_pred))), tol=1e-6)
    chk("lstm.test_mae", tm['mae'], float(mean_absolute_error(y_test, y_pred)), tol=1e-6)
    # structural consistency
    chk("lstm.forecast_length", len(r['forecast']), 6)
    fvals = [f['forecast_value'] for f in r['forecast']]
    chk("lstm.forecast_finite", 1.0 if np.all(np.isfinite(fvals)) else 0.0, 1.0)
    chk("lstm.n_test_matches", len(y_test), r['n_test_samples'])
    report("LSTM forecasting (Python)")
