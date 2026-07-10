
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def haversine_distance_matrix(coords):
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat/2)**2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        dist[i] = 6371 * c
    return dist

def create_spatial_weights_from_coords(lat, lon, method='knn', k=5, threshold=50, alpha=1.0):
    coords = np.column_stack([lat, lon])
    n = len(coords)
    dist_matrix = haversine_distance_matrix(coords)
    W = np.zeros((n, n))

    if method == 'knn':
        for i in range(n):
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            W[i, neighbors] = 1
    elif method == 'threshold':
        W = (dist_matrix < threshold).astype(float)
        np.fill_diagonal(W, 0)
    elif method == 'distance':  # 명시적으로 체크
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] > 0:
                    W[i, j] = 1 / (dist_matrix[i, j] ** alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'knn', 'distance', or 'threshold'")

    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, np.newaxis]
    return W

def morans_i(y, W):
    n = len(y)
    y_mean = np.mean(y)
    y_dev = y - y_mean
    numerator = np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    if denominator == 0: return 0.0
    S0 = np.sum(W)
    if S0 == 0: return 0.0
    return (n / S0) * (numerator / denominator)

def generate_interpretation(results):
    lambda_ = results['coefficients'].get('lambda_', 0)
    moran_i = results['diagnostics'].get('morans_i_ols_residuals', 0)
    
    interpretation = "### SEM Model Interpretation\n\n"
    
    # Lambda Interpretation
    if abs(lambda_) > 0.1:
        direction = "positive" if lambda_ > 0 else "negative"
        strength = "strong" if abs(lambda_) > 0.5 else "moderate"
        interpretation += f"- The spatial error coefficient (λ) is **{lambda_:.4f}**, indicating a **{strength} {direction} spatial autocorrelation** in the residuals. This confirms that unobserved factors influencing the outcome are spatially clustered.\n"
    else:
        interpretation += f"- The spatial error coefficient (λ) of **{lambda_:.4f}** is weak, suggesting that spatial autocorrelation in the error term is minimal.\n"
        
    # Moran's I Interpretation
    if abs(moran_i) > 0.1:
        direction = "positive" if moran_i > 0 else "negative"
        interpretation += f"- Moran's I on the OLS residuals was **{moran_i:.4f}**, which initially suggested the presence of spatial autocorrelation and justified using a spatial error model.\n"
    else:
        interpretation += f"- Moran's I on the OLS residuals was low (**{moran_i:.4f}**), indicating that spatial error correlation was weak to begin with. The SEM results confirm this.\n"
        
    # Overall Model Fit
    interpretation += f"- The model's fit statistics are AIC (**{results['aic']:.2f}**) and BIC (**{results['bic']:.2f}**). These are useful for comparing against other models like OLS or SAR to determine the best fit.\n"

    # Conclusion
    interpretation += "\n**Conclusion:** The SEM model explicitly accounts for the spatial patterns in the unobserved variables. The value of lambda indicates how strongly the errors of neighboring locations are related. This leads to more reliable estimates of the coefficients (β) compared to a standard OLS regression."
    
    return interpretation

class SEMModel:
    def __init__(self):
        self.lambda_ = None
        self.beta = None
        self.sigma2 = None
        self.loglik = None
        self.n = None
        self.k = None
        self.converged = False

    def fit(self, y, X, W, method='ml', lambda_bounds=(-0.99, 0.99)):
        y = np.asarray(y).flatten()
        X = np.asarray(X)
        W = np.asarray(W) if not issparse(W) else W.toarray()
        
        self.n = len(y)
        self.k = X.shape[1]

        if method == 'ml':
            self._fit_ml(y, X, W, lambda_bounds)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return self

    def _fit_ml(self, y, X, W, lambda_bounds):
        def neg_loglik(lambda_):
            try:
                B = np.eye(self.n) - lambda_ * W
                log_det = np.log(np.linalg.det(B))
            except:
                return np.inf

            y_star = B @ y
            X_star = B @ X
            
            try:
                XtX_inv = np.linalg.inv(X_star.T @ X_star)
                beta = XtX_inv @ X_star.T @ y_star
            except np.linalg.LinAlgError:
                return np.inf

            e = y_star - X_star @ beta
            sigma2 = (e.T @ e) / self.n
            if sigma2 <= 0: return np.inf

            loglik = -self.n / 2 * np.log(2 * np.pi * sigma2) + log_det - (e.T @ e) / (2 * sigma2)
            return -loglik

        result = minimize(neg_loglik, x0=0.0, method='L-BFGS-B', bounds=[lambda_bounds])
        
        self.converged = result.success
        self.lambda_ = result.x[0]
        self.loglik = -result.fun
        
        B_opt = np.eye(self.n) - self.lambda_ * W
        y_star_opt = B_opt @ y
        X_star_opt = B_opt @ X
        self.beta = np.linalg.inv(X_star_opt.T @ X_star_opt) @ X_star_opt.T @ y_star_opt
        e_opt = y_star_opt - X_star_opt @ self.beta
        self.sigma2 = (e_opt.T @ e_opt) / self.n


def main():
    try:
        payload = json.load(sys.stdin)
        data = pd.DataFrame(payload.get('data'))
        y_col = payload.get('y_col')
        x_cols = payload.get('x_cols')
        lat_col = payload.get('lat_col')
        lon_col = payload.get('lon_col')
        
        w_method = payload.get('w_method', 'knn')
        k_neighbors = payload.get('k_neighbors', 5)
        distance_threshold = payload.get('distance_threshold', 50)
        
        if not all([y_col, x_cols, lat_col, lon_col]):
            raise ValueError("Missing required columns")
        
        y = data[y_col].values
        X = data[x_cols].values
        lat = data[lat_col].values
        lon = data[lon_col].values
        X = np.column_stack([np.ones(len(y)), X])
        
        W = create_spatial_weights_from_coords(lat, lon, method=w_method, k=k_neighbors, threshold=distance_threshold)
        
        # Diagnostics on OLS residuals
        ols_beta = np.linalg.inv(X.T @ X) @ X.T @ y
        ols_residuals = y - X @ ols_beta
        moran_i = morans_i(ols_residuals, W)

        model = SEMModel()
        model.fit(y, X, W)
        
        aic = -2 * model.loglik + 2 * (model.k + 2)
        bic = -2 * model.loglik + np.log(model.n) * (model.k + 2)
        
        coef_names = ['intercept'] + x_cols
        coefficients = {
            'lambda_': model.lambda_,
            **{name: val for name, val in zip(coef_names, model.beta)}
        }
        
        results = {
            "coefficients": coefficients,
            "sigma2": model.sigma2,
            "log_likelihood": model.loglik,
            "aic": aic,
            "bic": bic,
            "n_obs": model.n,
            "converged": model.converged,
            "diagnostics": {
                "morans_i_ols_residuals": moran_i,
                "spatial_weights_method": w_method,
                "k_neighbors": k_neighbors if w_method == 'knn' else None,
                "distance_threshold": distance_threshold if w_method == 'threshold' else None
            }
        }
        
        results['interpretation'] = generate_interpretation(results)

        print(json.dumps({'results': results}, default=_to_native_type))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
    
