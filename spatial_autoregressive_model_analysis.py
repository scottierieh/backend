#!/usr/bin/env python3
"""
완전히 작동하는 SAR (Spatial Autoregressive) Model 백엔드
이 파일을 spatial_autoregressive_model.py에 전체 복사하세요
"""

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
    """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def haversine_distance_matrix(coords):
    """
    위도/경도 좌표로부터 실제 거리(km) 행렬 계산
    
    Parameters:
    - coords: (n, 2) 배열, [latitude, longitude]
    
    Returns:
    - dist: (n, n) 거리 행렬 (km)
    """
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    
    n = len(coords)
    dist = np.zeros((n, n))
    
    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat/2)**2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        dist[i] = 6371 * c  # Earth radius in km
    
    return dist


def create_spatial_weights_from_coords(lat, lon, method='knn', k=5, threshold=50, alpha=1.0):
    """
    실제 위도/경도 좌표로부터 공간 가중 행렬 생성
    
    Parameters:
    - lat, lon: 위도/경도 배열
    - method: 'knn', 'distance', 'threshold'
    - k: KNN 방식에서 이웃 개수
    - threshold: 거리 임계값 (km)
    - alpha: 역거리 가중치의 지수
    
    Returns:
    - W: (n, n) row-normalized 공간 가중 행렬
    """
    coords = np.column_stack([lat, lon])
    n = len(coords)
    
    # 실제 거리 계산
    dist_matrix = haversine_distance_matrix(coords)
    
    W = np.zeros((n, n))
    
    if method == 'knn':
        # K-nearest neighbors
        for i in range(n):
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            W[i, neighbors] = 1
            
    elif method == 'distance':
        # 역거리 가중치
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] > 0:
                    W[i, j] = 1 / (dist_matrix[i, j] ** alpha)
                    
    elif method == 'threshold':
        # 임계값 이내의 모든 지점
        W = (dist_matrix < threshold).astype(float)
        np.fill_diagonal(W, 0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Row-normalize
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, np.newaxis]
    
    return W


def morans_i(y, W):
    """
    Global Moran's I 통계량 계산
    
    Returns:
    - I: Moran's I 값 (-1 ~ 1)
    """
    n = len(y)
    y_mean = np.mean(y)
    y_dev = y - y_mean
    
    numerator = np.sum(W * np.outer(y_dev, y_dev))
    denominator = np.sum(y_dev ** 2)
    
    if denominator == 0:
        return 0.0
    
    S0 = np.sum(W)
    if S0 == 0:
        return 0.0
    
    I = (n / S0) * (numerator / denominator)
    
    return I

def generate_interpretation(results):
    rho = results['coefficients'].get('rho_', 0)
    moran_i = results['diagnostics'].get('morans_i', 0)
    
    interpretation = "### SAR Model Interpretation\n\n"
    
    # Rho Interpretation
    if abs(rho) > 0.1:
        direction = "positive" if rho > 0 else "negative"
        strength = "strong" if abs(rho) > 0.5 else "moderate"
        interpretation += f"- The spatial autoregressive coefficient (ρ) is **{rho:.4f}**, indicating a **{strength} {direction} spatial dependence**. This means that the value of the dependent variable in one location is significantly influenced by the values in its neighboring locations.\n"
    else:
        interpretation += f"- The spatial autoregressive coefficient (ρ) of **{rho:.4f}** is close to zero, suggesting weak or no spatial dependence in the dependent variable itself.\n"
        
    # Moran's I Interpretation
    if abs(moran_i) > 0.1:
        direction = "positive" if moran_i > 0 else "negative"
        interpretation += f"- Moran's I of **{moran_i:.4f}** confirms the presence of **{direction} spatial autocorrelation** in the dependent variable, justifying the use of a spatial model over standard OLS.\n"
    else:
        interpretation += f"- Moran's I of **{moran_i:.4f}** is low, indicating little spatial autocorrelation. While the SAR model was run, a standard OLS regression might also be appropriate.\n"
        
    # Overall Model Fit
    interpretation += f"- The model's fit is evaluated by AIC (**{results['aic']:.2f}**) and BIC (**{results['bic']:.2f}**). These values are useful for comparing this model to others (like SEM or OLS); lower values generally indicate a better fit.\n"

    # Conclusion
    interpretation += "\n**Conclusion:** The results suggest that spatial effects are an important component of the model. Standard regression that ignores these spatial relationships could lead to biased conclusions."
    
    return interpretation

class SARModel:
    """Spatial Autoregressive (Lag) Model"""
    
    def __init__(self):
        self.rho = None
        self.beta = None
        self.sigma2 = None
        self.loglik = None
        self.n = None
        self.k = None
        self.converged = False

    def fit(self, y, X, W, method='ml', rho_bounds=(-0.99, 0.99)):
        """
        SAR 모델 추정
        
        Model: y = ρWy + Xβ + ε
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)
        W = np.asarray(W) if not issparse(W) else W.toarray()
        
        self.n = len(y)
        self.k = X.shape[1]
        
        if X.shape[0] != self.n:
            raise ValueError(f"X has {X.shape[0]} rows but y has {self.n}")
        if W.shape[0] != self.n or W.shape[1] != self.n:
            raise ValueError(f"W must be {self.n}x{self.n}")
        
        if method == 'ml':
            self._fit_ml(y, X, W, rho_bounds)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return self
    
    def _fit_ml(self, y, X, W, rho_bounds):
        """Maximum Likelihood 추정"""
        
        def neg_loglik(rho):
            """음의 로그우도 함수"""
            try:
                log_det = np.log(np.linalg.det(np.eye(self.n) - rho * W))
                if not np.isfinite(log_det):
                    return np.inf
            except:
                return np.inf
            
            y_star = y - rho * (W @ y)
            
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                beta = XtX_inv @ X.T @ y_star
            except np.linalg.LinAlgError:
                return np.inf
            
            e = y_star - X @ beta
            sigma2 = (e.T @ e) / self.n
            
            if sigma2 <= 0:
                return np.inf
            
            loglik = (-0.5 * self.n * np.log(2 * np.pi) 
                      - 0.5 * self.n * np.log(sigma2) 
                      + log_det 
                      - (e.T @ e) / (2 * sigma2))
            
            return -loglik
        
        result = minimize(
            neg_loglik, 
            x0=0.0, 
            method='L-BFGS-B', 
            bounds=[rho_bounds]
        )
        
        self.converged = result.success
        
        self.rho = result.x[0]
        y_star = y - self.rho * (W @ y)
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y_star
        e = y_star - (X @ self.beta)
        self.sigma2 = (e.T @ e) / self.n
        self.loglik = -result.fun


def main():
    try:
        # Input parsing
        payload = json.load(sys.stdin)
        data = pd.DataFrame(payload.get('data'))
        y_col = payload.get('y_col')
        x_cols = payload.get('x_cols')
        lat_col = payload.get('lat_col')
        lon_col = payload.get('lon_col')
        
        # 공간 가중 행렬 옵션
        w_method = payload.get('w_method', 'knn')
        k_neighbors = payload.get('k_neighbors', 5)
        distance_threshold = payload.get('distance_threshold', 50)
        
        # Validation
        if not all([y_col, x_cols, lat_col, lon_col]):
            raise ValueError("Missing required columns: y_col, x_cols, lat_col, lon_col")
        
        if y_col not in data.columns:
            raise ValueError(f"Column '{y_col}' not found in data")
        if lat_col not in data.columns:
            raise ValueError(f"Column '{lat_col}' not found in data")
        if lon_col not in data.columns:
            raise ValueError(f"Column '{lon_col}' not found in data")
        
        for col in x_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Extract data
        y = data[y_col].values
        X_df = data[x_cols]
        X = X_df.values
        lat = data[lat_col].values
        lon = data[lon_col].values
        
        # Add intercept
        X = np.column_stack([np.ones(len(y)), X])
        
        # Create spatial weights matrix
        W = create_spatial_weights_from_coords(
            lat, lon, 
            method=w_method, 
            k=k_neighbors, 
            threshold=distance_threshold
        )
        
        # Calculate Moran's I
        moran_i_value = morans_i(y, W)
        
        # Fit SAR model
        model = SARModel()
        model.fit(y, X, W)
        
        # Model diagnostics
        aic = -2 * model.loglik + 2 * (model.k + 2)
        bic = -2 * model.loglik + np.log(model.n) * (model.k + 2)
        
        # Prepare results
        coef_names = ['intercept'] + x_cols
        coefficients = {
            'rho_': model.rho,
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
                "morans_i": moran_i_value,
                "spatial_weights_method": w_method,
                "k_neighbors": k_neighbors if w_method == 'knn' else None,
                "distance_threshold": distance_threshold if w_method == 'threshold' else None
            }
        }
        
        results['interpretation'] = generate_interpretation(results)
        
        print(json.dumps({'results': results}, default=_to_native_type))
        
    except Exception as e:
        error_msg = str(e)
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
