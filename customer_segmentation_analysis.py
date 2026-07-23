import sys
import json
import io
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

CUSTOMER_DETAIL_CAP = 500


def _native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def quality_verdict(silhouette):
    if silhouette >= 0.5:
        return "well separated"
    if silhouette >= 0.25:
        return "moderately separated"
    return "poorly separated"


def name_cluster(centroid_z: pd.Series, top_n=2):
    """Rule-based heuristic label from a cluster's top |z-score| deviations."""
    ordered = centroid_z.reindex(centroid_z.abs().sort_values(ascending=False).index)
    parts = []
    for feat, z in ordered.head(top_n).items():
        level = "High" if z > 0 else "Low"
        clean = feat.replace("_", " ").title()
        parts.append(f"{level} {clean}")
    return " / ".join(parts) if parts else "Undifferentiated"


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get("data")
        feature_cols = payload.get("feature_cols")
        method = (payload.get("method") or "kmeans").lower()
        n_clusters = payload.get("n_clusters") or 4
        dbscan_eps = float(payload.get("dbscan_eps") or 0.8)
        dbscan_min_samples = int(payload.get("dbscan_min_samples") or 5)
        max_k = int(payload.get("max_k") or 10)

        if not data or not feature_cols or len(feature_cols) < 2:
            raise ValueError("Provide 'data' and at least 2 'feature_cols'")

        df = pd.DataFrame(data)
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        clean = df[feature_cols].dropna()
        if len(clean) < 10:
            raise ValueError("Need at least 10 complete rows across the selected features")

        n = len(clean)
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(clean), columns=feature_cols, index=clean.index)

        # ---- (4) Optimal K diagnostic: always run K-means across a K range ----
        k_hi = min(max_k, n - 1, 10)
        k_range = list(range(2, max(3, k_hi) + 1))
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
            inertias.append(float(km.inertia_))
            labels_k = km.labels_
            sil_scores.append(float(silhouette_score(scaled, labels_k)) if len(set(labels_k)) > 1 else -1.0)
        recommended_k = int(k_range[int(np.argmax(sil_scores))]) if sil_scores else 3

        # ---- Fit chosen method ----
        method_labels = {
            "kmeans": "K-Means",
            "hierarchical": "Hierarchical (Agglomerative)",
            "dbscan": "DBSCAN",
            "gmm": "Gaussian Mixture Model",
        }
        method_name = method_labels.get(method, method)

        if method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=int(n_clusters))
            labels = model.fit_predict(scaled)
        elif method == "dbscan":
            model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            labels = model.fit_predict(scaled)
        elif method == "gmm":
            model = GaussianMixture(n_components=int(n_clusters), random_state=42)
            labels = model.fit_predict(scaled)
        else:
            method = "kmeans"
            model = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=42)
            labels = model.fit_predict(scaled)

        clean = clean.copy()
        clean["_cluster_raw"] = labels

        # Remap labels to contiguous 0..k-1 ordered by size, DBSCAN noise (-1) kept last as "Noise"
        unique_labels = [l for l in pd.Series(labels).value_counts().index if l != -1]
        remap = {old: f"Cluster {i+1}" for i, old in enumerate(unique_labels)}
        if -1 in set(labels):
            remap[-1] = "Noise"
        clean["_cluster"] = clean["_cluster_raw"].map(remap)

        n_effective_clusters = len(unique_labels)
        if n_effective_clusters < 2:
            raise ValueError("Clustering produced fewer than 2 clusters; adjust parameters (e.g. DBSCAN eps/min_samples) and retry")

        # For metrics: exclude noise points for DBSCAN if present
        metric_mask = clean["_cluster_raw"] != -1
        metric_labels = clean.loc[metric_mask, "_cluster_raw"]
        metric_data = scaled.loc[metric_mask]

        sil = float(silhouette_score(metric_data, metric_labels))
        cal = float(calinski_harabasz_score(metric_data, metric_labels))
        dav = float(davies_bouldin_score(metric_data, metric_labels))

        # ---- (5) Cluster distribution ----
        dist_rows = []
        for lbl in [remap[o] for o in unique_labels]:
            cnt = int((clean["_cluster"] == lbl).sum())
            dist_rows.append({"cluster": lbl, "count": cnt, "pct": round(cnt / n * 100, 2)})
        noise_count = int((clean["_cluster"] == "Noise").sum()) if "Noise" in clean["_cluster"].values else 0
        if noise_count:
            dist_rows.append({"cluster": "Noise", "count": noise_count, "pct": round(noise_count / n * 100, 2)})

        largest = max(dist_rows, key=lambda r: r["count"] if r["cluster"] != "Noise" else -1)

        # ---- (6) Cluster profile: raw feature means per cluster ----
        cluster_order = [remap[o] for o in unique_labels]
        profile_rows = []
        overall_mean = clean[feature_cols].mean()
        overall_std = clean[feature_cols].std().replace(0, 1)
        z_profiles = {}
        for lbl in cluster_order:
            sub = clean[clean["_cluster"] == lbl]
            means = sub[feature_cols].mean()
            row = {"cluster": lbl, "size": int(len(sub))}
            for f in feature_cols:
                row[f] = float(means[f])
            profile_rows.append(row)
            z_profiles[lbl] = (means - overall_mean) / overall_std

        # ---- (7) Heatmap data: z-scores feature x cluster ----
        heatmap_matrix = {lbl: {f: float(z_profiles[lbl][f]) for f in feature_cols} for lbl in cluster_order}

        # ---- (8) PCA projection ----
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(scaled)
        pca_points = [
            {"pc1": float(pca_coords[i, 0]), "pc2": float(pca_coords[i, 1]), "cluster": clean["_cluster"].iloc[i]}
            for i in range(n)
        ]
        explained = [float(v) for v in pca.explained_variance_ratio_]

        # ---- (9) Separation quality ----
        separation = {
            "silhouette": sil,
            "calinski_harabasz": cal,
            "davies_bouldin": dav,
            "verdict": quality_verdict(sil),
        }

        # ---- (10) Comparison: top 5 features by overall variance of cluster means ----
        variance_of_means = pd.DataFrame(z_profiles).abs().mean(axis=1).sort_values(ascending=False)
        top_features = variance_of_means.head(min(5, len(feature_cols))).index.tolist()
        comparison_rows = []
        for lbl in cluster_order:
            row = {"cluster": lbl}
            for f in top_features:
                row[f] = float(clean[clean["_cluster"] == lbl][f].mean())
            comparison_rows.append(row)

        # ---- (11) Cluster naming (heuristic) ----
        naming = []
        for lbl in cluster_order:
            label = name_cluster(z_profiles[lbl])
            top2 = z_profiles[lbl].abs().sort_values(ascending=False).head(2).index.tolist()
            naming.append({
                "cluster": lbl,
                "name": label,
                "size": int((clean["_cluster"] == lbl).sum()),
                "pct": round(float((clean["_cluster"] == lbl).mean() * 100), 2),
                "distinguishing_features": top2,
            })

        # ---- (12) Customer detail ----
        # Distance to assigned cluster centroid (Euclidean, in standardized space)
        centroids_scaled = {}
        for lbl in cluster_order:
            mask = clean["_cluster"] == lbl
            centroids_scaled[lbl] = scaled.loc[clean.index[mask]].mean()

        id_col = None
        for cand in ["customer_id", "CustomerID", "id", "ID"]:
            if cand in df.columns:
                id_col = cand
                break

        detail_rows = []
        for idx in clean.index[:CUSTOMER_DETAIL_CAP]:
            lbl = clean.loc[idx, "_cluster"]
            if lbl == "Noise":
                dist = None
            else:
                diff = scaled.loc[idx] - centroids_scaled[lbl]
                dist = float(np.sqrt((diff ** 2).sum()))
            key_feats = {f: float(clean.loc[idx, f]) for f in feature_cols[:2]}
            detail_rows.append({
                "customer_id": str(df.loc[idx, id_col]) if id_col else str(idx),
                "cluster": lbl,
                "distance_to_centroid": dist,
                **key_feats,
            })
        truncated = n > CUSTOMER_DETAIL_CAP

        # ---- Charts ----
        charts = {}

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(k_range, inertias, "bo-")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("WCSS (inertia)")
        ax.set_title("Elbow Method")
        ax.grid(alpha=0.3)
        charts["elbow"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=k_range, y=sil_scores, color="skyblue", ax=ax)
        ax.axvline(k_range.index(recommended_k), color="red", linestyle="--", alpha=0.6)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title(f"Silhouette vs K (recommended k={recommended_k})")
        charts["silhouette_k"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        order_lbls = [r["cluster"] for r in dist_rows]
        sns.barplot(x=order_lbls, y=[r["count"] for r in dist_rows], ax=ax, palette="viridis")
        ax.set_ylabel("Customers")
        ax.set_title("Cluster Distribution")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        charts["distribution"] = _png(fig)

        fig, ax = plt.subplots(figsize=(1.2 * len(feature_cols) + 3, 1 * len(cluster_order) + 2))
        heat_df = pd.DataFrame(heatmap_matrix).T[feature_cols]
        sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Cluster Profile Heatmap (Z-scores)")
        charts["heatmap"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        clusters_arr = clean["_cluster"].values
        palette = sns.color_palette("husl", len(set(clusters_arr)))
        for i, lbl in enumerate(sorted(set(clusters_arr))):
            mask = clusters_arr == lbl
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], label=lbl, alpha=0.7, s=25, color=palette[i])
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.set_title("Customer Cluster Map (PCA)")
        ax.legend(fontsize=8)
        charts["pca_map"] = _png(fig)

        fig, ax = plt.subplots(figsize=(1.3 * len(top_features) + 3, 5))
        comp_df = pd.DataFrame(comparison_rows).set_index("cluster")
        comp_df.plot(kind="bar", ax=ax)
        ax.set_title("Cluster Comparison — Top Features")
        ax.set_ylabel("Mean value")
        plt.setp(ax.get_xticklabels(), rotation=0)
        ax.legend(fontsize=7)
        charts["comparison"] = _png(fig)

        results = {
            "overview": {
                "total_customers": n,
                "n_clusters": n_effective_clusters,
                "method": method_name,
                "method_key": method,
                "silhouette_score": sil,
                "n_features": len(feature_cols),
                "largest_cluster": largest["cluster"],
                "largest_cluster_pct": largest["pct"],
                "noise_count": noise_count,
            },
            "optimal_k": {
                "k_range": k_range,
                "inertias": inertias,
                "silhouette_scores": sil_scores,
                "recommended_k": recommended_k,
            },
            "distribution": dist_rows,
            "profile": profile_rows,
            "feature_cols": feature_cols,
            "heatmap": heatmap_matrix,
            "pca_points": pca_points[:CUSTOMER_DETAIL_CAP] if n > CUSTOMER_DETAIL_CAP else pca_points,
            "pca_explained_variance": explained,
            "separation": separation,
            "comparison": {"features": top_features, "rows": comparison_rows},
            "naming": naming,
            "customer_detail": detail_rows,
            "customer_detail_truncated": truncated,
            "customer_detail_total": n,
            "charts": charts,
        }

        print(json.dumps({"results": results, "plot": charts.get("pca_map")}, default=_native))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
