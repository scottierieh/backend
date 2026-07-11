# k-Medoids clustering (Python) — Validation Report
- **Target**: `kmedoids_analysis.py` · uses `sklearn_extra.cluster.KMedoids(method='pam', random_state=42)`
- **Status**: Package-based by construction (sklearn-extra). **Local validation deferred** — `scikit-learn-extra` is unmaintained and its compiled extension is **not importable under numpy 2.x** (ABI: "numpy.dtype size changed") in this sandbox. Runs where numpy < 2 / a compatible `scikit-learn-extra` build is present.
- ⚠️ **Deployment risk**: `requirements.txt` pins `scikit-learn-extra` but numpy is unpinned — if the image resolves numpy 2.x, the k-medoids endpoint will crash on import. Recommend pinning `numpy<2` (or replacing with a maintained k-medoids) and verifying `/api/analysis/kmedoids` in the deployed image.
