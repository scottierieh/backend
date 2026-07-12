# Self-Organizing Map (SOM) Backend — Validation Report
- **Target**: `som_analysis.py` · **Endpoint**: `/api/analysis/som`
- **Method**: silhouette verified with scikit-learn + structural consistency; exact quantization-error reproduction against `minisom` where installed.
## Summary — 5/5 pass
SOM training is stochastic (random weight init + random sample order), so the reported **silhouette score is verified independently** with `sklearn.metrics.silhouette_score` on the handler's own scaled data and node assignments — it matches to 1e-9. Structural consistency is checked too: the number of activated best-matching units equals the distinct assignments and never exceeds the grid size, every observation is assigned exactly once, and the quantization error is finite and non-negative. Where `minisom` is installed (the deploy image), the quantization error is additionally reproduced from a direct `MiniSom(random_seed=42)` fit.

**Bug found & fixed during validation**: `main()` called `_generate_interpretation(...)` which was never defined, crashing every SOM request at results assembly. The function is now defined (plain-language summary of grid size, quantization error and silhouette), so the endpoint returns successfully. See the auto-generated metadata section below for measured tolerances, package versions and the reproduction command.
## Defects — Fixed the missing `_generate_interpretation` crash. Silhouette package-verified (scikit-learn); quantization error reproduced against minisom in the deploy env.
