"""
Integration smoke test: Worldwide Electricity Load + signatures + DTW + clustering
with the min_size=3 contract.

Run this once before the full pipeline to verify the environment is
healthy. Requires only the lightweight dependency stack
(numpy + pandas + scipy + scikit-learn + dtaidistance);
ETNA / CatBoost are not exercised here.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def main() -> int:
    from data_io import load_worldwide_electricity  # замени на реальное имя функции
    from signatures import SIGNATURE_REGISTRY
    from clustering import cluster_with_min_size, dtw_distance_matrix, zscore_rows

    print("[1/4] Loading Worldwide Electricity dataset ...")
    X, names, start_date = load_worldwide_electricity(local_dir="../data/worldwide_electricity")
    assert X is not None, "Worldwide Electricity loader returned None"
    assert X.ndim == 2, f"Expected 2D array, got shape {X.shape}"
    assert X.shape[0] >= 3, f"Need at least 3 series for clustering, got {X.shape[0]}"
    assert int(np.isfinite(X).sum()) == X.size, "Dataset contains NaNs after impute"
    print(f"      shape={X.shape}, start={start_date}, series={len(names)}")

    print("[2/4] Computing all signatures ...")
    for name, fn in SIGNATURE_REGISTRY.items():
        S = fn(X)
        assert np.isfinite(S).all(), f"signature {name} produced NaNs"
        print(f"      {name}: {S.shape}")

    print("[3/4] DTW distance matrix ...")
    t = time.time()
    D = dtw_distance_matrix(zscore_rows(X))
    elapsed = time.time() - t
    n = X.shape[0]
    assert D.shape == (n, n), f"Expected ({n},{n}), got {D.shape}"
    assert (D == D.T).all(), "DTW matrix not symmetric"
    assert (D.diagonal() == 0).all(), "DTW diagonal not zero"
    print(f"      shape={D.shape}, mean={D.mean():.3f}, time={elapsed:.1f}s")

    print("[4/4] Clustering at k in {3,4,5} with min_size=3 ...")
    max_k = min(6, n - 1)
    for k in [k for k in (3, 4, 5) if k < n]:
        labels = cluster_with_min_size(D, n_clusters=k, min_size=3, seed=42)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        assert min(sizes) >= 3, f"min_size=3 violated at k={k}: sizes={sizes}"
        assert len(sizes) <= k, f"got {len(sizes)} clusters when {k} requested"
        print(f"      k_req={k} -> k_actual={len(sizes)}, sizes={sizes}")

    print("\n[OK] Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
