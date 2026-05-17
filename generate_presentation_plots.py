import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style for presentation
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (16, 10),
    "figure.dpi": 150,
    "font.size": 12,
    "lines.linewidth": 1.5,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Mocking ETNA imports if running locally without it, 
# but this script is intended to be run inside the Docker environment.
try:
    from etna.datasets import TSDataset
    from etna.pipeline import Pipeline
    from etna.models import CatBoostMultiSegmentModel, SklearnMultiSegmentModel
    from etna.transforms import (
        LagTransform, DateFlagsTransform, StandardScalerTransform, 
        TimeSeriesImputerTransform, SegmentEncoderTransform
    )
    ETNA_AVAILABLE = True
except ImportError:
    ETNA_AVAILABLE = False
    print("[!] ETNA not found. This script should be run in the Docker environment.")

# --- ELEC Logic (simplified from elec.py) ---

def load_elec_data():
    from data_io import load_elec
    X, names, start_date = load_elec()
    return X, names, start_date

def run_elec_forecast(X, start_date, horizon=30):
    from forecast import forecast_global, forecast_clustered_local, cluster_labels_dtw
    
    # Train end
    train_end = X.shape[1] - horizon
    X_train = X[:, :train_end]
    X_actual = X[:, train_end:]
    
    # Increase pool to find best examples
    pool_size = min(20, X.shape[0])
    X_train_pool = X_train[:pool_size]
    
    # Global Forecast (on the pool)
    global_f = forecast_global("catboost", X_train_pool, start_date, horizon)
    
    # Ensemble Forecast (with real clustering for the pool)
    labels, mapping = cluster_labels_dtw(X_train_pool, start_date, n_clusters=3)
    ensemble_f = forecast_clustered_local(X_train_pool, labels, mapping, start_date, horizon)
    
    actual_dict = {f"region_{i}": X_actual[i] for i in range(pool_size)}
    
    # Calculate improvements
    improvements = []
    for i in range(pool_size):
        reg = f"region_{i}"
        act = actual_dict[reg]
        glob = global_f.get(reg, np.zeros_like(act))
        ens = ensemble_f.get(reg, glob)
        m_glob = np.mean(np.abs(act - glob))
        m_ens = np.mean(np.abs(act - ens))
        improvements.append((reg, m_glob - m_ens))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    best_regions = [x[0] for x in improvements[:4]]
    
    return actual_dict, global_f, ensemble_f, best_regions

# --- Rossmann Logic (simplified from rossmann_exog.py) ---

def load_rossmann_data():
    from rossmann_exog import _load_with_exog
    target_long, exog_long, stores = _load_with_exog()
    return target_long, exog_long, stores

def run_rossmann_forecast(target_long, exog_long, horizon=30):
    from rossmann_exog import _build_tsdataset, _forecast_method, _forecast_cl_occam_with_exog
    
    # Pool size for selection
    all_segs = target_long["segment"].unique()
    pool_size = min(30, len(all_segs))
    segments = all_segs[:pool_size]
    
    target_pool = target_long[target_long["segment"].isin(segments)].copy()
    exog_pool = exog_long[exog_long["segment"].isin(segments)].copy()
    
    timestamps = sorted(target_pool["timestamp"].unique())
    test_start = len(timestamps) - horizon
    train_ts_dates = timestamps[:test_start]
    test_ts_dates = timestamps[test_start:]
    
    train_target = target_pool[target_pool["timestamp"].isin(train_ts_dates)].copy()
    train_exog = exog_pool[exog_pool["timestamp"].isin(timestamps)].copy()
    tsd_train = _build_tsdataset(train_target, train_exog)
    
    test_target = target_pool[target_pool["timestamp"].isin(test_ts_dates)].copy()
    
    # Global
    global_f = _forecast_method("catboost", tsd_train, horizon)
    
    # Ensemble (CL-Occam)
    ensemble_f = _forecast_cl_occam_with_exog(train_target, train_exog, horizon)
    
    # Use original segment names (store IDs) for Rossmann
    actual_dict = {}
    improvements = []
    for seg in segments:
        reg = str(seg)
        act = test_target[test_target["segment"] == seg].sort_values("timestamp")["target"].values
        actual_dict[reg] = act
        
        glob = global_f.get(reg, np.zeros_like(act))
        ens = ensemble_f.get(reg, glob)
        m_glob = np.mean(np.abs(act - glob))
        m_ens = np.mean(np.abs(act - ens))
        improvements.append((reg, m_glob - m_ens))
        
    improvements.sort(key=lambda x: x[1], reverse=True)
    best_stores = [x[0] for x in improvements[:4]]
    
    return actual_dict, global_f, ensemble_f, best_stores

# --- Plotting logic ---

def plot_dataset_results(actuals, global_f, ensemble_f, dataset_name, output_path, regions=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{dataset_name} Comparison", fontsize=20, fontweight='bold')
    
    # Select 4 regions to plot
    if regions is None:
        regions = sorted(list(actuals.keys()))[:4]
    
    for i, reg in enumerate(regions):
        ax = axes[i // 2, i % 2]
        
        act = actuals[reg]
        # Robust lookup: try string and int keys if necessary
        glob = global_f.get(reg)
        if glob is None:
             glob = global_f.get(str(reg))
        if glob is None:
             glob = np.zeros_like(act)
             
        ens = ensemble_f.get(reg)
        if ens is None:
            ens = ensemble_f.get(str(reg))
        if ens is None:
            ens = glob
        
        # Calculate MAEs
        mae_glob = np.mean(np.abs(act - glob))
        mae_ens = np.mean(np.abs(act - ens))
        
        ax.plot(act, label="Actual", color="black", linewidth=2)
        ax.plot(glob, label=f"Global (MAE {mae_glob:.1f})", color="#3b6e96", linestyle="--")
        ax.plot(ens, label=f"Ensemble (MAE {mae_ens:.1f})", color="#c97f4a", linestyle="-.")
        
        # Use "Store" label for Rossmann, "Region" for others
        label_type = "Store" if dataset_name == "Rossmann" else "Region"
        ax.set_title(f"{label_type}: {reg}")
        ax.legend(fontsize=10)
        ax.set_ylabel("Value")
        ax.set_xlabel("Horizon (days)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

def main():
    if not ETNA_AVAILABLE:
        return
        
    os.makedirs("results/presentation", exist_ok=True)
    
    # 1. ELEC
    print("Generating ELEC plots...")
    try:
        X, names, start_date = load_elec_data()
        actual_dict, global_f, ensemble_f, best_regs = run_elec_forecast(X, start_date)
        plot_dataset_results(actual_dict, global_f, ensemble_f, "Worldwide ELEC", "results/presentation/elec_comparison.png", regions=best_regs)
    except Exception as e:
        print(f"ELEC plotting failed: {e}")

    # 2. Rossmann
    print("Generating Rossmann plots...")
    try:
        target_long, exog_long, stores = load_rossmann_data()
        actual_dict, global_f, ensemble_f, best_stores = run_rossmann_forecast(target_long, exog_long)
        plot_dataset_results(actual_dict, global_f, ensemble_f, "Rossmann", "results/presentation/rossmann_comparison.png", regions=best_stores)
    except Exception as e:
        print(f"Rossmann plotting failed: {e}")

if __name__ == "__main__":
    main()
