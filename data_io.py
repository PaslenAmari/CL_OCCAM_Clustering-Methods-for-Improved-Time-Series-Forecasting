"""
Dataset loaders for the CL-Occam clustered-local forecasting pipeline.

Two real-world daily panels used in the main experiment:
    ELEC     --- Worldwide Electricity Load (Mendeley ybggkc58fz)
    ROSSMANN --- Kaggle Rossmann Store Sales (50 German retail stores)

Each loader returns a tuple (X, names, start_date_str) where:
    X        ndarray of shape (n_series, n_timesteps), float64
    names    list of length n_series with human-readable identifiers
    start_date_str  ISO calendar date of the first column of X
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ELEC -- Worldwide Electricity Load (Mendeley)
# ---------------------------------------------------------------------------

# Resolve dataset locations. In a local dev environment data/ sits next
# to the script; inside Docker (/app) it is a child directory.
_HERE = os.path.abspath(os.path.dirname(__file__))
_PARENT = os.path.abspath(os.path.join(_HERE, os.pardir))

if os.path.exists(os.path.join(_PARENT, "data")):
    _PROJECT_ROOT = _PARENT
else:
    _PROJECT_ROOT = _HERE

def _project_path(*parts: str) -> str:
    return os.path.join(_PROJECT_ROOT, "data", *parts)


ELEC_BASE_PATH = _project_path("elec", "Worldwide Electricity Load Dataset", "GloElecLoad")

ELEC_WINDOW_START = pd.Timestamp("2022-01-01")
ELEC_WINDOW_END = pd.Timestamp("2022-12-31")
ELEC_COVERAGE_THRESHOLD = 0.70
ELEC_MIN_OBS = 200


def _detect_datetime_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        c = str(col).lower()
        if any(tok in c for tok in ("time", "date", "datetime", "timestamp")):
            return col
    return df.columns[0]


def _detect_target_col(df: pd.DataFrame, dt_col: str) -> Optional[str]:
    for col in df.columns:
        if col == dt_col:
            continue
        c = str(col).lower()
        if any(tok in c for tok in ("load", "mw", "demand", "value", "consumption")):
            return col
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c != dt_col]
    return numeric[0] if numeric else None


def _parse_elec_csv(path: str) -> Optional[pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    if df.empty or df.shape[1] < 2:
        return None
    dt_col = _detect_datetime_col(df)
    target_col = _detect_target_col(df, dt_col)
    if target_col is None:
        return None
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.dropna(subset=[dt_col, target_col])
    if df.empty:
        return None
    df = df[[dt_col, target_col]].set_index(dt_col).sort_index()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    daily = df[target_col].resample("D").mean()
    daily = daily.where(daily > 0, np.nan)
    if daily.notna().sum() < 100:
        return None
    return daily


def load_elec(base_path: str = ELEC_BASE_PATH) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    """Load and align the Mendeley Worldwide Electricity Load panel."""
    if not os.path.exists(base_path):
        return None, None, None

    all_series: dict[str, pd.Series] = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".csv") or "Additional_Information" in file or file.startswith("."):
                continue
            path = os.path.join(root, file)
            region = os.path.basename(root)
            if region == "GloElecLoad":
                region = file.replace(".csv", "")
            try:
                series = _parse_elec_csv(path)
                if series is None:
                    continue
                key = region.strip()
                if key in all_series:
                    key = f"{key}_{file.replace('.csv', '')}"
                all_series[key] = series
            except Exception:
                continue

    if not all_series:
        return None, None, None

    combined = pd.DataFrame(all_series)
    idx = pd.date_range(ELEC_WINDOW_START, ELEC_WINDOW_END, freq="D")
    combined = combined.reindex(idx)

    coverage = combined.notna().mean()
    valid_cols = coverage[coverage >= ELEC_COVERAGE_THRESHOLD].index.tolist()
    if len(valid_cols) < 8:
        valid_cols = coverage.sort_values(ascending=False).head(min(30, len(coverage))).index.tolist()

    combined = combined[valid_cols]
    combined = combined.loc[:, combined.notna().sum() >= ELEC_MIN_OBS]
    combined = combined.interpolate(method="linear", limit_direction="both").ffill().bfill()
    combined = combined.loc[:, combined.std(axis=0) > 1e-8]

    if combined.shape[1] < 4:
        return None, None, None

    X = combined.T.values.astype(float)
    return X, combined.columns.tolist(), ELEC_WINDOW_START.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# ROSSMANN -- Kaggle Rossmann Store Sales (third panel, daily)
# ---------------------------------------------------------------------------


# Kaggle Rossmann Store Sales: 1,115 stores, daily Sales 2013-01-01
# to 2015-07-31 (942 days for the well-covered subset).
# Zero-sales days correspond to closed stores (Open=0) and are retained
# because they are predictable from weekday and holiday signals --
# the per-segment heterogeneity (store types, assortments) is precisely
# what DTW clustering should identify.
ROSSMANN_LOCAL_DIR = _project_path("rossman")
ROSSMANN_TRAIN_FILE = "train.csv"
ROSSMANN_FULL_LENGTH = 942
ROSSMANN_TARGET_N = 50
ROSSMANN_SEED = 42


def load_rossmann(local_dir: str = ROSSMANN_LOCAL_DIR) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[str]]:
    """Load a balanced subset of the Rossmann daily sales panel."""
    train_path = os.path.join(local_dir, ROSSMANN_TRAIN_FILE)
    if not os.path.exists(train_path):
        print(f"  [!] Rossmann train.csv missing at {train_path}")
        return None, None, None
    df = pd.read_csv(
        train_path,
        usecols=["Store", "Date", "Sales"],
        parse_dates=["Date"],
        dtype={"Store": "int32", "Sales": "int64"},
    )
    df = df.sort_values(["Store", "Date"])
    counts = df.groupby("Store").size()
    full_stores = counts[counts == ROSSMANN_FULL_LENGTH].index.tolist()
    if len(full_stores) < 4:
        return None, None, None

    rng = np.random.default_rng(ROSSMANN_SEED)
    if len(full_stores) > ROSSMANN_TARGET_N:
        chosen = sorted(rng.choice(full_stores, size=ROSSMANN_TARGET_N, replace=False).tolist())
    else:
        chosen = sorted(full_stores)

    sub = df[df["Store"].isin(chosen)]
    pivot = sub.pivot(index="Date", columns="Store", values="Sales")
    pivot = pivot.sort_index()
    pivot = pivot.reindex(columns=chosen)
    pivot = pivot.interpolate(method="linear", limit_direction="both").ffill().bfill()
    matrix = pivot.T.values.astype(float)

    stds = matrix.std(axis=1)
    keep = stds > 1e-8
    matrix = matrix[keep]
    kept_names = [str(s) for s, k in zip(chosen, keep) if k]

    common_start = pivot.index.min().strftime("%Y-%m-%d")
    return matrix, kept_names, common_start


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "ELEC": load_elec,
    "ROSSMANN": lambda: load_rossmann(),
}


def load_dataset(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]()