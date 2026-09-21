#updated regression code: 
"""
regression.py
-------------
Financial Market RAG System — Regression Module (Leak-Proof Control)

Responsibility: Train and evaluate logistic regression models on the same
four time windows used by the RAG pipeline (5yr, 10yr, 20yr, 50yr, all
ending 2015-12-31), predicting binary next-month S&P 500 direction from
lagged macroeconomic features.

This module is structurally leak-proof: unlike generation.py, there is no
pretrained language model involved, so a model trained only on data through
2015 has no possible access to anything after that date. This serves as
the leak-proof triangulation check against the RAG track's results.

Training methodology:
    - Hyperparameter search (C) via TimeSeriesSplit + GridSearchCV, done
      entirely within each window's own training data — the 2016-2024
      eval set is never touched until the final model is already chosen.
    - Feature values are clipped post-scaling to prevent any single
      out-of-distribution input from saturating the sigmoid — this
      directly targets the collapse mechanism diagnosed during
      development (narrow windows produced eval-set z-scores as extreme
      as 70, versus ~2 for wider windows), rather than relying on
      regularization alone, which was found not to fix it.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("regression")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF = "2015-12-31"
EVAL_START = "2016-01-01"
EVAL_END = "2024-12-31"

WINDOW_YEARS = {"5yr": 5, "10yr": 10, "20yr": 20, "50yr": 50}

SOURCE_COLUMN_MAP = {
    "UNRATE": "unemployment",
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "GDPC1": "real_gdp",
    "T10Y2YM": "yield_spread",
    "S&P500": "sp500_price",
}

FEATURE_COLUMNS = [
    "unemployment_lag1",
    "fed_funds_rate_lag1",
    "cpi_lag1",
    "yield_spread_lag1",
    "sp500_return_lag1",
]

C_GRID = [0.01, 0.1, 1.0, 10.0]
CLIP_THRESHOLD = 4.0  # max allowed |z-score| after scaling


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_merge_raw_data(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load each relevant raw CSV, rename its value column, resample to
    monthly frequency, and merge into one wide dataframe indexed by
    month-end date.
    """
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    merged: Optional[pd.DataFrame] = None

    for stem, col_name in SOURCE_COLUMN_MAP.items():
        filepath = raw_dir / f"{stem}.csv"
        if not filepath.exists():
            logger.warning("Expected file not found, skipping: %s", filepath)
            continue

        df = pd.read_csv(filepath)
        date_col = next(
            (c for c in df.columns if c.lower() in
             ("date", "observation_date", "datetime")),
            None
        )
        if date_col is None:
            logger.warning("No date column found in %s, skipping.", filepath.name)
            continue

        value_col = [c for c in df.columns if c != date_col][0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])

        df = df[[date_col, value_col]].rename(
            columns={date_col: "date", value_col: col_name}
        )
        df = df.set_index("date").resample("MS").mean()

        merged = df if merged is None else merged.join(df, how="outer")
        logger.info("Loaded and merged %s -> column '%s' (%d rows)",
                    stem, col_name, len(df))

    if merged is None:
        raise FileNotFoundError(
            "No source files were loaded. Check SOURCE_COLUMN_MAP against "
            "your actual data/raw/ filenames."
        )

    if "real_gdp" in merged.columns:
        merged["real_gdp"] = merged["real_gdp"].ffill()

    merged = merged.sort_index()
    logger.info(
        "Merged dataframe: %d rows, %s -> %s",
        len(merged), merged.index.min().date(), merged.index.max().date()
    )
    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lagged features (1-month lag, to prevent look-ahead bias) and
    the binary target: 1 if S&P 500 price in month t+1 > month t, else 0.
    """
    df = df.copy()
    df["sp500_return"] = df["sp500_price"].pct_change()

    feature_cols = {
        "unemployment": "unemployment_lag1",
        "fed_funds_rate": "fed_funds_rate_lag1",
        "cpi": "cpi_lag1",
        "yield_spread": "yield_spread_lag1",
        "sp500_return": "sp500_return_lag1",
    }

    for source_col, lagged_col in feature_cols.items():
        if source_col in df.columns:
            df[lagged_col] = df[source_col].shift(1)
        else:
            logger.warning("Expected column '%s' missing from merged data.", source_col)

    df["target"] = (df["sp500_price"].shift(-1) > df["sp500_price"]).astype(int)

    required = list(feature_cols.values()) + ["target"]
    before = len(df)
    df = df.dropna(subset=required)
    logger.info("Feature engineering: %d -> %d rows after dropping NaNs", before, len(df))

    return df


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def get_window_slice(df: pd.DataFrame, window_name: str) -> pd.DataFrame:
    if window_name not in WINDOW_YEARS:
        raise ValueError(f"Unknown window '{window_name}'. Must be one of {list(WINDOW_YEARS)}.")

    cutoff_ts = pd.Timestamp(CUTOFF)
    start_ts = cutoff_ts - pd.DateOffset(years=WINDOW_YEARS[window_name])
    return df[(df.index >= start_ts) & (df.index <= cutoff_ts)]


def get_eval_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(EVAL_START)) & (df.index <= pd.Timestamp(EVAL_END))]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_class_balance(df: pd.DataFrame, label: str) -> None:
    counts = df["target"].value_counts(normalize=True).sort_index()
    up_pct = counts.get(1, 0.0) * 100
    down_pct = counts.get(0, 0.0) * 100
    logger.info(
        "%s class balance: %.1f%% up-months, %.1f%% down-months (n=%d)",
        label, up_pct, down_pct, len(df)
    )


def check_multicollinearity(df: pd.DataFrame, label: str, threshold: float = 0.8) -> None:
    """
    Flag any feature pair with |correlation| above threshold. On a small
    dataset, high collinearity makes coefficients unstable — worth
    reporting explicitly as a limitation rather than silently ignoring.
    """
    corr = df[FEATURE_COLUMNS].corr()
    flagged = []
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(i + 1, len(FEATURE_COLUMNS)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                flagged.append((FEATURE_COLUMNS[i], FEATURE_COLUMNS[j], val))
    if flagged:
        for f1, f2, val in flagged:
            logger.warning("[%s] High collinearity: %s <-> %s (r=%.2f)", label, f1, f2, val)
    else:
        logger.info("[%s] No feature pairs exceed collinearity threshold %.2f", label, threshold)


def naive_baseline_accuracy(eval_df: pd.DataFrame) -> float:
    majority_class = eval_df["target"].mode()[0]
    return (eval_df["target"] == majority_class).mean()


def clip_extreme_values(X_scaled: np.ndarray, clip_at: float = CLIP_THRESHOLD) -> np.ndarray:
    """
    Clip z-scores to a maximum absolute value. Directly targets the
    diagnosed collapse mechanism: a single feature value far outside the
    training distribution (e.g. z-score of 70 for the 5yr window's
    unemployment reading) dominates the sigmoid regardless of coefficient
    size, which is why regularization alone was found not to fix it.
    """
    return np.clip(X_scaled, -clip_at, clip_at)


# ---------------------------------------------------------------------------
# Model training with proper hyperparameter search
# ---------------------------------------------------------------------------

