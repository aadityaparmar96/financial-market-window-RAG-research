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

This module does NOT:
    - Use any text chunks or embeddings
    - Call any LLM
    - Share code with data_process.py — regression needs numeric columns,
      not natural-language sentences, so data loading is reimplemented
      here specifically for this purpose.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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

# Maps each raw CSV's filename stem to the column name it should be
# renamed to in the merged dataframe. Adjust this if your raw filenames
# differ — this is the one place that needs to match your data/raw/ folder.
SOURCE_COLUMN_MAP = {
    "UNRATE": "unemployment",
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "GDPC1": "real_gdp",
    "T10Y2YM": "yield_spread",
    "S&P500": "sp500_price",
}

# ---------------------------------------------------------------------------
# Data loading — separate from data_process.py by design
# ---------------------------------------------------------------------------
"""
    Load each relevant raw CSV, rename its value column per SOURCE_COLUMN_MAP,
    resample to monthly frequency, and merge everything into one wide
    dataframe indexed by month-end date.

    Unlike data_process.py's per-row document approach, regression needs
    one row per month with all indicators as columns, this is the
    classic "wide" format for a feature matrix.
    """
def load_and_merge_raw_data(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    
    if raw_dir is None:
        # Reuses the same project-root detection logic conceptually,
        # but kept local to this file to avoid a hard dependency on
        # data_process.py's internals.
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
        df = df.set_index("date").resample("MS").mean()  # monthly, start-of-month

        merged = df if merged is None else merged.join(df, how="outer")
        logger.info("Loaded and merged %s -> column '%s' (%d rows)",
                    stem, col_name, len(df))

    if merged is None:
        raise FileNotFoundError(
            "No source files were loaded. Check SOURCE_COLUMN_MAP against "
            "your actual data/raw/ filenames."
        )

    # GDP is quarterly natively; forward-fill to monthly rather than
    # interpolate here, since regression features should reflect only
    # information that would genuinely have been known at that point in
    # time — interpolation (used in the RAG text pipeline) can leak
    # slightly forward-looking smoothed values, which matters more for a
    # model that is directly fitting on these numbers.
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
    Build lagged features and the binary target variable.

    All features are lagged by 1 month to prevent look-ahead bias — a
    prediction made "as of" month t can only use information that was
    actually available at the end of month t, not month t's own outcome.

    Target: 1 if S&P 500 price in month t+1 > month t, else 0.
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

    # Target: did price go UP the following month?
    df["target"] = (df["sp500_price"].shift(-1) > df["sp500_price"]).astype(int)

    required = list(feature_cols.values()) + ["target"]
    before = len(df)
    df = df.dropna(subset=required)
    logger.info("Feature engineering: %d -> %d rows after dropping NaNs", before, len(df))

    return df

# ---------------------------------------------------------------------------
# Windowing — matches the RAG track's four windows exactly
# ---------------------------------------------------------------------------

def get_window_slice(df: pd.DataFrame, window_name: str) -> pd.DataFrame:
    """
    Slice the full feature/target dataframe down to one window's training
    range, using the same CUTOFF and WINDOW_YEARS as embeddings.py, so the
    regression track and RAG track are trained on identical time spans.
    """
    if window_name not in WINDOW_YEARS:
        raise ValueError(f"Unknown window '{window_name}'. Must be one of {list(WINDOW_YEARS)}.")

    cutoff_ts = pd.Timestamp(CUTOFF)
    start_ts = cutoff_ts - pd.DateOffset(years=WINDOW_YEARS[window_name])

    sliced = df[(df.index >= start_ts) & (df.index <= cutoff_ts)]
    return sliced


def get_eval_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    The fixed 2016-2024 holdout — identical across all four windows,
    exactly mirroring how every RAG window is asked the same eval
    questions regardless of its own training range.
    """
    return df[(df.index >= pd.Timestamp(EVAL_START)) & (df.index <= pd.Timestamp(EVAL_END))]


# ---------------------------------------------------------------------------
# Class balance check
# ---------------------------------------------------------------------------

def check_class_balance(df: pd.DataFrame, label: str) -> None:
    """
    Log the target class distribution. S&P 500 monthly returns are
    historically skewed toward positive months — if a window's training
    data is meaningfully imbalanced, class_weight='balanced' (already set
    in train_model()) becomes not just a nice-to-have but a requirement,
    otherwise the model can achieve deceptively high accuracy by always
    predicting the majority class.
    """
    counts = df["target"].value_counts(normalize=True).sort_index()
    up_pct = counts.get(1, 0.0) * 100
    down_pct = counts.get(0, 0.0) * 100
    logger.info(
        "%s class balance: %.1f%% up-months, %.1f%% down-months (n=%d)",
        label, up_pct, down_pct, len(df)
    )


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "unemployment_lag1",
    "fed_funds_rate_lag1",
    "cpi_lag1",
    "yield_spread_lag1",
    "sp500_return_lag1",
]


def train_and_evaluate_window(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    window_name: str,
) -> dict:
    """
    Fit a logistic regression on train_df, evaluate on eval_df (the fixed
    2016-2024 holdout), and return a results dict.

    class_weight='balanced' is set unconditionally here, not just when
    imbalance is detected — this keeps the four windows methodologically
    identical to each other rather than applying different settings per
    window based on their individual class balance, which would itself
    become a confound in the window-size comparison.
    """
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_eval = eval_df[FEATURE_COLUMNS]
    y_eval = eval_df["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)  # transform only, never re-fit on eval data

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_eval_scaled)
    accuracy = accuracy_score(y_eval, y_pred)
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)

    coefficients = dict(zip(FEATURE_COLUMNS, model.coef_[0]))

    logger.info(
        "[%s] train_n=%d  eval_n=%d  accuracy=%.1f%%",
        window_name, len(train_df), len(eval_df), accuracy * 100
    )

    return {
        "window": window_name,
        "train_n": len(train_df),
        "eval_n": len(eval_df),
        "accuracy": accuracy,
        "classification_report": report,
        "coefficients": coefficients,
    }


