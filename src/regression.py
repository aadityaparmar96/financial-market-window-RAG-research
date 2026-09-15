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