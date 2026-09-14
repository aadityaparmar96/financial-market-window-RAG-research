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