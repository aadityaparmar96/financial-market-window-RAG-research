"""
data_processor.py
-----------------
Financial Market RAG System — Data Processing Module

Responsibility: Load raw CSV datasets from the data/ directory, normalize them
into a common document schema, and return a list of dicts ready for embedding.

This module does NOT:
  - Generate embeddings
  - Connect to vector databases
  - Perform retrieval

Output schema per document:
    {
        "text":     str,   # Human-readable sentence(s) for the embedding model
        "metadata": {
            "source":       str,   # Original filename (without extension)
            "date":         str,   # ISO-8601 date string "YYYY-MM-DD"
            "dataset_type": str,   # Inferred category: macro | stock | news | generic
        }
    }
"""

import os
import logging
from pathlib import Path
from typing import Optional
import json

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_processor")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All recognised date-column spellings (checked case-insensitively).
DATE_COLUMN_CANDIDATES = [
    "date",
    "observation_date",
    "timestamp",
    "datetime",
    "time",
    "period",
]

# Keyword → dataset_type mapping (matched against the filename, lower-cased).
# Extend this dict as new dataset families are added to the project.
FILENAME_TYPE_MAP = {
    # Macro / FRED indicators
    "cpi": "macro",
    "gdp": "macro",
    "pce": "macro",
    "unrate": "macro",
    "fedfunds": "macro",
    "dgs": "macro",          # Treasury yields (DGS2, DGS10, …)
    "payems": "macro",
    "umcsent": "macro",      # UMich consumer sentiment
    "indpro": "macro",
    "houst": "macro",
    # Equity / market data
    "stock": "stock",
    "equity": "stock",
    "price": "stock",
    "ohlc": "stock",
    "close": "stock",
    "spy": "stock",
    "qqq": "stock",
    "etf": "stock",
    # News / sentiment
    "news": "news",
    "headline": "news",
    "article": "news",
    "sentiment": "news",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward from *start* (default: this file's directory) until a directory
    containing both 'data/' and 'src/' is found.  Falls back to the directory
    that contains 'data/' alone.  Raises FileNotFoundError if neither is found.
    """
    anchor = start or Path(__file__).resolve().parent
    for directory in [anchor, *anchor.parents]:
        has_src  = (directory / "src").is_dir()
        has_data = (directory / "data").is_dir()
        if has_data and has_src:
            logger.debug("Project root detected (src + data): %s", directory)
            return directory
        if has_data:
            logger.debug("Project root detected (data only): %s", directory)
            return directory
    raise FileNotFoundError(
        f"Cannot locate project root (no 'data/' folder found above {anchor})."
    )


def get_data_directory() -> Path:
    """Return the absolute path to the project's data/ directory."""
    root = find_project_root()
    data_dir = root / "data"
    logger.info("Data directory: %s", data_dir)
    return data_dir


# ---------------------------------------------------------------------------
# Dataset-type inference
# ---------------------------------------------------------------------------

def infer_dataset_type(filename: str) -> str:
    """
    Infer a semantic category from the CSV filename by checking whether any
    keyword in FILENAME_TYPE_MAP appears as a substring of the lower-cased
    stem.

    Returns one of: "macro" | "stock" | "news" | "generic"
    """
    stem = Path(filename).stem.lower()
    for keyword, dtype in FILENAME_TYPE_MAP.items():
        if keyword in stem:
            return dtype
    return "generic"


# ---------------------------------------------------------------------------
# Date-column detection
# ---------------------------------------------------------------------------

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return the name of the first column that matches a known date-column name
    (case-insensitive).  Returns None if no match is found.
    """
    col_lower_map = {col.lower(): col for col in df.columns}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in col_lower_map:
            return col_lower_map[candidate]
    return None


def parse_date_column(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Parse *col* in *df* into a normalised pd.Series of ISO-8601 date strings
    ("YYYY-MM-DD").  Rows that cannot be parsed become NaT and are later
    dropped by the cleaning step.
    """
    parsed = pd.to_datetime(df[col], errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Column-type classification
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame, date_col: str):
    """
    Split the DataFrame columns (excluding the date column) into two groups:

    Returns
    -------
    numeric_cols : list[str]
        Columns whose dtype is numeric (int / float).
    text_cols : list[str]
        All remaining non-date columns (object / string / bool …).
    """
    value_cols = [c for c in df.columns if c != date_col]
    numeric_cols = df[value_cols].select_dtypes(include="number").columns.tolist()
    text_cols    = [c for c in value_cols if c not in numeric_cols]
    return numeric_cols, text_cols


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def row_to_text(
    row: pd.Series,
    date_str: str,
    source: str,
    numeric_cols: list[str],
    text_cols: list[str],
    dataset_type: str,
) -> str:
    """
    Convert a single data row into a natural-language string suitable for an
    embedding model.

    Numeric columns → "On DATE, METRIC value was X."
    Text columns    → "On DATE, FIELD: VALUE."

    All non-null values are included; null values are skipped silently.
    """
    sentences = []

    for col in numeric_cols:
        val = row.get(col)
        if pd.isna(val):
            continue
        # Round to 4 significant figures to avoid floating-point noise
        val_fmt = f"{val:.4g}" if isinstance(val, float) else str(val)
        sentences.append(f"On {date_str}, {col} value was {val_fmt}.")

    for col in text_cols:
        val = row.get(col)
        if pd.isna(val) or str(val).strip() == "":
            continue
        sentences.append(f"On {date_str}, {col}: {str(val).strip()}.")

    if not sentences:
        return ""

    # Prepend a dataset-level context line so the embedding captures provenance
    prefix = f"[{dataset_type.upper()} | {source}] "
    return prefix + " ".join(sentences)


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_csv(filepath: Path) -> list[dict]:
    """
    Load and normalise one CSV file into a list of documents.

    Steps
    -----
    1. Load CSV, inferring dtypes.
    2. Detect and parse the date column.
    3. Drop rows with missing dates or all-null value columns.
    4. Classify remaining columns (numeric vs. text).
    5. Convert each row to a natural-language string + metadata dict.
    6. Skip rows that produce an empty text string.

    Parameters
    ----------
    filepath : Path
        Absolute path to the CSV file.

    Returns
    -------
    list[dict]
        One dict per valid row: {"text": ..., "metadata": {...}}
    """
    source       = filepath.stem                    # e.g. "CPIAUCSL"
    dataset_type = infer_dataset_type(filepath.name)

    logger.info("Loading %-30s  type=%-8s", filepath.name, dataset_type)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(filepath, dtype=str)  # Load as str first; parse later
    except Exception as exc:
        logger.error("Failed to read %s: %s", filepath, exc)
        return []

    if df.empty:
        logger.warning("File is empty, skipping: %s", filepath.name)
        return []

    logger.debug("  Columns: %s", list(df.columns))
    logger.debug("  Raw rows: %d", len(df))

    # ------------------------------------------------------------------
    # 2. Detect & parse date column
    # ------------------------------------------------------------------
    date_col = detect_date_column(df)

    if date_col is None:
        logger.warning(
            "No date column found in %s (columns: %s). Skipping.",
            filepath.name, list(df.columns),
        )
        return []

    df["_date_parsed"] = parse_date_column(df, date_col)

    # Drop the original date column to avoid duplication in text generation
    df = df.drop(columns=[date_col])

    # ------------------------------------------------------------------
    # 3. Clean
    # ------------------------------------------------------------------
    # Drop rows where the date could not be parsed
    before = len(df)
    df = df.dropna(subset=["_date_parsed"])
    dropped_dates = before - len(df)
    if dropped_dates:
        logger.warning("  Dropped %d rows with unparseable dates.", dropped_dates)

    # Coerce numeric-looking columns to float (they were loaded as str)
    value_cols = [c for c in df.columns if c != "_date_parsed"]
    for col in value_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Only replace column with numeric version if at least half the
        # non-null values converted successfully (avoids clobbering text cols)
        non_null = df[col].notna().sum()
        converted = coerced.notna().sum()
        if non_null > 0 and (converted / non_null) >= 0.5:
            df[col] = coerced

    # Drop rows where every value column is null (date-only rows carry no signal)
    before = len(df)
    df = df.dropna(subset=value_cols, how="all")
    dropped_empty = before - len(df)
    if dropped_empty:
        logger.warning("  Dropped %d fully-empty rows.", dropped_empty)

    logger.info("  Valid rows after cleaning: %d", len(df))

    # ------------------------------------------------------------------
    # 4. Classify columns
    # ------------------------------------------------------------------
    numeric_cols, text_cols = classify_columns(df, "_date_parsed")
    logger.debug("  Numeric cols: %s", numeric_cols)
    logger.debug("  Text cols:    %s", text_cols)

    # ------------------------------------------------------------------
    # 5. Convert rows → documents
    # ------------------------------------------------------------------
    documents = []
    for _, row in df.iterrows():
        date_str = row["_date_parsed"]

        text = row_to_text(
            row=row,
            date_str=date_str,
            source=source,
            numeric_cols=numeric_cols,
            text_cols=text_cols,
            dataset_type=dataset_type,
        )

        # 6. Skip empty documents
        if not text:
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source":       source,
                    "date":         date_str,
                    "dataset_type": dataset_type,
                },
            }
        )

    logger.info("  Documents produced: %d", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Multi-file entry point
# ---------------------------------------------------------------------------

def load_all_documents(data_dir: Optional[Path] = None) -> list[dict]:
    """
    Discover all CSV files under *data_dir* (defaults to the auto-detected
    project data/ directory), process each one, and return the concatenated
    list of documents.

    Parameters
    ----------
    data_dir : Path, optional
        Override the data directory (useful for unit tests or notebooks).

    Returns
    -------
    list[dict]
        All normalised documents across every CSV file, sorted by date then
        source — a convenient default ordering for window-based experiments.
    """
    if data_dir is None:
        data_dir = get_data_directory()

    csv_files = sorted(data_dir.glob("**/*.csv"))  # recursive; finds subdirs too

    if not csv_files:
        logger.warning("No CSV files found in %s", data_dir)
        return []

    logger.info("Found %d CSV file(s) to process.", len(csv_files))

    all_documents: list[dict] = []
    for csv_path in csv_files:
        docs = process_csv(csv_path)
        all_documents.extend(docs)

    # Sort by date (ISO strings sort correctly as plain strings), then source
    all_documents.sort(key=lambda d: (d["metadata"]["date"], d["metadata"]["source"]))

    logger.info(
        "Total documents loaded: %d  (spanning %s → %s)",
        len(all_documents),
        all_documents[0]["metadata"]["date"]  if all_documents else "N/A",
        all_documents[-1]["metadata"]["date"] if all_documents else "N/A",
    )

    return all_documents


# ---------------------------------------------------------------------------
# Convenience filter (used by the retriever in later pipeline stages)
# ---------------------------------------------------------------------------

def filter_by_window(
    documents: list[dict],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Return only the documents whose date falls within [start_date, end_date]
    (both inclusive).  Dates must be ISO-8601 strings "YYYY-MM-DD".

    This function is intentionally kept here (rather than in retriever.py) so
    that researchers can quickly test window effects directly on raw documents
    without setting up a vector store.

    Parameters
    ----------
    documents  : output of load_all_documents()
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"
    """
    return [
        doc for doc in documents
        if start_date <= doc["metadata"]["date"] <= end_date
    ]


#data saver:

def save_document(documents : list[dict], output_path : Path)-> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            documents,
            f,
            indent=4,
            ensure_ascii=False
        )
    print(f"Saved {len(documents)} documents to {output_path}")
    


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick sanity check.  Run from the project root:

        python src/data_processor.py

    Prints a summary and the first three documents.
    """
    docs = load_all_documents()

    print(f"\n{'='*60}")
    print(f"  Total documents : {len(docs)}")
    if docs:
        dates = [d["metadata"]["date"] for d in docs]
        print(f"  Date range      : {min(dates)}  →  {max(dates)}")
        types = {}
        for d in docs:
            t = d["metadata"]["dataset_type"]
            types[t] = types.get(t, 0) + 1
        print(f"  By dataset type : {types}")
    print(f"{'='*60}\n")

    print("Sample documents:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n[{i}] text     : {doc['text']}")
        print(f"    metadata : {doc['metadata']}")