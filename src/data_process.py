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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_processor")


DATE_COLUMN_CANDIDATES = [
    "date", "observation_date", "timestamp", "datetime", "time", "period",
]

FILENAME_TYPE_MAP = {
    "cpi": "macro", "gdp": "macro", "pce": "macro", "unrate": "macro",
    "fedfunds": "macro", "dgs": "macro", "payems": "macro", "umcsent": "macro",
    "indpro": "macro", "houst": "macro",
    "stock": "stock", "equity": "stock", "price": "stock", "ohlc": "stock",
    "close": "stock", "spy": "stock", "qqq": "stock", "etf": "stock",
    "news": "news", "headline": "news", "article": "news", "sentiment": "news",
}

# PE10/CAPE cannot legitimately be 0 — it requires 10 years of trailing
# earnings data, so a literal 0 in early rows of a long historical series
# almost always encodes "not yet computable," not a real observation.
SENTINEL_ZERO_COLUMNS = ["PE10", "CAPE"]


def find_project_root(start: Optional[Path] = None) -> Path:
    anchor = start or Path(__file__).resolve().parent
    for directory in [anchor, *anchor.parents]:
        has_src = (directory / "src").is_dir()
        has_data = (directory / "data").is_dir()
        if has_data and has_src:
            return directory
        if has_data:
            return directory
    raise FileNotFoundError(f"Cannot locate project root (no 'data/' folder found above {anchor}).")


def get_data_directory() -> Path:
    root = find_project_root()
    data_dir = root / "data"
    logger.info("Data directory: %s", data_dir)
    return data_dir


def infer_dataset_type(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for keyword, dtype in FILENAME_TYPE_MAP.items():
        if keyword in stem:
            return dtype
    return "generic"


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    col_lower_map = {col.lower(): col for col in df.columns}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in col_lower_map:
            return col_lower_map[candidate]
    return None


def parse_date_column(df: pd.DataFrame, col: str) -> pd.Series:
    parsed = pd.to_datetime(df[col], errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def classify_columns(df: pd.DataFrame, date_col: str):
    value_cols = [c for c in df.columns if c != date_col]
    numeric_cols = df[value_cols].select_dtypes(include="number").columns.tolist()
    text_cols = [c for c in value_cols if c not in numeric_cols]
    return numeric_cols, text_cols


def row_to_text(row, date_str, source, numeric_cols, text_cols, dataset_type) -> str:
    sentences = []
    for col in numeric_cols:
        val = row.get(col)
        if pd.isna(val):
            continue
        val_fmt = f"{val:.4g}" if isinstance(val, float) else str(val)
        sentences.append(f"On {date_str}, {col} value was {val_fmt}.")
    for col in text_cols:
        val = row.get(col)
        if pd.isna(val) or str(val).strip() == "":
            continue
        sentences.append(f"On {date_str}, {col}: {str(val).strip()}.")
    if not sentences:
        return ""
    prefix = f"[{dataset_type.upper()} | {source}] "
    return prefix + " ".join(sentences)


def process_csv(filepath: Path) -> list[dict]:
    source = filepath.stem
    dataset_type = infer_dataset_type(filepath.name)
    logger.info("Loading %-30s  type=%-8s", filepath.name, dataset_type)

    try:
        df = pd.read_csv(filepath, dtype=str)
    except Exception as exc:
        logger.error("Failed to read %s: %s", filepath, exc)
        return []

    if df.empty:
        logger.warning("File is empty, skipping: %s", filepath.name)
        return []

    date_col = detect_date_column(df)
    if date_col is None:
        logger.warning("No date column found in %s. Skipping.", filepath.name)
        return []

    df["_date_parsed"] = parse_date_column(df, date_col)
    df = df.drop(columns=[date_col])

    before = len(df)
    df = df.dropna(subset=["_date_parsed"])
    if before - len(df):
        logger.warning("  Dropped %d rows with unparseable dates.", before - len(df))

    value_cols = [c for c in df.columns if c != "_date_parsed"]
    for col in value_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        converted = coerced.notna().sum()
        if non_null > 0 and (converted / non_null) >= 0.5:
            df[col] = coerced

    # Treat known sentinel-missing values (e.g. PE10=0 before it's computable)
    # as actually missing, not real data.
    for col in value_cols:
        if col in SENTINEL_ZERO_COLUMNS and col in df.columns:
            df.loc[df[col] == 0, col] = pd.NA

    before = len(df)
    df = df.dropna(subset=value_cols, how="all")
    if before - len(df):
        logger.warning("  Dropped %d fully-empty rows.", before - len(df))

    logger.info("  Valid rows after cleaning: %d", len(df))

    numeric_cols, text_cols = classify_columns(df, "_date_parsed")

    documents = []
    for _, row in df.iterrows():
        date_str = row["_date_parsed"]
        text = row_to_text(row, date_str, source, numeric_cols, text_cols, dataset_type)
        if not text:
            continue
        documents.append({
            "text": text,
            "metadata": {"source": source, "date": date_str, "dataset_type": dataset_type},
        })

    logger.info("  Documents produced: %d", len(documents))
    return documents


def load_all_documents(data_dir: Optional[Path] = None) -> list[dict]:
    if data_dir is None:
        data_dir = get_data_directory()

    csv_files = sorted(data_dir.glob("**/*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", data_dir)
        return []

    logger.info("Found %d CSV file(s) to process.", len(csv_files))

    all_documents: list[dict] = []
    for csv_path in csv_files:
        all_documents.extend(process_csv(csv_path))

    all_documents.sort(key=lambda d: (d["metadata"]["date"], d["metadata"]["source"]))

    logger.info(
        "Total documents loaded: %d  (spanning %s → %s)",
        len(all_documents),
        all_documents[0]["metadata"]["date"] if all_documents else "N/A",
        all_documents[-1]["metadata"]["date"] if all_documents else "N/A",
    )
    return all_documents


def filter_by_window(documents: list[dict], start_date: str, end_date: str) -> list[dict]:
    return [doc for doc in documents if start_date <= doc["metadata"]["date"] <= end_date]


def save_document(documents: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(documents)} documents to {output_path}")


if __name__ == "__main__":
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

    output_path = get_data_directory() / "processed" / "all_documents.json"
    save_document(docs, output_path)