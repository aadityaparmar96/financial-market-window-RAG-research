"""
regression.py
-------------
Financial Market RAG System — Regression Module (Leak-Proof Control)

Training methodology:
    - Hyperparameter search (C) via TimeSeriesSplit + GridSearchCV, done
      entirely within each window's own training data.
    - Feature values clipped post-scaling to prevent out-of-distribution
      inputs from saturating the sigmoid.
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
    accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("regression")

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
    "unemployment_lag1", "fed_funds_rate_lag1", "cpi_lag1",
    "yield_spread_lag1", "sp500_return_lag1",
]

C_GRID = [0.01, 0.1, 1.0, 10.0]
CLIP_THRESHOLD = 4.0


def load_and_merge_raw_data(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    merged: Optional[pd.DataFrame] = None
    for stem, col_name in SOURCE_COLUMN_MAP.items():
        filepath = raw_dir / f"{stem}.csv"
        if not filepath.exists():
            logger.warning("Expected file not found, skipping: %s", filepath)
            continue

        df = pd.read_csv(filepath)
        date_col = next((c for c in df.columns if c.lower() in ("date", "observation_date", "datetime")), None)
        if date_col is None:
            logger.warning("No date column found in %s, skipping.", filepath.name)
            continue

        value_col = [c for c in df.columns if c != date_col][0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])
        df = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: col_name})
        df = df.set_index("date").resample("MS").mean()

        merged = df if merged is None else merged.join(df, how="outer")
        logger.info("Loaded and merged %s -> column '%s' (%d rows)", stem, col_name, len(df))

    if merged is None:
        raise FileNotFoundError("No source files were loaded. Check SOURCE_COLUMN_MAP.")

    if "real_gdp" in merged.columns:
        merged["real_gdp"] = merged["real_gdp"].ffill()

    merged = merged.sort_index()
    logger.info("Merged dataframe: %d rows, %s -> %s", len(merged), merged.index.min().date(), merged.index.max().date())
    return merged


def build_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sp500_return"] = df["sp500_price"].pct_change()

    feature_cols = {
        "unemployment": "unemployment_lag1", "fed_funds_rate": "fed_funds_rate_lag1",
        "cpi": "cpi_lag1", "yield_spread": "yield_spread_lag1", "sp500_return": "sp500_return_lag1",
    }
    for source_col, lagged_col in feature_cols.items():
        if source_col in df.columns:
            df[lagged_col] = df[source_col].shift(1)
        else:
            logger.warning("Expected column '%s' missing.", source_col)

    df["target"] = (df["sp500_price"].shift(-1) > df["sp500_price"]).astype(int)

    required = list(feature_cols.values()) + ["target"]
    before = len(df)
    df = df.dropna(subset=required)
    logger.info("Feature engineering: %d -> %d rows after dropping NaNs", before, len(df))
    return df


def get_window_slice(df: pd.DataFrame, window_name: str) -> pd.DataFrame:
    if window_name not in WINDOW_YEARS:
        raise ValueError(f"Unknown window '{window_name}'.")
    cutoff_ts = pd.Timestamp(CUTOFF)
    start_ts = cutoff_ts - pd.DateOffset(years=WINDOW_YEARS[window_name])
    return df[(df.index >= start_ts) & (df.index <= cutoff_ts)]


def get_eval_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(EVAL_START)) & (df.index <= pd.Timestamp(EVAL_END))]


def check_class_balance(df: pd.DataFrame, label: str) -> None:
    counts = df["target"].value_counts(normalize=True).sort_index()
    logger.info("%s class balance: %.1f%% up-months, %.1f%% down-months (n=%d)",
                label, counts.get(1, 0.0) * 100, counts.get(0, 0.0) * 100, len(df))


def check_multicollinearity(df: pd.DataFrame, label: str, threshold: float = 0.8) -> None:
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
    return np.clip(X_scaled, -clip_at, clip_at)


def train_and_evaluate_window(train_df, eval_df, window_name, use_clipping=True) -> dict:
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_eval = eval_df[FEATURE_COLUMNS]
    y_eval = eval_df["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    z_min, z_max = X_eval_scaled.min(), X_eval_scaled.max()
    logger.info("[%s] eval feature z-score range (pre-clip): min=%.1f, max=%.1f", window_name, z_min, z_max)

    if use_clipping:
        X_train_scaled = clip_extreme_values(X_train_scaled)
        X_eval_scaled = clip_extreme_values(X_eval_scaled)

    n_splits = min(5, max(2, len(train_df) // 12))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        param_grid={"C": C_GRID}, cv=tscv, scoring="balanced_accuracy",
    )
    grid.fit(X_train_scaled, y_train)
    best_C = grid.best_params_["C"]
    logger.info("[%s] GridSearchCV selected C=%.3f (cv folds=%d)", window_name, best_C, n_splits)

    model = grid.best_estimator_
    y_pred = model.predict(X_eval_scaled)
    y_proba = model.predict_proba(X_eval_scaled)[:, 1]

    accuracy = accuracy_score(y_eval, y_pred)
    balanced_acc = balanced_accuracy_score(y_eval, y_pred)
    predicted_up_rate = float(np.mean(y_pred))
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)

    try:
        auc = roc_auc_score(y_eval, y_proba)
    except ValueError:
        auc = None

    coefficients = dict(zip(FEATURE_COLUMNS, model.coef_[0]))

    logger.info("[%s] C=%.3f  accuracy=%.1f%%  balanced_acc=%.1f%%  pred_up_rate=%.1f%%  auc=%s",
                window_name, best_C, accuracy * 100, balanced_acc * 100,
                predicted_up_rate * 100, f"{auc:.3f}" if auc is not None else "N/A")

    return {
        "window": window_name, "best_C": best_C, "train_n": len(train_df), "eval_n": len(eval_df),
        "accuracy": accuracy, "balanced_accuracy": balanced_acc, "predicted_up_rate": predicted_up_rate,
        "roc_auc": auc, "eval_z_score_range": [float(z_min), float(z_max)],
        "classification_report": report, "coefficients": coefficients,
    }


def run_regression_experiment(raw_dir: Optional[Path] = None) -> dict[str, dict]:
    merged = load_and_merge_raw_data(raw_dir)
    featured = build_features_and_target(merged)

    eval_df = get_eval_slice(featured)
    if len(eval_df) == 0:
        raise ValueError("Evaluation slice (2016-2024) is empty.")
    check_class_balance(eval_df, "Eval (2016-2024)")

    results = {}
    for window_name in WINDOW_YEARS:
        train_df = get_window_slice(featured, window_name)
        if len(train_df) < 12:
            logger.warning("[%s] Only %d training rows — results will be unreliable.", window_name, len(train_df))
        check_class_balance(train_df, window_name)
        check_multicollinearity(train_df, window_name)
        results[window_name] = train_and_evaluate_window(train_df, eval_df, window_name)

    return results


if __name__ == "__main__":
    results = run_regression_experiment()

    merged = load_and_merge_raw_data()
    featured = build_features_and_target(merged)
    eval_df = get_eval_slice(featured)
    baseline = naive_baseline_accuracy(eval_df)

    print(f"\n{'='*70}\nREGRESSION RESULTS — Properly Tuned (TimeSeriesSplit + Clipping)\n{'='*70}")
    print(f"NAIVE BASELINE (always predict majority class): {baseline*100:.1f}%\n")

    for window_name, r in results.items():
        flag = "  <-- BELOW BASELINE" if r["accuracy"] < baseline else ""
        auc_str = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "N/A"
        print(f"{window_name:6s}  best_C={r['best_C']:6.3f}  train_n={r['train_n']:4d}  "
              f"accuracy={r['accuracy']*100:5.1f}%  balanced_acc={r['balanced_accuracy']*100:5.1f}%  "
              f"auc={auc_str}  pred_up_rate={r['predicted_up_rate']*100:5.1f}%{flag}")

    print(f"\n{'='*70}\nFeature coefficients by window\n{'='*70}")
    for window_name, r in results.items():
        print(f"\n[{window_name}] (C={r['best_C']})")
        for feat, coef in r["coefficients"].items():
            print(f"  {feat:24s} {coef:+.4f}")

    import json
    output_path = Path("results") / "regression_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")