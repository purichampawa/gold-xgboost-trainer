from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from old_patch2.config import CONFIG
from old_patch2.signals import DEFAULT_SIGNAL_ENGINE


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[CONFIG.data.timestamp_col] = pd.to_datetime(df[CONFIG.data.timestamp_col], errors="coerce")
    df = df.dropna(subset=[CONFIG.data.timestamp_col]).sort_values(CONFIG.data.timestamp_col).reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame) -> pd.Series:
    close = df[CONFIG.data.price_col]
    future_price = close.shift(-CONFIG.labels.horizon_bars)
    future_ret = (future_price - close) / close

    buy_th = CONFIG.labels.threshold_buy
    sell_th = CONFIG.labels.threshold_sell
    if CONFIG.labels.use_spread_aware_labels:
        buy_th += CONFIG.labels.spread_buffer
        sell_th -= CONFIG.labels.spread_buffer

    labels = pd.Series(CONFIG.labels.hold_label, index=df.index, dtype="object")
    labels[future_ret > buy_th] = CONFIG.labels.buy_label
    labels[future_ret < sell_th] = CONFIG.labels.sell_label
    return labels


def detect_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    excluded = {CONFIG.data.timestamp_col, target_col}
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric feature columns detected.")
    return numeric_cols


def time_series_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * CONFIG.split.train_ratio)
    n_val = int(n * CONFIG.split.val_ratio)
    if n_train <= 0 or n_val <= 0 or (n_train + n_val) >= n:
        raise ValueError("Invalid split ratios for dataset size.")
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def make_model(model_type: str, class_weight: dict[int, float] | None = None) -> Any:
    model_key = model_type.lower()
    common = {
        "n_estimators": CONFIG.model.n_estimators,
        "max_depth": CONFIG.model.max_depth,
        "random_state": CONFIG.model.random_state,
    }

    if model_key == "random_forest":
        return RandomForestClassifier(
            **common,
            n_jobs=-1,
            class_weight=class_weight,
        )

    if model_key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost") from exc

        return XGBClassifier(
            **common,
            learning_rate=CONFIG.model.learning_rate,
            subsample=CONFIG.model.subsample,
            colsample_bytree=CONFIG.model.colsample_bytree,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )

    if model_key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed. Install with: pip install lightgbm") from exc
        return LGBMClassifier(
            **common,
            learning_rate=CONFIG.model.learning_rate,
            subsample=CONFIG.model.subsample,
            colsample_bytree=CONFIG.model.colsample_bytree,
            class_weight=class_weight,
        )

    if model_key == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is not installed. Install with: pip install catboost") from exc
        return CatBoostClassifier(
            iterations=CONFIG.model.n_estimators,
            depth=CONFIG.model.max_depth,
            learning_rate=CONFIG.model.learning_rate,
            random_seed=CONFIG.model.random_state,
            verbose=False,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def compute_balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    labels = np.unique(y)
    weights = compute_class_weight("balanced", classes=labels, y=y)
    return {int(cls): float(w) for cls, w in zip(labels, weights)}


def evaluate_model(model: Any, x: pd.DataFrame, y_true: np.ndarray, split_name: str) -> dict[str, Any]:
    y_pred = model.predict(x)
    probs = model.predict_proba(x)
    actions = DEFAULT_SIGNAL_ENGINE.batch_probs_to_actions(probs)
    y_action_encoded = DEFAULT_SIGNAL_ENGINE.encode_labels(actions)

    metrics = {
        "split": split_name,
        "accuracy_pred_class": float(accuracy_score(y_true, y_pred)),
        "macro_f1_pred_class": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy_signal_action": float(accuracy_score(y_true, y_action_encoded)),
        "macro_f1_signal_action": float(f1_score(y_true, y_action_encoded, average="macro")),
        "classification_report_pred_class": classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


def run_training(model_type: str | None = None) -> None:
    df = load_dataset(CONFIG.data.csv_path)
    
    # ดึงคอลัมน์ Signal จาก CSV ตรงๆ (ไม่ต้อง build_labels ใหม่)
    target_col = CONFIG.data.raw_signal_col
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    feature_cols = detect_feature_columns(df, target_col=target_col)
    train_df, val_df, test_df = time_series_split(df)

    y_train = DEFAULT_SIGNAL_ENGINE.encode_labels(train_df[target_col])
    y_val = DEFAULT_SIGNAL_ENGINE.encode_labels(val_df[target_col])
    y_test = DEFAULT_SIGNAL_ENGINE.encode_labels(test_df[target_col])

    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    x_test = test_df[feature_cols]

    class_weight = None
    if CONFIG.model.class_weight_mode == "balanced":
        class_weight = compute_balanced_class_weights(y_train)

    model = make_model(model_type or CONFIG.model.model_type, class_weight=class_weight)
    model.fit(x_train, y_train)

    val_metrics = evaluate_model(model, x_val, y_val, "val")
    test_metrics = evaluate_model(model, x_test, y_test, "test")

    CONFIG.train_output.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CONFIG.train_output.model_path)

    with CONFIG.train_output.feature_columns_path.open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    out = {
        "model_type": model_type or CONFIG.model.model_type,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "label_config": asdict(CONFIG.labels),
        "signal_config": asdict(CONFIG.signals),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with CONFIG.train_output.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved model to: {CONFIG.train_output.model_path}")
    print(f"Saved metrics to: {CONFIG.train_output.metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BUY/HOLD/SELL classifier for gold trading.")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Override model type: xgboost | lightgbm | catboost | random_forest",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(model_type=args.model_type)
