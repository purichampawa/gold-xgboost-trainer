from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import asdict

from old_patch2.config import CONFIG
from old_patch2.signals import SignalEngine
from old_patch2.train import build_labels, detect_feature_columns, load_dataset, make_model, time_series_split


def evaluate_with_signal_engine(
    model: Any,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    signal_engine: SignalEngine,
) -> float:
    probs = model.predict_proba(x_val)
    actions = signal_engine.batch_probs_to_actions(probs)
    y_action = signal_engine.encode_labels(actions)
    # Weighted objective: emphasize macro F1 under imbalance.
    f1 = f1_score(y_val, y_action, average="macro")
    acc = accuracy_score(y_val, y_action)
    return float(0.7 * f1 + 0.3 * acc)


def trial_objective(
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> float:
    old_model_cfg = asdict(CONFIG.model).copy()
    old_signal_cfg = asdict(CONFIG.signals).copy()
    old_label_cfg = asdict(CONFIG.labels).copy()

    # Model hyperparameters.
    CONFIG.model.learning_rate = params["learning_rate"]
    CONFIG.model.max_depth = int(params["max_depth"])
    CONFIG.model.n_estimators = int(params["n_estimators"])

    # Signal thresholds.
    CONFIG.signals.threshold_buy = params["threshold_buy"]
    CONFIG.signals.threshold_sell = params["threshold_sell"]
    CONFIG.signals.hold_zone = params["hold_zone"]
    CONFIG.signals.confidence_filter = params["confidence_filter"]

    # Label horizon.
    # CONFIG.labels.horizon_bars = int(params["label_horizon"])

    selected_features = params.get("feature_subset")
    if selected_features is not None:
        x_train_trial = x_train[selected_features]
        x_val_trial = x_val[selected_features]
    else:
        x_train_trial = x_train
        x_val_trial = x_val

    model = make_model(CONFIG.model.model_type)
    model.fit(x_train_trial, y_train)
    score = evaluate_with_signal_engine(model, x_val_trial, y_val, SignalEngine(config=CONFIG.signals))

    # Restore global config objects to avoid trial leakage.
    for key, val in old_model_cfg.items():
        setattr(CONFIG.model, key, val)
    for key, val in old_signal_cfg.items():
        setattr(CONFIG.signals, key, val)
    for key, val in old_label_cfg.items():
        setattr(CONFIG.labels, key, val)

    return score


def run_optuna(n_trials: int, output_path: Path) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is not installed. Install with: pip install optuna") from exc

    df = load_dataset(CONFIG.data.csv_path)
    
    # อ่านจาก Signal column ตรงๆ เหมือนกัน
    target_col = CONFIG.data.raw_signal_col
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    feature_cols = detect_feature_columns(df, target_col=target_col)

    train_df, val_df, _ = time_series_split(df)
    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    y_train = SignalEngine(config=CONFIG.signals).encode_labels(train_df[target_col])
    y_val = SignalEngine(config=CONFIG.signals).encode_labels(val_df[target_col])

    def objective(trial: optuna.Trial) -> float:
        feature_subset = None
        if trial.suggest_categorical("feature_selection", [True, False]):
            k = trial.suggest_int("feature_count", max(4, len(feature_cols) // 3), len(feature_cols))
            
            # ให้ Optuna สุ่มเลือก 'ชื่อกลยุทธ์' แทนการโยน List เข้าไปตรงๆ
            subset_strategy = trial.suggest_categorical(
                "feature_subset_strategy",
                ["first_k", "last_k", "even_k", "odd_k"]
            )
            
            # นำกลยุทธ์ที่ Optuna เลือกมาสร้าง Subset เอง
            if subset_strategy == "first_k":
                feature_subset = feature_cols[:k]
            elif subset_strategy == "last_k":
                feature_subset = feature_cols[-k:]
            elif subset_strategy == "even_k":
                feature_subset = feature_cols[::2][:k]
            elif subset_strategy == "odd_k":
                feature_subset = feature_cols[1::2][:k]

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "threshold_buy": trial.suggest_float("threshold_buy", 0.30, 0.70),
            "threshold_sell": trial.suggest_float("threshold_sell", 0.30, 0.70),
            "hold_zone": trial.suggest_float("hold_zone", 0.02, 0.25),
            "confidence_filter": trial.suggest_float("confidence_filter", 0.20, 0.70),
            # "label_horizon": trial.suggest_int("label_horizon", 2, 24),
            "feature_subset": feature_subset,
        }
        return trial_objective(params, x_train, y_train, x_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_score": study.best_value,
        "best_params": study.best_params,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Best score: {study.best_value:.6f}")
    print(f"Saved tuning results to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for gold trading classifier.")
    parser.add_argument("--n-trials", type=int, default=40, help="Number of Optuna trials.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/finetune_results.json"),
        help="Where to save best tuning parameters.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_optuna(n_trials=args.n_trials, output_path=args.output)
