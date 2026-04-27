from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

def load_data(csv_path: str):
    """ โหลด Dataset ที่ผ่านการ Label Dual-Target มาแล้ว """
    df = pd.read_csv(csv_path)
    
    # ตรวจสอบว่ามี Column Target หรือไม่
    assert 'target_buy' in df.columns and 'target_sell' in df.columns, "ข้อมูลต้องมีคอลัมน์ target_buy และ target_sell"
    
    # ---------------------------------------------------------
    # 🟢 เพิ่มส่วนนี้: แปลง session_name เป็นตัวเลข (One-Hot Encoding)
    # ---------------------------------------------------------
    if 'session_name' in df.columns:
        df = pd.get_dummies(df, columns=['session_name'], drop_first=False)
        # แปลง True/False เป็น 1/0 ให้ XGBoost ไม่หงุดหงิด
        for col in df.columns:
            if col.startswith('session_name_'):
                df[col] = df[col].astype(int)
    # ---------------------------------------------------------

    # ดึงคอลัมน์ Features (ตัดพวก timestamp, session_id และ target ออก)
    # เพิ่ม 'session_name' ลงไปใน drop_cols ด้วยเผื่อตกค้าง
    drop_cols = [
        'timestamp', 'session_id', 'session_name', 
        'target_buy', 'target_sell', 
        'm1_buy_tpsl', 'm1_sell_tpsl', 
        'm2_buy_sniper', 'm2_sell_sniper'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    return df, feature_cols

def custom_evaluate(y_true, y_prob):
    """ 
    ใช้ PR-AUC (Average Precision) เป็นตัวชี้วัดหลัก 
    เพราะเหมาะกับข้อมูล Imbalance ขีดสุดแบบของเรา 
    และเสริมด้วย F1 ที่ Optimize Threshold แล้ว
    """
    pr_auc = average_precision_score(y_true, y_prob)
    
    # หา Threshold ที่ให้ F1 ดีที่สุด
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)
    
    # ให้น้ำหนัก PR-AUC 70% และ Best F1 30%
    return (0.7 * pr_auc) + (0.3 * best_f1)

def build_model(model_type: str, params: dict):
    if model_type == "xgboost":
        return xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric="logloss")
    elif model_type == "lightgbm":
        return lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError("รองรับเฉพาะ xgboost หรือ lightgbm")

def run_optuna_for_side(df: pd.DataFrame, feature_cols: list, target_col: str, model_type: str, n_trials: int) -> dict:
    """ จูนพารามิเตอร์ให้ด้านใดด้านหนึ่ง (BUY หรือ SELL) """
    import optuna

    X = df[feature_cols]
    y = df[target_col]

    # คำนวณ Imbalance Ratio คร่าวๆ เพื่อเป็นไกด์ให้ Optuna
    ratio = float(len(y[y == 0]) / max(len(y[y == 1]), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # หัวใจสำคัญแก้ Hold Bias! ให้ Optuna สุ่มค่าน้ำหนัก
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, ratio * 1.5)
        }

        # ใช้ TimeSeriesSplit เพื่อจำลองการเทรดแบบ Forward-walking
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = build_model(model_type, params)
            model.fit(X_train, y_train)

            # ให้ทำนายความน่าจะเป็น (Probability)
            y_prob = model.predict_proba(X_val)[:, 1]
            score = custom_evaluate(y_val, y_prob)
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return {
        "best_score": study.best_value,
        "best_params": study.best_params
    }

def main(csv_path: str, model_type: str, n_trials: int, output_dir: str):
    print(f"🚀 เริ่มการ Finetune Dual-Model ด้วยข้อมูล: {csv_path}")
    df, feature_cols = load_data(csv_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print("🟢 กำลัง Finetune โมเดลสาย BUY...")
    print("="*50)
    buy_results = run_optuna_for_side(df, feature_cols, "target_buy", model_type, n_trials)
    
    print("\n" + "="*50)
    print("🔴 กำลัง Finetune โมเดลสาย SELL...")
    print("="*50)
    sell_results = run_optuna_for_side(df, feature_cols, "target_sell", model_type, n_trials)

    # บันทึกผลลัพธ์
    payload = {
        "buy_model": buy_results,
        "sell_model": sell_results,
        "model_type": model_type
    }
    
    result_file = out_path / "finetune_dual_results.json"
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print("\n✅ เสร็จสิ้น! สรุปคะแนน (PR-AUC + Best F1):")
    print(f"BUY Model Best Score : {buy_results['best_score']:.4f}")
    print(f"SELL Model Best Score: {sell_results['best_score']:.4f}")
    print(f"บันทึกพารามิเตอร์ที่ดีที่สุดไว้ที่: {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/label/gold_train_2025.csv", help="ไฟล์ Label ฉบับใหม่")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--trials", type=int, default=50, help="จำนวนรอบที่ให้ Optuna จูนต่อ 1 หน้าเทรด")
    parser.add_argument("--out", type=str, default="artifacts")
    args = parser.parse_args()

    main(args.csv, args.model, args.trials, args.out)