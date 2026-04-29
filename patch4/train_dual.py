import joblib
import pandas as pd
import xgboost as xgb
from pathlib import Path
import lightgbm as lgb

from config import CONFIG

def load_training_data(csv_path: str):
    print(f"📥 กำลังโหลดข้อมูลสำหรับเทรนจาก: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # ดรอปคอลัมน์ที่ไม่ใช้ และคอลัมน์ที่เป็น Data Leakage
    drop_cols = [
        'timestamp', 'session_id', 'session_name', 'target_buy', 'target_sell', 
        'm1_buy_tpsl', 'm1_sell_tpsl', 'm2_buy_sniper', 'm2_sell_sniper',
        'buy_score', 'sell_score'
    ]
    
    # เก็บเฉพาะคอลัมน์ที่มีอยู่จริง
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    X = df[feature_cols]
    y_buy = df['target_buy']
    y_sell = df['target_sell']
    
    return X, y_buy, y_sell, feature_cols

def train_master_models():
    CONFIG.model.buy_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. โหลดข้อมูลเทรน
    X, y_buy, y_sell, feature_cols = load_training_data(str(CONFIG.data.csv_path))
    
    model_type = CONFIG.model.model_type.lower()
        
    print("\n" + "="*50)
    print("🟢 กำลังเทรนโมเดลสาย BUY (Model Buy)...")
    print("="*50)
    
    # 2. ผูกพารามิเตอร์ของหน้า BUY จาก config.py
    if model_type == "xgboost":
        buy_model = xgb.XGBClassifier(
            n_estimators=CONFIG.model.buy_n_estimators,
            learning_rate=CONFIG.model.buy_learning_rate,
            max_depth=CONFIG.model.buy_max_depth,
            subsample=CONFIG.model.buy_subsample,
            colsample_bytree=CONFIG.model.buy_colsample_bytree,
            scale_pos_weight=CONFIG.model.buy_scale_pos_weight,
            random_state=CONFIG.model.random_state,
            n_jobs=CONFIG.model.n_jobs,
            eval_metric="logloss",
        )
    else:
        buy_model = lgb.LGBMClassifier(
            n_estimators=CONFIG.model.buy_n_estimators,
            learning_rate=CONFIG.model.buy_learning_rate,
            max_depth=CONFIG.model.buy_max_depth,
            subsample=CONFIG.model.buy_subsample,
            colsample_bytree=CONFIG.model.buy_colsample_bytree,
            scale_pos_weight=CONFIG.model.buy_scale_pos_weight,
            random_state=CONFIG.model.random_state,
            n_jobs=CONFIG.model.n_jobs,
            verbose=-1,
        )
    buy_model.fit(X, y_buy)
    
    print("\n" + "="*50)
    print("🔴 กำลังเทรนโมเดลสาย SELL (Model Sell)...")
    print("="*50)
    
    # 3. ผูกพารามิเตอร์ของหน้า SELL จาก config.py
    if model_type == "xgboost":
        sell_model = xgb.XGBClassifier(
            n_estimators=CONFIG.model.sell_n_estimators,
            learning_rate=CONFIG.model.sell_learning_rate,
            max_depth=CONFIG.model.sell_max_depth,
            subsample=CONFIG.model.sell_subsample,
            colsample_bytree=CONFIG.model.sell_colsample_bytree,
            scale_pos_weight=CONFIG.model.sell_scale_pos_weight,
            random_state=CONFIG.model.random_state,
            n_jobs=CONFIG.model.n_jobs,
            eval_metric="logloss",
        )
    else:
        sell_model = lgb.LGBMClassifier(
            n_estimators=CONFIG.model.sell_n_estimators,
            learning_rate=CONFIG.model.sell_learning_rate,
            max_depth=CONFIG.model.sell_max_depth,
            subsample=CONFIG.model.sell_subsample,
            colsample_bytree=CONFIG.model.sell_colsample_bytree,
            scale_pos_weight=CONFIG.model.sell_scale_pos_weight,
            random_state=CONFIG.model.random_state,
            n_jobs=CONFIG.model.n_jobs,
            verbose=-1,
        )
    sell_model.fit(X, y_sell)

    # 4. บันทึกผลลัพธ์ลงโฟลเดอร์ outputs/models
    buy_model_path = CONFIG.model.buy_model_path
    sell_model_path = CONFIG.model.sell_model_path
    features_path = CONFIG.model.feature_columns_path
    
    joblib.dump(buy_model, buy_model_path)
    joblib.dump(sell_model, sell_model_path)
    
    with open(features_path, "w", encoding="utf-8") as f:
        import json
        json.dump(feature_cols, f, indent=4)

    print("\n✅ เทรนเสร็จสมบูรณ์!")
    print(f"📦 บันทึก Model BUY ไว้ที่ : {buy_model_path}")
    print(f"📦 บันทึก Model SELL ไว้ที่: {sell_model_path}")
    print(f"📋 บันทึก Features ไว้ที่   : {features_path}")

if __name__ == "__main__":
    train_master_models()