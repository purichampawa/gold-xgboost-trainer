import pandas as pd
import numpy as np
from pathlib import Path

# ======================================================
# GOLD DATA CLEANER PRO
# Final Feature Set for 2-Model XGBoost System
# Buy Model + Sell Model
# ======================================================

INPUT_FILE = "data/processed/gold_feature_dataset.parquet"

OUTPUT_CSV = "data/label/gold_data.csv"
OUTPUT_PARQUET = "data/processed/gold_master_pro.parquet"


# ======================================================
# LOAD DATA
# ======================================================
def load_data(path):

    print(f"Loading: {path}")
    df = pd.read_parquet(path)

    # timestamp detect
    if "timestamp" not in df.columns:

        if "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp": "timestamp"})

        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})

        else:
            df = df.reset_index()

    if "timestamp" not in df.columns:
        raise ValueError("No timestamp column found.")

    # convert timestamp
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ======================================================
# FEATURE ENGINEERING
# ======================================================
def create_features(df):

    # ------------------------------------------
    # Returns
    # ------------------------------------------
    df["xauusd_ret1"] = df["xauusd_close"].pct_change(1)
    df["xauusd_ret3"] = df["xauusd_close"].pct_change(3)
    df["usdthb_ret1"] = df["usdthb_close"].pct_change(1)

    # ------------------------------------------
    # Momentum Change
    # ------------------------------------------
    df["xau_macd_delta1"] = df["xauusd_macd_hist"].diff(1)
    df["xau_rsi_delta1"] = df["xauusd_rsi14"].diff(1)

    # ------------------------------------------
    # Trend
    # ------------------------------------------
    df["trend_regime"] = (df["xauusd_dist_ema21"] > 0).astype(int)

    # ------------------------------------------
    # Volatility Rank
    # ------------------------------------------
    df["atr_rank50"] = (
        df["xauusd_atr_norm"]
        .rolling(50)
        .rank(pct=True)
    )

    # ------------------------------------------
    # Candle Reversal
    # ------------------------------------------
    df["wick_bias"] = (
        df["xauusd_lower_wick"] -
        df["xauusd_upper_wick"]
    )

    # body strength
    if all(col in df.columns for col in [
        "xauusd_open",
        "xauusd_high",
        "xauusd_low",
        "xauusd_close"
    ]):

        candle_range = (
            df["xauusd_high"] -
            df["xauusd_low"]
        ).replace(0, np.nan)

        body = abs(
            df["xauusd_close"] -
            df["xauusd_open"]
        )

        df["body_strength"] = body / candle_range

    else:
        df["body_strength"] = 0.0

    # ------------------------------------------
    # Session ID
    # ------------------------------------------
    df["session_id"] = df["timestamp"].apply(get_session_id)

    return df


# ======================================================
# SESSION CLASSIFIER
# ======================================================
def get_session_id(ts):

    h = ts.hour
    m = ts.minute
    minute = h * 60 + m
    wd = ts.weekday()

    # weekend special
    if wd >= 5:
        if 570 <= minute < 1050:
            return 4
        return 0

    # weekday
    if 375 <= minute < 720:
        return 1   # 06:15-12:00

    elif 720 <= minute < 1080:
        return 2   # 12:00-18:00

    elif minute >= 1080 or minute < 120:
        return 3   # 18:00-02:00

    return 0


# ======================================================
# SELECT FINAL FEATURES
# ======================================================
def select_features(df):

    cols = [
        "timestamp",

        # =======================================
        # NEW: Price (keep for labeling & plotting)
        # =======================================
        "xauusd_open",
        "xauusd_high",
        "xauusd_low",
        "xauusd_close",

        # momentum
        "xauusd_ret1",
        "xauusd_ret3",
        "usdthb_ret1",
        "xau_macd_delta1",

        # trend
        "xauusd_dist_ema21",
        "xauusd_dist_ema50",
        "usdthb_dist_ema21",
        "trend_regime",

        # oscillator
        "xauusd_rsi14",
        "xau_rsi_delta1",
        "xauusd_macd_hist",

        # volatility
        "xauusd_atr_norm",
        "xauusd_bb_width",
        "atr_rank50",

        # reversal
        "wick_bias",
        "body_strength",

        # time
        "hour_sin",
        "session_id",
    ]

    valid = [c for c in cols if c in df.columns]

    return df[valid].copy()


# ======================================================
# CLEAN
# ======================================================
def clean_data(df):

    df = df.replace([np.inf, -np.inf], np.nan)

    # warmup rows
    df = df.iloc[60:].copy()

    df = df.dropna().reset_index(drop=True)

    return df


# ======================================================
# SAVE
# ======================================================
def save_files(df):

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/label").mkdir(parents=True, exist_ok=True)

    # CSV
    df_csv = df.copy()
    df_csv["timestamp"] = df_csv["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df_csv.to_csv(OUTPUT_CSV, index=False)

    # Parquet
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Saved CSV     :", OUTPUT_CSV)
    print("Saved Parquet :", OUTPUT_PARQUET)


# ======================================================
# MAIN
# ======================================================
def main():

    df = load_data(INPUT_FILE)

    print("Creating features...")
    df = create_features(df)

    print("Selecting features...")
    df = select_features(df)

    print("Cleaning...")
    df = clean_data(df)

    print("Saving...")
    save_files(df)

    print("-" * 60)
    print("Rows :", len(df))
    print("Cols :", len(df.columns))
    print(df.columns.tolist())


if __name__ == "__main__":
    main()