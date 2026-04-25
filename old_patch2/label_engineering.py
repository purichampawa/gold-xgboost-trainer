import pandas as pd
import numpy as np
from pathlib import Path

# ==================================
# CONFIG
# ==================================
TP_PCT = 0.0018
SL_PCT = 0.0010
HORIZON = 6

INPUT_FILE = "data/label/gold_data.csv"
OUTPUT_FILE = "data/label/gold_master_labeled.csv"


# ==================================
# CORE LABELING
# ==================================
def create_labels(df):

    n = len(df)

    buy = np.zeros(n, dtype=int)
    sell = np.zeros(n, dtype=int)

    close = df["xauusd_close"].values
    high = df["xauusd_high"].values
    low = df["xauusd_low"].values

    for i in range(n - HORIZON):

        entry = close[i]

        buy_tp = entry * (1 + TP_PCT)
        buy_sl = entry * (1 - SL_PCT)

        sell_tp = entry * (1 - TP_PCT)
        sell_sl = entry * (1 + SL_PCT)

        # ---------- BUY ----------
        for j in range(i+1, i+HORIZON+1):

            if low[j] <= buy_sl:
                break

            if high[j] >= buy_tp:
                buy[i] = 1
                break

        # ---------- SELL ----------
        for j in range(i+1, i+HORIZON+1):

            if high[j] >= sell_sl:
                break

            if low[j] <= sell_tp:
                sell[i] = 1
                break

        # conflict remove
        if buy[i] == 1 and sell[i] == 1:
            buy[i] = 0
            sell[i] = 0

    df["buy_label"] = buy
    df["sell_label"] = sell

    # remove rows with incomplete future
    df = df.iloc[:-HORIZON].copy()

    return df


# ==================================
# REPORT
# ==================================
def report(df):

    print("="*60)

    print("BUY:")
    print(df["buy_label"].value_counts(normalize=True))

    print()

    print("SELL:")
    print(df["sell_label"].value_counts(normalize=True))

    print("="*60)


# ==================================
# MAIN
# ==================================
def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = create_labels(df)

    Path("data/label").mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    report(df)

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()