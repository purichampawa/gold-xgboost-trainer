import pandas as pd
import numpy as np

# ==========================================================
# GOLD LABEL ENGINE V5 (Top-K Dual-Model Optimized)
# Physical Gold Signal / Highly Sensitive / Noise Resistant
# ==========================================================

# CONFIG
SPREAD_PCT      = 0.0014      

# 1. ลดเป้าหมายลงเหลือ 0.16% (ที่ราคา 2692 กราฟลงแค่ ~4.3 ดอลลาร์ บอทก็แจกสัญญาณ Sell ทันที!)
TARGET_MOVE_PCT = 0.0018      # (ลดจาก 0.0030) 

MAX_RISK_PCT    = 0.0043      # คงไว้ที่ 0.40% (ทนแรงสะบัดไส้เทียนได้ถึง 10 ดอลลาร์)

BUY_PRICE_MOVE  = TARGET_MOVE_PCT 
SELL_PRICE_MOVE = TARGET_MOVE_PCT 

# 2. ขยายเวลาให้กราฟไหลลงมาชนเป้าแบบไม่ต้องรีบ
HORIZON         = 36          # (เพิ่มจาก 16) ให้เวลาไปเลย 48 แท่งเทียน

# 3. ให้บอทคายสัญญาณเป็นโซนกว้างๆ จะได้ไม่ตกรถ
COOLDOWN_BARS   = 2           # (ลดจาก 3) ถ้ายอดเขา 2692 มันกว้าง คุณจะได้สัญญาณ Sell รัวๆ 2-3 แท่งติดกันเลย          # (ลดจาก 3) ถ้ายอดเขามีหลายแท่งสูสีกัน จะได้สัญญาณ 2-3 แท่งติดกันไปให้โมเดลเรียนรู้

MAX_SCORE = 5.0

# ==========================================================
# SESSION DEFINITIONS (อัปเดตเวลาตามเป้าหมายเป๊ะๆ)
# ==========================================================
WEEKDAY_SESSIONS = [
    ("Morning",   (6, 15), (12, 0), False),
    ("Afternoon", (12, 0), (18, 0), False),
    ("Night",     (18, 0), (2, 0), True),
]

WEEKEND_SESSIONS = [
    ("Weekend",   (9, 30), (17, 30), False),
]

def assign_sessions(df):
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["session_id"] = None
    df["session_name"] = None

    for date in sorted(ts.dt.date.unique()):
        base = pd.Timestamp(date)
        dow = base.weekday()
        defs = WEEKDAY_SESSIONS if dow < 5 else WEEKEND_SESSIONS

        for name, (sh, sm), (eh, em), next_day in defs:
            start = base + pd.Timedelta(hours=sh, minutes=sm)
            end_base = base + pd.Timedelta(days=1) if next_day else base
            end = end_base + pd.Timedelta(hours=eh, minutes=em)
            mask = (ts >= start) & (ts <= end)
            sid = f"{date}_{name}"
            df.loc[mask, "session_id"] = sid
            df.loc[mask, "session_name"] = name
    return df

def ensure_prices(df):
    df = df.copy()
    if "xauusd_close" not in df.columns:
        df["xauusd_close"] = 2500 * (1 + df["xauusd_ret1"]).cumprod()
    if "xauusd_high" not in df.columns:
        df["xauusd_high"] = df["xauusd_close"] * 1.0005
    if "xauusd_low" not in df.columns:
        df["xauusd_low"] = df["xauusd_close"] * 0.9995
    return df

def safe_score(reward, risk):
    rr = reward / max(risk, 1e-4)
    return min(rr, MAX_SCORE)

def generate_dual_targets(df):
    df = ensure_prices(df)
    df = assign_sessions(df)
    n = len(df)

    close_p = df["xauusd_close"].values
    high_p  = df["xauusd_high"].values
    low_p   = df["xauusd_low"].values

    buy = np.zeros(n, dtype=int)
    sell = np.zeros(n, dtype=int)
    buy_score_arr = np.zeros(n)
    sell_score_arr = np.zeros(n)

    for i in range(n - HORIZON):
        if pd.isna(df.at[i, "session_id"]):
            continue

        entry = close_p[i]
        future_high = np.max(high_p[i+1:i+HORIZON+1])
        future_low  = np.min(low_p[i+1:i+HORIZON+1])

        up_move = (future_high - entry) / entry
        down_move = (entry - future_low) / entry

        # --------------------------------------------------
        # PHYSICAL GOLD REWARD LOGIC (หัก Spread แค่ขา Buy)
        # --------------------------------------------------
        buy_reward = max(up_move - SPREAD_PCT, 0)
        sell_reward = max(down_move, 0) 

        buy_score = safe_score(buy_reward, down_move) 
        sell_score = safe_score(sell_reward, up_move)
        
        buy_score_arr[i] = buy_score
        sell_score_arr[i] = sell_score

        # --------------------------------------------------
        # PATHWISE VALIDATION
        # --------------------------------------------------
        buy_valid, sell_valid = False, False
        for j in range(i+1, i+HORIZON+1):
            if low_p[j] <= entry * (1 - MAX_RISK_PCT):
                break
            if high_p[j] >= entry * (1 + BUY_PRICE_MOVE):
                buy_valid = True
                break

        for j in range(i+1, i+HORIZON+1):
            if high_p[j] >= entry * (1 + MAX_RISK_PCT):
                break
            if low_p[j] <= entry * (1 - SELL_PRICE_MOVE):
                sell_valid = True
                break

        if buy_valid: buy[i] = 1
        if sell_valid: sell[i] = 1

    # ------------------------------------------------------
    # 1. SMART CONFLICT RESOLUTION
    # ------------------------------------------------------
    conflict = (buy == 1) & (sell == 1)
    if conflict.any():
        for k in np.where(conflict)[0]:
            if buy_score_arr[k] >= sell_score_arr[k]:
                sell[k] = 0
            else:
                buy[k] = 0

    # ------------------------------------------------------
    # 2. CLUSTER SUPPRESSION (NMS)
    # ------------------------------------------------------
    for i in range(n):
        if buy[i] == 1:
            start = max(0, i - COOLDOWN_BARS)
            end = min(n, i + COOLDOWN_BARS + 1)
            local_scores = buy_score_arr[start:end] * buy[start:end]
            best_local_idx = start + np.argmax(local_scores)
            if i != best_local_idx:
                buy[i] = 0

    for i in range(n):
        if sell[i] == 1:
            start = max(0, i - COOLDOWN_BARS)
            end = min(n, i + COOLDOWN_BARS + 1)
            local_scores = sell_score_arr[start:end] * sell[start:end]
            best_local_idx = start + np.argmax(local_scores)
            if i != best_local_idx:
                sell[i] = 0

    df["target_buy"] = buy
    df["target_sell"] = sell
    df["buy_score"] = buy_score_arr
    df["sell_score"] = sell_score_arr

    # # ------------------------------------------------------
    # # 🌟 3. TOP-K SESSION RANKING (บังคับคัดเฉพาะตัวท็อปใน Session)
    # # ------------------------------------------------------
    # for sid in df["session_id"].dropna().unique():
    #     idx = df[df["session_id"] == sid].index
        
    #     # กรองฝั่ง BUY
    #     buy_idx = idx[df.loc[idx, "target_buy"] == 1]
    #     if len(buy_idx) > TOP_K_PER_SESSION:
    #         # เรียงคะแนนจากมากไปน้อย แล้วเอาแค่ K อันดับแรก
    #         top_buy = df.loc[buy_idx, "buy_score"].sort_values(ascending=False).head(TOP_K_PER_SESSION).index
    #         # สัญญาณที่เหลือปรับเป็น 0 (ลบทิ้ง ไม่ให้โมเดลจำไปใช้)
    #         df.loc[buy_idx.difference(top_buy), "target_buy"] = 0
            
    #     # กรองฝั่ง SELL
    #     sell_idx = idx[df.loc[idx, "target_sell"] == 1]
    #     if len(sell_idx) > TOP_K_PER_SESSION:
    #         top_sell = df.loc[sell_idx, "sell_score"].sort_values(ascending=False).head(TOP_K_PER_SESSION).index
    #         df.loc[sell_idx.difference(top_sell), "target_sell"] = 0

    return df

def print_summary(df):
    active = df.dropna(subset=["session_id"])
    buy_n = int(active["target_buy"].sum())
    sell_n = int(active["target_sell"].sum())
    ratio = sell_n / buy_n if buy_n else 0
    sessions = active["session_id"].nunique()

    print("\n" + "=" * 60)
    print("GOLD LABEL ENGINE V5 (Top-K Dual-Model)")
    print("=" * 60)
    print(f"Total Evaluated Rows  : {len(active)}")
    print(f"Total Sessions        : {sessions}")
    print(f"BUY Signals (Top-K)   : {buy_n}")
    print(f"SELL Signals (Top-K)  : {sell_n}")
    print(f"SELL / BUY Ratio      : {ratio:.2f} (เข้าใกล้ 1.0 คือดีเยี่ยม)")
    print(f"Avg Signals/Session   : {(buy_n + sell_n) / sessions:.2f} (ควร > 2.0 ตามกฎ)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    print("📁 Loading raw data...")
    df = pd.read_csv("gold_data.csv")
    print("🎯 Generating Labels...")
    out = generate_dual_targets(df)
    out = out.dropna(subset=["session_id"]).reset_index(drop=True)
    out.to_csv("gold_data_labeled_v4.csv", index=False)
    print_summary(out)
    print("✅ Saved: gold_data_labeled_v4.csv")