import pandas as pd
import numpy as np
import sys

# ─── CORE CONFIG (Thai Physical Gold Optimized) ────────────
SPREAD_PCT      = 0.0014      # Spread ทองไทย (โดนหักเฉพาะฝั่งขาเข้า/Buy)

# ปรับ Target ให้สอดคล้องกับการเล่นสั้นจบใน Session (M5)
TARGET_MOVE_PCT = 0.0020      # เป้าหมายทำกำไรสั้นๆ ~4.5 - 5 USD 
MAX_RISK_PCT    = 0.0032      # ทนแรงสะบัดได้ ~10 USD (เน้น Win Rate ให้รอดจาก Noise)

BUY_PRICE_MOVE  = TARGET_MOVE_PCT 
SELL_PRICE_MOVE = TARGET_MOVE_PCT 

HORIZON         = 36          # 36 แท่ง (3 ชั่วโมงบน M5) เหมาะสมกับรอบ Session
COOLDOWN_BARS   = 2           # ป้องกันสัญญาณกระจุกตัว (Non-Maximum Suppression)
MAX_SCORE       = 5.0

# ─── SESSION DEFINITIONS ──────────────────────────────────
WEEKDAY_SESSIONS = [
    ("Morning",   (6, 15), (12, 0),  False),
    ("Afternoon", (12, 0), (18, 0),  False),
    ("Night",     (18, 0), (2, 0),   True),
]
WEEKEND_SESSIONS = [
    ("Weekend",   (9, 30), (17, 30), False),
]

def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["session_id"]       = None
    df["session_name"]     = None
    df["session_start_ts"] = pd.NaT
    df["session_end_ts"]   = pd.NaT

    for date in sorted(ts.dt.date.unique()):
        base = pd.Timestamp(date)
        dow  = base.weekday()
        defs = WEEKDAY_SESSIONS if dow < 5 else WEEKEND_SESSIONS

        for name, (sh, sm), (eh, em), next_day in defs:
            start    = base + pd.Timedelta(hours=sh, minutes=sm)
            end_base = base + pd.Timedelta(days=1) if next_day else base
            end      = end_base + pd.Timedelta(hours=eh, minutes=em)
            mask     = (ts >= start) & (ts <= end)
            sid      = f"{date}_{name}"
            df.loc[mask, "session_id"]       = sid
            df.loc[mask, "session_name"]     = name
            df.loc[mask, "session_start_ts"] = start
            df.loc[mask, "session_end_ts"]   = end

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    hour, minute = ts.dt.hour, ts.dt.minute

    # แปลงเวลาเป็นวงกลม (Cyclical encoding)
    df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
    df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60)

    start_ts = pd.to_datetime(df["session_start_ts"], errors="coerce")
    end_ts   = pd.to_datetime(df["session_end_ts"], errors="coerce")
    
    total_dur = (end_ts - start_ts).dt.total_seconds()
    elapsed   = (ts - start_ts).dt.total_seconds()

    # ความคืบหน้าของ Session (0.0 ถึง 1.0)
    df["session_progress"] = (elapsed / total_dur.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    df["day_of_week"] = ts.dt.dayofweek
    
    return df

def ensure_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "xauusd_close" not in df.columns:
        df["xauusd_close"] = 2500 * (1 + df["xauusd_ret1"]).cumprod()
    if "xauusd_high" not in df.columns:
        df["xauusd_high"] = df["xauusd_close"] * 1.0005
    if "xauusd_low" not in df.columns:
        df["xauusd_low"]  = df["xauusd_close"] * 0.9995
    return df

def safe_score(reward: float, risk: float) -> float:
    rr = reward / max(risk, 1e-4)
    return min(rr, MAX_SCORE)

def generate_dual_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_prices(df)
    df = assign_sessions(df)
    df = add_time_features(df)
    n  = len(df)

    close_p = df["xauusd_close"].values
    high_p  = df["xauusd_high"].values
    low_p   = df["xauusd_low"].values

    buy, sell = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    buy_score_arr, sell_score_arr = np.zeros(n), np.zeros(n)

    for i in range(n - HORIZON):
        if pd.isna(df.at[i, "session_id"]):
            continue

        entry       = close_p[i]  # ราคา Bid ปัจจุบันของกราฟ
        future_high = np.max(high_p[i+1 : i+HORIZON+1])
        future_low  = np.min(low_p[i+1  : i+HORIZON+1])

        # ── Thai Physical Gold Logic ──────────────────────
        # 1. ฝั่ง BUY (หาจุดเข้าซื้อที่ดีที่สุด)
        # ซื้อหน้าร้านต้องโดนบวก Spread จึงต้องหา "ต้นทุนซื้อจริง" ก่อน
        actual_buy_entry = entry * (1 + SPREAD_PCT)
        
        buy_up_move   = (future_high - actual_buy_entry) / actual_buy_entry
        buy_down_move = (actual_buy_entry - future_low)  / actual_buy_entry

        buy_reward = max(buy_up_move, 0)
        buy_risk   = buy_down_move

        # 2. ฝั่ง SELL (หาจุดขายคืนร้าน / Exit Point หรือจุดสูงสุดก่อนราคาลง)
        # เวลาขายคืนร้าน เราขายที่ราคา Bid ตามกราฟได้เลย ไม่โดน Spread ซ้ำ
        sell_down_move = (entry - future_low) / entry
        sell_up_move   = (future_high - entry) / entry

        sell_reward = max(sell_down_move, 0)
        sell_risk   = sell_up_move

        buy_score_arr[i]  = safe_score(buy_reward,  buy_risk)
        sell_score_arr[i] = safe_score(sell_reward, sell_risk)

        # ── Pathwise Validation ───────────────────────────
        buy_valid = sell_valid = False

        # ตรวจสอบการวิ่งของราคาฝั่ง BUY (อิงจากต้นทุนจริงที่แพงกว่ากราฟ)
        for j in range(i+1, i+HORIZON+1):
            if low_p[j] <= actual_buy_entry * (1 - MAX_RISK_PCT): 
                break # ชน SL (คิดจากต้นทุนที่บวก Spread แล้ว)
            if high_p[j] >= actual_buy_entry * (1 + BUY_PRICE_MOVE):
                buy_valid = True
                break # ถึงเป้า TP (กราฟต้องวิ่งไกลพอที่จะชนะทั้ง Spread และบวกกำไร)

        # ตรวจสอบการวิ่งของราคาฝั่ง SELL (อิงจากราคา Bid ตามกราฟได้เลย)
        for j in range(i+1, i+HORIZON+1):
            if high_p[j] >= entry * (1 + MAX_RISK_PCT): 
                break # ชน SL (ราคาวิ่งสวนขึ้นไปทำ New High)
            if low_p[j] <= entry * (1 - SELL_PRICE_MOVE):
                sell_valid = True
                break # ราคาลงถึงเป้า (หาจุดจบรอบขาขึ้นได้สมบูรณ์)

        if buy_valid:  buy[i]  = 1
        if sell_valid: sell[i] = 1

    # ── Conflict Resolution ───────────────────────────────
    # กรณีสวิงแรงจนชนทั้งสองเป้าหมาย เลือกทางที่ R:R ดีกว่า
    conflict = (buy == 1) & (sell == 1)
    for k in np.where(conflict)[0]:
        if buy_score_arr[k] >= sell_score_arr[k]: 
            sell[k] = 0
        else: 
            buy[k] = 0

    # ── Cluster Suppression (NMS) ─────────────────────────
    # ป้องกันการเปิดออเดอร์ซ้ำซ้อนในสวิงเดียวกัน คัดเอาแค่แท่งที่ดีที่สุด
    for i in range(n):
        if buy[i] == 1:
            s, e = max(0, i - COOLDOWN_BARS), min(n, i + COOLDOWN_BARS + 1)
            best = s + int(np.argmax(buy_score_arr[s:e] * buy[s:e]))
            if i != best: buy[i] = 0

        if sell[i] == 1:
            s, e = max(0, i - COOLDOWN_BARS), min(n, i + COOLDOWN_BARS + 1)
            best = s + int(np.argmax(sell_score_arr[s:e] * sell[s:e]))
            if i != best: sell[i] = 0

    df["target_buy"], df["target_sell"] = buy, sell
    df["buy_score"], df["sell_score"] = buy_score_arr, sell_score_arr

    return df

def print_summary(df: pd.DataFrame) -> None:
    active   = df.dropna(subset=["session_id"])
    buy_n    = int(active["target_buy"].sum())
    sell_n   = int(active["target_sell"].sum())
    sessions = active["session_id"].nunique()

    print("\n" + "=" * 65)
    print("GOLD LABEL ENGINE - FINAL VERSION (Thai Physical & Dual Model)")
    print("=" * 65)
    print(f"Total Evaluated Rows  : {len(active):,}")
    print(f"Total Sessions        : {sessions:,}")
    print(f"BUY  Signals          : {buy_n:,}")
    print(f"SELL Signals          : {sell_n:,}")
    print(f"Avg Signals / Session : {(buy_n + sell_n) / sessions:.2f} (Natural Frequency)")
    print("─" * 65)
    print(f"TARGET_MOVE_PCT       : {TARGET_MOVE_PCT:.4f}  ({TARGET_MOVE_PCT*100:.2f}%)")
    print(f"MAX_RISK_PCT          : {MAX_RISK_PCT:.4f}  ({MAX_RISK_PCT*100:.2f}%)")
    print(f"HORIZON               : {HORIZON} bars")
    print(f"COOLDOWN_BARS         : {COOLDOWN_BARS} bars")
    
    time_cols = ["hour_sin", "hour_cos", "minute_sin", "minute_cos",
                 "session_progress", "day_of_week"]
    present = [c for c in time_cols if c in df.columns]
    print(f"Time Features Added   : {len(present)}/{len(time_cols)}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    input_csv  = sys.argv[1] if len(sys.argv) > 1 else "gold_data.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "gold_data_labeled_v6.csv"

    print(f"📁 Loading : {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ Error: ไม่พบไฟล์ {input_csv}")
        sys.exit(1)

    print("🎯 Generating Final Labels...")
    out = generate_dual_targets(df)
    
    # 1. ตัดข้อมูลเฉพาะที่มี session (เฉพาะช่วงเวลาที่เราสนใจเทรด)
    out = out.dropna(subset=["session_id"]).reset_index(drop=True)
    
    # 2. ลบ columns ตัวช่วยที่ไม่จำเป็นต้องใช้เทรนทิ้ง
    out = out.drop(columns=["session_start_ts", "session_end_ts"], errors="ignore")
    
    out.to_csv(output_csv, index=False)
    print_summary(out)
    print(f"✅ Saved : {output_csv}")