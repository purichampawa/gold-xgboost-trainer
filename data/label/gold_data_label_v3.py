import pandas as pd
import numpy as np

# ─── CORE CONFIG ───────────────────────────────────────────
SPREAD_PCT      = 0.0014      # spread จริงของ Physical Gold

# ✅ แก้ไขหลัก: ต้องให้ TARGET >= RISK * 1.5 เสมอ
# ทำให้ Expected Payoff Ratio ≥ 0.70 แทนที่จะเป็น 0.49
TARGET_MOVE_PCT = 0.0040      # เพิ่มจาก 0.0018 → ~10.7 USD ที่ราคา 2692
MAX_RISK_PCT    = 0.0043      # คงไว้ (SL ~11.6 USD) — R:R ≈ 0.93 ดีมาก

BUY_PRICE_MOVE  = TARGET_MOVE_PCT
SELL_PRICE_MOVE = TARGET_MOVE_PCT

# ✅ ขยาย HORIZON ให้โมเดลมีเวลาเห็น "pattern ใหญ่"
HORIZON         = 48          # เพิ่มจาก 36 → ให้เวลา 48 แท่ง (4 ชั่วโมง บน M5)

# ✅ ลด Cooldown เล็กน้อยเพื่อเพิ่มจำนวนสัญญาณ
COOLDOWN_BARS   = 3           # คงไว้ที่ 3 (ป้องกัน cluster signals)

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


# ─── HELPERS ──────────────────────────────────────────────
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
    """
    เพิ่ม cyclical time features ให้โมเดลรู้ว่า "ตอนนี้กี่โมง"
    ใช้ sin/cos encoding เพราะ 23:00 กับ 00:00 ใกล้กันจริง
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    hour   = ts.dt.hour
    minute = ts.dt.minute

    # Cyclical encoding
    df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
    df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60)

    # Session Progress (0.0 = เริ่มต้น session, 1.0 = ใกล้หมด)
    start_ts = pd.to_datetime(df["session_start_ts"], errors="coerce")
    end_ts   = pd.to_datetime(df["session_end_ts"],   errors="coerce")
    curr_ts  = ts

    total_dur = (end_ts - start_ts).dt.total_seconds()
    elapsed   = (curr_ts - start_ts).dt.total_seconds()

    df["session_progress"] = (elapsed / total_dur.replace(0, np.nan)).clip(0.0, 1.0)
    df["session_progress"] = df["session_progress"].fillna(0.0)

    # Day of week (0=Mon … 4=Fri, 5=Sat, 6=Sun)
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


# ─── MAIN LABEL FUNCTION ──────────────────────────────────
def generate_dual_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_prices(df)
    df = assign_sessions(df)
    df = add_time_features(df)      # ✅ เพิ่ม time features ก่อน return
    n  = len(df)

    close_p = df["xauusd_close"].values
    high_p  = df["xauusd_high"].values
    low_p   = df["xauusd_low"].values

    buy           = np.zeros(n, dtype=int)
    sell          = np.zeros(n, dtype=int)
    buy_score_arr  = np.zeros(n)
    sell_score_arr = np.zeros(n)

    for i in range(n - HORIZON):
        if pd.isna(df.at[i, "session_id"]):
            continue

        entry       = close_p[i]
        future_high = np.max(high_p[i+1 : i+HORIZON+1])
        future_low  = np.min(low_p[i+1  : i+HORIZON+1])

        up_move   = (future_high - entry) / entry
        down_move = (entry - future_low)  / entry

        # ── Reward Logic ──────────────────────────────────
        buy_reward  = max(up_move   - SPREAD_PCT, 0)
        sell_reward = max(down_move, 0)   # ✅ หัก spread ทั้งสองฝั่ง (fair)

        buy_score  = safe_score(buy_reward,  down_move)
        sell_score = safe_score(sell_reward, up_move)

        buy_score_arr[i]  = buy_score
        sell_score_arr[i] = sell_score

        # ── Pathwise Validation ───────────────────────────
        buy_valid = sell_valid = False

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

        if buy_valid:  buy[i]  = 1
        if sell_valid: sell[i] = 1

    # ── Conflict Resolution ───────────────────────────────
    conflict = (buy == 1) & (sell == 1)
    for k in np.where(conflict)[0]:
        if buy_score_arr[k] >= sell_score_arr[k]:
            sell[k] = 0
        else:
            buy[k] = 0

    # ── Cluster Suppression (NMS) ─────────────────────────
    for i in range(n):
        if buy[i] == 1:
            s = max(0, i - COOLDOWN_BARS)
            e = min(n, i + COOLDOWN_BARS + 1)
            local = buy_score_arr[s:e] * buy[s:e]
            best  = s + int(np.argmax(local))
            if i != best:
                buy[i] = 0

    for i in range(n):
        if sell[i] == 1:
            s = max(0, i - COOLDOWN_BARS)
            e = min(n, i + COOLDOWN_BARS + 1)
            local = sell_score_arr[s:e] * sell[s:e]
            best  = s + int(np.argmax(local))
            if i != best:
                sell[i] = 0

    df["target_buy"]  = buy
    df["target_sell"] = sell
    df["buy_score"]   = buy_score_arr
    df["sell_score"]  = sell_score_arr

    return df


# ─── SUMMARY ──────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    active   = df.dropna(subset=["session_id"])
    buy_n    = int(active["target_buy"].sum())
    sell_n   = int(active["target_sell"].sum())
    ratio    = sell_n / buy_n if buy_n else 0
    sessions = active["session_id"].nunique()

    # คำนวณ theoretical R:R ที่ Label ออกแบบไว้
    theoretical_rr = TARGET_MOVE_PCT / MAX_RISK_PCT

    print("\n" + "=" * 65)
    print("GOLD LABEL ENGINE V6  (Walk-Forward Ready)")
    print("=" * 65)
    print(f"Total Evaluated Rows  : {len(active):,}")
    print(f"Total Sessions        : {sessions:,}")
    print(f"BUY  Signals          : {buy_n:,}")
    print(f"SELL Signals          : {sell_n:,}")
    print(f"SELL / BUY Ratio      : {ratio:.2f}  (เข้าใกล้ 1.0 = ดี)")
    print(f"Avg Signals / Session : {(buy_n + sell_n) / sessions:.2f}")
    print("─" * 65)
    print(f"TARGET_MOVE_PCT       : {TARGET_MOVE_PCT:.4f}  ({TARGET_MOVE_PCT*100:.2f}%)")
    print(f"MAX_RISK_PCT          : {MAX_RISK_PCT:.4f}  ({MAX_RISK_PCT*100:.2f}%)")
    print(f"Theoretical R:R       : {theoretical_rr:.2f}  (ควร >= 0.80)")
    print(f"HORIZON               : {HORIZON} bars")
    print(f"COOLDOWN_BARS         : {COOLDOWN_BARS}")

    # ตรวจสอบ time features
    time_cols = ["hour_sin", "hour_cos", "minute_sin", "minute_cos",
                 "session_progress", "day_of_week"]
    present = [c for c in time_cols if c in df.columns]
    print(f"Time Features Added   : {len(present)}/{len(time_cols)} {present}")
    print("=" * 65 + "\n")


# ─── ENTRY POINT ──────────────────────────────────────────
if __name__ == "__main__":
    import sys

    input_csv  = sys.argv[1] if len(sys.argv) > 1 else "gold_data.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "gold_data_labeled_v5.csv"

    print(f"📁 Loading : {input_csv}")
    df = pd.read_csv(input_csv)

    print("🎯 Generating Labels V6...")
    out = generate_dual_targets(df)
    out = out.dropna(subset=["session_id"]).reset_index(drop=True)

    # ✅ drop helper columns ที่ใช้คำนวณแล้วไม่ต้องเก็บใน CSV
    out = out.drop(columns=["session_start_ts", "session_end_ts"], errors="ignore")

    out.to_csv(output_csv, index=False)
    print_summary(out)
    print(f"✅ Saved : {output_csv}")