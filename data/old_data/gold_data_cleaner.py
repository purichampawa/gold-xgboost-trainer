"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          GOLD_XGBOOST_TRADER — Feature Engineering Pipeline                ║
║          Stage 1: Dataset Construction (No Labels)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Supported raw formats
─────────────────────
GLD965  → timestamp(int), datetime(YYYY-MM-DD HH:MM:SS), open, high, low, close
          (comma-sep, NO volume/spread)
XAUUSD  → <DATE>\t<TIME>\t<OPEN>…<TICKVOL>\t<VOL>\t<SPREAD>  (tab MT5)
USDTHB  → same MT5 tab format as XAUUSD
Optional → DXY, US10Y, SPX, BTCUSD  (MT5 format, auto-detected)
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("GoldPipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE   = PROCESSED_DIR / "gold_feature_dataset.parquet"
ARTIFACT_DIR  = Path("artifacts")

PRIORITY_SYMBOLS = ["GLD965", "XAUUSD", "USDTHB"]
OPTIONAL_SYMBOLS = ["DXY", "US10Y", "SPX", "BTCUSD"]

SYMBOL_HINTS: Dict[str, str] = {
    "GLD965": "GLD965",
    "XAUUSD": "XAUUSD",
    "USDTHB": "USDTHB",
    "DXY":    "DXY",
    "US10Y":  "US10Y",
    "SPX":    "SPX",
    "BTCUSD": "BTCUSD",
}

CORR_WINDOWS = [12, 24, 48]   # 5-min bars: 1h / 2h / 4h
LAG_PERIODS  = [1, 2, 3]
ROLL_WINDOWS = [3, 6, 12]


# ═════════════════════════════════════════════════════════════════════════════
# 1.  I/O — format-aware loaders
# ═════════════════════════════════════════════════════════════════════════════

def detect_symbol(fp: Path) -> str:
    name = fp.stem.upper()
    for hint, sym in SYMBOL_HINTS.items():
        if hint in name:
            return sym
    return fp.stem.split("_")[0].upper()


def _load_gld965(fp: Path) -> pd.DataFrame:
    """
    GLD965 CSV format:
        timestamp (int row id), datetime, open, high, low, close
    No TICKVOL / VOL / SPREAD columns.
    """
    df = pd.read_csv(fp, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df["tick_vol"] = np.nan
    df["volume"]   = np.nan
    df["spread"]   = np.nan
    return df


def _load_mt5(fp: Path) -> pd.DataFrame:
    """
    MT5 tab-separated:
        <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
    """
    df = pd.read_csv(fp, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["TIME"].astype(str)
    )
    col_map = {
        "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
        "TICKVOL": "tick_vol", "VOL": "volume", "SPREAD": "spread",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    for c in ["tick_vol", "volume", "spread"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["timestamp", "open", "high", "low", "close", "tick_vol", "volume", "spread"]]


def load_raw_csv(fp: Path) -> pd.DataFrame:
    symbol = detect_symbol(fp)
    log.info(f"Loading {symbol:8s} ← {fp.name}")

    # Peek at first line to detect format
    with open(fp, "r", encoding="utf-8-sig") as fh:
        first = fh.readline()

    if "\t" in first:
        raw = _load_mt5(fp)
    else:
        # Comma-separated: GLD965 has 'datetime' column, others do not
        cols_lower = [c.strip().lower().strip("<>") for c in first.split(",")]
        raw = _load_gld965(fp) if "datetime" in cols_lower else _load_mt5(fp)

    # Shared cleanup
    for col in ["open", "high", "low", "close", "tick_vol", "volume", "spread"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = (
        raw.sort_values("timestamp")
           .drop_duplicates(subset="timestamp")
           .set_index("timestamp")
    )
    raw = raw.ffill(limit=3).dropna(subset=["open", "high", "low", "close"])

    # Prefix every column with symbol name
    raw.columns = [f"{symbol.lower()}_{c}" for c in raw.columns]

    log.info(f"  → {len(raw):>8,} bars  [{raw.index.min()}  …  {raw.index.max()}]")
    return raw


def load_all_symbols(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for fp in sorted(raw_dir.glob("*.csv")):
        sym = detect_symbol(fp)
        try:
            frames[sym] = load_raw_csv(fp)
        except Exception as e:
            log.warning(f"Skip {fp.name}: {e}")
    return frames


def align_symbols(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "GLD965" not in frames:
        raise RuntimeError("GLD965 is required as anchor symbol.")
    log.info("Aligning all symbols on GLD965 timestamp …")

    df = frames["GLD965"].copy()
    for sym, odf in frames.items():
        if sym == "GLD965":
            continue
        df = df.join(odf, how="left")
        sym_cols = [c for c in df.columns if c.startswith(sym.lower())]
        df[sym_cols] = df[sym_cols].ffill(limit=3)   # fill gaps up to 3 bars

    df = df.dropna(subset=["gld965_close"]).sort_index()
    log.info(f"Aligned: {len(df):,} rows × {len(df.columns)} raw columns")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Technical indicator helpers  (strictly no-lookahead)
# ═════════════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.DataFrame:
    up   = h - h.shift(1)
    down = l.shift(1) - l
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    ndm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    atr_s = _atr(h, l, c, period)
    di_p  = 100 * pdm.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_n  = 100 * ndm.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx    = 100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan)
    return pd.DataFrame({"adx":     dx.ewm(com=period - 1, adjust=False).mean(),
                         "di_plus": di_p,
                         "di_minus": di_n})


def _macd(s: pd.Series, fast=12, slow=26, sig=9) -> pd.DataFrame:
    ml = _ema(s, fast) - _ema(s, slow)
    sl = _ema(ml, sig)
    return pd.DataFrame({"macd": ml, "macd_sig": sl, "macd_hist": ml - sl})


def _bollinger(s: pd.Series, w=20, n=2) -> pd.DataFrame:
    mid = s.rolling(w).mean()
    std = s.rolling(w).std(ddof=0)
    bw  = 2 * n * std
    return pd.DataFrame({
        "bb_upper": mid + n * std,
        "bb_lower": mid - n * std,
        "bb_mid":   mid,
        "bb_width": bw / mid.replace(0, np.nan),
        "bb_pct":   (s - (mid - n * std)) / bw.replace(0, np.nan),
    })


def _roc(s: pd.Series, period: int = 10) -> pd.Series:
    return (s - s.shift(period)) / s.shift(period).replace(0, np.nan) * 100


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Feature builders
# ═════════════════════════════════════════════════════════════════════════════

def build_trend(df: pd.DataFrame, p: str) -> pd.DataFrame:
    c   = df[f"{p}_close"]
    out = pd.DataFrame(index=df.index)
    for span in [9, 21, 50, 100, 200]:
        out[f"{p}_ema{span}"] = _ema(c, span)
    for span in [21, 50, 200]:
        out[f"{p}_dist_ema{span}"] = (c - out[f"{p}_ema{span}"]) / out[f"{p}_ema{span}"].replace(0, np.nan)
    out[f"{p}_above_ema21"]  = (c > out[f"{p}_ema21"]).astype(int)
    out[f"{p}_above_ema50"]  = (c > out[f"{p}_ema50"]).astype(int)
    out[f"{p}_above_ema200"] = (c > out[f"{p}_ema200"]).astype(int)
    return out


def build_momentum(df: pd.DataFrame, p: str) -> pd.DataFrame:
    c   = df[f"{p}_close"]
    out = pd.DataFrame(index=df.index)
    out[f"{p}_rsi7"]       = _rsi(c, 7)
    out[f"{p}_rsi14"]      = _rsi(c, 14)
    out[f"{p}_rsi_slope"]  = out[f"{p}_rsi14"].diff(3)
    out[f"{p}_roc5"]       = _roc(c, 5)
    out[f"{p}_roc10"]      = _roc(c, 10)
    out[f"{p}_momentum5"]  = c - c.shift(5)
    out[f"{p}_momentum10"] = c - c.shift(10)
    for col, val in _macd(c).items():
        out[f"{p}_{col}"] = val
    return out


def build_volatility(df: pd.DataFrame, p: str) -> pd.DataFrame:
    h, l, c = df[f"{p}_high"], df[f"{p}_low"], df[f"{p}_close"]
    out = pd.DataFrame(index=df.index)
    atr = _atr(h, l, c, 14)
    out[f"{p}_atr14"]      = atr
    out[f"{p}_atr_norm"]   = atr / c.replace(0, np.nan)
    out[f"{p}_roll_std12"] = c.rolling(12).std(ddof=0)
    out[f"{p}_roll_std24"] = c.rolling(24).std(ddof=0)
    for col, val in _bollinger(c).items():
        out[f"{p}_{col}"] = val
    return out


def build_strength(df: pd.DataFrame, p: str) -> pd.DataFrame:
    h, l, c = df[f"{p}_high"], df[f"{p}_low"], df[f"{p}_close"]
    out = pd.DataFrame(index=df.index)
    for col, val in _adx(h, l, c, 14).items():
        out[f"{p}_{col}"] = val
    return out


def build_price_action(df: pd.DataFrame, p: str) -> pd.DataFrame:
    o, h, l, c = df[f"{p}_open"], df[f"{p}_high"], df[f"{p}_low"], df[f"{p}_close"]
    out = pd.DataFrame(index=df.index)

    body   = (c - o).abs()
    rng    = (h - l).replace(0, np.nan)
    hi_oc  = pd.concat([o, c], axis=1).max(axis=1)
    lo_oc  = pd.concat([o, c], axis=1).min(axis=1)
    u_wick = h - hi_oc
    l_wick = lo_oc - l

    out[f"{p}_body_size"]       = body
    out[f"{p}_body_pct"]        = body / rng
    out[f"{p}_upper_wick"]      = u_wick
    out[f"{p}_lower_wick"]      = l_wick
    out[f"{p}_wick_body_ratio"] = (u_wick + l_wick) / body.replace(0, np.nan)
    out[f"{p}_candle_range"]    = rng
    out[f"{p}_close_pos"]       = (c - l) / rng
    out[f"{p}_is_bullish"]      = (c >= o).astype(int)

    out[f"{p}_inside_bar"]  = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    out[f"{p}_outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(int)

    out[f"{p}_bull_engulf"] = (
        (o.shift(1) > c.shift(1)) & (c > o) &
        (c > o.shift(1)) & (o < c.shift(1))
    ).astype(int)
    out[f"{p}_bear_engulf"] = (
        (c.shift(1) > o.shift(1)) & (o > c) &
        (o > c.shift(1)) & (c < o.shift(1))
    ).astype(int)

    def _consec(s: pd.Series) -> pd.Series:
        res, cnt = [], 0
        for v in s:
            cnt = cnt + 1 if v else 0
            res.append(cnt)
        return pd.Series(res, index=s.index)

    out[f"{p}_consec_bull"] = _consec((c >= o).astype(int))
    out[f"{p}_consec_bear"] = _consec((c < o).astype(int))
    out[f"{p}_consec_hh"]   = _consec((h > h.shift(1)).astype(int))
    out[f"{p}_consec_ll"]   = _consec((l < l.shift(1)).astype(int))
    return out


def build_temporal(df: pd.DataFrame) -> pd.DataFrame:
    ts  = df.index
    out = pd.DataFrame(index=df.index)
    out["hour"]          = ts.hour
    out["minute"]        = ts.minute
    out["day_of_week"]   = ts.dayofweek
    out["day_of_month"]  = ts.day
    out["week_of_month"] = (ts.day - 1) // 7 + 1
    out["month"]         = ts.month

    out["hour_sin"]   = np.sin(2 * math.pi * ts.hour   / 24)
    out["hour_cos"]   = np.cos(2 * math.pi * ts.hour   / 24)
    out["minute_sin"] = np.sin(2 * math.pi * ts.minute / 60)
    out["minute_cos"] = np.cos(2 * math.pi * ts.minute / 60)
    out["dow_sin"]    = np.sin(2 * math.pi * ts.dayofweek / 5)
    out["dow_cos"]    = np.cos(2 * math.pi * ts.dayofweek / 5)
    out["month_sin"]  = np.sin(2 * math.pi * ts.month  / 12)
    out["month_cos"]  = np.cos(2 * math.pi * ts.month  / 12)

    h = ts.hour
    out["session_asia"]    = ((h >= 7)  & (h < 16)).astype(int)
    out["session_london"]  = ((h >= 15) & (h < 23)).astype(int)
    out["session_newyork"] = ((h >= 20) | (h < 4 )).astype(int)
    out["session_overlap"] = ((h >= 20) & (h < 23)).astype(int)

    out["is_friday"]    = (ts.dayofweek == 4).astype(int)
    out["is_monday"]    = (ts.dayofweek == 0).astype(int)
    out["monday_open"]  = ((ts.dayofweek == 0) & (h == 0) & (ts.minute < 10)).astype(int)
    out["is_month_end"] = ts.is_month_end.astype(int)
    out["is_qtr_end"]   = ts.is_quarter_end.astype(int)
    return out


def build_lag(df: pd.DataFrame, p: str = "gld965") -> pd.DataFrame:
    c   = df[f"{p}_close"]
    ret = c.pct_change()

    # GLD965 has no volume — use XAUUSD tick_vol as proxy if available
    vol_col = f"{p}_tick_vol"
    if vol_col not in df.columns or df[vol_col].isna().all():
        vol_col = "xauusd_tick_vol"
    v = df[vol_col] if vol_col in df.columns else pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(index=df.index)
    for lag in LAG_PERIODS:
        out[f"{p}_close_lag{lag}"]  = c.shift(lag)
        out[f"{p}_return_lag{lag}"] = ret.shift(lag)
    out[f"{p}_vol_lag1"] = v.shift(1)

    for w in ROLL_WINDOWS:
        out[f"{p}_roll_mean{w}"]  = c.rolling(w).mean()
        out[f"{p}_roll_max{w}"]   = c.rolling(w).max()
        out[f"{p}_roll_min{w}"]   = c.rolling(w).min()
        out[f"{p}_roll_range{w}"] = out[f"{p}_roll_max{w}"] - out[f"{p}_roll_min{w}"]
        out[f"{p}_return{w}"]     = ret.rolling(w).sum()
    return out


def build_intermarket(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    gld = df.get("gld965_close")
    xau = df.get("xauusd_close")
    usd = df.get("usdthb_close")

    has_gld = gld is not None and not gld.isna().all()
    has_xau = xau is not None and not xau.isna().all()
    has_usd = usd is not None and not usd.isna().all()

    if has_gld and has_xau:
        out["gld_xau_ratio"] = gld / xau.replace(0, np.nan)
        if has_usd:
            xau_thb = xau * usd
            out["xau_thb_synthetic"] = xau_thb
            out["gold_premium"]      = gld - xau_thb
            out["gold_premium_pct"]  = out["gold_premium"] / xau_thb.replace(0, np.nan)
        for w in CORR_WINDOWS:
            out[f"corr_gld_xau_{w}"] = gld.rolling(w).corr(xau)

    if has_gld and has_usd:
        for w in CORR_WINDOWS:
            out[f"corr_gld_usdthb_{w}"] = gld.rolling(w).corr(usd)

    if has_xau and has_usd:
        out["xauusd_x_usdthb"] = xau * usd

    # Spread change — USDTHB and XAUUSD have spread; GLD965 does not
    for sym in ["xauusd", "usdthb"]:
        sc = f"{sym}_spread"
        if sc in df.columns and df[sc].notna().any():
            out[f"{sym}_spread_chg"] = df[sc].diff()
            out[f"{sym}_spread_pct"] = df[sc] / df[f"{sym}_close"].replace(0, np.nan) * 100

    # Volume divergence: XAUUSD tick_vol vs USDTHB tick_vol
    xv = df.get("xauusd_tick_vol")
    uv = df.get("usdthb_tick_vol")
    if xv is not None and uv is not None and not xv.isna().all() and not uv.isna().all():
        xv_n = xv / xv.rolling(12).mean().replace(0, np.nan)
        uv_n = uv / uv.rolling(12).mean().replace(0, np.nan)
        out["xau_usdthb_vol_divergence"] = xv_n - uv_n

    return out


def build_returns(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for sym in symbols:
        col = f"{sym.lower()}_close"
        if col in df.columns:
            out[f"{sym.lower()}_ret1"]    = df[col].pct_change(1)
            out[f"{sym.lower()}_logret1"] = np.log(df[col] / df[col].shift(1))
            out[f"{sym.lower()}_ret3"]    = df[col].pct_change(3)
            out[f"{sym.lower()}_ret12"]   = df[col].pct_change(12)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Master pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> pd.DataFrame:
    log.info("━" * 65)
    log.info("GOLD XGBOOST TRADER — Feature Engineering Pipeline v1.1")
    log.info("━" * 65)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    log.info("STEP 1 │ Loading raw CSV files …")
    frames = load_all_symbols(RAW_DIR)
    found  = list(frames.keys())
    log.info(f"Symbols loaded: {found}")

    # ── 2. Align ─────────────────────────────────────────────────────────────
    log.info("STEP 2 │ Aligning on GLD965 timestamp …")
    base = align_symbols(frames)

    # ── 3. Feature blocks ────────────────────────────────────────────────────
    log.info("STEP 3 │ Building feature blocks …")
    blocks: List[pd.DataFrame] = [base]

    primary = [s for s in PRIORITY_SYMBOLS if s in found]
    for sym in primary:
        p = sym.lower()
        if f"{p}_close" not in base.columns:
            continue
        log.info(f"  [{sym}] trend / momentum / volatility / strength / price-action")
        blocks += [
            build_trend(base, p),
            build_momentum(base, p),
            build_volatility(base, p),
            build_strength(base, p),
            build_price_action(base, p),
        ]

    log.info("  [ALL]  temporal features")
    blocks.append(build_temporal(base))

    log.info("  [GLD965] lag & rolling features")
    blocks.append(build_lag(base, "gld965"))

    log.info("  [ALL]  intermarket features")
    blocks.append(build_intermarket(base))

    log.info("  [ALL]  return features")
    blocks.append(build_returns(base, found))

    # ── 4. Concatenate ───────────────────────────────────────────────────────
    log.info("STEP 4 │ Concatenating …")
    df = pd.concat(blocks, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # ── 5. NaN handling ───────────────────────────────────────────────────────
    log.info("STEP 5 │ NaN handling …")
    n_before = len(df)
    df = df.iloc[200:].copy()   # drop EMA-200 warm-up
    df = df.ffill(limit=3)

    for col in df.columns:
        if df[col].isna().any():
            if any(x in col for x in ["vol", "spread", "corr", "diverge"]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())

    log.info(f"  Warm-up dropped: {n_before - len(df)} bars | Remaining: {len(df):,} rows")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    log.info("STEP 6 │ Saving parquet …")
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy")
    size_mb = OUTPUT_FILE.stat().st_size / 1_048_576
    log.info(f"  → {OUTPUT_FILE}  ({size_mb:.1f} MB)")

    # ── 7. Report ─────────────────────────────────────────────────────────────
    log.info("STEP 7 │ Generating report & heatmap …")
    _report(df)
    _heatmap(df)

    log.info("━" * 65)
    log.info(f"DONE  │ {len(df):,} rows × {len(df.columns)} features")
    log.info("━" * 65)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Report & heatmap
# ═════════════════════════════════════════════════════════════════════════════

def _report(df: pd.DataFrame) -> None:
    import json

    feat_list = df.columns.tolist()
    null_pct  = (df.isnull().sum() / len(df) * 100).round(2)
    nan_cols  = null_pct[null_pct > 0].to_dict()

    summary = {
        "row_count":     len(df),
        "feature_count": len(feat_list),
        "date_range":    {"start": str(df.index.min()), "end": str(df.index.max())},
        "nan_summary":   nan_cols,
    }
    (ARTIFACT_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (ARTIFACT_DIR / "feature_list.txt").write_text("\n".join(feat_list))

    print("\n" + "─" * 62)
    print("  FEATURE DATASET SUMMARY")
    print("─" * 62)
    print(f"  Rows          : {len(df):,}")
    print(f"  Features      : {len(feat_list)}")
    print(f"  Date range    : {df.index.min()}  →  {df.index.max()}")
    total_nan = df.isnull().sum().sum()
    print(f"  Total NaN     : {total_nan}  {'✓ Clean' if total_nan == 0 else '⚠ check nan_summary'}")

    groups = {
        "OHLCV (GLD965)":  lambda c: c.startswith("gld965_") and any(x in c for x in ["open","high","low","close","vol","spread"]),
        "OHLCV (XAUUSD)":  lambda c: c.startswith("xauusd_") and any(x in c for x in ["open","high","low","close","vol","spread"]),
        "OHLCV (USDTHB)":  lambda c: c.startswith("usdthb_") and any(x in c for x in ["open","high","low","close","vol","spread"]),
        "Trend / EMA":      lambda c: "ema" in c or "dist_ema" in c or "above_ema" in c,
        "Momentum":         lambda c: any(x in c for x in ["rsi","roc","macd","momentum"]),
        "Volatility":       lambda c: any(x in c for x in ["atr","bb_","roll_std"]),
        "Strength (ADX)":  lambda c: any(x in c for x in ["adx","di_plus","di_minus"]),
        "Price Action":     lambda c: any(x in c for x in ["body","wick","engulf","inside","outside","consec","close_pos","is_bullish","candle_range"]),
        "Temporal":         lambda c: any(x in c for x in ["hour","minute","day","week","month","session","friday","monday","qtr"]),
        "Lag / Rolling":    lambda c: any(x in c for x in ["lag","roll_mean","roll_max","roll_min","roll_range","return3","return6","return12"]),
        "Intermarket":      lambda c: any(x in c for x in ["premium","ratio","corr_","synthetic","divergence","xauusd_x","spread_chg","spread_pct"]),
        "Returns":          lambda c: any(x in c for x in ["_ret1","_ret3","_ret12","_logret"]),
    }
    print("\n  Feature Groups:")
    cols = df.columns.tolist()
    for g, fn in groups.items():
        n = sum(1 for c in cols if fn(c))
        print(f"    {g:<24}  {n:>3}")
    print(f"    {'TOTAL':<24}  {len(cols):>3}")
    print("─" * 62 + "\n")


def _heatmap(df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        target = "gld965_close"
        if target not in df.columns:
            return
        top = (df.corr()[target]
                  .drop(target)
                  .abs()
                  .sort_values(ascending=False)
                  .head(30)
                  .index.tolist())
        sub = df[[target] + top]

        fig, ax = plt.subplots(figsize=(18, 14))
        sns.heatmap(sub.corr(), cmap="RdYlGn", center=0, annot=False, linewidths=0.3, ax=ax)
        ax.set_title("Correlation Heatmap — Top 30 Features vs GLD965 Close", fontsize=13, pad=10)
        plt.tight_layout()
        out_path = ARTIFACT_DIR / "correlation_heatmap.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        log.info(f"  Heatmap → {out_path}")
    except Exception as e:
        log.warning(f"Heatmap skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    run_pipeline()