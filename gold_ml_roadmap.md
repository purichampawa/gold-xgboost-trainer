# 🔬 Gold Trading ML — Label Engineering & Roadmap
### วิเคราะห์ปัญหา → แก้ไข → แผนอนาคต (Ensemble + News AI)

---

## สรุปผลการวิเคราะห์ไฟล์จริง

หลังจากอ่านโค้ดและ dataset ของคุณ พบปัญหาสำคัญ **4 จุด** ใน `label_engineering.py` เดิม:

---

## 🔴 ปัญหาที่พบ (v1) + หลักฐาน

### ปัญหาที่ 1 — TP ต่ำกว่า Spread (ร้ายแรงที่สุด)

```
TP เดิม   = 0.25% net
Spread RT = 0.14% × 2 = 0.28%  ← round-trip spread

TP gross  = 0.25% + 0.28% = 0.53%  → ต้องวิ่ง ~18 USD
ATR_14 mean = 3.18 USD → ต้อง 4.3× ATR ภายใน 48 แท่ง

ผลลัพธ์: label เดิมหลายตัวที่เรียกว่า BUY จริงๆ ขาดทุนหลัง spread
```

### ปัญหาที่ 2 — Path ไม่ถูกต้อง (argmax bug)

```python
# โค้ดเดิม (ผิด):
buy_hit_tp_idx = np.argmax(future_highs >= buy_tp_price)  # ❌
buy_hit_sl_idx = np.argmax(future_lows  <= buy_sl_price)  # ❌

# ปัญหา: ถ้าทั้งสองเป็น False ทั้งหมด argmax คืน 0
# แปลว่าเหมือนชนทั้ง TP และ SL ที่ bar แรก → ผลผิดพลาด

# โค้ดใหม่ (ถูก):
for j in range(i+1, i+horizon+1):
    if bar_low <= buy_sl: buy_result = 'SL'; break-candidate
    if bar_high >= buy_tp: buy_result = 'TP'; break-candidate
```

### ปัญหาที่ 3 — BUY cluster 77.7% (ไม่ตรงความเป็นจริง)

```
ข้อจำกัด portfolio: เงิน 1500 THB, min_order = 1000 THB
→ เปิดได้ 1 ไม้เท่านั้น (เหลือ 500 ซึ่ง < min_order)
→ แต่ BUY ติดกัน 77% = โมเดลเรียนรู้ pattern ที่ทำไม่ได้จริง
```

### ปัญหาที่ 4 — Session gate ไม่ตรงกับ session_gate.py

```python
# label_engineering.py เดิม กรองแค่:
if 5 <= current_hour < 8: labels.append('HOLD')

# แต่ session_gate.py กำหนด weekday: 06:15 – 02:00
# ช่องว่าง: 02:00–05:00 ไม่ถูกกรอง → signal นอก session เข้าโมเดล
```

---

## ✅ การแก้ไข (v2) + ผลลัพธ์จริง

### พารามิเตอร์ใหม่ (จาก grid search บน dataset จริง)

| ค่า | เดิม | ใหม่ | เหตุผล |
|-----|------|------|--------|
| TP net | 0.25% | **0.20%** | ต่ำลงเพื่อ hit TP ได้ง่ายขึ้น |
| SL net | 0.25% | **0.25%** | คงเดิม (กว้างกว่า TP = RR=0.8) |
| Horizon | 48 bars | **72 bars (6h)** | ให้เวลา TP วิ่งได้จริง |
| Spread RT | ไม่รวม | **0.28% (0.14%×2)** | รวมต้นทุนจริง |
| Session gate | ไม่ครบ | **ครบทุก session** | ตรงกับ session_gate.py |
| Path method | argmax (ผิด) | **bar-by-bar loop** | accurate 100% |

### Label distribution เปรียบเทียบ

```
เดิม (v1):  BUY=16.7%  SELL=13.9%  HOLD=69.4%  ← conservative เกินไป
ใหม่ (v2):  BUY=25.4%  SELL=21.1%  HOLD=53.4%  ← sensitive และสมดุลกว่า
```

### Quality validation (ผ่านทุก check)

```
✅ BUY  mean future_return: +0.435%  (ทิศทางถูก)
✅ SELL mean future_return: -0.343%  (ทิศทางถูก)
✅ HOLD mean future_return: +0.049%  (ใกล้ศูนย์)
✅ ไม่มีสัญญาณนอก session
```

### 2 คอลัมน์ใหม่ใน output

```
Signal        → raw label สำหรับเทรน XGBoost (sensitive, ครอบคลุมทุก pattern)
Signal_Entry  → squashed label สำหรับ backtest (1 entry ต่อกลุ่ม)
```

---

## 🔧 วิธีใช้ label_engineering.py v2

```bash
# รัน label ใหม่พร้อม validate
python label_engineering.py \
  --input  gold_features_cleaned.csv \
  --output gold_features_labeled.csv \
  --validate

# ปรับพารามิเตอร์ (ถ้าต้องการ sensitive มากขึ้น)
python label_engineering.py --tp 0.0018 --sl 0.0025 --horizon 72

# ปรับ conservative มากขึ้น (ถ้า backtest ผ่านแต่ live ไม่ผ่าน)
python label_engineering.py --tp 0.0025 --sl 0.0025 --horizon 96
```

### คำอธิบาย RR = 0.8

```
RR = TP / SL = 0.20% / 0.25% = 0.80

Expectancy formula:
  E = (WinRate × TP) - (LossRate × SL)
  E > 0 เมื่อ WinRate > SL/(TP+SL) = 0.25/(0.20+0.25) = 55.6%

ถ้า XGBoost ได้ WinRate = 60%:
  E = (0.60 × 0.20%) - (0.40 × 0.25%) = 0.12% - 0.10% = +0.02% per trade
  → บวกแม้ RR < 1.0 ✅
```

---

## 📋 ลำดับขั้นตอนหลังจากนี้ (ทำตามลำดับ)

### Phase 1 — ปรับ Label + เทรน Baseline (ทำได้เดี๋ยวนี้)

```bash
# 1. รัน label engine ใหม่
python label_engineering.py --input gold_features_cleaned.csv --validate

# 2. เทรน XGBoost
python train.py --model-type xgboost

# 3. Backtest ทันที (อย่าข้าม)
python backtest.py

# 4. ดูผล
cat outputs/backtests/backtest_summary.json
```

**สิ่งที่ต้องแก้ใน `config.py` ด้วย:**

```python
# แก้ label config ให้ตรงกับ label_engineering.py v2
CONFIG.labels.horizon_bars    = 6       # ใช้แค่ build_labels ใน train.py
CONFIG.labels.threshold_buy   = 0.0020  # TP net
CONFIG.labels.threshold_sell  = -0.0020
CONFIG.labels.spread_buffer   = 0.0014  # spread entry

# Signal config: ให้ model กล้า emit สัญญาณมากขึ้น
CONFIG.signals.threshold_buy   = 0.40
CONFIG.signals.threshold_sell  = 0.40
CONFIG.signals.confidence_filter = 0.38
CONFIG.signals.hold_zone       = 0.05

# Data: ชี้ไปที่ column ใหม่
CONFIG.data.raw_signal_col = "Signal"   # raw label สำหรับ train
```

### Phase 2 — Feature Engineering (เพิ่ม sensitivity)

เพิ่ม features ที่ช่วยจับ local high/low ซึ่งเป็นจุดเทรดทองที่ดีที่สุด:

```python
# เพิ่มใน preprocessing script หรือ train.py

# 1. Swing high/low detector
df['swing_high_5'] = (df['High'] == df['High'].rolling(5, center=True).max()).astype(int)
df['swing_low_5']  = (df['Low']  == df['Low'].rolling(5, center=True).min()).astype(int)

# 2. Distance from recent high/low (normalized)
df['dist_from_20h_high'] = (df['High'].rolling(240).max() - df['Close']) / df['Close']
df['dist_from_20h_low']  = (df['Close'] - df['Low'].rolling(240).min()) / df['Close']

# 3. ATR-normalized price position
df['bb_pos'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()

# 4. Session features (สำคัญสำหรับทอง)
df['is_london']    = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
df['is_ny']        = ((df['hour'] >= 13) & (df['hour'] < 20)).astype(int)
df['is_overlap']   = ((df['hour'] >= 13) & (df['hour'] < 18)).astype(int)
df['is_asia']      = ((df['hour'] >= 1) & (df['hour'] < 6)).astype(int)

# 5. Momentum multi-timeframe
df['mom_1h']  = df['Close'].pct_change(12)   # 1 ชั่วโมง (12 × M5)
df['mom_4h']  = df['Close'].pct_change(48)   # 4 ชั่วโมง
df['mom_1d']  = df['Close'].pct_change(288)  # 1 วัน

# 6. Volatility regime
df['vol_regime'] = (df['ATR_14'] / df['ATR_14'].rolling(288).mean())  # ATR สัมพัทธ์
```

### Phase 3 — Finetune (หลัง baseline stable)

```bash
# รัน 80 trials (เพิ่มจาก 40)
python finetune.py --n-trials 80

# ดูผล
cat artifacts/finetune_results.json

# Apply best params → config.py → train → backtest
python train.py && python backtest.py
```

**ตัวแปรที่ finetune ควรเน้น:**
- `label_horizon` (2–24) — ตัวสำคัญที่สุด
- `threshold_buy/sell` (0.30–0.65) — ความมั่นใจก่อน emit สัญญาณ
- `hold_zone` (0.02–0.20) — dead zone กลาง

---

## 🚀 แผนอนาคต — Ensemble + News AI

### สถาปัตยกรรม Multi-Model System

```
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL AGGREGATOR                        │
│           (weighted vote by session + confidence)           │
└──────────┬──────────────────────────┬──────────────────────┘
           │                          │
    ┌──────▼──────┐           ┌───────▼──────┐
    │ QUANT MODEL │           │  NEWS MODEL  │
    │  (XGBoost)  │           │  (LLM/BERT)  │
    │             │           │              │
    │ Input:      │           │ Input:       │
    │ - OHLCV     │           │ - News feed  │
    │ - EMAs      │           │ - Sentiment  │
    │ - ATR       │           │ - Keywords   │
    │ - RSI/MACD  │           │ - Source     │
    │ - Session   │           │   weight     │
    │ - Momentum  │           │              │
    │             │           │ Output:      │
    │ Output:     │           │ - Direction  │
    │ - P(BUY)    │           │ - Magnitude  │
    │ - P(HOLD)   │           │ - Urgency    │
    │ - P(SELL)   │           │              │
    └─────────────┘           └──────────────┘
```

### Session-Weighted Aggregation

```python
# น้ำหนักโมเดลต่างกันตาม session
SESSION_WEIGHTS = {
    "london_open":  {"quant": 0.55, "news": 0.45},  # ข่าวมีผลมาก
    "ny_open":      {"quant": 0.50, "news": 0.50},  # ข่าว US มีผลสูงสุด
    "overlap":      {"quant": 0.45, "news": 0.55},  # Volatility สูง ข่าวครอง
    "asia":         {"quant": 0.75, "news": 0.25},  # ตลาดเงียบ quant ดีกว่า
    "weekend":      {"quant": 0.80, "news": 0.20},  # ข่าวน้อย
}

def aggregate_signals(quant_probs, news_signal, session):
    w = SESSION_WEIGHTS[session]
    combined_buy  = quant_probs['BUY']  * w['quant'] + news_signal['BUY']  * w['news']
    combined_sell = quant_probs['SELL'] * w['quant'] + news_signal['SELL'] * w['news']
    # → ส่งไป signal engine เพื่อตัดสินใจ BUY/HOLD/SELL
```

### React Loop สำหรับ News Analysis

```
[News Feed API]
      ↓
[LLM (Claude API)]
  Prompt: "วิเคราะห์ข่าวนี้ต่อราคาทอง
           ทิศทาง: UP/DOWN/NEUTRAL
           ความมั่นใจ: 0-1
           ระยะเวลาผล: SHORT/MEDIUM/LONG"
      ↓
[News Signal: direction + confidence + urgency]
      ↓
[Weighted Combiner] ← Quant Signal
      ↓
[Final Signal: BUY/HOLD/SELL]
      ↓
[Execute / Alert]
```

**ตัวอย่าง News Prompt Template:**

```python
NEWS_PROMPT = """
คุณเป็น analyst ทองคำ วิเคราะห์ข่าวต่อไปนี้:

ข่าว: {news_text}
เวลา: {timestamp}
Session ปัจจุบัน: {session_name}

ตอบใน JSON เท่านั้น:
{{
  "direction": "UP" | "DOWN" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "duration": "SHORT" | "MEDIUM" | "LONG",
  "reason": "สรุปสั้นๆ 1 ประโยค"
}}
"""
```

### Timeline การพัฒนา

```
สัปดาห์นี้:
  ✅ label_engineering.py v2 (เสร็จแล้ว)
  → เทรน XGBoost baseline
  → backtest และ validate

สัปดาห์ 2-3:
  → เพิ่ม features (swing high/low, momentum MTF, session)
  → finetune Optuna 80 trials
  → benchmark XGBoost vs LightGBM

เดือน 2:
  → สร้าง News pipeline (API + LLM prompt)
  → ออกแบบ weighted combiner
  → backtest Ensemble vs Quant-only

เดือน 3:
  → Paper trading (ไม่ใช้เงินจริง)
  → ติดตาม expectancy จริงเทียบ backtest
  → ปรับ session weights จากผลจริง

เดือน 4+:
  → Live trading ด้วยทุนจริงขั้นต่ำ
  → Monitor และ retrain รายเดือน
```

---

## ⚠️ กฎที่ต้องจำสำหรับ Portfolio นี้

```
ทุน: 1500 THB | Min order: 1000 THB | Blowup: ขาดทุน > 500 THB

กฎเหล็ก:
  1. เปิด 1 ไม้ต่อครั้งเท่านั้น (เงินไม่พอเปิด 2)
  2. ถ้า trade ยังเปิดอยู่ → ไม่รับ signal ใหม่จากโมเดล
  3. ขาดทุนรวม > 500 THB → หยุดทันที (blowup protection)
  4. ทุกสัญญาณจาก AI เชื่อ 100% ตาม config → ไม่ override ด้วยมือ

Backtest ต้องใช้ Signal_Entry (not Signal) เพื่อให้ได้ผลสมจริง
```

---

*label_engineering.py v2 — แก้ไขปัญหา 4 จุด, ผ่าน quality validation ทุก check*
*dataset: 56,985 rows (2025-01-02 → 2025-12-31), M5 timeframe, ทองคำ*