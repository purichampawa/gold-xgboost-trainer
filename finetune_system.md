# 📘 คู่มือการใช้งาน ML Trading System สำหรับซื้อขายทอง
### สัญญาณ BUY / HOLD / SELL — ฉบับปฏิบัติจริง

---

## สารบัญ

1. [ภาพรวมระบบ](#1-ภาพรวมระบบ)
2. [วงรอบการทำงานประจำวัน](#2-วงรอบการทำงานประจำวัน-daily-loop)
3. [ไฟล์ในระบบ — หน้าที่และลำดับการรัน](#3-ไฟล์ในระบบ--หน้าที่และลำดับการรัน)
4. [การตั้งค่า config.py](#4-การตั้งค่า-configpy)
5. [การเทรน Train Baseline](#5-การเทรน-train-baseline)
6. [การ Finetune ด้วย Optuna](#6-การ-finetune-ด้วย-optuna)
7. [การรัน Backtest](#7-การรัน-backtest)
8. [การอ่านผลลัพธ์ Backtest](#8-การอ่านผลลัพธ์-backtest)
9. [ลำดับความสำคัญในการ Tune](#9-ลำดับความสำคัญในการ-tune)
10. [สัญญาณอันตราย — ต้องระวัง](#10-สัญญาณอันตราย--ต้องระวัง)
11. [ตัวอย่าง Experiment Log](#11-ตัวอย่าง-experiment-log)
12. [Mindset นักวิจัยเชิงปริมาณ](#12-mindset-นักวิจัยเชิงปริมาณ)

---

## 1. ภาพรวมระบบ

ระบบนี้เป็น **ML-based trading system** สำหรับสร้างสัญญาณ **BUY / HOLD / SELL** บนทอง (XAUUSD หรือสัญลักษณ์ทองอื่น)

```
ข้อมูลราคาทอง (CSV)
        ↓
   config.py  ← ศูนย์กลางทุกพารามิเตอร์
        ↓
   train.py   ← สร้างโมเดล + บันทึก artifacts
        ↓
  backtest.py ← ทดสอบในตลาดจำลอง
        ↓
  finetune.py ← ค้นหาพารามิเตอร์ที่ดีที่สุด (Optuna)
        ↓
   ผลลัพธ์: equity_curve.csv, backtest_summary.json
```

---

## 2. วงรอบการทำงานประจำวัน (Daily Loop)

ทำตามขั้นตอนนี้ **ทุกวันที่ทดสอบ** — อย่าข้ามขั้น

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1 │ แช่แข็งข้อมูล                                 │
│         │ บันทึก CSV ด้วยชื่อวันที่                      │
│         │ เช่น gold_features_2026-04-25.csv              │
├─────────────────────────────────────────────────────────┤
│  STEP 2 │ ตั้ง Hypothesis ก่อนรัน                        │
│         │ ตัวอย่าง: "เพิ่ม horizon_bars จะลด noise หรือไม่" │
│         │ เปลี่ยนแค่ 1–3 ตัวแปรต่อรอบ                   │
├─────────────────────────────────────────────────────────┤
│  STEP 3 │ แก้ config.py                                  │
│         │ (labels, broker, session, risk)                │
├─────────────────────────────────────────────────────────┤
│  STEP 4 │ python train.py                                │
├─────────────────────────────────────────────────────────┤
│  STEP 5 │ python backtest.py                             │
├─────────────────────────────────────────────────────────┤
│  STEP 6 │ วิเคราะห์ผล: expectancy, drawdown, trade count │
├─────────────────────────────────────────────────────────┤
│  STEP 7 │ บันทึก Experiment Log                         │
│         │ (เปลี่ยนอะไร → ผลเป็นอย่างไร)                  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. ไฟล์ในระบบ — หน้าที่และลำดับการรัน

| ไฟล์ | หน้าที่ | รันตรง? |
|------|---------|---------|
| `config.py` | ศูนย์กลางทุก parameter — แก้ตรงนี้ก่อนเสมอ | ❌ (แก้ค่า) |
| `train.py` | โหลดข้อมูล → สร้าง label → เทรนโมเดล → บันทึก artifacts | ✅ |
| `backtest.py` | จำลองการเทรดพร้อม spread/session/risk แบบสมจริง | ✅ |
| `finetune.py` | ค้นหา hyperparameter ด้วย Optuna | ✅ |
| `signals.py` | แปลง probability → BUY/HOLD/SELL | ❌ (ใช้โดย train/backtest) |
| `risk.py` | คำนวณ position size, blowup check, stop logic | ❌ (ใช้โดย backtest) |
| `session_gate.py` | ตรวจว่าตลาดเปิดหรือไม่ | ❌ (ใช้โดย backtest) |

**Artifacts ที่ระบบบันทึก (โฟลเดอร์ `artifacts/`):**

```
artifacts/
├── model.pkl                  ← โมเดลที่เทรนแล้ว
├── feature_columns.json       ← รายชื่อ features
├── train_metrics.json         ← ผล metrics ตอนเทรน
├── backtest_trades.csv        ← ทุก trade ที่เกิดขึ้น
├── equity_curve.csv           ← กราฟ equity
└── backtest_summary.json      ← สรุปผล backtest
```

---

## 4. การตั้งค่า `config.py`

### 4.1 ส่วน Broker — ความสมจริงของต้นทุน

```python
CONFIG.broker = {
    "starting_capital_thb": 100000,   # ทุนเริ่มต้น (บาท)
    "min_order_size_thb":   500,      # ขนาด order ขั้นต่ำ
    "spread_rate_entry":    0.0003,   # ค่า spread ตอนเข้า (0.03%)
    "spread_rate_exit":     0.0003,   # ค่า spread ตอนออก
    "spread_buffer":        0.0001,   # buffer เผื่อ slippage
}
```

> ⚠️ **กฎเหล็ก:** `spread_rate_entry + spread_rate_exit` คือต้นทุนขั้นต่ำที่โมเดลต้อง **ชนะ** ก่อนจะเป็น +EV

### 4.2 ส่วน Labels — นิยาม "การเทรดที่ดี"

```python
CONFIG.labels = {
    "horizon_bars":    10,     # มองไปข้างหน้ากี่แท่งเทียน
    "threshold_buy":   0.003,  # ขึ้น 0.3% = BUY
    "threshold_sell":  0.003,  # ลง 0.3% = SELL
    # ไม่ถึงเกณฑ์ทั้งสอง = HOLD
}
```

> 💡 `horizon_bars` คือตัวแปรที่มีผลมากที่สุด — ปรับตัวนี้ก่อน

### 4.3 ส่วน Signals — ความมั่นใจก่อนส่งสัญญาณ

```python
CONFIG.signals = {
    "threshold_buy":      0.60,  # ต้องมั่นใจ ≥60% จึง BUY
    "threshold_sell":     0.60,  # ต้องมั่นใจ ≥60% จึง SELL
    "hold_zone":          0.10,  # zone กลาง ±10% รอบ 50%
    "confidence_filter":  0.55,  # กรองสัญญาณอ่อน
}
```

### 4.4 ส่วน Session — เวลาเทรดที่อนุญาต

```python
CONFIG.session = {
    "deny_new_entries_outside_session": True,
    "force_close_at_session_end":       True,
    "allow_carry_overnight":            False,
}
```

> ⚠️ ถ้า backtest อนุญาต carry overnight แต่ในชีวิตจริงไม่ได้ทำ = ผล backtest ไม่น่าเชื่อถือ

### 4.5 ส่วน Risk — การจัดการความเสี่ยง

```python
CONFIG.risk = {
    "max_daily_loss_pct":          0.05,   # หยุดเทรดถ้าขาดทุนเกิน 5%/วัน
    "max_consecutive_losses":      5,      # หยุดถ้าแพ้ติดต่อกัน 5 ครั้ง
    "position_size_pct":           0.10,   # ใช้ 10% ของทุนต่อ trade
}
```

---

## 5. การเทรน Train Baseline

### คำสั่งพื้นฐาน

```bash
# เทรนด้วยโมเดล default (ที่ตั้งใน config.py)
python train.py

# ระบุประเภทโมเดลโดยตรง
python train.py --model-type xgboost
python train.py --model-type lightgbm
python train.py --model-type randomforest
```

### สิ่งที่ `train.py` ทำ

1. โหลด CSV จาก path ใน config
2. สร้าง label จาก future return (BUY/HOLD/SELL)
3. ตรวจหา numeric features อัตโนมัติ
4. แบ่ง train/val/test ตาม timeline (ไม่ shuffle!)
5. เทรนโมเดล → บันทึก `artifacts/model.pkl`
6. บันทึก feature list และ train metrics

### ✅ ตรวจสอบหลังเทรน

```bash
cat artifacts/train_metrics.json
```

ดูค่าเหล่านี้:
- `accuracy` — ความแม่นยำ (ระวัง: สูงไม่ได้แปลว่าดี)
- `f1_macro` — ความสมดุลระหว่าง class
- `class_distribution` — สัดส่วน BUY/HOLD/SELL

---

## 6. การ Finetune ด้วย Optuna

### เมื่อไรควร Finetune

ทำ **หลังจาก** baseline มีเสถียรภาพแล้ว คือ:
- backtest ได้ expectancy > 0
- ไม่ blown_up
- trade count สมเหตุสมผล (ไม่น้อยหรือมากเกินไป)

### คำสั่ง

```bash
# รัน 60 trials (แนะนำ)
python finetune.py --n-trials 60

# รัน 120 trials สำหรับการค้นหาละเอียด
python finetune.py --n-trials 120
```

### ขั้นตอนหลังจาก Finetune

```bash
# 1. ดูผลที่ดีที่สุด
cat artifacts/best_params.json

# 2. นำค่าไปใส่ใน config.py ด้วยมือ
#    (อย่า auto-apply โดยไม่เข้าใจว่าค่าแต่ละตัวหมายความว่าอะไร)

# 3. เทรนใหม่ด้วยค่าที่ tuned
python train.py

# 4. Backtest เสมอ — อย่าเชื่อ Optuna score อย่างเดียว!
python backtest.py
```

> ⚠️ **กฎสำคัญ:** Optuna optimize บน validation set แต่ประสิทธิภาพที่แท้จริงต้องดูจาก **backtest บน test period** เท่านั้น

### พารามิเตอร์ที่ Optuna จะค้นหา

| กลุ่ม | ตัวอย่างพารามิเตอร์ | ผลกระทบ |
|------|-------------------|---------|
| โมเดล | `learning_rate`, `max_depth`, `n_estimators` | ความแม่นยำ |
| สัญญาณ | `threshold_buy/sell`, `hold_zone` | จำนวน trade |
| Label | `horizon_bars`, `threshold_buy/sell` | นิยาม "การเทรดดี" |

---

## 7. การรัน Backtest

### คำสั่ง

```bash
python backtest.py
```

### สิ่งที่ `backtest.py` จำลอง

- ✅ Spread ทั้งขาเข้าและขาออก
- ✅ ขนาด order ขั้นต่ำ (min_order_size)
- ✅ Session gate (ห้ามเปิด trade นอกเวลา)
- ✅ Risk gate (หยุดเมื่อขาดทุนเกินกำหนด)
- ✅ Blowup detection (ทุนหมด)
- ✅ Position sizing จาก risk config

### Loop การทดสอบโมเดลหลายแบบ

```bash
# ทดสอบ XGBoost
python train.py --model-type xgboost
python backtest.py
# บันทึกผล

# ทดสอบ LightGBM
python train.py --model-type lightgbm
python backtest.py
# เปรียบเทียบผล
```

---

## 8. การอ่านผลลัพธ์ Backtest

### ค่าหลักใน `backtest_summary.json`

| ค่า | ความหมาย | เป้าหมาย |
|-----|---------|---------|
| `total_return_pct` | ผลตอบแทนรวม | > 0 (หลังต้นทุน) |
| `expectancy` | กำไรเฉลี่ยต่อ trade | **ต้องเป็นบวก** |
| `max_drawdown_pct` | drawdown สูงสุด | ให้น้อยที่สุด |
| `sharpe` | ผลตอบแทนต่อความเสี่ยง | > 1.0 ดี, > 1.5 ดีมาก |
| `win_rate` | % trade ที่กำไร | รอง (ดู expectancy ก่อน) |
| `total_trades` | จำนวน trade ทั้งหมด | 30+ ถึงจะมีนัยสถิติ |
| `blown_up` | ทุนหมดระหว่างทาง? | ต้องเป็น False |

### การอ่าน Equity Curve

```bash
# ดู equity curve
cat artifacts/equity_curve.csv

# วิเคราะห์ด้วย Python
python -c "
import pandas as pd
df = pd.read_csv('artifacts/equity_curve.csv')
print(df.tail(20))
print('Max DD:', df['equity'].pct_change().min())
"
```

**สิ่งที่ดูใน equity curve:**
- เส้นควรขึ้นสม่ำเสมอ ไม่ขึ้นแบบ spike แล้วร่วง
- ช่วง drawdown ต้องฟื้นตัวได้ในเวลาสมเหตุสมผล
- ไม่พึ่งพา trade เดี่ยว 1-2 ตัวในการดัน return

---

## 9. ลำดับความสำคัญในการ Tune

### 🔴 High Impact — Tune ก่อน

```
1. horizon_bars (labels.horizon_bars)
   → นิยามว่า "trade ดี" คืออะไร
   → ลอง: 5, 10, 15, 20 bars

2. Label thresholds (labels.threshold_buy/sell)
   → ควบคุมความบริสุทธิ์ของ label
   → ลอง: 0.002, 0.003, 0.005

3. Spread assumptions (broker.spread_rate_*)
   → ให้ตรงกับ broker จริง
   → ถ้า tuning ด้วย spread ต่ำกว่าจริง = หลอกตัวเอง
```

### 🟡 Medium Impact — Tune หลัง

```
4. model_type (xgboost / lightgbm / randomforest)
5. Model hyperparams (learning_rate, max_depth, n_estimators)
6. Signal thresholds (signals.threshold_buy/sell)
7. hold_zone, confidence_filter
```

### 🟢 Low Impact — Tune ท้าย

```
8. random_seed (กระทบแค่ reproducibility)
9. logging settings
```

---

## 10. สัญญาณอันตราย — ต้องระวัง

### 🚨 ปฏิเสธโมเดลทันทีถ้า...

```
❌ Accuracy สูง แต่ expectancy ติดลบ
   → โมเดลทำนายได้ แต่ต้นทุน spread กินหมด

❌ blown_up = True บ่อยครั้ง
   → ขนาด position หรือ risk config ไม่เหมาะสม

❌ total_trades < 20
   → ไม่มีนัยสถิติ ผลที่ดีอาจเป็นโชค

❌ ผล backtest ดีเฉพาะช่วงสั้น 1-2 เดือน
   → อาจ curve-fit กับ regime นั้น

❌ เปลี่ยน config เล็กน้อย แล้วผลพัง
   → โมเดลไม่ robust

❌ Equity curve ขึ้นแบบ step function
   → พึ่งพา trade เดี่ยวจำนวนน้อย
```

### 🔍 Data Leakage — ตรวจสอบเสมอ

```python
# Feature ไม่ควรใช้ข้อมูลอนาคต
# ตัวอย่างที่ผิด:
feature = df['close'].shift(-5)  # ❌ ใช้ราคาอีก 5 แท่งข้างหน้า

# ตัวอย่างที่ถูก:
feature = df['close'].shift(1)   # ✅ ใช้ราคาแท่งที่ผ่านมา
```

---

## 11. ตัวอย่าง Experiment Log

บันทึกทุกการทดสอบในรูปแบบนี้ — เป็นหัวใจของ reproducibility:

```markdown
## Experiment #007 — 2026-04-25

**Hypothesis:**
เพิ่ม horizon_bars จาก 10 → 15 จะลด noise และเพิ่ม expectancy

**สิ่งที่เปลี่ยน:**
- labels.horizon_bars: 10 → 15
- ไม่มีอย่างอื่น

**ข้อมูลที่ใช้:**
- gold_features_2026-04-25.csv
- Train: 2024-01 ถึง 2025-06
- Test: 2025-07 ถึง 2026-03

**ผลลัพธ์:**
| ค่า | ก่อน | หลัง |
|-----|------|------|
| expectancy | 12.5 THB | 18.2 THB |
| total_trades | 145 | 98 |
| max_drawdown_pct | 8.2% | 6.1% |
| sharpe | 0.92 | 1.15 |
| blown_up | False | False |

**สรุป:**
✅ ยืนยัน hypothesis — horizon ยาวขึ้นช่วยจริง
→ เก็บค่า horizon_bars = 15 ไปใช้ต่อ
→ รอบถัดไป: ทดสอบ threshold_buy = 0.004
```

---

## 12. Mindset นักวิจัยเชิงปริมาณ

### กฎ 3 ข้อที่ต้องจำ

```
1. HYPOTHESIS FIRST
   ตั้งสมมติฐานก่อนรัน ไม่ใช่ดูผลแล้วค่อยแต่งเหตุผล

2. ONE CHANGE AT A TIME
   เปลี่ยนแค่ 1-3 ตัวแปรต่อรอบ
   ถ้าเปลี่ยนหลายอย่างพร้อมกัน = ไม่รู้ว่าอะไรทำให้ดีขึ้น

3. OUT-OF-SAMPLE ALWAYS
   ผล validation ดีไม่พอ
   ต้องดูผล backtest บน test period ที่ไม่เคยเห็น
```

### ความผิดพลาดที่พบบ่อย

| ความผิดพลาด | ผลที่ตามมา | วิธีหลีกเลี่ยง |
|-------------|-----------|--------------|
| Test บน spread ต่ำกว่าจริง | Backtest ดีกว่าการเทรดจริงมาก | ใช้ spread จาก broker จริง |
| ไม่แช่แข็ง dataset | ผลเปรียบเทียบ experiment ไม่น่าเชื่อถือ | ตั้งชื่อไฟล์ด้วยวันที่ |
| Tune บน test set | Overfitting ที่ไม่รู้ตัว | แยก validation/test อย่างเด็ดขาด |
| เพิ่มทุน backtest เพื่อหนี blowup | ซ่อนปัญหาจริง | แก้ที่ logic ไม่ใช่เพิ่มทุน |
| Model shopping ก่อนแก้ data | โมเดลเก่งไม่ช่วยถ้า data/label แย่ | แก้ label quality ก่อน |

---

## คำสั่งสรุป — Quick Reference

```bash
# ═══ วงรอบปกติ ═══
python train.py
python backtest.py

# ═══ วงรอบ Finetune ═══
python finetune.py --n-trials 60
# → แก้ config.py ด้วยมือ
python train.py
python backtest.py

# ═══ ทดสอบหลายโมเดล ═══
python train.py --model-type xgboost && python backtest.py
python train.py --model-type lightgbm && python backtest.py

# ═══ ดูผลลัพธ์ ═══
cat artifacts/backtest_summary.json
cat artifacts/train_metrics.json
```

---

> **เป้าหมายสูงสุด:**
> ไม่ใช่ "หา backtest ที่ดีที่สุด"
> แต่คือ **สร้างกระบวนการที่ตรวจสอบได้, สมจริง, และโกงตัวเองได้ยาก**

---

*คู่มือนี้ใช้คู่กับไฟล์: `config.py`, `train.py`, `finetune.py`, `backtest.py`, `signals.py`, `risk.py`, `session_gate.py`*