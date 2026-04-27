import pandas as pd
import joblib
from old_patch2.session_gate import SessionGate
from patch3.risk import RiskManager

# โหลดระบบ
df = pd.read_csv('gold_features_cleaned.csv')
model = joblib.load('gold_ml_model.pkl')
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
X_test = df_test.drop(columns=['Timestamp', 'Signal'])

probabilities = model.predict_proba(X_test)
df_test['Prob_BUY'] = probabilities[:, 1]
df_test['Prob_SELL'] = probabilities[:, 2]

gate = SessionGate(is_mt5_data=False) 
risk = RiskManager(initial_capital=1500.0, spread_pct=0.0014)

# 🔥 พารามิเตอร์ที่ผ่านการวิเคราะห์มาแล้ว
CONFIDENCE_THRESHOLD = 0.95  # คัดเน้นๆ เฉพาะที่มั่นใจสุดๆ
NEWS_FILTER = 0.15           # กรองข่าวให้เข้มขึ้น
ATR_MULTIPLIER_TP = 4.0      # รอกินคำใหญ่ขึ้น (4 เท่าของ ATR)
ATR_MULTIPLIER_SL = 2.5      # เผื่อระยะให้กราฟหายใจ (2.5 เท่าของ ATR)

for i in range(len(df_test)):
    row = df_test.iloc[i]
    ts, px, atr, news = row['Timestamp'], row['Close'], row['ATR_14'], row['News_Sentiment']
    
    is_open, _ = gate.check_session(ts)
    if not is_open or risk.is_game_over: continue
    
    signal = "HOLD"
    # --- HYBRID LOGIC ---
    # จะ BUY: กราฟต้องบอก BUY และ ข่าวต้องไม่แย่ (Sentiment > -NEWS_FILTER)
    if row['Prob_BUY'] > CONFIDENCE_THRESHOLD and news > -NEWS_FILTER:
        signal = "BUY"
    # จะ SELL: กราฟต้องบอก SELL และ ข่าวต้องไม่ดี (Sentiment < NEWS_FILTER)
    elif row['Prob_SELL'] > CONFIDENCE_THRESHOLD and news < NEWS_FILTER:
        signal = "SELL"
        
    if signal != "HOLD":
        if risk.current_position == 0:
            action, msg = risk.process_llm_signal(signal, px)
            if action == "EXECUTED_BUY":
                # ปรับ TP/SL ให้กว้างขึ้นเพื่อสู้กับ Spread
                risk.tp_price = px + (atr * 3.5) if signal == "BUY" else px - (atr * 3.5)
                risk.sl_price = px - (atr * 2.0) if signal == "BUY" else px + (atr * 2.0)
        else:
            # เช็คปิดออเดอร์
            if (risk.current_position == 1 and (px >= risk.tp_price or px <= risk.sl_price)):
                risk.process_llm_signal("SELL", px)
            elif (risk.current_position == -1 and (px <= risk.tp_price or px >= risk.sl_price)):
                risk.process_llm_signal("BUY", px)

# (ส่วนสรุปผลเหมือนเดิม...)
print(f"💰 ทุนสุดท้าย: ${risk.capital:.2f}")