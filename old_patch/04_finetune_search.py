import pandas as pd
import joblib
from old_patch2.session_gate import SessionGate
from old_patch2.risk import RiskManager

# โหลดข้อมูลรอไว้เลย (ลดเวลาโหลดใน Loop)
df = pd.read_csv('gold_features_cleaned.csv')
model = joblib.load('gold_ml_model.pkl')
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
X_test = df_test.drop(columns=['Timestamp', 'Signal'])

# คำนวณความมั่นใจล่วงหน้า
probabilities = model.predict_proba(X_test)
df_test['Prob_BUY'] = probabilities[:, 1]
df_test['Prob_SELL'] = probabilities[:, 2]

# --- ตั้งค่าตัวแปรที่เราจะลองสุ่ม (Grid Search) ---
thresholds = [0.85, 0.90, 0.92, 0.95]
tp_multipliers = [1.5, 2.0, 2.5, 3.0] # ลองกำไรกี่เท่าของ ATR ดี?

best_profit = -999999
best_params = {}

print("🚀 เริ่มต้นระบบ Automated Finetuning Search...")

for th in thresholds:
    for tp_m in tp_multipliers:
        # รีเซ็ตระบบผู้คุมกฎใหม่ทุกรอบ
        gate = SessionGate(is_mt5_data=False)
        risk = RiskManager(initial_capital=1500.0, spread_pct=0.0014)
        
        for i in range(len(df_test)):
            ts = df_test.loc[i, 'Timestamp']
            px = df_test.loc[i, 'Close']
            atr = df_test.loc[i, 'ATR_14']
            
            is_open, _ = gate.check_session(ts)
            if not is_open: continue
            
            # ตัดสินใจตาม Threshold ปัจจุบันใน Loop
            sig = "HOLD"
            if df_test.loc[i, 'Prob_BUY'] > th: sig = "BUY"
            elif df_test.loc[i, 'Prob_SELL'] > th: sig = "SELL"
            
            if sig != "HOLD":
                # ปรับแต่ง Risk Manager ชั่วคราว (แบบ Hardcode ในลูป)
                if risk.current_position == 0:
                    # จังหวะเข้าซื้อ เราจะตั้ง TP/SL ตามค่าที่สุ่มใน Loop
                    action, msg = risk.process_llm_signal(sig, px)
                    if action == "EXECUTED_BUY":
                        risk.tp_price = px + (atr * tp_m) if sig == "BUY" else px - (atr * tp_m)
                        risk.sl_price = px - (atr * 1.0) if sig == "BUY" else px + (atr * 1.0)
                else:
                    # จังหวะถือครอง (เช็ค TP/SL)
                    # (หมายเหตุ: ลอจิกนี้ควรอยู่ใน Risk Manager แต่เราดัดแปลงเพื่อทดสอบไวๆ)
                    if (risk.current_position == 1 and px >= risk.tp_price) or \
                       (risk.current_position == 1 and px <= risk.sl_price):
                        risk.process_llm_signal("SELL", px)
            
            if risk.is_game_over: break
            
        current_profit = risk.capital - 1500.0
        print(f"🔍 ทดสอบ: Threshold={th}, TP={tp_m}x ATR | กำไรสุทธิ: ${current_profit:.2f}")
        
        if current_profit > best_profit:
            best_profit = current_profit
            best_params = {'Threshold': th, 'TP_Mult': tp_m}

print("\n" + "="*50)
print(f"🏆 ผลการ Finetune ที่ดีที่สุด!")
print(f"ค่าที่ควรใช้: {best_params}")
print(f"กำไรที่ทำได้: ${best_profit:.2f}")
print("="*50)