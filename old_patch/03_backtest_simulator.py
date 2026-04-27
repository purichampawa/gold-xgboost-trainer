import pandas as pd
import joblib
from old_patch2.session_gate import SessionGate
from patch3.risk import RiskManager

print("📈 โหลดระบบ Backtest V3 (Integrated Session & Risk)...")

# 1. โหลดข้อมูลและสมองกล
df = pd.read_csv('gold_features_cleaned.csv')
model = joblib.load('gold_ml_model.pkl')

# 2. แบ่งข้อมูลมาสอบ (20% สุดท้าย)
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
X_test = df_test.drop(columns=['Timestamp', 'Signal'])

# 3. ให้สมองกลประเมินความมั่นใจ
probabilities = model.predict_proba(X_test)
df_test['Prob_BUY'] = probabilities[:, 1]
df_test['Prob_SELL'] = probabilities[:, 2]

# 4. 🛠️ ติดตั้งระบบผู้คุมกฎ 
# เปิดโหมด MT5 เพื่อบวกเวลาชดเชย 7 ชม. ให้ตรงกับเวลาไทย
gate = SessionGate(is_mt5_data=False) # สมมติว่าไฟล์ CSV เราเป็นเวลาไทยแล้ว
# ตั้งค่าทุน 1500, ขั้นต่ำ 1000, พอร์ตแตก 1000, Spread 0.14%
risk = RiskManager(initial_capital=1500.0, min_trade_size=1000.0, game_over_level=1000.0, spread_pct=0.0014)

CONFIDENCE_THRESHOLD = 0.85 # 🔥 บังคับโมเดล: ต้องมั่นใจ 85% ถึงจะส่งซิกแนล

print(f"⏳ เริ่มจำลองการเทรด {len(df_test):,} แท่งเทียน...")

win_count = 0
loss_count = 0
rejected_count = 0

for i in range(len(df_test)):
    timestamp = df_test.loc[i, 'Timestamp']
    current_price = df_test.loc[i, 'Close']
    prob_buy = df_test.loc[i, 'Prob_BUY']
    prob_sell = df_test.loc[i, 'Prob_SELL']
    
    # ----------------------------------------------------
    # ด่านที่ 1: เช็คเวลาตลาดเปิด (Session Gate)
    # ----------------------------------------------------
    is_open, session_name = gate.check_session(timestamp)
    
    signal = "HOLD"
    if is_open:
        # ถ้าตลาดเปิด ให้เชื่อ LLM (.pkl) แบบ 100% ตามโจทย์
        if prob_buy > CONFIDENCE_THRESHOLD:
            signal = "BUY"
        elif prob_sell > CONFIDENCE_THRESHOLD:
            signal = "SELL"
    
    # ----------------------------------------------------
    # ด่านที่ 2: เช็คทุนและกติกาพอร์ต (Risk Manager)
    # ----------------------------------------------------
    if signal != "HOLD":
        action, msg = risk.process_llm_signal(signal, current_price)
        
        # จดบันทึกผลลัพธ์
        if action == "EXECUTED_SELL":
            if "PROFIT" in msg:
                win_count += 1
            else:
                loss_count += 1
        elif action == "REJECTED":
            rejected_count += 1
            
    # ----------------------------------------------------
    # ด่านที่ 3: เช็คพอร์ตแตก (Game Over Trap)
    # ----------------------------------------------------
    if risk.is_game_over:
        print(f"\n💀 GAME OVER ที่แท่งเทียน: {timestamp} (Session: {session_name})")
        print(f"🚨 สาเหตุ: {msg}")
        break # หยุดการจำลองทันที พอร์ตแตกแล้ว!

# ==========================================
# สรุปผลลัพธ์
# ==========================================
stats = risk.get_portfolio_status()
total_trades = win_count + loss_count
win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
net_profit = stats['Capital'] - 1500.0

print("\n" + "="*50)
print("🎯 📊 สรุปผลการ Backtest V3 (กฏเข้มงวด)")
print("="*50)
print(f"💰 ทุนเริ่มต้น   : $1,500.00")
print(f"💵 ทุนคงเหลือ   : ${stats['Capital']:,.2f}")
print(f"📈 กำไรสุทธิ    : ${net_profit:,.2f}")

if stats['Game_Over']:
    print("🚨 สถานะพอร์ต : 🔴 พอร์ตแตก (Game Over)")
else:
    print("✅ สถานะพอร์ต : 🟢 รอดชีวิต")
print("-" * 50)
print(f"🔄 เทรดจบไปแล้ว : {total_trades} ไม้ (ซื้อและขายครบ Loop)")
print(f"✅ ชนะ (Win)   : {win_count} ไม้")
print(f"❌ แพ้ (Loss)  : {loss_count} ไม้")
print(f"🏆 Win Rate  : {win_rate:.2f}%")
print(f"⚠️ ถูกปฏิเสธคำสั่ง: {rejected_count} ครั้ง (ติดกติกา One Bullet หรือทุนไม่พอ)")
print("="*50)