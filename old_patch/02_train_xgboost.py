import pandas as pd
import time
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

input_file = 'gold_features_cleaned.csv'
model_file = 'gold_ml_model.pkl'

print(f"📥 กำลังโหลดข้อมูลจาก {input_file}...")
df = pd.read_csv(input_file)

# 1. แปลง Label ข้อความให้เป็นตัวเลขที่ ML เข้าใจ
label_mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
df['Target'] = df['Signal'].map(label_mapping)

# 2. แยก Features (ข้อสอบ) และ Target (เฉลย)
# ตัดคอลัมน์ที่ไม่เกี่ยวกับการคำนวณทิ้งไป
X = df.drop(columns=['Timestamp', 'Signal', 'Target'])
y = df['Target']

# 3. 🔥 Time-Series Split 🔥
# แบ่งข้อมูล 80% แรก (ม.ค. - ต.ค.) ให้โมเดลเรียน
# แบ่งข้อมูล 20% หลัง (พ.ย. - ธ.ค.) ให้โมเดลลองสอบ
# สำคัญมาก: shuffle=False เพื่อไม่ให้โมเดลแอบดูข้อสอบอนาคต
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"📊 แบ่งข้อมูลสำเร็จ:")
print(f"   - ชุดเรียน (Train): {len(X_train):,} แถว")
print(f"   - ชุดสอบ (Test)  : {len(X_test):,} แถว")

# 4. เริ่มกระบวนการ Train & Finetune
print("\n🧠 กำลังเทรนและจูนโมเดล (HistGradientBoosting)...")
start_time = time.time()

# ตรงนี้คือการ "Finetune (Hyperparameter Tuning)" เบื้องต้น
# เราตั้งค่า max_depth (ความลึกในการคิด) และ learning_rate (ความเร็วในการเรียนรู้)
model = HistGradientBoostingClassifier(
    max_iter=500,           # เพิ่มจำนวนรอบให้เรียนรู้ได้ลึกขึ้น
    learning_rate=0.01,     # ลดความเร็วในการเรียนรู้เพื่อให้เก็บรายละเอียด Pattern ได้ดีขึ้น
    max_depth=12,           # เพิ่มความลึกของต้นไม้เพื่อจับเงื่อนไขที่ซับซ้อน
    l2_regularization=2.0,  # เพิ่ม Penalty เพื่อไม่ให้โมเดลมั่นใจในตัวเองเกินเหตุ (ป้องกันการเดามั่ว)
    min_samples_leaf=50,    # บังคับให้แต่ละกิ่งต้องมีข้อมูลรองรับมากพอ (ป้องกันสัญญาณหลอก)
    random_state=42
)
# สั่งให้โมเดลเริ่มเรียน!
model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"⚡ เทรนเสร็จสมบูรณ์ใน {training_time:.2f} วินาที! (ลาก่อน 26 ชั่วโมง!)")

# 5. ทดสอบความแม่นยำ (Evaluation)
print("\n🎯 กำลังตรวจข้อสอบ...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🌟 ความแม่นยำรวม (Accuracy): {acc * 100:.2f}%\n")
print("📊 รายละเอียดความแม่นยำแยกตาม Signal (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['HOLD', 'BUY', 'SELL']))

# 6. บันทึกสมองกลเก็บไว้ใช้งาน
joblib.dump(model, model_file)
print("="*50)
print(f"💾 เซฟโมเดลสำเร็จ! บันทึกเป็นไฟล์: {model_file}")
print("   พร้อมนำไปใช้เชื่อมกับ FinBERT และทำ Backtest แล้วครับ!")
print("="*50)