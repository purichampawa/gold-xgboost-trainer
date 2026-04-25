import json
import re
import pandas as pd

# 📌 ใส่ชื่อไฟล์ทั้งสองก้อนเข้าไปใน List
input_files = ['data/train_qwen.jsonl', 'data/val_qwen.jsonl']
output_file = 'gold_features_cleaned.csv'

data_list = []

print(f"⏳ กำลังอ่านและรวมข้อมูลจากไฟล์ทั้งหมด...")

# วนลูปอ่านทีละไฟล์
for file_name in input_files:
    print(f"   ▶️ กำลังดูดข้อมูลจาก: {file_name}")
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                
                # เจาะเข้าไปในโครงสร้าง Messages
                user_content = next(msg['content'] for msg in record['messages'] if msg['role'] == 'user')
                assistant_content = next(msg['content'] for msg in record['messages'] if msg['role'] == 'assistant')
                
                # ดึงคำตอบ (Label)
                signal = json.loads(assistant_content).get('signal', 'HOLD')
                
                # ใช้ Regex ควานหาตัวเลข
                ts_match = re.search(r'Timestamp T = ([\d\-]+\s[\d:]+)', user_content)
                candle = re.search(r'O=([\d\.]+)\s+H=([\d\.]+)\s+L=([\d\.]+)\s+C=([\d\.]+)', user_content)
                ema = re.search(r'EMA 9/21/50\s+:\s+([\d\.\-]+)\s+/\s+([\d\.\-]+)\s+/\s+([\d\.\-]+)', user_content)
                rsi = re.search(r'RSI\(14\)\s+:\s+([\d\.\-]+)', user_content)
                macd = re.search(r'MACD/Sig/H\s+:\s+([\d\.\-]+)\s+/\s+([\d\.\-]+)\s+/\s+([\d\.\-]+)', user_content)
                atr = re.search(r'ATR\(14\)\s+:\s+([\d\.\-]+)', user_content)
                sentiment = re.search(r'Sentiment\s+:\s+([\d\.\-]+)', user_content)
                
                if candle and ema and rsi and macd and atr and ts_match:
                    row_data = {
                        'Timestamp': ts_match.group(1),
                        'Open': float(candle.group(1)),
                        'High': float(candle.group(2)),
                        'Low': float(candle.group(3)),
                        'Close': float(candle.group(4)),
                        'EMA_9': float(ema.group(1)),
                        'EMA_21': float(ema.group(2)),
                        'EMA_50': float(ema.group(3)),
                        'RSI_14': float(rsi.group(1)),
                        'MACD': float(macd.group(1)),
                        'MACD_Signal': float(macd.group(2)),
                        'MACD_Hist': float(macd.group(3)),
                        'ATR_14': float(atr.group(1)),
                        'News_Sentiment': float(sentiment.group(1)) if sentiment else 0.0,
                        'Signal': signal
                    }
                    data_list.append(row_data)
                    
            except Exception as e:
                continue

# แปลงเป็นตาราง DataFrame
df = pd.DataFrame(data_list)

# 🔥 ทำการ Sort วันที่ (Time Series) ของทั้ง 2 ไฟล์ให้เรียงร้อยเป็นเส้นเดียวกัน
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# บันทึกเป็นไฟล์ CSV
df.to_csv(output_file, index=False)

print("="*50)
print(f"✅ สกัดและรวมข้อมูลสำเร็จ (Sort วันที่เรียบร้อย)!")
print(f"📊 ได้ข้อมูลทั้งหมด: {len(df):,} แถว (Train + Val)")
print(f"📅 ข้อมูลแท่งแรก: {df['Timestamp'].min()}")
print(f"📅 ข้อมูลแท่งสุดท้าย: {df['Timestamp'].max()}")
print(f"💾 บันทึกไฟล์พร้อมเทรน: {output_file}")
print("="*50)