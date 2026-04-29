import pandas as pd

# โหลดข้อมูล
df = pd.read_csv('gold_data_labeled_v4.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ฟังก์ชันสำหรับบันทึกไฟล์
def save_split(train_df, test_df, set_name):
    train_df.to_csv(f"{set_name}_train.csv", index=False)
    test_df.to_csv(f"{set_name}_backtest.csv", index=False)
    print(f"✅ {set_name} saved: Train {len(train_df)} rows, Backtest {len(test_df)} rows")

# --- ชุดที่ 1: Train ปี 2025 | Backtest ม.ค. 2026 ---
train_1 = df[df['timestamp'] < '2026-01-01']
test_1 = df[(df['timestamp'] >= '2026-01-01') & (df['timestamp'] < '2026-02-01')]
save_split(train_1, test_1, "set1_jan")

# --- ชุดที่ 2: Train ปี 2025 + ม.ค. 2026 | Backtest ก.พ. 2026 ---
train_2 = df[df['timestamp'] < '2026-02-01']
test_2 = df[(df['timestamp'] >= '2026-02-01') & (df['timestamp'] < '2026-03-01')]
save_split(train_2, test_2, "set2_feb")

# --- ชุดที่ 3: Train ปี 2025 + ม.ค.-ก.พ. 2026 | Backtest มี.ค. 2026 ---
train_3 = df[df['timestamp'] < '2026-03-01']
test_3 = df[(df['timestamp'] >= '2026-03-01') & (df['timestamp'] < '2026-04-01')]
save_split(train_3, test_3, "set3_mar")

# --- ชุดที่ 4: Train ปี 2025 + ม.ค.-มี.ค. 2026 | Backtest เม.ย. 2026 ---
train_4 = df[df['timestamp'] < '2026-04-01']
test_4 = df[(df['timestamp'] >= '2026-04-01') & (df['timestamp'] < '2026-05-01')]
save_split(train_4, test_4, "set4_apr")