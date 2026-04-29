import pandas as pd
from pathlib import Path

# ชื่อไฟล์ต้นฉบับ
INPUT_FILE = "gold_data_labeled_v6.csv"

def split_dataset_4_sets():
    print(f"Loading data from {INPUT_FILE}...\n")
    df = pd.read_csv(INPUT_FILE)
    
    # แปลง timestamp เป็น datetime เพื่อใช้ในการกรองข้อมูล
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # ---------------------------------------------
    # ชุดที่ 1: Train ต้นปี 2025 - มกราคม 2026 | Backtest กุมภาพันธ์ 2026 เป็นต้นไป
    # ---------------------------------------------
    train_1 = df[df['timestamp'] < '2026-02-01'].copy()
    backtest_1 = df[df['timestamp'] >= '2026-02-01'].copy()
    
    train_1.to_csv("set1/set1_train_2025_to_jan2026.csv", index=False)
    backtest_1.to_csv("set1/set1_backtest_feb2026_onwards.csv", index=False)
    print(f"✅ ชุดที่ 1:")
    print(f"   Train (2025 - ม.ค. 26): {len(train_1)} แถว")
    print(f"   Backtest (ก.พ. 26 เป็นต้นไป): {len(backtest_1)} แถว\n")

    # ---------------------------------------------
    # ชุดที่ 2: Train ต้นปี 2025 - กุมภาพันธ์ 2026 | Backtest มีนาคม 2026 เป็นต้นไป
    # ---------------------------------------------
    train_2 = df[df['timestamp'] < '2026-03-01'].copy()
    backtest_2 = df[df['timestamp'] >= '2026-03-01'].copy()
    
    train_2.to_csv("set2/set2_train_2025_to_feb2026.csv", index=False)
    backtest_2.to_csv("set2/set2_backtest_mar2026_onwards.csv", index=False)
    print(f"✅ ชุดที่ 2:")
    print(f"   Train (2025 - ก.พ. 26): {len(train_2)} แถว")
    print(f"   Backtest (มี.ค. 26 เป็นต้นไป): {len(backtest_2)} แถว\n")

    # ---------------------------------------------
    # ชุดที่ 3: Train ต้นปี 2025 - มีนาคม 2026 | Backtest เมษายน 2026 เป็นต้นไป
    # ---------------------------------------------
    train_3 = df[df['timestamp'] < '2026-04-01'].copy()
    backtest_3 = df[df['timestamp'] >= '2026-04-01'].copy()
    
    train_3.to_csv("set3/set3_train_2025_to_mar2026.csv", index=False)
    backtest_3.to_csv("set3/set3_backtest_apr2026_onwards.csv", index=False)
    print(f"✅ ชุดที่ 3:")
    print(f"   Train (2025 - มี.ค. 26): {len(train_3)} แถว")
    print(f"   Backtest (เม.ย. 26 เป็นต้นไป): {len(backtest_3)} แถว\n")

    # ---------------------------------------------
    # ชุดที่ 4: Train ปี 2025 | Backtest ปี 2026 ทั้งหมด (เทียบเท่ากับเป้าหมายที่ 1 ของคุณ)
    # ---------------------------------------------
    train_4 = df[df['timestamp'] < '2026-01-01'].copy()
    backtest_4 = df[df['timestamp'] >= '2026-01-01'].copy()
    
    train_4.to_csv("set4/set4_train_2025.csv", index=False)
    backtest_4.to_csv("set4/set4_backtest_2026.csv", index=False)
    print(f"✅ ชุดที่ 4:")
    print(f"   Train (2025 ทั้งปี): {len(train_4)} แถว")
    print(f"   Backtest (2026 ทั้งปี): {len(backtest_4)} แถว\n")

if __name__ == "__main__":
    split_dataset_4_sets()