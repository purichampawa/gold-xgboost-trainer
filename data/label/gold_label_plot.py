import pandas as pd
import plotly.graph_objects as go

# ==========================================
# CONFIG
# ==========================================
INPUT_FILE = "gold_data_labeled_v6.csv"  # ใส่ชื่อไฟล์ที่ได้จากการ Label ใหม่
PLOT_CANDLES = 5000  # จำนวนแท่งเทียนที่ต้องการพล็อต 
START_IDX = 75000     # จุดเริ่มต้นของแท่งเทียน (เพื่อให้ข้ามช่วงแรกที่กราฟอาจจะไม่นิ่ง)

# ==========================================
# LOAD DATA
# ==========================================
print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ตัดข้อมูลมาพล็อตเพื่อไม่ให้กราฟหน่วงเกินไป
df_plot = df.iloc[START_IDX : START_IDX + PLOT_CANDLES].copy()

# คำนวณขอบเขตแกน Y ให้พอดี (บวก/ลบ เผื่อระยะลูกศร)
y_min = df_plot['xauusd_close'].min() - 5
y_max = df_plot['xauusd_close'].max() + 5

# ==========================================
# CREATE PLOTLY FIGURE (1 ช่องเน้นๆ)
# ==========================================
fig = go.Figure()

# ------------------------------------------
# 1. เส้นราคาปิด (Close Price)
# ------------------------------------------
fig.add_trace(go.Scatter(
    x=df_plot['timestamp'], 
    y=df_plot['xauusd_close'],
    mode='lines', 
    line=dict(color='#B2B5BE', width=1.5),
    name='Close Price'
))

# ==========================================
# 2. PLOT SIGNALS
# ==========================================
# กรองเฉพาะจุดที่มีสัญญาณ 1
buy_signals = df_plot[df_plot['target_buy'] == 1]
sell_signals = df_plot[df_plot['target_sell'] == 1]

offset = 2.0  # ระยะห่างลูกศรจากเส้นราคา เพื่อไม่ให้บังทับกัน

# ลูกศรชี้ขึ้นสำหรับ BUY
fig.add_trace(go.Scatter(
    x=buy_signals['timestamp'], 
    y=buy_signals['xauusd_close'] - offset,
    mode='markers', 
    marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(width=1, color='black')),
    name='TARGET BUY'
))

# ลูกศรชี้ลงสำหรับ SELL
fig.add_trace(go.Scatter(
    x=sell_signals['timestamp'], 
    y=sell_signals['xauusd_close'] + offset,
    mode='markers', 
    marker=dict(symbol='triangle-down', size=14, color='fuchsia', line=dict(width=1, color='black')),
    name='TARGET SELL'
))

# ==========================================
# LAYOUT & DISPLAY
# ==========================================
fig.update_layout(
    title_text=f"Gold DUAL-MODEL Targets Review ({PLOT_CANDLES} candles)",
    height=700,
    template="plotly_dark",
    hovermode='x unified',             
    plot_bgcolor='#131722',            
    paper_bgcolor='#131722',           
    margin=dict(l=50, r=50, t=80, b=50),
    showlegend=True,
    yaxis=dict(range=[y_min, y_max], showgrid=True, gridcolor='#2a2e39', dtick=5),
    xaxis=dict(showgrid=True, gridcolor='#2a2e39')
)

print("Opening browser to show plot...")
fig.show()