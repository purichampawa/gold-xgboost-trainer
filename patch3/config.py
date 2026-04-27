from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


def _versioned_path(version: str, *parts: str) -> Path:
    return Path("outputs") / version / Path(*parts)

@dataclass(slots=True)
class DataConfig:
    csv_path: Path = Path("data/label/gold_data_labeled_v6.csv") 
    backtest_path: Path = Path("data/label/set2_apr_backtest.csv")
    timestamp_col: str = "timestamp"
    price_col: str = "xauusd_close"
    target_buy_col: str = "target_buy"
    target_sell_col: str = "target_sell"
    raw_signal_col: str = "target_buy" # สำหรับรองรับไฟล์เก่า

@dataclass(slots=True)
class LabelConfig:
    horizon_bars: int = 12
    spread_buffer: float = 0.0007
    # Legacy thresholds สำหรับ reporting.py
    threshold_buy: float = 0.0030 
    threshold_sell: float = 0.0030
    buy_label: str = "BUY"
    hold_label: str = "HOLD"
    sell_label: str = "SELL"
    use_spread_aware_labels: bool = True

@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

@dataclass(slots=True)
class ModelConfig:
    model_type: str = "xgboost"

    buy_model_path: Path = _versioned_path("latest_model", "models", "model_buy.pkl")
    sell_model_path: Path = _versioned_path("latest_model", "models", "model_sell.pkl")
    feature_columns_path: Path = _versioned_path("latest_model", "models", "feature_columns.json")

    random_state: int = 42
    n_jobs: int = -1

    # 🟢 BUY MODEL BEST PARAMS (จาก Optuna)
    buy_n_estimators: int = 270
    buy_learning_rate: float = 0.02568177356257289
    buy_max_depth: int = 7
    buy_subsample: float = 0.8469126011150903
    buy_colsample_bytree: float = 0.9962714834928094
    buy_scale_pos_weight: float = 1.3777090544750767  # ลดจาก 1.42

    # 🔴 SELL MODEL BEST PARAMS (จาก Optuna)
    sell_n_estimators: int = 579
    sell_learning_rate: float = 0.01012015792641747
    sell_max_depth: int = 7
    sell_subsample: float = 0.8973233413246483
    sell_colsample_bytree: float = 0.96746657749076
    sell_scale_pos_weight: float = 1.0836476460180258

    # buy_n_estimators: int = 100
    # buy_learning_rate: float = 0.1
    # buy_max_depth: int = 6
    # buy_subsample: float = 1.0
    # buy_colsample_bytree: float = 1.0
    # buy_scale_pos_weight: float = 1.0 # ยังไม่เพิ่มน้ำหนักให้ Label 1

    # # SELL MODEL PARAMS
    # sell_n_estimators: int = 100
    # sell_learning_rate: float = 0.1
    # sell_max_depth: int = 6
    # sell_subsample: float = 1.0
    # sell_colsample_bytree: float = 1.0
    # sell_scale_pos_weight: float = 1.0 # ยังไม่เพิ่มน้ำหนักให้ Label 1

@dataclass(slots=True)
class SignalConfig:
    base_threshold: float = 0.70  # กลับมาตั้งให้ปลอดภัย
    min_threshold: float = 0.55   # ยอมต่ำสุดแค่นี้ตอนใกล้หมดเวลา
    conflict_gap: float = 0.15
    
    # สำหรับรองรับไฟล์เก่า
    threshold_buy: float = 0.75 
    threshold_sell: float = 0.65
    hold_zone: float = 0.15
    confidence_filter: float = 0.60
    
    buy_label: str = "BUY"
    hold_label: str = "HOLD"
    sell_label: str = "SELL"

@dataclass(slots=True)
class BrokerConfig:
    starting_capital_thb: float = 1500.0
    min_order_size_thb: float = 1000.0
    spread_rate: float = 0.0014       # 0.20%
    slippage_rate: float = 0.0001
    commission_rate: float = 0.0
    fee_rate: float = 0.0
    spread_rate_entry: float = 0.0010 # สำหรับ backtest.py เก่า
    spread_rate_exit: float = 0.0010  # สำหรับ backtest.py เก่า

@dataclass(slots=True)
class RiskConfig:
    risk_fraction_per_trade: float = 1.0
    max_daily_loss_thb: float = 500.0
    max_consecutive_losses: int = 1000
    
    # 🟢 ปรับให้ตรงกับ MAX_RISK_PCT ในไฟล์ Label
    stop_loss_pct: float = 0.0032     # 0.43% (ให้ระยะหายใจบอทเท่ากับตอนที่สอน)
    
    # 🟢 ปรับให้ตรงกับ TARGET_MOVE_PCT ในไฟล์ Label
    take_profit_pct: float = 0.0020
    
    blowup_equity_thb: float = 500.0  
    blowup_loss_thb: float = 1000.0

@dataclass(slots=True)
class SessionConfig:
    timezone: ZoneInfo = ZoneInfo("Asia/Bangkok")
    allow_overnight_holding: bool = False
    allow_carry_overnight: bool = False # สำหรับ session_gate.py
    deny_new_entries_outside_session: bool = True
    close_all_at_session_end: bool = True
    force_close_at_session_end: bool = True # สำหรับ session_gate.py

@dataclass(slots=True)
class MetricsConfig:
    risk_free_rate_annual: float = 0.02
    annualization_days: int = 252
    annualization_bars_per_day: int = 288
    min_days_for_trade_annualization: float = 5.0
    xirr_day_count: float = 365.25

@dataclass(slots=True)
class TrainOutputConfig:
    base_dir: Path = _versioned_path("latest_model", "train")
    model_path: Path = _versioned_path("latest_model", "models", "model_buy.pkl")
    metrics_path: Path = _versioned_path("latest_model", "train", "metrics.json")
    feature_columns_path: Path = _versioned_path("latest_model", "models", "feature_columns.json")

@dataclass(slots=True)
class BacktestOutputConfig:
    base_dir: Path = _versioned_path("latest_model", "backtests")
    history_path: Path = _versioned_path("latest_model", "backtests", "history.csv")

@dataclass(slots=True)
class LoggingConfig:
    base_dir: Path = _versioned_path("latest_model", "logs")

@dataclass(slots=True)
class ProjectConfig:
    version: str = "latest_model"
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    train_output: TrainOutputConfig = field(default_factory=TrainOutputConfig)
    backtest_output: BacktestOutputConfig = field(default_factory=BacktestOutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

CONFIG = ProjectConfig()