from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(slots=True)
class DataConfig:
    csv_path: Path = Path("gold_features_labeled.csv")
    timestamp_col: str = "Timestamp"
    price_col: str = "Close"
    # Existing label column in dataset (optional if using generated labels).
    raw_signal_col: str = "Signal"


@dataclass(slots=True)
class LabelConfig:
    # Predict return over N future bars.
    horizon_bars: int = 24
    threshold_buy: float = 0.0010
    threshold_sell: float = -0.0010
    # Optional spread-aware adjustment for labels.
    spread_buffer: float = 0.0014
    use_spread_aware_labels: bool = True
    buy_label: str = "BUY"
    hold_label: str = "HOLD"
    sell_label: str = "SELL"


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    require_strict_time_order: bool = True


@dataclass(slots=True)
class ModelConfig:
    model_type: str = "xgboost"  # xgboost | lightgbm | catboost | random_forest
    random_state: int = 42
    n_estimators: int = 177                 # อัปเดตแล้ว
    max_depth: int = 7                      # อัปเดตแล้ว
    learning_rate: float = 0.15568653516107106
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    class_weight_mode: str = "balanced"  # balanced | none


@dataclass(slots=True)
class SignalConfig:
    hold_zone: float = 0.21151677743433517        # อัปเดตแล้ว
    threshold_buy: float = 0.3704270897954325     # อัปเดตแล้ว
    threshold_sell: float = 0.370029944647675     # อัปเดตแล้ว
    confidence_filter: float = 0.3902296457849091 # <--- ขยับขึ้นมาเป็น 38%
    buy_label: str = "BUY"
    hold_label: str = "HOLD"
    sell_label: str = "SELL"


@dataclass(slots=True)
class BrokerConfig:
    starting_capital_thb: float = 1500.0
    min_order_size_thb: float = 1000.0
    spread_rate_entry: float = 0.0014  # เปลี่ยนจาก 0.0016
    spread_rate_exit: float = 0.0014
    fee_rate: float = 0.0
    slippage_rate: float = 0.0


@dataclass(slots=True)
class RiskConfig:
    blowup_equity_thb: float = 1000.0
    blowup_loss_thb: float = 500.0
    max_daily_loss_thb: float = 99999.0  # <--- ปรับเป็นตัวเลขสูงๆ เพื่อให้ไม่หยุดเทรดรายวัน
    max_consecutive_losses: int = 9999  # <--- ผิดติดกัน 4 ไม้ หยุดเทรด
    risk_fraction_per_trade: float = 1.0


@dataclass(slots=True)
class SessionConfig:
    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo("Asia/Bangkok"))
    force_close_at_session_end: bool = True   # <--- เปลี่ยนเป็น True (ปิดจบในวัน)
    allow_carry_overnight: bool = False       # <--- เปลี่ยนเป็น False (ห้ามถือข้ามคืน)
    deny_new_entries_outside_session: bool = True


@dataclass(slots=True)
class TrainOutputConfig:
    model_path: Path = Path("outputs/models/model.pkl")
    metrics_path: Path = Path("outputs/models/train_metrics.json")
    feature_columns_path: Path = Path("outputs/models/feature_columns.json")


@dataclass(slots=True)
class BacktestOutputConfig:
    base_dir: Path = Path("outputs/backtests")
    history_path: Path = Path("outputs/backtests/history.csv")


@dataclass(slots=True)
class LoggingConfig:
    base_dir: Path = Path("outputs/logs")


@dataclass(slots=True)
class MetricsConfig:
    risk_free_rate_annual: float = 0.02
    annualization_days: int = 252
    annualization_bars_per_day: int = 288
    min_days_for_trade_annualization: float = 5.0
    xirr_day_count: float = 365.25


@dataclass(slots=True)
class ProjectConfig:
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
