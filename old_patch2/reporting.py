from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from old_patch2.config import CONFIG
from old_patch2.metrics import BacktestMetrics


def _serialize_value(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _serialize_value(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(v) for v in obj]
    return obj


def build_config_payload() -> dict[str, Any]:
    payload = {
        "data": _serialize_value(CONFIG.data),
        "labels": _serialize_value(CONFIG.labels),
        "split": _serialize_value(CONFIG.split),
        "model": _serialize_value(CONFIG.model),
        "signals": _serialize_value(CONFIG.signals),
        "broker": _serialize_value(CONFIG.broker),
        "risk": _serialize_value(CONFIG.risk),
        "session": _serialize_value(CONFIG.session),
        "metrics": _serialize_value(CONFIG.metrics),
    }
    return payload


def config_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def ensure_output_dirs() -> None:
    Path("outputs").mkdir(parents=True, exist_ok=True)
    CONFIG.backtest_output.base_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.train_output.model_path.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.logging.base_dir.mkdir(parents=True, exist_ok=True)


def save_backtest_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    metrics: BacktestMetrics,
    model_name: str,
    feature_columns: list[str],
) -> dict[str, Path]:
    ensure_output_dirs()
    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    prefix = CONFIG.backtest_output.base_dir / run_ts

    trades_path = Path(f"{prefix}_trades.csv")
    equity_path = Path(f"{prefix}_equity.csv")
    summary_path = Path(f"{prefix}_summary.json")

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    cfg_payload = build_config_payload()
    payload = {
        "runtime_timestamp": run_ts,
        "model_version": model_name,
        "config_hash": config_hash(cfg_payload),
        "feature_columns": feature_columns,
        "thresholds": {
            "label_threshold_buy": CONFIG.labels.threshold_buy,
            "label_threshold_sell": CONFIG.labels.threshold_sell,
            "signal_threshold_buy": CONFIG.signals.threshold_buy,
            "signal_threshold_sell": CONFIG.signals.threshold_sell,
            "hold_zone": CONFIG.signals.hold_zone,
            "confidence_filter": CONFIG.signals.confidence_filter,
        },
        "config": cfg_payload,
        "metrics": metrics.to_dict(),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    history_row = {
        "timestamp": run_ts,
        "model_name": model_name,
        "return_pct": metrics.total_return_pct,
        "sharpe": metrics.sharpe_ratio,
        "drawdown": metrics.max_drawdown_pct,
        "win_rate": metrics.win_rate,
        "trades": metrics.total_closed_trades,
        "config_hash": payload["config_hash"],
    }
    history = pd.DataFrame([history_row])
    if CONFIG.backtest_output.history_path.exists():
        history.to_csv(CONFIG.backtest_output.history_path, mode="a", index=False, header=False)
    else:
        history.to_csv(CONFIG.backtest_output.history_path, index=False)

    return {
        "summary_path": summary_path,
        "trades_path": trades_path,
        "equity_path": equity_path,
        "history_path": CONFIG.backtest_output.history_path,
    }


def print_console_summary(metrics: BacktestMetrics) -> None:
    print("===== BACKTEST SUMMARY =====")
    print(f"Trades: {metrics.total_closed_trades}")
    print(f"Win Rate: {metrics.win_rate * 100:.2f}%")
    print(f"Net Profit: {metrics.net_profit:+,.2f} THB")
    print(f"Unrealized P/L: {metrics.unrealized_pnl:+,.2f} THB")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"Sharpe: {metrics.sharpe_ratio:.3f}")
    print(f"Profit Factor: {metrics.profit_factor:.3f}")
    print(f"Expectancy: {metrics.expectancy_per_trade:+,.2f} THB/trade")
    print(f"XIRR: {metrics.xirr * 100:.2f}%")
    print("===========================")
