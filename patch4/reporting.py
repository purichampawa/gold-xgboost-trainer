from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from metrics import BacktestMetrics
from overfit_detector import detect_overfit


# ==========================================================
# HELPERS
# ==========================================================

def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return {
            k: _serialize(v)
            for k, v in asdict(obj).items()
        }

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            k: _serialize(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]

    return obj


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ==========================================================
# CONFIG
# ==========================================================

def build_config_payload(
    config: Any,
) -> dict[str, Any]:
    return {
        "version": getattr(
            config,
            "version",
            "unknown",
        ),
        "data": _serialize(config.data),
        "labels": _serialize(config.labels),
        "split": _serialize(config.split),
        "model": _serialize(config.model),
        "signals": _serialize(config.signals),
        "broker": _serialize(config.broker),
        "risk": _serialize(config.risk),
        "session": _serialize(config.session),
        "metrics": _serialize(config.metrics),
    }


def config_hash(
    payload: dict[str, Any],
) -> str:

    raw = json.dumps(
        payload,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()[:16]


# ==========================================================
# PATHS
# ==========================================================

def ensure_output_dirs(
    config: Any,
) -> None:

    Path("outputs").mkdir(
        parents=True,
        exist_ok=True,
    )

    config.backtest_output.base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================
# SCORE
# ==========================================================

def compute_run_score(
    metrics: BacktestMetrics,
    robustness_score: float,
    overfit_risk: str,
) -> float:

    score = 0.0

    # Profitability
    score += min(
        metrics.total_return_pct,
        100.0,
    ) * 0.18

    score += min(
        metrics.net_profit / 10.0,
        30.0,
    )

    score += min(
        metrics.profit_factor,
        3.0,
    ) * 8.0

    score += min(
        metrics.payoff_ratio,
        4.0,
    ) * 5.0

    # Win quality
    score += (
        metrics.win_rate * 10
    )

    # Sharpe (deployable only)
    score += min(
        metrics.sharpe_deployable,
        3.0,
    ) * 12.0

    score += min(
        metrics.trade_sharpe_deflated,
        3.0,
    ) * 8.0

    # Risk
    score -= abs(
        metrics.max_drawdown_pct
    ) * 1.5

    # Reality
    score += (
        metrics.reality_score
        * 2.0
    )

    # Overfit
    score += (
        robustness_score
        * 2.5
    )

    if overfit_risk == "HIGH":
        score -= 15
    elif overfit_risk == "MEDIUM":
        score -= 6

    return round(score, 2)


def grade_score(
    score: float,
) -> str:

    if score >= 92:
        return "A+"
    if score >= 84:
        return "A"
    if score >= 76:
        return "B+"
    if score >= 68:
        return "B"
    if score >= 60:
        return "C+"
    if score >= 52:
        return "C"
    return "D"


# ==========================================================
# SAVE
# ==========================================================

def save_backtest_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    metrics: BacktestMetrics,
    model_name: str,
    feature_columns: list[str],
    config: Any,
):

    ensure_output_dirs(config)

    run_ts = datetime.now().strftime(
        "%Y-%m-%d_%H%M%S"
    )

    prefix = (
        config.backtest_output.base_dir
        / run_ts
    )

    trades_path = Path(
        f"{prefix}_trades.csv"
    )
    equity_path = Path(
        f"{prefix}_equity.csv"
    )
    summary_path = Path(
        f"{prefix}_summary.json"
    )

    trades_df.to_csv(
        trades_path,
        index=False,
    )

    equity_df.to_csv(
        equity_path,
        index=False,
    )

    cfg = build_config_payload(
        config
    )

    cfg_hash = config_hash(cfg)

    overfit = detect_overfit(
        trades_df=trades_df,
        equity_df=equity_df,
    )

    score = compute_run_score(
        metrics,
        overfit.robustness_score,
        overfit.overfit_risk,
    )

    grade = grade_score(score)

    payload = {
        "runtime_timestamp":
            run_ts,
        "model_version":
            model_name,
        "config_hash":
            cfg_hash,
        "score":
            score,
        "grade":
            grade,
        "feature_columns":
            feature_columns,
        "config":
            cfg,
        "metrics":
            metrics.to_dict(),
        "overfit_report":
            overfit.to_dict(),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=str,
        )

    return {
        "summary_path":
            summary_path,
        "trades_path":
            trades_path,
        "equity_path":
            equity_path,
    }


# ==========================================================
# CONSOLE
# ==========================================================

def print_console_summary(
    metrics: BacktestMetrics,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
):

    overfit = detect_overfit(
        trades_df=trades_df,
        equity_df=equity_df,
    )

    score = compute_run_score(
        metrics,
        overfit.robustness_score,
        overfit.overfit_risk,
    )

    grade = grade_score(score)

    print("")
    print("============== BACKTEST REPORT ==============")

    # ------------------------------------------------------
    print("TRADES")
    print(
        f"Closed Trades      : "
        f"{metrics.total_closed_trades}"
    )
    print(
        f"Wins / Losses      : "
        f"{metrics.total_wins} / "
        f"{metrics.total_losses}"
    )
    print(
        f"Win Rate           : "
        f"{metrics.win_rate*100:.2f}%"
    )
    print(
        f"Profit Factor      : "
        f"{metrics.profit_factor:.3f}"
    )
    print(
        f"Payoff Ratio       : "
        f"{metrics.payoff_ratio:.3f}"
    )
    print(
        f"Expectancy         : "
        f"{metrics.expectancy_per_trade:+,.2f}"
    )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print("PROFITABILITY")
    print(
        f"Net Profit         : "
        f"{metrics.net_profit:+,.2f}"
    )
    print(
        f"Ending Equity      : "
        f"{metrics.ending_equity:,.2f}"
    )
    print(
        f"Total Return       : "
        f"{metrics.total_return_pct:.2f}%"
    )
    print(
        f"XIRR (Annualized)  : "
        f"{metrics.xirr_pct:.2f}%"
    )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print("SHARPE FRAMEWORK")

    print(
        f"Sharpe Raw         : "
        f"{metrics.sharpe_raw:.3f}"
    )

    print(
        f"Sharpe Annualized  : "
        f"{metrics.sharpe_annualized:.3f}"
    )

    print(
        f"Sharpe Deflated    : "
        f"{metrics.sharpe_deflated:.3f}"
    )

    print(
        f"Sharpe Haircut     : "
        f"{metrics.sharpe_haircut:.3f}"
    )

    print(
        f"Sharpe Deployable  : "
        f"{metrics.sharpe_deployable:.3f}"
    )

    print(
        f"Trade Sharpe Raw   : "
        f"{metrics.trade_sharpe_raw:.3f}"
    )

    print(
        f"Trade Sharpe Defl. : "
        f"{metrics.trade_sharpe_deflated:.3f}"
    )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print("ANNUALIZED TRADE")

    print(
        f"Best Trade        : "
        f"{metrics.best_annualized_trade:.2f}%"
    )

    print(
        f"Worst Trade       : "
        f"{metrics.worst_annualized_trade:.2f}%"
    )

    print(
        f"Median Trade      : "
        f"{metrics.median_annualized_trade:.2f}%"
    )

    print(
        f"Top 10% Trade     : "
        f"{metrics.top10_annualized_trade:.2f}%"
    )

    print(
        f"Bottom 10% Trade  : "
        f"{metrics.bottom10_annualized_trade:.2f}%"
    )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print("RISK")

    print(
        f"Max Drawdown       : "
        f"{metrics.max_drawdown_pct:.2f}%"
    )

    print(
        f"Drawdown Value     : "
        f"{metrics.max_drawdown_value:,.2f}"
    )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print("EFFICIENCY")

    print(
        f"Backtest Days      : "
        f"{metrics.backtest_days:.2f}"
    )

    print(
        f"Trades / Day       : "
        f"{metrics.trades_per_day:.2f}"
    )

    print(
        f"Exposure Time      : "
        f"{metrics.exposure_pct:.2f}%"
    )

    print(
        f"Active Day Ratio   : "
        f"{metrics.active_day_ratio:.2f}%"
    )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print("ADVANCED")

    print(
        f"Reality Score      : "
        f"{metrics.reality_score:.2f}/10"
    )

    if metrics.metric_warning:
        print(
            f"Warnings           : "
            f"{metrics.metric_warning}"
        )
    else:
        print(
            "Warnings           : None"
        )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print("OVERFIT CHECK")

    print(
        f"Robustness Score   : "
        f"{overfit.robustness_score:.2f}/10"
    )

    print(
        f"Overfit Risk       : "
        f"{overfit.overfit_risk}"
    )

    print(
        f"Confidence Grade   : "
        f"{overfit.confidence_grade}"
    )

    print("--------------------------------------------")

    # ------------------------------------------------------
    print(
        f"RUN SCORE          : "
        f"{score:.2f}"
    )

    print(
        f"GRADE              : "
        f"{grade}"
    )

    print("============================================")
    print("")