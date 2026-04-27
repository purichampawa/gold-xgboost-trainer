from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


# ==========================================================
# HELPERS
# ==========================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default

    if not np.isfinite(v):
        return default

    return v


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-12:
        return default
    return a / b


def _pick_time_col(df: pd.DataFrame) -> str | None:
    for c in ("timestamp", "datetime", "time", "date"):
        if c in df.columns:
            return c
    return None


# ==========================================================
# SHARPE TOOLKIT
# ==========================================================

def _raw_sharpe(
    returns: np.ndarray,
) -> float:
    """
    Mean / Std (non annualized)
    """
    if len(returns) < 3:
        return 0.0

    mu = returns.mean()
    sd = returns.std(ddof=1)

    return _safe_div(mu, sd)


def _annualized_sharpe(
    returns: np.ndarray,
    periods_per_year: float,
) -> float:
    """
    Standard annualized Sharpe
    """
    return (
        _raw_sharpe(returns)
        * math.sqrt(
            max(periods_per_year, 0.0)
        )
    )


def _haircut_sharpe(
    annual_sharpe: float,
    sample_days: float,
    exposure_pct: float,
) -> float:
    """
    Penalize short sample + low exposure
    """

    if annual_sharpe <= 0:
        return annual_sharpe

    # sample penalty
    sample_factor = min(
        1.0,
        sample_days / 252.0
    )

    # low exposure penalty
    exp_factor = min(
        1.0,
        max(
            exposure_pct / 100.0,
            0.20
        )
    )

    factor = math.sqrt(
        sample_factor * exp_factor
    )

    return annual_sharpe * factor


def _deflated_sharpe(
    annual_sharpe: float,
    n_obs: int,
) -> float:
    """
    Simple conservative deflation
    """

    if n_obs < 20:
        return annual_sharpe * 0.35

    if n_obs < 60:
        return annual_sharpe * 0.50

    if n_obs < 120:
        return annual_sharpe * 0.65

    if n_obs < 252:
        return annual_sharpe * 0.75

    return annual_sharpe * 0.85


# ==========================================================
# OUTPUT MODEL
# ==========================================================

@dataclass(slots=True)
class BacktestMetrics:

    total_closed_trades: int
    total_wins: int
    total_losses: int

    win_rate: float

    gross_profit: float
    gross_loss: float
    net_profit: float

    avg_win: float
    avg_loss: float

    profit_factor: float
    payoff_ratio: float
    expectancy_per_trade: float

    starting_capital: float
    ending_equity: float
    total_return_pct: float

    max_drawdown_pct: float
    max_drawdown_value: float

    backtest_days: float
    trades_per_day: float

    exposure_pct: float
    active_day_ratio: float

    # ------------------------------------------------------
    # Sharpe Family
    # ------------------------------------------------------
    sharpe_raw: float
    sharpe_annualized: float
    sharpe_deflated: float
    sharpe_haircut: float
    sharpe_deployable: float

    trade_sharpe_raw: float
    trade_sharpe_annualized: float
    trade_sharpe_deflated: float

    # legacy alias
    sharpe_ratio: float
    trade_sharpe_ratio: float
    realistic_sharpe_ratio: float

    # annualized trade stats
    best_annualized_trade: float
    worst_annualized_trade: float
    median_annualized_trade: float
    top10_annualized_trade: float
    bottom10_annualized_trade: float

    # misc
    metric_warning: str
    reality_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================================
# MAIN
# ==========================================================

def compute_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    starting_capital: float,
    config: Any = None,
) -> BacktestMetrics:

    # ------------------------------------------------------
    # Trades
    # ------------------------------------------------------
    closed = trades_df.copy()

    if "status" in closed.columns:
        closed = closed[
            closed["status"] == "CLOSED"
        ].copy()

    pnl = closed["pnl"].astype(float).to_numpy()
    trade_ret = closed[
        "pnl_pct"
    ].astype(float).to_numpy()

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_closed = len(closed)
    total_wins = len(wins)
    total_losses = len(losses)

    win_rate = _safe_div(
        total_wins,
        total_closed
    )

    gross_profit = wins.sum()
    gross_loss = losses.sum()
    net_profit = pnl.sum()

    avg_win = (
        wins.mean()
        if len(wins) else 0.0
    )

    avg_loss = abs(
        losses.mean()
    ) if len(losses) else 0.0

    profit_factor = _safe_div(
        gross_profit,
        abs(gross_loss)
    )

    payoff_ratio = _safe_div(
        avg_win,
        avg_loss
    )

    expectancy = (
        net_profit / total_closed
        if total_closed else 0.0
    )

    # ------------------------------------------------------
    # Equity / Drawdown
    # ------------------------------------------------------
    eq = equity_df[
        "equity"
    ].astype(float).to_numpy()

    ending_equity = eq[-1]

    total_return_pct = (
        (ending_equity - starting_capital)
        / starting_capital
        * 100
    )

    peak = np.maximum.accumulate(eq)

    dd = eq - peak
    dd_pct = dd / peak

    max_dd_val = abs(dd.min())
    max_dd_pct = dd_pct.min() * 100

    # ------------------------------------------------------
    # Time
    # ------------------------------------------------------
    tcol = _pick_time_col(
        equity_df
    )

    ts = pd.to_datetime(
        equity_df[tcol],
        errors="coerce"
    ).dropna()

    backtest_days = (
        ts.iloc[-1] - ts.iloc[0]
    ).total_seconds() / 86400.0

    trades_per_day = _safe_div(
        total_closed,
        backtest_days
    )

    # ------------------------------------------------------
    # Exposure
    # ------------------------------------------------------
    exposure_pct = 0.0

    if "position_open" in equity_df.columns:
        exposure_pct = (
            equity_df[
                "position_open"
            ].mean()
            * 100
        )

    # ------------------------------------------------------
    # Daily Returns
    # ------------------------------------------------------
    daily_eq = (
        equity_df
        .assign(
            _ts=pd.to_datetime(
                equity_df[tcol],
                errors="coerce"
            )
        )
        .dropna()
        .set_index("_ts")["equity"]
        .resample("1D")
        .last()
        .dropna()
    )

    daily_ret = (
        daily_eq
        .pct_change()
        .dropna()
        .to_numpy()
    )

    active_day_ratio = (
        np.mean(
            np.abs(daily_ret) > 1e-12
        ) * 100
        if len(daily_ret)
        else 0.0
    )

    # ======================================================
    # DAILY SHARPE
    # ======================================================
    sharpe_raw = _raw_sharpe(
        daily_ret
    )

    sharpe_ann = _annualized_sharpe(
        daily_ret,
        252
    )

    sharpe_def = _deflated_sharpe(
        sharpe_ann,
        len(daily_ret)
    )

    sharpe_hc = _haircut_sharpe(
        sharpe_ann,
        backtest_days,
        exposure_pct,
    )

    sharpe_dep = min(
        sharpe_def,
        sharpe_hc
    )

    # ======================================================
    # TRADE SHARPE
    # ======================================================
    trade_raw = _raw_sharpe(
        trade_ret
    )

    annual_trades = (
        trades_per_day * 252
    )

    trade_ann = _annualized_sharpe(
        trade_ret,
        annual_trades
    )

    trade_def = _deflated_sharpe(
        trade_ann,
        len(trade_ret)
    )

    # ------------------------------------------------------
    # Annualized Trade Metrics
    # ------------------------------------------------------
    # ------------------------------------------------------
    # Trade Efficiency Metrics
    # ------------------------------------------------------
    best_annualized_trade = 0.0
    worst_annualized_trade = 0.0
    median_annualized_trade = 0.0
    top10_annualized_trade = 0.0
    bottom10_annualized_trade = 0.0

    if (
        len(closed) > 0
        and "days_held" in closed.columns
        and "pnl_pct" in closed.columns
    ):

        hold_days = (
            closed["days_held"]
            .astype(float)
            .clip(lower=1/48)
            .to_numpy()
        )

        trade_ret = (
            closed["pnl_pct"]
            .astype(float)
            .to_numpy()
        )

        score = (
            trade_ret / hold_days
        ) * 100.0

        score = np.clip(
            score,
            -300.0,
            300.0
        )

        best_annualized_trade = score.max()
        worst_annualized_trade = score.min()
        median_annualized_trade = np.median(score)
        top10_annualized_trade = np.percentile(score, 90)
        bottom10_annualized_trade = np.percentile(score, 10)

    # ------------------------------------------------------
    # Warnings
    # ------------------------------------------------------
    warn = []

    if sharpe_ann > 4:
        warn.append(
            "Annualized Sharpe inflated"
        )

    if backtest_days < 180:
        warn.append(
            "Short sample period"
        )

    if exposure_pct < 25:
        warn.append(
            "Low exposure strategy"
        )

    warning = ", ".join(warn)

    # ------------------------------------------------------
    # Reality Score
    # ------------------------------------------------------
    reality = 10.0

    if sharpe_ann > 3:
        reality -= (
            sharpe_ann - 3
        ) * 1.5

    if backtest_days < 180:
        reality -= 1.0

    if exposure_pct < 25:
        reality -= 1.0

    reality = max(
        0.0,
        min(10.0, reality)
    )

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------
    return BacktestMetrics(
        total_closed_trades=total_closed,
        total_wins=total_wins,
        total_losses=total_losses,

        win_rate=win_rate,

        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,

        avg_win=avg_win,
        avg_loss=avg_loss,

        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        expectancy_per_trade=expectancy,

        starting_capital=starting_capital,
        ending_equity=ending_equity,
        total_return_pct=total_return_pct,

        max_drawdown_pct=max_dd_pct,
        max_drawdown_value=max_dd_val,

        backtest_days=backtest_days,
        trades_per_day=trades_per_day,

        exposure_pct=exposure_pct,
        active_day_ratio=active_day_ratio,

        sharpe_raw=sharpe_raw,
        sharpe_annualized=sharpe_ann,
        sharpe_deflated=sharpe_def,
        sharpe_haircut=sharpe_hc,
        sharpe_deployable=sharpe_dep,

        trade_sharpe_raw=trade_raw,
        trade_sharpe_annualized=trade_ann,
        trade_sharpe_deflated=trade_def,

        # aliases
        sharpe_ratio=sharpe_dep,
        trade_sharpe_ratio=trade_def,
        realistic_sharpe_ratio=sharpe_dep,

        best_annualized_trade=best_annualized_trade,
        worst_annualized_trade=worst_annualized_trade,
        median_annualized_trade=median_annualized_trade,
        top10_annualized_trade=top10_annualized_trade,
        bottom10_annualized_trade=bottom10_annualized_trade,

        metric_warning=warning,
        reality_score=reality,
    )