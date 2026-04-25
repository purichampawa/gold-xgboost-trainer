from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from old_patch2.config import CONFIG


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(val):
        return default
    return val


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def _to_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _xnpv(rate: float, cashflows: list[tuple[datetime, float]], day_count: float) -> float:
    if not cashflows:
        return 0.0
    t0 = cashflows[0][0]
    total = 0.0
    for dt, amount in cashflows:
        years = (dt - t0).total_seconds() / (86400.0 * day_count)
        total += amount / ((1.0 + rate) ** years)
    return total


def _xirr(cashflows: list[tuple[datetime, float]], day_count: float) -> float:
    if len(cashflows) < 2:
        return 0.0
    has_pos = any(c > 0 for _, c in cashflows)
    has_neg = any(c < 0 for _, c in cashflows)
    if not (has_pos and has_neg):
        return 0.0

    low, high = -0.999, 10.0
    f_low = _xnpv(low, cashflows, day_count)
    f_high = _xnpv(high, cashflows, day_count)

    tries = 0
    while f_low * f_high > 0 and tries < 8:
        high *= 2.0
        f_high = _xnpv(high, cashflows, day_count)
        tries += 1
        if high > 1e6:
            return 0.0

    if f_low * f_high > 0:
        return 0.0

    for _ in range(120):
        mid = (low + high) / 2.0
        f_mid = _xnpv(mid, cashflows, day_count)
        if abs(f_mid) < 1e-9:
            return _safe_float(mid)
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return _safe_float((low + high) / 2.0)


@dataclass(slots=True)
class BacktestMetrics:
    total_closed_trades: int
    total_open_trades: int
    total_wins: int
    total_losses: int
    win_rate: float
    loss_rate: float
    breakeven_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy_per_trade: float
    unrealized_pnl: float
    starting_capital: float
    ending_equity: float
    total_return_pct: float
    cagr: float
    max_drawdown_pct: float
    max_drawdown_value: float
    recovery_factor: float
    calmar_ratio: float
    volatility_daily: float
    sharpe_ratio: float
    sortino_ratio: float
    downside_deviation: float
    ulcer_index: float
    avg_days_held: float
    median_days_held: float
    max_days_held: float
    min_days_held: float
    best_annualized_trade: float
    worst_annualized_trade: float
    median_annualized_trade: float
    xirr: float
    average_capital_deployed: float
    capital_turnover: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    starting_capital: float,
) -> BacktestMetrics:
    closed = trades_df[trades_df["status"] == "CLOSED"].copy() if not trades_df.empty else pd.DataFrame()
    open_trades = trades_df[trades_df["status"] != "CLOSED"].copy() if not trades_df.empty else pd.DataFrame()

    pnls = closed["pnl"].astype(float).to_numpy() if not closed.empty else np.array([], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    breakeven = pnls[np.isclose(pnls, 0.0)]

    total_closed = int(len(closed))
    total_open = int(len(open_trades))
    total_wins = int(len(wins))
    total_losses = int(len(losses))
    win_rate = _safe_div(total_wins, total_closed)
    loss_rate = _safe_div(total_losses, total_closed)
    breakeven_rate = _safe_div(len(breakeven), total_closed)

    gross_profit = _safe_float(wins.sum())
    gross_loss = _safe_float(losses.sum())
    net_profit = _safe_float(pnls.sum())

    avg_win = _safe_float(wins.mean()) if total_wins > 0 else 0.0
    avg_loss_abs = abs(_safe_float(losses.mean())) if total_losses > 0 else 0.0
    largest_win = _safe_float(wins.max()) if total_wins > 0 else 0.0
    largest_loss = _safe_float(losses.min()) if total_losses > 0 else 0.0
    profit_factor = _safe_div(gross_profit, abs(gross_loss), default=0.0)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss_abs)

    ending_equity = _safe_float(equity_df["equity"].iloc[-1] if not equity_df.empty else starting_capital)
    total_return_pct = _safe_div(ending_equity - starting_capital, starting_capital) * 100.0

    if equity_df.empty:
        drawdowns = np.array([0.0], dtype=float)
    else:
        eq = equity_df["equity"].astype(float).to_numpy()
        running_peak = np.maximum.accumulate(eq)
        drawdowns = _safe_div(eq - running_peak, 1.0) / np.where(running_peak == 0, 1.0, running_peak)
    max_dd_pct = _safe_float(drawdowns.min() * 100.0)
    max_dd_value = _safe_float((starting_capital - ending_equity) if max_dd_pct == 0 else abs(max_dd_pct / 100.0) * starting_capital)
    recovery_factor = _safe_div(net_profit, abs(max_dd_value), default=0.0)

    start_dt = _to_datetime_series(equity_df["datetime"]).iloc[0] if not equity_df.empty else pd.NaT
    end_dt = _to_datetime_series(equity_df["datetime"]).iloc[-1] if not equity_df.empty else pd.NaT
    years = 0.0
    if pd.notna(start_dt) and pd.notna(end_dt):
        years = max((end_dt - start_dt).total_seconds() / (365.25 * 86400.0), 0.0)
    cagr = 0.0
    if years >= 1.0 and starting_capital > 0 and ending_equity > 0:
        cagr = (ending_equity / starting_capital) ** (1.0 / years) - 1.0
    calmar = _safe_div(cagr, abs(max_dd_pct) / 100.0, default=0.0)

    if not equity_df.empty:
        eq_series = equity_df.set_index(_to_datetime_series(equity_df["datetime"]))["equity"].astype(float)
        daily_eq = eq_series.resample("1D").last().dropna()
        daily_returns = daily_eq.pct_change().dropna()
    else:
        daily_returns = pd.Series(dtype=float)

    risk_free_daily = CONFIG.metrics.risk_free_rate_annual / CONFIG.metrics.annualization_days
    if daily_returns.empty:
        volatility_daily = 0.0
        sharpe = 0.0
        sortino = 0.0
        downside_deviation = 0.0
    else:
        excess = daily_returns - risk_free_daily
        volatility_daily = _safe_float(daily_returns.std(ddof=0))
        sharpe = _safe_div(
            float(excess.mean()) * math.sqrt(CONFIG.metrics.annualization_days),
            float(daily_returns.std(ddof=0)),
            default=0.0,
        )
        downside = np.minimum(excess.to_numpy(), 0.0)
        downside_deviation = _safe_float(np.sqrt(np.mean(downside**2)))
        sortino = _safe_div(
            float(excess.mean()) * math.sqrt(CONFIG.metrics.annualization_days),
            downside_deviation,
            default=0.0,
        )

    ulcer_index = _safe_float(np.sqrt(np.mean((drawdowns * 100.0) ** 2)))

    if closed.empty:
        days_held = np.array([], dtype=float)
    else:
        entry = _to_datetime_series(closed["entry_time"])
        exit_ = _to_datetime_series(closed["exit_time"])
        days_held = ((exit_ - entry).dt.total_seconds() / 86400.0).fillna(0.0).to_numpy()
    avg_days = _safe_float(np.mean(days_held)) if len(days_held) else 0.0
    med_days = _safe_float(np.median(days_held)) if len(days_held) else 0.0
    max_days = _safe_float(np.max(days_held)) if len(days_held) else 0.0
    min_days = _safe_float(np.min(days_held)) if len(days_held) else 0.0

    annualized_trade_returns = []
    if not closed.empty:
        for _, tr in closed.iterrows():
            pnl_pct = _safe_float(tr.get("pnl_pct", 0.0))
            held = _safe_float(tr.get("days_held", 0.0))
            floor_days = max(CONFIG.metrics.min_days_for_trade_annualization, 1e-9)
            effective_days = max(held, floor_days)
            ann = (1.0 + pnl_pct) ** (CONFIG.metrics.xirr_day_count / effective_days) - 1.0
            if np.isfinite(ann):
                annualized_trade_returns.append(float(ann))

    if annualized_trade_returns:
        best_ann = float(max(annualized_trade_returns))
        worst_ann = float(min(annualized_trade_returns))
        median_ann = float(np.median(annualized_trade_returns))
    else:
        best_ann = worst_ann = median_ann = 0.0

    cashflows: list[tuple[datetime, float]] = []
    if not closed.empty:
        for _, tr in closed.iterrows():
            entry_dt = pd.to_datetime(tr["entry_time"], utc=True, errors="coerce")
            exit_dt = pd.to_datetime(tr["exit_time"], utc=True, errors="coerce")
            notional = _safe_float(tr.get("entry_notional", 0.0))
            exit_value = _safe_float(tr.get("exit_value", 0.0))
            if pd.notna(entry_dt):
                cashflows.append((entry_dt.to_pydatetime(), -notional))
            if pd.notna(exit_dt):
                cashflows.append((exit_dt.to_pydatetime(), exit_value))
    cashflows.sort(key=lambda x: x[0])
    xirr = _xirr(cashflows, day_count=CONFIG.metrics.xirr_day_count)

    if equity_df.empty:
        avg_capital_deployed = 0.0
        capital_turnover = 0.0
        unrealized = 0.0
    else:
        deployed = equity_df["capital_deployed"].astype(float).to_numpy()
        avg_capital_deployed = _safe_float(np.mean(deployed))
        total_notional = _safe_float(closed["entry_notional"].sum()) if not closed.empty else 0.0
        capital_turnover = _safe_div(total_notional, starting_capital, default=0.0)
        unrealized = _safe_float(equity_df["open_pnl"].iloc[-1])

    return BacktestMetrics(
        total_closed_trades=total_closed,
        total_open_trades=total_open,
        total_wins=total_wins,
        total_losses=total_losses,
        win_rate=win_rate,
        loss_rate=loss_rate,
        breakeven_rate=breakeven_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        avg_win=avg_win,
        avg_loss=avg_loss_abs,
        largest_win=largest_win,
        largest_loss=largest_loss,
        profit_factor=profit_factor,
        expectancy_per_trade=expectancy,
        unrealized_pnl=unrealized,
        starting_capital=starting_capital,
        ending_equity=ending_equity,
        total_return_pct=total_return_pct,
        cagr=float(cagr),
        max_drawdown_pct=max_dd_pct,
        max_drawdown_value=max_dd_value,
        recovery_factor=recovery_factor,
        calmar_ratio=calmar,
        volatility_daily=volatility_daily,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        downside_deviation=downside_deviation,
        ulcer_index=ulcer_index,
        avg_days_held=avg_days,
        median_days_held=med_days,
        max_days_held=max_days,
        min_days_held=min_days,
        best_annualized_trade=best_ann,
        worst_annualized_trade=worst_ann,
        median_annualized_trade=median_ann,
        xirr=float(xirr),
        average_capital_deployed=avg_capital_deployed,
        capital_turnover=capital_turnover,
    )
