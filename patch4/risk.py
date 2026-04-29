from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


def can_open_trade(cash: float, min_order: float) -> bool:
    return cash >= min_order


def calc_position_size(cash: float, broker: BrokerConfig, risk: RiskConfig) -> float:
    if cash <= 0:
        return 0.0
    max_alloc = cash * risk.risk_fraction_per_trade
    if max_alloc < broker.min_order_size_thb:
        return 0.0
    return min(max_alloc, cash)


def apply_spread(price: float, spread_rate: float, side: str) -> float:
    side_norm = side.upper()
    if side_norm == "BUY":
        return price * (1.0 + spread_rate)
    if side_norm == "SELL":
        return price * (1.0 - spread_rate)
    raise ValueError(f"Unsupported side for spread: {side}")


def apply_slippage(price: float, slippage_rate: float, side: str) -> float:
    side_norm = side.upper()
    if side_norm == "BUY":
        return price * (1.0 + slippage_rate)
    if side_norm == "SELL":
        return price * (1.0 - slippage_rate)
    raise ValueError(f"Unsupported side for slippage: {side}")


def stop_if_blowup(
    starting_capital: float,
    current_equity: float,
    blowup_equity: float,
    blowup_loss: float,
) -> bool:
    if current_equity <= blowup_equity:
        return True
    loss_abs = starting_capital - current_equity
    return loss_abs >= blowup_loss


@dataclass(slots=True)
class RiskController:
    broker: Any = None
    risk: Any = None
    starting_capital: float = 0.0
    daily_loss_realized: float = 0.0
    current_day: date | None = None
    consecutive_losses: int = 0

    @classmethod
    def from_config(cls, config: Any) -> "RiskController":
        return cls(
            broker=config.broker,
            risk=config.risk,
            starting_capital=float(config.broker.starting_capital_thb),
        )

    def reset_day_if_needed(self, day: date) -> None:
        if self.current_day != day:
            self.current_day = day
            self.daily_loss_realized = 0.0

    def register_trade_result(self, pnl_realized: float) -> None:
        if pnl_realized < 0:
            self.daily_loss_realized += abs(pnl_realized)
            self.consecutive_losses += 1
        elif pnl_realized > 0:
            self.consecutive_losses = 0

    def max_daily_loss(self) -> bool:
        return self.daily_loss_realized >= self.risk.max_daily_loss_thb

    def max_consecutive_losses_hit(self) -> bool:
        return self.consecutive_losses >= self.risk.max_consecutive_losses

    def can_trade_now(self, cash: float) -> bool:
        if not can_open_trade(cash, self.broker.min_order_size_thb):
            return False
        if self.max_daily_loss():
            return False
        if self.max_consecutive_losses_hit():
            return False
        return True

    def is_blowup(self, equity: float) -> bool:
        return stop_if_blowup(
            starting_capital=self.starting_capital,
            current_equity=equity,
            blowup_equity=self.risk.blowup_equity_thb,
            blowup_loss=self.risk.blowup_loss_thb,
        )
