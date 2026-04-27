from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import joblib
import pandas as pd

from config import CONFIG, ProjectConfig
from metrics import compute_metrics
from reporting import print_console_summary, save_backtest_run
from risk import (
    RiskController,
    apply_slippage,
    apply_spread,
    calc_position_size,
)
from session_gate import SessionGate
from signals import SignalEngine


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass(slots=True)
class Position:
    side: str
    qty: float

    entry_time: datetime

    entry_price_raw: float
    entry_price_effective: float

    entry_notional: float

    entry_prob_buy: float
    entry_prob_sell: float

    session_name: str

    # telemetry
    bars_held: int = 0
    max_favorable_pnl: float = 0.0
    max_adverse_pnl: float = 0.0
    spread_paid_entry: float = 0.0
    slippage_paid_entry: float = 0.0


@dataclass(slots=True)
class BacktestResult:
    trades_df: pd.DataFrame
    equity_df: pd.DataFrame
    model_name: str
    feature_cols: list[str]


# ==========================================================
# ENGINE
# ==========================================================

class DualBacktester:

    def __init__(
        self,
        buy_model,
        sell_model,
        feature_cols: list[str],
        config: ProjectConfig = CONFIG,
    ):
        self.config = config

        self.buy_model = buy_model
        self.sell_model = sell_model
        self.feature_cols = feature_cols

        self.signal_engine = SignalEngine.from_signal_config(
            self.config.signals
        )

        self.session_gate = SessionGate(
            config=self.config.session
        )

        self.risk_controller = RiskController.from_config(
            self.config
        )

        self.cash = float(
            self.config.broker.starting_capital_thb
        )

        self.position: Position | None = None

        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

        self.current_session_name: str | None = None
        self.trades_in_session = 0

        self.fee_rate_total = float(
            self.config.broker.fee_rate
            + self.config.broker.commission_rate
        )

    # ======================================================
    # MAIN
    # ======================================================

    def run(self, df: pd.DataFrame) -> BacktestResult:

        print(
            "⏳ กำลังจำลองการเทรดแบบ Dual-Model "
            "พร้อม Intrabar SL/TP..."
        )

        features_df = df[self.feature_cols].copy()

        for c in ("target_buy", "target_sell"):
            if c in features_df.columns:
                features_df = features_df.drop(columns=[c])

        probs_buy_all = self.buy_model.predict_proba(
            features_df
        )[:, 1]

        probs_sell_all = self.sell_model.predict_proba(
            features_df
        )[:, 1]

        for i, row in enumerate(df.itertuples(index=False)):

            dt = getattr(
                row,
                self.config.data.timestamp_col
            )

            price_close = getattr(
                row,
                "xauusd_close"
            )

            price_high = getattr(
                row,
                "xauusd_high",
                price_close
            )

            price_low = getattr(
                row,
                "xauusd_low",
                price_close
            )

            prob_buy = float(probs_buy_all[i])
            prob_sell = float(probs_sell_all[i])

            self.risk_controller.reset_day_if_needed(
                dt.date()
            )

            session_name = self.session_gate.get_session_name(
                dt
            )

            if session_name != self.current_session_name:
                self.current_session_name = session_name
                self.trades_in_session = 0

            # ----------------------------------------------
            # update telemetry
            # ----------------------------------------------
            if self.position is not None:
                self._update_open_position_stats(
                    price_close=price_close,
                    price_high=price_high,
                    price_low=price_low,
                )

            # ----------------------------------------------
            # EXIT
            # ----------------------------------------------
            if self.position is not None:
                self._handle_exit(
                    dt=dt,
                    price_close=price_close,
                    price_high=price_high,
                    price_low=price_low,
                )

            # ----------------------------------------------
            # ENTRY
            # ----------------------------------------------
            if self.position is None:
                self._handle_entry(
                    dt=dt,
                    price_close=price_close,
                    prob_buy=prob_buy,
                    prob_sell=prob_sell,
                    session_name=session_name,
                )

            # ----------------------------------------------
            # EQUITY
            # ----------------------------------------------
            self._record_equity(
                dt=dt,
                session_name=session_name,
                mark_price=price_close,
            )

        # force close if still open
        if self.position is not None:
            self._force_close_last(df)

        return BacktestResult(
            trades_df=pd.DataFrame(self.trades),
            equity_df=pd.DataFrame(self.equity_curve),
            model_name="Dual-Math-XGBoost",
            feature_cols=self.feature_cols,
        )

    # ======================================================
    # ENTRY
    # ======================================================

    def _handle_entry(
        self,
        dt: datetime,
        price_close: float,
        prob_buy: float,
        prob_sell: float,
        session_name: str | None,
    ) -> None:

        if not self.risk_controller.can_trade_now(
            self.cash
        ):
            return

        if not self.session_gate.can_open_new_trade(
            dt
        ):
            return

        progress = self.session_gate.get_current_progress(
            dt
        )

        action = self.signal_engine.evaluate_dual_probs(
            prob_buy=prob_buy,
            prob_sell=prob_sell,
            session_progress=progress,
            trades_done_in_session=self.trades_in_session,
        )

        if action not in ("BUY", "SELL"):
            return

        order_size = calc_position_size(
            self.cash,
            self.config.broker,
            self.config.risk,
        )

        if order_size < self.config.broker.min_order_size_thb:
            return

        self.cash -= order_size

        entry_raw = float(price_close)

        spread_rate = self.config.broker.spread_rate
        slip_rate = self.config.broker.slippage_rate

        entry_spread = apply_spread(
            entry_raw,
            spread_rate,
            action,
        )

        entry_eff = apply_slippage(
            entry_spread,
            slip_rate,
            action,
        )

        qty = (
            order_size / entry_eff
            if entry_eff > 0
            else 0.0
        )

        self.position = Position(
            side=action,
            qty=qty,

            entry_time=dt,

            entry_price_raw=entry_raw,
            entry_price_effective=entry_eff,

            entry_notional=order_size,

            entry_prob_buy=prob_buy,
            entry_prob_sell=prob_sell,

            session_name=session_name or "UNKNOWN",

            spread_paid_entry=abs(
                entry_spread - entry_raw
            ),
            slippage_paid_entry=abs(
                entry_eff - entry_spread
            ),
        )

        self.trades_in_session += 1

    # ======================================================
    # OPEN POSITION TELEMETRY
    # ======================================================

    def _update_open_position_stats(
        self,
        price_close: float,
        price_high: float,
        price_low: float,
    ) -> None:

        pos = self.position
        if pos is None:
            return

        pos.bars_held += 1

        if pos.side == "BUY":

            favorable = (
                price_high
                - pos.entry_price_effective
            ) * pos.qty

            adverse = (
                price_low
                - pos.entry_price_effective
            ) * pos.qty

        else:

            favorable = (
                pos.entry_price_effective
                - price_low
            ) * pos.qty

            adverse = (
                pos.entry_price_effective
                - price_high
            ) * pos.qty

        pos.max_favorable_pnl = max(
            pos.max_favorable_pnl,
            favorable,
        )

        pos.max_adverse_pnl = min(
            pos.max_adverse_pnl,
            adverse,
        )

    # ======================================================
    # EXIT
    # ======================================================

    def _handle_exit(
        self,
        dt: datetime,
        price_close: float,
        price_high: float,
        price_low: float,
    ) -> None:

        pos = self.position
        if pos is None:
            return

        close_trade = False
        exit_price = float(price_close)
        exit_reason = "CLOSE"

        if self.session_gate.should_force_close(dt):
            close_trade = True
            exit_reason = "SESSION_END"

        else:
            sl_pct = self.config.risk.stop_loss_pct
            
            # 🟢 เพิ่มลอจิก Dynamic TP ตรงนี้
            if "Morning" in pos.session_name:
                tp_pct = 0.0018  # เช้าตลาดซึม เก็บสั้น 0.15%
            elif "Evening" in pos.session_name or "Night" in pos.session_name:
                tp_pct = 0.0022  # ดึกตลาดวิ่งแรง เก็บยาว 0.22%
            else:
                tp_pct = self.config.risk.take_profit_pct # ค่าปกติจาก Config
                
            slip = self.config.broker.slippage_rate

            if pos.side == "BUY":

                tp_price = (
                    pos.entry_price_effective
                    * (1 + tp_pct)
                    / (1 - slip)
                )

                sl_price = (
                    pos.entry_price_effective
                    * (1 - sl_pct)
                    / (1 - slip)
                )

                if price_low <= sl_price:
                    close_trade = True
                    exit_price = sl_price
                    exit_reason = "STOP_LOSS"

                elif price_high >= tp_price:
                    close_trade = True
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"

            else:

                tp_price = (
                    pos.entry_price_effective
                    * (1 - tp_pct)
                    / (1 + slip)
                )

                sl_price = (
                    pos.entry_price_effective
                    * (1 + sl_pct)
                    / (1 + slip)
                )

                if price_high >= sl_price:
                    close_trade = True
                    exit_price = sl_price
                    exit_reason = "STOP_LOSS"

                elif price_low <= tp_price:
                    close_trade = True
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"

        if not close_trade:
            return

        self._finalize_trade(
            dt=dt,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

    # ======================================================
    # FINALIZE TRADE
    # ======================================================

    def _finalize_trade(
        self,
        dt: datetime,
        exit_price: float,
        exit_reason: str,
    ) -> None:

        pos = self.position
        if pos is None:
            return

        action_exit = (
            "SELL"
            if pos.side == "BUY"
            else "BUY"
        )

        slip_rate = self.config.broker.slippage_rate

        exit_eff = apply_slippage(
            exit_price,
            slip_rate,
            action_exit,
        )

        exit_slippage_paid = abs(
            exit_eff - exit_price
        )

        if pos.side == "BUY":
            pnl_gross = (
                exit_eff
                - pos.entry_price_effective
            ) * pos.qty
        else:
            pnl_gross = (
                pos.entry_price_effective
                - exit_eff
            ) * pos.qty

        exit_value_gross = (
            pos.entry_notional + pnl_gross
        )

        fees = (
            max(exit_value_gross, 0.0)
            * self.fee_rate_total
        )

        pnl_net = pnl_gross - fees
        exit_value_net = exit_value_gross - fees

        self.cash += exit_value_net

        self.risk_controller.register_trade_result(
            pnl_net
        )

        days_held = (
            dt - pos.entry_time
        ).total_seconds() / 86400.0

        minutes_held = days_held * 1440.0

        pnl_pct = (
            pnl_net / pos.entry_notional
            if pos.entry_notional > 0
            else 0.0
        )

        self.trades.append(
            {
                "entry_time": pos.entry_time,
                "exit_time": dt,
                "status": "CLOSED",

                "side": pos.side,
                "qty": pos.qty,

                "entry_price_raw":
                    pos.entry_price_raw,
                "entry_price":
                    pos.entry_price_effective,
                "exit_price":
                    exit_eff,

                "entry_notional":
                    pos.entry_notional,
                "exit_value":
                    exit_value_net,

                "gross_pnl":
                    pnl_gross,
                "fees":
                    fees,
                "pnl":
                    pnl_net,
                "pnl_pct":
                    pnl_pct,

                "days_held":
                    days_held,
                "minutes_held":
                    minutes_held,
                "bars_held":
                    pos.bars_held,

                "reason":
                    exit_reason,
                "session_name":
                    pos.session_name,

                "entry_prob_buy":
                    pos.entry_prob_buy,
                "entry_prob_sell":
                    pos.entry_prob_sell,

                "mfe":
                    pos.max_favorable_pnl,
                "mae":
                    pos.max_adverse_pnl,

                "spread_paid_entry":
                    pos.spread_paid_entry,
                "slippage_paid_entry":
                    pos.slippage_paid_entry,
                "slippage_paid_exit":
                    exit_slippage_paid,
                "total_execution_cost":
                    pos.spread_paid_entry
                    + pos.slippage_paid_entry
                    + exit_slippage_paid
                    + fees,
            }
        )

        self.position = None

    # ======================================================
    # FORCE CLOSE LAST BAR
    # ======================================================

    def _force_close_last(
        self,
        df: pd.DataFrame,
    ) -> None:

        if self.position is None:
            return

        last_row = df.iloc[-1]

        dt = last_row[
            self.config.data.timestamp_col
        ]

        px = last_row["xauusd_close"]

        self._finalize_trade(
            dt=dt,
            exit_price=float(px),
            exit_reason="END_OF_DATA",
        )

    # ======================================================
    # EQUITY CURVE
    # ======================================================

    def _record_equity(
        self,
        dt: datetime,
        session_name: str | None,
        mark_price: float,
    ) -> None:

        equity = self.cash

        open_pnl = 0.0
        capital_deployed = 0.0
        market_value = 0.0
        free_cash = self.cash

        if self.position is not None:

            pos = self.position

            capital_deployed = (
                pos.entry_notional
            )

            action_exit = (
                "SELL"
                if pos.side == "BUY"
                else "BUY"
            )

            sim_exit = apply_slippage(
                mark_price,
                self.config.broker.slippage_rate,
                action_exit,
            )

            if pos.side == "BUY":
                pnl_gross = (
                    sim_exit
                    - pos.entry_price_effective
                ) * pos.qty
            else:
                pnl_gross = (
                    pos.entry_price_effective
                    - sim_exit
                ) * pos.qty

            sim_value = (
                pos.entry_notional
                + pnl_gross
            )

            sim_fees = (
                max(sim_value, 0.0)
                * self.fee_rate_total
            )

            open_pnl = pnl_gross - sim_fees

            market_value = (
                capital_deployed
                + open_pnl
            )

            equity += market_value

        self.equity_curve.append(
            {
                "timestamp": dt,
                "cash": self.cash,
                "free_cash": free_cash,
                "market_value": market_value,
                "equity": equity,
                "open_pnl": open_pnl,
                "capital_deployed": capital_deployed,
                "position_open": int(
                    self.position is not None
                ),
                "session": session_name,
            }
        )


# ==========================================================
# HELPERS
# ==========================================================

def _load_feature_columns(
    path: str,
) -> list[str]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        cols = json.load(f)

    return list(cols)


def _load_backtest_df() -> pd.DataFrame:

    df = pd.read_csv(
        CONFIG.data.backtest_path
    )

    tcol = CONFIG.data.timestamp_col

    df[tcol] = pd.to_datetime(
        df[tcol],
        errors="coerce",
    )

    df = df.dropna(
        subset=[tcol]
    )

    df = df.sort_values(
        tcol
    ).reset_index(drop=True)

    return df


def _clean_trade_output(
    trades_df: pd.DataFrame,
) -> pd.DataFrame:

    out = trades_df.copy()

    for c in (
        "entry_time",
        "exit_time",
        "timestamp",
    ):
        if c in out.columns:
            out[c] = pd.to_datetime(
                out[c],
                errors="coerce",
            )

    return out


def _clean_equity_output(
    equity_df: pd.DataFrame,
) -> pd.DataFrame:

    out = equity_df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(
            out["timestamp"],
            errors="coerce",
        )

        out = out.dropna(
            subset=["timestamp"]
        ).sort_values(
            "timestamp"
        )

    return out.reset_index(
        drop=True
    )


# ==========================================================
# MAIN
# ==========================================================

def run_backtest():

    print(
        "📁 โหลดไฟล์ Models & Data..."
    )

    # ------------------------------------------------------
    # MODELS
    # ------------------------------------------------------
    buy_model = joblib.load(
        CONFIG.model.buy_model_path
    )

    sell_model = joblib.load(
        CONFIG.model.sell_model_path
    )

    feature_cols = _load_feature_columns(
        CONFIG.model.feature_columns_path
    )

    # ------------------------------------------------------
    # DATA
    # ------------------------------------------------------
    df = _load_backtest_df()

    print(
        "⏳ กำลังจำลองการเทรดแบบ "
        "Dual-Model พร้อม Intrabar SL/TP..."
    )

    # ------------------------------------------------------
    # ENGINE (ของเดิม)
    # ------------------------------------------------------
    engine = DualBacktester(
        buy_model=buy_model,
        sell_model=sell_model,
        feature_cols=feature_cols,
        config=CONFIG,
    )

    result = engine.run(df)

    # ------------------------------------------------------
    # OUTPUT CLEAN
    # ------------------------------------------------------
    trades_df = _clean_trade_output(
        result.trades_df
    )

    equity_df = _clean_equity_output(
        result.equity_df
    )

    # ------------------------------------------------------
    # METRICS (SharpeFix)
    # ------------------------------------------------------
    metrics = compute_metrics(
        trades_df=trades_df,
        equity_df=equity_df,
        starting_capital=CONFIG.broker.starting_capital_thb,
        config=CONFIG,
    )

    # ------------------------------------------------------
    # SAVE REPORT FILES
    # ------------------------------------------------------
    model_name = getattr(
        result,
        "model_name",
        "dual_model",
    )

    files = save_backtest_run(
        trades_df=trades_df,
        equity_df=equity_df,
        metrics=metrics,
        model_name=model_name,
        feature_columns=feature_cols,
        config=CONFIG,
    )

    # ------------------------------------------------------
    # CONSOLE REPORT
    # ------------------------------------------------------
    print_console_summary(
        metrics=metrics,
        trades_df=trades_df,
        equity_df=equity_df,
    )

    # ------------------------------------------------------
    # RAW JSON
    # ------------------------------------------------------
    payload = metrics.to_dict()

    rounded = {
        k: (
            round(v, 4)
            if isinstance(v, float)
            else v
        )
        for k, v in payload.items()
    }

    print(
        json.dumps(
            rounded,
            indent=2,
            default=str,
        )
    )

    # ------------------------------------------------------
    # PATHS
    # ------------------------------------------------------
    print("")
    print("Saved Files")
    print(
        f"- Summary : "
        f"{files['summary_path']}"
    )
    print(
        f"- Trades  : "
        f"{files['trades_path']}"
    )
    print(
        f"- Equity  : "
        f"{files['equity_path']}"
    )
    print("")


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    run_backtest()