from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from old_patch2.config import CONFIG
from old_patch2.metrics import compute_metrics
from old_patch2.reporting import print_console_summary, save_backtest_run
from old_patch2.risk import RiskController, apply_slippage, apply_spread, calc_position_size
from old_patch2.session_gate import SessionGate
from old_patch2.signals import SignalEngine
from old_patch2.train import detect_feature_columns, load_dataset, time_series_split


@dataclass(slots=True)
class Position:
    side: str  # เพิ่ม 'LONG' หรือ 'SHORT'
    qty: float
    entry_price_raw: float
    entry_price_effective: float
    entry_notional: float
    entry_time: datetime


@dataclass(slots=True)
class BacktestResult:
    trades_df: pd.DataFrame
    equity_df: pd.DataFrame
    model_name: str
    feature_cols: list[str]


class Backtester:
    def __init__(self, model: Any, feature_cols: list[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.signal_engine = SignalEngine(config=CONFIG.signals)
        self.session_gate = SessionGate(config=CONFIG.session)
        self.risk_controller = RiskController(
            broker=CONFIG.broker,
            risk=CONFIG.risk,
            starting_capital=CONFIG.broker.starting_capital_thb,
        )

    def _exit_value_long(self, qty: float, raw_price: float) -> float:
        px = apply_spread(raw_price, CONFIG.broker.spread_rate_exit, "SELL")
        px = apply_slippage(px, CONFIG.broker.slippage_rate, "SELL")
        gross = qty * px
        return gross - (gross * CONFIG.broker.fee_rate)

    def _exit_cost_short(self, qty: float, raw_price: float) -> float:
        px = apply_spread(raw_price, CONFIG.broker.spread_rate_exit, "BUY")
        px = apply_slippage(px, CONFIG.broker.slippage_rate, "BUY")
        gross = qty * px
        return gross + (gross * CONFIG.broker.fee_rate)

    def _entry_effective_price(self, raw_price: float, side: str) -> float:
        px = apply_spread(raw_price, CONFIG.broker.spread_rate_entry, side)
        px = apply_slippage(px, CONFIG.broker.slippage_rate, side)
        return px
    
    def _create_trade_record(self, position: Position, exit_time: datetime, exit_price: float, pnl: float, confidence: float, close_reason: str) -> dict:
        """ฟังก์ชันสำหรับแพ็กข้อมูล Trade กลับไปเป็น Dictionary เพื่อบันทึกลง CSV"""
        pnl_pct = pnl / position.entry_notional if position.entry_notional > 0 else 0.0
        days_held = (exit_time - position.entry_time).total_seconds() / 86400.0
        
        # คำนวณยอดเงินรวมตอนออกเพื่อคิดค่าธรรมเนียม
        if position.side == "LONG":
            exit_gross = position.qty * apply_spread(exit_price, CONFIG.broker.spread_rate_exit, "SELL")
        else: # SHORT
            exit_gross = position.qty * apply_spread(exit_price, CONFIG.broker.spread_rate_exit, "BUY")
            
        fees = exit_gross * CONFIG.broker.fee_rate

        return {
            "entry_time": position.entry_time,
            "exit_time": exit_time,
            "status": "CLOSED",
            "side": position.side,
            "entry_price": position.entry_price_effective,
            "exit_price": exit_price,
            "qty": position.qty,
            "entry_notional": position.entry_notional,
            "exit_value": exit_gross,
            "fees": fees,
            "spread": CONFIG.broker.spread_rate_entry + CONFIG.broker.spread_rate_exit,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "days_held": days_held,
            "signal_confidence": confidence,
            "close_reason": close_reason,
        }

    def run(self, test_df: pd.DataFrame) -> BacktestResult:
        cash = CONFIG.broker.starting_capital_thb
        equity = cash
        position: Position | None = None
        trade_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []

        x_test = test_df[self.feature_cols]
        probs = self.model.predict_proba(x_test)
        actions = self.signal_engine.batch_probs_to_actions(probs)

        for idx, row in test_df.iterrows():
            ts = row[CONFIG.data.timestamp_col]
            close_px = float(row[CONFIG.data.price_col])
            action = actions[idx - test_df.index[0]]
            confidence = float(max(probs[idx - test_df.index[0]]))

            self.risk_controller.reset_day_if_needed(ts.date())

            # 1. เช็ค Hard SL / TP ก่อน (ถ้ามี position)
            if position is not None:
                # ตั้งค่าให้ตรงกับ label_engineering.py (TP 0.20% และ SL 0.25%)
                sl_net = 0.0025  
                tp_net = 0.0020  
                
                is_hit_tp = False
                is_hit_sl = False
                
                # เช็คการชน Price Level โดยตรง 
                if position.side == "LONG":
                    buy_tp_price = position.entry_price_raw * (1.0 + CONFIG.broker.spread_rate_entry + tp_net)
                    buy_sl_price = position.entry_price_raw * (1.0 + CONFIG.broker.spread_rate_entry - sl_net)
                    
                    if close_px >= buy_tp_price:
                        is_hit_tp = True
                    elif close_px <= buy_sl_price:
                        is_hit_sl = True
                        
                else: # SHORT
                    sell_tp_price = position.entry_price_raw * (1.0 - CONFIG.broker.spread_rate_entry - tp_net)
                    sell_sl_price = position.entry_price_raw * (1.0 - CONFIG.broker.spread_rate_entry + sl_net)
                    
                    if close_px <= sell_tp_price:
                        is_hit_tp = True
                    elif close_px >= sell_sl_price:
                        is_hit_sl = True
                
                # ถ้าชนเงื่อนไข ให้สั่งปิดออเดอร์
                if is_hit_tp or is_hit_sl:
                    close_reason = "TAKE_PROFIT" if is_hit_tp else "STOP_LOSS"
                    if position.side == "LONG":
                        exit_val = self._exit_value_long(position.qty, close_px)
                        pnl = exit_val - position.entry_notional
                        cash += exit_val
                    else: # SHORT
                        exit_cost = self._exit_cost_short(position.qty, close_px)
                        pnl = position.entry_notional - exit_cost
                        cash += position.entry_notional + pnl
                        
                    self.risk_controller.register_trade_result(pnl)
                    trade_rows.append(self._create_trade_record(position, ts, close_px, pnl, confidence, close_reason))
                    position = None

            # 2. บังคับปิดออเดอร์เมื่อหมด Session ตลาด
            if self.session_gate.should_force_close(ts) and position is not None:
                if position.side == "LONG":
                    exit_val = self._exit_value_long(position.qty, close_px)
                    pnl = exit_val - position.entry_notional
                    cash += exit_val
                else: # SHORT
                    exit_cost = self._exit_cost_short(position.qty, close_px)
                    pnl = position.entry_notional - exit_cost
                    cash += position.entry_notional + pnl
                    
                self.risk_controller.register_trade_result(pnl)
                trade_rows.append(self._create_trade_record(position, ts, close_px, pnl, confidence, "FORCE_CLOSE"))
                position = None

            # 3. เงื่อนไขการเปิด / ปิด ออเดอร์ตาม Signal ของโมเดล
            if position is None:
                if action in [CONFIG.signals.buy_label, CONFIG.signals.sell_label]:
                    if self.session_gate.can_open_new_trade(ts) and self.risk_controller.can_trade_now(equity):
                        notional = 1000.0  # หรือจะใช้ฟังก์ชัน calc_position_size() เพื่อคำนวณขนาดตำแหน่งตามความเสี่ยงก็ได้
                        if notional >= CONFIG.broker.min_order_size_thb:
                            side = "BUY" if action == CONFIG.signals.buy_label else "SELL"
                            px_eff = self._entry_effective_price(close_px, side)
                            qty = notional / px_eff if px_eff > 0 else 0.0
                            
                            if qty > 0:
                                pos_side = "LONG" if side == "BUY" else "SHORT"
                                cash -= notional  # วางมัดจำทั้ง Long และ Short
                                position = Position(
                                    side=pos_side,
                                    qty=qty,
                                    entry_price_raw=close_px,
                                    entry_price_effective=px_eff,
                                    entry_notional=notional,
                                    entry_time=ts,
                                )

            elif position is not None:
                # มีของอยู่ แล้วเจอสัญญาณให้ปิด
                if position.side == "LONG" and action == CONFIG.signals.sell_label:
                    if self.session_gate.can_hold_position(ts):
                        exit_val = self._exit_value_long(position.qty, close_px)
                        pnl = exit_val - position.entry_notional
                        cash += exit_val
                        self.risk_controller.register_trade_result(pnl)
                        trade_rows.append(self._create_trade_record(position, ts, close_px, pnl, confidence, "SIGNAL_CLOSE"))
                        position = None
                
                elif position.side == "SHORT" and action == CONFIG.signals.buy_label:
                    if self.session_gate.can_hold_position(ts):
                        exit_cost = self._exit_cost_short(position.qty, close_px)
                        pnl = position.entry_notional - exit_cost
                        cash += position.entry_notional + pnl
                        self.risk_controller.register_trade_result(pnl)
                        trade_rows.append(self._create_trade_record(position, ts, close_px, pnl, confidence, "SIGNAL_CLOSE"))
                        position = None

            # 4. [สำคัญ!] คำนวณ open_pnl และ equity ก่อนบันทึกลง Log
            if position is not None:
                if position.side == "LONG":
                    mtm_val = self._exit_value_long(position.qty, close_px)
                    open_pnl = mtm_val - position.entry_notional
                    equity = cash + mtm_val
                else: 
                    mtm_cost = self._exit_cost_short(position.qty, close_px)
                    open_pnl = position.entry_notional - mtm_cost
                    equity = cash + position.entry_notional + open_pnl
            else:
                open_pnl = 0.0
                equity = cash

            # ตัดจบถ้าพอร์ตแตก
            if self.risk_controller.is_blowup(equity):
                break

            # 5. บันทึกผลรายแท่งเทียน
            equity_rows.append({
                "datetime": ts,
                "equity": equity,
                "cash": cash,
                "open_pnl": open_pnl,
                "capital_deployed": position.entry_notional if position is not None else 0.0,
                "drawdown": 0.0,
                "signal": action,
            })

            if self.risk_controller.is_blowup(equity):
                break

        if position is not None:
            last_ts = test_df.iloc[-1][CONFIG.data.timestamp_col]
            last_px = float(test_df.iloc[-1][CONFIG.data.price_col])
            
            if position.side == "LONG":
                exit_val = self._exit_value_long(position.qty, last_px)
                pnl = exit_val - position.entry_notional
                cash += exit_val
            else: # SHORT
                exit_cost = self._exit_cost_short(position.qty, last_px)
                pnl = position.entry_notional - exit_cost
                cash += position.entry_notional + pnl
                
            trade_rows.append(self._create_trade_record(position, last_ts, last_px, pnl, 0.0, "FINAL_CLOSE"))
            position = None
            equity = cash

        trades_df = pd.DataFrame(trade_rows)
        equity_df = pd.DataFrame(equity_rows)
        
        if not equity_df.empty:
            running_peak = equity_df["equity"].cummax()
            equity_df["drawdown"] = (equity_df["equity"] - running_peak) / running_peak.replace(0, 1.0)

        return BacktestResult(
            trades_df=trades_df,
            equity_df=equity_df,
            model_name=self.model.__class__.__name__,
            feature_cols=self.feature_cols,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realistic backtest for BUY/HOLD/SELL model signals.")
    parser.add_argument("--model-path", type=Path, default=CONFIG.train_output.model_path)
    parser.add_argument("--feature-path", type=Path, default=CONFIG.train_output.feature_columns_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)

    if args.feature_path.exists():
        with args.feature_path.open("r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        df_tmp = load_dataset(CONFIG.data.csv_path)
        feature_cols = detect_feature_columns(df_tmp.assign(target_label="HOLD"), target_col="target_label")

    df = load_dataset(CONFIG.data.csv_path)
    _, _, test_df = time_series_split(df)
    test_df = test_df.reset_index(drop=True)

    backtester = Backtester(model=model, feature_cols=feature_cols)
    result = backtester.run(test_df)
    metrics = compute_metrics(
        trades_df=result.trades_df,
        equity_df=result.equity_df,
        starting_capital=CONFIG.broker.starting_capital_thb,
    )
    paths = save_backtest_run(
        trades_df=result.trades_df,
        equity_df=result.equity_df,
        metrics=metrics,
        model_name=result.model_name,
        feature_columns=result.feature_cols,
    )

    print_console_summary(metrics)
    
    # วนลูปเพื่อปัดเศษ 2 ตำแหน่งเฉพาะตัวแปรที่เป็นตัวเลขทศนิยม (float)
    metrics_dict = metrics.to_dict()
    rounded_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics_dict.items()}
    
    print(json.dumps(rounded_metrics, indent=2))
    
    # ปรับให้บันทึกลงไฟล์เป็นค่าที่ปัดเศษแล้วด้วย (ถ้าต้องการ)
    with open(paths['summary_path'], 'w', encoding='utf-8') as f:
        json.dump(rounded_metrics, f, indent=2)

    print(f"Summary: {paths['summary_path']}")
    print(f"Trades: {paths['trades_path']}")
    print(f"Equity: {paths['equity_path']}")
    print(f"History: {paths['history_path']}")


if __name__ == "__main__":
    main()
