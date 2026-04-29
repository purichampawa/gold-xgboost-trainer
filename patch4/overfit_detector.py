from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


# ==========================================================
# OUTPUT MODEL
# ==========================================================

@dataclass(slots=True)
class OverfitReport:
    sample_size_score: float
    consistency_score: float
    concentration_score: float
    streak_score: float
    robustness_score: float

    overfit_risk: str
    confidence_grade: str

    monthly_returns: dict[str, float]
    monthly_trade_count: dict[str, int]

    best_month_pct: float
    worst_month_pct: float

    longest_win_streak: int
    longest_loss_streak: int

    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _grade(score: float) -> str:

    if score >= 8.5:
        return "A"
    if score >= 7.0:
        return "B"
    if score >= 5.5:
        return "C"
    if score >= 4.0:
        return "D"
    return "F"


# ==========================================================
# STREAKS
# ==========================================================

def _calc_streaks(pnls: np.ndarray) -> tuple[int, int]:

    longest_win = 0
    longest_loss = 0

    cur_win = 0
    cur_loss = 0

    for p in pnls:

        if p > 0:
            cur_win += 1
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = 0
            cur_loss = 0

        longest_win = max(longest_win, cur_win)
        longest_loss = max(longest_loss, cur_loss)

    return longest_win, longest_loss


# ==========================================================
# CORE
# ==========================================================

def detect_overfit(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame | None = None,
) -> OverfitReport:
    """
    Evaluate if backtest result looks robust or likely overfit.
    """

    notes: list[str] = []

    # ------------------------------------------------------
    # Basic Trades
    # ------------------------------------------------------
    if trades_df.empty:
        return OverfitReport(
            sample_size_score=0.0,
            consistency_score=0.0,
            concentration_score=0.0,
            streak_score=0.0,
            robustness_score=0.0,
            overfit_risk="HIGH",
            confidence_grade="F",
            monthly_returns={},
            monthly_trade_count={},
            best_month_pct=0.0,
            worst_month_pct=0.0,
            longest_win_streak=0,
            longest_loss_streak=0,
            notes=["No trades found"],
        )

    closed = trades_df.copy()

    if "status" in closed.columns:
        closed = closed[
            closed["status"] == "CLOSED"
        ].copy()

    if closed.empty:
        notes.append("No closed trades")

    total_trades = len(closed)

    pnls = (
        closed["pnl"].astype(float).to_numpy()
        if "pnl" in closed.columns
        else np.array([], dtype=float)
    )

    # ======================================================
    # 1. SAMPLE SIZE SCORE
    # ======================================================
    if total_trades >= 300:
        sample_size = 10.0
    elif total_trades >= 200:
        sample_size = 8.5
    elif total_trades >= 100:
        sample_size = 7.0
    elif total_trades >= 50:
        sample_size = 5.5
    else:
        sample_size = 3.0
        notes.append("Low trade count")

    # ======================================================
    # 2. MONTHLY CONSISTENCY
    # ======================================================
    monthly_returns = {}
    monthly_trade_count = {}

    consistency = 5.0
    best_month = 0.0
    worst_month = 0.0

    if "exit_time" in closed.columns:

        closed["exit_time"] = pd.to_datetime(
            closed["exit_time"],
            errors="coerce",
        )

        closed = closed.dropna(
            subset=["exit_time"]
        )

        if not closed.empty:

            closed["month"] = closed[
                "exit_time"
            ].dt.to_period("M").astype(str)

            grp = closed.groupby("month")

            pnl_month = grp["pnl"].sum()
            cnt_month = grp["pnl"].count()

            monthly_returns = {
                k: round(v, 4)
                for k, v in pnl_month.items()
            }

            monthly_trade_count = {
                k: int(v)
                for k, v in cnt_month.items()
            }

            arr = pnl_month.to_numpy()

            if len(arr) >= 2:

                pos_months = np.sum(arr > 0)
                ratio = pos_months / len(arr)

                consistency = ratio * 10.0

                best_month = _safe_float(arr.max())
                worst_month = _safe_float(arr.min())

                if ratio < 0.55:
                    notes.append(
                        "Low positive month ratio"
                    )

                if (
                    abs(best_month)
                    > abs(arr.sum()) * 0.50
                ):
                    notes.append(
                        "Return concentrated in one month"
                    )

            else:
                consistency = 5.0
                notes.append(
                    "Few monthly observations"
                )

    # ======================================================
    # 3. CONCENTRATION SCORE
    # ======================================================
    concentration = 8.0

    if len(pnls) >= 5:

        top5 = np.sort(pnls)[-5:]
        total_profit = pnls[pnls > 0].sum()

        if total_profit > 0:
            ratio = top5.sum() / total_profit

            concentration = max(
                0.0,
                10.0 - ratio * 10.0
            )

            if ratio > 0.45:
                notes.append(
                    "Too dependent on top winners"
                )

    # ======================================================
    # 4. STREAK SCORE
    # ======================================================
    longest_win, longest_loss = _calc_streaks(
        pnls
    )

    streak = 8.0

    if longest_loss >= 10:
        streak -= 3.0
        notes.append(
            "Long losing streak"
        )

    if longest_win >= 12:
        streak -= 1.5

    streak = max(0.0, streak)

    # ======================================================
    # 5. ROBUSTNESS SCORE
    # ======================================================
    robustness = (
        sample_size * 0.30
        + consistency * 0.30
        + concentration * 0.20
        + streak * 0.20
    )

    robustness = round(
        max(0.0, min(10.0, robustness)),
        2
    )

    # ======================================================
    # RISK LABEL
    # ======================================================
    if robustness >= 8.0:
        risk = "LOW"
    elif robustness >= 6.0:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    grade = _grade(robustness)

    # ======================================================
    # RETURN
    # ======================================================
    return OverfitReport(
        sample_size_score=round(sample_size, 2),
        consistency_score=round(consistency, 2),
        concentration_score=round(concentration, 2),
        streak_score=round(streak, 2),
        robustness_score=robustness,

        overfit_risk=risk,
        confidence_grade=grade,

        monthly_returns=monthly_returns,
        monthly_trade_count=monthly_trade_count,

        best_month_pct=round(best_month, 4),
        worst_month_pct=round(worst_month, 4),

        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,

        notes=notes,
    )


# ==========================================================
# CONSOLE PRINT
# ==========================================================

def print_overfit_report(
    report: OverfitReport,
) -> None:

    print("")
    print("=========== OVERFIT REPORT ===========")

    print(
        f"Robustness Score : "
        f"{report.robustness_score:.2f}/10"
    )

    print(
        f"Confidence Grade : "
        f"{report.confidence_grade}"
    )

    print(
        f"Overfit Risk     : "
        f"{report.overfit_risk}"
    )

    print("-------------------------------------")

    print(
        f"Sample Size      : "
        f"{report.sample_size_score:.2f}"
    )

    print(
        f"Consistency      : "
        f"{report.consistency_score:.2f}"
    )

    print(
        f"Concentration    : "
        f"{report.concentration_score:.2f}"
    )

    print(
        f"Streak Quality   : "
        f"{report.streak_score:.2f}"
    )

    print("-------------------------------------")

    print(
        f"Longest Win Seq  : "
        f"{report.longest_win_streak}"
    )

    print(
        f"Longest Loss Seq : "
        f"{report.longest_loss_streak}"
    )

    print("-------------------------------------")

    if report.notes:
        print("Notes:")
        for n in report.notes:
            print(f"- {n}")
    else:
        print("Notes: None")

    print("=====================================")