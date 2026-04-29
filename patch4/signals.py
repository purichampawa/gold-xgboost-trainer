from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import numpy as np

from config import SignalConfig


@dataclass(slots=True)
class CalibrationConfig:
    base_threshold: float = 0.85
    min_threshold: float = 0.60
    conflict_gap: float = 0.15

@dataclass(slots=True)
class SignalEngine:
    calib_config: CalibrationConfig = field(default_factory=CalibrationConfig)

    @classmethod
    def from_signal_config(cls, signal_cfg: SignalConfig) -> "SignalEngine":
        return cls(
            calib_config=CalibrationConfig(
                base_threshold=float(signal_cfg.base_threshold),
                min_threshold=float(signal_cfg.min_threshold),
                conflict_gap=float(signal_cfg.conflict_gap),
            )
        )
    
    # 🚀 เปลี่ยนมารับค่าจาก 2 โมเดล (Dual-Model) พร้อมพารามิเตอร์เวลา
    def evaluate_dual_probs(
        self, 
        prob_buy: float, 
        prob_sell: float, 
        session_progress: float, 
        trades_done_in_session: int
    ) -> str:
        
        # 1. ป้องกันสัญญาณตีกันก่อนเลย! (สำคัญมาก)
        if abs(prob_buy - prob_sell) < self.calib_config.conflict_gap:
            return "HOLD"

        # 2. Dynamic Threshold ที่ฉลาดขึ้น
        # ช่วงครึ่งแรกของ Session (0.0 - 0.5) ให้เข้มงวดสุดๆ ไปเลย (Prob > 0.80)
        # ช่วงครึ่งหลัง (0.5 - 1.0) ค่อยๆ ลดเกณฑ์ลงมา ถ้ายังไม่ได้ออเดอร์
        
        if trades_done_in_session == 0:
            # --- เริ่มส่วนที่แก้ไขสำหรับ Min Threshold ท้ายรอบ ---
            if session_progress > 0.9:
                # 5-10 นาทีสุดท้ายของรอบ ถ้ายังไม่ได้เทรด ยอมรับเกณฑ์ที่ 0.50 เพื่อรักษาโควต้า
                current_threshold = 0.55
            elif session_progress < 0.5:
                current_threshold = self.calib_config.base_threshold + 0.10 
            else:
                decay_factor = (session_progress - 0.5) * 2
                decay_amount = (self.calib_config.base_threshold - self.calib_config.min_threshold) * decay_factor
                current_threshold = self.calib_config.base_threshold - decay_amount
            # --- จบส่วนที่แก้ไข ---
        else:
            current_threshold = self.calib_config.base_threshold + 0.15

        # 3. ตัดสินใจ
        if prob_buy >= current_threshold and prob_buy > prob_sell:
            return "BUY"
            
        if prob_sell >= current_threshold and prob_sell > prob_buy:
            return "SELL"
            
        return "HOLD"

    def encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        # ฟังก์ชันเดิมสำหรับแปลงกลับเป็นตัวเลข (ถ้าจำเป็นต้องใช้)
        mapping = {"HOLD": 0, "BUY": 1, "SELL": -1}
        return np.array([mapping.get(str(lbl).upper(), 0) for lbl in labels])