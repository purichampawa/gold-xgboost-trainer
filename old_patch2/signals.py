from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from old_patch2.config import CONFIG, SignalConfig


@dataclass(slots=True)
class SignalEngine:
    config: SignalConfig = field(default_factory=SignalConfig)
    class_order: tuple[str, str, str] = ("BUY", "HOLD", "SELL")

    def probs_to_action(self, probs: np.ndarray) -> str:
        buy_idx = self.class_order.index(self.config.buy_label)
        hold_idx = self.class_order.index(self.config.hold_label)
        sell_idx = self.class_order.index(self.config.sell_label)

        p_buy = float(probs[buy_idx])
        p_hold = float(probs[hold_idx])
        p_sell = float(probs[sell_idx])
        confidence = max(p_buy, p_hold, p_sell)

        if confidence < self.config.confidence_filter:
            return self.config.hold_label

        if abs(p_buy - p_sell) <= self.config.hold_zone:
            return self.config.hold_label

        if p_buy >= self.config.threshold_buy and p_buy > p_sell:
            return self.config.buy_label
        if p_sell >= self.config.threshold_sell and p_sell > p_buy:
            return self.config.sell_label
        return self.config.hold_label

    def batch_probs_to_actions(self, prob_matrix: np.ndarray) -> list[str]:
        return [self.probs_to_action(row) for row in prob_matrix]

    def labels_to_index(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.class_order)}

    def encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        mapping = self.labels_to_index()
        return np.array([mapping[x] for x in labels], dtype=np.int64)

    def decode_labels(self, encoded: Iterable[int]) -> list[str]:
        return [self.class_order[idx] for idx in encoded]


DEFAULT_SIGNAL_ENGINE = SignalEngine(config=CONFIG.signals)
