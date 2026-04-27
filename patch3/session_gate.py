from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

@dataclass(frozen=True, slots=True)
class SessionWindow:
    name: str
    start: time
    end: time

    @property
    def spans_midnight(self) -> bool:
        return self.start > self.end

    def contains(self, local_time: time) -> bool:
        if self.spans_midnight:
            return local_time >= self.start or local_time < self.end
        return self.start <= local_time < self.end

    # 🚀 สิ่งที่เพิ่มเข้ามา: คำนวณว่าเวลาผ่านไปกี่ % แล้ว
    def get_progress(self, local_time: time) -> float:
        if not self.contains(local_time):
            return 0.0
            
        def time_to_seconds(t: time) -> int:
            return t.hour * 3600 + t.minute * 60 + t.second
            
        start_sec = time_to_seconds(self.start)
        end_sec = time_to_seconds(self.end)
        curr_sec = time_to_seconds(local_time)
        
        if self.spans_midnight:
            if curr_sec >= start_sec:
                end_sec += 24 * 3600
            else:
                start_sec -= 24 * 3600
                
        total_duration = end_sec - start_sec
        elapsed = curr_sec - start_sec
        
        return max(0.0, min(1.0, elapsed / total_duration))

# (กำหนด WEEKDAY_SESSIONS และ WEEKEND_SESSIONS ตามเดิม)
WEEKDAY_SESSIONS = (
    SessionWindow("Weekday_Morning", time(6, 15), time(12, 0)),
    SessionWindow("Weekday_Afternoon", time(12, 0), time(18, 0)),
    SessionWindow("Weekday_Evening", time(18, 0), time(2, 0)),
)
WEEKEND_SESSIONS = (SessionWindow("Weekend_Special", time(9, 30), time(17, 30)),)

class SessionGate:
    def __init__(self, config: Any):
        self.config = config
        self.tz: ZoneInfo = self.config.timezone

    def _to_local_time(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz)

    def _get_sessions_for_date(self, dt: datetime) -> tuple[SessionWindow, ...]:
        if dt.weekday() < 5:
            return WEEKDAY_SESSIONS
        return WEEKEND_SESSIONS

    def get_session_name(self, dt: datetime) -> str | None:
        local_dt = self._to_local_time(dt)
        local_t = local_dt.time()
        for window in self._get_sessions_for_date(local_dt):
            if window.contains(local_t):
                return window.name
        return None

    def is_market_open(self, dt: datetime) -> bool:
        return self.get_session_name(dt) is not None

    def can_open_new_trade(self, dt: datetime) -> bool:
        if not self.config.deny_new_entries_outside_session:
            return True
        return self.is_market_open(dt)

    def can_hold_position(self, dt: datetime) -> bool:
        if self.is_market_open(dt):
            return True
        return bool(self.config.allow_overnight_holding or self.config.allow_carry_overnight)

    def should_force_close(self, dt: datetime) -> bool:
        if not bool(self.config.close_all_at_session_end or self.config.force_close_at_session_end):
            return False
        return not self.is_market_open(dt)

    def get_current_progress(self, dt: datetime) -> float:
        local_dt = self._to_local_time(dt)
        local_t = local_dt.time()
        for window in self._get_sessions_for_date(local_dt):
            if window.contains(local_t):
                return window.get_progress(local_t)
        return 0.0
