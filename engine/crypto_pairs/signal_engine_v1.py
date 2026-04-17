"""Rule-based z-score signal engine for crypto pairs trading V1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import SignalConfig


class Signal(Enum):
    HOLD = "HOLD"
    LONG_A_SHORT_B = "LONG_A_SHORT_B"
    SHORT_A_LONG_B = "SHORT_A_LONG_B"
    EXIT = "EXIT"


@dataclass(slots=True)
class SignalDecision:
    signal: Signal
    reason: str


class SignalEngineV1:
    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        self.open_positions: dict[str, dict[str, float | int | str]] = {}
        self.last_exit_timestamps_ms: dict[str, int] = {}

    def evaluate(self, *, pair_key: str, zscore: float, features: dict[str, float], timestamp_ms: int) -> SignalDecision:
        if pair_key in self.open_positions:
            return self._evaluate_exit(pair_key=pair_key, zscore=zscore, timestamp_ms=timestamp_ms)
        return self._evaluate_entry(pair_key=pair_key, zscore=zscore, features=features, timestamp_ms=timestamp_ms)

    def _evaluate_entry(self, *, pair_key: str, zscore: float, features: dict[str, float], timestamp_ms: int) -> SignalDecision:
        del features  # V1 is intentionally pure z-score rules.
        last_exit_ms = self.last_exit_timestamps_ms.get(pair_key)
        if last_exit_ms is not None and timestamp_ms - last_exit_ms < self.config.cooldown_seconds * 1000:
            return SignalDecision(Signal.HOLD, "cooldown")
        if zscore >= self.config.entry_z:
            return SignalDecision(Signal.SHORT_A_LONG_B, "entry_z_high")
        if zscore <= -self.config.entry_z:
            return SignalDecision(Signal.LONG_A_SHORT_B, "entry_z_low")
        return SignalDecision(Signal.HOLD, "inside_entry_band")

    def _evaluate_exit(self, *, pair_key: str, zscore: float, timestamp_ms: int) -> SignalDecision:
        position = self.open_positions[pair_key]
        direction = str(position["direction"])
        hold_seconds = (timestamp_ms - int(position["entry_time_ms"])) / 1000
        if direction == Signal.SHORT_A_LONG_B.value and zscore <= self.config.exit_z:
            return SignalDecision(Signal.EXIT, "take_profit_mean_reversion")
        if direction == Signal.LONG_A_SHORT_B.value and zscore >= -self.config.exit_z:
            return SignalDecision(Signal.EXIT, "take_profit_mean_reversion")
        if abs(zscore) >= self.config.stop_z:
            return SignalDecision(Signal.EXIT, "stop_loss")
        if hold_seconds >= self.config.max_hold_seconds:
            return SignalDecision(Signal.EXIT, "max_hold_timeout")
        return SignalDecision(Signal.HOLD, "position_open")

    def confirm_entry(self, *, pair_key: str, signal: Signal, zscore: float, timestamp_ms: int) -> None:
        if signal not in {Signal.LONG_A_SHORT_B, Signal.SHORT_A_LONG_B}:
            return
        self.open_positions[pair_key] = {
            "direction": signal.value,
            "entry_time_ms": timestamp_ms,
            "entry_zscore": zscore,
        }

    def confirm_exit(self, *, pair_key: str, timestamp_ms: int) -> None:
        self.open_positions.pop(pair_key, None)
        self.last_exit_timestamps_ms[pair_key] = timestamp_ms
