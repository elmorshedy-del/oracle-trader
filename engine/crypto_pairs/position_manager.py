"""Position tracking and risk limits for the crypto pairs lane."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import RiskConfig


@dataclass(slots=True)
class Position:
    pair_key: str
    direction: str
    entry_trade: dict[str, object]
    entry_time_ms: int
    entry_zscore: float
    max_hold_seconds: int


class PositionManager:
    def __init__(self, *, total_capital: float, risk_config: RiskConfig | None = None):
        self.total_capital = total_capital
        self.risk = risk_config or RiskConfig()
        self.positions: dict[str, Position] = {}
        self.daily_pnl_usd = 0.0
        self.closed_trades: list[dict[str, object]] = []

    def can_open(self, pair_key: str) -> tuple[bool, str]:
        if len(self.positions) >= self.risk.max_positions:
            return False, "max_positions_reached"
        if pair_key in self.positions:
            return False, "already_in_pair"
        if self.total_capital > 0 and self.daily_pnl_usd / self.total_capital <= -self.risk.max_daily_loss_pct:
            return False, "daily_loss_limit"
        current_exposure_pct = len(self.positions) * self.risk.max_capital_per_pair_pct
        if current_exposure_pct >= self.risk.max_total_exposure_pct:
            return False, "max_exposure"
        new_tokens = set(pair_key.split("/"))
        shared_positions = sum(1 for key in self.positions if set(key.split("/")) & new_tokens)
        if shared_positions >= self.risk.max_correlation_overlap:
            return False, "token_overlap_limit"
        return True, "ok"

    def get_position_size_per_leg(self) -> float:
        return self.total_capital * self.risk.max_capital_per_pair_pct / 2

    def open_position(self, *, pair_key: str, direction: str, entry_trade: dict[str, object], zscore: float, max_hold_seconds: int) -> None:
        self.positions[pair_key] = Position(
            pair_key=pair_key,
            direction=direction,
            entry_trade=entry_trade,
            entry_time_ms=int(time.time() * 1000),
            entry_zscore=zscore,
            max_hold_seconds=max_hold_seconds,
        )

    def close_position(self, *, pair_key: str, exit_trade: dict[str, object]) -> None:
        position = self.positions.pop(pair_key, None)
        if position is None:
            return
        pnl_usd = float(exit_trade.get("pnl_usd", 0.0))
        self.daily_pnl_usd += pnl_usd
        hold_seconds = (int(time.time() * 1000) - position.entry_time_ms) / 1000
        self.closed_trades.append(
            {
                "pair": pair_key,
                "direction": position.direction,
                "entry_zscore": position.entry_zscore,
                "pnl_usd": pnl_usd,
                "pnl_bps": float(exit_trade.get("pnl_bps", 0.0)),
                "hold_seconds": hold_seconds,
            }
        )

    def check_stale_positions(self, current_time_ms: int) -> list[str]:
        stale: list[str] = []
        for pair_key, position in self.positions.items():
            hold_seconds = (current_time_ms - position.entry_time_ms) / 1000
            if hold_seconds >= position.max_hold_seconds:
                stale.append(pair_key)
        return stale

    def reset_daily(self) -> None:
        self.daily_pnl_usd = 0.0

