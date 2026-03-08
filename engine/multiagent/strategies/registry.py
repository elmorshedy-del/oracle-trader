from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StaticStrategyRegistry:
    strategies: list[Any] = field(default_factory=list)

    def active_strategies(self) -> list[Any]:
        return list(self.strategies)
