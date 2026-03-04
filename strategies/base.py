"""
Base Strategy
=============
Abstract base class for all strategy implementations.
"""

from abc import ABC, abstractmethod
from data.models import Market, Event, Signal
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """All strategies implement this interface."""

    name: str = "base"
    description: str = ""

    def __init__(self, config):
        self.config = config
        self.enabled = True
        self._stats = {
            "signals_generated": 0,
            "scans_completed": 0,
            "errors": 0,
        }

    @abstractmethod
    async def scan(
        self, markets: list[Market], events: list[Event]
    ) -> list[Signal]:
        """Scan current market state and return any trade signals."""
        pass

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
        }
