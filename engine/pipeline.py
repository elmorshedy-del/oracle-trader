"""
Pipeline Orchestrator
=====================
Runs the main algo loop: scan → signal → execute → log.
Coordinates all strategies and the paper trading engine.
"""

import asyncio
import logging
from datetime import datetime, timezone
from config import PipelineConfig
from data.collector import PolymarketCollector
from data.models import DashboardState, Signal
from strategies.liquidity import HedgedLiquidityStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.whale import WhaleTrackingStrategy
from strategies.news import NewsLatencyStrategy
from strategies.mean_reversion import MeanReversionStrategy
from engine.paper_trader import PaperTrader
from engine.slippage import SlippageModel
from engine.ab_tester import ABTester
from engine.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class Pipeline:
    """Main orchestrator — runs everything."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.start_time = datetime.now(timezone.utc)
        self._running = False

        # Data
        self.collector = PolymarketCollector(
            gamma_host=self.config.api.gamma_host,
            clob_host=self.config.api.clob_host,
            data_host=self.config.api.data_host,
        )

        # Strategies
        self.strategies = {
            "liquidity": HedgedLiquidityStrategy(self.config),
            "arbitrage": ArbitrageStrategy(self.config),
            "whale": WhaleTrackingStrategy(self.config, collector=self.collector),
            "news": NewsLatencyStrategy(self.config),
            "mean_reversion": MeanReversionStrategy(self.config, collector=self.collector),
        }

        # Engine
        self.trader = PaperTrader(
            starting_capital=self.config.risk.max_total_exposure_usd,
            log_dir="logs",
        )

        # Slippage model (self-calibrating)
        self.slippage = SlippageModel(initial_k=0.1, log_dir="logs")

        # A/B tester
        self.ab_tester = ABTester(log_dir="logs")
        self.ab_tester.create_test(
            "mean_reversion",
            "conservative", {"drop_threshold": 0.15, "exit_reversion": 0.50, "min_days_left": 30},
            "aggressive", {"drop_threshold": 0.10, "exit_reversion": 0.70, "min_days_left": 14},
        )
        self.ab_tester.create_test(
            "liquidity_sizing",
            "kelly_25pct", {"kelly_cap": 0.25, "competitor_estimate": 10},
            "kelly_10pct", {"kelly_cap": 0.10, "competitor_estimate": 20},
        )
        