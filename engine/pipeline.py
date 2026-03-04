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

        # Health monitor
        self.health = HealthMonitor(log_dir="logs")

        # State
        self.dashboard_state = DashboardState(mode=self.config.mode)
        self._markets = []
        self._events = []
        self._all_signals: list[Signal] = []
        self._errors: list[str] = []
        self._scan_count = 0

    async def start(self):
        """Start the main loop."""
        logger.info(f"=== Polymarket Algo Pipeline Starting ({self.config.mode} mode) ===")
        self._running = True

        # Initial data fetch
        await self._refresh_data()
        # Initial whale refresh
        whale: WhaleTrackingStrategy = self.strategies["whale"]
        await whale.refresh_whales()

        while self._running:
            try:
                await self._scan_cycle()
            except Exception as e:
                err_msg = f"Scan cycle error: {e}"
                logger.error(err_msg)
                self._errors.append(err_msg)
                if len(self._errors) > 100:
                    self._errors = self._errors[-50:]

            await asyncio.sleep(self.config.scan_interval_secs)

    async def stop(self):
        """Stop the pipeline gracefully."""
        self._running = False
        await self.collector.close()
        logger.info("Pipeline stopped")

    async def _scan_cycle(self):
        """One full scan cycle: fetch data → run strategies → execute signals."""
        self._scan_count += 1
        cycle_start = datetime.now(timezone.utc)

        # Refresh market data every cycle
        await self._refresh_data()

        if not self._markets:
            logger.warning("No markets available, skipping cycle")
            return

        # Run all enabled strategies
        all_signals: list[Signal] = []

        import time as _time
        for name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            try:
                _strat_start = _time.time()
                signals = await strategy.scan(self._markets, self._events)
                _strat_dur = (_time.time() - _strat_start) * 1000
                all_signals.extend(signals)
                self.health.record_strategy_run(name, len(signals), _strat_dur)
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
                strategy._stats["errors"] += 1
                self.health.record_strategy_error(name, str(e))

        # Apply whale confirmation to directional signals
        whale: WhaleTrackingStrategy = self.strategies["whale"]
        if whale.enabled and whale.whale_wallets:
            for signal in all_signals:
                if signal.source != "whale_tracking":
                    try:
                        whale_data = await whale.get_whale_sentiment(signal.condition_id)
                        signal = whale.confirm_signal(signal, whale_data)
                    except Exception:
                        pass

        # Sort by confidence (highest first)
        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        # Execute signals through paper trader
        current_prices = self._build_price_map()
        executed = 0
        for signal in all_signals:
            trade = self.trader.execute_signal(signal, current_prices)
            if trade:
                executed += 1

        # Update positions with current prices
        self.trader.update_positions(current_prices)

        # Store signals for dashboard
        self._all_signals = all_signals
        self.dashboard_state.last_scan = cycle_start
        self.dashboard_state.active_signals = all_signals[:20]
        self.dashboard_state.recent_trades = self.trader.trade_log[-50:]
        self.dashboard_state.portfolio = self.trader.portfolio

        cycle_time = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        self.health.record_scan(cycle_time, len(self._markets), len(all_signals), executed)
        logger.info(
            f"Scan #{self._scan_count}: {len(self._markets)} markets | "
            f"{len(all_signals)} signals | {executed} executed | "
            f"{cycle_time:.1f}s | Portfolio: ${self.trader.portfolio.total_value:.2f}"
        )

    async def _refresh_data(self):
        """Fetch fresh market and event data."""
        try:
            self._markets = await self.collector.get_all_active_markets()
            self._events = await self.collector.get_events(limit=100)
            self.dashboard_state.active_markets = len(self._markets)
            self.health.record_api_success("gamma")
            self.health.record_data_freshness("markets")
        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
            self._errors.append(f"Data refresh: {e}")
            self.health.record_api_error("gamma", str(e))

    def _build_price_map(self) -> dict[str, float]:
        """Build a token_id -> price map from current market data."""
        prices = {}
        for market in self._markets:
            for outcome in market.outcomes:
                prices[outcome.token_id] = outcome.price
        return prices

    def get_state(self) -> dict:
        """Get full dashboard state."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        news_strat: NewsLatencyStrategy = self.strategies["news"]
        whale_strat: WhaleTrackingStrategy = self.strategies["whale"]

        return {
            "mode": self.config.mode,
            "uptime_seconds": int(uptime),
            "uptime_human": self._format_uptime(uptime),
            "scan_count": self._scan_count,
            "active_markets": len(self._markets),
            "portfolio": {
                "total_value": round(self.trader.portfolio.total_value, 2),
                "cash": round(self.trader.portfolio.cash, 2),
                "positions_value": round(self.trader.portfolio.positions_value, 2),
                "total_pnl": round(self.trader.portfolio.total_pnl, 2),
                "total_pnl_pct": round(self.trader.portfolio.total_pnl_pct, 2),
                "total_trades": self.trader.portfolio.total_trades,
                "win_rate": round(self.trader.portfolio.win_rate * 100, 1),
                "max_drawdown": round(self.trader.portfolio.max_drawdown * 100, 2),
                "total_fees": round(self.trader.portfolio.total_fees_paid, 2),
                "positions": [
                    {
                        "market": p.market_slug,
                        "side": p.side,
                        "shares": round(p.shares, 2),
                        "entry": round(p.avg_entry_price, 3),
                        "current": round(p.current_price, 3),
                        "pnl": round(p.unrealized_pnl, 2),
                        "source": p.source.value,
                    }
                    for p in self.trader.portfolio.positions
                ],
            },
            "signals": [
                {
                    "id": s.id,
                    "time": s.timestamp.isoformat(),
                    "source": s.source.value,
                    "action": s.action.value,
                    "market": s.market_slug,
                    "confidence": round(s.confidence, 2),
                    "edge": round(s.expected_edge, 2),
                    "size": round(s.suggested_size_usd, 2),
                    "whale": s.whale_confirmed,
                    "reasoning": s.reasoning,
                }
                for s in self._all_signals[:30]
            ],
            "trades": [
                {
                    "id": t.id,
                    "time": t.timestamp.isoformat(),
                    "source": t.source.value,
                    "market": t.market_slug,
                    "side": t.side.value,
                    "price": round(t.price, 3),
                    "shares": round(t.size_shares, 2),
                    "usd": round(t.size_usd, 2),
                    "pnl": round(t.realized_pnl, 2) if t.realized_pnl else None,
                }
                for t in self.trader.trade_log[-30:]
            ],
            "strategies": {
                name: strat.stats for name, strat in self.strategies.items()
            },
            "whale_wallets": [
                {
                    "address": w.address[:8] + "..." + w.address[-4:] if len(w.address) > 12 else w.address,
                    "name": w.name,
                    "pnl": round(w.total_pnl, 0),
                    "win_rate": round(w.win_rate * 100, 1),
                }
                for w in whale_strat.whale_wallets[:10]
            ],
            "recent_news": [
                {
                    "title": h.title,
                    "source": h.source,
                    "classification": h.classification,
                }
                for h in news_strat.get_recent_headlines()[-10:]
            ],
            "performance": self.trader.get_performance_report(),
            "errors": self._errors[-10:],
            "health": self.health.get_health_report(),
            "slippage": self.slippage.get_stats(),
            "ab_tests": self.ab_tester.get_report(),
            "markets_sample": [
                {
                    "slug": m.slug,
                    "question": m.question[:80],
                    "yes_price": round(m.outcomes[0].price, 3) if m.outcomes else 0,
                    "volume_24h": round(m.volume_24h, 0),
                    "liquidity": round(m.liquidity, 0),
                    "spread": round(m.spread, 4),
                    "reward_pool": round(m.reward_pool, 2),
                }
                for m in sorted(self._markets, key=lambda x: x.volume_24h, reverse=True)[:20]
            ],
        }

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"
