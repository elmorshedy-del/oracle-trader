"""
Pipeline Orchestrator
=====================
Runs the main algo loop: scan → signal → execute → log.
Coordinates all strategies and the paper trading engine.
"""

import json
import asyncio
import time as _time
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
from strategies.crypto_arb import CryptoTemporalArbStrategy
from strategies.weather import WeatherForecastStrategy
from engine.paper_trader import PaperTrader
from engine.slippage import SlippageModel
from engine.ab_tester import ABTester
from engine.health_monitor import HealthMonitor
from runtime_paths import LOG_DIR, STATE_PATH

logger = logging.getLogger(__name__)

COMPARISON_VIEW_CONFIG = {
    "all": {"label": "All", "strategy": None, "source": "all"},
    "news": {"label": "News", "strategy": "news", "source": "news_latency"},
    "bitcoin": {"label": "Bitcoin", "strategy": "crypto_arb", "source": "crypto_temporal_arb"},
    "weather": {"label": "Weather", "strategy": "weather", "source": "weather_forecast"},
    "arbitrage": {"label": "Arbitrage", "strategy": "arbitrage", "source": "multi_outcome_arbitrage"},
    "momentum": {"label": "Momentum", "strategy": "mean_reversion", "source": "mean_reversion"},
}


class Pipeline:
    """Main orchestrator — runs everything."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.start_time = datetime.now(timezone.utc)
        self._running = False
        self._cycle_lock = asyncio.Lock()

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
            "crypto_arb": CryptoTemporalArbStrategy(self.config),
            "weather": WeatherForecastStrategy(self.config),
        }

        # Engine
        self.trader = PaperTrader(
            starting_capital=self.config.risk.max_total_exposure_usd,
            log_dir=str(LOG_DIR),
            state_path=str(STATE_PATH),
        )
        self.comparison_traders = {
            view_key: PaperTrader(
                starting_capital=self.config.risk.max_total_exposure_usd,
                log_dir=str(LOG_DIR / "comparison" / view_key),
                state_path=str(STATE_PATH.with_name(f"{STATE_PATH.stem}-{view_key}{STATE_PATH.suffix}")),
            )
            for view_key, meta in COMPARISON_VIEW_CONFIG.items()
            if meta["strategy"] is not None
        }

        # Slippage model (self-calibrating)
        self.slippage = SlippageModel(initial_k=0.1, log_dir=str(LOG_DIR))

        # A/B tester
        self.ab_tester = ABTester(log_dir=str(LOG_DIR))
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
        self.health = HealthMonitor(log_dir=str(LOG_DIR))

        # State
        self.dashboard_state = DashboardState(mode=self.config.mode)
        self._markets = []
        self._events = []
        self._all_signals: list[Signal] = []
        self._latest_comparison_signals: dict[str, list[Signal]] = {
            key: [] for key in COMPARISON_VIEW_CONFIG
        }
        self._errors: list[str] = []
        self._scan_count = 0
        self._latest_diagnostics: dict = {}

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
                self.health.record_error("scan_cycle", str(e))
                if len(self._errors) > 100:
                    self._errors = self._errors[-50:]

            await asyncio.sleep(self.config.scan_interval_secs)

    async def stop(self):
        """Stop the pipeline gracefully."""
        self._running = False
        self.trader.save_state()
        for trader in self.comparison_traders.values():
            trader.save_state()
        await self.collector.close()
        logger.info("Pipeline stopped")

    async def _scan_cycle(self):
        """One full scan cycle: fetch data → run strategies → execute signals."""
        async with self._cycle_lock:
            self._scan_count += 1
            cycle_start = datetime.now(timezone.utc)

            # Refresh market data every cycle
            await self._refresh_data()

            if not self._markets:
                logger.warning("No markets available, skipping cycle")
                return

            current_prices = self._build_price_map()

            # Free capacity before looking for new entries.
            self.trader.update_positions(current_prices)
            exit_count = self.trader.check_exits(current_prices)
            resolved_count = self.trader.resolve_positions(self._markets)
            for trader in self.comparison_traders.values():
                trader.update_positions(current_prices)
                trader.check_exits(current_prices)
                trader.resolve_positions(self._markets)

            held_conditions = {p.condition_id for p in self.trader.portfolio.positions}
            candidate_markets = [
                market for market in self._markets if market.condition_id not in held_conditions
            ]

            # Run all enabled strategies on markets that are not already held.
            all_signals: list[Signal] = []
            strategy_signal_counts: dict[str, int] = {}
            strategy_signals: dict[str, list[Signal]] = {}
            for name, strategy in self.strategies.items():
                if not strategy.enabled:
                    strategy_signals[name] = []
                    continue
                try:
                    _strat_start = _time.time()
                    signals = await strategy.scan(self._markets, self._events)
                    _strat_dur = (_time.time() - _strat_start) * 1000
                    all_signals.extend(signals)
                    strategy_signal_counts[name] = len(signals)
                    strategy_signals[name] = signals
                    self.health.record_strategy_run(name, len(signals), _strat_dur)
                except Exception as e:
                    logger.error(f"Strategy {name} error: {e}")
                    strategy._stats["errors"] += 1
                    strategy_signal_counts[name] = 0
                    strategy_signals[name] = []
                    self.health.record_strategy_error(name, str(e))

            # Skip per-signal whale confirmation (was making 600+ API calls per scan)
            # Whale data is still loaded and available for the dashboard
            # TODO: implement cached whale sentiment lookup instead of per-signal API calls

            # Highest-confidence signals get first claim on remaining capacity.
            all_signals.sort(key=lambda s: s.confidence, reverse=True)
            all_signals, filtered = self.trader.select_candidate_signals(all_signals)
            if filtered:
                logger.info(
                    "[PIPELINE] Filtered signals before execution: "
                    + ", ".join(f"{reason}={count}" for reason, count in filtered.items())
                )

            executed = 0
            executed_by_strategy: dict[str, int] = {}
            for signal in all_signals:
                trade = self.trader.execute_signal(signal, current_prices)
                if trade:
                    executed += 1
                    source = trade.source.value
                    executed_by_strategy[source] = executed_by_strategy.get(source, 0) + 1

            for view_key, meta in COMPARISON_VIEW_CONFIG.items():
                strategy_name = meta["strategy"]
                if strategy_name is None:
                    self._latest_comparison_signals[view_key] = all_signals[:30]
                    continue

                trader = self.comparison_traders[view_key]
                view_signals = list(strategy_signals.get(strategy_name, []))
                view_signals.sort(key=lambda s: s.confidence, reverse=True)
                view_signals, _ = trader.select_candidate_signals(view_signals)
                self._latest_comparison_signals[view_key] = view_signals[:30]
                for signal in view_signals:
                    trader.execute_signal(signal, current_prices)
                trader.save_state()

            # Store signals for dashboard
            self._all_signals = all_signals
            self.dashboard_state.last_scan = cycle_start
            self.dashboard_state.active_signals = all_signals[:20]
            self.dashboard_state.recent_trades = self.trader.trade_log[-50:]
            self.dashboard_state.portfolio = self.trader.portfolio

            cycle_time = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            self.trader.save_state()

            self.health.record_scan(cycle_time, len(self._markets), len(all_signals), executed)
            logger.info(
                f"Scan #{self._scan_count}: {len(self._markets)} markets | "
                f"{len(all_signals)} signals | {executed} executed | "
                f"{cycle_time:.1f}s | Portfolio: ${self.trader.portfolio.total_value:.2f}"
            )
            self._write_diagnostic_entry(
                cycle_start=cycle_start,
                cycle_time=cycle_time,
                candidate_markets=candidate_markets,
                strategy_signal_counts=strategy_signal_counts,
                filtered=filtered,
                executed=executed,
                executed_by_strategy=executed_by_strategy,
                exits=exit_count,
                resolved=resolved_count,
            )

    async def reset_state(self):
        """Reset portfolio and per-strategy transient state without deleting persisted logs."""
        async with self._cycle_lock:
            self.trader.reset(self.config.risk.max_total_exposure_usd)
            self._all_signals = []
            self._latest_comparison_signals = {
                key: [] for key in COMPARISON_VIEW_CONFIG
            }
            self._errors = []
            self._scan_count = 0
            self._latest_diagnostics = {}
            self.dashboard_state = DashboardState(mode=self.config.mode)
            for trader in self.comparison_traders.values():
                trader.reset(self.config.risk.max_total_exposure_usd)

            news: NewsLatencyStrategy = self.strategies["news"]
            news._seen_headlines.clear()
            news._recent_headlines.clear()
            news._api_calls_this_hour = 0
            news._hour_start = datetime.now(timezone.utc)
            news._market_index.clear()

            mean_reversion: MeanReversionStrategy = self.strategies["mean_reversion"]
            mean_reversion._baselines.clear()

            crypto: CryptoTemporalArbStrategy = self.strategies["crypto_arb"]
            crypto._price_history = {"BTC": [], "ETH": [], "SOL": []}
            crypto._matched_markets = {}
            crypto._last_market_scan = 0

            weather: WeatherForecastStrategy = self.strategies["weather"]
            weather._forecasts.clear()
            weather._matched_markets = []
            weather._last_forecast_fetch = 0
            weather._last_market_scan = 0

            self.health = HealthMonitor(log_dir=str(LOG_DIR))
            await self._refresh_data()
            self.trader.save_state()
            logger.info("[PIPELINE] Reset live paper trading state")

    def _write_diagnostic_entry(
        self,
        *,
        cycle_start: datetime,
        cycle_time: float,
        candidate_markets: list,
        strategy_signal_counts: dict[str, int],
        filtered: dict[str, int],
        executed: int,
        executed_by_strategy: dict[str, int],
        exits: int,
        resolved: int,
    ):
        """Persist one compact diagnostic record per scan for later review."""
        positions_by_source: dict[str, int] = {}
        exposure_by_source: dict[str, float] = {}
        for pos in self.trader.portfolio.positions:
            source = pos.source.value
            positions_by_source[source] = positions_by_source.get(source, 0) + 1
            exposure_by_source[source] = exposure_by_source.get(source, 0.0) + (
                pos.shares * pos.current_price
            )

        entry = {
            "timestamp": cycle_start.isoformat(),
            "scan": self._scan_count,
            "duration_secs": round(cycle_time, 2),
            "markets_total": len(self._markets),
            "markets_tradeable": len(candidate_markets),
            "signals_by_strategy": strategy_signal_counts,
            "filtered_signals": filtered,
            "executed": executed,
            "executed_by_strategy": executed_by_strategy,
            "exits": exits,
            "resolved": resolved,
            "portfolio": {
                "cash": round(self.trader.portfolio.cash, 2),
                "total_value": round(self.trader.portfolio.total_value, 2),
                "open_positions": len(self.trader.portfolio.positions),
                "positions_by_source": positions_by_source,
                "exposure_by_source": {
                    source: round(exposure, 2)
                    for source, exposure in exposure_by_source.items()
                },
            },
        }
        self._latest_diagnostics = entry
        try:
            with open(LOG_DIR / "diagnostics.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning(f"[PIPELINE] Failed to write diagnostics log: {e}")

    async def _refresh_data(self):
        """Fetch fresh market and event data."""
        try:
            self._markets = await self.collector.get_all_active_markets()
            self._events = await self.collector.get_events(limit=100)
            # Keep held markets in the working set so open positions keep receiving
            # fresh prices and closed positions can still be resolved cleanly.
            if self.trader.portfolio.positions:
                seen_conditions = {m.condition_id for m in self._markets}
                held_slugs = {p.market_slug for p in self.trader.portfolio.positions}
                for slug in held_slugs:
                    try:
                        m = await self.collector.get_market_by_slug(slug)
                        if m and m.condition_id not in seen_conditions:
                            self._markets.append(m)
                            seen_conditions.add(m.condition_id)
                    except Exception:
                        logger.debug("Held market refresh failed for slug=%s", slug, exc_info=True)
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
        comparison_views = {}
        comparison_views["all"] = self._serialize_view(
            view_key="all",
            label=COMPARISON_VIEW_CONFIG["all"]["label"],
            source=COMPARISON_VIEW_CONFIG["all"]["source"],
            trader=self.trader,
            signals=self._all_signals[:30],
        )
        for view_key, meta in COMPARISON_VIEW_CONFIG.items():
            if view_key == "all":
                continue
            comparison_views[view_key] = self._serialize_view(
                view_key=view_key,
                label=meta["label"],
                source=meta["source"],
                trader=self.comparison_traders[view_key],
                signals=self._latest_comparison_signals.get(view_key, []),
            )

        return {
            "mode": self.config.mode,
            "uptime_seconds": int(uptime),
            "uptime_human": self._format_uptime(uptime),
            "scan_count": self._scan_count,
            "active_markets": len(self._markets),
            "portfolio": comparison_views["all"]["portfolio"],
            "signals": comparison_views["all"]["signals"],
            "trades": comparison_views["all"]["trades"],
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
            "health": self.health.get_health_report(),
            "slippage": self.slippage.get_stats(),
            "ab_tests": self.ab_tester.get_report(),
            "performance": self.trader.get_performance_report(),
            "comparison_views": comparison_views,
            "diagnostics": self._latest_diagnostics,
            "errors": self._errors[-10:],
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

    def _serialize_view(
        self,
        *,
        view_key: str,
        label: str,
        source: str,
        trader: PaperTrader,
        signals: list[Signal],
    ) -> dict:
        return {
            "key": view_key,
            "label": label,
            "source": source,
            "portfolio": {
                "total_value": round(trader.portfolio.total_value, 2),
                "cash": round(trader.portfolio.cash, 2),
                "positions_value": round(trader.portfolio.positions_value, 2),
                "total_pnl": round(trader.portfolio.total_pnl, 2),
                "total_pnl_pct": round(trader.portfolio.total_pnl_pct, 2),
                "total_trades": trader.portfolio.total_trades,
                "win_rate": round(trader.portfolio.win_rate * 100, 1),
                "max_drawdown": round(
                    getattr(trader.portfolio, "current_drawdown", trader.portfolio.max_drawdown)
                    * 100,
                    2,
                ),
                "total_fees": round(trader.portfolio.total_fees_paid, 2),
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
                    for p in trader.portfolio.positions
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
                for s in signals[:30]
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
                for t in trader.trade_log[-30:]
            ],
            "performance": trader.get_performance_report(),
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
