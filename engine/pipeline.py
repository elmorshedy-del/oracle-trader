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
import re
from datetime import datetime, timezone, timedelta
from config import PipelineConfig
from data.collector import PolymarketCollector
from data.models import DashboardState, Signal, SignalAction, SignalSource
from strategies.liquidity import HedgedLiquidityStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.whale import WhaleTrackingStrategy
from strategies.news import NewsLatencyStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.crypto_arb import CryptoTemporalArbStrategy
from strategies.bitcoin_model import BitcoinModelStrategy
from strategies.bitcoin_meanrev_shadow import BitcoinMeanRevShadowStrategy
from strategies.sports_model import SportsModelStrategy
from strategies.weather import WeatherForecastStrategy
from strategies.weather_model import WeatherModelStrategy
from strategies.weather_model_v2 import WeatherModelStrategyV2
from strategies.bundle_arb import BundleArbitrageStrategy
from engine.paper_trader import PaperTrader
from engine.slippage import SlippageModel
from engine.ab_tester import ABTester
from engine.health_monitor import HealthMonitor
from runtime_paths import LOG_DIR, STATE_PATH

logger = logging.getLogger(__name__)
BITCOIN_SIGNAL_PATTERN = re.compile(r"\b(?:btc|bitcoin)\b", re.IGNORECASE)
LEGACY_DIAGNOSTICS_LOG_PATH = LOG_DIR / "diagnostics.jsonl"
LEGACY_APP_LOG_PATH = LOG_DIR / "app.log"
OVERLAY_SIGNAL_BUFFER_LIMIT = 120
OVERLAY_SEED_CONFIDENCE = 0.55
OVERLAY_SEED_MIN_SIZE_USD = 10.0
OVERLAY_SEED_MAX_SIZE_USD = 50.0
OVERLAY_SEED_POSITION_LIMIT = 20

COMPARISON_VIEW_CONFIG = {
    "all": {"label": "All", "strategy": None, "source": "all"},
    "weather_all": {
        "label": "Weather All",
        "strategy": "weather",
        "source": "weather_all",
        "signal_sources": (
            SignalSource.WEATHER_SNIPER.value,
            SignalSource.WEATHER_LATENCY.value,
            SignalSource.WEATHER_SWING.value,
        ),
    },
    "weather_sniper": {
        "label": "Sniper",
        "strategy": "weather",
        "source": SignalSource.WEATHER_SNIPER.value,
        "signal_sources": (SignalSource.WEATHER_SNIPER.value,),
    },
    "weather_latency": {
        "label": "Latency Hunter",
        "strategy": "weather",
        "source": SignalSource.WEATHER_LATENCY.value,
        "signal_sources": (SignalSource.WEATHER_LATENCY.value,),
    },
    "weather_swing": {
        "label": "Swing Trader",
        "strategy": "weather",
        "source": SignalSource.WEATHER_SWING.value,
        "signal_sources": (SignalSource.WEATHER_SWING.value,),
    },
    "weather_model_trader": {
        "label": "Weather ML Trader",
        "strategy": "weather_model_trader",
        "source": SignalSource.WEATHER_MODEL_TRADER.value,
        "signal_sources": (SignalSource.WEATHER_MODEL_TRADER.value,),
    },
    "weather_model_signal": {
        "label": "Weather ML Signal",
        "strategy": "weather_model_signal",
        "source": SignalSource.WEATHER_MODEL_SIGNAL.value,
        "signal_sources": (SignalSource.WEATHER_MODEL_SIGNAL.value,),
    },
    "weather_model_v2_trader": {
        "label": "Weather ML V2 Trader",
        "strategy": "weather_model_v2_trader",
        "source": SignalSource.WEATHER_MODEL_V2_TRADER.value,
        "signal_sources": (SignalSource.WEATHER_MODEL_V2_TRADER.value,),
    },
    "weather_model_v2_signal": {
        "label": "Weather ML V2 Signal",
        "strategy": "weather_model_v2_signal",
        "source": SignalSource.WEATHER_MODEL_V2_SIGNAL.value,
        "signal_sources": (SignalSource.WEATHER_MODEL_V2_SIGNAL.value,),
    },
    "news": {"label": "News", "strategy": "news", "source": "news_latency"},
    "news_whale": {"label": "News + Whale", "strategy": "news", "source": "news_whale"},
    "bitcoin": {"label": "Bitcoin", "strategy": "crypto_arb", "source": "crypto_temporal_arb"},
    "bitcoin_model": {
        "label": "Bitcoin ML",
        "strategy": "bitcoin_model",
        "source": SignalSource.BITCOIN_MODEL.value,
        "signal_sources": (SignalSource.BITCOIN_MODEL.value,),
    },
    "bitcoin_meanrev_shadow": {
        "label": "BTC MeanRev Shadow",
        "strategy": "bitcoin_meanrev_shadow",
        "source": SignalSource.BITCOIN_MEANREV_SHADOW.value,
        "signal_sources": (),
    },
    "sports": {
        "label": "Sports",
        "strategy": "sports_model",
        "source": SignalSource.SPORTS_MODEL.value,
        "signal_sources": (SignalSource.SPORTS_MODEL.value,),
    },
    "bitcoin_whale": {
        "label": "Bitcoin + Whale",
        "strategy": "crypto_arb",
        "source": "bitcoin_whale",
    },
    "arbitrage": {"label": "Arbitrage", "strategy": "arbitrage", "source": "multi_outcome_arbitrage"},
    "bundle_arb": {"label": "Bundle Arb", "strategy": "bundle_arb", "source": SignalSource.BUNDLE_ARB.value},
    "whale_follow": {"label": "Whale Follow", "strategy": "whale", "source": SignalSource.WHALE.value},
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
            "weather": WeatherForecastStrategy(
                self.config,
                collector=self.collector,
                state_path=str(STATE_PATH.with_name("weather_state.json")),
            ),
        }
        self.bundle_arb_strategy = BundleArbitrageStrategy(self.config)
        self.comparison_only_strategies = {
            "weather_model": WeatherModelStrategy(
                self.config,
                weather_strategy=self.strategies["weather"],
            ),
            "weather_model_v2": WeatherModelStrategyV2(
                self.config,
                weather_strategy=self.strategies["weather"],
            ),
            "bitcoin_model": BitcoinModelStrategy(
                self.config,
                crypto_strategy=self.strategies["crypto_arb"],
            ),
            "bitcoin_meanrev_shadow": BitcoinMeanRevShadowStrategy(self.config),
            "sports_model": SportsModelStrategy(self.config),
        }

        # Engine
        self.trader = PaperTrader(
            starting_capital=self.config.risk.max_total_exposure_usd,
            log_dir=str(LOG_DIR),
            state_path=str(STATE_PATH),
        )
        self.comparison_traders = {
            view_key: PaperTrader(
                starting_capital=self._comparison_starting_capital(view_key),
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
        self._recent_overlay_signals: dict[str, list[Signal]] = {
            "news": [],
            "crypto_arb": [],
        }
        self._errors: list[str] = []
        self._scan_count = 0
        self._latest_diagnostics: dict = {}
        self._cached_state: dict | None = None

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
        weather: WeatherForecastStrategy = self.strategies["weather"]
        weather.save_state()
        for strategy in self.comparison_only_strategies.values():
            close = getattr(strategy, "close", None)
            if close is None:
                continue
            result = close()
            if asyncio.iscoroutine(result):
                await result
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

            try:
                _bundle_start = _time.time()
                bundle_signals = await self.bundle_arb_strategy.scan(self._markets, self._events)
                _bundle_dur = (_time.time() - _bundle_start) * 1000
                strategy_signals["bundle_arb"] = bundle_signals
                self.health.record_strategy_run("bundle_arb", len(bundle_signals), _bundle_dur)
            except Exception as e:
                logger.error(f"Strategy bundle_arb error: {e}")
                self.bundle_arb_strategy._stats["errors"] += 1
                strategy_signals["bundle_arb"] = []
                self.health.record_strategy_error("bundle_arb", str(e))

            weather_model: WeatherModelStrategy = self.comparison_only_strategies["weather_model"]
            try:
                _weather_model_start = _time.time()
                weather_model_outputs = weather_model.scan_variants()
                _weather_model_dur = (_time.time() - _weather_model_start) * 1000
                for key, value in weather_model_outputs.items():
                    strategy_signals[key] = value
                self.health.record_strategy_run(
                    "weather_model",
                    sum(len(value) for value in weather_model_outputs.values()),
                    _weather_model_dur,
                )
            except Exception as e:
                logger.error(f"Strategy weather_model error: {e}")
                weather_model._stats["errors"] += 1
                strategy_signals["weather_model_trader"] = []
                strategy_signals["weather_model_signal"] = []
                self.health.record_strategy_error("weather_model", str(e))

            weather_model_v2: WeatherModelStrategyV2 = self.comparison_only_strategies["weather_model_v2"]
            try:
                _weather_model_v2_start = _time.time()
                weather_model_v2_outputs = weather_model_v2.scan_variants()
                _weather_model_v2_dur = (_time.time() - _weather_model_v2_start) * 1000
                for key, value in weather_model_v2_outputs.items():
                    strategy_signals[key] = value
                self.health.record_strategy_run(
                    "weather_model_v2",
                    sum(len(value) for value in weather_model_v2_outputs.values()),
                    _weather_model_v2_dur,
                )
            except Exception as e:
                logger.error(f"Strategy weather_model_v2 error: {e}")
                weather_model_v2._stats["errors"] += 1
                strategy_signals["weather_model_v2_trader"] = []
                strategy_signals["weather_model_v2_signal"] = []
                self.health.record_strategy_error("weather_model_v2", str(e))

            bitcoin_model: BitcoinModelStrategy = self.comparison_only_strategies["bitcoin_model"]
            try:
                _bitcoin_model_start = _time.time()
                bitcoin_model_signals = await bitcoin_model.scan(self._markets, self._events)
                _bitcoin_model_dur = (_time.time() - _bitcoin_model_start) * 1000
                strategy_signals["bitcoin_model"] = bitcoin_model_signals
                self.health.record_strategy_run(
                    "bitcoin_model",
                    len(bitcoin_model_signals),
                    _bitcoin_model_dur,
                )
            except Exception as e:
                logger.error(f"Strategy bitcoin_model error: {e}")
                bitcoin_model._stats["errors"] += 1
                strategy_signals["bitcoin_model"] = []
                self.health.record_strategy_error("bitcoin_model", str(e))

            bitcoin_meanrev_shadow: BitcoinMeanRevShadowStrategy = self.comparison_only_strategies["bitcoin_meanrev_shadow"]
            try:
                _bitcoin_shadow_start = _time.time()
                bitcoin_shadow_signals = await bitcoin_meanrev_shadow.scan(self._markets, self._events)
                _bitcoin_shadow_dur = (_time.time() - _bitcoin_shadow_start) * 1000
                strategy_signals["bitcoin_meanrev_shadow"] = bitcoin_shadow_signals
                self.health.record_strategy_run(
                    "bitcoin_meanrev_shadow",
                    len(bitcoin_shadow_signals),
                    _bitcoin_shadow_dur,
                )
            except Exception as e:
                logger.error(f"Strategy bitcoin_meanrev_shadow error: {e}")
                bitcoin_meanrev_shadow._stats["errors"] += 1
                strategy_signals["bitcoin_meanrev_shadow"] = []
                self.health.record_strategy_error("bitcoin_meanrev_shadow", str(e))

            sports_model: SportsModelStrategy = self.comparison_only_strategies["sports_model"]
            try:
                _sports_model_start = _time.time()
                sports_model_signals = await sports_model.scan(self._markets, self._events)
                _sports_model_dur = (_time.time() - _sports_model_start) * 1000
                strategy_signals["sports_model"] = sports_model_signals
                self.health.record_strategy_run(
                    "sports_model",
                    len(sports_model_signals),
                    _sports_model_dur,
                )
            except Exception as e:
                logger.error(f"Strategy sports_model error: {e}")
                sports_model._stats["errors"] += 1
                strategy_signals["sports_model"] = []
                self.health.record_strategy_error("sports_model", str(e))

            # Skip per-signal whale confirmation (was making 600+ API calls per scan)
            # Whale data is still loaded and available for the dashboard
            # TODO: implement cached whale sentiment lookup instead of per-signal API calls
            self._refresh_recent_overlay_signals("news", strategy_signals.get("news", []))
            self._refresh_recent_overlay_signals("crypto_arb", strategy_signals.get("crypto_arb", []))

            # Highest-confidence signals get first claim on remaining capacity.
            all_signals.sort(key=lambda s: s.confidence, reverse=True)
            all_signals, filtered = self.trader.select_candidate_signals(all_signals)
            if filtered:
                logger.info(
                    "[PIPELINE] Filtered signals before execution: "
                    + ", ".join(f"{reason}={count}" for reason, count in filtered.items())
                )

            early_arb_releases = self.trader.release_locked_arb_positions(all_signals)
            if early_arb_releases:
                exit_count += early_arb_releases

            executed = 0
            executed_by_strategy: dict[str, int] = {}
            for signal in all_signals:
                trade = self.trader.execute_signal(signal, current_prices)
                if trade:
                    executed += 1
                    source = trade.source.value
                    executed_by_strategy[source] = executed_by_strategy.get(source, 0) + 1

            whale: WhaleTrackingStrategy = self.strategies["whale"]
            for view_key, meta in COMPARISON_VIEW_CONFIG.items():
                strategy_name = meta["strategy"]
                if strategy_name is None:
                    self._latest_comparison_signals[view_key] = all_signals[:30]
                    continue

                trader = self.comparison_traders[view_key]
                view_signals = self._build_comparison_signals(
                    view_key=view_key,
                    strategy_name=strategy_name,
                    strategy_signals=strategy_signals,
                    whale_strategy=whale,
                )
                view_signals.sort(key=lambda s: s.confidence, reverse=True)
                view_signals, _ = trader.select_candidate_signals(view_signals)
                self._latest_comparison_signals[view_key] = view_signals[:30]
                if view_key == "bitcoin_meanrev_shadow":
                    continue
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
            weather: WeatherForecastStrategy = self.strategies["weather"]
            weather.save_state()
            self._cached_state = self._build_state()

    async def reset_state(self):
        """Reset portfolio and per-strategy transient state without deleting persisted logs."""
        async with self._cycle_lock:
            self.trader.reset(self.config.risk.max_total_exposure_usd)
            self._all_signals = []
            self._latest_comparison_signals = {
                key: [] for key in COMPARISON_VIEW_CONFIG
            }
            self._recent_overlay_signals = {
                "news": [],
                "crypto_arb": [],
            }
            self._errors = []
            self._scan_count = 0
            self._latest_diagnostics = {}
            self.dashboard_state = DashboardState(mode=self.config.mode)
            for view_key, trader in self.comparison_traders.items():
                trader.reset(self._comparison_starting_capital(view_key))

            news: NewsLatencyStrategy = self.strategies["news"]
            news._processed_headlines.clear()
            news._pending_headlines.clear()
            news._recent_headlines.clear()
            news._api_calls_this_hour = 0
            news._hour_start = datetime.now(timezone.utc)
            news._anthropic_pause_until = None
            news._anthropic_last_error = ""
            news._market_index.clear()

            mean_reversion: MeanReversionStrategy = self.strategies["mean_reversion"]
            mean_reversion._baselines.clear()

            crypto: CryptoTemporalArbStrategy = self.strategies["crypto_arb"]
            crypto._price_history = {"BTC": [], "ETH": [], "SOL": []}
            crypto._matched_markets = {}
            crypto._last_market_scan = 0

            weather: WeatherForecastStrategy = self.strategies["weather"]
            weather.reset_state()

            self.health = HealthMonitor(log_dir=str(LOG_DIR))
            await self._refresh_data()
            self.trader.save_state()
            self._cached_state = self._build_state()
            logger.info("[PIPELINE] Reset live paper trading state")

    def _filter_comparison_signals(self, *, view_key: str, signals: list[Signal]) -> list[Signal]:
        filtered = list(signals)
        allowed_sources = COMPARISON_VIEW_CONFIG.get(view_key, {}).get("signal_sources")
        if allowed_sources:
            filtered = [signal for signal in filtered if signal.source.value in allowed_sources]
        if view_key == "bitcoin":
            filtered = [
                signal for signal in filtered
                if BITCOIN_SIGNAL_PATTERN.search(signal.market_slug)
                or BITCOIN_SIGNAL_PATTERN.search(signal.reasoning)
            ]
        return filtered

    def _build_comparison_signals(
        self,
        *,
        view_key: str,
        strategy_name: str,
        strategy_signals: dict[str, list[Signal]],
        whale_strategy: WhaleTrackingStrategy,
    ) -> list[Signal]:
        base_signals = strategy_signals.get(strategy_name, [])
        if view_key == "news_whale":
            source_signals = self._merge_overlay_signal_sets(
                self._recent_overlay_signals.get("news", []),
                self._build_overlay_seed_signals("news"),
            )
            return self._apply_whale_overlay(source_signals, whale_strategy)
        if view_key == "bitcoin_whale":
            bitcoin_signals = self._filter_comparison_signals(
                view_key="bitcoin",
                signals=self._merge_overlay_signal_sets(
                    self._recent_overlay_signals.get("crypto_arb", []),
                    self._build_overlay_seed_signals("bitcoin"),
                ),
            )
            return self._apply_whale_overlay(bitcoin_signals, whale_strategy)
        if view_key == "whale_follow":
            return whale_strategy.build_standalone_signals(self._markets)
        return self._filter_comparison_signals(view_key=view_key, signals=base_signals)

    @staticmethod
    def _apply_whale_overlay(
        signals: list[Signal],
        whale_strategy: WhaleTrackingStrategy,
    ) -> list[Signal]:
        overlaid: list[Signal] = []
        for signal in signals:
            adjusted, applied = whale_strategy.apply_cached_confirmation(signal)
            if applied:
                overlaid.append(adjusted)
        return overlaid

    def _refresh_recent_overlay_signals(self, strategy_name: str, new_signals: list[Signal]) -> None:
        ttl = timedelta(minutes=self.config.whale.signal_ttl_minutes)
        cutoff = datetime.now(timezone.utc) - ttl
        existing = [
            signal
            for signal in self._recent_overlay_signals.get(strategy_name, [])
            if signal.timestamp >= cutoff
        ]
        merged: dict[tuple[str, str], Signal] = {}
        for signal in existing + list(new_signals):
            key = (signal.condition_id, signal.action.value)
            current = merged.get(key)
            if current is None or signal.timestamp > current.timestamp:
                merged[key] = signal
        ordered = sorted(merged.values(), key=lambda signal: signal.timestamp, reverse=True)
        self._recent_overlay_signals[strategy_name] = ordered[:OVERLAY_SIGNAL_BUFFER_LIMIT]

    @staticmethod
    def _merge_overlay_signal_sets(*signal_sets: list[Signal]) -> list[Signal]:
        merged: dict[tuple[str, str], Signal] = {}
        for signal_set in signal_sets:
            for signal in signal_set or []:
                key = (signal.condition_id, signal.action.value)
                current = merged.get(key)
                if current is None or signal.timestamp > current.timestamp:
                    merged[key] = signal
        return sorted(merged.values(), key=lambda signal: signal.timestamp, reverse=True)

    def _build_overlay_seed_signals(self, base_view_key: str) -> list[Signal]:
        trader = self.comparison_traders.get(base_view_key)
        if not trader:
            return []

        now = datetime.now(timezone.utc)
        seeds: dict[tuple[str, str], Signal] = {}
        positions = sorted(
            trader.portfolio.positions,
            key=lambda position: position.opened_at,
            reverse=True,
        )
        for position in positions[:OVERLAY_SEED_POSITION_LIMIT]:
            if position.side not in {"YES", "NO"}:
                continue
            token_id = position.token_id or None
            if not token_id or token_id in {"HEDGE", "ARB_ALL"}:
                continue
            action = SignalAction.BUY_YES if position.side == "YES" else SignalAction.BUY_NO
            key = (position.condition_id, action.value)
            suggested_size_usd = min(
                max(position.shares * max(position.avg_entry_price, 0.01), OVERLAY_SEED_MIN_SIZE_USD),
                OVERLAY_SEED_MAX_SIZE_USD,
            )
            seeds[key] = Signal(
                source=position.source,
                action=action,
                market_slug=position.market_slug,
                condition_id=position.condition_id,
                token_id=token_id,
                confidence=OVERLAY_SEED_CONFIDENCE,
                expected_edge=0.0,
                reasoning=f"Whale overlay seed from active {base_view_key} sleeve position",
                suggested_size_usd=suggested_size_usd,
                group_key=position.group_key,
                timestamp=now,
            )
        return list(seeds.values())

    def _comparison_starting_capital(self, view_key: str) -> float:
        if view_key == "weather_all":
            return self.config.weather.combined_budget_usd
        if view_key == "weather_sniper":
            return self.config.weather.sniper_budget_usd
        if view_key == "weather_latency":
            return self.config.weather.latency_budget_usd
        if view_key == "weather_swing":
            return self.config.weather.swing_budget_usd
        if view_key == "weather_model_trader":
            return self.config.weather_model.trader_budget_usd
        if view_key == "weather_model_signal":
            return self.config.weather_model.signal_budget_usd
        if view_key == "weather_model_v2_trader":
            return self.config.weather_model_v2.trader_budget_usd
        if view_key == "weather_model_v2_signal":
            return self.config.weather_model_v2.signal_budget_usd
        if view_key == "bitcoin_model":
            return self.config.bitcoin_model.budget_usd
        if view_key == "bitcoin_meanrev_shadow":
            return self.config.bitcoin_meanrev_shadow.budget_usd
        if view_key == "sports":
            return self.config.sports_model.budget_usd
        return self.config.risk.max_total_exposure_usd

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
            weather: WeatherForecastStrategy = self.strategies["weather"]
            extra_weather_markets = await weather.get_supplemental_markets()
            if extra_weather_markets:
                seen_conditions = {m.condition_id for m in self._markets}
                for market in extra_weather_markets:
                    if market.condition_id not in seen_conditions:
                        self._markets.append(market)
                        seen_conditions.add(market.condition_id)
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
        if self._cached_state is not None:
            return self._cached_state

        return self._build_state()

    def _build_state(self) -> dict:
        """Build the full dashboard state payload."""
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
            if view_key == "bitcoin_meanrev_shadow":
                comparison_views[view_key] = self.comparison_only_strategies["bitcoin_meanrev_shadow"].serialize_view()
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
            } | {
                "bundle_arb": self.bundle_arb_strategy.stats,
                "weather_model": self.comparison_only_strategies["weather_model"].stats,
                "weather_model_v2": self.comparison_only_strategies["weather_model_v2"].stats,
                "bitcoin_model": self.comparison_only_strategies["bitcoin_model"].stats,
                "bitcoin_meanrev_shadow": self.comparison_only_strategies["bitcoin_meanrev_shadow"].stats,
                "sports_model": self.comparison_only_strategies["sports_model"].stats,
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

    def llm_context(self) -> dict:
        """Build a compact, legacy-only diagnostic payload for the old engine consultant."""
        state = self.get_state()
        diagnostics = state.get("diagnostics") or {}
        scan_tape = self._read_recent_diagnostics(limit=24)
        strategy_rollup = self._aggregate_rollup(scan_tape, key="signals_by_strategy")
        rejection_rollup = self._aggregate_rollup(scan_tape, key="filtered_signals")
        trade_tape = [self._serialize_trade_row(trade) for trade in self.trader.trade_log[-60:]]
        close_tape = [trade for trade in trade_tape if trade.get("realized_pnl") is not None][-30:]
        blockers = self._legacy_blockers(state)
        current_blocker_summary = self._current_blocker_summary(state)
        recent_pattern_summary = self._recent_pattern_summary(scan_tape)

        comparison_summary = []
        for key, view in (state.get("comparison_views") or {}).items():
            portfolio = view.get("portfolio") or {}
            performance = view.get("performance") or {}
            comparison_summary.append(
                {
                    "key": key,
                    "label": view.get("label"),
                    "source": view.get("source"),
                    "total_value": portfolio.get("total_value", 0.0),
                    "cash": portfolio.get("cash", 0.0),
                    "positions_value": portfolio.get("positions_value", 0.0),
                    "total_pnl": performance.get("total_pnl", portfolio.get("total_pnl", 0.0)),
                    "win_rate": performance.get("win_rate", portfolio.get("win_rate", 0.0)),
                    "total_trades": performance.get("total_trades", portfolio.get("total_trades", 0)),
                    "open_positions": len(portfolio.get("positions") or []),
                    "signals": len(view.get("signals") or []),
                }
            )

        return {
            "scope": "legacy_engine_only",
            "notes": {
                "current_state": "summary, portfolio, performance, health, diagnostics, comparison_summary, blockers, current_blocker_summary",
                "recent_window": "scan_tape, trade_tape, close_tape, strategy_rollup, rejection_rollup, recent_pattern_summary",
                "separation": "This payload is legacy-engine only. Do not infer anything from Opus or multi-agent runtime data.",
            },
            "summary": {
                "mode": state.get("mode"),
                "uptime_human": state.get("uptime_human"),
                "scan_count": state.get("scan_count", 0),
                "active_markets": state.get("active_markets", 0),
                "error_count": len(state.get("errors") or []),
            },
            "warm_start": {
                "active": int(state.get("scan_count", 0) or 0) < 3,
                "reason": "Treat current blockers as provisional during the first few scans after restart.",
            },
            "portfolio": state.get("portfolio"),
            "performance": state.get("performance"),
            "health": state.get("health"),
            "latest_scan": diagnostics,
            "blockers": blockers,
            "current_blocker_summary": current_blocker_summary,
            "strategies": state.get("strategies"),
            "recent_news": state.get("recent_news", [])[:20],
            "comparison_summary": comparison_summary,
            "scan_tape": scan_tape,
            "trade_tape": trade_tape,
            "close_tape": close_tape,
            "strategy_rollup": strategy_rollup,
            "rejection_rollup": rejection_rollup,
            "recent_pattern_summary": recent_pattern_summary,
            "policy_snapshot": {
                "scan_interval_secs": self.config.scan_interval_secs,
                "starting_capital": self.config.risk.max_total_exposure_usd,
                "weather_budgets": {
                    "combined": self.config.weather.combined_budget_usd,
                    "sniper": self.config.weather.sniper_budget_usd,
                    "latency": self.config.weather.latency_budget_usd,
                    "swing": self.config.weather.swing_budget_usd,
                },
                "weather_model_budgets": {
                    "trader": self.config.weather_model.trader_budget_usd,
                    "signal": self.config.weather_model.signal_budget_usd,
                },
                "weather_model_v2_budgets": {
                    "trader": self.config.weather_model_v2.trader_budget_usd,
                    "signal": self.config.weather_model_v2.signal_budget_usd,
                },
                "bitcoin_model_budget": self.config.bitcoin_model.budget_usd,
                "bitcoin_meanrev_shadow_budget": self.config.bitcoin_meanrev_shadow.budget_usd,
            },
            "diagnostics_log_path": str(LEGACY_DIAGNOSTICS_LOG_PATH),
            "app_log_path": str(LEGACY_APP_LOG_PATH),
            "state_path": str(STATE_PATH),
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
                "starting_capital": round(trader.portfolio.starting_capital, 2),
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

    def _read_recent_diagnostics(self, *, limit: int = 24) -> list[dict]:
        if not LEGACY_DIAGNOSTICS_LOG_PATH.exists():
            return []
        try:
            lines = LEGACY_DIAGNOSTICS_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            logger.warning("[PIPELINE] Failed to read diagnostics log for consult: %s", exc)
            return []

        entries: list[dict] = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    @staticmethod
    def _aggregate_rollup(scan_tape: list[dict], *, key: str) -> list[dict]:
        totals: dict[str, int] = {}
        for scan in scan_tape:
            for name, count in (scan.get(key) or {}).items():
                totals[str(name)] = totals.get(str(name), 0) + int(count or 0)
        return [
            {"name": name, "count": count}
            for name, count in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:20]
        ]

    @staticmethod
    def _serialize_trade_row(trade) -> dict:
        return {
            "id": trade.id,
            "timestamp": trade.timestamp.isoformat(),
            "source": trade.source.value,
            "market_slug": trade.market_slug,
            "side": trade.side.value,
            "price": round(trade.price, 4),
            "shares": round(trade.size_shares, 4),
            "size_usd": round(trade.size_usd, 2),
            "fees": round(getattr(trade, "fees_paid", 0.0) or 0.0, 4),
            "exit_price": round(trade.exit_price, 4) if getattr(trade, "exit_price", None) is not None else None,
            "realized_pnl": round(trade.realized_pnl, 4) if trade.realized_pnl is not None else None,
        }

    @staticmethod
    def _legacy_blockers(state: dict) -> list[dict[str, str]]:
        diagnostics = state.get("diagnostics") or {}
        health = state.get("health") or {}
        blockers: list[dict[str, str]] = []

        if not diagnostics or diagnostics.get("scan") is None:
            blockers.append(
                {
                    "title": "Legacy runtime warming up",
                    "detail": "No completed legacy diagnostic scan is available yet after startup.",
                }
            )
            return blockers

        if (health.get("overall_status") or "").lower() not in {"", "healthy"}:
            blockers.append(
                {
                    "title": "System health degraded",
                    "detail": f"Legacy engine health is {health.get('overall_status', 'unknown')}.",
                }
            )

        filtered = diagnostics.get("filtered_signals") or {}
        if diagnostics.get("executed", 0) == 0:
            if filtered:
                top_reason, count = max(filtered.items(), key=lambda item: item[1])
                blockers.append(
                    {
                        "title": "Signals are being filtered",
                        "detail": f"Latest scan rejected {count} signals because of {top_reason}.",
                    }
                )
            elif sum((diagnostics.get("signals_by_strategy") or {}).values()) == 0:
                blockers.append(
                    {
                        "title": "No strategies emitted candidates",
                        "detail": f"{diagnostics.get('markets_tradeable', 0)} tradeable markets were scanned but no strategy produced a live signal.",
                    }
                )

        exposure = ((diagnostics.get("portfolio") or {}).get("exposure_by_source")) or {}
        total_exposure = sum(float(value or 0.0) for value in exposure.values())
        if total_exposure > 0:
            source, value = max(exposure.items(), key=lambda item: item[1])
            share = (float(value or 0.0) / total_exposure) if total_exposure else 0.0
            if share >= 0.8:
                blockers.append(
                    {
                        "title": "One strategy dominates exposure",
                        "detail": f"{source} currently represents {share * 100:.0f}% of deployed legacy capital.",
                    }
                )

        if not blockers:
            blockers.append(
                {
                    "title": "No dominant blocker detected",
                    "detail": "The latest legacy scan did not surface a single obvious failure mode.",
                }
            )
        return blockers[:6]

    @staticmethod
    def _current_blocker_summary(state: dict) -> dict:
        diagnostics = state.get("diagnostics") or {}
        if not diagnostics or diagnostics.get("scan") is None:
            return {
                "authoritative": True,
                "scan": None,
                "executed": 0,
                "markets_tradeable": 0,
                "total_signals": 0,
                "top_filtered_reason": None,
                "top_filtered_count": 0,
                "blockers": Pipeline._legacy_blockers(state),
                "instruction": "No current scan is available yet. Treat this as warm-start state, not a settled blocker diagnosis.",
            }
        filtered = diagnostics.get("filtered_signals") or {}
        signals_by_strategy = diagnostics.get("signals_by_strategy") or {}
        total_signals = sum(int(value or 0) for value in signals_by_strategy.values())
        top_filtered = None
        if filtered:
            top_filtered = max(filtered.items(), key=lambda item: item[1])
        return {
            "authoritative": True,
            "scan": diagnostics.get("scan"),
            "executed": diagnostics.get("executed", 0),
            "markets_tradeable": diagnostics.get("markets_tradeable", 0),
            "total_signals": total_signals,
            "top_filtered_reason": top_filtered[0] if top_filtered else None,
            "top_filtered_count": int(top_filtered[1]) if top_filtered else 0,
            "blockers": Pipeline._legacy_blockers(state),
            "instruction": "Use this section first for the current blocker. Treat recent rollups only as background pattern.",
        }

    @staticmethod
    def _recent_pattern_summary(scan_tape: list[dict]) -> dict:
        if not scan_tape:
            return {
                "window_scans": 0,
                "avg_executed": 0.0,
                "top_recent_filter": None,
                "top_recent_filter_count": 0,
            }

        total_executed = sum(int(scan.get("executed", 0) or 0) for scan in scan_tape)
        total_signals = sum(
            sum(int(value or 0) for value in (scan.get("signals_by_strategy") or {}).values())
            for scan in scan_tape
        )
        filter_totals: dict[str, int] = {}
        for scan in scan_tape:
            for reason, count in (scan.get("filtered_signals") or {}).items():
                filter_totals[str(reason)] = filter_totals.get(str(reason), 0) + int(count or 0)
        top_recent = max(filter_totals.items(), key=lambda item: item[1]) if filter_totals else None
        return {
            "window_scans": len(scan_tape),
            "avg_executed": round(total_executed / len(scan_tape), 2),
            "avg_signals": round(total_signals / len(scan_tape), 2),
            "top_recent_filter": top_recent[0] if top_recent else None,
            "top_recent_filter_count": int(top_recent[1]) if top_recent else 0,
            "instruction": "Use this only for recent trend context, not as proof of the current blocker.",
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
