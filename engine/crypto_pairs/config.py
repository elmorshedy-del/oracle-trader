"""Shared configuration objects for the crypto pairs lane."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BINANCE_SPOT_WS_URLS = (
    "wss://stream.binance.com:9443/stream",
    "wss://data-stream.binance.vision/stream",
)
DEFAULT_BAR_INTERVAL_SECONDS = 1
DEFAULT_MAX_LEG_LAG_MS = 1_500
DEFAULT_RECONNECT_DELAY_SECONDS = 5
DEFAULT_WARMUP_SECONDS = 3_600
DEFAULT_STATS_SAMPLE_SECONDS = 3_600
DEFAULT_PAPER_FEE_BPS = 1.0
DEFAULT_PAPER_SLIPPAGE_BPS = 0.5
DEFAULT_PAPER_QTY_PRECISION = 8
DEFAULT_DISCOVERY_PROJECT_ROOT = "research/crypto_pairs/projects/crypto-pairs-v1"
DEFAULT_LOG_ROOT = "output/crypto_pairs/sessions"


@dataclass(slots=True)
class PairRuntimeConfig:
    pair_key: str
    token_a: str
    token_b: str
    lookback_seconds: int
    halflife_hours: float
    discovery_score: float
    spread_mean: float
    spread_std: float


@dataclass(slots=True)
class PriceStreamerConfig:
    ws_urls: tuple[str, ...] = DEFAULT_BINANCE_SPOT_WS_URLS
    bar_interval_seconds: int = DEFAULT_BAR_INTERVAL_SECONDS
    reconnect_delay_seconds: int = DEFAULT_RECONNECT_DELAY_SECONDS


@dataclass(slots=True)
class SignalConfig:
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0
    max_hold_seconds: int = 21_600
    cooldown_seconds: int = 60


@dataclass(slots=True)
class RiskConfig:
    max_positions: int = 5
    max_capital_per_pair_pct: float = 0.20
    max_total_exposure_pct: float = 0.80
    max_daily_loss_pct: float = 0.03
    max_correlation_overlap: int = 2


@dataclass(slots=True)
class ExecutionConfig:
    fee_bps: float = DEFAULT_PAPER_FEE_BPS
    slippage_bps: float = DEFAULT_PAPER_SLIPPAGE_BPS
    quantity_precision: int = DEFAULT_PAPER_QTY_PRECISION
    paper_trade: bool = True


@dataclass(slots=True)
class ShadowRunnerConfig:
    total_capital: float = 10_000.0
    top_pairs: int = 5
    runtime_seconds: int | None = None
