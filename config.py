"""
Polymarket Algo Trading Pipeline — Configuration
=================================================
All tunable parameters in one place. Override with environment variables.
"""

import os
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_host: str = "https://data-api.polymarket.com"


@dataclass
class WalletConfig:
    private_key: str = os.getenv("POLY_PRIVATE_KEY", "")
    funder_address: str = os.getenv("POLY_FUNDER_ADDRESS", "")
    chain_id: int = 137


@dataclass
class LiquidityProvisionConfig:
    """Hedged liquidity provision (Layer 1 — the salary)."""
    enabled: bool = True
    # Max spread from mid to qualify for rewards
    max_spread_cents: int = 3
    # Target distance from midpoint (0 = at mid, higher = safer but less reward)
    target_distance_cents: float = 0.5
    # Min shares per order
    min_shares: int = 10
    # Max overpayment above $1.00 for the hedge (Δ_max from the paper)
    max_overpayment: float = 0.02
    # Market selection filters
    prefer_price_near_50: bool = True
    max_volatility_24h: float = 0.15
    min_reward_pool_usd: float = 50.0
    # Kelly fraction cap (never bet more than this fraction of capital)
    kelly_fraction_cap: float = 0.25


@dataclass
class ArbitrageConfig:
    """Multi-outcome arbitrage (Layer 2 — the bonus)."""
    enabled: bool = True
    # Min profit after fees to trigger
    min_profit_cents: float = 2.0
    # Per-trade arb sizing cap (kept separate from global directional max position sizing)
    max_position_usd: float = float(os.getenv("ARB_MAX_POSITION_USD", "150"))
    # Max number of outcomes in a multi-outcome market to consider
    max_outcomes: int = 20
    # Min liquidity per outcome
    min_liquidity_usd: float = 1000.0
    # Execution timeout — abort if can't fill all legs in N seconds
    execution_timeout_secs: int = 10


@dataclass
class BundleArbitrageConfig:
    """Strict bundle arbitrage experiment (comparison-book only)."""
    enabled: bool = True
    min_profit_cents: float = 3.0
    max_position_usd: float = float(os.getenv("BUNDLE_ARB_MAX_POSITION_USD", "200"))
    max_outcomes: int = 12
    min_liquidity_usd: float = 1500.0
    min_event_volume_usd: float = 10000.0
    require_neg_risk: bool = False


@dataclass
class WhaleTrackingConfig:
    """Whale wallet tracking (Layer 3 — the advisor)."""
    enabled: bool = True
    # Min historical PnL to qualify as a "whale"
    min_pnl_usd: float = 1000.0
    # Min win rate (0.0 - 1.0)
    min_win_rate: float = 0.40
    # How many top wallets to track
    top_n_wallets: int = 50
    # Refresh wallet rankings every N hours
    refresh_interval_hours: int = 24
    # Refresh recent whale activity cache every N minutes
    activity_refresh_minutes: int = 15
    # Number of whale wallets to inspect when building cached sentiment
    activity_wallet_limit: int = 20
    # Recent trades to fetch per whale wallet
    activity_trades_per_wallet: int = 25
    # Cached whale sentiment must be fresher than this to be used
    signal_ttl_minutes: int = 720
    # Minimum activity needed before whale sentiment can influence another strategy
    overlay_min_whales: int = 1
    overlay_min_total_size: float = 10.0
    # Standalone whale experiment settings (comparison-book only)
    standalone_enabled: bool = True
    standalone_min_whales: int = 1
    standalone_min_total_size: float = 25.0
    standalone_min_confidence: float = 0.54
    standalone_max_entry_price: float = 0.75
    standalone_min_size_usd: float = 15.0
    standalone_max_size_usd: float = 100.0
    # Confidence boost multiplier when whale confirms a signal
    confirmation_boost: float = 1.5


@dataclass
class NewsConfig:
    """News-to-price latency engine (optional — requires LLM API key)."""
    enabled: bool = bool(os.getenv("FIREWORKS_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
    primary_provider: str = os.getenv(
        "NEWS_LLM_PRIMARY_PROVIDER",
        "fireworks" if os.getenv("FIREWORKS_API_KEY", "") else "anthropic",
    )
    fallback_provider: str = os.getenv("NEWS_LLM_FALLBACK_PROVIDER", "")
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    fireworks_model: str = os.getenv("FIREWORKS_NEWS_MODEL", "accounts/fireworks/models/glm-5")
    escalation_model: str = os.getenv("ANTHROPIC_ESCALATION_MODEL", "claude-sonnet-4-6")
    # RSS / news sources
    rss_feeds: list = field(default_factory=lambda: [
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://feeds.npr.org/1004/rss.xml",
        "https://feeds.npr.org/1006/rss.xml",
        "https://feeds.npr.org/1014/rss.xml",
        "https://www.theguardian.com/world/rss",
        "https://www.theguardian.com/us/business/rss",
        "https://www.theguardian.com/us-news/us-politics/rss",
    ])
    # Min confidence to generate a signal (0.0 - 1.0)
    min_confidence: float = 0.5  # lowered from 0.7 — let more signals through for paper testing
    # Scan interval in seconds
    scan_interval_secs: int = 30
    # Max API calls per hour (cost control)
    max_calls_per_hour: int = int(os.getenv("NEWS_MAX_CALLS_PER_HOUR", "24"))
    # Hard cap on candidate headlines we will send to the LLM in one scan
    max_headlines_per_scan: int = int(os.getenv("NEWS_MAX_HEADLINES_PER_SCAN", "4"))


@dataclass
class CryptoArbConfig:
    """Crypto temporal arbitrage — exploit exchange-to-Polymarket price lag."""
    enabled: bool = True
    # Min price move on exchange to trigger (0.3% = significant)
    min_move_pct: float = 0.003
    # Lookback window in seconds to measure the move
    lookback_seconds: int = 120  # 2 minutes
    # Max entry price (don't buy YES at 0.90, the edge is gone)
    max_entry_price: float = 0.75
    # Symbols to track
    symbols: list = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    # Structure strategy: exploit ladder / implication violations on barrier markets
    structure_enabled: bool = True
    structure_min_adjacent_edge: float = 0.025
    structure_min_equivalence_edge: float = 0.02
    structure_min_implication_edge: float = 0.04
    structure_max_entry_price: float = 0.80
    structure_min_size_usd: float = 8.0
    structure_max_size_usd: float = 35.0


@dataclass
class WeatherForecastConfig:
    """Weather forecast variants — Open-Meteo model consensus vs Polymarket."""
    enabled: bool = True
    cities: list = field(default_factory=lambda: [
        "new-york",
        "chicago",
        "miami",
        "los-angeles",
        "london",
        "seoul",
    ])
    forecast_refresh_secs: int = 120
    market_refresh_secs: int = 120
    forecast_days: int = 4
    model_agreement_max_spread_f: float = 3.6
    min_edge: float = 0.08

    sniper_budget_usd: float = float(os.getenv("WEATHER_SNIPER_BUDGET_USD", "150"))
    latency_budget_usd: float = float(os.getenv("WEATHER_LATENCY_BUDGET_USD", "150"))
    swing_budget_usd: float = float(os.getenv("WEATHER_SWING_BUDGET_USD", "150"))
    combined_budget_usd: float = float(os.getenv("WEATHER_COMBINED_BUDGET_USD", "450"))

    sniper_max_yes_price: float = 0.05
    sniper_min_prob: float = 0.94
    sniper_min_size_usd: float = 1.0
    sniper_max_size_usd: float = 3.0

    latency_min_probability_shift: float = 0.08
    latency_min_edge: float = 0.04
    latency_max_entry_price: float = 0.58
    latency_take_profit_price: float = 0.60
    latency_min_size_usd: float = 8.0
    latency_max_size_usd: float = 30.0

    swing_min_prob: float = 0.58
    swing_max_prob: float = 0.42
    swing_min_token_dip: float = 0.05
    swing_lookback_minutes: int = 180
    swing_min_edge: float = 0.04
    swing_min_size_usd: float = 6.0
    swing_max_size_usd: float = 24.0


@dataclass
class WeatherModelConfig:
    """External-only weather ML experiments (comparison-book only)."""
    enabled: bool = True
    model_dir: str = os.getenv(
        "WEATHER_MODEL_DIR",
        "models/weather_ml/external_only/legacy-weather-ml-v1",
    )

    trader_budget_usd: float = float(os.getenv("WEATHER_MODEL_TRADER_BUDGET_USD", "750"))
    signal_budget_usd: float = float(os.getenv("WEATHER_MODEL_SIGNAL_BUDGET_USD", "600"))

    trader_min_edge: float = 0.07
    trader_min_prob_distance: float = 0.10
    trader_max_token_price: float = 0.82
    trader_min_size_usd: float = 18.0
    trader_max_size_usd: float = 140.0

    signal_min_edge: float = 0.11
    signal_min_prob_distance: float = 0.18
    signal_max_token_price: float = 0.72
    signal_min_size_usd: float = 15.0
    signal_max_size_usd: float = 110.0


@dataclass
class BitcoinModelConfig:
    """Standalone BTC futures-impulse sleeve (comparison-book only)."""
    enabled: bool = os.getenv("BITCOIN_MODEL_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    model_dir: str = os.getenv(
        "BITCOIN_MODEL_DIR",
        "models/bitcoin_ml/impulse_baseline",
    )

    budget_usd: float = float(os.getenv("BITCOIN_MODEL_BUDGET_USD", "600"))
    long_threshold: float = float(os.getenv("BITCOIN_MODEL_LONG_THRESHOLD", "0.6552729109"))
    short_threshold: float = float(os.getenv("BITCOIN_MODEL_SHORT_THRESHOLD", "0.7616295043"))
    degraded_threshold: float = float(os.getenv("BITCOIN_MODEL_DEGRADED_THRESHOLD", "0.40"))
    min_direction_margin: float = float(os.getenv("BITCOIN_MODEL_DIRECTION_MARGIN", "0.04"))
    degraded_direction_margin: float = float(os.getenv("BITCOIN_MODEL_DEGRADED_DIRECTION_MARGIN", "0.015"))
    min_source_fresh_score: float = float(os.getenv("BITCOIN_MODEL_MIN_FRESH_SCORE", "0.50"))
    min_barrier_edge: float = float(os.getenv("BITCOIN_MODEL_MIN_BARRIER_EDGE", "0.10"))
    max_entry_price: float = float(os.getenv("BITCOIN_MODEL_MAX_ENTRY_PRICE", "0.82"))
    max_resolution_days: int = int(os.getenv("BITCOIN_MODEL_MAX_RESOLUTION_DAYS", "365"))
    max_barrier_distance_pct: float = float(os.getenv("BITCOIN_MODEL_MAX_BARRIER_DISTANCE_PCT", "1.00"))
    min_size_usd: float = float(os.getenv("BITCOIN_MODEL_MIN_SIZE_USD", "20"))
    max_size_usd: float = float(os.getenv("BITCOIN_MODEL_MAX_SIZE_USD", "120"))
    max_signals_per_scan: int = int(os.getenv("BITCOIN_MODEL_MAX_SIGNALS_PER_SCAN", "6"))

    symbol: str = os.getenv("BITCOIN_MODEL_SYMBOL", "BTCUSDT")
    bucket_seconds: int = int(os.getenv("BITCOIN_MODEL_BUCKET_SECONDS", "5"))
    horizon_seconds: int = int(os.getenv("BITCOIN_MODEL_HORIZON_SECONDS", "60"))
    cost_bps: float = float(os.getenv("BITCOIN_MODEL_COST_BPS", "4.0"))
    min_signed_ratio: float = float(os.getenv("BITCOIN_MODEL_MIN_SIGNED_RATIO", "0.04"))
    min_depth_imbalance: float = float(os.getenv("BITCOIN_MODEL_MIN_DEPTH_IMBALANCE", "0.01"))
    min_trade_z: float = float(os.getenv("BITCOIN_MODEL_MIN_TRADE_Z", "0.25"))
    min_directional_efficiency: float = float(os.getenv("BITCOIN_MODEL_MIN_DIRECTIONAL_EFFICIENCY", "0.15"))
    warmup_buckets: int = int(os.getenv("BITCOIN_MODEL_WARMUP_BUCKETS", "72"))

    depth_poll_seconds: int = int(os.getenv("BITCOIN_MODEL_DEPTH_POLL_SECONDS", "5"))
    metrics_poll_seconds: int = int(os.getenv("BITCOIN_MODEL_METRICS_POLL_SECONDS", "60"))
    funding_poll_seconds: int = int(os.getenv("BITCOIN_MODEL_FUNDING_POLL_SECONDS", "300"))
    max_trade_age_buckets: int = int(os.getenv("BITCOIN_MODEL_MAX_TRADE_AGE_BUCKETS", "12"))
    max_depth_age_buckets: int = int(os.getenv("BITCOIN_MODEL_MAX_DEPTH_AGE_BUCKETS", "12"))
    max_metrics_age_buckets: int = int(os.getenv("BITCOIN_MODEL_MAX_METRICS_AGE_BUCKETS", "120"))
    max_funding_age_buckets: int = int(os.getenv("BITCOIN_MODEL_MAX_FUNDING_AGE_BUCKETS", "5760"))

    book_ticker_enabled: bool = os.getenv("BITCOIN_MODEL_BOOK_TICKER_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    max_polymarket_quote_spread: float = float(os.getenv("BITCOIN_MODEL_MAX_POLY_SPREAD", "0.18"))
    min_live_quote_edge: float = float(os.getenv("BITCOIN_MODEL_MIN_LIVE_QUOTE_EDGE", "0.08"))
    polymarket_market_ws_url: str = os.getenv(
        "BITCOIN_MODEL_POLYMARKET_MARKET_WS_URL",
        "wss://ws-subscriptions-clob.polymarket.com/ws/market",
    )
    polymarket_ping_seconds: int = int(os.getenv("BITCOIN_MODEL_POLYMARKET_PING_SECONDS", "10"))
    polymarket_quote_ttl_seconds: int = int(os.getenv("BITCOIN_MODEL_POLYMARKET_QUOTE_TTL_SECONDS", "60"))
    polymarket_recent_quote_grace_seconds: int = int(
        os.getenv("BITCOIN_MODEL_POLYMARKET_RECENT_QUOTE_GRACE_SECONDS", "180")
    )
    polymarket_max_watch_assets: int = int(os.getenv("BITCOIN_MODEL_POLYMARKET_MAX_WATCH_ASSETS", "120"))

    context_enabled: bool = os.getenv("BITCOIN_MODEL_CONTEXT_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    context_query: str = os.getenv(
        "BITCOIN_MODEL_CONTEXT_QUERY",
        "\"bitcoin\" OR BTC OR \"bitcoin etf\" OR crypto OR sec OR fed OR treasury",
    )
    context_shock_window_minutes: int = int(os.getenv("BITCOIN_MODEL_CONTEXT_SHOCK_WINDOW_MINUTES", "45"))
    context_block_intensity: float = float(os.getenv("BITCOIN_MODEL_CONTEXT_BLOCK_INTENSITY", "0.78"))
    context_aligned_size_multiplier: float = float(os.getenv("BITCOIN_MODEL_CONTEXT_ALIGNED_SIZE_MULTIPLIER", "1.20"))
    context_opposing_size_multiplier: float = float(os.getenv("BITCOIN_MODEL_CONTEXT_OPPOSING_SIZE_MULTIPLIER", "0.60"))
    context_aligned_confidence_bonus: float = float(os.getenv("BITCOIN_MODEL_CONTEXT_ALIGNED_CONFIDENCE_BONUS", "0.04"))
    context_opposing_confidence_penalty: float = float(os.getenv("BITCOIN_MODEL_CONTEXT_OPPOSING_CONFIDENCE_PENALTY", "0.06"))

    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    newsapi_poll_seconds: int = int(os.getenv("BITCOIN_MODEL_NEWSAPI_POLL_SECONDS", "120"))
    newsapi_page_size: int = int(os.getenv("BITCOIN_MODEL_NEWSAPI_PAGE_SIZE", "20"))

    gdelt_enabled: bool = os.getenv("BITCOIN_MODEL_GDELT_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    gdelt_poll_seconds: int = int(os.getenv("BITCOIN_MODEL_GDELT_POLL_SECONDS", "180"))
    gdelt_max_records: int = int(os.getenv("BITCOIN_MODEL_GDELT_MAX_RECORDS", "20"))

    x_bearer_token: str = os.getenv("X_BEARER_TOKEN", "")
    x_stream_enabled: bool = os.getenv("BITCOIN_MODEL_X_STREAM_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    x_rule_tag: str = os.getenv("BITCOIN_MODEL_X_RULE_TAG", "oracle-btc-context")
    x_rule_value: str = os.getenv(
        "BITCOIN_MODEL_X_RULE_VALUE",
        "(bitcoin OR btc OR #bitcoin OR #btc OR \"bitcoin etf\" OR crypto OR sec OR fed) lang:en -is:retweet",
    )


@dataclass
class MeanReversionConfig:
    """Mean reversion — competing signal strategy."""
    enabled: bool = True
    drop_threshold_pct: float = 0.05  # lowered: running avg smooths out moves
    lookback_hours: int = 72
    trigger_window_hours: int = 6
    exit_reversion_pct: float = 0.60  # exit at 60% reversion


@dataclass
class RiskConfig:
    """Risk management — applies across all strategies."""
    max_position_usd: float = float(os.getenv("MAX_POSITION_USD", "50"))
    max_total_exposure_usd: float = float(os.getenv("MAX_EXPOSURE_USD", "2000"))
    max_drawdown_pct: float = 0.15  # pause trading if portfolio drops 15%
    min_liquidity_usd: float = 5000.0
    max_spread_pct: float = 0.05


@dataclass
class PipelineConfig:
    api: APIConfig = field(default_factory=APIConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)
    liquidity: LiquidityProvisionConfig = field(default_factory=LiquidityProvisionConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    bundle_arb: BundleArbitrageConfig = field(default_factory=BundleArbitrageConfig)
    whale: WhaleTrackingConfig = field(default_factory=WhaleTrackingConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    crypto_arb: CryptoArbConfig = field(default_factory=CryptoArbConfig)
    weather: WeatherForecastConfig = field(default_factory=WeatherForecastConfig)
    weather_model: WeatherModelConfig = field(default_factory=WeatherModelConfig)
    bitcoin_model: BitcoinModelConfig = field(default_factory=BitcoinModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Pipeline modes
    mode: str = os.getenv("TRADING_MODE", "paper")  # paper | shadow | live
    scan_interval_secs: int = 30
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Dashboard
    dashboard_port: int = int(os.getenv("PORT", "8000"))

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"
