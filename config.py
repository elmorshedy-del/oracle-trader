"""
Polymarket Algo Trading Pipeline — Configuration
=================================================
All tunable parameters in one place. Override with environment variables.
"""

import os
from dataclasses import dataclass, field

from engine.crypto_pairs.config import DEFAULT_BINANCE_SPOT_WS_URLS
from engine.weather_edge_config import RULE_MIN_EDGE
from engine.weather_edge_live_support import WEATHER_EDGE_LIVE_ALLOWED_LEAD_TIMES_HOURS


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
        "models/weather_ml/external_only/frozen-legacy-weather-ml-v1",
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
class WeatherModelV2Config:
    """Stacked weather ML v2 experiment built on the frozen weather baseline."""
    enabled: bool = os.getenv("WEATHER_MODEL_V2_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    model_dir: str = os.getenv(
        "WEATHER_MODEL_V2_DIR",
        "models/weather_ml/external_only/legacy-weather-ml-v2",
    )
    fallback_model_dir: str = os.getenv(
        "WEATHER_MODEL_V2_FALLBACK_DIR",
        "models/weather_ml/external_only/frozen-legacy-weather-ml-v1",
    )
    history_manifest_path: str = os.getenv(
        "WEATHER_MODEL_V2_HISTORY_MANIFEST",
        "models/weather_ml/weather_v2_history_sources.json",
    )

    trader_budget_usd: float = float(os.getenv("WEATHER_MODEL_V2_TRADER_BUDGET_USD", "750"))
    signal_budget_usd: float = float(os.getenv("WEATHER_MODEL_V2_SIGNAL_BUDGET_USD", "600"))

    trader_min_edge: float = float(os.getenv("WEATHER_MODEL_V2_TRADER_MIN_EDGE", "0.03"))
    trader_min_prob_distance: float = float(os.getenv("WEATHER_MODEL_V2_TRADER_MIN_PROB_DISTANCE", "0.10"))
    trader_max_token_price: float = float(os.getenv("WEATHER_MODEL_V2_TRADER_MAX_TOKEN_PRICE", "0.80"))
    trader_min_size_usd: float = float(os.getenv("WEATHER_MODEL_V2_TRADER_MIN_SIZE_USD", "5"))
    trader_max_size_usd: float = float(os.getenv("WEATHER_MODEL_V2_TRADER_MAX_SIZE_USD", "45"))

    signal_min_edge: float = float(os.getenv("WEATHER_MODEL_V2_SIGNAL_MIN_EDGE", "0.05"))
    signal_min_prob_distance: float = float(os.getenv("WEATHER_MODEL_V2_SIGNAL_MIN_PROB_DISTANCE", "0.15"))
    signal_max_token_price: float = float(os.getenv("WEATHER_MODEL_V2_SIGNAL_MAX_TOKEN_PRICE", "0.65"))
    signal_min_size_usd: float = float(os.getenv("WEATHER_MODEL_V2_SIGNAL_MIN_SIZE_USD", "5"))
    signal_max_size_usd: float = float(os.getenv("WEATHER_MODEL_V2_SIGNAL_MAX_SIZE_USD", "35"))


@dataclass
class WeatherEdgeLiveConfig:
    """Standalone live weather-edge-v1 lane."""

    enabled: bool = os.getenv("WEATHER_EDGE_LIVE_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    model_dir: str = os.getenv(
        "WEATHER_EDGE_LIVE_MODEL_DIR",
        "models/weather_ml/external_only/frozen-legacy-weather-ml-v1",
    )
    starting_bankroll_usd: float = float(os.getenv("WEATHER_EDGE_LIVE_STARTING_BANKROLL_USD", "500"))
    min_edge: float = float(os.getenv("WEATHER_EDGE_LIVE_MIN_EDGE", str(RULE_MIN_EDGE)))
    max_position_fraction: float = float(os.getenv("WEATHER_EDGE_LIVE_MAX_POSITION_FRACTION", "0.02"))
    daily_summary_hour_utc: int = int(os.getenv("WEATHER_EDGE_LIVE_DAILY_SUMMARY_HOUR_UTC", "0"))
    allowed_lead_times_hours: tuple[int, ...] = field(
        default_factory=lambda: tuple(WEATHER_EDGE_LIVE_ALLOWED_LEAD_TIMES_HOURS)
    )
    label: str = os.getenv("WEATHER_EDGE_LIVE_LABEL", "Weather Edge Live")
    view_key: str = os.getenv("WEATHER_EDGE_LIVE_VIEW_KEY", "weather_edge_live")
    source: str = os.getenv("WEATHER_EDGE_LIVE_SOURCE", "weather_edge_live")
    session_label: str = os.getenv("WEATHER_EDGE_LIVE_SESSION_LABEL", "weather_edge_live")
    audit_root: str = os.getenv("WEATHER_EDGE_LIVE_AUDIT_ROOT", "")


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
class BitcoinMeanRevShadowConfig:
    """Frozen BTC multivenue mean-reversion shadow sleeve."""
    enabled: bool = os.getenv("BITCOIN_MEANREV_SHADOW_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    spec_path: str = os.getenv(
        "BITCOIN_MEANREV_SHADOW_SPEC_PATH",
        "research/btc/projects/btc-meanrev-downshock30-v1/validation_spec.json",
    )
    budget_usd: float = float(os.getenv("BITCOIN_MEANREV_SHADOW_BUDGET_USD", "600"))
    trade_notional_usd: float = float(os.getenv("BITCOIN_MEANREV_SHADOW_TRADE_NOTIONAL_USD", "600"))
    symbol: str = os.getenv("BITCOIN_MEANREV_SHADOW_SYMBOL", "BTCUSDT")
    product_id: str = os.getenv("BITCOIN_MEANREV_SHADOW_PRODUCT_ID", "BTC-USD")
    bucket_seconds: int = int(os.getenv("BITCOIN_MEANREV_SHADOW_BUCKET_SECONDS", "1"))
    levels: int = int(os.getenv("BITCOIN_MEANREV_SHADOW_LEVELS", "20"))
    warmup_buckets: int = int(os.getenv("BITCOIN_MEANREV_SHADOW_WARMUP_BUCKETS", "45"))
    evaluation_interval_seconds: float = float(os.getenv("BITCOIN_MEANREV_SHADOW_EVAL_INTERVAL_SECONDS", "0.5"))
    session_label: str = os.getenv("BITCOIN_MEANREV_SHADOW_SESSION_LABEL", "runtime_meanrev_shadow_v1")
    capture_root: str = os.getenv("BITCOIN_MEANREV_SHADOW_CAPTURE_ROOT", "")


@dataclass(frozen=True)
class CryptoPairsShadowProfileConfig:
    strategy_key: str
    view_key: str
    pair_key: str
    label: str
    source: str
    session_label: str
    margin_multiple: float = 1.0


CRYPTO_PAIRS_SHADOW_DEFAULT_PROFILE_SPECS = (
    ("crypto_pairs_shadow", "crypto_pairs_aave_doge", "AAVE/DOGE", "AAVE/DOGE Shadow", "crypto_pairs_aave_doge_shadow", "crypto_pairs_aave_doge_shadow", 3.0),
    ("crypto_pairs_shadow_comp_floki", "crypto_pairs_comp_floki", "COMP/FLOKI", "COMP/FLOKI Shadow", "crypto_pairs_comp_floki_shadow", "crypto_pairs_comp_floki_shadow", 3.0),
    ("crypto_pairs_shadow_comp_link", "crypto_pairs_comp_link", "COMP/LINK", "COMP/LINK Shadow", "crypto_pairs_comp_link_shadow", "crypto_pairs_comp_link_shadow", 3.0),
    ("crypto_pairs_shadow_bonk_grt", "crypto_pairs_bonk_grt", "BONK/GRT", "BONK/GRT Shadow", "crypto_pairs_bonk_grt_shadow", "crypto_pairs_bonk_grt_shadow", 3.0),
)


def build_crypto_pairs_shadow_profiles() -> list[CryptoPairsShadowProfileConfig]:
    raw_profiles = os.getenv("CRYPTO_PAIRS_SHADOW_PROFILES", "").strip()
    if raw_profiles:
        profiles: list[CryptoPairsShadowProfileConfig] = []
        for raw_profile in raw_profiles.split(";"):
            parts = [part.strip() for part in raw_profile.split("|")]
            if len(parts) not in {6, 7}:
                raise ValueError(
                    "CRYPTO_PAIRS_SHADOW_PROFILES entries must use "
                    "strategy_key|view_key|pair_key|label|source|session_label[|margin_multiple]"
                )
            if len(parts) == 6:
                parts.append("1.0")
            parts[-1] = float(parts[-1])
            profiles.append(CryptoPairsShadowProfileConfig(*parts))
        return profiles

    legacy_pair_keys = [
        pair.strip()
        for pair in os.getenv("CRYPTO_PAIRS_SHADOW_PAIR_KEYS", "").split(",")
        if pair.strip()
    ]
    legacy_label = os.getenv("CRYPTO_PAIRS_SHADOW_LABEL", "").strip()
    legacy_session_label = os.getenv("CRYPTO_PAIRS_SHADOW_SESSION_LABEL", "").strip()
    legacy_single_profile_requested = bool(legacy_pair_keys or legacy_label or legacy_session_label)
    if legacy_single_profile_requested:
        pair_key = legacy_pair_keys[0] if legacy_pair_keys else "AAVE/DOGE"
        return [
            CryptoPairsShadowProfileConfig(
                strategy_key="crypto_pairs_shadow",
                view_key="crypto_pairs_aave_doge",
                pair_key=pair_key,
                label=legacy_label or f"{pair_key} Shadow",
                source="crypto_pairs_aave_doge_shadow",
                session_label=legacy_session_label or "crypto_pairs_aave_doge_shadow",
                margin_multiple=3.0 if pair_key == "AAVE/DOGE" else 1.0,
            )
        ]

    return [CryptoPairsShadowProfileConfig(*spec) for spec in CRYPTO_PAIRS_SHADOW_DEFAULT_PROFILE_SPECS]


@dataclass
class CryptoPairsShadowConfig:
    """Focused crypto-pairs shadow sleeves for live Oracle paper validation."""
    enabled: bool = os.getenv("CRYPTO_PAIRS_SHADOW_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    discovery_report: str = os.getenv("CRYPTO_PAIRS_SHADOW_DISCOVERY_REPORT", "")
    profiles: list[CryptoPairsShadowProfileConfig] = field(default_factory=build_crypto_pairs_shadow_profiles)
    top_pairs: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_TOP_PAIRS", "3"))
    budget_usd: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_BUDGET_USD", "10000"))
    capital_per_pair_pct: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_CAPITAL_PER_PAIR_PCT", "0.20"))
    max_total_exposure_pct: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_MAX_TOTAL_EXPOSURE_PCT", "0.20"))
    max_daily_loss_pct: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_MAX_DAILY_LOSS_PCT", "0.03"))
    entry_z: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_ENTRY_Z", "2.0"))
    exit_z: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_EXIT_Z", "0.0"))
    stop_z: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_STOP_Z", "4.0"))
    max_hold_seconds: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_MAX_HOLD_SECONDS", "21600"))
    cooldown_seconds: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_COOLDOWN_SECONDS", "60"))
    fee_bps: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_FEE_BPS", "1.0"))
    slippage_bps: float = float(os.getenv("CRYPTO_PAIRS_SHADOW_SLIPPAGE_BPS", "0.5"))
    quantity_precision: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_QUANTITY_PRECISION", "8"))
    bar_interval_seconds: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_BAR_INTERVAL_SECONDS", "1"))
    max_leg_lag_ms: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_MAX_LEG_LAG_MS", "10000"))
    reconnect_delay_seconds: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_RECONNECT_DELAY_SECONDS", "5"))
    ws_urls: list[str] = field(
        default_factory=lambda: [
            url.strip()
            for url in os.getenv(
                "CRYPTO_PAIRS_SHADOW_WS_URLS",
                ",".join((DEFAULT_BINANCE_SPOT_WS_URLS[1], DEFAULT_BINANCE_SPOT_WS_URLS[0])),
            ).split(",")
            if url.strip()
        ]
    )
    hourly_check_seconds: int = int(os.getenv("CRYPTO_PAIRS_SHADOW_HOURLY_CHECK_SECONDS", "3600"))
    audit_root: str = os.getenv("CRYPTO_PAIRS_SHADOW_AUDIT_ROOT", "")


@dataclass
class CopyTraderShadowConfig:
    """Live paper copy-trader sleeve driven by top-wallet activity."""

    enabled: bool = os.getenv("COPY_TRADER_SHADOW_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    budget_usd: float = float(os.getenv("COPY_TRADER_SHADOW_BUDGET_USD", "1000"))
    max_open_positions: int = int(os.getenv("COPY_TRADER_SHADOW_MAX_OPEN_POSITIONS", "8"))
    top_wallets: int = int(os.getenv("COPY_TRADER_SHADOW_TOP_WALLETS", "3"))
    leaderboard_limit: int = int(os.getenv("COPY_TRADER_SHADOW_LEADERBOARD_LIMIT", "20"))
    min_wallet_pnl_usd: float = float(os.getenv("COPY_TRADER_SHADOW_MIN_WALLET_PNL_USD", "100000"))
    leaderboard_refresh_minutes: int = int(os.getenv("COPY_TRADER_SHADOW_LEADERBOARD_REFRESH_MINUTES", "60"))
    activity_refresh_seconds: int = int(os.getenv("COPY_TRADER_SHADOW_ACTIVITY_REFRESH_SECONDS", "45"))
    activity_trades_per_wallet: int = int(os.getenv("COPY_TRADER_SHADOW_ACTIVITY_TRADES_PER_WALLET", "12"))
    copy_size_multiplier: float = float(os.getenv("COPY_TRADER_SHADOW_COPY_SIZE_MULTIPLIER", "0.20"))
    min_trade_usd: float = float(os.getenv("COPY_TRADER_SHADOW_MIN_TRADE_USD", "10"))
    max_trade_usd: float = float(os.getenv("COPY_TRADER_SHADOW_MAX_TRADE_USD", "80"))
    max_entry_price: float = float(os.getenv("COPY_TRADER_SHADOW_MAX_ENTRY_PRICE", "0.90"))
    min_wallet_sell_usd: float = float(os.getenv("COPY_TRADER_SHADOW_MIN_WALLET_SELL_USD", "5"))
    tracked_wallets: list[str] = field(
        default_factory=lambda: [
            wallet.strip()
            for wallet in os.getenv("COPY_TRADER_SHADOW_WALLETS", "").split(",")
            if wallet.strip()
        ]
    )
    label: str = os.getenv("COPY_TRADER_SHADOW_LABEL", "Copy Trader Shadow")
    view_key: str = os.getenv("COPY_TRADER_SHADOW_VIEW_KEY", "copy_trader_shadow")
    source: str = os.getenv("COPY_TRADER_SHADOW_SOURCE", "copy_trader_shadow")
    session_label: str = os.getenv("COPY_TRADER_SHADOW_SESSION_LABEL", "copy_trader_shadow")
    audit_root: str = os.getenv("COPY_TRADER_SHADOW_AUDIT_ROOT", "")


@dataclass
class WalletCopyResearchConfig:
    """Collection-only wallet-copy research subsystem."""

    enabled: bool = os.getenv("WALLET_COPY_RESEARCH_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    db_path: str = os.getenv("WALLET_COPY_RESEARCH_DB_PATH", "")
    db_filename: str = os.getenv("WALLET_COPY_RESEARCH_DB_FILENAME", "wallet_copy_research.sqlite")
    schema_version: str = os.getenv("WALLET_COPY_RESEARCH_SCHEMA_VERSION", "v1")
    collector_version: str = os.getenv("WALLET_COPY_RESEARCH_COLLECTOR_VERSION", "wallet-copy-v1")
    target_labeled_buys: int = int(os.getenv("WALLET_COPY_RESEARCH_TARGET_LABELED_BUYS", "2000"))

    leaderboard_refresh_seconds: int = int(os.getenv("WALLET_COPY_RESEARCH_LEADERBOARD_REFRESH_SECONDS", "86400"))
    leaderboard_all_limit: int = int(os.getenv("WALLET_COPY_RESEARCH_LEADERBOARD_ALL_LIMIT", "50"))
    leaderboard_30d_profit_limit: int = int(os.getenv("WALLET_COPY_RESEARCH_LEADERBOARD_30D_PROFIT_LIMIT", "30"))
    leaderboard_30d_volume_limit: int = int(os.getenv("WALLET_COPY_RESEARCH_LEADERBOARD_30D_VOLUME_LIMIT", "20"))
    tracked_wallet_limit: int = int(os.getenv("WALLET_COPY_RESEARCH_TRACKED_WALLET_LIMIT", "50"))

    wallet_activity_poll_seconds: int = int(os.getenv("WALLET_COPY_RESEARCH_WALLET_ACTIVITY_POLL_SECONDS", "10"))
    wallet_activity_limit: int = int(os.getenv("WALLET_COPY_RESEARCH_WALLET_ACTIVITY_LIMIT", "20"))
    positions_refresh_seconds: int = int(os.getenv("WALLET_COPY_RESEARCH_POSITIONS_REFRESH_SECONDS", "300"))
    labeler_poll_seconds: int = int(os.getenv("WALLET_COPY_RESEARCH_LABELER_POLL_SECONDS", "60"))

    market_cache_ttl_seconds: int = int(os.getenv("WALLET_COPY_RESEARCH_MARKET_CACHE_TTL_SECONDS", "30"))
    orderbook_cache_ttl_seconds: int = int(os.getenv("WALLET_COPY_RESEARCH_ORDERBOOK_CACHE_TTL_SECONDS", "10"))
    activity_concurrency: int = int(os.getenv("WALLET_COPY_RESEARCH_ACTIVITY_CONCURRENCY", "6"))
    positions_concurrency: int = int(os.getenv("WALLET_COPY_RESEARCH_POSITIONS_CONCURRENCY", "4"))

    btc_context_enabled: bool = os.getenv("WALLET_COPY_RESEARCH_BTC_CONTEXT_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    btc_price_poll_seconds: int = int(os.getenv("WALLET_COPY_RESEARCH_BTC_PRICE_POLL_SECONDS", "10"))
    btc_price_source: str = os.getenv("WALLET_COPY_RESEARCH_BTC_PRICE_SOURCE", "coinbase")

    price_history_interval: str = os.getenv("WALLET_COPY_RESEARCH_PRICE_HISTORY_INTERVAL", "1m")
    price_history_fidelity: int = int(os.getenv("WALLET_COPY_RESEARCH_PRICE_HISTORY_FIDELITY", "10"))


@dataclass
class CopyHeuristicShadowProfileConfig:
    """Config for one heuristic wallet-copy paper sleeve."""

    strategy_key: str
    view_key: str
    label: str
    source: str
    session_label: str
    kind: str
    budget_usd: float
    min_trade_usd: float
    max_trade_usd: float
    score_threshold: float = 0.30
    max_open_positions: int = 6
    scan_trade_limit: int = 300
    min_wallet_labeled_trades: int = 3
    min_wallet_win_rate: float = 0.60
    max_wallet_trade_count_24h: int = 15
    min_size_vs_avg: float = 1.0
    max_market_spread: float = 0.10
    max_detection_delay_seconds: float = 30.0
    require_first_entry: bool = False
    min_consensus_wallets: int = 3
    min_consensus_avg_win_rate: float = 0.58
    consensus_window_seconds: int = 3600
    min_hold_hours: float = 24.0
    min_market_hours_remaining: float = 12.0
    max_market_age_minutes: float = 30.0
    min_trade_size_usd: float = 500.0
    audit_root: str = ""
    decisions_limit: int = 30


COPY_HEURISTIC_SHADOW_DEFAULTS = (
    {
        "strategy_key": "wallet_selective_copy_shadow",
        "view_key": "wallet_selective_copy_shadow",
        "label": "Selective Copy",
        "source": "wallet_selective_copy_shadow",
        "session_label": "wallet_selective_copy_shadow",
        "kind": "selective_copy",
        "budget_usd": 700.0,
        "min_trade_usd": 50.0,
        "max_trade_usd": 300.0,
        "score_threshold": 0.30,
        "max_open_positions": 6,
        "scan_trade_limit": 300,
        "min_wallet_labeled_trades": 3,
        "min_wallet_win_rate": 0.62,
        "max_wallet_trade_count_24h": 8,
        "min_size_vs_avg": 1.5,
        "max_market_spread": 0.05,
        "max_detection_delay_seconds": 15.0,
        "require_first_entry": True,
    },
    {
        "strategy_key": "wallet_whale_consensus_shadow",
        "view_key": "wallet_whale_consensus_shadow",
        "label": "Whale Consensus",
        "source": "wallet_whale_consensus_shadow",
        "session_label": "wallet_whale_consensus_shadow",
        "kind": "whale_consensus",
        "budget_usd": 900.0,
        "min_trade_usd": 50.0,
        "max_trade_usd": 300.0,
        "score_threshold": 0.35,
        "max_open_positions": 4,
        "scan_trade_limit": 300,
        "min_wallet_labeled_trades": 3,
        "min_wallet_win_rate": 0.58,
        "max_market_spread": 0.08,
        "min_consensus_wallets": 3,
        "min_consensus_avg_win_rate": 0.58,
        "consensus_window_seconds": 3600,
    },
    {
        "strategy_key": "wallet_contrarian_exit_shadow",
        "view_key": "wallet_contrarian_exit_shadow",
        "label": "Contrarian Exit",
        "source": "wallet_contrarian_exit_shadow",
        "session_label": "wallet_contrarian_exit_shadow",
        "kind": "contrarian_exit",
        "budget_usd": 600.0,
        "min_trade_usd": 50.0,
        "max_trade_usd": 300.0,
        "score_threshold": 0.30,
        "max_open_positions": 5,
        "scan_trade_limit": 300,
        "min_wallet_labeled_trades": 3,
        "min_wallet_win_rate": 0.60,
        "min_hold_hours": 24.0,
        "min_market_hours_remaining": 12.0,
    },
    {
        "strategy_key": "wallet_fresh_market_shadow",
        "view_key": "wallet_fresh_market_shadow",
        "label": "Fresh Market Sniper",
        "source": "wallet_fresh_market_shadow",
        "session_label": "wallet_fresh_market_shadow",
        "kind": "fresh_market",
        "budget_usd": 700.0,
        "min_trade_usd": 50.0,
        "max_trade_usd": 300.0,
        "score_threshold": 0.30,
        "max_open_positions": 5,
        "scan_trade_limit": 300,
        "min_wallet_labeled_trades": 3,
        "min_wallet_win_rate": 0.60,
        "max_market_spread": 0.10,
        "max_market_age_minutes": 30.0,
        "min_trade_size_usd": 500.0,
    },
)


def build_copy_heuristic_shadow_profiles() -> list[CopyHeuristicShadowProfileConfig]:
    profiles: list[CopyHeuristicShadowProfileConfig] = []
    for defaults in COPY_HEURISTIC_SHADOW_DEFAULTS:
        strategy_key = defaults["strategy_key"]
        env_prefix = strategy_key.upper()
        profiles.append(
            CopyHeuristicShadowProfileConfig(
                strategy_key=strategy_key,
                view_key=os.getenv(f"{env_prefix}_VIEW_KEY", defaults["view_key"]),
                label=os.getenv(f"{env_prefix}_LABEL", defaults["label"]),
                source=os.getenv(f"{env_prefix}_SOURCE", defaults["source"]),
                session_label=os.getenv(f"{env_prefix}_SESSION_LABEL", defaults["session_label"]),
                kind=defaults["kind"],
                budget_usd=float(os.getenv(f"{env_prefix}_BUDGET_USD", str(defaults["budget_usd"]))),
                min_trade_usd=float(os.getenv(f"{env_prefix}_MIN_TRADE_USD", str(defaults["min_trade_usd"]))),
                max_trade_usd=float(os.getenv(f"{env_prefix}_MAX_TRADE_USD", str(defaults["max_trade_usd"]))),
                score_threshold=float(os.getenv(f"{env_prefix}_SCORE_THRESHOLD", str(defaults["score_threshold"]))),
                max_open_positions=int(os.getenv(f"{env_prefix}_MAX_OPEN_POSITIONS", str(defaults["max_open_positions"]))),
                scan_trade_limit=int(os.getenv(f"{env_prefix}_SCAN_TRADE_LIMIT", str(defaults["scan_trade_limit"]))),
                min_wallet_labeled_trades=int(os.getenv(f"{env_prefix}_MIN_WALLET_LABELED_TRADES", str(defaults.get("min_wallet_labeled_trades", 3)))),
                min_wallet_win_rate=float(os.getenv(f"{env_prefix}_MIN_WALLET_WIN_RATE", str(defaults.get("min_wallet_win_rate", 0.60)))),
                max_wallet_trade_count_24h=int(os.getenv(f"{env_prefix}_MAX_WALLET_TRADE_COUNT_24H", str(defaults.get("max_wallet_trade_count_24h", 15)))),
                min_size_vs_avg=float(os.getenv(f"{env_prefix}_MIN_SIZE_VS_AVG", str(defaults.get("min_size_vs_avg", 1.0)))),
                max_market_spread=float(os.getenv(f"{env_prefix}_MAX_MARKET_SPREAD", str(defaults.get("max_market_spread", 0.10)))),
                max_detection_delay_seconds=float(os.getenv(f"{env_prefix}_MAX_DETECTION_DELAY_SECONDS", str(defaults.get("max_detection_delay_seconds", 30.0)))),
                require_first_entry=os.getenv(f"{env_prefix}_REQUIRE_FIRST_ENTRY", str(defaults.get("require_first_entry", False))).lower() in {"1", "true", "yes", "on"},
                min_consensus_wallets=int(os.getenv(f"{env_prefix}_MIN_CONSENSUS_WALLETS", str(defaults.get("min_consensus_wallets", 3)))),
                min_consensus_avg_win_rate=float(os.getenv(f"{env_prefix}_MIN_CONSENSUS_AVG_WIN_RATE", str(defaults.get("min_consensus_avg_win_rate", 0.58)))),
                consensus_window_seconds=int(os.getenv(f"{env_prefix}_CONSENSUS_WINDOW_SECONDS", str(defaults.get("consensus_window_seconds", 3600)))),
                min_hold_hours=float(os.getenv(f"{env_prefix}_MIN_HOLD_HOURS", str(defaults.get("min_hold_hours", 24.0)))),
                min_market_hours_remaining=float(os.getenv(f"{env_prefix}_MIN_MARKET_HOURS_REMAINING", str(defaults.get("min_market_hours_remaining", 12.0)))),
                max_market_age_minutes=float(os.getenv(f"{env_prefix}_MAX_MARKET_AGE_MINUTES", str(defaults.get("max_market_age_minutes", 30.0)))),
                min_trade_size_usd=float(os.getenv(f"{env_prefix}_MIN_TRADE_SIZE_USD", str(defaults.get("min_trade_size_usd", 500.0)))),
                audit_root=os.getenv(f"{env_prefix}_AUDIT_ROOT", ""),
                decisions_limit=int(os.getenv(f"{env_prefix}_DECISIONS_LIMIT", str(defaults.get("decisions_limit", 30)))),
            )
        )
    return profiles


@dataclass
class CopyHeuristicShadowConfig:
    """Heuristic wallet-copy paper sleeves driven by the real wallet-copy store."""

    profiles: list[CopyHeuristicShadowProfileConfig] = field(default_factory=build_copy_heuristic_shadow_profiles)


@dataclass
class KalshiBtcArbShadowConfig:
    """Cross-venue BTC hourly overlap arbitrage paper sleeve."""

    enabled: bool = os.getenv("KALSHI_BTC_ARB_SHADOW_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    budget_usd: float = float(os.getenv("KALSHI_BTC_ARB_SHADOW_BUDGET_USD", "1200"))
    max_trade_usd: float = float(os.getenv("KALSHI_BTC_ARB_SHADOW_MAX_TRADE_USD", "180"))
    min_trade_usd: float = float(os.getenv("KALSHI_BTC_ARB_SHADOW_MIN_TRADE_USD", "20"))
    min_net_margin_dollars: float = float(os.getenv("KALSHI_BTC_ARB_SHADOW_MIN_NET_MARGIN_DOLLARS", "0.02"))
    trade_fee_buffer_dollars: float = float(os.getenv("KALSHI_BTC_ARB_SHADOW_FEE_BUFFER_DOLLARS", "0.02"))
    kalshi_neighbor_count: int = int(os.getenv("KALSHI_BTC_ARB_SHADOW_NEIGHBOR_COUNT", "4"))
    max_open_positions: int = int(os.getenv("KALSHI_BTC_ARB_SHADOW_MAX_OPEN_POSITIONS", "3"))
    resolution_grace_minutes: int = int(os.getenv("KALSHI_BTC_ARB_SHADOW_RESOLUTION_GRACE_MINUTES", "15"))
    label: str = os.getenv("KALSHI_BTC_ARB_SHADOW_LABEL", "Poly/Kalshi BTC Arb")
    view_key: str = os.getenv("KALSHI_BTC_ARB_SHADOW_VIEW_KEY", "kalshi_btc_arb_shadow")
    source: str = os.getenv("KALSHI_BTC_ARB_SHADOW_SOURCE", "kalshi_btc_arb_shadow")
    session_label: str = os.getenv("KALSHI_BTC_ARB_SHADOW_SESSION_LABEL", "kalshi_btc_arb_shadow")
    audit_root: str = os.getenv("KALSHI_BTC_ARB_SHADOW_AUDIT_ROOT", "")


@dataclass
class BitcoinLatencyShadowConfig:
    """Dedicated BTC-only latency/dislocation shadow sleeve."""

    enabled: bool = os.getenv("BITCOIN_LATENCY_SHADOW_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    budget_usd: float = float(os.getenv("BITCOIN_LATENCY_SHADOW_BUDGET_USD", "600"))
    min_move_pct: float = float(os.getenv("BITCOIN_LATENCY_SHADOW_MIN_MOVE_PCT", "0.003"))
    lookback_seconds: int = int(os.getenv("BITCOIN_LATENCY_SHADOW_LOOKBACK_SECONDS", "120"))
    max_entry_price: float = float(os.getenv("BITCOIN_LATENCY_SHADOW_MAX_ENTRY_PRICE", "0.75"))
    min_trade_usd: float = float(os.getenv("BITCOIN_LATENCY_SHADOW_MIN_TRADE_USD", "12"))
    max_trade_usd: float = float(os.getenv("BITCOIN_LATENCY_SHADOW_MAX_TRADE_USD", "90"))
    label: str = os.getenv("BITCOIN_LATENCY_SHADOW_LABEL", "BTC Latency Shadow")
    view_key: str = os.getenv("BITCOIN_LATENCY_SHADOW_VIEW_KEY", "bitcoin_latency_shadow")
    source: str = os.getenv("BITCOIN_LATENCY_SHADOW_SOURCE", "bitcoin_latency_shadow")


@dataclass
class SportsModelConfig:
    """Standalone NBA sportsbook-anchor sleeve (comparison-book only)."""
    enabled: bool = os.getenv("SPORTS_MODEL_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
    budget_usd: float = float(os.getenv("SPORTS_MODEL_BUDGET_USD", "800"))
    league: str = os.getenv("SPORTS_MODEL_LEAGUE", "nba")
    scoreboard_url: str = os.getenv(
        "SPORTS_MODEL_SCOREBOARD_URL",
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    )
    refresh_seconds: int = int(os.getenv("SPORTS_MODEL_REFRESH_SECONDS", "90"))
    market_horizon_hours: float = float(os.getenv("SPORTS_MODEL_MARKET_HORIZON_HOURS", "30"))
    min_hours_to_tip: float = float(os.getenv("SPORTS_MODEL_MIN_HOURS_TO_TIP", "0.35"))
    max_hours_to_tip: float = float(os.getenv("SPORTS_MODEL_MAX_HOURS_TO_TIP", "30"))
    min_edge: float = float(os.getenv("SPORTS_MODEL_MIN_EDGE", "0.08"))
    max_entry_price: float = float(os.getenv("SPORTS_MODEL_MAX_ENTRY_PRICE", "0.84"))
    min_size_usd: float = float(os.getenv("SPORTS_MODEL_MIN_SIZE_USD", "25"))
    max_size_usd: float = float(os.getenv("SPORTS_MODEL_MAX_SIZE_USD", "140"))
    max_signals_per_scan: int = int(os.getenv("SPORTS_MODEL_MAX_SIGNALS_PER_SCAN", "8"))
    min_market_liquidity_usd: float = float(os.getenv("SPORTS_MODEL_MIN_MARKET_LIQUIDITY_USD", "2000"))
    min_market_volume_usd: float = float(os.getenv("SPORTS_MODEL_MIN_MARKET_VOLUME_USD", "1000"))
    line_move_points_threshold: float = float(os.getenv("SPORTS_MODEL_LINE_MOVE_POINTS_THRESHOLD", "1.0"))
    spread_points_scale: float = float(os.getenv("SPORTS_MODEL_SPREAD_POINTS_SCALE", "5.5"))
    total_points_scale: float = float(os.getenv("SPORTS_MODEL_TOTAL_POINTS_SCALE", "9.0"))
    max_spread_line_gap: float = float(os.getenv("SPORTS_MODEL_MAX_SPREAD_LINE_GAP", "4.0"))
    max_total_line_gap: float = float(os.getenv("SPORTS_MODEL_MAX_TOTAL_LINE_GAP", "12.0"))
    win_prob_edge_floor: float = float(os.getenv("SPORTS_MODEL_WIN_PROB_EDGE_FLOOR", "0.06"))
    total_prob_edge_floor: float = float(os.getenv("SPORTS_MODEL_TOTAL_PROB_EDGE_FLOOR", "0.08"))
    spread_prob_edge_floor: float = float(os.getenv("SPORTS_MODEL_SPREAD_PROB_EDGE_FLOOR", "0.08"))


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
    weather_model_v2: WeatherModelV2Config = field(default_factory=WeatherModelV2Config)
    weather_edge_live: WeatherEdgeLiveConfig = field(default_factory=WeatherEdgeLiveConfig)
    bitcoin_model: BitcoinModelConfig = field(default_factory=BitcoinModelConfig)
    bitcoin_meanrev_shadow: BitcoinMeanRevShadowConfig = field(default_factory=BitcoinMeanRevShadowConfig)
    crypto_pairs_shadow: CryptoPairsShadowConfig = field(default_factory=CryptoPairsShadowConfig)
    copy_trader_shadow: CopyTraderShadowConfig = field(default_factory=CopyTraderShadowConfig)
    copy_heuristic_shadow: CopyHeuristicShadowConfig = field(default_factory=CopyHeuristicShadowConfig)
    wallet_copy_research: WalletCopyResearchConfig = field(default_factory=WalletCopyResearchConfig)
    kalshi_btc_arb_shadow: KalshiBtcArbShadowConfig = field(default_factory=KalshiBtcArbShadowConfig)
    bitcoin_latency_shadow: BitcoinLatencyShadowConfig = field(default_factory=BitcoinLatencyShadowConfig)
    sports_model: SportsModelConfig = field(default_factory=SportsModelConfig)
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
