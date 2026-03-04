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
    # Max number of outcomes in a multi-outcome market to consider
    max_outcomes: int = 20
    # Min liquidity per outcome
    min_liquidity_usd: float = 1000.0
    # Execution timeout — abort if can't fill all legs in N seconds
    execution_timeout_secs: int = 10


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
    # Confidence boost multiplier when whale confirms a signal
    confirmation_boost: float = 1.5


@dataclass
class NewsConfig:
    """News-to-price latency engine (optional — requires LLM API key)."""
    enabled: bool = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = "claude-sonnet-4-5-20250929"  # cheap first pass
    escalation_model: str = "claude-sonnet-4-5-20250929"  # for high-confidence
    # RSS / news sources
    rss_feeds: list = field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    ])
    # Min confidence to generate a signal (0.0 - 1.0)
    min_confidence: float = 0.7
    # Scan interval in seconds
    scan_interval_secs: int = 30
    # Max API calls per hour (cost control)
    max_calls_per_hour: int = 100


@dataclass
class MeanReversionConfig:
    """Mean reversion — competing signal strategy."""
    enabled: bool = True
    drop_threshold_pct: float = 0.10
    lookback_hours: int = 72
    trigger_window_hours: int = 6
    exit_reversion_pct: float = 0.60  # exit at 60% reversion


@dataclass
class RiskConfig:
    """Risk management — applies across all strategies."""
    max_position_usd: float = float(os.getenv("MAX_POSITION_USD", "50"))
    max_total_exposure_usd: float = float(os.getenv("MAX_EXPOSURE_USD", "500"))
    max_drawdown_pct: float = 0.15  # pause trading if portfolio drops 15%
    min_liquidity_usd: float = 5000.0
    max_spread_pct: float = 0.05


@dataclass
class PipelineConfig:
    api: APIConfig = field(default_factory=APIConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)
    liquidity: LiquidityProvisionConfig = field(default_factory=LiquidityProvisionConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    whale: WhaleTrackingConfig = field(default_factory=WhaleTrackingConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
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
