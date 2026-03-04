"""
Data Models
===========
Pydantic models for markets, signals, trades, and portfolio state.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalSource(str, Enum):
    LIQUIDITY = "liquidity_provision"
    ARBITRAGE = "multi_outcome_arbitrage"
    WHALE = "whale_tracking"
    NEWS = "news_latency"
    MEAN_REVERSION = "mean_reversion"


class TradeStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SignalAction(str, Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"
    HEDGE_BOTH = "hedge_both"      # liquidity provision
    ARB_ALL = "arb_all_outcomes"   # multi-outcome arb


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

class Outcome(BaseModel):
    token_id: str
    name: str  # "Yes" / "No" / candidate name etc.
    price: float
    book_bid: Optional[float] = None
    book_ask: Optional[float] = None


class Market(BaseModel):
    condition_id: str
    question: str
    slug: str
    outcomes: list[Outcome]
    volume_24h: float = 0.0
    volume_total: float = 0.0
    liquidity: float = 0.0
    spread: float = 0.0
    midpoint: float = 0.0
    end_date: Optional[str] = None
    active: bool = True
    closed: bool = False
    neg_risk: bool = False
    tags: list[str] = Field(default_factory=list)
    reward_pool: float = 0.0
    max_spread_for_rewards: float = 0.0
    min_shares_for_rewards: int = 0
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Event(BaseModel):
    event_id: str
    slug: str
    title: str
    markets: list[Market] = Field(default_factory=list)
    total_volume: float = 0.0


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: SignalSource
    action: SignalAction
    market_slug: str
    condition_id: str
    token_id: Optional[str] = None
    confidence: float = 0.0        # 0.0 - 1.0
    expected_edge: float = 0.0     # expected profit in cents
    reasoning: str = ""
    whale_confirmed: bool = False
    suggested_size_usd: float = 0.0

    # For arbitrage signals
    arb_outcomes: list[str] = Field(default_factory=list)  # token_ids
    arb_total_cost: float = 0.0
    arb_guaranteed_payout: float = 0.0


# ---------------------------------------------------------------------------
# Trades & Portfolio
# ---------------------------------------------------------------------------

class PaperTrade(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    signal_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: SignalSource
    market_slug: str
    condition_id: str
    token_id: str
    side: Side
    price: float
    size_shares: float
    size_usd: float
    status: TradeStatus = TradeStatus.FILLED
    # P&L tracking
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    realized_pnl: Optional[float] = None


class Position(BaseModel):
    token_id: str
    condition_id: str
    market_slug: str
    side: str  # "YES" or "NO" or outcome name
    shares: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    source: SignalSource


class Portfolio(BaseModel):
    starting_capital: float = 1000.0
    cash: float = 1000.0
    positions: list[Position] = Field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_realized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    peak_value: float = 1000.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0

    @property
    def positions_value(self) -> float:
        return sum(p.shares * p.current_price for p in self.positions)

    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.starting_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return (self.total_pnl / self.starting_capital) * 100

    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total


# ---------------------------------------------------------------------------
# Whale Data
# ---------------------------------------------------------------------------

class WhaleWallet(BaseModel):
    address: str
    name: Optional[str] = None
    total_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    last_active: Optional[datetime] = None
    recent_trades: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

class NewsHeadline(BaseModel):
    title: str
    source: str
    url: str
    published: Optional[datetime] = None
    classification: Optional[dict] = None  # {market_slug: "bullish"/"bearish", confidence: 0.8}


# ---------------------------------------------------------------------------
# Dashboard State
# ---------------------------------------------------------------------------

class DashboardState(BaseModel):
    mode: str = "paper"
    uptime_seconds: int = 0
    portfolio: Portfolio = Field(default_factory=Portfolio)
    active_markets: int = 0
    active_signals: list[Signal] = Field(default_factory=list)
    recent_trades: list[PaperTrade] = Field(default_factory=list)
    strategy_stats: dict = Field(default_factory=dict)
    whale_wallets: list[WhaleWallet] = Field(default_factory=list)
    recent_news: list[NewsHeadline] = Field(default_factory=list)
    arb_opportunities: list[dict] = Field(default_factory=list)
    liquidity_positions: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    last_scan: Optional[datetime] = None
