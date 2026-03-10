from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any

from .contracts import PortfolioSnapshot, PositionState, SignalCandidate


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "before",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "will",
    "with",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class LedgerEntry:
    key: str
    last_seen_at: datetime
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineContext:
    generated_at: datetime = field(default_factory=utc_now)
    open_market_ids: frozenset[str] = field(default_factory=frozenset)
    open_family_counts: dict[str, int] = field(default_factory=dict)
    open_theme_counts: dict[str, int] = field(default_factory=dict)
    open_theme_exposure: dict[str, float] = field(default_factory=dict)
    recent_signal_entries: dict[str, LedgerEntry] = field(default_factory=dict)
    recent_headline_entries: dict[str, LedgerEntry] = field(default_factory=dict)
    recent_family_entries: dict[str, LedgerEntry] = field(default_factory=dict)
    recent_theme_entries: dict[str, LedgerEntry] = field(default_factory=dict)

    def has_open_market(self, market_id: str) -> bool:
        return market_id in self.open_market_ids

    def family_positions(self, family_key: str | None) -> int:
        if not family_key:
            return 0
        return self.open_family_counts.get(family_key, 0)

    def theme_positions(self, theme_key: str | None) -> int:
        if not theme_key:
            return 0
        return self.open_theme_counts.get(theme_key, 0)

    def theme_exposure(self, theme_key: str | None) -> float:
        if not theme_key:
            return 0.0
        return self.open_theme_exposure.get(theme_key, 0.0)

    def seen_signal(self, key: str | None, ttl_hours: float) -> bool:
        return _seen_recent(self.recent_signal_entries, key, ttl_hours, self.generated_at)

    def seen_headline(self, key: str | None, ttl_hours: float) -> bool:
        return _seen_recent(self.recent_headline_entries, key, ttl_hours, self.generated_at)

    def seen_family(self, key: str | None, ttl_hours: float) -> bool:
        return _seen_recent(self.recent_family_entries, key, ttl_hours, self.generated_at)

    def seen_theme(self, key: str | None, ttl_hours: float) -> bool:
        return _seen_recent(self.recent_theme_entries, key, ttl_hours, self.generated_at)


def _seen_recent(
    entries: dict[str, LedgerEntry],
    key: str | None,
    ttl_hours: float,
    now: datetime,
) -> bool:
    if not key or ttl_hours <= 0:
        return False
    entry = entries.get(key)
    if entry is None:
        return False
    age_hours = (now - entry.last_seen_at).total_seconds() / 3600
    return age_hours <= ttl_hours


def compact_topic_key(text: str, prefix: str = "") -> str:
    tokens = [
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if len(token) > 2 and token not in STOP_WORDS
    ]
    topic = "-".join(tokens[:4]) if tokens else "unknown"
    return f"{prefix}{topic}" if prefix else topic


def family_key_from_signal(signal: SignalCandidate) -> str | None:
    metadata = signal.metadata or {}
    family_key = metadata.get("family_key")
    if family_key:
        return str(family_key)
    if signal.strategy_name == "news_signal":
        headline = metadata.get("headline")
        if headline:
            slug = metadata.get("market_slug") or signal.market_id
            return f"news:{compact_topic_key(str(headline))}:{slug}"
    if signal.strategy_name.startswith("weather_"):
        city = metadata.get("city")
        target_date = metadata.get("target_date")
        variant = metadata.get("variant") or signal.strategy_name
        if city and target_date:
            return f"weather:{variant}:{city}:{target_date}"
    if signal.strategy_name.startswith("crypto_"):
        symbol = metadata.get("symbol")
        kind = metadata.get("kind") or signal.strategy_name
        if symbol:
            return f"crypto:{symbol}:{kind}:{signal.market_id}"
    return f"{signal.strategy_name}:{signal.market_id}:{signal.outcome}"


def theme_key_from_signal(signal: SignalCandidate) -> str | None:
    metadata = signal.metadata or {}
    theme_key = metadata.get("theme_key")
    if theme_key:
        return str(theme_key)
    if signal.strategy_name.startswith("weather_"):
        city = metadata.get("city")
        if city:
            return f"weather:{compact_topic_key(str(city))}"
    if signal.strategy_name.startswith("crypto_") or signal.strategy_name == "relationship_arbitrage":
        symbol = metadata.get("symbol")
        if symbol:
            return f"crypto:{str(symbol).lower()}"
    if signal.strategy_name == "news_signal":
        headline = metadata.get("headline") or (
            signal.market_snapshot.question if signal.market_snapshot is not None else ""
        )
        return f"news:{compact_topic_key(str(headline))}"
    question = signal.market_snapshot.question if signal.market_snapshot is not None else signal.market_id
    return compact_topic_key(question, prefix="market:")


def signal_memory_key(signal: SignalCandidate) -> str:
    family_key = family_key_from_signal(signal) or signal.market_id
    return f"{signal.strategy_name}:{signal.market_id}:{signal.outcome}:{family_key}"


def headline_memory_key(signal: SignalCandidate) -> str | None:
    headline = (signal.metadata or {}).get("headline")
    if not headline:
        return None
    source = (signal.metadata or {}).get("source") or "unknown"
    return f"{source}:{compact_topic_key(str(headline))}"


def family_key_from_position(position: PositionState) -> str | None:
    metadata = position.metadata or {}
    family_key = metadata.get("family_key")
    if family_key:
        return str(family_key)
    if position.strategy_name.startswith("weather_"):
        city = metadata.get("city")
        target_date = metadata.get("target_date")
        variant = metadata.get("variant") or position.strategy_name
        if city and target_date:
            return f"weather:{variant}:{city}:{target_date}"
    if position.strategy_name.startswith("crypto_") or position.strategy_name == "relationship_arbitrage":
        symbol = metadata.get("symbol")
        if symbol:
            kind = metadata.get("kind") or metadata.get("opportunity_type") or position.strategy_name
            return f"crypto:{symbol}:{kind}:{position.market_id}"
    return f"{position.strategy_name}:{position.market_id}:{position.outcome}"


def theme_key_from_position(position: PositionState) -> str | None:
    metadata = position.metadata or {}
    theme_key = metadata.get("theme_key")
    if theme_key:
        return str(theme_key)
    if position.strategy_name.startswith("weather_"):
        city = metadata.get("city")
        if city:
            return f"weather:{compact_topic_key(str(city))}"
    if position.strategy_name.startswith("crypto_") or position.strategy_name == "relationship_arbitrage":
        symbol = metadata.get("symbol")
        if symbol:
            return f"crypto:{str(symbol).lower()}"
    if position.strategy_name == "news_signal":
        headline = metadata.get("headline") or position.market_question
        return f"news:{compact_topic_key(str(headline))}"
    return compact_topic_key(position.market_question, prefix="market:")


def build_pipeline_context(
    portfolio: PortfolioSnapshot,
    *,
    signal_entries: dict[str, LedgerEntry],
    headline_entries: dict[str, LedgerEntry],
    family_entries: dict[str, LedgerEntry],
    theme_entries: dict[str, LedgerEntry],
) -> PipelineContext:
    open_family_counts: dict[str, int] = {}
    open_theme_counts: dict[str, int] = {}
    open_theme_exposure: dict[str, float] = {}

    for position in portfolio.positions:
        family_key = family_key_from_position(position)
        if family_key:
            open_family_counts[family_key] = open_family_counts.get(family_key, 0) + 1

        theme_key = theme_key_from_position(position)
        if theme_key:
            open_theme_counts[theme_key] = open_theme_counts.get(theme_key, 0) + 1
            exposure = position.current_price * position.shares
            open_theme_exposure[theme_key] = open_theme_exposure.get(theme_key, 0.0) + exposure

    return PipelineContext(
        open_market_ids=portfolio.open_market_ids,
        open_family_counts=open_family_counts,
        open_theme_counts=open_theme_counts,
        open_theme_exposure=open_theme_exposure,
        recent_signal_entries=signal_entries,
        recent_headline_entries=headline_entries,
        recent_family_entries=family_entries,
        recent_theme_entries=theme_entries,
    )
