from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from runtime_paths import LOG_DIR


@dataclass(frozen=True)
class StrategyCap:
    max_capital_pct: float = 1.0
    max_positions: int = 999
    max_single_position_usd: float = 1500.0


@dataclass(frozen=True)
class ValidationConfig:
    min_volume_24h: float = 5000.0
    strategy_min_volume_24h: dict[str, float] = field(
        default_factory=lambda: {
            "relationship_arbitrage": 0.0,
            "weather_sniper": 0.0,
            "weather_latency": 0.0,
            "weather_swing": 0.0,
            "crypto_latency": 0.0,
            "news_signal": 0.0,
        }
    )
    min_liquidity: float = 2000.0
    min_hours_to_resolution: float = 4.0
    min_market_age_hours: float = 2.0
    min_edge_absolute: float = 0.03
    max_positions_per_category: int = 999
    max_enrichment_age_seconds: dict[str, float] = field(
        default_factory=lambda: {
            "weather": 7200.0,
            "crypto": 600.0,
            "news": 14400.0,
        }
    )


@dataclass(frozen=True)
class SizingConfig:
    method: str = "fixed_fractional"
    base_fraction: float = 0.02
    edge_normalization: float = 0.10
    min_position_usd: float = 10.0
    max_slippage_pct: float = 0.02
    execution_size_safety_factor: float = 0.85


@dataclass(frozen=True)
class SlippageConfig:
    impact_factor: float = 0.1
    min_slippage_pct: float = 0.001


def recommended_strategy_caps() -> dict[str, StrategyCap]:
    return {
        "relationship_arbitrage": StrategyCap(
            max_capital_pct=1.0,
            max_positions=999,
            max_single_position_usd=1500.0,
        ),
        "weather_sniper": StrategyCap(
            max_capital_pct=1.0,
            max_positions=999,
            max_single_position_usd=1500.0,
        ),
        "weather_latency": StrategyCap(
            max_capital_pct=1.0,
            max_positions=999,
            max_single_position_usd=1500.0,
        ),
        "weather_swing": StrategyCap(
            max_capital_pct=1.0,
            max_positions=999,
            max_single_position_usd=1500.0,
        ),
        "crypto_structure": StrategyCap(
            max_capital_pct=1.0,
            max_positions=999,
            max_single_position_usd=1500.0,
        ),
        "crypto_latency": StrategyCap(
            max_capital_pct=1.0,
            max_positions=999,
            max_single_position_usd=1500.0,
        ),
        "news_signal": StrategyCap(
            max_capital_pct=1.0,
            max_positions=999,
            max_single_position_usd=1500.0,
        ),
    }


@dataclass(frozen=True)
class RiskLimits:
    max_portfolio_utilization_pct: float = 0.98
    min_reserve_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    max_single_position_pct: float = 0.20
    max_single_position_usd: float = 2500.0
    strategy_caps: dict[str, StrategyCap] = field(default_factory=recommended_strategy_caps)
    category_caps: dict[str, float] = field(
        default_factory=lambda: {
            "weather": 1.0,
            "crypto": 1.0,
            "politics": 1.0,
            "sports": 1.0,
            "other": 1.0,
        }
    )
    reentry_cooldown_hours: float = 0.0
    max_correlated_positions: int = 999


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "paper"
    paper: SlippageConfig = field(default_factory=SlippageConfig)
    live_api_key_env: str = "POLYMARKET_API_KEY"
    live_api_secret_env: str = "POLYMARKET_API_SECRET"


@dataclass(frozen=True)
class AuditConfig:
    snapshot_dir: Path = field(default_factory=lambda: LOG_DIR / "multiagent_snapshots")
    max_snapshots: int = 1000
    log_level: str = "INFO"
    log_format: str = "json"


@dataclass(frozen=True)
class LLMTaskConfig:
    enabled: bool = False
    primary_provider: str = "anthropic"
    primary_model: str = "claude-sonnet-4-6"
    fallback_provider: str = "fireworks"
    fallback_model: str = "accounts/fireworks/models/glm-5"
    shadow_providers: tuple[str, ...] = ()
    max_calls_per_cycle: int = 6
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = bool(
        os.getenv("ANTHROPIC_API_KEY", "")
        or os.getenv("FIREWORKS_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    tasks: dict[str, LLMTaskConfig] = field(
        default_factory=lambda: {
            "news_relevance": LLMTaskConfig(
                enabled=bool(
                    os.getenv("ANTHROPIC_API_KEY", "")
                    or os.getenv("FIREWORKS_API_KEY", "")
                    or os.getenv("OPENAI_API_KEY", "")
                )
            ),
            "relationship_linking": LLMTaskConfig(enabled=False),
            "rule_extraction": LLMTaskConfig(enabled=False),
        }
    )


@dataclass(frozen=True)
class OrchestratorConfig:
    scanner: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, Any] = field(
        default_factory=lambda: {
            "relationship_arbitrage": {
                "duplicate_min_edge": 0.045,
                "ladder_min_edge": 0.04,
                "implication_min_edge": 0.05,
                "max_entry_price": 0.82,
                "max_candidates_per_cycle": 8,
                "min_volume_focus": 10000.0,
            },
            "weather_sniper": {
                "min_edge": 0.08,
                "max_yes_price": 0.08,
                "min_probability": 0.92,
                "min_models": 3,
            },
            "weather_latency": {
                "min_edge": 0.04,
                "min_probability_shift": 0.07,
                "max_entry_price": 0.62,
            },
            "weather_swing": {
                "min_edge": 0.04,
                "swing_min_prob": 0.58,
                "swing_max_prob": 0.42,
                "min_token_dip": 0.05,
                "min_history_points": 3,
            },
            "crypto_latency": {
                "temporal_min_move_pct": 0.003,
                "temporal_max_entry_price": 0.75,
                "barrier_min_edge": 0.04,
                "max_entry_price": 0.80,
                "max_days_to_resolution": 90.0,
                "max_candidates_per_cycle": 8,
            },
            "news_signal": {
                "min_confidence": 0.58,
                "min_edge": 0.03,
                "max_entry_price": 0.75,
                "max_candidates_per_cycle": 4,
            },
        }
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
