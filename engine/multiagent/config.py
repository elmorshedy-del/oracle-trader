from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from runtime_paths import LOG_DIR


@dataclass(frozen=True)
class StrategyCap:
    max_capital_pct: float = 0.25
    max_positions: int = 8
    max_single_position_usd: float = 600.0


@dataclass(frozen=True)
class ValidationConfig:
    allow_duplicate_positions: bool = False
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
    strategy_min_liquidity: dict[str, float] = field(
        default_factory=lambda: {
            "relationship_arbitrage": 2000.0,
            "weather_sniper": 500.0,
            "weather_latency": 900.0,
            "weather_swing": 900.0,
            "crypto_latency": 1500.0,
            "news_signal": 750.0,
        }
    )
    min_hours_to_resolution: float = 4.0
    min_market_age_hours: float = 2.0
    min_edge_absolute: float = 0.03
    max_positions_per_category: int = 18
    recent_signal_ttl_hours: float = 1.0
    recent_headline_ttl_hours: float = 2.0
    family_reentry_cooldown_hours: float = 1.0
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
            max_capital_pct=0.35,
            max_positions=18,
            max_single_position_usd=700.0,
        ),
        "weather_sniper": StrategyCap(
            max_capital_pct=0.25,
            max_positions=8,
            max_single_position_usd=450.0,
        ),
        "weather_latency": StrategyCap(
            max_capital_pct=0.18,
            max_positions=6,
            max_single_position_usd=350.0,
        ),
        "weather_swing": StrategyCap(
            max_capital_pct=0.18,
            max_positions=6,
            max_single_position_usd=325.0,
        ),
        "crypto_structure": StrategyCap(
            max_capital_pct=0.20,
            max_positions=8,
            max_single_position_usd=400.0,
        ),
        "crypto_latency": StrategyCap(
            max_capital_pct=0.12,
            max_positions=4,
            max_single_position_usd=250.0,
        ),
        "news_signal": StrategyCap(
            max_capital_pct=0.25,
            max_positions=10,
            max_single_position_usd=500.0,
        ),
    }


@dataclass(frozen=True)
class RiskLimits:
    max_portfolio_utilization_pct: float = 0.90
    min_reserve_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_single_position_pct: float = 0.20
    max_single_position_usd: float = 2500.0
    strategy_caps: dict[str, StrategyCap] = field(default_factory=recommended_strategy_caps)
    category_caps: dict[str, float] = field(
        default_factory=lambda: {
            "weather": 0.40,
            "crypto": 0.18,
            "politics": 0.45,
            "sports": 0.20,
            "entertainment": 0.18,
            "economics": 0.18,
            "other": 0.18,
        }
    )
    family_caps: dict[str, int] = field(
        default_factory=lambda: {
            "relationship_arbitrage": 1,
            "weather_sniper": 1,
            "weather_latency": 1,
            "weather_swing": 1,
            "crypto_structure": 1,
            "crypto_latency": 1,
            "news_signal": 1,
        }
    )
    max_positions_per_theme: int = 6
    max_theme_capital_pct: float = 0.30
    reentry_cooldown_hours: float = 0.0
    max_correlated_positions: int = 4


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
    primary_provider: str = "fireworks"
    primary_model: str = "accounts/fireworks/models/glm-5"
    fallback_provider: str = ""
    fallback_model: str = "claude-sonnet-4-6"
    shadow_providers: tuple[str, ...] = ()
    max_calls_per_cycle: int = 6
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class DecisioningConfig:
    trade_gate_batch_size: int = 8
    trade_gate_fail_open: bool = True
    trade_gate_memory_hours: float = 6.0
    exit_judge_max_positions_per_cycle: int = 8
    exit_judge_min_hold_hours: float = 1.5
    exit_judge_fail_open: bool = True
    exit_judge_require_nontrivial_context: bool = True


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = bool(os.getenv("FIREWORKS_API_KEY", ""))
    tasks: dict[str, LLMTaskConfig] = field(
        default_factory=lambda: {
            "news_relevance": LLMTaskConfig(
                enabled=bool(os.getenv("FIREWORKS_API_KEY", "")),
                fallback_provider="",
                max_calls_per_cycle=int(os.getenv("OPUS_NEWS_MAX_CALLS_PER_CYCLE", "3")),
            ),
            "relationship_linking": LLMTaskConfig(enabled=False),
            "rule_extraction": LLMTaskConfig(enabled=False),
            "trade_gate": LLMTaskConfig(
                enabled=bool(
                    os.getenv("OPUS_ENABLE_LLM_TRADE_GATE", "1").lower() in {"1", "true", "yes", "on"}
                    and os.getenv("FIREWORKS_API_KEY", "")
                ),
                primary_provider=os.getenv("OPUS_TRADE_GATE_PROVIDER", "fireworks"),
                primary_model=os.getenv("OPUS_TRADE_GATE_MODEL", "accounts/fireworks/models/glm-5"),
                fallback_provider="",
                fallback_model=os.getenv("OPUS_TRADE_GATE_FALLBACK_MODEL", "claude-sonnet-4-6"),
                max_calls_per_cycle=int(os.getenv("OPUS_TRADE_GATE_MAX_CALLS_PER_CYCLE", "1")),
                timeout_seconds=float(os.getenv("OPUS_TRADE_GATE_TIMEOUT_SECONDS", "25")),
            ),
            "exit_judge": LLMTaskConfig(
                enabled=bool(
                    os.getenv("OPUS_ENABLE_LLM_EXIT_JUDGE", "1").lower() in {"1", "true", "yes", "on"}
                    and os.getenv("FIREWORKS_API_KEY", "")
                ),
                primary_provider=os.getenv("OPUS_EXIT_JUDGE_PROVIDER", "fireworks"),
                primary_model=os.getenv("OPUS_EXIT_JUDGE_MODEL", "accounts/fireworks/models/glm-5"),
                fallback_provider="",
                fallback_model=os.getenv("OPUS_EXIT_JUDGE_FALLBACK_MODEL", "claude-sonnet-4-6"),
                max_calls_per_cycle=int(os.getenv("OPUS_EXIT_JUDGE_MAX_CALLS_PER_CYCLE", "1")),
                timeout_seconds=float(os.getenv("OPUS_EXIT_JUDGE_TIMEOUT_SECONDS", "25")),
            ),
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
                "enabled": os.getenv("OPUS_ENABLE_CRYPTO_LATENCY", "0").lower() in {"1", "true", "yes", "on"},
                "temporal_min_move_pct": 0.003,
                "temporal_max_entry_price": 0.75,
                "barrier_min_edge": 0.04,
                "max_entry_price": 0.80,
                "max_days_to_resolution": 90.0,
                "max_candidates_per_cycle": 8,
            },
            "news_signal": {
                "min_confidence": 0.35,
                "min_edge": 0.015,
                "max_entry_price": 0.85,
                "max_candidates_per_cycle": 6,
            },
        }
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    decisioning: DecisioningConfig = field(default_factory=DecisioningConfig)
