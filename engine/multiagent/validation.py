from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from .config import ValidationConfig
from .contracts import (
    MarketContext,
    PortfolioSnapshot,
    RejectedSignal,
    SignalCandidate,
    ValidatedSignal,
    ValidationCheck,
)
from .enums import PositionStatus, RejectionReason


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ValidationRule(Protocol):
    name: str
    blocking: bool

    def check(
        self,
        candidate: SignalCandidate,
        portfolio: PortfolioSnapshot,
    ) -> ValidationCheck:
        ...


def _require_snapshot(candidate: SignalCandidate) -> MarketContext:
    if candidate.market_snapshot is None:
        raise ValueError("candidate missing market_snapshot")
    return candidate.market_snapshot


@dataclass
class MinVolumeRule:
    config: ValidationConfig
    name: str = "min_volume"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        snapshot = _require_snapshot(candidate)
        actual = snapshot.volume_24h
        threshold = self.config.strategy_min_volume_24h.get(
            candidate.strategy_name,
            self.config.min_volume_24h,
        )
        passed = actual >= threshold
        return ValidationCheck(
            rule_name=self.name,
            passed=passed,
            blocking=self.blocking,
            threshold=threshold,
            actual_value=actual,
            reason=f"24h volume ${actual:.0f} {'>=' if passed else '<'} ${threshold:.0f}",
            rejection_code=RejectionReason.VOLUME_BELOW_MINIMUM if not passed else None,
        )


@dataclass
class MinLiquidityRule:
    config: ValidationConfig
    name: str = "min_liquidity"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        snapshot = _require_snapshot(candidate)
        actual = snapshot.liquidity
        threshold = self.config.min_liquidity
        passed = actual >= threshold
        return ValidationCheck(
            rule_name=self.name,
            passed=passed,
            blocking=self.blocking,
            threshold=threshold,
            actual_value=actual,
            reason=f"liquidity ${actual:.0f} {'>=' if passed else '<'} ${threshold:.0f}",
            rejection_code=RejectionReason.LIQUIDITY_BELOW_MINIMUM if not passed else None,
        )


@dataclass
class ResolutionProximityRule:
    config: ValidationConfig
    name: str = "resolution_proximity"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        snapshot = _require_snapshot(candidate)
        hours = snapshot.hours_to_resolution
        if hours is None:
            return ValidationCheck(
                rule_name=self.name,
                passed=True,
                blocking=self.blocking,
                reason="no resolution date available",
            )
        threshold = self.config.min_hours_to_resolution
        passed = hours >= threshold
        return ValidationCheck(
            rule_name=self.name,
            passed=passed,
            blocking=self.blocking,
            threshold=threshold,
            actual_value=hours,
            reason=f"{hours:.1f}h to resolution {'>=' if passed else '<'} {threshold:.1f}h",
            rejection_code=RejectionReason.RESOLUTION_TOO_SOON if not passed else None,
        )


@dataclass
class MarketAgeRule:
    config: ValidationConfig
    name: str = "market_age"
    blocking: bool = False

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        snapshot = _require_snapshot(candidate)
        age_hours = (utc_now() - snapshot.created_date).total_seconds() / 3600
        threshold = self.config.min_market_age_hours
        passed = age_hours >= threshold
        return ValidationCheck(
            rule_name=self.name,
            passed=passed,
            blocking=self.blocking,
            threshold=threshold,
            actual_value=age_hours,
            reason=f"market age {age_hours:.1f}h {'>=' if passed else '<'} {threshold:.1f}h",
            rejection_code=RejectionReason.MARKET_TOO_NEW if not passed else None,
        )


@dataclass
class MinEdgeRule:
    config: ValidationConfig
    name: str = "min_edge"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        actual = abs(candidate.edge_estimate)
        threshold = self.config.min_edge_absolute
        passed = actual >= threshold
        return ValidationCheck(
            rule_name=self.name,
            passed=passed,
            blocking=self.blocking,
            threshold=threshold,
            actual_value=actual,
            reason=f"|edge| {actual:.4f} {'>=' if passed else '<'} {threshold:.4f}",
            rejection_code=RejectionReason.EDGE_BELOW_THRESHOLD if not passed else None,
        )


@dataclass
class ReasoningPresentRule:
    config: ValidationConfig
    name: str = "reasoning_present"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        has_reasoning = bool(candidate.reasoning and len(candidate.reasoning.strip()) > 20)
        return ValidationCheck(
            rule_name=self.name,
            passed=has_reasoning,
            blocking=self.blocking,
            reason="reasoning present" if has_reasoning else "reasoning missing or too short",
            rejection_code=RejectionReason.MISSING_REASONING if not has_reasoning else None,
        )


@dataclass
class EdgeBasisPresentRule:
    config: ValidationConfig
    name: str = "edge_basis_present"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        has_basis = bool(candidate.edge_basis and candidate.edge_basis.strip())
        return ValidationCheck(
            rule_name=self.name,
            passed=has_basis,
            blocking=self.blocking,
            reason="edge basis present" if has_basis else "no edge basis specified",
            rejection_code=RejectionReason.MISSING_EDGE_BASIS if not has_basis else None,
        )


@dataclass
class EvidencePresentRule:
    config: ValidationConfig
    name: str = "evidence_present"
    blocking: bool = False

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        has_evidence = len(candidate.evidence) > 0
        return ValidationCheck(
            rule_name=self.name,
            passed=has_evidence,
            blocking=self.blocking,
            reason=f"{len(candidate.evidence)} evidence items" if has_evidence else "no evidence listed",
            rejection_code=RejectionReason.MISSING_EVIDENCE if not has_evidence else None,
        )


@dataclass
class DuplicatePositionRule:
    config: ValidationConfig
    name: str = "duplicate_position"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        duplicate = portfolio.has_position_in(candidate.market_id)
        return ValidationCheck(
            rule_name=self.name,
            passed=not duplicate,
            blocking=self.blocking,
            reason="no existing position" if not duplicate else f"already have position in {candidate.market_id}",
            rejection_code=RejectionReason.DUPLICATE_POSITION if duplicate else None,
        )


@dataclass
class MaxMarketsPerCategoryRule:
    config: ValidationConfig
    name: str = "max_markets_per_category"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        snapshot = _require_snapshot(candidate)
        category = snapshot.category.value
        current_count = sum(
            1
            for position in portfolio.positions
            if position.status == PositionStatus.OPEN and position.category.value == category
        )
        threshold = self.config.max_positions_per_category
        passed = current_count < threshold
        return ValidationCheck(
            rule_name=self.name,
            passed=passed,
            blocking=self.blocking,
            threshold=threshold,
            actual_value=current_count,
            reason=f"{current_count} open {category} positions {'<' if passed else '>='} {threshold}",
            rejection_code=RejectionReason.MAX_MARKETS_PER_CATEGORY if not passed else None,
        )


@dataclass
class EnrichmentStalenessRule:
    config: ValidationConfig
    name: str = "enrichment_staleness"
    blocking: bool = True

    def check(self, candidate: SignalCandidate, portfolio: PortfolioSnapshot) -> ValidationCheck:
        critical_provider = self._critical_provider_for(candidate.strategy_name)
        if critical_provider is None:
            return ValidationCheck(
                rule_name=self.name,
                passed=True,
                blocking=self.blocking,
                reason="no critical enrichment required",
            )
        snapshot = _require_snapshot(candidate)
        enrichment = snapshot.get_enrichment(critical_provider)
        if enrichment is None:
            return ValidationCheck(
                rule_name=self.name,
                passed=False,
                blocking=self.blocking,
                reason=f"critical enrichment '{critical_provider}' missing",
                rejection_code=RejectionReason.ENRICHMENT_MISSING_CRITICAL,
            )
        if enrichment.error:
            return ValidationCheck(
                rule_name=self.name,
                passed=False,
                blocking=self.blocking,
                reason=f"critical enrichment '{critical_provider}' errored: {enrichment.error}",
                rejection_code=RejectionReason.ENRICHMENT_MISSING_CRITICAL,
            )
        max_age = self.config.max_enrichment_age_seconds.get(critical_provider, 3600.0)
        age = enrichment.staleness_seconds
        passed = age <= max_age
        return ValidationCheck(
            rule_name=self.name,
            passed=passed,
            blocking=self.blocking,
            threshold=max_age,
            actual_value=age,
            reason=f"{critical_provider} age {age:.0f}s {'<=' if passed else '>'} {max_age:.0f}s",
            rejection_code=RejectionReason.ENRICHMENT_STALE if not passed else None,
        )

    @staticmethod
    def _critical_provider_for(strategy_name: str) -> str | None:
        mapping = {
            "weather_sniper": "weather",
            "weather_latency": "weather",
            "weather_swing": "weather",
            "crypto_structure": "crypto",
            "crypto_latency": "crypto",
            "news_signal": "news",
        }
        return mapping.get(strategy_name)


class Validator:
    def __init__(
        self,
        config: ValidationConfig | None = None,
        rules: list[ValidationRule] | None = None,
    ):
        self.config = config or ValidationConfig()
        self.rules = rules or [
            MinVolumeRule(self.config),
            MinLiquidityRule(self.config),
            ResolutionProximityRule(self.config),
            MarketAgeRule(self.config),
            MinEdgeRule(self.config),
            ReasoningPresentRule(self.config),
            EdgeBasisPresentRule(self.config),
            EvidencePresentRule(self.config),
            DuplicatePositionRule(self.config),
            MaxMarketsPerCategoryRule(self.config),
            EnrichmentStalenessRule(self.config),
        ]

    def validate(
        self,
        candidates: list[SignalCandidate],
        portfolio: PortfolioSnapshot,
    ) -> tuple[list[ValidatedSignal], list[RejectedSignal]]:
        validated: list[ValidatedSignal] = []
        rejected: list[RejectedSignal] = []

        for candidate in candidates:
            checks: list[ValidationCheck] = []
            try:
                for rule in self.rules:
                    checks.append(rule.check(candidate, portfolio))
            except Exception as exc:
                checks.append(
                    ValidationCheck(
                        rule_name="validation_error",
                        passed=False,
                        blocking=True,
                        reason=str(exc),
                        rejection_code=RejectionReason.VALIDATION_ERROR,
                    )
                )

            blocking_failures = [check for check in checks if check.blocking and not check.passed]
            if blocking_failures:
                rejected.append(
                    RejectedSignal(
                        signal=candidate,
                        checks=tuple(checks),
                        blocking_rules=tuple(check.rule_name for check in blocking_failures),
                        rejection_codes=tuple(
                            check.rejection_code
                            for check in blocking_failures
                            if check.rejection_code is not None
                        ),
                    )
                )
                continue

            warnings = tuple(
                check.reason for check in checks if not check.blocking and not check.passed
            )
            validated.append(
                ValidatedSignal(
                    signal=candidate,
                    checks=tuple(checks),
                    warnings=warnings,
                )
            )

        return validated, rejected
