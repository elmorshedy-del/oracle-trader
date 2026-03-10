from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .config import DecisioningConfig
from .context import family_key_from_position, family_key_from_signal, theme_key_from_position, theme_key_from_signal
from .contracts import ModuleHealth, PortfolioSnapshot, PositionState, RejectedSignal, SignalCandidate, ValidatedSignal, ValidationCheck
from .enums import ModuleStatus, RejectionReason
from .llm import LLMTaskResult, MultiagentLLMRouter


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OpusDecisionLayer:
    def __init__(self, llm_router: MultiagentLLMRouter, config: DecisioningConfig) -> None:
        self.llm_router = llm_router
        self.config = config

    async def gate_signals(
        self,
        validated: list[ValidatedSignal],
        *,
        portfolio: PortfolioSnapshot,
        context: Any,
    ) -> tuple[list[ValidatedSignal], list[RejectedSignal], ModuleHealth, dict[str, int], list[dict[str, Any]], list[SignalCandidate]]:
        task_cfg = self.llm_router.config.tasks.get("trade_gate")
        if not validated or task_cfg is None or not self.llm_router.config.enabled or not task_cfg.enabled:
            return (
                validated,
                [],
                ModuleHealth(
                    module_name="llm.trade_gate",
                    status=ModuleStatus.HEALTHY,
                    last_run_at=utc_now(),
                    last_duration_seconds=0.0,
                    items_in=len(validated),
                    items_out=len(validated),
                ),
                {},
                [],
                [],
            )

        batch = sorted(validated, key=lambda item: item.signal.edge_estimate, reverse=True)[: self.config.trade_gate_batch_size]
        decisions_result, attempts = await self.llm_router.complete_json(
            task_name="trade_gate",
            system_prompt=_trade_gate_system_prompt(),
            user_payload=_trade_gate_payload(batch, portfolio, context),
            required_keys=("decisions",),
        )

        decision_rows: list[dict[str, Any]] = []
        for attempt in attempts:
            decision_rows.append(
                {
                    "task": "trade_gate",
                    "provider": attempt.provider,
                    "model": attempt.model,
                    "success": attempt.success,
                    "parsed": attempt.parsed,
                    "latency_ms": attempt.latency_ms,
                    "prompt_hash": attempt.prompt_hash,
                    "error": attempt.error,
                    "role": attempt.role,
                    "created_at": attempt.created_at,
                }
            )

        if decisions_result is None or decisions_result.parsed_json is None:
            status = ModuleStatus.DEGRADED if self.config.trade_gate_fail_open else ModuleStatus.FAILED
            health = ModuleHealth(
                module_name="llm.trade_gate",
                status=status,
                last_run_at=utc_now(),
                last_duration_seconds=(attempts[-1].latency_ms / 1000.0) if attempts else 0.0,
                last_error=attempts[-1].error if attempts else "llm_trade_gate_unavailable",
                items_in=len(batch),
                items_out=len(batch) if self.config.trade_gate_fail_open else 0,
            )
            if self.config.trade_gate_fail_open:
                return validated, [], health, {}, decision_rows, []
            rejected = [_llm_reject(item, "defer", "trade gate unavailable", "llm_trade_gate_unavailable") for item in batch]
            approved_ids = {item.signal.signal_id for item in validated if item not in batch}
            approved = [item for item in validated if item.signal.signal_id in approved_ids]
            remembered = [item.signal for item in batch]
            return approved, rejected, health, {"defer": len(rejected)}, decision_rows, remembered

        decision_map = {
            item.get("signal_id"): item
            for item in decisions_result.parsed_json.get("decisions", [])
            if isinstance(item, dict) and item.get("signal_id")
        }
        approved: list[ValidatedSignal] = []
        rejected: list[RejectedSignal] = []
        remembered: list[SignalCandidate] = []
        action_counts: dict[str, int] = {}

        batch_ids = {item.signal.signal_id for item in batch}
        for item in validated:
            if item.signal.signal_id not in batch_ids:
                approved.append(item)
                continue
            decision = decision_map.get(item.signal.signal_id)
            if decision is None:
                if self.config.trade_gate_fail_open:
                    approved.append(item)
                    continue
                rejected.append(_llm_reject(item, "defer", "trade gate returned no decision", "llm_trade_gate_no_decision"))
                remembered.append(item.signal)
                action_counts["defer"] = action_counts.get("defer", 0) + 1
                continue

            action = str(decision.get("action", "defer")).lower()
            reason = str(decision.get("reason", "") or "")
            confidence = float(decision.get("confidence", 0.0) or 0.0)
            action_counts[action] = action_counts.get(action, 0) + 1
            decision_rows.append(
                {
                    "task": "trade_gate",
                    "signal_id": item.signal.signal_id,
                    "action": action,
                    "confidence": confidence,
                    "reason": reason,
                    "provider": decisions_result.provider,
                    "model": decisions_result.model,
                    "created_at": decisions_result.created_at,
                }
            )
            if action == "take":
                approved.append(_llm_annotate_validated(item, decisions_result, action, reason, confidence))
            else:
                rejected.append(_llm_reject(item, action, reason, None))
                remembered.append(item.signal)

        health = ModuleHealth(
            module_name="llm.trade_gate",
            status=ModuleStatus.HEALTHY,
            last_run_at=utc_now(),
            last_duration_seconds=decisions_result.latency_ms / 1000.0,
            items_in=len(batch),
            items_out=len(approved),
        )
        return approved, rejected, health, action_counts, decision_rows, remembered

    async def judge_exits(
        self,
        *,
        portfolio: PortfolioSnapshot,
        context: Any,
    ) -> tuple[dict[str, dict[str, Any]], ModuleHealth, dict[str, int], list[dict[str, Any]]]:
        task_cfg = self.llm_router.config.tasks.get("exit_judge")
        eligible = _eligible_positions_for_exit(portfolio.positions, context, self.config)
        if not eligible or task_cfg is None or not self.llm_router.config.enabled or not task_cfg.enabled:
            return (
                {},
                ModuleHealth(
                    module_name="llm.exit_judge",
                    status=ModuleStatus.HEALTHY,
                    last_run_at=utc_now(),
                    last_duration_seconds=0.0,
                    items_in=len(eligible),
                    items_out=0,
                ),
                {},
                [],
            )

        decisions_result, attempts = await self.llm_router.complete_json(
            task_name="exit_judge",
            system_prompt=_exit_judge_system_prompt(),
            user_payload=_exit_judge_payload(eligible, portfolio, context),
            required_keys=("decisions",),
        )
        decision_rows: list[dict[str, Any]] = []
        for attempt in attempts:
            decision_rows.append(
                {
                    "task": "exit_judge",
                    "provider": attempt.provider,
                    "model": attempt.model,
                    "success": attempt.success,
                    "parsed": attempt.parsed,
                    "latency_ms": attempt.latency_ms,
                    "prompt_hash": attempt.prompt_hash,
                    "error": attempt.error,
                    "role": attempt.role,
                    "created_at": attempt.created_at,
                }
            )

        if decisions_result is None or decisions_result.parsed_json is None:
            status = ModuleStatus.DEGRADED if self.config.exit_judge_fail_open else ModuleStatus.FAILED
            return (
                {},
                ModuleHealth(
                    module_name="llm.exit_judge",
                    status=status,
                    last_run_at=utc_now(),
                    last_duration_seconds=(attempts[-1].latency_ms / 1000.0) if attempts else 0.0,
                    last_error=attempts[-1].error if attempts else "llm_exit_judge_unavailable",
                    items_in=len(eligible),
                    items_out=0,
                ),
                {},
                decision_rows,
            )

        actions: dict[str, dict[str, Any]] = {}
        action_counts: dict[str, int] = {}
        for item in decisions_result.parsed_json.get("decisions", []):
            if not isinstance(item, dict):
                continue
            position_id = str(item.get("position_id", "")).strip()
            if not position_id:
                continue
            action = str(item.get("action", "hold")).lower()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            reason = str(item.get("reason", "") or "")
            action_counts[action] = action_counts.get(action, 0) + 1
            actions[position_id] = {
                "action": action,
                "confidence": confidence,
                "reason": reason,
                "provider": decisions_result.provider,
                "model": decisions_result.model,
                "created_at": decisions_result.created_at,
            }
            decision_rows.append(
                {
                    "task": "exit_judge",
                    "position_id": position_id,
                    "action": action,
                    "confidence": confidence,
                    "reason": reason,
                    "provider": decisions_result.provider,
                    "model": decisions_result.model,
                    "created_at": decisions_result.created_at,
                }
            )

        return (
            actions,
            ModuleHealth(
                module_name="llm.exit_judge",
                status=ModuleStatus.HEALTHY,
                last_run_at=utc_now(),
                last_duration_seconds=decisions_result.latency_ms / 1000.0,
                items_in=len(eligible),
                items_out=action_counts.get("exit", 0),
            ),
            action_counts,
            decision_rows,
        )


def _llm_annotate_validated(
    item: ValidatedSignal,
    result: LLMTaskResult,
    action: str,
    reason: str,
    confidence: float,
) -> ValidatedSignal:
    candidate_signal = replace(
        item.signal,
        llm_involved=True,
        reasoning=f"{item.signal.reasoning} | LLM TRADE GATE ({action} {confidence:.0%}): {reason}".strip(),
        metadata={
            **item.signal.metadata,
            "llm_trade_gate_action": action,
            "llm_trade_gate_reason": reason,
            "llm_trade_gate_confidence": confidence,
            "llm_trade_gate_provider": result.provider,
            "llm_trade_gate_model": result.model,
            "llm_trade_gate_prompt_hash": result.prompt_hash,
        },
    )
    return replace(item, signal=candidate_signal)


def _llm_reject(
    item: ValidatedSignal,
    action: str,
    reason: str,
    health_error: str | None,
) -> RejectedSignal:
    rejection_code = (
        RejectionReason.LLM_TRADE_GATE_DEFERRED
        if action == "defer"
        else RejectionReason.LLM_TRADE_GATE_REJECTED
    )
    check = ValidationCheck(
        rule_name="llm_trade_gate",
        passed=False,
        blocking=True,
        reason=reason or health_error or f"llm_trade_gate:{action}",
        rejection_code=rejection_code,
    )
    return RejectedSignal(
        signal=item.signal,
        checks=item.checks + (check,),
        blocking_rules=item.blocking_rules + ("llm_trade_gate",),
        rejection_codes=item.rejection_codes + (rejection_code,),
    )


def _trade_gate_payload(
    validated: list[ValidatedSignal],
    portfolio: PortfolioSnapshot,
    context: Any,
) -> dict[str, Any]:
    top_theme_exposure = sorted(context.open_theme_exposure.items(), key=lambda item: item[1], reverse=True)[:8]
    open_positions = sorted(portfolio.positions, key=lambda item: item.current_price * item.shares, reverse=True)[:12]
    return {
        "portfolio": {
            "total_capital": portfolio.total_capital,
            "available_capital": portfolio.available_capital,
            "deployed_capital": portfolio.deployed_capital,
            "open_positions": portfolio.position_count,
            "mark_win_rate": portfolio.mark_win_rate,
            "top_theme_exposure": [
                {"theme": key, "exposure": value, "positions": context.open_theme_counts.get(key, 0)}
                for key, value in top_theme_exposure
            ],
            "open_positions_preview": [
                {
                    "position_id": position.position_id,
                    "market_id": position.market_id,
                    "market_question": position.market_question,
                    "strategy_name": position.strategy_name,
                    "current_price": position.current_price,
                    "entry_price": position.entry_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "family_key": family_key_from_position(position),
                    "theme_key": theme_key_from_position(position),
                }
                for position in open_positions
            ],
        },
        "memory": {
            "recent_signal_keys": len(context.recent_signal_entries),
            "recent_headlines": len(context.recent_headline_entries),
            "recent_families": len(context.recent_family_entries),
            "recent_themes": len(context.recent_theme_entries),
        },
        "candidates": [
            {
                "signal_id": item.signal.signal_id,
                "strategy_name": item.signal.strategy_name,
                "market_id": item.signal.market_id,
                "question": item.signal.market_snapshot.question if item.signal.market_snapshot else item.signal.market_id,
                "direction": item.signal.direction.value,
                "outcome": item.signal.outcome,
                "current_price": item.signal.current_price,
                "estimated_fair_value": item.signal.estimated_fair_value,
                "edge_estimate": item.signal.edge_estimate,
                "edge_basis": item.signal.edge_basis,
                "reasoning": item.signal.reasoning,
                "family_key": family_key_from_signal(item.signal),
                "theme_key": theme_key_from_signal(item.signal),
                "open_family_positions": context.family_positions(family_key_from_signal(item.signal)),
                "open_theme_positions": context.theme_positions(theme_key_from_signal(item.signal)),
                "open_theme_exposure": context.theme_exposure(theme_key_from_signal(item.signal)),
                "checks": [check.reason for check in item.checks if check.passed],
            }
            for item in validated
        ],
    }


def _eligible_positions_for_exit(
    positions: tuple[PositionState, ...],
    context: Any,
    config: DecisioningConfig,
) -> list[PositionState]:
    now = utc_now()
    ranked: list[tuple[float, PositionState]] = []
    for position in positions:
        held_hours = (now - position.opened_at).total_seconds() / 3600
        if held_hours < config.exit_judge_min_hold_hours:
            continue
        theme_key = theme_key_from_position(position)
        family_key = family_key_from_position(position)
        crowding = context.theme_positions(theme_key) + context.family_positions(family_key)
        score = crowding + max(0.0, held_hours - config.exit_judge_min_hold_hours)
        if config.exit_judge_require_nontrivial_context and crowding <= 1 and position.unrealized_pnl > 0:
            continue
        ranked.append((score, position))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[: config.exit_judge_max_positions_per_cycle]]


def _exit_judge_payload(
    positions: list[PositionState],
    portfolio: PortfolioSnapshot,
    context: Any,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "portfolio": {
            "total_capital": portfolio.total_capital,
            "available_capital": portfolio.available_capital,
            "deployed_capital": portfolio.deployed_capital,
            "open_positions": portfolio.position_count,
            "mark_win_rate": portfolio.mark_win_rate,
        },
        "positions": [
            {
                "position_id": position.position_id,
                "market_id": position.market_id,
                "market_question": position.market_question,
                "strategy_name": position.strategy_name,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "shares": position.shares,
                "unrealized_pnl": position.unrealized_pnl,
                "held_hours": (now - position.opened_at).total_seconds() / 3600,
                "target_price": float(position.metadata.get("target_price", 0.0) or 0.0),
                "family_key": family_key_from_position(position),
                "theme_key": theme_key_from_position(position),
                "theme_positions": context.theme_positions(theme_key_from_position(position)),
                "theme_exposure": context.theme_exposure(theme_key_from_position(position)),
            }
            for position in positions
        ],
    }


def _trade_gate_system_prompt() -> str:
    return (
        "You are the Opus trade gate for a paper-trading research lab. "
        "Your job is not to size or execute. Your job is to decide whether a validated candidate is truly worth taking now, "
        "given what is already held. Be skeptical of duplicate theses, repeated family entries, crowded themes, stale repeated headlines, "
        "and trades that add little novelty. Favor new, clearly differentiated, high-edge ideas. "
        "Return strict JSON: {\"decisions\":[{\"signal_id\":\"...\",\"action\":\"take|skip|defer\",\"confidence\":0.0-1.0,\"reason\":\"...\"}]}."
    )


def _exit_judge_system_prompt() -> str:
    return (
        "You are the Opus exit judge for a paper-trading research lab. "
        "Review open positions and decide if capital should be recycled now. Prefer exit when a position is overcrowded by theme/family, "
        "has weak follow-through, or appears to be tying up capital without enough progress. Do not replace hard stops or guaranteed target exits; "
        "this is an additional intelligence layer for recycling capital sooner. "
        "Return strict JSON: {\"decisions\":[{\"position_id\":\"...\",\"action\":\"hold|exit\",\"confidence\":0.0-1.0,\"reason\":\"...\"}]}."
    )
