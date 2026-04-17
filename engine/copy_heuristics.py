from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from config import CopyHeuristicShadowProfileConfig


POSITION_SIZE_TIERS = (
    (0.75, 300.0),
    (0.60, 200.0),
    (0.45, 100.0),
    (0.30, 50.0),
)


@dataclass(slots=True)
class HeuristicEvaluation:
    score: float
    should_copy: bool
    suggested_size_usd: float
    reasons: list[str]
    context: dict[str, Any]
    position_outcome: str | None = None
    position_token_id: str | None = None
    position_side: str | None = None


def evaluate_trade(
    *,
    profile: CopyHeuristicShadowProfileConfig,
    trade_row: dict[str, Any],
    wallet_performance: dict[str, Any],
    consensus_rows: list[dict[str, Any]] | None = None,
    consensus_avg_win_rate: float | None = None,
    market_first_seen_timestamp: float | None = None,
    prior_buy_row: dict[str, Any] | None = None,
    available_cash_usd: float,
) -> HeuristicEvaluation:
    consensus_rows = consensus_rows or []
    context: dict[str, Any] = {
        "strategy_key": profile.strategy_key,
        "profile_kind": profile.kind,
        "wallet_labeled_trades": int(wallet_performance.get("labeled_trades") or 0),
        "wallet_observed_win_rate": _normalize_rate(wallet_performance.get("win_rate")),
        "wallet_avg_hold_hours": _as_float(wallet_performance.get("avg_hold_hours")),
        "leaderboard_win_rate": _normalize_rate(trade_row.get("wallet_leaderboard_win_rate")),
        "wallet_effective_win_rate": None,
        "wallet_win_rate_source": "none",
        "market_spread": _market_spread(trade_row),
        "detection_delay_seconds": _as_float(trade_row.get("detection_delay_seconds")) or 0.0,
        "wallet_trade_count_24h": int(trade_row.get("wallet_trade_count_24h") or 0),
        "size_vs_wallet_avg": _as_float(trade_row.get("size_vs_wallet_avg")),
        "is_adding_to_position": bool(trade_row.get("is_adding_to_position")),
        "market_age_minutes": _market_age_minutes(trade_row, market_first_seen_timestamp),
        "market_hours_remaining": _market_hours_remaining(trade_row),
        "consensus_wallets": 0,
        "consensus_avg_win_rate": _normalize_rate(consensus_avg_win_rate),
        "consensus_strength": None,
        "prior_hold_hours": None,
        "prior_loss_cut_pct": None,
        "first_tracked_wallet": None,
    }
    effective_win_rate, win_rate_source = _effective_wallet_win_rate(
        trade_row=trade_row,
        wallet_performance=wallet_performance,
        min_labeled=profile.min_wallet_labeled_trades,
    )
    context["wallet_effective_win_rate"] = effective_win_rate
    context["wallet_win_rate_source"] = win_rate_source

    score = 0.0
    reasons: list[str] = []
    position_outcome = str(trade_row.get("outcome") or "") or None
    position_token_id = str(trade_row.get("asset_token_id") or "") or None

    if profile.kind == "selective_copy":
        score, reasons = _score_selective_copy(
            profile=profile,
            trade_row=trade_row,
            effective_win_rate=effective_win_rate,
            context=context,
        )
    elif profile.kind == "whale_consensus":
        score, reasons = _score_whale_consensus(
            profile=profile,
            trade_row=trade_row,
            effective_win_rate=effective_win_rate,
            consensus_rows=consensus_rows,
            consensus_avg_win_rate=consensus_avg_win_rate,
            context=context,
        )
    elif profile.kind == "contrarian_exit":
        score, reasons, position_outcome, position_token_id = _score_contrarian_exit(
            profile=profile,
            trade_row=trade_row,
            effective_win_rate=effective_win_rate,
            prior_buy_row=prior_buy_row,
            context=context,
        )
    elif profile.kind == "fresh_market":
        score, reasons = _score_fresh_market(
            profile=profile,
            trade_row=trade_row,
            effective_win_rate=effective_win_rate,
            market_first_seen_timestamp=market_first_seen_timestamp,
            context=context,
        )

    score, reasons = _apply_common_penalties(
        score=score,
        reasons=reasons,
        trade_row=trade_row,
        profile=profile,
        context=context,
    )
    score = max(0.0, min(score, 1.0))
    suggested_size_usd = _position_size_for_score(
        score=score,
        profile=profile,
        available_cash_usd=available_cash_usd,
    )
    should_copy = (
        score >= profile.score_threshold
        and suggested_size_usd >= profile.min_trade_usd
        and position_token_id is not None
        and position_outcome is not None
    )
    position_side = _binary_side_from_outcome(position_outcome)
    return HeuristicEvaluation(
        score=score,
        should_copy=should_copy,
        suggested_size_usd=suggested_size_usd,
        reasons=reasons,
        context=context,
        position_outcome=position_outcome,
        position_token_id=position_token_id,
        position_side=position_side,
    )


def build_decision_key(*, strategy_key: str, trade_id: int) -> str:
    return f"{strategy_key}:{trade_id}"


def parse_json_payload(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        payload = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _score_selective_copy(
    *,
    profile: CopyHeuristicShadowProfileConfig,
    trade_row: dict[str, Any],
    effective_win_rate: float | None,
    context: dict[str, Any],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    size_vs_avg = _as_float(trade_row.get("size_vs_wallet_avg")) or 0.0
    spread = _market_spread(trade_row)
    detection_delay = _as_float(trade_row.get("detection_delay_seconds")) or 0.0
    wallet_trade_count = int(trade_row.get("wallet_trade_count_24h") or 0)
    is_first_entry = not bool(trade_row.get("is_adding_to_position"))
    if (
        effective_win_rate is not None
        and effective_win_rate >= profile.min_wallet_win_rate
        and wallet_trade_count < profile.max_wallet_trade_count_24h
        and size_vs_avg >= profile.min_size_vs_avg
        and spread is not None
        and spread < profile.max_market_spread
        and detection_delay < profile.max_detection_delay_seconds
        and (not profile.require_first_entry or is_first_entry)
    ):
        score += 0.30
        reasons.append("selective_conviction")
    context["first_entry"] = is_first_entry
    return score, reasons


def _score_whale_consensus(
    *,
    profile: CopyHeuristicShadowProfileConfig,
    trade_row: dict[str, Any],
    effective_win_rate: float | None,
    consensus_rows: list[dict[str, Any]],
    consensus_avg_win_rate: float | None,
    context: dict[str, Any],
) -> tuple[float, list[str]]:
    del effective_win_rate
    score = 0.0
    reasons: list[str] = []
    spread = _market_spread(trade_row)
    wallet_ids = {str(trade_row.get("wallet_address") or "")}
    sized_scores: list[float] = []
    for row in consensus_rows:
        wallet_address = str(row.get("wallet_address") or "")
        if not wallet_address:
            continue
        wallet_ids.add(wallet_address)
        row_wr = _normalize_rate(row.get("effective_wallet_win_rate") or row.get("wallet_leaderboard_win_rate"))
        if row_wr is None:
            continue
        sized_scores.append(row_wr * math.log1p(max(_as_float(row.get("size_usd")) or 0.0, 0.0)))
    consensus_wallets = len(wallet_ids)
    context["consensus_wallets"] = consensus_wallets
    context["consensus_strength"] = round(sum(sized_scores) / max(len(sized_scores), 1), 6) if sized_scores else 0.0
    if (
        consensus_wallets >= profile.min_consensus_wallets
        and consensus_avg_win_rate is not None
        and consensus_avg_win_rate >= profile.min_consensus_avg_win_rate
        and spread is not None
        and spread < profile.max_market_spread
    ):
        score += 0.35
        score += min(0.15, max(consensus_wallets - profile.min_consensus_wallets, 0) * 0.05)
        score += min(0.10, max(consensus_avg_win_rate - profile.min_consensus_avg_win_rate, 0.0))
        reasons.append(f"consensus_{consensus_wallets}_wallets")
    return score, reasons


def _score_contrarian_exit(
    *,
    profile: CopyHeuristicShadowProfileConfig,
    trade_row: dict[str, Any],
    effective_win_rate: float | None,
    prior_buy_row: dict[str, Any] | None,
    context: dict[str, Any],
) -> tuple[float, list[str], str | None, str | None]:
    score = 0.0
    reasons: list[str] = []
    if str(trade_row.get("side") or "").upper() != "SELL" or prior_buy_row is None:
        return score, reasons, None, None
    position_outcome, position_token_id = _opposite_binary_outcome(
        raw_market_json=trade_row.get("raw_market_json"),
        current_outcome=str(prior_buy_row.get("outcome") or ""),
        current_token_id=str(prior_buy_row.get("asset_token_id") or ""),
    )
    if not position_outcome or not position_token_id:
        return score, reasons, None, None
    hold_hours = max((_as_float(trade_row.get("timestamp")) or 0.0) - (_as_float(prior_buy_row.get("timestamp")) or 0.0), 0.0) / 3600.0
    sell_price = _as_float(trade_row.get("price")) or 0.0
    prior_buy_price = _as_float(prior_buy_row.get("price")) or 0.0
    loss_cut_pct = ((prior_buy_price - sell_price) / prior_buy_price) if prior_buy_price > 0 and sell_price < prior_buy_price else None
    context["prior_hold_hours"] = hold_hours
    context["prior_loss_cut_pct"] = loss_cut_pct
    if (
        effective_win_rate is not None
        and effective_win_rate >= profile.min_wallet_win_rate
        and hold_hours >= profile.min_hold_hours
        and loss_cut_pct is not None
        and loss_cut_pct > 0
        and (_market_hours_remaining(trade_row) or 0.0) >= profile.min_market_hours_remaining
    ):
        score += 0.25
        score += min(0.15, loss_cut_pct)
        if hold_hours >= (profile.min_hold_hours * 2):
            score += 0.05
        reasons.append("smart_money_cutting")
    return score, reasons, position_outcome, position_token_id


def _score_fresh_market(
    *,
    profile: CopyHeuristicShadowProfileConfig,
    trade_row: dict[str, Any],
    effective_win_rate: float | None,
    market_first_seen_timestamp: float | None,
    context: dict[str, Any],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    trade_timestamp = _as_float(trade_row.get("timestamp")) or 0.0
    market_age_minutes = _market_age_minutes(trade_row, market_first_seen_timestamp)
    is_first_tracked_wallet = (
        market_first_seen_timestamp is not None and abs(trade_timestamp - market_first_seen_timestamp) <= 60.0
    )
    context["first_tracked_wallet"] = is_first_tracked_wallet
    if (
        market_age_minutes is not None
        and market_age_minutes <= profile.max_market_age_minutes
        and effective_win_rate is not None
        and effective_win_rate >= profile.min_wallet_win_rate
        and (_as_float(trade_row.get("size_usd")) or 0.0) >= profile.min_trade_size_usd
        and (_market_spread(trade_row) or 1.0) < profile.max_market_spread
    ):
        score += 0.25
        if is_first_tracked_wallet:
            score += 0.05
        if (_as_float(trade_row.get("size_usd")) or 0.0) >= profile.min_trade_size_usd * 2:
            score += 0.05
        reasons.append("fresh_market_first_mover")
    return score, reasons


def _apply_common_penalties(
    *,
    score: float,
    reasons: list[str],
    trade_row: dict[str, Any],
    profile: CopyHeuristicShadowProfileConfig,
    context: dict[str, Any],
) -> tuple[float, list[str]]:
    del profile
    detection_delay = _as_float(trade_row.get("detection_delay_seconds")) or 0.0
    spread = _market_spread(trade_row)
    wallet_trade_count = int(trade_row.get("wallet_trade_count_24h") or 0)
    if detection_delay > 30.0:
        score *= 0.5
        reasons.append("stale_signal")
    if spread is not None and spread > 0.10:
        score *= 0.7
        reasons.append("wide_spread")
    if wallet_trade_count > 15:
        score *= 0.5
        reasons.append("spray_trader")
    context["post_penalty_score"] = round(score, 6)
    return score, reasons


def _position_size_for_score(
    *,
    score: float,
    profile: CopyHeuristicShadowProfileConfig,
    available_cash_usd: float,
) -> float:
    base_size = 0.0
    for min_score, size in POSITION_SIZE_TIERS:
        if score >= min_score:
            base_size = size
            break
    if base_size <= 0:
        return 0.0
    capped = min(base_size, profile.max_trade_usd, available_cash_usd)
    if capped < profile.min_trade_usd:
        return 0.0
    return round(capped, 4)


def _effective_wallet_win_rate(
    *,
    trade_row: dict[str, Any],
    wallet_performance: dict[str, Any],
    min_labeled: int,
) -> tuple[float | None, str]:
    observed = _normalize_rate(wallet_performance.get("win_rate"))
    labeled_trades = int(wallet_performance.get("labeled_trades") or 0)
    if observed is not None and labeled_trades >= min_labeled:
        return observed, "observed"
    leaderboard = _normalize_rate(trade_row.get("wallet_leaderboard_win_rate"))
    if leaderboard is not None:
        return leaderboard, "leaderboard"
    return observed, "observed_sparse" if observed is not None else "none"


def _market_spread(trade_row: dict[str, Any]) -> float | None:
    spread = _as_float(trade_row.get("market_spread"))
    if spread is not None:
        return spread
    bid = _as_float(trade_row.get("token_best_bid"))
    ask = _as_float(trade_row.get("token_best_ask"))
    if bid is None or ask is None:
        return None
    return max(ask - bid, 0.0)


def _market_age_minutes(trade_row: dict[str, Any], market_first_seen_timestamp: float | None) -> float | None:
    trade_timestamp = _as_float(trade_row.get("timestamp"))
    if trade_timestamp is None:
        return None
    raw_market = parse_json_payload(trade_row.get("raw_market_json"))
    created_ts = _timestamp_from_market(raw_market)
    if created_ts is None:
        created_ts = market_first_seen_timestamp
    if created_ts is None:
        return None
    return max(trade_timestamp - created_ts, 0.0) / 60.0


def _market_hours_remaining(trade_row: dict[str, Any]) -> float | None:
    seconds = _as_float(trade_row.get("market_seconds_to_expiry"))
    if seconds is None:
        return None
    return seconds / 3600.0


def _timestamp_from_market(raw_market: dict[str, Any]) -> float | None:
    for key in ("createdAt", "startDate", "start_time", "created_at"):
        value = raw_market.get(key)
        if not value:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return _iso_to_timestamp(str(value))
            except ValueError:
                continue
    return None


def _opposite_binary_outcome(
    *,
    raw_market_json: Any,
    current_outcome: str,
    current_token_id: str,
) -> tuple[str | None, str | None]:
    raw_market = parse_json_payload(raw_market_json)
    outcomes = _parse_string_list(raw_market.get("outcomes"))
    token_ids = _parse_string_list(raw_market.get("clobTokenIds"))
    if len(outcomes) < 2 or len(token_ids) < 2:
        return None, None
    normalized = {outcomes[index].strip().lower(): token_ids[index] for index in range(min(len(outcomes), len(token_ids)))}
    current_name = current_outcome.strip().lower()
    if current_name == "yes" and "no" in normalized:
        return "No", normalized["no"]
    if current_name == "no" and "yes" in normalized:
        return "Yes", normalized["yes"]
    for index, token_id in enumerate(token_ids):
        if token_id == current_token_id:
            for other_index, other_token_id in enumerate(token_ids):
                if other_index != index:
                    return outcomes[other_index], other_token_id
    return None, None


def _binary_side_from_outcome(outcome: str | None) -> str | None:
    if outcome is None:
        return None
    normalized = outcome.strip().lower()
    if normalized == "yes":
        return "YES"
    if normalized == "no":
        return "NO"
    return outcome


def _normalize_rate(value: Any) -> float | None:
    rate = _as_float(value)
    if rate is None:
        return None
    if rate > 1.0 and rate <= 100.0:
        rate /= 100.0
    return rate


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_string_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(payload, list):
            return [str(item) for item in payload]
    return []


def _iso_to_timestamp(raw_value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).timestamp()
