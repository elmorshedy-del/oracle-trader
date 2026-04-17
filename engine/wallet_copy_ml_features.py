from __future__ import annotations

import math
from typing import Any

import pandas as pd


def build_wallet_copy_feature_frame(
    *,
    trade_row: dict[str, Any],
    wallet_performance: dict[str, Any],
    market_first_seen_timestamp: float | None,
    feature_columns: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    row = _feature_values(
        trade_row=trade_row,
        wallet_performance=wallet_performance,
        market_first_seen_timestamp=market_first_seen_timestamp,
    )
    for column in feature_columns:
        row.setdefault(column, None)
    for column in categorical_features:
        row[column] = str(row.get(column) or "unknown")
    return pd.DataFrame([{column: row.get(column) for column in feature_columns}])


def _feature_values(
    *,
    trade_row: dict[str, Any],
    wallet_performance: dict[str, Any],
    market_first_seen_timestamp: float | None,
) -> dict[str, Any]:
    price = _as_float(trade_row.get("price"))
    size_shares = _as_float(trade_row.get("size_shares"))
    size_usd = _as_float(trade_row.get("size_usd"))
    token_best_bid = _as_float(trade_row.get("token_best_bid"))
    token_best_ask = _as_float(trade_row.get("token_best_ask"))
    market_midpoint = _as_float(trade_row.get("market_midpoint"))
    market_spread = _as_float(trade_row.get("market_spread"))
    wallet_rank = _as_float(trade_row.get("wallet_leaderboard_rank"))
    wallet_profit = _as_float(trade_row.get("wallet_leaderboard_profit"))
    wallet_trade_count_24h = _as_float(trade_row.get("wallet_trade_count_24h"))
    market_seconds_to_expiry = _as_float(trade_row.get("market_seconds_to_expiry"))
    hour_of_day = _as_float(trade_row.get("hour_of_day")) or 0.0
    timestamp = _as_float(trade_row.get("timestamp")) or 0.0

    token_spread = _diff(token_best_ask, token_best_bid)
    token_midpoint = _midpoint(token_best_bid, token_best_ask)
    radians = 2.0 * math.pi * hour_of_day / 24.0

    observed_market_age_hours = None
    if market_first_seen_timestamp is not None and timestamp:
        observed_market_age_hours = max(timestamp - float(market_first_seen_timestamp), 0.0) / 3600.0

    return {
        "price": price,
        "price_distance_from_50": abs(price - 0.5) if price is not None else None,
        "size_shares": size_shares,
        "size_usd": size_usd,
        "log_size_shares": _log1p_nonnegative(size_shares),
        "log_size_usd": _log1p_nonnegative(size_usd),
        "token_best_bid": token_best_bid,
        "token_best_ask": token_best_ask,
        "token_best_bid_size": _as_float(trade_row.get("token_best_bid_size")),
        "token_best_ask_size": _as_float(trade_row.get("token_best_ask_size")),
        "token_depth_within_2pct": _as_float(trade_row.get("token_depth_within_2pct")),
        "token_spread": token_spread,
        "token_midpoint": token_midpoint,
        "price_vs_token_mid": _diff(price, token_midpoint),
        "market_yes_bid": _as_float(trade_row.get("market_yes_bid")),
        "market_yes_ask": _as_float(trade_row.get("market_yes_ask")),
        "market_no_bid": _as_float(trade_row.get("market_no_bid")),
        "market_no_ask": _as_float(trade_row.get("market_no_ask")),
        "market_spread": market_spread,
        "spread_frac_of_price": (market_spread / price) if price and market_spread is not None else None,
        "market_volume_24h": _as_float(trade_row.get("market_volume_24h")),
        "market_volume_total": _as_float(trade_row.get("market_volume_total")),
        "market_liquidity": _as_float(trade_row.get("market_liquidity")),
        "market_midpoint": market_midpoint,
        "price_vs_market_mid": _diff(price, market_midpoint),
        "market_reward_pool": _as_float(trade_row.get("market_reward_pool")),
        "hours_to_expiry": (market_seconds_to_expiry / 3600.0) if market_seconds_to_expiry is not None else None,
        "wallet_leaderboard_rank": wallet_rank,
        "wallet_leaderboard_rank_recip": (1.0 / (max(wallet_rank, 0.0) + 1.0)) if wallet_rank is not None else None,
        "wallet_leaderboard_profit": wallet_profit,
        "wallet_leaderboard_profit_log": _signed_log1p(wallet_profit),
        "wallet_leaderboard_win_rate_norm": _normalize_rate(trade_row.get("wallet_leaderboard_win_rate")),
        "wallet_open_positions": _as_float(trade_row.get("wallet_open_positions")),
        "wallet_trade_count_24h": wallet_trade_count_24h,
        "wallet_trade_rate_per_hour": (wallet_trade_count_24h / 24.0) if wallet_trade_count_24h is not None else None,
        "is_adding_to_position": 1.0 if bool(trade_row.get("is_adding_to_position")) else 0.0,
        "size_vs_wallet_avg": _as_float(trade_row.get("size_vs_wallet_avg")),
        "detection_delay_seconds": _as_float(trade_row.get("detection_delay_seconds")),
        "hour_of_day": hour_of_day,
        "hour_sin": math.sin(radians),
        "hour_cos": math.cos(radians),
        "day_of_week": _as_float(trade_row.get("day_of_week")),
        "btc_price": _as_float(trade_row.get("btc_price")),
        "btc_momentum_60s": _as_float(trade_row.get("btc_momentum_60s")),
        "observed_market_age_hours": observed_market_age_hours,
        "prior_wallet_resolution_trades": _as_float(wallet_performance.get("labeled_trades")),
        "prior_wallet_resolution_win_rate": _normalize_rate(wallet_performance.get("win_rate")),
        "prior_wallet_avg_resolution_return": _as_float(wallet_performance.get("avg_return")),
        "prior_wallet_avg_hold_hours": _as_float(wallet_performance.get("avg_hold_hours")),
        "market_category": str(trade_row.get("market_category") or "unknown"),
        "market_primary_tag": str(trade_row.get("market_primary_tag") or "unknown"),
    }


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_rate(value: Any) -> float | None:
    rate = _as_float(value)
    if rate is None or rate < 0:
        return None
    if rate > 1.0:
        rate /= 100.0
    return min(rate, 1.0)


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _log1p_nonnegative(value: float | None) -> float | None:
    if value is None:
        return None
    return math.log1p(max(value, 0.0))


def _signed_log1p(value: float | None) -> float | None:
    if value is None:
        return None
    return math.copysign(math.log1p(abs(value)), value)
