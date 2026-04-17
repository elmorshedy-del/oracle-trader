"""Compute rolling ratio state, z-scores, and features for active pairs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_MAX_LEG_LAG_MS, DEFAULT_STATS_SAMPLE_SECONDS, DEFAULT_WARMUP_SECONDS, PairRuntimeConfig
from .price_streamer import PriceBar
from .stats import compute_halflife, compute_hurst


@dataclass(slots=True)
class PairState:
    pair_key: str
    token_a: str
    token_b: str
    lookback_seconds: int
    halflife_hours: float
    discovery_score: float
    spread_mean_baseline: float
    spread_std_baseline: float
    ratio_history: deque[float] = field(default_factory=deque)
    timestamp_history: deque[int] = field(default_factory=deque)
    current_ratio: float = 0.0
    current_zscore: float = 0.0
    rolling_mean: float = 0.0
    rolling_std: float = 0.0
    estimated_halflife_seconds: float | None = None
    estimated_hurst: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    last_price_a: float = 0.0
    last_price_b: float = 0.0
    last_timestamp_a_ms: int = 0
    last_timestamp_b_ms: int = 0
    last_ratio_timestamp_ms: int = 0
    ready: bool = False
    min_warmup_seconds: int = DEFAULT_WARMUP_SECONDS


class RatioEngine:
    """Maintain live ratio statistics for the active pair set."""

    def __init__(self, pairs: list[PairRuntimeConfig], *, max_leg_lag_ms: int = DEFAULT_MAX_LEG_LAG_MS):
        self.max_leg_lag_ms = max_leg_lag_ms
        self.pairs: dict[str, PairState] = {}
        self.symbol_to_pairs: dict[str, list[str]] = {}
        self.stats: dict[str, object] = {
            "ratio_updates": 0,
            "no_price_reject": 0,
            "lag_reject": 0,
            "warmup_reject": 0,
            "per_pair": {},
        }
        for pair in pairs:
            state = PairState(
                pair_key=pair.pair_key,
                token_a=pair.token_a,
                token_b=pair.token_b,
                lookback_seconds=pair.lookback_seconds,
                halflife_hours=pair.halflife_hours,
                discovery_score=pair.discovery_score,
                spread_mean_baseline=pair.spread_mean,
                spread_std_baseline=pair.spread_std,
                ratio_history=deque(maxlen=pair.lookback_seconds),
                timestamp_history=deque(maxlen=pair.lookback_seconds),
            )
            self.pairs[pair.pair_key] = state
            self.symbol_to_pairs.setdefault(pair.token_a, []).append(pair.pair_key)
            self.symbol_to_pairs.setdefault(pair.token_b, []).append(pair.pair_key)
            self.stats["per_pair"][pair.pair_key] = {
                "ratio_updates": 0,
                "no_price_reject": 0,
                "lag_reject": 0,
                "warmup_reject": 0,
            }

    def on_price_bar(self, bar: PriceBar) -> list[str]:
        """Update all affected pairs and return the pair keys that produced new ratio ticks."""
        updated_pairs: list[str] = []
        for pair_key in self.symbol_to_pairs.get(bar.symbol, []):
            state = self.pairs[pair_key]
            if bar.symbol == state.token_a:
                state.last_price_a = bar.close
                state.last_timestamp_a_ms = bar.timestamp_ms
            elif bar.symbol == state.token_b:
                state.last_price_b = bar.close
                state.last_timestamp_b_ms = bar.timestamp_ms
            else:
                continue
            if bar.timestamp_ms <= state.last_ratio_timestamp_ms:
                continue
            reject_reason = self._rejection_reason(state)
            if reject_reason is not None:
                self._count_reject(pair_key, reject_reason)
                continue
            self._update_ratio(state, bar.timestamp_ms)
            if not state.ready:
                self._count_reject(pair_key, "warmup_reject")
            updated_pairs.append(pair_key)
        return updated_pairs

    def get_state(self, pair_key: str) -> PairState | None:
        return self.pairs.get(pair_key)

    def get_ready_states(self) -> dict[str, PairState]:
        return {pair_key: state for pair_key, state in self.pairs.items() if state.ready}

    def _rejection_reason(self, state: PairState) -> str | None:
        if state.last_price_a <= 0 or state.last_price_b <= 0:
            return "no_price_reject"
        if state.last_timestamp_a_ms <= 0 or state.last_timestamp_b_ms <= 0:
            return "no_price_reject"
        if abs(state.last_timestamp_a_ms - state.last_timestamp_b_ms) > self.max_leg_lag_ms:
            return "lag_reject"
        return None

    def _update_ratio(self, state: PairState, timestamp_ms: int) -> None:
        ratio = float(np.log(state.last_price_a / state.last_price_b))
        state.current_ratio = ratio
        state.last_ratio_timestamp_ms = timestamp_ms
        state.ratio_history.append(ratio)
        state.timestamp_history.append(timestamp_ms)
        self._count_stat(state.pair_key, "ratio_updates")

        count = len(state.ratio_history)
        if count < state.min_warmup_seconds:
            state.ready = False
            return

        ratios = np.asarray(state.ratio_history, dtype=float)
        state.rolling_mean = float(np.mean(ratios))
        state.rolling_std = float(np.std(ratios))
        if state.rolling_std <= 0:
            state.current_zscore = 0.0
            state.ready = False
            return

        state.ready = True
        state.current_zscore = (ratio - state.rolling_mean) / state.rolling_std
        state.features = self._compute_features(ratios, state)

    def _compute_features(self, ratios: np.ndarray, state: PairState) -> dict[str, float]:
        count = len(ratios)
        zscore = float(state.current_zscore)
        z_velocity_1m = self._z_velocity(ratios, state, 60)
        z_velocity_5m = self._z_velocity(ratios, state, 300)
        z_velocity_1h = self._z_velocity(ratios, state, 3_600)
        recent_std = float(np.std(ratios[-300:])) if count > 300 else state.rolling_std
        vol_ratio = recent_std / state.rolling_std if state.rolling_std > 0 else 1.0
        above_pct = float(np.mean(ratios[-300:] > state.rolling_mean)) if count > 300 else 0.5
        consecutive_seconds = float(self._consecutive_seconds(ratios, state))

        stats_sample = ratios[-min(len(ratios), DEFAULT_STATS_SAMPLE_SECONDS):]
        spread_sample = pd.Series(stats_sample)
        halflife_seconds = compute_halflife(spread_sample)
        hurst = compute_hurst(spread_sample)
        state.estimated_halflife_seconds = None if not np.isfinite(halflife_seconds) else float(halflife_seconds)
        state.estimated_hurst = None if not np.isfinite(hurst) else float(hurst)

        return {
            "zscore": zscore,
            "abs_zscore": abs(zscore),
            "z_positive": 1.0 if zscore > 0 else 0.0,
            "z_velocity_1m": z_velocity_1m,
            "z_velocity_5m": z_velocity_5m,
            "z_velocity_1h": z_velocity_1h,
            "vol_ratio": float(vol_ratio),
            "above_mean_pct": above_pct,
            "consecutive_seconds": consecutive_seconds,
            "rolling_std": float(state.rolling_std),
            "estimated_halflife_seconds": float(state.estimated_halflife_seconds or 0.0),
            "estimated_hurst": float(state.estimated_hurst or 0.5),
        }

    def _z_velocity(self, ratios: np.ndarray, state: PairState, offset: int) -> float:
        if len(ratios) <= offset or state.rolling_std <= 0:
            return 0.0
        prior_z = (ratios[-offset] - state.rolling_mean) / state.rolling_std
        return float(state.current_zscore - prior_z)

    def _consecutive_seconds(self, ratios: np.ndarray, state: PairState) -> int:
        if state.rolling_std <= 0 or len(ratios) == 0:
            return 0
        direction_positive = state.current_zscore >= 0
        consecutive = 0
        for ratio in ratios[::-1]:
            prior_z = (ratio - state.rolling_mean) / state.rolling_std
            if (prior_z >= 0) == direction_positive:
                consecutive += 1
            else:
                break
        return consecutive

    def _count_reject(self, pair_key: str, reason: str) -> None:
        self._count_stat(pair_key, reason)

    def _count_stat(self, pair_key: str, key: str) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + 1
        per_pair = self.stats["per_pair"][pair_key]
        per_pair[key] = int(per_pair.get(key, 0)) + 1
