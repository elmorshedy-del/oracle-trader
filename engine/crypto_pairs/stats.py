"""Shared statistics helpers for crypto pairs discovery and runtime."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_halflife(spread: pd.Series) -> float:
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    merged = pd.DataFrame({"lag": spread_lag, "diff": spread_diff}).dropna()
    if len(merged) < 20:
        return float("inf")
    theta = float(np.polyfit(merged["lag"], merged["diff"], 1)[0])
    if theta >= 0:
        return float("inf")
    return -math.log(2) / theta


def compute_hurst(spread: pd.Series, *, max_lag: int = 50) -> float:
    values = spread.to_numpy(dtype=float)
    lags = range(2, min(max_lag, len(values) // 2))
    tau = []
    used_lags = []
    for lag in lags:
        diff = values[lag:] - values[:-lag]
        std = np.std(diff)
        if std <= 0:
            continue
        tau.append(std)
        used_lags.append(lag)
    if len(tau) < 2:
        return 0.5
    return float(np.polyfit(np.log(used_lags), np.log(tau), 1)[0])

