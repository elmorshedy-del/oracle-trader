"""Cointegration helpers for live crypto-pairs monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

try:  # pragma: no cover - optional runtime dependency
    from statsmodels.tsa.stattools import coint as _statsmodels_coint
except Exception:  # pragma: no cover - runtime dependency gate
    _statsmodels_coint = None


MIN_SAMPLES = 50
MAX_STAT = 0.92
MIN_STAT = -18.86
STAR_STAT = -2.62
SMALL_P_COEFFS = (2.92, 1.5012, 0.039796)
LARGE_P_COEFFS = (2.1945, 0.64695, -0.29198, -0.042377)


@dataclass(slots=True)
class CointegrationResult:
    statistic: float
    pvalue: float
    method: str


def compute_cointegration_result(a: Iterable[float], b: Iterable[float]) -> CointegrationResult | None:
    series_a = _coerce_series(a)
    series_b = _coerce_series(b)
    if len(series_a) < MIN_SAMPLES or len(series_b) < MIN_SAMPLES or len(series_a) != len(series_b):
        return None

    if _statsmodels_coint is not None:
        try:
            statistic, pvalue, _ = _statsmodels_coint(series_a, series_b)
            statistic = float(statistic)
            pvalue = float(pvalue)
            if math.isfinite(statistic) and math.isfinite(pvalue):
                return CointegrationResult(statistic=statistic, pvalue=_clamp_probability(pvalue), method="statsmodels")
        except Exception:
            pass

    return _fallback_cointegration(series_a, series_b)


def _coerce_series(values: Iterable[float]) -> list[float]:
    series: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            series.append(numeric)
    return series


def _fallback_cointegration(a: list[float], b: list[float]) -> CointegrationResult | None:
    if len(a) != len(b) or len(a) < MIN_SAMPLES:
        return None
    resid = _ols_residuals(y=a, x=b)
    if resid is None:
        return CointegrationResult(statistic=float("-inf"), pvalue=0.0, method="fallback-perfect-collinearity")
    statistic = _adf_tstat_no_lag(resid)
    if statistic is None or not math.isfinite(statistic):
        return None
    pvalue = _mackinnon_pvalue_n2_c(statistic)
    return CointegrationResult(statistic=statistic, pvalue=_clamp_probability(pvalue), method="fallback-eg-adf0")


def _ols_residuals(*, y: list[float], x: list[float]) -> list[float] | None:
    nobs = len(y)
    mean_x = math.fsum(x) / nobs
    mean_y = math.fsum(y) / nobs
    var_x = math.fsum((value - mean_x) ** 2 for value in x)
    if var_x <= 1e-12:
        return None
    cov_xy = math.fsum((x[idx] - mean_x) * (y[idx] - mean_y) for idx in range(nobs))
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    resid = [y[idx] - (alpha + beta * x[idx]) for idx in range(nobs)]
    if max(abs(value) for value in resid) <= 1e-12:
        return None
    return resid


def _adf_tstat_no_lag(resid: list[float]) -> float | None:
    if len(resid) < MIN_SAMPLES:
        return None
    lagged = resid[:-1]
    delta = [resid[idx] - resid[idx - 1] for idx in range(1, len(resid))]
    denom = math.fsum(value * value for value in lagged)
    if denom <= 1e-12:
        return None
    gamma = math.fsum(lagged[idx] * delta[idx] for idx in range(len(delta))) / denom
    errors = [delta[idx] - gamma * lagged[idx] for idx in range(len(delta))]
    dof = len(delta) - 1
    if dof <= 0:
        return None
    sse = math.fsum(err * err for err in errors)
    sigma2 = sse / dof
    if sigma2 <= 0:
        return None
    se_gamma = math.sqrt(sigma2 / denom)
    if se_gamma <= 0:
        return None
    return gamma / se_gamma


def _mackinnon_pvalue_n2_c(teststat: float) -> float:
    if teststat > MAX_STAT:
        return 1.0
    if teststat < MIN_STAT:
        return 0.0
    coeffs = SMALL_P_COEFFS if teststat <= STAR_STAT else LARGE_P_COEFFS
    z = _polyval_ascending(coeffs, teststat)
    return _norm_cdf(z)


def _polyval_ascending(coeffs: tuple[float, ...], x: float) -> float:
    total = 0.0
    power = 1.0
    for coeff in coeffs:
        total += coeff * power
        power *= x
    return total


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))
