"""Helpers to load and convert frozen crypto-pairs discovery output."""

from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_DISCOVERY_PROJECT_ROOT, DEFAULT_WARMUP_SECONDS, PairRuntimeConfig


DISCOVERY_FILE_NAME = "pair_discovery_results.json"
LOOKBACK_MULTIPLIER = 4
USDT_SUFFIX = "USDT"


def load_discovery_report(report_path: str | Path | None = None) -> dict[str, object]:
    path = resolve_discovery_report_path(report_path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_discovery_report_path(report_path: str | Path | None = None) -> Path:
    if report_path is not None:
        report_path_str = str(report_path).strip()
        if report_path_str:
            return Path(report_path_str).expanduser().resolve()
    project_root = Path(DEFAULT_DISCOVERY_PROJECT_ROOT).expanduser().resolve()
    candidates = sorted(project_root.glob(f"crypto_pairs_discovery_*_v1/{DISCOVERY_FILE_NAME}"))
    if not candidates:
        raise FileNotFoundError(f"No discovery reports found under {project_root}")
    return candidates[-1]


def build_runtime_configs(
    discovery_report: dict[str, object],
    *,
    top_pairs: int = 5,
    pair_keys: list[str] | None = None,
) -> tuple[list[str], list[PairRuntimeConfig], list[dict[str, object]]]:
    tradeable_pairs = list(discovery_report.get("tradeable_pairs", []))
    if pair_keys:
        requested = {pair_key.upper() for pair_key in pair_keys}
        active_pairs = [
            row for row in tradeable_pairs if f"{row['token_a']}/{row['token_b']}".upper() in requested
        ]
        if len(active_pairs) != len(requested):
            found = {f"{row['token_a']}/{row['token_b']}".upper() for row in active_pairs}
            missing = sorted(requested - found)
            raise ValueError(f"Requested pair(s) not found in discovery report: {', '.join(missing)}")
    else:
        active_pairs = tradeable_pairs[:top_pairs]
    if not active_pairs:
        raise ValueError("No tradeable pairs available in discovery report")
    symbols: set[str] = set()
    configs: list[PairRuntimeConfig] = []
    for row in active_pairs:
        token_a = normalize_symbol(str(row["token_a"]))
        token_b = normalize_symbol(str(row["token_b"]))
        symbols.update([token_a, token_b])
        halflife_hours = float(row["halflife_hours"])
        lookback_seconds = max(DEFAULT_WARMUP_SECONDS, int(round(halflife_hours * 3600 * LOOKBACK_MULTIPLIER)))
        configs.append(
            PairRuntimeConfig(
                pair_key=f"{row['token_a']}/{row['token_b']}",
                token_a=token_a,
                token_b=token_b,
                lookback_seconds=lookback_seconds,
                halflife_hours=halflife_hours,
                discovery_score=float(row["score"]),
                spread_mean=float(row["spread_mean"]),
                spread_std=float(row["spread_std"]),
            )
        )
    return sorted(symbols), configs, active_pairs


def normalize_symbol(token: str) -> str:
    token = token.upper()
    return token if token.endswith(USDT_SUFFIX) else f"{token}{USDT_SUFFIX}"
