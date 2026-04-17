#!/usr/bin/env python3
"""
Run a first crypto pairs discovery sweep on Binance spot 1h klines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.crypto_pairs.historical import load_binance_spot_klines
from engine.crypto_pairs.stats import compute_halflife, compute_hurst

DEFAULT_PYTHON = Path("/Users/ahmedelmorshedy/.local/bin/oracle-btc-python")
DEFAULT_RAW_ROOT = Path("output/crypto_pairs/raw/spot_klines_1h")
DEFAULT_PROJECT_ROOT = Path("research/crypto_pairs/projects/crypto-pairs-v1")
DEFAULT_DIARY_PATH = Path("research/crypto_pairs/diary.md")
DEFAULT_INTERVAL = "1h"
DEFAULT_MIN_CORRELATION = 0.80
DEFAULT_MAX_COINT_PVALUE = 0.05
DEFAULT_MIN_HALFLIFE_HOURS = 1.0
DEFAULT_MAX_HALFLIFE_HOURS = 72.0
DEFAULT_SYMBOLS = [
    "ETHUSDT", "BTCUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT",
    "DOTUSDT", "ATOMUSDT", "ADAUSDT", "MATICUSDT", "SUIUSDT",
    "UNIUSDT", "AAVEUSDT", "LINKUSDT", "MKRUSDT",
    "ARBUSDT", "OPUSDT", "DOGEUSDT", "SHIBUSDT",
]
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crypto pairs discovery on Binance spot 1h klines.")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--diary-path", default=str(DEFAULT_DIARY_PATH))
    parser.add_argument("--min-correlation", type=float, default=DEFAULT_MIN_CORRELATION)
    parser.add_argument("--max-coint-pvalue", type=float, default=DEFAULT_MAX_COINT_PVALUE)
    parser.add_argument("--min-halflife-hours", type=float, default=DEFAULT_MIN_HALFLIFE_HOURS)
    parser.add_argument("--max-halflife-hours", type=float, default=DEFAULT_MAX_HALFLIFE_HOURS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = datetime.now(UTC).date() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.lookback_days - 1)
    if end_date < start_date:
        raise SystemExit("end-date must be on or after start-date")

    repo_root = Path(__file__).resolve().parent.parent
    python_exe = resolve_path(repo_root, args.python)
    raw_root = resolve_path(repo_root, args.raw_root)
    project_root = resolve_path(repo_root, args.project_root)
    diary_path = resolve_path(repo_root, args.diary_path)
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root.parent.parent / "checkpoints").mkdir(parents=True, exist_ok=True)
    if not diary_path.exists():
        diary_path.parent.mkdir(parents=True, exist_ok=True)
        diary_path.write_text("# Crypto Pairs Research Diary\n", encoding="utf-8")
    checkpoint_index = project_root.parent.parent / "checkpoints" / "index.json"
    if not checkpoint_index.exists():
        checkpoint_index.write_text("[]\n", encoding="utf-8")

    download_report = run_json(
        [
            str(python_exe),
            str(repo_root / "tools" / "download_crypto_pairs_klines.py"),
            "--start-date",
            start_date.isoformat(),
            "--end-date",
            end_date.isoformat(),
            "--interval",
            args.interval,
            "--output-root",
            str(raw_root),
            *sum([["--symbol", s] for s in (args.symbol or DEFAULT_SYMBOLS)], []),
        ],
        cwd=repo_root,
    )

    symbols = [symbol.upper() for symbol in (args.symbol or DEFAULT_SYMBOLS)]
    price_data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = load_binance_spot_klines(
            raw_root=raw_root,
            symbol=symbol,
            interval=args.interval,
            start_date=start_date,
            end_date=end_date,
        )
        if frame is not None and len(frame) >= 200:
            price_data[symbol] = frame

    results = []
    for symbol_a, symbol_b in combinations(sorted(price_data.keys()), 2):
        result = evaluate_pair(
            prices_a=price_data[symbol_a],
            prices_b=price_data[symbol_b],
            name_a=symbol_a.replace("USDT", ""),
            name_b=symbol_b.replace("USDT", ""),
            min_correlation=args.min_correlation,
            max_coint_pvalue=args.max_coint_pvalue,
            min_halflife_hours=args.min_halflife_hours,
            max_halflife_hours=args.max_halflife_hours,
        )
        results.append(result)

    tradeable = sorted((row for row in results if row["tradeable"]), key=lambda row: row["score"])
    rejected = [row for row in results if not row["tradeable"]]

    run_name = f"crypto_pairs_discovery_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_v1"
    run_root = project_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    report = {
        "run_name": run_name,
        "started_at": datetime.now(UTC).isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "interval": args.interval,
        "symbols_requested": symbols,
        "symbols_loaded": sorted(price_data.keys()),
        "download_report": download_report,
        "tradeable_pairs": tradeable,
        "rejected_pairs": rejected,
        "total_pairs_tested": len(results),
        "tradeable_pair_count": len(tradeable),
    }
    report_path = run_root / "pair_discovery_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    checkpoint_id = f"crypto-pairs-discovery-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    subprocess.check_call(
        [
            str(python_exe),
            str(repo_root / "tools" / "freeze_crypto_pairs_checkpoint.py"),
            "--checkpoint-id",
            checkpoint_id,
            "--status",
            "initial_discovery",
            "--category",
            "pair_discovery",
            "--summary",
            "Initial crypto pairs discovery from Binance spot 1h klines over a 30-day lookback.",
            "--artifact",
            f"project_root={project_root}",
            "--artifact",
            f"report_json={report_path}",
            "--metric",
            f"symbols_loaded={len(price_data)}",
            "--metric",
            f"pairs_tested={len(results)}",
            "--metric",
            f"tradeable_pairs={len(tradeable)}",
        ],
        cwd=str(repo_root),
    )

    write_plan(
        project_root=project_root,
        report_path=report_path,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        checkpoint_id=checkpoint_id,
        tradeable=tradeable,
    )
    append_diary(
        diary_path=diary_path,
        checkpoint_id=checkpoint_id,
        report_path=report_path,
        start_date=start_date,
        end_date=end_date,
        symbols_loaded=len(price_data),
        pairs_tested=len(results),
        tradeable=tradeable,
    )
    print(json.dumps({"checkpoint_id": checkpoint_id, "report_json": str(report_path)}, indent=2))


def run_json(command: list[str], *, cwd: Path) -> dict[str, object]:
    output = subprocess.check_output(command, cwd=str(cwd), text=True)
    return json.loads(output)


def resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def evaluate_pair(
    *,
    prices_a: pd.DataFrame,
    prices_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    min_correlation: float,
    max_coint_pvalue: float,
    min_halflife_hours: float,
    max_halflife_hours: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "pair": f"{name_a}/{name_b}",
        "token_a": name_a,
        "token_b": name_b,
        "tradeable": False,
    }
    merged = pd.merge(
        prices_a[["close"]].rename(columns={"close": "a"}),
        prices_b[["close"]].rename(columns={"close": "b"}),
        left_index=True,
        right_index=True,
        how="inner",
    )
    if len(merged) < 200:
        result["reject_reason"] = "insufficient_data"
        return result

    corr = float(merged["a"].corr(merged["b"]))
    result["correlation"] = round(corr, 4)
    if not np.isfinite(corr) or corr < min_correlation:
        result["reject_reason"] = "low_correlation"
        return result

    score, pvalue, _ = coint(merged["a"], merged["b"])
    result["coint_score"] = round(float(score), 6)
    result["coint_pvalue"] = round(float(pvalue), 6)
    if not np.isfinite(pvalue) or pvalue > max_coint_pvalue:
        result["reject_reason"] = "not_cointegrated"
        return result

    spread = np.log(merged["a"] / merged["b"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(spread) < 200:
        result["reject_reason"] = "bad_spread"
        return result

    halflife = compute_halflife(spread)
    result["halflife_hours"] = round(float(halflife), 2) if np.isfinite(halflife) else None
    if not np.isfinite(halflife) or halflife < min_halflife_hours or halflife > max_halflife_hours:
        result["reject_reason"] = "bad_halflife"
        return result

    hurst = compute_hurst(spread)
    result["hurst"] = round(float(hurst), 4)
    if not np.isfinite(hurst) or hurst >= 0.5:
        result["reject_reason"] = "not_mean_reverting"
        return result

    adf_stat, adf_pvalue, *_ = adfuller(spread.to_numpy())
    result["adf_stat"] = round(float(adf_stat), 6)
    result["adf_pvalue"] = round(float(adf_pvalue), 6)
    if not np.isfinite(adf_pvalue) or adf_pvalue > 0.05:
        result["reject_reason"] = "spread_not_stationary"
        return result

    result["spread_mean"] = round(float(spread.mean()), 6)
    result["spread_std"] = round(float(spread.std()), 6)
    result["spread_z_range"] = round(float((spread.max() - spread.min()) / spread.std()), 2) if float(spread.std()) > 0 else 0.0
    result["tradeable"] = True
    result["score"] = round(
        pvalue * 100 + halflife / 24 + hurst * 10 + (1 - corr) * 10,
        4,
    )
    return result


def write_plan(
    *,
    project_root: Path,
    report_path: Path,
    start_date,
    end_date,
    symbols: list[str],
    checkpoint_id: str,
    tradeable: list[dict[str, object]],
) -> None:
    preview = "\n".join(f"- `{row['pair']}` score `{row['score']}` halflife `{row['halflife_hours']}h`" for row in tradeable[:10]) or "- none"
    text = f"""# Crypto Pairs v1

Initial pair discovery lane based on Binance spot 1h klines.

- Date range: `{start_date}` to `{end_date}`
- Symbols: `{', '.join(symbols)}`
- Report: `{report_path}`
- Checkpoint: `{checkpoint_id}`

Top tradeable pairs:
{preview}
"""
    (project_root / "plan.md").write_text(text, encoding="utf-8")


def append_diary(
    *,
    diary_path: Path,
    checkpoint_id: str,
    report_path: Path,
    start_date,
    end_date,
    symbols_loaded: int,
    pairs_tested: int,
    tradeable: list[dict[str, object]],
) -> None:
    top = "\n".join(f"  - `{row['pair']}` score `{row['score']}` halflife `{row['halflife_hours']}h`" for row in tradeable[:5]) or "  - none"
    entry = f"""

## {datetime.now(UTC).date().isoformat()} - Initial pair discovery

- Checkpoint: `{checkpoint_id}`
- Date range: `{start_date}` to `{end_date}`
- Symbols loaded: `{symbols_loaded}`
- Pairs tested: `{pairs_tested}`
- Tradeable pairs: `{len(tradeable)}`
- Report: `{report_path}`
- Top pairs:
{top}
"""
    with diary_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


if __name__ == "__main__":
    main()
