"""
A/B Testing Engine
==================
Run two variants of a strategy simultaneously with different parameters.
Each variant gets its own paper portfolio. Compare performance side-by-side.

Usage:
    ab = ABTester("mean_reversion")
    ab.add_variant("conservative", {"drop_threshold_pct": 0.15, "exit_reversion_pct": 0.5})
    ab.add_variant("aggressive", {"drop_threshold_pct": 0.10, "exit_reversion_pct": 0.7})
    # Then call ab.record_signal() and ab.record_outcome() as trades play out
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)


class ABVariant:
    """One variant in an A/B test."""

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        self.signals: list[dict] = []
        self.trades: list[dict] = []
        self.wins: int = 0
        self.losses: int = 0
        self.total_pnl: float = 0.0
        self.total_trades: int = 0
        self.peak_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        self.started_at: str = datetime.now(timezone.utc).isoformat()

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    def record_signal(self, signal: dict):
        """Record a signal this variant would have generated."""
        self.signals.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **signal,
        })

    def record_trade(self, pnl: float, details: dict = None):
        """Record a completed trade outcome."""
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1

        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        if self.peak_pnl > 0:
            dd = (self.peak_pnl - self.total_pnl) / self.peak_pnl
            self.max_drawdown = max(self.max_drawdown, dd)

        self.trades.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pnl": pnl,
            "cumulative_pnl": self.total_pnl,
            **(details or {}),
        })

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate * 100, 1),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 3),
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "started_at": self.started_at,
            "recent_trades": self.trades[-10:],
        }


class ABTester:
    """Manages A/B tests for strategy comparison."""

    def __init__(self, log_dir: str = "logs"):
        self.tests: dict[str, dict] = {}  # test_name -> {"a": ABVariant, "b": ABVariant}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def create_test(
        self, test_name: str,
        variant_a_name: str, variant_a_params: dict,
        variant_b_name: str, variant_b_params: dict,
    ):
        """Create a new A/B test with two variants."""
        self.tests[test_name] = {
            "a": ABVariant(variant_a_name, variant_a_params),
            "b": ABVariant(variant_b_name, variant_b_params),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "strategy": test_name,
        }
        logger.info(
            f"[A/B] Created test '{test_name}': "
            f"{variant_a_name} vs {variant_b_name}"
        )

    def evaluate_signal(
        self, test_name: str, market_data: dict,
        evaluate_fn_a, evaluate_fn_b
    ) -> dict:
        """
        Run both variants' logic on the same market data.
        evaluate_fn_a/b take market_data and return (should_trade: bool, signal: dict)
        """
        if test_name not in self.tests:
            return {}

        test = self.tests[test_name]

        result_a = evaluate_fn_a(market_data)
        result_b = evaluate_fn_b(market_data)

        if result_a[0]:
            test["a"].record_signal(result_a[1])
        if result_b[0]:
            test["b"].record_signal(result_b[1])

        return {
            "a_triggered": result_a[0],
            "b_triggered": result_b[0],
        }

    def record_outcome(self, test_name: str, variant: str, pnl: float, details: dict = None):
        """Record an outcome for a specific variant."""
        if test_name not in self.tests:
            return
        if variant in self.tests[test_name]:
            self.tests[test_name][variant].record_trade(pnl, details)
            self._log_outcome(test_name, variant, pnl, details)

    def get_report(self) -> dict:
        """Get full A/B test report for all tests."""
        report = {}
        for test_name, test in self.tests.items():
            a = test["a"]
            b = test["b"]

            # Determine leader
            if a.total_trades > 0 and b.total_trades > 0:
                if a.total_pnl > b.total_pnl:
                    leader = a.name
                elif b.total_pnl > a.total_pnl:
                    leader = b.name
                else:
                    leader = "tied"
            else:
                leader = "insufficient data"

            report[test_name] = {
                "strategy": test.get("strategy", test_name),
                "created_at": test.get("created_at"),
                "leader": leader,
                "variant_a": a.to_dict(),
                "variant_b": b.to_dict(),
            }
        return report

    def _log_outcome(self, test_name: str, variant: str, pnl: float, details: dict = None):
        """Log A/B outcome to file."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test": test_name,
                "variant": variant,
                "pnl": pnl,
                **(details or {}),
            }
            with open(self.log_dir / "ab_tests.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log A/B outcome: {e}")
