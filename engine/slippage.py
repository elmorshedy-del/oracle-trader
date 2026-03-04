"""
Self-Calibrating Slippage Model
===============================
Tracks predicted vs actual execution quality and adjusts
the slippage parameter k over time.

Formula: slippage = k * sqrt(size / liquidity)

k starts at 0.1 (conservative default) and self-calibrates
as real execution data comes in.
"""

import json
import math
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class SlippageModel:
    """
    Self-calibrating slippage estimator.

    slippage = k * sqrt(size / liquidity)

    k is updated via exponential moving average of observed errors.
    Each observation compares predicted fill price to simulated/actual fill price.
    """

    def __init__(self, initial_k: float = 0.1, log_dir: str = "logs"):
        self.k = initial_k
        self.min_k = 0.01
        self.max_k = 0.5
        self.ema_error: float = 0.0
        self.ema_alpha: float = 0.05  # smoothing factor — slow adaptation
        self.calibration_count: int = 0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.observations: list[dict] = []

        # Load saved state if exists
        self._load_state()

    def estimate_slippage(self, size_usd: float, liquidity: float) -> float:
        """Estimate slippage for a given trade size and market liquidity."""
        if liquidity <= 0:
            return 0.05  # 5% default for unknown liquidity
        return self.k * math.sqrt(size_usd / liquidity)

    def estimate_fill_price(
        self, mid_price: float, side: str, size_usd: float,
        liquidity: float, spread: float = 0.0
    ) -> float:
        """
        Estimate realistic fill price including spread + slippage.

        For BUY: fill = mid + spread/2 + slippage
        For SELL: fill = mid - spread/2 - slippage
        """
        half_spread = spread / 2
        slip = self.estimate_slippage(size_usd, liquidity)

        if side.upper() == "BUY":
            return min(mid_price + half_spread + slip, 0.99)
        else:
            return max(mid_price - half_spread - slip, 0.01)

    def observe(
        self,
        market_slug: str,
        mid_price: float,
        predicted_fill: float,
        simulated_fill: float,
        size_usd: float,
        liquidity: float,
        spread: float,
    ):
        """
        Record an observation and update k.

        In paper mode: simulated_fill is estimated from order book depth.
        In live mode: simulated_fill is the actual fill price.
        """
        error = abs(simulated_fill - predicted_fill)
        direction = 1.0 if simulated_fill > predicted_fill else -1.0

        # Update EMA of signed error
        signed_error = (simulated_fill - predicted_fill)
        self.ema_error = (
            self.ema_alpha * signed_error +
            (1 - self.ema_alpha) * self.ema_error
        )

        # Adjust k based on error direction
        # Underestimating slippage (fills worse than predicted) → increase k
        # Overestimating (fills better) → decrease k
        if liquidity > 0 and size_usd > 0:
            observed_k = abs(simulated_fill - mid_price) / math.sqrt(size_usd / liquidity)
            # Blend toward observed k slowly
            self.k = (1 - self.ema_alpha) * self.k + self.ema_alpha * observed_k
            self.k = max(self.min_k, min(self.max_k, self.k))

        self.calibration_count += 1

        obs = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market": market_slug,
            "mid": round(mid_price, 4),
            "predicted": round(predicted_fill, 4),
            "simulated": round(simulated_fill, 4),
            "error": round(error, 5),
            "size_usd": round(size_usd, 2),
            "liquidity": round(liquidity, 0),
            "spread": round(spread, 4),
            "k_after": round(self.k, 5),
        }
        self.observations.append(obs)

        # Log observation
        try:
            with open(self.log_dir / "slippage.jsonl", "a") as f:
                f.write(json.dumps(obs) + "\n")
        except Exception as e:
            logger.error(f"Failed to log slippage obs: {e}")

        # Periodically save state
        if self.calibration_count % 50 == 0:
            self._save_state()
            logger.info(
                f"[SLIPPAGE] Calibrated: k={self.k:.4f} | "
                f"EMA error={self.ema_error:.5f} | "
                f"observations={self.calibration_count}"
            )

    def simulate_fill_from_book(
        self, book: dict, side: str, size_usd: float, mid_price: float
    ) -> float:
        """
        Walk the order book to simulate a realistic fill price.
        Used in paper mode to generate observations for calibration.
        """
        if side.upper() == "BUY":
            levels = book.get("asks", [])
        else:
            levels = book.get("bids", [])

        if not levels:
            # No book data — use mid + spread estimate
            return mid_price * (1.005 if side.upper() == "BUY" else 0.995)

        remaining = size_usd
        total_cost = 0.0
        total_shares = 0.0

        for level in levels:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
            if price <= 0 or size <= 0:
                continue

            level_value = price * size
            if level_value >= remaining:
                shares_here = remaining / price
                total_cost += remaining
                total_shares += shares_here
                remaining = 0
                break
            else:
                total_cost += level_value
                total_shares += size
                remaining -= level_value

        if total_shares <= 0:
            return mid_price * (1.01 if side.upper() == "BUY" else 0.99)

        return total_cost / total_shares

    def get_stats(self) -> dict:
        """Get calibration stats for dashboard."""
        return {
            "k": round(self.k, 5),
            "ema_error": round(self.ema_error, 5),
            "calibration_count": self.calibration_count,
            "recent_observations": self.observations[-10:],
        }

    def _save_state(self):
        """Persist k and calibration state."""
        state = {
            "k": self.k,
            "ema_error": self.ema_error,
            "calibration_count": self.calibration_count,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self.log_dir / "slippage_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save slippage state: {e}")

    def _load_state(self):
        """Load saved calibration state."""
        state_file = self.log_dir / "slippage_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.k = state.get("k", self.k)
                self.ema_error = state.get("ema_error", 0.0)
                self.calibration_count = state.get("calibration_count", 0)
                logger.info(
                    f"[SLIPPAGE] Loaded state: k={self.k:.4f}, "
                    f"observations={self.calibration_count}"
                )
            except Exception as e:
                logger.error(f"Failed to load slippage state: {e}")
