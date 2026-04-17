"""Paper execution engine for simultaneous crypto pairs leg fills."""

from __future__ import annotations

import time
from typing import Callable

from .config import ExecutionConfig
from .signal_engine_v1 import Signal


class ExecutionEngine:
    def __init__(self, quote_lookup: Callable[[str], float | None], config: ExecutionConfig | None = None):
        self.quote_lookup = quote_lookup
        self.config = config or ExecutionConfig()
        self.trade_log: list[dict[str, object]] = []

    def execute_entry(
        self,
        *,
        pair_key: str,
        direction: Signal,
        token_a: str,
        token_b: str,
        capital_per_leg_usdt: float,
    ) -> dict[str, object]:
        if direction == Signal.LONG_A_SHORT_B:
            side_a = "BUY"
            side_b = "SELL"
        elif direction == Signal.SHORT_A_LONG_B:
            side_a = "SELL"
            side_b = "BUY"
        else:
            raise ValueError(f"Unsupported entry direction: {direction}")

        price_a = self._require_quote(token_a)
        price_b = self._require_quote(token_b)
        qty_a = round(capital_per_leg_usdt / price_a, self.config.quantity_precision)
        qty_b = round(capital_per_leg_usdt / price_b, self.config.quantity_precision)
        fill_a = self._apply_slippage(price_a, side_a)
        fill_b = self._apply_slippage(price_b, side_b)
        slippage_usd = abs(fill_a - price_a) * qty_a + abs(fill_b - price_b) * qty_b

        trade = {
            "pair": pair_key,
            "direction": direction.value,
            "timestamp_ms": int(time.time() * 1000),
            "token_a": token_a,
            "token_b": token_b,
            "side_a": side_a,
            "side_b": side_b,
            "qty_a": qty_a,
            "qty_b": qty_b,
            "raw_price_a": price_a,
            "raw_price_b": price_b,
            "fill_a": fill_a,
            "fill_b": fill_b,
            "capital_per_leg": capital_per_leg_usdt,
            "fee_bps": self.config.fee_bps,
            "slippage_bps": self.config.slippage_bps,
            "slippage_usd": round(slippage_usd, 6),
            "paper": self.config.paper_trade,
        }
        self.trade_log.append(trade)
        return trade

    def execute_exit(self, entry_trade: dict[str, object]) -> dict[str, object]:
        exit_side_a = "SELL" if entry_trade["side_a"] == "BUY" else "BUY"
        exit_side_b = "SELL" if entry_trade["side_b"] == "BUY" else "BUY"
        price_a = self._require_quote(str(entry_trade["token_a"]))
        price_b = self._require_quote(str(entry_trade["token_b"]))
        fill_a = self._apply_slippage(price_a, exit_side_a)
        fill_b = self._apply_slippage(price_b, exit_side_b)

        qty_a = float(entry_trade["qty_a"])
        qty_b = float(entry_trade["qty_b"])
        capital_per_leg = float(entry_trade["capital_per_leg"])
        if str(entry_trade["side_a"]) == "BUY":
            pnl_a = (fill_a - float(entry_trade["fill_a"])) * qty_a
            pnl_b = (float(entry_trade["fill_b"]) - fill_b) * qty_b
        else:
            pnl_a = (float(entry_trade["fill_a"]) - fill_a) * qty_a
            pnl_b = (fill_b - float(entry_trade["fill_b"])) * qty_b
        gross_pnl = pnl_a + pnl_b
        total_fee = capital_per_leg * 4 * self.config.fee_bps / 10_000
        entry_slippage_usd = float(entry_trade.get("slippage_usd") or 0.0)
        exit_slippage_usd = abs(fill_a - price_a) * qty_a + abs(fill_b - price_b) * qty_b
        total_slippage_usd = entry_slippage_usd + exit_slippage_usd
        total_pnl = gross_pnl - total_fee
        gross_bps = gross_pnl / (capital_per_leg * 2) * 10_000 if capital_per_leg > 0 else 0.0
        total_bps = total_pnl / (capital_per_leg * 2) * 10_000 if capital_per_leg > 0 else 0.0

        trade = {
            "pair": entry_trade["pair"],
            "timestamp_ms": int(time.time() * 1000),
            "token_a": entry_trade["token_a"],
            "token_b": entry_trade["token_b"],
            "side_a": exit_side_a,
            "side_b": exit_side_b,
            "raw_price_a": price_a,
            "raw_price_b": price_b,
            "fill_a": fill_a,
            "fill_b": fill_b,
            "gross_pnl_usd": round(gross_pnl, 6),
            "gross_pnl_bps": round(gross_bps, 4),
            "pnl_usd": round(total_pnl, 6),
            "pnl_bps": round(total_bps, 4),
            "pnl_a": round(pnl_a, 6),
            "pnl_b": round(pnl_b, 6),
            "fees_usd": round(total_fee, 6),
            "slippage_usd": round(total_slippage_usd, 6),
            "slippage_bps": round(total_slippage_usd / (capital_per_leg * 2) * 10_000, 4) if capital_per_leg > 0 else 0.0,
            "paper": self.config.paper_trade,
        }
        self.trade_log.append(trade)
        return trade

    def _apply_slippage(self, price: float, side: str) -> float:
        direction = 1 if side == "BUY" else -1
        return price * (1 + direction * self.config.slippage_bps / 10_000)

    def _require_quote(self, symbol: str) -> float:
        quote = self.quote_lookup(symbol)
        if quote is None or quote <= 0:
            raise ValueError(f"Missing live quote for {symbol}")
        return float(quote)
