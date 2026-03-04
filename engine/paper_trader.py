"""
Paper Trading Engine
====================
Simulates trade execution and tracks portfolio P&L.
Every decision is logged with full context for later analysis.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from data.models import (
    Signal, PaperTrade, Position, Portfolio, Side, TradeStatus, SignalAction
)

logger = logging.getLogger(__name__)

BASE_FEE_RATE = 0.02


class PaperTrader:
    """Simulates trade execution in paper mode."""

    def __init__(self, starting_capital: float = 1000.0, log_dir: str = "logs"):
        self.portfolio = Portfolio(starting_capital=starting_capital, cash=starting_capital)
        self.trade_log: list[PaperTrade] = []
        self.signal_log: list[Signal] = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def execute_signal(self, signal: Signal, current_prices: dict[str, float]) -> PaperTrade | None:
        """
        Execute a signal in paper mode.
        Returns the simulated trade or None if rejected by risk checks.
        """
        self.signal_log.append(signal)

        # Risk checks
        if not self._passes_risk_checks(signal):
            logger.info(f"[PAPER] Signal {signal.id} rejected by risk checks")
            self._log_signal(signal, "REJECTED_RISK")
            return None

        # Route to appropriate execution method
        if signal.action == SignalAction.HEDGE_BOTH:
            return self._execute_hedge(signal, current_prices)
        elif signal.action == SignalAction.ARB_ALL:
            return self._execute_arb(signal, current_prices)
        elif signal.action in (SignalAction.BUY_YES, SignalAction.BUY_NO):
            return self._execute_directional(signal, current_prices)
        else:
            logger.warning(f"[PAPER] Unknown action: {signal.action}")
            return None

    def _execute_directional(
        self, signal: Signal, current_prices: dict[str, float]
    ) -> PaperTrade | None:
        """Execute a directional buy (YES or NO)."""
        if not signal.token_id:
            return None

        price = current_prices.get(signal.token_id, 0)
        if price <= 0:
            return None

        # Calculate size
        # Kelly-inspired sizing: size = (edge / odds) * bankroll, capped
        if signal.expected_edge > 0 and signal.confidence > 0:
            odds = (1.0 / max(signal.confidence, 0.1)) - 1.0  # implied odds
            edge = signal.expected_edge / 100  # convert cents to dollars
            kelly_raw = (signal.confidence * odds - (1 - signal.confidence)) / max(odds, 0.01)
            kelly_fraction = max(0, min(kelly_raw * 0.25, 0.20))  # quarter-Kelly, cap 20%
            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * kelly_fraction)
        else:
            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * 0.05)  # minimum sizing
        if size_usd <= 0:
            return None

        # Apply slippage (0.5%)
        fill_price = price * 1.005
        shares = size_usd / fill_price
        fee = BASE_FEE_RATE * min(fill_price, 1 - fill_price) * shares

        # Simulate fill
        price = fill_price
        total_cost = size_usd + fee
        if total_cost > self.portfolio.cash:
            size_usd = (self.portfolio.cash - fee) * 0.95
            shares = size_usd / price
            fee = BASE_FEE_RATE * min(price, 1 - price) * shares
            total_cost = size_usd + fee

        self.portfolio.cash -= total_cost
        self.portfolio.total_fees_paid += fee
        self.portfolio.total_trades += 1

        # Add position
        position = Position(
            token_id=signal.token_id,
            condition_id=signal.condition_id,
            market_slug=signal.market_slug,
            side="YES" if signal.action == SignalAction.BUY_YES else "NO",
            shares=shares,
            avg_entry_price=price,
            current_price=price,
            source=signal.source,
        )
        self.portfolio.positions.append(position)

        trade = PaperTrade(
            signal_id=signal.id,
            source=signal.source,
            market_slug=signal.market_slug,
            condition_id=signal.condition_id,
            token_id=signal.token_id,
            side=Side.BUY,
            price=price,
            size_shares=shares,
            size_usd=size_usd,
            status=TradeStatus.FILLED,
        )
        self.trade_log.append(trade)
        self._log_trade(trade, signal)

        logger.info(
            f"[PAPER] Executed: {signal.action.value} on {signal.market_slug} | "
            f"{shares:.1f} shares @ ${price:.3f} = ${size_usd:.2f} (fee: ${fee:.3f})"
        )
        return trade

    def _execute_hedge(
        self, signal: Signal, current_prices: dict[str, float]
    ) -> PaperTrade | None:
        """Execute a hedged liquidity provision order (both sides)."""
        # Simplified: track as a single hedge position
        trade = PaperTrade(
            signal_id=signal.id,
            source=signal.source,
            market_slug=signal.market_slug,
            condition_id=signal.condition_id,
            token_id=signal.token_id or "",
            side=Side.BUY,
            price=0.0,
            size_shares=0,
            size_usd=signal.suggested_size_usd,
            status=TradeStatus.FILLED,
        )
        self.trade_log.append(trade)
        self._log_trade(trade, signal)
        logger.info(f"[PAPER] Hedge position on {signal.market_slug}: ${signal.suggested_size_usd:.2f}")
        return trade

    def _execute_arb(
        self, signal: Signal, current_prices: dict[str, float]
    ) -> PaperTrade | None:
        """Execute a multi-outcome arb (buy all outcomes)."""
        if signal.arb_total_cost <= 0:
            return None

        # Kelly-inspired sizing: size = (edge / odds) * bankroll, capped
        if signal.expected_edge > 0 and signal.confidence > 0:
            odds = (1.0 / max(signal.confidence, 0.1)) - 1.0  # implied odds
            edge = signal.expected_edge / 100  # convert cents to dollars
            kelly_raw = (signal.confidence * odds - (1 - signal.confidence)) / max(odds, 0.01)
            kelly_fraction = max(0, min(kelly_raw * 0.25, 0.20))  # quarter-Kelly, cap 20%
            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * kelly_fraction)
        else:
            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * 0.05)  # minimum sizing
        if size_usd <= 0:
            return None

        # The arb locks in guaranteed profit
        units = size_usd / signal.arb_total_cost
        guaranteed_profit = units * (signal.arb_guaranteed_payout - signal.arb_total_cost)
        fee_estimate = size_usd * BASE_FEE_RATE * 0.5  # rough fee estimate

        net_profit = guaranteed_profit - fee_estimate
        if net_profit <= 0:
            return None

        self.portfolio.cash -= size_usd
        self.portfolio.total_fees_paid += fee_estimate
        self.portfolio.total_trades += 1
        # Track as open position — profit only realized at resolution
        position = Position(
            token_id="ARB_ALL",
            condition_id=signal.condition_id,
            market_slug=signal.market_slug,
            side="ARB",
            shares=units,
            avg_entry_price=signal.arb_total_cost,
            current_price=signal.arb_guaranteed_payout,
            unrealized_pnl=net_profit,
            source=signal.source,
        )
        self.portfolio.positions.append(position)

        trade = PaperTrade(
            signal_id=signal.id,
            source=signal.source,
            market_slug=signal.market_slug,
            condition_id=signal.condition_id,
            token_id="ARB_ALL",
            side=Side.BUY,
            price=signal.arb_total_cost,
            size_shares=units,
            size_usd=size_usd,
            status=TradeStatus.FILLED,
            realized_pnl=net_profit,
        )
        self.trade_log.append(trade)
        self._log_trade(trade, signal)

        logger.info(
            f"[PAPER] Arb executed: {signal.market_slug} | "
            f"Cost: ${size_usd:.2f} | Net profit: ${net_profit:.3f}"
        )
        return trade

    def update_positions(self, current_prices: dict[str, float]):
        """Update position mark-to-market and portfolio stats."""
        for pos in self.portfolio.positions:
            new_price = current_prices.get(pos.token_id, pos.current_price)
            pos.current_price = new_price
            pos.unrealized_pnl = (new_price - pos.avg_entry_price) * pos.shares

        # Update peak and drawdown
        total_val = self.portfolio.total_value
        if total_val > self.portfolio.peak_value:
            self.portfolio.peak_value = total_val
        if self.portfolio.peak_value > 0:
            dd = (self.portfolio.peak_value - total_val) / self.portfolio.peak_value
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, dd)

    def _update_drawdown(self):
        """Recalculate drawdown properly."""
        current = self.portfolio.cash + sum(
            p.shares * p.current_price for p in self.portfolio.positions
        )
        if current >= self.portfolio.starting_capital:
            self.portfolio.max_drawdown = 0.0
        elif self.portfolio.starting_capital > 0:
            dd = (self.portfolio.starting_capital - current) / self.portfolio.starting_capital
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, dd)

    def _passes_risk_checks(self, signal: Signal) -> bool:
        """Check if a signal passes risk management rules."""
        self._update_drawdown()
        # Check max total exposure
        if self.portfolio.positions_value >= self.portfolio.starting_capital * 2:
            logger.info("[RISK] Max exposure reached")
            return False
        # Check max drawdown
        if self.portfolio.max_drawdown > 0.30:
            logger.warning("[RISK] Max drawdown exceeded — pausing trading")
            return False
        # Check cash available
        if self.portfolio.cash < signal.suggested_size_usd * 0.5:
            return False
        # No duplicate positions on same market
        existing = [p for p in self.portfolio.positions if p.condition_id == signal.condition_id]
        if existing:
            logger.info(f"[RISK] Already have position on {signal.market_slug}")
            return False
        # Max 3 positions per resolution week
        if signal.market_slug:
            same_source = [p for p in self.portfolio.positions if p.source == signal.source]
            if len(same_source) >= 5:
                logger.info(f"[RISK] Too many positions from {signal.source.value}")
                return False
        # Max 30% of capital in any single strategy
        strategy_exposure = sum(
            p.shares * p.current_price for p in self.portfolio.positions
            if p.source == signal.source
        )
        if strategy_exposure > self.portfolio.starting_capital * 0.30:
            logger.info(f"[RISK] Strategy {signal.source.value} at 30% cap")
            return False
        return True

    # ------------------------------------------------------------------
    # Logging (for tuning analysis)
    # ------------------------------------------------------------------

    def _log_signal(self, signal: Signal, status: str = "GENERATED"):
        """Log a signal with full context."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "signal": signal.model_dump(mode="json"),
        }
        self._append_jsonl("signals.jsonl", entry)

    def _log_trade(self, trade: PaperTrade, signal: Signal):
        """Log a trade with full context."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade": trade.model_dump(mode="json"),
            "signal": signal.model_dump(mode="json"),
            "portfolio_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
        }
        self._append_jsonl("trades.jsonl", entry)

    def _append_jsonl(self, filename: str, data: dict):
        """Append a JSON line to a log file."""
        try:
            filepath = self.log_dir / filename
            with open(filepath, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def get_performance_report(self) -> dict:
        """Generate a performance report for tuning analysis."""
        by_strategy = {}
        for trade in self.trade_log:
            src = trade.source.value
            if src not in by_strategy:
                by_strategy[src] = {
                    "trades": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0.0, "avg_pnl": 0.0
                }
            by_strategy[src]["trades"] += 1
            pnl = trade.realized_pnl or 0
            by_strategy[src]["total_pnl"] += pnl
            if pnl > 0:
                by_strategy[src]["wins"] += 1
            elif pnl < 0:
                by_strategy[src]["losses"] += 1

        for src, stats in by_strategy.items():
            if stats["trades"] > 0:
                stats["avg_pnl"] = stats["total_pnl"] / stats["trades"]
                stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0

        return {
            "total_value": self.portfolio.total_value,
            "total_pnl": self.portfolio.total_pnl,
            "total_pnl_pct": self.portfolio.total_pnl_pct,
            "total_trades": self.portfolio.total_trades,
            "win_rate": self.portfolio.win_rate,
            "max_drawdown": self.portfolio.max_drawdown,
            "total_fees": self.portfolio.total_fees_paid,
            "by_strategy": by_strategy,
        }
