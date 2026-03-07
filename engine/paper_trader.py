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
    Signal, PaperTrade, Position, Portfolio, Side, TradeStatus, SignalAction, SignalSource
)

logger = logging.getLogger(__name__)

BASE_FEE_RATE = 0.02
EXIT_FEE_ESTIMATE_RATE = BASE_FEE_RATE * 0.5
MIN_TRADE_CASH_USD = 2.0
MAX_HEDGE_POSITIONS = 10
MAX_DIRECTIONAL_POSITIONS = 10
MAX_NEW_HEDGE_POSITIONS_PER_SCAN = 8
MAX_NEW_ARB_POSITIONS_PER_SCAN = 2
MAX_NEW_DIRECTIONAL_POSITIONS_PER_SCAN = 4
DIRECTIONAL_TAKE_PROFIT_PCT = 0.20
DIRECTIONAL_STOP_LOSS_PCT = -0.25
TOKEN_CONVERGENCE_PRICE = 0.95
HEDGE_ROTATION_HOURS = 24.0
STRATEGY_EXPOSURE_CAPS = {
    SignalSource.LIQUIDITY: 0.25,
    SignalSource.ARBITRAGE: 0.20,
    SignalSource.NEWS: 0.15,
    SignalSource.MEAN_REVERSION: 0.20,
    SignalSource.CRYPTO_ARB: 0.20,
    SignalSource.WEATHER: 0.15,
    SignalSource.WHALE: 0.10,
}


class PaperTrader:
    """Simulates trade execution in paper mode."""

    def __init__(self, starting_capital: float = 1000.0, log_dir: str = "logs", state_path: str = "/data/state.json"):
        self.state_path = Path(state_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Try to restore state from disk, fall back to fresh start
        if not self._load_state():
            self.portfolio = Portfolio(starting_capital=starting_capital, cash=starting_capital, peak_value=starting_capital)
            self.trade_log: list[PaperTrade] = []
            self.signal_log: list[Signal] = []
            logger.info(f"[PAPER] Fresh start with ${starting_capital}")
        else:
            # Fix peak_value if it was initialized with wrong default
            if self.portfolio.peak_value > self.portfolio.total_value * 1.2:
                self.portfolio.peak_value = max(self.portfolio.starting_capital, self.portfolio.total_value)
                logger.info(f"[PAPER] Reset peak_value to ${self.portfolio.peak_value:.2f}")
            logger.info(f"[PAPER] Restored state: ${self.portfolio.total_value:.2f} | {len(self.trade_log)} trades")

    def _load_state(self) -> bool:
        """Load state from disk if available."""
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                self.portfolio = Portfolio(**data["portfolio"])
                # Reconstruct positions (they're nested in portfolio)
                self.trade_log = [PaperTrade(**t) for t in data.get("trade_log", [])]
                self.signal_log = []  # don't restore signals, they're transient
                return True
        except Exception as e:
            logger.warning(f"[PAPER] Failed to load state: {e}")
        return False

    def save_state(self):
        """Persist current state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "portfolio": self.portfolio.model_dump(),
                "trade_log": [t.model_dump() for t in self.trade_log[-200:]],  # keep last 200
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            # Write to temp file first, then rename (atomic)
            tmp = self.state_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, default=str))
            tmp.rename(self.state_path)
        except Exception as e:
            logger.error(f"[PAPER] Failed to save state: {e}")

    def reset(self, starting_capital: float | None = None):
        """Reset paper state without wiping persisted logs."""
        capital = starting_capital or self.portfolio.starting_capital
        self.portfolio = Portfolio(
            starting_capital=capital,
            cash=capital,
            peak_value=capital,
        )
        self.trade_log = []
        self.signal_log = []
        try:
            self.state_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"[PAPER] Failed to delete state file: {e}")
        self.save_state()
        logger.info(f"[PAPER] Reset portfolio state to ${capital:.2f}")

    def _strategy_exposure_cap_pct(
        self, source: SignalSource, action: SignalAction | None = None
    ) -> float:
        if action == SignalAction.HEDGE_BOTH:
            return STRATEGY_EXPOSURE_CAPS[SignalSource.LIQUIDITY]
        return STRATEGY_EXPOSURE_CAPS.get(source, 0.20)

    def _position_value(self, pos: Position) -> float:
        return pos.shares * pos.current_price

    def _strategy_exposure(self, source: SignalSource) -> float:
        return sum(
            self._position_value(pos)
            for pos in self.portfolio.positions
            if pos.source == source
        )

    def select_candidate_signals(self, signals: list[Signal]) -> tuple[list[Signal], dict[str, int]]:
        """Drop signals that are already impossible given current holdings and source caps."""
        selected: list[Signal] = []
        filtered = {
            "held_market": 0,
            "duplicate_market": 0,
            "source_scan_cap": 0,
            "source_position_cap": 0,
            "source_exposure_cap": 0,
        }
        held_conditions = {p.condition_id for p in self.portfolio.positions}
        seen_conditions: set[str] = set()
        source_counts: dict = {}
        source_exposure: dict = {}
        selected_per_source: dict = {}

        for pos in self.portfolio.positions:
            source_counts[pos.source] = source_counts.get(pos.source, 0) + 1
            source_exposure[pos.source] = (
                source_exposure.get(pos.source, 0.0) + (pos.shares * pos.current_price)
            )

        for signal in signals:
            if signal.condition_id in held_conditions:
                filtered["held_market"] += 1
                continue
            if signal.condition_id in seen_conditions:
                filtered["duplicate_market"] += 1
                continue

            per_scan_cap = MAX_NEW_DIRECTIONAL_POSITIONS_PER_SCAN
            if signal.action == SignalAction.HEDGE_BOTH:
                per_scan_cap = MAX_NEW_HEDGE_POSITIONS_PER_SCAN
            elif signal.action == SignalAction.ARB_ALL:
                per_scan_cap = MAX_NEW_ARB_POSITIONS_PER_SCAN
            if selected_per_source.get(signal.source, 0) >= per_scan_cap:
                filtered["source_scan_cap"] += 1
                continue

            max_positions = (
                MAX_HEDGE_POSITIONS
                if signal.action == SignalAction.HEDGE_BOTH
                else MAX_DIRECTIONAL_POSITIONS
            )
            if source_counts.get(signal.source, 0) >= max_positions:
                filtered["source_position_cap"] += 1
                continue

            cap = self._strategy_exposure_cap_pct(signal.source, signal.action)
            current_exposure = source_exposure.get(signal.source, 0.0)
            projected_exposure = current_exposure + max(signal.suggested_size_usd, 0.0)
            if projected_exposure > self.portfolio.starting_capital * cap:
                filtered["source_exposure_cap"] += 1
                continue

            selected.append(signal)
            seen_conditions.add(signal.condition_id)
            selected_per_source[signal.source] = selected_per_source.get(signal.source, 0) + 1
            source_counts[signal.source] = source_counts.get(signal.source, 0) + 1
            source_exposure[signal.source] = projected_exposure

        return selected, {k: v for k, v in filtered.items() if v}

    def execute_signal(self, signal: Signal, current_prices: dict[str, float]) -> PaperTrade | None:
        """
        Execute a signal in paper mode.
        Returns the simulated trade or None if rejected by risk checks.
        """
        self.signal_log.append(signal)

        # Risk checks
        if not self._passes_risk_checks(signal):
            logger.debug(f"[PAPER] Signal {signal.id} rejected by risk checks")
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
        """Execute a directional buy (YES or NO) with Kelly-inspired sizing."""
        if not signal.token_id:
            return None

        price = current_prices.get(signal.token_id, 0)
        if price <= 0:
            return None

        # Apply slippage (0.5%)
        fill_price = price * 1.005
        if fill_price >= 0.99:
            return None

        # Kelly-inspired sizing for directional bets
        if signal.expected_edge > 0 and signal.confidence > 0:
            odds = (1.0 / max(signal.confidence, 0.1)) - 1.0
            kelly_raw = (signal.confidence * odds - (1 - signal.confidence)) / max(odds, 0.01)
            kelly_fraction = max(0.01, min(kelly_raw * 0.25, 0.20))  # quarter-Kelly, floor 1%, cap 20%
            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * kelly_fraction)
        else:
            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * 0.05)

        if size_usd < 1.0:  # minimum $1 trade
            logger.info(f"[PAPER] Directional size too small: ${size_usd:.2f}")
            return None

        shares = size_usd / fill_price
        fee = BASE_FEE_RATE * min(fill_price, 1 - fill_price) * shares

        # Check total cost doesn't exceed cash
        total_cost = size_usd + fee
        if total_cost > self.portfolio.cash:
            size_usd = (self.portfolio.cash - fee) * 0.90
            shares = size_usd / fill_price
            fee = BASE_FEE_RATE * min(fill_price, 1 - fill_price) * shares
            total_cost = size_usd + fee

        if total_cost > self.portfolio.cash or size_usd < 1.0:
            return None

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
            avg_entry_price=fill_price,
            current_price=fill_price,
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
            price=fill_price,
            size_shares=shares,
            size_usd=size_usd,
            status=TradeStatus.FILLED,
        )
        self.trade_log.append(trade)
        self._log_trade(trade, signal)

        logger.info(
            f"[PAPER] Executed: {signal.action.value} on {signal.market_slug} | "
            f"{shares:.1f} shares @ ${fill_price:.3f} = ${size_usd:.2f} (fee: ${fee:.3f})"
        )
        return trade

    def _execute_hedge(
        self, signal: Signal, current_prices: dict[str, float]
    ) -> PaperTrade | None:
        """Execute a hedged liquidity provision order (both sides)."""
        size_usd = min(signal.suggested_size_usd, self.portfolio.cash * 0.25)
        if size_usd < 1.0:
            return None

        self.portfolio.cash -= size_usd
        self.portfolio.total_trades += 1

        position = Position(
            token_id=signal.token_id or "HEDGE",
            condition_id=signal.condition_id,
            market_slug=signal.market_slug,
            side="HEDGE",
            shares=size_usd,
            avg_entry_price=1.0,
            current_price=1.0,
            source=signal.source,
        )
        self.portfolio.positions.append(position)

        trade = PaperTrade(
            signal_id=signal.id,
            source=signal.source,
            market_slug=signal.market_slug,
            condition_id=signal.condition_id,
            token_id=signal.token_id or "HEDGE",
            side=Side.BUY,
            price=1.0,
            size_shares=size_usd,
            size_usd=size_usd,
            status=TradeStatus.FILLED,
        )
        self.trade_log.append(trade)
        self._log_trade(trade, signal)
        logger.info(f"[PAPER] Hedge position on {signal.market_slug}: ${size_usd:.2f}")
        return trade

    def _execute_arb(
        self, signal: Signal, current_prices: dict[str, float]
    ) -> PaperTrade | None:
        """
        Execute a multi-outcome arb (buy all outcomes).
        Arb sizing is deterministic (not Kelly) — arb is guaranteed profit, not a bet.
        """
        if signal.arb_total_cost <= 0:
            logger.warning(f"[PAPER] Arb rejected: zero cost on {signal.market_slug}")
            return None

        # Arb sizing: edge-scaled with hard cap at 25% of cash
        edge_pct = signal.expected_edge / 100.0 if signal.expected_edge > 0 else 0
        max_size = self.portfolio.cash * 0.25
        edge_scaled = max_size * min(edge_pct / 0.10, 1.0)  # full allocation at 10%+ edge

        if signal.suggested_size_usd > 0:
            size_usd = min(signal.suggested_size_usd, edge_scaled)
        else:
            size_usd = edge_scaled

        size_usd = max(size_usd, 0)
        if size_usd < 1.0:
            logger.info(
                f"[PAPER] Arb size too small: ${size_usd:.2f} | "
                f"edge={edge_pct:.3f} | cash=${self.portfolio.cash:.2f}"
            )
            return None

        # Calculate guaranteed profit
        units = size_usd / signal.arb_total_cost
        guaranteed_profit = units * (signal.arb_guaranteed_payout - signal.arb_total_cost)
        fee_estimate = size_usd * BASE_FEE_RATE * 0.5

        net_profit = guaranteed_profit - fee_estimate
        if net_profit <= 0:
            logger.info(
                f"[PAPER] Arb unprofitable after fees: gross=${guaranteed_profit:.3f} "
                f"fee=${fee_estimate:.3f} net=${net_profit:.3f}"
            )
            return None

        # Execute — track as open position (profit at resolution, not instant)
        self.portfolio.cash -= size_usd
        self.portfolio.total_fees_paid += fee_estimate
        self.portfolio.total_trades += 1

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
            realized_pnl=0.0,  # credited at resolution, not now
        )
        self.trade_log.append(trade)
        self._log_trade(trade, signal)

        logger.info(
            f"[PAPER] Arb executed: {signal.market_slug} | "
            f"Cost: ${size_usd:.2f} | Units: {units:.1f} | "
            f"Expected profit: ${net_profit:.3f} (at resolution)"
        )
        return trade

    def check_exits(self, current_prices: dict[str, float]):
        """Check if any positions should be exited (take-profit, stop-loss, time)."""
        exits = []
        queued_positions: set[int] = set()
        now = datetime.now(timezone.utc)

        for pos in self.portfolio.positions[:]:
            if pos.side == "HEDGE":
                age_hours = (now - pos.opened_at).total_seconds() / 3600
                if age_hours >= HEDGE_ROTATION_HOURS:
                    exits.append((pos, 1.0, f"HEDGE_ROTATE: {age_hours:.1f}h"))
                    queued_positions.add(id(pos))
                continue
            if pos.side == "ARB":
                continue

            # Get current price
            current = current_prices.get(pos.token_id, pos.current_price)
            if current <= 0:
                continue

            entry = pos.avg_entry_price
            if entry <= 0:
                continue

            pnl_pct = (current - entry) / entry
            reason = None

            # Take profit: price moved +20% from entry
            if pnl_pct >= DIRECTIONAL_TAKE_PROFIT_PCT:
                reason = f"TAKE_PROFIT: +{pnl_pct:.1%}"

            # Stop loss: price moved -25% from entry
            elif pnl_pct <= DIRECTIONAL_STOP_LOSS_PCT:
                reason = f"STOP_LOSS: {pnl_pct:.1%}"

            # Close out directional bets once the held token has effectively converged.
            elif current >= TOKEN_CONVERGENCE_PRICE:
                reason = f"CONVERGED: {pos.side} at ${current:.3f}"

            if reason:
                exits.append((pos, current, reason))
                queued_positions.add(id(pos))

        # Keep liquidity from monopolizing the portfolio. Trim the oldest hedge
        # positions until the source is back within its dedicated budget/slot cap.
        liquidity_positions = [
            pos
            for pos in self.portfolio.positions
            if pos.source == SignalSource.LIQUIDITY
            and pos.side == "HEDGE"
            and id(pos) not in queued_positions
        ]
        liquidity_cap_usd = (
            self.portfolio.starting_capital
            * self._strategy_exposure_cap_pct(SignalSource.LIQUIDITY)
        )
        liquidity_exposure = sum(self._position_value(pos) for pos in liquidity_positions)
        if liquidity_positions and (
            liquidity_exposure > liquidity_cap_usd
            or len(liquidity_positions) > MAX_HEDGE_POSITIONS
        ):
            liquidity_positions.sort(key=lambda pos: pos.opened_at)
            trimmed = 0
            while liquidity_positions and (
                liquidity_exposure > liquidity_cap_usd
                or len(liquidity_positions) > MAX_HEDGE_POSITIONS
            ):
                pos = liquidity_positions.pop(0)
                exits.append((pos, 1.0, "REBALANCE: liquidity reserve"))
                queued_positions.add(id(pos))
                liquidity_exposure -= self._position_value(pos)
                trimmed += 1
            logger.info(
                "[RISK] Rebalancing liquidity: trimmed %s hedge positions to %.0f%% cap",
                trimmed,
                self._strategy_exposure_cap_pct(SignalSource.LIQUIDITY) * 100,
            )

        # Execute exits
        for pos, exit_price, reason in exits:
            # Calculate realized P&L
            if pos.side == "HEDGE":
                pnl = 0.0
                proceeds = pos.shares
                fee = 0.0
            elif pos.side == "YES":
                pnl = (exit_price - pos.avg_entry_price) * pos.shares
                proceeds = pos.shares * exit_price
                fee = proceeds * EXIT_FEE_ESTIMATE_RATE
            else:  # NO
                pnl = (exit_price - pos.avg_entry_price) * pos.shares
                proceeds = pos.shares * exit_price
                fee = proceeds * EXIT_FEE_ESTIMATE_RATE

            # Return capital + P&L to cash
            self.portfolio.cash += proceeds
            self.portfolio.total_realized_pnl += pnl
            self.portfolio.total_fees_paid += fee

            if pnl > 0:
                self.portfolio.winning_trades += 1
            elif pnl < 0:
                self.portfolio.losing_trades += 1

            self.portfolio.positions.remove(pos)
            logger.info(
                f"[EXIT] {pos.market_slug} | {pos.side} | {reason} | "
                f"Entry: ${pos.avg_entry_price:.3f} → Exit: ${exit_price:.3f} | "
                f"PnL: ${pnl:+.2f} | Cash freed: ${proceeds:.2f}"
            )
        return len(exits)

    def update_positions(self, current_prices: dict[str, float]):
        """Update position mark-to-market and portfolio stats."""
        # Fix stale peak_value (may be wrong from default or corrupted state)
        if self.portfolio.peak_value > self.portfolio.total_value * 1.5:
            self.portfolio.peak_value = self.portfolio.starting_capital
        if self.portfolio.peak_value < self.portfolio.starting_capital * 0.5:
            self.portfolio.peak_value = self.portfolio.starting_capital

        for pos in self.portfolio.positions:
            if pos.token_id == "ARB_ALL":
                continue  # arb profit is fixed, no mark-to-market
            if pos.side == "HEDGE":
                continue  # hedge P&L comes from reward accrual
            new_price = current_prices.get(pos.token_id, pos.current_price)
            pos.current_price = new_price
            pos.unrealized_pnl = (new_price - pos.avg_entry_price) * pos.shares

        # Update peak and drawdown
        total_val = self.portfolio.total_value
        if total_val > self.portfolio.peak_value:
            self.portfolio.peak_value = total_val
        if self.portfolio.peak_value > 0:
            dd = (self.portfolio.peak_value - total_val) / self.portfolio.peak_value
            self.portfolio.current_drawdown = dd
            if dd > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = dd

    def resolve_positions(self, markets: list):
        """Check if any positions are on resolved markets and settle them."""
        # Build a lookup of closed markets
        closed = {}
        for m in markets:
            if m.closed and m.outcomes:
                # Determine winning outcome — price closest to 1.0
                winner = max(m.outcomes, key=lambda o: o.price)
                closed[m.condition_id] = {
                    "slug": m.slug,
                    "winner_token": winner.token_id,
                    "winner_name": winner.name,
                }

        if not closed:
            return 0

        # Check each position
        resolved = []
        for pos in self.portfolio.positions[:]:
            if pos.condition_id not in closed:
                continue

            result = closed[pos.condition_id]
            pnl = 0.0

            if pos.side == "ARB":
                # Arb always wins — collect $1.00 per unit
                payout = pos.shares * 1.0
                cost = pos.shares * pos.avg_entry_price
                pnl = payout - cost
            elif pos.side == "HEDGE":
                # Hedge collects $1.00 per share (both sides = guaranteed $1)
                pnl = 0.0  # break even on the hedge, profit was from rewards
                # Return the capital
            elif pos.side in ("YES", "NO"):
                # Directional bet
                if pos.side == "YES" and pos.token_id == result["winner_token"]:
                    pnl = pos.shares * (1.0 - pos.avg_entry_price)  # won
                elif pos.side == "NO" and pos.token_id == result["winner_token"]:
                    pnl = pos.shares * (1.0 - pos.avg_entry_price)  # won
                else:
                    pnl = -(pos.shares * pos.avg_entry_price)  # lost

            # Return capital + P&L to cash
            if pos.side == "HEDGE":
                self.portfolio.cash += pos.shares  # return hedge capital
            elif pos.side == "ARB":
                self.portfolio.cash += pos.shares * 1.0  # arb payout
            else:
                self.portfolio.cash += max(0, pos.shares * pos.avg_entry_price + pnl)

            # Track wins/losses
            self.portfolio.total_realized_pnl += pnl
            if pnl > 0:
                self.portfolio.winning_trades += 1
            elif pnl < 0:
                self.portfolio.losing_trades += 1

            resolved.append((pos, pnl))

        # Remove resolved positions
        for pos, pnl in resolved:
            self.portfolio.positions.remove(pos)
            logger.info(
                f"[RESOLVED] {pos.market_slug} | {pos.side} | "
                f"PnL: ${pnl:+.2f} | Cash returned to portfolio"
            )
        return len(resolved)

    def _update_drawdown(self):
        """Recalculate drawdown from actual portfolio value."""
        current = self.portfolio.cash + sum(
            p.shares * p.current_price for p in self.portfolio.positions
        )
        if current >= self.portfolio.starting_capital:
            self.portfolio.current_drawdown = 0.0
        elif self.portfolio.starting_capital > 0:
            dd = (self.portfolio.starting_capital - current) / self.portfolio.starting_capital
            self.portfolio.current_drawdown = dd
            if dd > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = dd

    def _passes_risk_checks(self, signal: Signal) -> bool:
        """Check if a signal passes risk management rules."""
        self._update_drawdown()

        # Check max total exposure
        if self.portfolio.positions_value >= self.portfolio.starting_capital * 2:
            logger.debug("[RISK] Max exposure reached")
            return False

        # Check current drawdown (30% threshold)
        if self.portfolio.current_drawdown > 0.30:
            logger.warning("[RISK] Max drawdown exceeded — pausing trading")
            return False

        # Check cash available (minimum $2 to trade, not 50% of suggested size)
        if self.portfolio.cash < MIN_TRADE_CASH_USD:
            logger.debug(
                f"[RISK] Insufficient cash: ${self.portfolio.cash:.2f} < ${MIN_TRADE_CASH_USD}"
            )
            return False

        # No duplicate positions on same market
        existing = [p for p in self.portfolio.positions if p.condition_id == signal.condition_id]
        if existing:
            logger.debug(f"[RISK] Already have position on {signal.market_slug}")
            return False

        # Max positions per strategy (higher limit for hedged positions)
        same_source = [p for p in self.portfolio.positions if p.source == signal.source]
        max_positions = (
            MAX_HEDGE_POSITIONS
            if signal.action == SignalAction.HEDGE_BOTH
            else MAX_DIRECTIONAL_POSITIONS
        )
        if len(same_source) >= max_positions:
            logger.debug(f"[RISK] Too many positions ({len(same_source)}/{max_positions}) from {signal.source.value}")
            return False

        # Max exposure per strategy (higher for hedged positions since risk is bounded)
        strategy_exposure = sum(
            self._position_value(p) for p in self.portfolio.positions
            if p.source == signal.source
        )
        cap = self._strategy_exposure_cap_pct(signal.source, signal.action)
        if strategy_exposure > self.portfolio.starting_capital * cap:
            logger.debug(f"[RISK] Strategy {signal.source.value} at {cap:.0%} cap (exposure: ${strategy_exposure:.0f})")
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
                stats["win_rate"] = stats["wins"] / stats["trades"]

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
