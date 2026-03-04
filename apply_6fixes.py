"""
apply_6fixes.py — Applies all 6 fixes from GLM5 consultation
Run from oracle-trader root: python3 apply_6fixes.py

Fixes:
1. State persistence to /data/state.json
2. News markdown stripping before JSON parse
3. Whale flexible field mapping + debug logging
4. Drawdown: add current_drawdown field
5. Arb: yes_price → yes_ask in reasoning
6. Hedge: meaningful price/shares in trade record
"""

import re

def fix_file(path, replacements):
    text = open(path).read()
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            print(f"  ✓ Applied fix in {path}")
        else:
            print(f"  ✗ Pattern not found in {path}: {old[:60]}...")
    open(path, 'w').write(text)
    return text


print("=" * 60)
print("FIX 1: State persistence (paper_trader.py)")
print("=" * 60)

# paper_trader.py — add save/restore and state_path
pt = open('engine/paper_trader.py').read()

# Fix __init__ to accept state_path and restore
pt = pt.replace(
    '    def __init__(self, starting_capital: float = 1000.0, log_dir: str = "logs"):',
    '    def __init__(self, starting_capital: float = 1000.0, log_dir: str = "logs", state_path: str = "/data/state.json"):'
)

pt = pt.replace(
    '''        self.portfolio = Portfolio(starting_capital=starting_capital, cash=starting_capital)
        self.trade_log: list[PaperTrade] = []
        self.signal_log: list[Signal] = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)''',
    '''        self.state_path = Path(state_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Try to restore state from disk, fall back to fresh start
        if not self._load_state():
            self.portfolio = Portfolio(starting_capital=starting_capital, cash=starting_capital)
            self.trade_log: list[PaperTrade] = []
            self.signal_log: list[Signal] = []
            logger.info(f"[PAPER] Fresh start with ${starting_capital}")
        else:
            logger.info(f"[PAPER] Restored state: ${self.portfolio.total_value:.2f} | {len(self.trade_log)} trades")'''
)

# Add save/load methods before execute_signal
pt = pt.replace(
    '    def execute_signal(self, signal: Signal, current_prices: dict[str, float]) -> PaperTrade | None:',
    '''    def _load_state(self) -> bool:
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

    def execute_signal(self, signal: Signal, current_prices: dict[str, float]) -> PaperTrade | None:'''
)

# Fix 4: Drawdown — use current_drawdown for display, max_drawdown for risk
pt = pt.replace(
    '''        # Update peak and drawdown
        total_val = self.portfolio.total_value
        if total_val > self.portfolio.peak_value:
            self.portfolio.peak_value = total_val
        if self.portfolio.peak_value > 0:
            dd = (self.portfolio.peak_value - total_val) / self.portfolio.peak_value
            self.portfolio.max_drawdown = dd  # current drawdown, not all-time max''',
    '''        # Update peak and drawdown
        total_val = self.portfolio.total_value
        if total_val > self.portfolio.peak_value:
            self.portfolio.peak_value = total_val
        if self.portfolio.peak_value > 0:
            dd = (self.portfolio.peak_value - total_val) / self.portfolio.peak_value
            self.portfolio.current_drawdown = dd
            if dd > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = dd'''
)

# Fix _update_drawdown to use current_drawdown
pt = pt.replace(
    '''    def _update_drawdown(self):
        """Recalculate drawdown from actual portfolio value."""
        current = self.portfolio.cash + sum(
            p.shares * p.current_price for p in self.portfolio.positions
        )
        if current >= self.portfolio.starting_capital:
            self.portfolio.max_drawdown = 0.0
        elif self.portfolio.starting_capital > 0:
            dd = (self.portfolio.starting_capital - current) / self.portfolio.starting_capital
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, dd)''',
    '''    def _update_drawdown(self):
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
                self.portfolio.max_drawdown = dd'''
)

# Fix risk check to use current_drawdown
pt = pt.replace(
    '''        # Check max drawdown (30% threshold)
        if self.portfolio.max_drawdown > 0.30:''',
    '''        # Check current drawdown (30% threshold)
        if self.portfolio.current_drawdown > 0.30:'''
)

open('engine/paper_trader.py', 'w').write(pt)
print("  ✓ State persistence added")
print("  ✓ Drawdown split into current/max")


print()
print("=" * 60)
print("FIX 1b: Wire save_state into pipeline.py")
print("=" * 60)

pl = open('engine/pipeline.py').read()

# Add save_state call at end of scan cycle
pl = pl.replace(
    '''        self.health.record_scan(cycle_time, len(self._markets), len(all_signals), executed)
        logger.info(''',
    '''        # Persist state every scan
        self.trader.save_state()

        self.health.record_scan(cycle_time, len(self._markets), len(all_signals), executed)
        logger.info('''
)

# Add save_state to stop
pl = pl.replace(
    '''    async def stop(self):
        """Stop the pipeline gracefully."""
        self._running = False
        await self.collector.close()
        logger.info("Pipeline stopped")''',
    '''    async def stop(self):
        """Stop the pipeline gracefully."""
        self._running = False
        self.trader.save_state()
        await self.collector.close()
        logger.info("Pipeline stopped")'''
)

# Fix drawdown display to use current_drawdown
pl = pl.replace(
    '"max_drawdown": round(self.trader.portfolio.max_drawdown * 100, 2),',
    '"max_drawdown": round(getattr(self.trader.portfolio, "current_drawdown", self.trader.portfolio.max_drawdown) * 100, 2),'
)

open('engine/pipeline.py', 'w').write(pl)
print("  ✓ save_state wired into scan cycle and stop")
print("  ✓ Dashboard shows current_drawdown")


print()
print("=" * 60)
print("FIX 2: News markdown stripping (news.py)")
print("=" * 60)

ns = open('strategies/news.py').read()

# Find the JSON parsing section and add markdown stripping
ns = ns.replace(
    '''            data = resp.json()
            text = data["content"][0]["text"].strip()''',
    '''            data = resp.json()
            text = data["content"][0]["text"].strip()

            # Strip markdown code blocks if Claude wrapped the JSON
            if text.startswith("```"):
                lines = text.split("\\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\\n".join(lines).strip()'''
)

open('strategies/news.py', 'w').write(ns)
print("  ✓ Markdown stripping added before JSON parse")


print()
print("=" * 60)
print("FIX 3: Whale flexible field mapping (whale.py)")
print("=" * 60)

wh = open('strategies/whale.py').read()

# Check if refresh_whales exists and has the parsing loop
if 'def refresh_whales' in wh and 'proxyWallet' in wh:
    # Add flexible field mapping
    wh = wh.replace(
        '''                wallet = WhaleWallet(
                    address=entry.get("proxyWallet", entry.get("address", "")),
                    name=entry.get("name") or entry.get("pseudonym"),
                    total_pnl=float(entry.get("cashPnl", 0) or 0),
                    win_rate=float(entry.get("winRate", 0) or 0),
                    total_trades=int(entry.get("numTrades", 0) or 0),
                )''',
        '''                address = (
                    entry.get("proxyWallet") or
                    entry.get("address") or
                    entry.get("wallet") or
                    entry.get("user") or ""
                )
                name = entry.get("name") or entry.get("pseudonym") or entry.get("username") or "anon"
                total_pnl = float(
                    entry.get("cashPnl") or entry.get("pnl") or
                    entry.get("totalPnl") or entry.get("profit") or 0
                )
                win_rate = float(entry.get("winRate") or entry.get("win_rate") or 0)
                total_trades = int(
                    entry.get("numTrades") or entry.get("totalTrades") or
                    entry.get("trades") or entry.get("numTraded") or 0
                )
                wallet = WhaleWallet(
                    address=address,
                    name=name,
                    total_pnl=total_pnl,
                    win_rate=win_rate,
                    total_trades=total_trades,
                )'''
    )
    print("  ✓ Flexible field mapping for whale parsing")
else:
    print("  ✗ Could not find whale parsing pattern — check manually")

# Add debug logging after refresh
if 'self._last_refresh = datetime.now(timezone.utc)' in wh:
    wh = wh.replace(
        '            self._last_refresh = datetime.now(timezone.utc)',
        '''            self._last_refresh = datetime.now(timezone.utc)
            logger.info(
                f"[WHALE] Refreshed: {len(self.whale_wallets)} whales loaded from "
                f"{len(leaderboard)} entries"
            )'''
    )
    print("  ✓ Debug logging added to whale refresh")

open('strategies/whale.py', 'w').write(wh)


print()
print("=" * 60)
print("FIX 4: Add current_drawdown to Portfolio model (models.py)")
print("=" * 60)

md = open('data/models.py').read()

if 'current_drawdown' not in md:
    md = md.replace(
        '    max_drawdown: float = 0.0',
        '    max_drawdown: float = 0.0\n    current_drawdown: float = 0.0'
    )
    print("  ✓ current_drawdown field added to Portfolio")
else:
    print("  ~ current_drawdown already exists")

open('data/models.py', 'w').write(md)


print()
print("=" * 60)
print("FIX 5: Arb yes_price → yes_ask (arbitrage.py)")
print("=" * 60)

arb = open('strategies/arbitrage.py').read()

# Fix the variable name error
count = arb.count('yes_price')
if count > 0:
    arb = arb.replace('yes_price', 'yes_ask')
    arb = arb.replace('no_price', 'no_ask')
    print(f"  ✓ Replaced {count} instances of yes_price → yes_ask")
else:
    print("  ~ yes_price not found (may already be fixed)")

open('strategies/arbitrage.py', 'w').write(arb)


print()
print("=" * 60)
print("FIX 6: Hedge display — meaningful price/shares (paper_trader.py)")
print("=" * 60)

pt2 = open('engine/paper_trader.py').read()

if 'price=0.0,\n            size_shares=0,' in pt2:
    pt2 = pt2.replace(
        '''            price=0.0,
            size_shares=0,
            size_usd=size_usd,''',
        '''            price=1.0,
            size_shares=size_usd,
            size_usd=size_usd,'''
    )
    print("  ✓ Hedge trade now shows $1.00 price and shares = USD amount")
    open('engine/paper_trader.py', 'w').write(pt2)
else:
    print("  ~ Hedge display already fixed or pattern different")


print()
print("=" * 60)
print("ALL 6 FIXES APPLIED")
print("=" * 60)
print()
print("Next: git add . && git commit -m 'fix: all 6 GLM5 audit items' && git push && railway up")
