# Polymarket Algo Trader

Autonomous algo trading pipeline for Polymarket prediction markets with a real-time dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              ALWAYS-ON LAYERS                       │
│                                                     │
│  Layer 1: Hedged Liquidity Provision (the salary)   │
│  Layer 2: Multi-Outcome Arbitrage   (the bonus)     │
│  Layer 3: Whale Wallet Tracking     (the advisor)   │
│                                                     │
├─────────────────────────────────────────────────────┤
│           COMPETING SIGNAL STRATEGIES               │
│                                                     │
│  • News-to-Price Latency (requires ANTHROPIC_API_KEY│
│  • Mean Reversion                                   │
│                                                     │
│  Paper trade both → keep the winner                 │
├─────────────────────────────────────────────────────┤
│               ENGINE                                │
│                                                     │
│  Paper Trader → Risk Manager → Trade Logger         │
│                                                     │
├─────────────────────────────────────────────────────┤
│             DASHBOARD (port 8000)                   │
│                                                     │
│  Portfolio • Signals • Trades • Markets • Whales    │
└─────────────────────────────────────────────────────┘
```

## Quick Start (Local)

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your settings (paper mode needs no wallet keys)

# 3. Run
python main.py

# 4. Open dashboard
# → http://localhost:8000
```

## Deploy to Railway

```bash
# 1. Push to GitHub

# 2. Connect repo to Railway
#    → railway.app → New Project → Deploy from GitHub

# 3. Set environment variables in Railway dashboard:
#    TRADING_MODE=paper
#    PORT=8000
#    ANTHROPIC_API_KEY=sk-...  (optional)

# 4. Deploy — your dashboard will be live at your Railway URL
```

## Trading Modes

| Mode | Behavior |
|------|----------|
| `paper` | Full simulation. No real trades. Logs everything for analysis. |
| `shadow` | Paper trades + shows exact orders that *would* be submitted. |
| `live` | Real trades via Polymarket CLOB API. Requires wallet keys. |

## Strategies

### Layer 1: Hedged Liquidity Provision
- Places resting orders on both YES and NO sides
- Earns Polymarket's daily Liquidity Rewards
- Risk-bounded by hedge (if both fill → risk-free)
- Based on Kelly-optimal position sizing

### Layer 2: Multi-Outcome Arbitrage
- Scans binary markets for YES+NO ≠ $1.00
- Scans multi-outcome events for sum of YES prices ≠ $1.00
- Accounts for fees and slippage before triggering

### Layer 3: Whale Wallet Tracking
- Tracks top-performing wallets from the leaderboard
- Used as a confirmation filter (not standalone signals)
- When whales agree with a signal → size up
- When whales disagree → reduce confidence

### News-to-Price Latency (Optional)
- Ingests RSS news feeds in real-time
- Pre-filters headlines with keyword matching
- Classifies relevant headlines via Claude API
- Generates directional signals before market reacts
- Requires `ANTHROPIC_API_KEY` — costs ~$10-30/month

### Mean Reversion
- Tracks 72-hour price baselines per market
- Flags markets with >10% moves in 6 hours
- Bets on reversion to baseline
- Competing with News strategy — paper trade both, keep the winner

## Tuning

Every signal and trade is logged to `logs/signals.jsonl` and `logs/trades.jsonl`.

Bring these logs to Claude and ask:
- "Which strategy is performing best?"
- "Should I tighten the confidence threshold?"
- "Are there patterns in my losing trades?"

The dashboard also shows per-strategy P&L, win rates, and drawdowns.

## Project Structure

```
polymarket-algo/
├── main.py                 # FastAPI server + entry point
├── config.py               # All tunable parameters
├── data/
│   ├── collector.py        # Polymarket API client
│   └── models.py           # Pydantic data models
├── strategies/
│   ├── base.py             # Strategy interface
│   ├── liquidity.py        # Hedged liquidity provision
│   ├── arbitrage.py        # Multi-outcome arbitrage
│   ├── whale.py            # Whale wallet tracking
│   ├── news.py             # News-to-price (LLM)
│   └── mean_reversion.py   # Mean reversion
├── engine/
│   └── paper_trader.py     # Paper trading + P&L tracking
├── dashboard/
│   └── index.html          # Real-time web dashboard
├── logs/                   # Auto-created trade/signal logs
├── requirements.txt
├── Procfile                # Railway deployment
├── railway.json
└── .env.example
```

## Risk Warning

This is experimental software for educational purposes. Prediction market trading carries financial risk. Always paper trade first. Never trade with money you can't afford to lose.
