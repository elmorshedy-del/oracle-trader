# Crypto Pairs Research Diary

This diary is append-only.
Each new crypto pairs discovery, backtest, or execution experiment gets a new entry.


## 2026-03-14 - Initial pair discovery

- Checkpoint: `crypto-pairs-discovery-20260314T222257`
- Date range: `2026-02-12` to `2026-03-13`
- Symbols loaded: `16`
- Pairs tested: `120`
- Tradeable pairs: `3`
- Report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/research/crypto_pairs/projects/crypto-pairs-v1/crypto_pairs_discovery_20260314T222257_v1/pair_discovery_results.json`
- Top pairs:
  - `LINK/SOL` score `4.0985` halflife `13.95h`
  - `AVAX/ETH` score `6.2311` halflife `20.15h`
  - `AAVE/DOGE` score `7.5988` halflife `27.97h`


## 2026-03-14 - Runtime scaffold and first backtest

- Status: `implemented`
- Runtime modules:
  - `engine/crypto_pairs/price_streamer.py`
  - `engine/crypto_pairs/ratio_engine.py`
  - `engine/crypto_pairs/signal_engine_v1.py`
  - `engine/crypto_pairs/execution_engine.py`
  - `engine/crypto_pairs/position_manager.py`
  - `engine/crypto_pairs/logger.py`
- Shadow runner:
  - `tools/run_crypto_pairs_shadow.py`
- Backtester:
  - `tools/backtest_crypto_pairs_v1.py`
- First backtest:
  - pair `LINK/SOL`
  - report `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260314T223158_v1/report.json`
  - `44` trades
  - `70.45%` win rate
  - `+233.7625 bps`
- Live smoke:
  - session `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/sessions/crypto_pairs_shadow_20260314T231223_v1/summary.json`
  - primary Binance websocket returned `HTTP 451`
  - fallback `data-stream.binance.vision` worked
  - `17` messages, `6` bars emitted
