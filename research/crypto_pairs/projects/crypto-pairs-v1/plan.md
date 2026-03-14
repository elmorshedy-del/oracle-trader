# Crypto Pairs v1

Follow the architecture document in order and freeze each phase before moving on.

## Phase 1: Pair Discovery

- Status: `frozen`
- Date range: `2026-02-12` to `2026-03-13`
- Symbols: `ETHUSDT, BTCUSDT, SOLUSDT, AVAXUSDT, NEARUSDT, DOTUSDT, ATOMUSDT, ADAUSDT, MATICUSDT, SUIUSDT, UNIUSDT, AAVEUSDT, LINKUSDT, MKRUSDT, ARBUSDT, OPUSDT, DOGEUSDT, SHIBUSDT`
- Report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/research/crypto_pairs/projects/crypto-pairs-v1/crypto_pairs_discovery_20260314T222257_v1/pair_discovery_results.json`
- Checkpoint: `crypto-pairs-discovery-20260314T222257`

Top tradeable pairs:
- `LINK/SOL` score `4.0985` halflife `13.95h`
- `AVAX/ETH` score `6.2311` halflife `20.15h`
- `AAVE/DOGE` score `7.5988` halflife `27.97h`

## Phase 2-7: Runtime Scaffold

- Status: `implemented and frozen`
- Modules:
  - `engine/crypto_pairs/price_streamer.py`
  - `engine/crypto_pairs/ratio_engine.py`
  - `engine/crypto_pairs/signal_engine_v1.py`
  - `engine/crypto_pairs/execution_engine.py`
  - `engine/crypto_pairs/position_manager.py`
  - `engine/crypto_pairs/logger.py`
  - `tools/run_crypto_pairs_shadow.py`
- Live streamer note:
  - primary Binance spot websocket returned `HTTP 451` in this environment
  - runtime now falls back to `wss://data-stream.binance.vision/stream`
- Live smoke session:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/sessions/crypto_pairs_shadow_20260314T231223_v1/summary.json`
  - `17` websocket messages, `6` emitted bars, `3` active pairs

## Phase 8: Backtester

- Status: `implemented`
- First coarse archive backtest:
  - pair: `LINK/SOL`
  - report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260314T223158_v1/report.json`
  - trades: `44`
  - win rate: `70.45%`
  - total PnL: `233.7625 bps`

## Next

- Freeze this scaffold as the baseline runtime checkpoint.
- Expand backtests from the top pair to the full active-pair basket using the same rule set.
- Only after the rule-based lane is characterized, consider the CatBoost upgrade from the architecture.
