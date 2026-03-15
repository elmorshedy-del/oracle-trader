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
- Historical loader fix:
  - Binance spot archive timestamps were in microseconds, not milliseconds
  - corrected in `engine/crypto_pairs/historical.py`
  - earlier coarse single-pair backtest numbers are superseded
- Corrected single-pair backtests:
  - `LINK/SOL`: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001358537377_v1/report.json`
    - `22` trades
    - `77.27%` win rate
    - `+624.5312 bps`
  - `AVAX/ETH`: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001358857144_v1/report.json`
    - `21` trades
    - `42.86%` win rate
    - `-166.8219 bps`
  - `AAVE/DOGE`: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001358640288_v1/report.json`
    - `18` trades
    - `50.00%` win rate
    - `+475.8089 bps`
- Corrected 3-pair basket:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001400250642_v1/report.json`
  - `69` trades
  - `57.97%` win rate
  - `+687.4537 bps`

## Next

- Freeze this scaffold as the baseline runtime checkpoint.
- Expand backtests from the top pair to the full active-pair basket using the same rule set.
- Only after the rule-based lane is characterized, consider the CatBoost upgrade from the architecture.

## External Supervision

- Status: `implemented`
- Tool:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/tools/supervise_crypto_pairs_shadow.py`
- Verified lifecycle:
  - detached `start`
  - independent `status`
  - controlled `stop`
- Active live supervisor:
  - id `crypto_pairs_shadow_supervisor_live_20260315T0031_v1`
  - root `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/shadow_supervision/crypto_pairs_shadow_supervisor_live_20260315T0031_v1`
  - worker runtime `3900s`
  - status file shows supervisor and child both alive
