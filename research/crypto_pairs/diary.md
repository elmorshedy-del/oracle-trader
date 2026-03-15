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


## 2026-03-15 - Corrected V1 pair and basket backtests

- Historical data fix:
  - Binance spot kline archive timestamps were parsed as milliseconds but are microseconds in the downloaded files
  - corrected in `engine/crypto_pairs/historical.py`
  - the earlier first coarse backtest is superseded
- Corrected single-pair results:
  - `LINK/SOL`: `22` trades, `77.27%` win rate, `+624.5312 bps`
  - `AVAX/ETH`: `21` trades, `42.86%` win rate, `-166.8219 bps`
  - `AAVE/DOGE`: `18` trades, `50.00%` win rate, `+475.8089 bps`
- Corrected basket result:
  - top `3` pairs
  - `69` trades
  - `57.97%` win rate
  - `+687.4537 bps`
- Reports:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001358537377_v1/report.json`
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001358857144_v1/report.json`
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001358640288_v1/report.json`
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T001400250642_v1/report.json`


## 2026-03-15 - External supervisor added

- Tool:
  - `tools/supervise_crypto_pairs_shadow.py`
- What it does:
  - detached external supervisor process
  - `start / status / stop`
  - restart-aware state management
  - separate supervisor root with `state.json`, `supervisor.log`, and worker logs
- Verified smoke lifecycle:
  - start detached
  - status while child running
  - stop transitions final state to `stopped`
- Live long run started:
  - supervisor id `crypto_pairs_shadow_supervisor_live_20260315T0031_v1`
  - root `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/shadow_supervision/crypto_pairs_shadow_supervisor_live_20260315T0031_v1`
  - runtime `3900s`


## 2026-03-15 - Live ratio pipeline unblocked

- Runtime fix:
  - widened live leg-lag tolerance from `1500ms` to `10000ms`
  - compute ratio on every fresh leg bar using the last known opposite leg price
  - added counters for:
    - `no_price_reject`
    - `lag_reject`
    - `warmup_reject`
  - shadow logger now records ratio updates even before warmup is complete
- Verified live session:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/sessions/crypto_pairs_shadow_20260315T033639_v1/summary.json`
  - `5` ratio ticks
  - `283` websocket messages
  - `33` emitted bars
  - reject counts:
    - `no_price_reject: 27`
    - `lag_reject: 0`
    - `warmup_reject: 5`
- Read:
  - pair synchronization is no longer the blocker
  - warmup depth is now the reason there are still no live signals/trades in a short run


## 2026-03-15 - Extended 60-day V1 backtests

- Archive extension:
  - downloaded `2026-01-13` through `2026-02-11` for:
    - `LINKUSDT`
    - `SOLUSDT`
    - `AVAXUSDT`
    - `ETHUSDT`
    - `AAVEUSDT`
    - `DOGEUSDT`
- 60-day results using the same frozen pair selection and V1 rules:
  - `LINK/SOL`: `50` trades, `62.00%` win rate, `+439.3625 bps`
  - `AVAX/ETH`: `49` trades, `46.94%` win rate, `-263.6010 bps`
  - `AAVE/DOGE`: `43` trades, `58.14%` win rate, `+634.6798 bps`
  - basket: `156` trades, `53.21%` win rate, `+595.2121 bps`
- Reports:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T034351038366_v1/report.json`
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T034351038837_v1/report.json`
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T034351040225_v1/report.json`
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/crypto_pairs/backtests/crypto_pairs_backtest_20260315T034351328792_v1/report.json`
