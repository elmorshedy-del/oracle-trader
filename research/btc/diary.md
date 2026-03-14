# BTC Research Diary

## 2026-03-13 - Bookkeeping started

This diary was created to stop BTC research from drifting across ad hoc output folders.
From this point forward, every BTC strategy track should be logged here and in the checkpoint ledger.

### Frozen history brought under bookkeeping

1. `btc-impulse-catboost-90d-v1`
   - Main 5-second BTC futures impulse continuation baseline.
   - Frozen report:
     `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/futures_ml_impulse_compare/binance_btcusdt_5s_impulse_60s_tp8_sl6_sig004_dep001_tz025_eff015_ctxcore_20251211_20260310_compare_train-c075-c100_eval-c075-c100_v1/reports/comparison_report.md`
   - Key metrics:
     - Long AUC `0.7412`
     - Short AUC `0.7282`
   - Status: best BTC model baseline so far.

2. `btc-impulse-xgboost-90d-v1`
   - XGBoost challenger on the same 5-second impulse setup.
   - Frozen report:
     `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/futures_ml_xgboost_compare/binance_btcusdt_5s_impulse_xgboost_20251211_20260310_tp8_sl6_sig004_dep001_tz025_eff015_src100_v1/reports/comparison_report.md`
   - Key metrics:
     - Long AUC `0.7371`
     - Short AUC `0.6989`
   - Status: real challenger, but weaker than CatBoost.

3. `btc-impulse-replay-14d-v1`
   - Realistic post-cost replay on the 5-second impulse baseline.
   - Frozen report:
     `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/futures_ml_replay/binance_btcusdt_5s_replay_20260224_20260309_tp8_sl6_sig004_dep001_tz025_eff015_src100_v1/reports/replay_report.md`
   - Key metrics:
     - Held-out test trades `57`
     - Net PnL `-15.09 USD`
     - Win rate `43.9%`
   - Status: proved that model signal existed, but monetization after costs was weak.

4. `btc-lob-deeplob-smoke-v1`
   - First proper L2-based LOB comparison on captured Binance futures depth.
   - Frozen report:
     `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/futures_lob_compare_cpu_smoke/lob_compare_btcusdt_2026-03-11_lvl20_seq030_ev020_t10s_v1/reports/comparison_report.md`
   - Key metrics:
     - XGBoost event AUC `0.7104`, net `-163.7070 bps`
     - DeepLOB-style event AUC `0.6020`, net `-166.1949 bps`
   - Status: proved that L2 compare framework works, but not enough data for DeepLOB to win yet.

### Current conclusion

- Keep the CatBoost impulse baseline frozen.
- Do not promote BTC models into production based on AUC alone.
- New BTC work should be multi-venue and execution-aware:
  - Binance futures
  - Binance spot
  - Coinbase level2
- New research split:
  - continuation track
  - mean-reversion track
  - regime filter on top

## 2026-03-13 - BTC multivenue v1 started

New active project:

- `btc-multivenue-v1`

Goal:

- collect new L2-quality data without mutating existing BTC champions
- add missing venue state:
  - Binance futures L2
  - Binance spot L2
  - Coinbase level2
- train new continuation and mean-reversion models later on a better state space

Why this track exists:

- the prior BTC models found predictive signal, but post-cost edge was too thin
- the likely missing ingredient is better market-state information, not another random threshold sweep

Freeze policy for this project:

- old checkpoints remain immutable
- new outputs live under a separate capture root
- new models will get their own checkpoint ids before any Oracle sleeve uses them

### Live session started

- Active multivenue capture session:
  `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_capture/sessions/20260313_130748_multivenue_v1_live`
- Reusable shared capture Python:
  `/Users/ahmedelmorshedy/.local/bin/oracle-btc-python`

### First multivenue base dataset built

- Builder:
  `/Users/ahmedelmorshedy/Downloads/oracle-trader/tools/build_btc_multivenue_dataset.py`
- Output:
  `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T131709_3sessions_v1/dataset/features.csv.gz`
- Metadata:
  `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T131709_3sessions_v1/dataset/metadata.json`

Initial dataset stats:

- rows: `593`
- columns: `67`
- span: `2026-03-13T13:04:13+00:00` to `2026-03-13T13:17:09+00:00`
- venues included:
  - Binance futures BTCUSDT
  - Binance spot BTCUSDT
  - Coinbase BTC-USD

Initial read:

- this is enough to verify the aligned multivenue feature pipeline
- this is not enough to train a serious new BTC model yet
- continue capture first, then train fresh continuation and mean-reversion baselines on the same immutable track

## 2026-03-13 - Auto-trained multivenue CatBoost baselines

- Capture session: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_capture/sessions/20260313_144242_multivenue_v1_live`
- Dataset: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T154112_5sessions_v1/dataset/features.csv.gz`
- Training run: `btc_multivenue_catboost_20260313T203335_v1`
- Checkpoint: `btc-multivenue-catboost-20260313T203338`
- continuation_long_30s: status `trained`, test AUC `0.3501516628750948`
- continuation_short_30s: status `trained`, test AUC `0.6693014545771274`
- meanrev_after_upshock_30s: status `skipped`, test AUC `None`
- meanrev_after_downshock_30s: status `skipped`, test AUC `None`

## 2026-03-13 - Impulse-conditioned multivenue CatBoost baselines

- Dataset: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T154112_5sessions_v1/dataset/features.csv.gz`
- Training run: `btc_multivenue_catboost_impulse_20260313T210101_v1`
- Checkpoint: `btc-multivenue-catboost-impulse-20260313T210104`
- Impulse definition: `past 5s absolute move >= 5 bps`, future horizon `30s`, profit/revert threshold `8 bps`
- impulse_continuation_after_upshock_30s: test AUC `0.4697`
- impulse_continuation_after_downshock_30s: test AUC `0.1500`
- meanrev_after_upshock_30s: test AUC `0.3492`
- meanrev_after_downshock_30s: test AUC `0.6305`
- Read: filtering to impulse rows made the models event-specific, but this sample is still small and unstable; only downshock mean-reversion has a non-terrible first test read.

## 2026-03-13 - Tuned downshock mean reversion

- Dataset: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T154112_5sessions_v1/dataset/features.csv.gz`
- Training run: `btc_multivenue_catboost_impulse_20260313T222428_v1`
- Checkpoint: `btc-multivenue-catboost-meanrev-downshock30-v2-20260313T222431`
- Tuning change: kept the impulse definition at `past 5s <= -5 bps`, but relaxed the required rebound from `+8 bps` to `+4 bps` over `30s`.
- meanrev_after_downshock_30s: valid AUC `0.7565`, test AUC `0.7703`, test precision@top-decile `0.6667`.
- Read: this is the first BTC multivenue result that looks strong enough to deserve dedicated replay instead of more generic retraining first.

## 2026-03-13 - Hybrid replay search for tuned downshock mean reversion

- Dataset: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T154112_5sessions_v1/dataset/features.csv.gz`
- Model: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_models/btc_multivenue_catboost_impulse_20260313T222428_v1/models/meanrev_after_downshock_30s.cbm`
- Search run: `btc_meanrev_hybrid_search_20260313T224152_v1`
- Checkpoint: `btc-meanrev-hybrid-search-20260313T224152`
- Search method: exhaustive grid over `900` core entry/exit combinations, then execution-only Monte Carlo stress on the top `20` cores, then bootstrap on the best stressed configuration.
- Candidate events: `265`
- Best core config: threshold `0.35`, take-profit `8 bps`, stop-loss `8-10 bps`, max hold `30s`
- Best core replay: `27` trades, `88.9%` win rate, total net `98.04 bps`
- Best stressed result: mean total net `91.63 bps`, p05 total net `49.46 bps`, positive total share `1.000`
- Bootstrap on best stressed trade list: mean total net `147.23 bps`, p05 total net `110.55 bps`, positive total share `1.000`
- Read: this is the first BTC replay checkpoint that looks economically promising under the tested fee/slippage/cooldown ranges. It is still a small-window result and should be treated as a replay candidate, not a production claim.

## 2026-03-13 - Frozen validation protocol registered

- Project spec: `/Users/ahmedelmorshedy/Downloads/oracle-trader/research/btc/projects/btc-meanrev-downshock30-v1/validation_spec.json`
- Validation runner: `/Users/ahmedelmorshedy/Downloads/oracle-trader/tools/run_btc_meanrev_frozen_validation.py`
- Builder update: `/Users/ahmedelmorshedy/Downloads/oracle-trader/tools/build_btc_multivenue_dataset.py` now supports `--session-dir` so future out-of-sample datasets can be built from new session roots only.
- Protocol checkpoint: `btc-meanrev-downshock30-validation-protocol-v1`
- Locked validation setup:
  - signal: `past 5s <= -5 bps`
  - model score threshold: `0.35`
  - take-profit: `8 bps`
  - stop-loss: `10 bps`
  - max hold: `30s`
  - cooldown: `3s`
  - fees: `1.0 bps/side`
  - slippage: `0.5 bps` entry and exit
- Promotion policy:
  - minimum new days: `5`
  - minimum total trades: `50`
  - positive-day-share target: `>= 0.55`
  - total net bps: `> 0`
  - bootstrap p05 total net bps: `> 0`
- Reference run on the discovery-day dataset:
  - validation run: `btc_meanrev_validation_20260313T230821_v1`
  - trades: `27`
  - win rate: `88.9%`
  - total net: `98.04 bps`
  - day count: `1`
- Read: this reference run is only a protocol sanity check because it uses the same discovery-day dataset. Future days must be evaluated unchanged through the frozen spec before this strategy can be promoted.

## 2026-03-14 - Frozen mean-reversion out-of-sample validation

- Checkpoint: `btc-meanrev-downshock30-validation-oos-20260314T015231`
- Capture session: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_capture/sessions/20260314_002124_meanrev_validation_v1`
- Dataset: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260314T002125_20260314T012124_1sessions_v1/dataset/features.csv.gz`
- Validation run: `btc_meanrev_validation_20260314T015230_v1`
- Trades: `2`
- Win rate: `0.5`
- Total net bps: `-7.232742120291565`
- Aggregate validation day count: `1`
- Aggregate positive-day share: `0.0`
- Aggregate bootstrap p05 total net bps: `-7.232742120291565`

## 2026-03-14 - Replay family across all 30s impulse quadrants

- Dataset: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T154112_5sessions_v1/dataset/features.csv.gz`
- Runner: `/Users/ahmedelmorshedy/Downloads/oracle-trader/tools/run_btc_meanrev_hybrid_search.py`
- Family checkpoint: `btc-quadrant-hybrid-family-20260314T035538`
- Purpose: rerun the same hybrid replay structure across all four 30-second impulse quadrants on the same March 13 discovery sample.

Results:

- `downshock -> long mean reversion`
  - report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_hybrid_family/down_meanrev_long/btc_meanrev_hybrid_search_20260314T035538_v1/reports/report.md`
  - candidate events: `265`
  - stressed mean total net: `91.63 bps`
  - stressed p05 total net: `49.46 bps`
  - positive total share: `1.000`

- `downshock -> short continuation`
  - report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_hybrid_family/down_cont_short/btc_meanrev_hybrid_search_20260314T035538_v1/reports/report.md`
  - candidate events: `265`
  - stressed mean total net: `43.80 bps`
  - stressed p05 total net: `-2.92 bps`
  - positive total share: `0.944`

- `upshock -> short mean reversion`
  - report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_hybrid_family/up_meanrev_short/btc_meanrev_hybrid_search_20260314T035538_v1/reports/report.md`
  - candidate events: `218`
  - stressed mean total net: `7.29 bps`
  - stressed p05 total net: `-14.74 bps`
  - positive total share: `0.731`

- `upshock -> long continuation`
  - report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_hybrid_family/up_cont_long/btc_meanrev_hybrid_search_20260314T035538_v1/reports/report.md`
  - candidate events: `218`
  - no replay configuration survived the minimum trade filter

- Read:
  - the March 13 monster was real, but it was concentrated in the `downshock -> long mean reversion` quadrant
  - `downshock -> short continuation` is the only other quadrant that looks even somewhat promising, but it is much less robust because the stressed lower bound dipped slightly negative
  - both `upshock` quadrants were materially weaker on the same discovery sample

## 2026-03-14 - Binance historical bulk-archive smoke

- Project: `/Users/ahmedelmorshedy/Downloads/oracle-trader/research/btc/projects/btc-binance-historical-meanrev-v1/plan.md`
- Checkpoint: `btc-binance-historical-smoke-20260314T115752`
- Downloader: `/Users/ahmedelmorshedy/Downloads/oracle-trader/tools/download_btc_binance_historical.py`
- Dataset builder: `/Users/ahmedelmorshedy/Downloads/oracle-trader/tools/build_btc_binance_historical_dataset.py`
- Dataset: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_binance_historical/datasets/btc_binance_historical_1s_20260301_20260301_v1/dataset/features.csv.gz`
- Training report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_binance_historical/models/btc_multivenue_catboost_impulse_20260314T115354_v1/reports/metadata.json`
- Smoke replay report: `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_binance_historical/validation/btc_meanrev_validation_20260314T115751_v1/reports/metadata.json`
- Dataset stats:
  - rows: `86,400`
  - columns: `68`
  - candidate downshock events: `2,204`
- First Binance-only model read:
  - `meanrev_after_downshock_30s` test AUC `0.4911`
  - test precision@top-decile `0.1972`
- Frozen post-cost smoke replay:
  - trades: `567`
  - win rate: `30.5%`
  - total net: `-2014.59 bps`
- Read:
  - official Binance bulk data is now a working BTC backtest source inside Oracle
  - this first Binance-only historical sample is materially weaker than the multivenue discovery track
  - so Binance historical should stay a separate research lane, not be used to “confirm” the Coinbase-aware shadow model

## 2026-03-14 - March 14 runtime multivenue same-day validation slice

- Source runtime session import:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_runtime_imports/20260314_065637_runtime_meanrev_shadow_v1`
- Dataset:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset_runtime_shadow_single_v3/btc_multivenue_1s_20260314T065738_20260314T121119_1sessions_v1/dataset/features.csv.gz`
- Validation report:
  - `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_meanrev_validation_runtime_shadow/btc_meanrev_validation_20260314T130746_v1/reports/report.md`
- Dataset stats:
  - rows: `9,372`
  - columns: `67`
  - span: `2026-03-14 06:57:38 UTC` to `2026-03-14 12:11:19 UTC`
- Frozen same-day runtime validation read:
  - candidate events: `1`
  - trades: `1`
  - win rate: `0.0%`
  - total net: `-0.58 bps`
  - exit: `timeout`
- Read:
  - the real March 14 runtime shadow archive did contain enough multivenue data to reconstruct a same-day dataset
  - the frozen downshock mean-reversion winner barely triggered in this runtime window
  - this is useful same-day exploratory evidence, but it is not a clean independent multi-day confirmation because it uses the same March 14 live regime
