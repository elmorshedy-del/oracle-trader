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
