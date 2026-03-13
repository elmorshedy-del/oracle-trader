# BTC Multivenue V1

Status: `collecting`

Objective:

- Build the next BTC research track without mutating any prior checkpoint.
- Collect multi-venue state for future continuation and mean-reversion models.

Data sources:

1. Binance futures BTCUSDT
   - `aggTrade`
   - `bookTicker`
   - `depth20@100ms`
   - `markPrice@1s`
   - `forceOrder`

2. Binance spot BTCUSDT
   - `aggTrade`
   - `bookTicker`
   - `depth20@100ms`

3. Coinbase BTC-USD
   - `level2_batch`
   - `ticker`
   - `heartbeat`

Planned model stack:

1. CatBoost continuation baseline
2. CatBoost mean-reversion baseline
3. TCN challengers
4. DeepLOB only after enough L2 history exists
5. Regime filter on top

Evaluation:

- walk-forward only
- replay after fees
- no promotion on AUC alone
- no reuse of mutable run folders

Freeze rule:

- new model families get a checkpoint before any Oracle sleeve consumes them
- prior checkpoints remain immutable even if the new track fails

Current live collection session:

- `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_capture/sessions/20260313_130748_multivenue_v1_live`

Reusable shared capture env:

- Python: `/Users/ahmedelmorshedy/.local/bin/oracle-btc-python`
- Pip: `/Users/ahmedelmorshedy/.local/bin/oracle-btc-pip`

First aligned dataset:

- `/Users/ahmedelmorshedy/Downloads/oracle-trader/output/btc_multivenue_dataset/btc_multivenue_1s_20260313T130413_20260313T131709_3sessions_v1/dataset/features.csv.gz`

Current base-table contents:

- futures book/tape/depth
- spot book/tape/depth
- Coinbase level2 + ticker
- cross-venue gaps
- future BTC return targets for `5s`, `10s`, `30s`, `60s`, `90s`

Current constraint:

- sample is still too short for a serious model promotion
- keep capture running, then train from this same versioned track
