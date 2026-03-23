# Oracle Research Ideas

## GLM-5 Ideas

### Weather model v2 data / ML enhancement side project

- Keep this as a side project, separate from current live sleeve changes.
- Main thesis: `weather_model_v2_trader` is the best candidate for ML/data enhancement because it already has richer model infrastructure but is underperforming the older v1 sleeve.

#### Why it looks promising

- `weather_model_v2_trader` already runs a larger ML stack:
  - `15` models loaded
  - CatBoost / LightGBM / XGBoost ensembles
  - `48` features
- It underperforms the simpler v1 sleeve:
  - `weather_model_trader (v1)`: total PnL `117.17`, win rate `42.5%`, trades `352`
  - `weather_model_v2_trader`: total PnL `38.66`, win rate `35.3%`, trades `306`
- Thresholds are more permissive in v2:
  - `min_edge 0.03` vs `0.07` in v1
- This suggests optimization opportunity rather than missing architecture.

#### Comparison snapshot

| Sleeve | Total PnL | Win Rate | Trades | ML Models |
|---|---:|---:|---:|---:|
| `weather_model_trader (v1)` | `117.17` | `42.5%` | `352` | `5` |
| `weather_model_v2_trader` | `38.66` | `35.3%` | `306` | `15` |
| `weather_sniper` | `105.93` | `39.8%` | `425` | `0` |
| `weather_swing` | `-4.03` | `48.7%` | `121` | `0` |

#### Working assumptions

- v1 performance implies the ML approach itself is viable.
- v2 likely has feature noise, data-quality issues, threshold problems, or calibration drift.
- `weather_edge_live` showing `100%` win rate on a tiny sample is not strong evidence by itself, but it reinforces keeping weather research alive.

#### Best next research tasks

1. Compare v1 vs v2 feature importance.
2. Review v2 training-data quality and bundle composition.
3. Check whether `legacy-weather-ml-v2` is frozen on stale or weaker external-only data.
4. Add or improve live weather API data inputs for future retraining.
5. Revisit v2 thresholds, especially whether `min_edge 0.03` is too permissive.

#### Guardrails

- Do not change the currently frozen weather-edge research lanes because of this note.
- Treat this as a separate ML-improvement project for the core weather model stack.
