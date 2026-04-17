# Weather Edge V3

This is a separate research lane from the frozen legacy weather edge sleeves.

## Thesis

Weather edge comes from forecast-information timing, not from inventing a better raw weather predictor. V3 only proceeds if we can reconstruct enough horizon-aligned historical data to test whether forecast revisions and model disagreement beat Polymarket pricing.

## Frozen V3 Scope

- Resolved Polymarket weather markets only
- Horizon-aligned market probabilities at `48h`, `24h`, `12h`, `6h`, `2h`
- Prior-run weather forecast values at the same horizons
- Multi-model features:
  - consensus probability
  - model agreement
  - model spread
- Revision features:
  - revision over the last `24h`
  - revision direction
- Score by edge, not by accuracy

## Data Collection Checklist

1. Market data
- resolved market id
- question
- city
- metric type
- threshold
- target date
- resolution timestamp
- resolved outcome
- Polymarket probability at `48h`, `24h`, `12h`, `6h`, `2h`

2. Forecast data
- one prior-run forecast record per model per horizon
- models:
  - `gfs_seamless`
  - `ecmwf_ifs025`
  - `icon_seamless`
  - `gem_seamless`
  - `jma_seamless`

3. Alignment rules
- market probability timestamp must be at or before the target horizon anchor
- forecast run must reflect what was knowable at that horizon
- no hindsight-filled features

## Model Configurations To Test

1. Pooled
- one model across all horizons
- horizon hours included as a feature

2. Bucketed
- early: `48h + 24h`
- mid: `12h + 6h`
- late: `2h`

3. Separate
- one model per horizon
- only allowed if day depth is strong enough

## Hard Data Gate

If this gate fails, V3 stops.

Minimum acceptance bar:
- enough independent historical days to support real backtesting
- enough horizon coverage to support the chosen config
- split-half and short-window checks must be possible

Practical thresholds:
- pooled: target `150-200` independent days
- bucketed: target `100-150` independent days per bucket
- separate: target `300+` independent days per horizon

## Backtest Checklist

1. Full-period backtest
- realized net PnL
- win rate
- max drawdown

2. Split-half validation
- first half positive
- second half positive

3. Short-window validation
- no deeply negative short block

4. Calibration check
- higher-edge bucket should outperform lower-edge bucket

## Promotion Rule

V3 does not become a live paper sleeve unless:
- the data gate passes
- the backtest passes full-period validation
- the split-half check passes
- the short-window check passes

If the data gate fails, V3 is killed early instead of forcing a weak model.
