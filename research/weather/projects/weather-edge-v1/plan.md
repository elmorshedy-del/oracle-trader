# Weather Edge v1

Standalone replay/backtest lane built on the frozen legacy CatBoost weather model.

## Scope

- Load resolved weather markets from the preserved history dataset.
- Sample Polymarket odds at `48h / 24h / 12h / 6h / 2h` before resolution.
- Replay the frozen weather model against those market snapshots.
- Compute raw edge, rule-based filter results, Kelly sizing, split-half, and 15-day summaries.
- Keep the live baseline weather sleeves untouched.
