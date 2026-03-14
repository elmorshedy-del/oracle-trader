# BTC Binance Historical Mean Reversion v1

This project is separate from the live multivenue BTC shadow sleeve.

- Reason: the frozen live model depends on Coinbase features that do not exist in Binance's bulk archive.
- Goal: use official Binance spot/futures bulk archives to expand historical BTC mean-reversion coverage without mutating the live multivenue track.
- Data sources:
  - Binance USD-M futures `aggTrades`
  - Binance USD-M futures `bookDepth`
  - Binance spot `aggTrades`
- Current v1 dataset family:
  - `btc_binance_historical_1s`
- First smoke date range:
  - `2026-03-01` to `2026-03-01`
- First checkpoint:
  - `btc-binance-historical-smoke-20260314T115752`

Rules for this project:

1. Keep the existing multivenue shadow model frozen.
2. Treat Binance historical as a separate research lane.
3. Do not call Binance-only historical results “validation” for the Coinbase-aware live model.
4. Freeze each historical sweep as its own checkpoint before expanding the date range.
