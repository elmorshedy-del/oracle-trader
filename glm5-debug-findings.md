# GLM-5 Debug Findings: ARB Signals Generated, 0 Trades Executed

Date: 2026-03-04  
Repo: `oracle-trader`  
Scope: `engine/paper_trader.py`, `engine/pipeline.py`

## Problem Summary

- Pipeline scans markets and generates arbitrage signals.
- Logs show repeated: `2 signals | 0 executed`.
- Portfolio remains unchanged (`$500.00`, `0 trades`).

## Primary Root Cause

The arb execution path computes position size using a Kelly formulation that collapses to zero in the current setup.

- File: `engine/paper_trader.py`
- Key lines: `161-170` and `169-170`
- Blocking behavior:
  - `size_usd` becomes `0`
  - Execution exits early with `return None`
  - Trade is never recorded

### Why This Happens

Current logic derives `odds` from `confidence`:

```python
odds = (1.0 / max(signal.confidence, 0.1)) - 1.0
kelly_raw = (signal.confidence * odds - (1 - signal.confidence)) / max(odds, 0.01)
```

This setup can produce `kelly_raw ~= 0`, which then yields:

- `kelly_fraction = 0`
- `size_usd = 0`
- early return in `_execute_arb()`

## Ranked Causes (with confidence)

1. **High**: Arb sizing path returns zero size due to Kelly math in `engine/paper_trader.py:161-170`.
2. **Medium**: Silent risk rejection at `engine/paper_trader.py:270` (cash check fails without useful logging).
3. **Low**: `signal.suggested_size_usd` could be oversized/zero from upstream strategy output.

## Minimal Fix Direction

Use non-Kelly sizing for arbitrage (arb is structurally different from directional bets), then keep guardrails:

- In `_execute_arb()`: replace Kelly-based size with bounded edge-based or fixed-cap sizing.
- Keep hard cap, e.g. `10%-25%` of available cash.
- Keep existing profitability check (`net_profit > 0`).

Example patch shape:

```diff
--- a/engine/paper_trader.py
+++ b/engine/paper_trader.py
@@ -160,14 +160,16 @@ def _execute_arb(...):
-        # Kelly-inspired sizing...
-        if signal.expected_edge > 0 and signal.confidence > 0:
-            ...
-            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * kelly_fraction)
-        else:
-            size_usd = min(signal.suggested_size_usd, self.portfolio.cash * 0.05)
+        # Arb sizing: deterministic bounded sizing (not Kelly)
+        edge_pct = signal.expected_edge / 100.0
+        max_size = self.portfolio.cash * 0.25
+        edge_scaled = max_size * min(edge_pct / 0.10, 1.0)
+        if signal.suggested_size_usd > 0:
+            size_usd = min(signal.suggested_size_usd, edge_scaled)
+        else:
+            size_usd = edge_scaled
         if size_usd <= 0:
+            logger.warning(\"[PAPER] Arb size calculation returned 0\")
             return None
```

## Logs to Add (to prove fix in production)

1. **Pre-size arb input log** in `_execute_arb()`:
   - market, edge, confidence, suggested size, cash.
2. **Post-size arb decision log** in `_execute_arb()`:
   - computed `size_usd`, cap used, reason for clamping.
3. **Risk-check rejection log** in `_passes_risk_checks()` at cash gate (`line 270`):
   - required vs available cash, signal id, market slug.

## Secondary Bugs Found

- Duplicate `_update_drawdown()` method definitions:
  - `engine/paper_trader.py:235-244`
  - `engine/paper_trader.py:246-255`
- Duplicate `_update_drawdown()` calls in risk checks:
  - `engine/paper_trader.py:259-260`
- Duplicate A/B test creation blocks:
  - `engine/pipeline.py:62-71`
  - `engine/pipeline.py:72-81`
- Directional sizing likely has the same Kelly issue:
  - `engine/paper_trader.py:68-77`

## Validation Checklist After Patch

1. Run one cycle and confirm `signals > 0` and `executed > 0`.
2. Confirm arb trade entries appear in `logs/trades.jsonl`.
3. Confirm new logs show non-zero `size_usd` for arb signals.
4. Confirm cash decreases when arb positions open (and PnL behavior matches intended accounting model).
