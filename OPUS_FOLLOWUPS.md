# Opus Follow-Ups

This file tracks the questions and tuning items we still want to take back to Opus later.

The goal is simple:
- keep building the isolated Opus runtime now
- avoid forgetting the deeper architecture and calibration questions
- separate "current implementation default" from "later Opus recommendation"

## Current Context

- Legacy Oracle engine stays untouched and keeps running.
- New Opus runtime is isolated under `/multiagent` and `engine/multiagent/`.
- Current Opus trade path is deterministic.
- First live Opus-native strategy is `relationship_arbitrage`.
- LLM consult exists only as an on-request diagnostic tool, not part of trading.

## Ask Opus Later

## 1. LLM Enrichment Boundaries

Ask:
- What exact structured outputs should we require for `news`, `relationships`, and `rule extraction`?
- What should the exact JSON schemas be?
- What hard guardrails should reject LLM output before it enters enrichment?
- What should be persisted for audit: prompt hash, model, raw response, parsed JSON, or all of them?

Current default:
- Defer all LLM trade-path use.
- Only on-request diagnostic consult is connected.

## 2. Weather Strategy Exact Math

Ask:
- Exact signal formulas for `weather_sniper`, `weather_latency`, and `weather_swing`
- Exact edge calculation for each
- Exact entry / exit rules
- Exact degraded-mode behavior when forecasts are stale or partially missing
- Whether the first slice should include early exits or resolution-only fallback for each variant

Current default:
- Weather is not yet migrated into the isolated Opus runtime.

## 3. Crypto Strategy Exact Math

Ask:
- Exact separation between `crypto_structure` and `crypto_latency`
- Exact formulas for ladder breaks, implication breaks, duplicate barriers, and spot-lag trades
- Exact exit rules for structure trades
- Exact anti-overlap rules so `crypto_structure` and `relationship_arbitrage` do not duplicate the same idea
- Exact rules for avoiding fuzzy directional guessing

Current default:
- `relationship_arbitrage` currently covers the first crypto structure slice.

## 4. Relationship / Arbitrage Scope

Ask:
- Should `relationship_arbitrage` stay as one strategy or split into:
  - duplicate equivalence
  - implication arbitrage
  - ladder arbitrage
- Should any of those be treated as separate sleeves in allocation?
- Should we add true bundle planning later or stay with single-leg relative value in paper mode first?

Current default:
- One strategy: `relationship_arbitrage`
- Single-leg paper entries only

## 5. Validation Thresholds By Strategy

Ask:
- Which validation thresholds should be global vs strategy-specific?
- Exact recommended minimums for:
  - volume
  - liquidity
  - market age
  - hours to resolution
  - edge
- Which strategies should bypass generic volume rules, if any?

Current default:
- `relationship_arbitrage` uses a strategy-specific `min_volume_24h = 0`
- Liquidity and edge checks still apply

## 6. Allocation / Risk Defaults

Ask:
- Exact sleeve percentages Opus recommends once more strategies are live
- Exact reserve cash policy for paper-trading comparison mode
- Exact per-strategy caps and max positions
- Exact re-entry cooldown policy
- Exact stale-rotation policy
- Exact correlated-position handling between related markets

Current default:
- Moderate paper-trading settings
- Reserve and caps exist, but are intentionally not ultra-conservative

## 7. Exit Policy

Ask:
- Exact exit policy Opus recommends for:
  - structure arbitrage
  - weather latency
  - crypto latency
  - future news / relationship trades
- Which should exit on convergence, stop loss, time decay, or resolution?
- Which close reasons should be first-class metrics in the dashboard?

Current default:
- Isolated runtime uses:
  - target convergence
  - stop loss
  - stale rotation

## 8. State and Persistence

Ask:
- Should Opus runtime state stay in-memory first, or move immediately to persisted state files / SQLite?
- What exact persistence model should be used for:
  - open positions
  - closed positions
  - cycle reports
  - provider freshness
  - prior forecast / spot snapshots

Current default:
- In-memory portfolio state
- Snapshot JSONs + compact JSONL metrics + SQLite runtime metrics store

## 9. Compact Metrics for LLM Consult

Ask:
- What is the best compact schema for LLM-readable diagnostics without wasting tokens?
- Which fields should be summarized vs raw?
- How much cycle history is useful for consultation?
- What should be omitted entirely from the consult context?

Current default:
- Compact SQLite rollups
- Recent cycle summaries
- Recent closes
- Recent fills
- Blockers, health, diagnostics, and snapshot summaries

## 10. Multi-LLM Connector Design

Ask:
- Best architecture for side-task LLM connector supporting:
  - ChatGPT
  - Sonnet
  - GLM-5
- Best abstraction for model selection, prompt shaping, audit logging, and fallback
- Whether all models should share one normalized consult contract

Current default:
- Diagnostic consult currently uses Anthropic-style endpoint only
- This is intentionally a side task, not in the trading path
- Comparison architecture draft is captured in `LLM_ENRICHMENT_COMPARISON_PLAN.md`

## 11. Operator UI / Dashboard Detail

Ask:
- Exact dashboard views Opus wants for:
  - signal funnel
  - allocation blockers
  - execution trace
  - recent closes
  - per-strategy sleeve usage
  - provider freshness
- Exact red/yellow/green sanity checks that matter most
- Which panels should be operator-facing by default vs hidden drill-downs

Current default:
- `/multiagent` already shows:
  - runtime health
  - workflow illustration
  - sanity checks
  - blockers
  - performance
  - strategy/module cards
  - on-request LLM consult

## 12. Provider Layer Design

Ask:
- Exact provider split Opus recommends:
  - metadata
  - relationships
  - weather
  - crypto
  - news
- Which providers should enrich every scan vs only candidate markets?
- Which providers should use TTL cache vs rolling snapshot history?

Current default:
- Minimal bare enricher in isolated runtime
- Priority CLOB enrichment already added for top markets

## 13. Testing Priorities

Ask:
- Exact golden-path tests Opus wants first
- Exact rejection-path tests per strategy
- Exact degraded-mode tests for provider failure
- Exact acceptance tests before migrating weather and crypto into Opus runtime

Current default:
- No full multiagent test suite yet

## 14. Final Target Comparison Model

Ask:
- Best apples-to-apples comparison structure between:
  - legacy Oracle engine
  - isolated Opus runtime
- Whether comparison should be:
  - same starting capital
  - same feed cadence
  - same market universe
  - separate sleeves per strategy inside Opus

Current default:
- Legacy and Opus are already isolated
- Both share external market data only

## 15. What To Ask Before Shipping More Strategies

Highest-priority next Opus questions:
1. Exact weather formulas and exits
2. Exact crypto latency formulas and exits
3. Exact sleeve / cap recommendations for paper mode
4. Exact compact consult schema for multi-LLM diagnostics
5. Exact relationship-provider / rule-extraction LLM guardrails

## Notes

Anything in this file is a "bring back to Opus later" item, not a blocker for the current deterministic runtime unless explicitly marked otherwise.
