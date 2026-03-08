# LLM Enrichment Comparison Plan

This plan is for the future engine-side LLM layer only.

It does not change the current deterministic Opus trade path.

## Goal

Make it possible to compare engine-side LLM providers against each other on the same bounded tasks:

- `news_relevance`
- `relationship_linking`
- `rule_extraction`

Primary runtime choice:
- `sonnet`

Fallback choice:
- `glm-5`

Later optional:
- `chatgpt`

## Non-Negotiable Rule

The LLM layer never owns:

- probabilities
- edge calculation
- sizing
- validation decisions
- allocation
- execution

The LLM layer only produces structured enrichment artifacts.

## Architecture

## 1. One LLM Task Interface

Create one internal contract for every engine-side LLM task:

```python
LLMTaskRequest(
    task_name: str,
    market_id: str,
    payload: dict,
    schema_name: str,
    primary_provider: str,
    fallback_provider: str | None,
    shadow_providers: list[str],
)
```

And one normalized response:

```python
LLMTaskResult(
    task_name: str,
    provider: str,
    model: str,
    success: bool,
    parsed: bool,
    latency_ms: int,
    token_estimate: int | None,
    raw_text: str | None,
    parsed_json: dict | None,
    error: str | None,
    prompt_hash: str,
    created_at: str,
)
```

## 2. Provider Abstraction

Each provider implements the same interface:

```python
class LLMProvider(Protocol):
    name: str

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict,
        schema_name: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMTaskResult:
        ...
```

Providers:

- `sonnet_provider`
- `glm5_provider`
- later `chatgpt_provider`

## 3. Task Router

One task router decides:

- which provider is primary
- which provider is fallback
- whether shadow mode is on
- which schema/prompt to use

Example defaults:

- `news_relevance`
  - primary: `sonnet`
  - fallback: `glm-5`
- `relationship_linking`
  - primary: `sonnet`
  - fallback: `glm-5`
- `rule_extraction`
  - primary: `sonnet`
  - fallback: `glm-5`

## 4. Shadow Mode

Shadow mode is how we compare models safely.

Behavior:

- one provider is `primary`
- one or more providers are `shadow`
- only the primary output enters enrichment
- shadow outputs are logged for evaluation only

This lets us compare models without changing live engine behavior.

## 5. Fallback Mode

If primary fails because of:

- no API key
- no funds
- timeout
- malformed output
- provider error

then:

- try `glm-5`
- record fallback reason
- record that fallback output became the active enrichment

## Comparison Modes

## Mode A: Primary Only

Use one model only.

Use when:

- low complexity
- no benchmarking needed
- cost sensitivity matters

## Mode B: Primary + Fallback

Use primary first, fallback only on failure.

Use when:

- production-style resilience matters

## Mode C: Primary + Shadow

Use primary for actual enrichment and run one or more shadow models in parallel.

Use when:

- comparing output quality
- comparing parse reliability
- comparing cost and latency

## Mode D: Scheduled A/B Windows

Use one provider as primary for a defined time window, then switch.

Example:

- week 1: Sonnet primary
- week 2: GLM-5 primary

Only do this once the enrichment layer is stable and the metrics store is ready.

## What To Compare

## 1. Reliability

- parse success rate
- schema validation pass rate
- provider timeout rate
- fallback rate

## 2. Quality

- false positive rate
- false negative rate
- operator override rate
- downstream usefulness in the strategy

Examples:

- `news_relevance`
  - did it include the truly relevant articles?
  - did it drag in junk?

- `relationship_linking`
  - did it falsely claim unrelated markets are related?
  - did it miss obvious related markets?

- `rule_extraction`
  - did it correctly extract date / threshold / jurisdiction / resolution source?

## 3. Operational Cost

- average latency
- average token estimate
- estimated dollar spend

## 4. Trading Usefulness

Do not compare models on raw “smartness”.

Compare them on:

- how often their enrichment survives deterministic validation
- how often their enrichment contributes to trades
- whether those trades later look useful

## Storage Model

## 1. LLM Calls Table

Store one row per provider call:

- task name
- market id
- provider
- model
- primary / fallback / shadow role
- prompt hash
- success
- parsed
- latency
- token estimate
- error
- created at

## 2. LLM Outputs Table

Store compact normalized outputs:

- task name
- provider
- market id
- parsed JSON
- schema version
- active vs shadow

## 3. Enrichment Outcome Table

Link the accepted active result to:

- the market
- the scan cycle
- the strategy that later consumed it

## 4. Evaluation Table

Later, store review labels such as:

- correct
- partially correct
- false positive
- false negative
- malformed
- useful
- not useful

## Dashboard Views Later

When this is implemented, the `/multiagent` section should add:

## 1. LLM Provider Health

- Sonnet status
- GLM-5 status
- ChatGPT status
- fallback events

## 2. Task Quality View

- parse success by task
- average latency by task
- error rate by task

## 3. Model Comparison View

For each task:

- provider
- active uses
- shadow uses
- parse success
- average latency
- fallback count
- operator-rated quality

## 4. Output Diff View

Show side-by-side:

- primary output
- shadow output
- chosen active output
- downstream outcome

## Recommended Rollout

## Stage 1

Implement provider abstraction only.

- Sonnet primary
- GLM-5 fallback
- no shadow mode yet

## Stage 2

Add logging + compact storage.

- persist each LLM call
- persist parsed output
- persist fallback reason

## Stage 3

Add shadow mode for one task first.

Best first task:
- `relationship_linking`

Reason:
- low execution risk
- easy to compare
- bounded output

## Stage 4

Add side-by-side dashboard evaluation.

## Stage 5

Run week-based A/B if needed.

Only after:

- stable schemas
- stable logging
- good operator visibility

## Recommended Task Order

If we later enable engine-side LLMs, implement in this order:

1. `relationship_linking`
2. `rule_extraction`
3. `news_relevance`

Reason:

- relationship linking is bounded and easier to audit
- rule extraction is second because it is structured
- news is the most open-ended and easiest to overdo

## Current Decision

For now:

- engine trading stays deterministic
- Sonnet is the intended primary model later
- GLM-5 is the intended fallback later
- comparison support should be built into the architecture from the start
- actual model testing can happen later without rewriting the engine
