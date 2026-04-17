from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import httpx

from .config import LLMConfig, LLMTaskConfig


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
OPENAI_API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")


@dataclass(frozen=True)
class LLMTaskResult:
    task_name: str
    provider: str
    model: str
    success: bool
    parsed: bool
    latency_ms: int
    token_estimate: int | None
    prompt_hash: str
    raw_text: str | None
    parsed_json: dict[str, Any] | None
    error: str | None
    role: str = "primary"
    created_at: str = ""


class MultiagentLLMRouter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._calls_this_cycle: dict[str, int] = {}

    def begin_cycle(self) -> None:
        self._calls_this_cycle = {}

    async def complete_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        required_keys: tuple[str, ...],
    ) -> tuple[LLMTaskResult | None, list[LLMTaskResult]]:
        task_cfg = self.config.tasks.get(task_name)
        if not self.config.enabled or task_cfg is None or not task_cfg.enabled:
            return None, []

        used = self._calls_this_cycle.get(task_name, 0)
        if used >= task_cfg.max_calls_per_cycle:
            return (
                LLMTaskResult(
                    task_name=task_name,
                    provider="none",
                    model="disabled",
                    success=False,
                    parsed=False,
                    latency_ms=0,
                    token_estimate=None,
                    prompt_hash=_prompt_hash(system_prompt, user_payload),
                    raw_text=None,
                    parsed_json=None,
                    error="task_call_cap_reached",
                    created_at=_now_iso(),
                ),
                [],
            )

        self._calls_this_cycle[task_name] = used + 1
        attempts: list[tuple[str, str, str]] = [
            ("primary", task_cfg.primary_provider, task_cfg.primary_model),
        ]
        if task_cfg.fallback_provider and task_cfg.fallback_model:
            attempts.append(("fallback", task_cfg.fallback_provider, task_cfg.fallback_model))

        results: list[LLMTaskResult] = []
        for role, provider, model in attempts:
            result = await _call_provider(
                provider=provider,
                model=model,
                task_name=task_name,
                system_prompt=system_prompt,
                user_payload=user_payload,
                required_keys=required_keys,
                timeout_seconds=task_cfg.timeout_seconds,
                role=role,
            )
            results.append(result)
            if result.success and result.parsed:
                return result, results
        return None, results


async def _call_provider(
    *,
    provider: str,
    model: str,
    task_name: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    required_keys: tuple[str, ...],
    timeout_seconds: float,
    role: str,
) -> LLMTaskResult:
    started = datetime.now(timezone.utc)
    prompt_hash = _prompt_hash(system_prompt, user_payload)
    provider = provider.lower()
    try:
        if provider == "anthropic":
            raw_text, usage = await _call_anthropic(model, system_prompt, user_payload, timeout_seconds)
        elif provider == "fireworks":
            raw_text, usage = await _call_fireworks(model, system_prompt, user_payload, timeout_seconds)
        elif provider == "openai":
            raw_text, usage = await _call_openai(model, system_prompt, user_payload, timeout_seconds)
        else:
            raise RuntimeError(f"unsupported_provider:{provider}")

        parsed_raw = _parse_json(raw_text)
        if isinstance(parsed_raw, list) and task_name in {"trade_gate", "exit_judge"}:
            parsed_raw = {"decisions": parsed_raw}
        if not isinstance(parsed_raw, dict):
            raise ValueError("llm_output_not_object")
        parsed_json = _normalize_task_payload(task_name, parsed_raw)
        missing = [key for key in required_keys if key not in parsed_json]
        if missing:
            return LLMTaskResult(
                task_name=task_name,
                provider=provider,
                model=model,
                success=False,
                parsed=False,
                latency_ms=_elapsed_ms(started),
                token_estimate=usage,
                prompt_hash=prompt_hash,
                raw_text=raw_text,
                parsed_json=None,
                error=f"missing_required_keys:{','.join(missing)}",
                role=role,
                created_at=_now_iso(),
            )

        return LLMTaskResult(
            task_name=task_name,
            provider=provider,
            model=model,
            success=True,
            parsed=True,
            latency_ms=_elapsed_ms(started),
            token_estimate=usage,
            prompt_hash=prompt_hash,
            raw_text=raw_text,
            parsed_json=parsed_json,
            error=None,
            role=role,
            created_at=_now_iso(),
        )
    except Exception as exc:
        return LLMTaskResult(
            task_name=task_name,
            provider=provider,
            model=model,
            success=False,
            parsed=False,
            latency_ms=_elapsed_ms(started),
            token_estimate=None,
            prompt_hash=prompt_hash,
            raw_text=None,
            parsed_json=None,
            error=str(exc),
            role=role,
            created_at=_now_iso(),
        )


async def _call_anthropic(
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, int | None]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("anthropic_api_key_missing")
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 600,
                "temperature": 0.1,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=True)}],
                    }
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
    text = "\n".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    ).strip()
    usage = payload.get("usage", {})
    tokens = None
    if usage:
        tokens = int((usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0))
    return text, tokens


async def _call_fireworks(
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, int | None]:
    api_key = os.getenv("FIREWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("fireworks_api_key_missing")
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    request_payload = {
        "model": model,
        "max_tokens": 900,
        "temperature": 0.1,
        "reasoning_effort": "medium",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ],
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            FIREWORKS_API_URL,
            headers=headers,
            json=request_payload,
        )
        # 412 often indicates unsupported request knobs on specific model routes.
        if response.status_code == 412:
            relaxed_payload = dict(request_payload)
            # First retry: keep JSON mode, only drop reasoning controls.
            relaxed_payload.pop("reasoning_effort", None)
            response = await client.post(
                FIREWORKS_API_URL,
                headers=headers,
                json=relaxed_payload,
            )
            if response.status_code == 412:
                minimal_payload = dict(relaxed_payload)
                minimal_payload.pop("response_format", None)
                response = await client.post(
                    FIREWORKS_API_URL,
                    headers=headers,
                    json=minimal_payload,
                )
        response.raise_for_status()
        payload = _decode_json_response(response, "fireworks")
    choice = (payload.get("choices") or [{}])[0]
    raw = _extract_chat_content(choice.get("message", {}).get("content"))
    if not raw:
        raw = _extract_chat_content(choice.get("message", {}).get("reasoning_content"))
    if not raw:
        raw = _extract_chat_content(choice.get("text"))
    usage = payload.get("usage", {})
    tokens = int((usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)) if usage else None
    return raw.strip(), tokens


async def _call_openai(
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, int | None]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("openai_api_key_missing")
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            OPENAI_API_URL,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 600,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
    choice = (payload.get("choices") or [{}])[0]
    raw = _extract_chat_content(choice.get("message", {}).get("content"))
    usage = payload.get("usage", {})
    tokens = int((usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)) if usage else None
    return raw.strip(), tokens


def _parse_json(raw_text: str) -> Any:
    text = raw_text.strip()
    if not text:
        raise ValueError("empty_llm_output")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            return json.loads(text[obj_start : obj_end + 1])
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            return json.loads(text[arr_start : arr_end + 1])
        raise


def _decode_json_response(response: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception as exc:
        snippet = (response.text or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = f"{snippet[:240]}..."
        raise RuntimeError(
            f"{provider}_non_json_response status={response.status_code} body={snippet or '<empty>'}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{provider}_invalid_json_payload")
    return payload


def _extract_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
    return ""


def _normalize_task_payload(task_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if task_name == "trade_gate":
        normalized = dict(payload)
        decisions = normalized.get("decisions")
        if not isinstance(decisions, list):
            normalized["decisions"] = []
            return normalized
        cleaned = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "defer")).strip().lower()
            if action not in {"take", "skip", "defer"}:
                action = "defer"
            confidence = item.get("confidence", 0.0)
            try:
                confidence = float(confidence)
                if confidence > 1.0 and confidence <= 100.0:
                    confidence = confidence / 100.0
            except (TypeError, ValueError):
                confidence = 0.0
            cleaned.append(
                {
                    "signal_id": str(item.get("signal_id", "")).strip(),
                    "action": action,
                    "confidence": max(0.0, min(confidence, 1.0)),
                    "reason": str(item.get("reason", "") or ""),
                }
            )
        normalized["decisions"] = cleaned
        return normalized

    if task_name == "exit_judge":
        normalized = dict(payload)
        decisions = normalized.get("decisions")
        if not isinstance(decisions, list):
            normalized["decisions"] = []
            return normalized
        cleaned = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "hold")).strip().lower()
            if action not in {"hold", "exit"}:
                action = "hold"
            confidence = item.get("confidence", 0.0)
            try:
                confidence = float(confidence)
                if confidence > 1.0 and confidence <= 100.0:
                    confidence = confidence / 100.0
            except (TypeError, ValueError):
                confidence = 0.0
            cleaned.append(
                {
                    "position_id": str(item.get("position_id", "")).strip(),
                    "action": action,
                    "confidence": max(0.0, min(confidence, 1.0)),
                    "reason": str(item.get("reason", "") or ""),
                }
            )
        normalized["decisions"] = cleaned
        return normalized

    if task_name != "news_relevance":
        return payload

    normalized = dict(payload)
    if "market_slug" not in normalized:
        for alias in ("slug", "candidate_slug", "selected_market_slug", "best_market_slug"):
            if alias in normalized:
                normalized["market_slug"] = normalized[alias]
                break
    if "direction" not in normalized:
        for alias in ("bias", "sentiment", "market_direction"):
            if alias in normalized:
                normalized["direction"] = normalized[alias]
                break
    if "confidence" not in normalized:
        for alias in ("confidence_score", "score", "probability"):
            if alias in normalized:
                normalized["confidence"] = normalized[alias]
                break
    if "expected_impact_cents" not in normalized:
        for alias in ("impact_cents", "expected_move_cents", "impact", "price_impact_cents"):
            if alias in normalized:
                normalized["expected_impact_cents"] = normalized[alias]
                break

    direction = str(normalized.get("direction", "neutral")).strip().lower()
    direction_map = {
        "positive": "bullish",
        "up": "bullish",
        "yes": "bullish",
        "negative": "bearish",
        "down": "bearish",
        "no": "bearish",
    }
    normalized["direction"] = direction_map.get(direction, direction if direction in {"bullish", "bearish", "neutral"} else "neutral")

    confidence = normalized.get("confidence", 0.0)
    try:
        confidence = float(confidence)
        if confidence > 1.0 and confidence <= 100.0:
            confidence = confidence / 100.0
    except (TypeError, ValueError):
        confidence = 0.0
    normalized["confidence"] = max(0.0, min(confidence, 1.0))

    impact = normalized.get("expected_impact_cents", 0.0)
    try:
        impact = float(str(impact).replace("c", "").replace("¢", "").strip())
    except (TypeError, ValueError):
        impact = 0.0
    normalized["expected_impact_cents"] = max(0.0, impact)

    reasoning = normalized.get("reasoning")
    if reasoning is None:
        reasoning = normalized.get("explanation", "")
    normalized["reasoning"] = str(reasoning or "")
    return normalized


def _prompt_hash(system_prompt: str, user_payload: dict[str, Any]) -> str:
    return sha256(
        f"{system_prompt}\n{json.dumps(user_payload, sort_keys=True, default=str)}".encode()
    ).hexdigest()[:16]


def _elapsed_ms(started: datetime) -> int:
    return int((datetime.now(timezone.utc) - started).total_seconds() * 1000)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
