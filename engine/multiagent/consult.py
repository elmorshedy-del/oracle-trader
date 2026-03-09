from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
OPENAI_API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
DEFAULT_FIREWORKS_MODEL = os.getenv("FIREWORKS_CONSULT_MODEL", "accounts/fireworks/models/glm-5")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_CONSULT_MODEL", "gpt-5")


async def consult_multiagent_logs(
    *,
    question: str,
    context: dict[str, Any],
    preferred_provider: str | None = None,
) -> dict[str, Any]:
    system_prompt = (
        "You are diagnosing an isolated Polymarket paper-trading runtime. "
        "Use only the provided metrics, cycle reports, blockers, health data, news terminal, "
        "recent trades, and closed-position history. "
        "Do not invent market behavior that is not in the payload. "
        "Answer concisely with: 1) what is happening, 2) main blockers or failure modes, "
        "3) what is working, 4) the next best engineering action. "
        "If evidence is insufficient, say so clearly."
    )

    user_content = {
        "question": question,
        "runtime_context": context,
    }

    last_error = "no_provider_configured"
    for provider, model in _provider_chain(preferred_provider):
        try:
            answer = await _call_provider(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_content=user_content,
            )
            return {
                "ok": True,
                "answer": answer,
                "provider": provider,
                "model": model,
                "generated_at": _now_iso(),
            }
        except Exception as exc:
            last_error = f"{provider}:{exc}"

    return {
        "ok": False,
        "answer": (
            "No configured LLM provider could answer the Opus consult request. "
            f"Last error: {last_error}"
        ),
        "provider": None,
        "model": None,
        "generated_at": _now_iso(),
    }


def _provider_chain(preferred_provider: str | None = None) -> list[tuple[str, str]]:
    available: dict[str, tuple[str, str]] = {}
    if os.getenv("FIREWORKS_API_KEY", ""):
        available["fireworks"] = ("fireworks", DEFAULT_FIREWORKS_MODEL)
    if os.getenv("ANTHROPIC_API_KEY", ""):
        available["anthropic"] = ("anthropic", DEFAULT_ANTHROPIC_MODEL)
    if os.getenv("OPENAI_API_KEY", ""):
        available["openai"] = ("openai", DEFAULT_OPENAI_MODEL)

    order = ["fireworks", "anthropic", "openai"]
    preferred = (preferred_provider or "auto").strip().lower()
    if preferred in available:
        order = [preferred] + [name for name in order if name != preferred]

    return [available[name] for name in order if name in available]


async def _call_provider(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_content: dict[str, Any],
) -> str:
    provider = provider.lower()
    if provider == "anthropic":
        return await _call_anthropic(model, system_prompt, user_content)
    if provider == "fireworks":
        return await _call_fireworks(model, system_prompt, user_content)
    if provider == "openai":
        return await _call_openai(model, system_prompt, user_content)
    raise RuntimeError(f"unsupported_provider:{provider}")


async def _call_anthropic(
    model: str,
    system_prompt: str,
    user_content: dict[str, Any],
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("anthropic_api_key_missing")

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 900,
                "temperature": 0.2,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _safe_json(user_content),
                            }
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()

    text_blocks = [
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    ]
    answer = "\n".join(part for part in text_blocks if part).strip()
    if not answer:
        raise RuntimeError("anthropic_empty_response")
    return answer


async def _call_fireworks(
    model: str,
    system_prompt: str,
    user_content: dict[str, Any],
) -> str:
    api_key = os.getenv("FIREWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("fireworks_api_key_missing")

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            FIREWORKS_API_URL,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            json={
                "model": model,
                "max_tokens": 1200,
                "temperature": 0.2,
                "reasoning_effort": "high",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _safe_json(user_content)},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()

    choice = (payload.get("choices") or [{}])[0]
    answer = (choice.get("message", {}) or {}).get("content", "")
    answer = answer.strip()
    if not answer:
        raise RuntimeError("fireworks_empty_response")
    return answer


async def _call_openai(
    model: str,
    system_prompt: str,
    user_content: dict[str, Any],
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("openai_api_key_missing")

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            OPENAI_API_URL,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_completion_tokens": 900,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _safe_json(user_content)},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()

    choice = (payload.get("choices") or [{}])[0]
    answer = (choice.get("message", {}) or {}).get("content", "")
    answer = answer.strip()
    if not answer:
        raise RuntimeError("openai_empty_response")
    return answer


def _safe_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
