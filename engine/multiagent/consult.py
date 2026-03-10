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
CONSULT_MAX_TOKENS = int(os.getenv("MULTIAGENT_CONSULT_MAX_TOKENS", "2200"))
ALLOW_ANTHROPIC_AUTO = os.getenv("CONSULT_ALLOW_ANTHROPIC_AUTO", "").lower() in {"1", "true", "yes", "on"}


async def consult_multiagent_logs(
    *,
    question: str,
    context: dict[str, Any],
    preferred_provider: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    system_prompt = (
        "You are diagnosing an isolated Polymarket paper-trading runtime. "
        "Use only the provided metrics, cycle reports, blockers, health data, policy snapshot, news terminal, "
        "recent trades, and closed-position history. "
        "Do not invent market behavior that is not in the payload. "
        "Distinguish explicitly between current-state fields and recent-window historical rollups. "
        "Do not present historical rollup counts as current blockers unless the latest_scan or blockers section confirms that. "
        "Answer concisely with: 1) what is happening, 2) main blockers or failure modes, "
        "3) what is working, 4) the next best engineering action. "
        "If evidence is insufficient, say so clearly."
    )
    return await _consult_runtime(
        question=question,
        context=context,
        preferred_provider=preferred_provider,
        history=history,
        system_prompt=system_prompt,
        unavailable_message="No configured LLM provider could answer the Opus consult request.",
    )


async def consult_legacy_logs(
    *,
    question: str,
    context: dict[str, Any],
    preferred_provider: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    system_prompt = (
        "You are diagnosing the legacy Oracle paper-trading engine. "
        "Use only the provided current state, compact diagnostics, comparison-view summaries, strategy stats, "
        "recent headlines, trade tape, legacy blocker summary, current_blocker_summary, and recent_pattern_summary. "
        "Do not mix Opus or multi-agent assumptions into this answer. "
        "Distinguish clearly between current-state data and recent historical rollups. "
        "Treat current_blocker_summary and latest_scan as authoritative for what is happening right now. "
        "Treat recent_pattern_summary, scan_tape, strategy_rollup, and rejection_rollup only as background trend context. "
        "If warm_start.active is true, explicitly say the engine has recently restarted and current blocker attribution may still be stabilizing. "
        "Do not say '0 trades executed' if current_blocker_summary.executed or latest_scan.executed is nonzero. "
        "Do not present aggregated rejection totals as live blockers unless the latest diagnostics or blockers section confirms them. "
        "Answer concisely with: 1) what is happening, 2) main blockers or failure modes, "
        "3) what is working, 4) the next best engineering action. "
        "If evidence is insufficient, say so clearly."
    )
    return await _consult_runtime(
        question=question,
        context=context,
        preferred_provider=preferred_provider,
        history=history,
        system_prompt=system_prompt,
        unavailable_message="No configured LLM provider could answer the legacy-engine consult request.",
    )


async def _consult_runtime(
    *,
    question: str,
    context: dict[str, Any],
    preferred_provider: str | None,
    history: list[dict[str, str]] | None,
    system_prompt: str,
    unavailable_message: str,
) -> dict[str, Any]:
    user_content = {
        "question": question,
        "conversation_history": _sanitize_history(history or []),
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
        "answer": f"{unavailable_message} Last error: {last_error}",
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

    order = ["fireworks", "openai"]
    preferred = (preferred_provider or "auto").strip().lower()
    if preferred in available:
        order = [preferred] + [name for name in order if name != preferred]
    elif preferred == "auto" and ALLOW_ANTHROPIC_AUTO and "anthropic" in available:
        order.append("anthropic")
    elif preferred == "auto":
        # Auto is intentionally cheap-first. Anthropic should be an explicit operator choice.
        order = [name for name in order if name in available]
    if preferred != "auto" and preferred != "anthropic" and "anthropic" in available and ALLOW_ANTHROPIC_AUTO:
        order.append("anthropic")

    return [available[name] for name in order if name in available]


def _sanitize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for item in history[-12:]:
        role = str(item.get("role", "") or "").strip().lower()
        content = str(item.get("content", "") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        sanitized.append(
            {
                "role": role,
                "content": content[:4000],
            }
        )
    return sanitized


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
                "max_tokens": CONSULT_MAX_TOKENS,
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

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    request_payload = {
        "model": model,
        "max_tokens": CONSULT_MAX_TOKENS,
        "temperature": 0.2,
        "reasoning_effort": "high",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _safe_json(user_content)},
        ],
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            FIREWORKS_API_URL,
            headers=headers,
            json=request_payload,
        )
        if response.status_code == 412:
            relaxed_payload = dict(request_payload)
            relaxed_payload.pop("reasoning_effort", None)
            response = await client.post(
                FIREWORKS_API_URL,
                headers=headers,
                json=relaxed_payload,
            )
        response.raise_for_status()
        payload = _decode_json_response(response, "fireworks")

    choice = (payload.get("choices") or [{}])[0]
    message = (choice.get("message") or {})
    answer = _extract_chat_content(message.get("content")).strip()
    if not answer:
        answer = _extract_chat_content(message.get("reasoning_content")).strip()
    if not answer:
        answer = _extract_chat_content(choice.get("text")).strip()
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
                "max_completion_tokens": CONSULT_MAX_TOKENS,
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
    answer = _extract_chat_content((choice.get("message") or {}).get("content")).strip()
    if not answer:
        answer = _extract_chat_content(choice.get("text")).strip()
    if not answer:
        raise RuntimeError("openai_empty_response")
    return answer


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


def _safe_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
