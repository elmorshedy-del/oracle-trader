from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


async def consult_multiagent_logs(
    *,
    question: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {
            "ok": False,
            "answer": "No Anthropic API key is configured for the multi-agent consult connector.",
            "model": None,
            "generated_at": _now_iso(),
        }

    system_prompt = (
        "You are diagnosing an isolated Polymarket paper-trading runtime. "
        "Use only the provided metrics, cycle reports, blockers, health data, and closed-position history. "
        "Do not invent market behavior that is not in the payload. "
        "Answer concisely with: 1) what is happening, 2) main blockers or failure modes, "
        "3) what is working, 4) the next best engineering action. "
        "If evidence is insufficient, say so clearly."
    )

    user_content = {
        "question": question,
        "runtime_context": context,
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
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
    return {
        "ok": True,
        "answer": answer,
        "model": payload.get("model", DEFAULT_MODEL),
        "generated_at": _now_iso(),
    }


def _safe_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True, default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
