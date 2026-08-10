"""Answer generation providers.

The generator is told to answer strictly from the retrieved context and to say
so when the context is insufficient. Grounding is the whole point of RAG: an
answer that ignores the sources is worse than no answer, because it looks
equally confident.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite the passages you rely on as [1], [2], and so on. "
    "If the context does not contain the answer, say that you cannot answer "
    "from the provided sources instead of guessing."
)


@runtime_checkable
class GenerationProvider(Protocol):
    async def generate(self, *, question: str, context: str) -> str:
        ...


def build_user_prompt(question: str, context: str) -> str:
    return f"Context passages:\n{context}\n\nQuestion: {question}"


class FakeGenerator:
    """Offline generator that echoes grounding metadata.

    Keeps tests deterministic and lets the API be demonstrated without keys.
    """

    async def generate(self, *, question: str, context: str) -> str:
        if not context.strip():
            return "I cannot answer from the provided sources."
        passages = context.count("[")
        return f"Answer to {question!r} grounded in {passages} passage(s). [1]"


class AnthropicGenerator:
    """Answer generation through the Anthropic Messages API."""

    _ENDPOINT = "https://api.anthropic.com/v1/messages"
    _VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("An Anthropic API key is required")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def generate(self, *, question: str, context: str) -> str:
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_prompt(question, context)}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._VERSION,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        blocks = data.get("content", [])
        return "".join(
            block.get("text", "") for block in blocks if block.get("type") == "text"
        ).strip()
