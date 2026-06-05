"""
llm.py — Amazon Bedrock LLM (Claude Opus 4.7).

All agent intelligence calls go through this module. There is no Anthropic API-key
path and no heuristic fallback when a call is made — failures raise LLMError.

Requires AWS credentials with Bedrock access to anthropic.claude-opus-4-7 in
BEDROCK_REGION (default us-east-1). Override model via BEDROCK_MODEL_ID.
"""
from __future__ import annotations

import json
import re

from config import BEDROCK_MODEL_ID, BEDROCK_REGION


class LLMError(RuntimeError):
    """Bedrock LLM unavailable or returned an invalid response."""


def _opus_47_plus(model: str) -> bool:
    """Opus 4.7+ on Bedrock rejects legacy sampling params (temperature, top_p)."""
    m = model.lower()
    return "opus-4-7" in m or "opus-4-8" in m


class LLM:
    def __init__(self, model: str = BEDROCK_MODEL_ID, region: str = BEDROCK_REGION):
        self.model = model
        self.region = region
        self.client = None
        try:
            from anthropic import AnthropicBedrock
            self.client = AnthropicBedrock(aws_region=region)
        except Exception as e:
            raise LLMError(
                f"Failed to initialize Bedrock client (region={region}): {e}. "
                "Ensure AWS credentials are configured and Bedrock is enabled."
            ) from e

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        if not self.client:
            raise LLMError("Bedrock client not initialized")
        kw: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kw["system"] = system
        if not _opus_47_plus(self.model):
            kw["temperature"] = temperature
        try:
            resp = self.client.messages.create(**kw)
            return resp.content[0].text
        except Exception as e:
            raise LLMError(
                f"Bedrock completion failed (model={self.model}, region={self.region}): {e}"
            ) from e

    def json(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> dict:
        txt = self.complete(prompt, system=system, max_tokens=max_tokens)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise LLMError(f"Bedrock response contained no JSON object: {txt[:200]!r}")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(f"Bedrock returned invalid JSON: {e}") from e


_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM


def bedrock_model_label() -> str:
    return f"Bedrock {BEDROCK_MODEL_ID} ({BEDROCK_REGION})"
