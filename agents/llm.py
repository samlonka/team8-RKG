"""
llm.py — Bedrock-backed LLM helper (Claude Sonnet 4.6).

All "intelligence" calls in the agent pipeline go through here. Uses the
Anthropic SDK's AnthropicBedrock client with the account's AWS credentials.
Falls back gracefully (returns None) so the pipeline still runs heuristically
if Bedrock is unavailable.
"""
from __future__ import annotations

import os
import re
import json

# Sonnet 4.6 on Bedrock (discovered via `aws bedrock list-foundation-models`).
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-20860715-v1:0")
BEDROCK_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))


class LLM:
    def __init__(self, model: str = BEDROCK_MODEL_ID, region: str = BEDROCK_REGION):
        self.model = model
        self.region = region
        self.client = None
        self.available = False
        try:
            from anthropic import AnthropicBedrock
            self.client = AnthropicBedrock(aws_region=region)
            self.available = True
        except Exception:
            self.available = False

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.0) -> str | None:
        if not self.available:
            return None
        try:
            kw = dict(model=self.model, max_tokens=max_tokens, temperature=temperature,
                      messages=[{"role": "user", "content": prompt}])
            if system:
                kw["system"] = system
            resp = self.client.messages.create(**kw)
            return resp.content[0].text
        except Exception:
            return None

    def json(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> dict | None:
        txt = self.complete(prompt, system=system, max_tokens=max_tokens)
        if not txt:
            return None
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# module-level singleton (lazy)
_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM
