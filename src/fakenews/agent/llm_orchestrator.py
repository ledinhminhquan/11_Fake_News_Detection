"""Optional LLM fact-check brain (anthropic), with rule fallback.

Advisory only: may write a one-line plain-language rationale over the evidence the
rules already selected. Disabled by default; validates its own output and on any
problem the caller keeps the rule result. Default deployment makes zero paid API
calls and is fully deterministic. **Never invents a verdict or a citation** — it
only rephrases the rule decision; citations come verbatim from the retrieved
corpus (transparency is non-negotiable for this project).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..config import AgentConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        return bool(self.cfg.llm_fallback_enabled and os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic
            key = os.environ.get(self.cfg.llm_api_key_env)
            self._client = anthropic.Anthropic(api_key=key) if key else None
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def explain(self, claim: str, verdict: str, evidence: List[Dict]) -> Optional[str]:
        """A one-line rationale over the given evidence. None to keep the rule rationale."""
        if not self.available() or not evidence:
            return None
        client = self._get_client()
        if client is None:
            return None
        refs = "; ".join(f"[{e.get('source','?')}] {e.get('text','')[:120]} (stance={e.get('stance')})"
                         for e in evidence[:3])
        prompt = ("Given a claim, a rule-decided verdict, and evidence passages, write ONE plain-language "
                  "sentence explaining the verdict using ONLY the evidence. Do not change the verdict or "
                  "invent facts.\n\n"
                  f"Claim: {claim}\nVerdict: {verdict}\nEvidence: {refs}\n\nReturn ONLY the sentence.")
        try:
            msg = client.messages.create(model=self.cfg.llm_model, max_tokens=160, temperature=0.0,
                                         messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") for b in msg.content).strip()
            return text or None
        except Exception as exc:
            logger.info("LLM explain failed (%s)", exc)
            return None


__all__ = ["LLMBrain"]
