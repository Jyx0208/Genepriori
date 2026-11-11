"""Optional LLM agents speaking any OpenAI-compatible chat API.

The implementation uses plain ``requests`` so no extra dependency is needed.
Configuration comes from the environment:

``GENEPRIORI_LLM_BASE_URL``  default ``https://api.deepseek.com/v1``
``GENEPRIORI_LLM_MODEL``     default ``deepseek-chat``
``GENEPRIORI_LLM_API_KEY``   falls back to ``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``

If a call fails or the response cannot be parsed, the agent silently falls
back to its heuristic counterpart so a debate always completes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional

import requests

from ..network import SUPPORTED_ALGORITHMS
from .base import Agent, DebateContext, Proposal, make_claim

LOG = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are one of four research agents in GenePriori, a leakage-aware gene
prioritization framework. You propose parameterised hypotheses ("claims") that
will be executed by an evaluation pipeline.

Hard constraints:
- The holdout evidence type must receive weight 0.0 in evidence_type_weights.
- Weights are non-negative and are renormalized automatically.
- network_algorithm must be one of: ppr, rwr, neighbor_voting, hd, ensemble, none.
- network_params may contain: damping, restart_prob, blend, network_discount,
  max_network_genes, max_neighbors_per_seed, min_score, heat_steps, heat_step_size.

Return ONLY a JSON object with this shape:
{"claims": [{
  "title": "short title",
  "hypothesis": "one or two sentences",
  "evidence_type_weights": {"drug": 0.0, "genetics": 0.5, "expression": 0.1,
                            "pathway": 0.2, "literature": 0.2, "animal_model": 0.0},
  "network_algorithm": "ppr",
  "network_params": {"damping": 0.85, "blend": 0.7},
  "expected_auprc_gain": 0.01,
  "expected_recall_gain": 0.01,
  "risk_tags": ["genetics-heavy"],
  "falsification_test": "what result would refute this claim"
}]}
Propose at most 3 claims. Do not include markdown fences or prose."""


class LLMClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 120.0,
        max_retries: int = 2,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("GENEPRIORI_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("GENEPRIORI_LLM_MODEL") or DEFAULT_MODEL
        self.api_key = (
            api_key
            or os.environ.get("GENEPRIORI_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.session = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key configured. Set GENEPRIORI_LLM_API_KEY "
                "(or DEEPSEEK_API_KEY / OPENAI_API_KEY), or use --mode heuristic."
            )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"HTTP {response.status_code}")
                response.raise_for_status()
                body = response.json()
                return str(body["choices"][0]["message"]["content"])
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
        raise RuntimeError(f"LLM request failed: {last_error}") from last_error


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_claims(
    payload: str,
    *,
    context: DebateContext,
    agent_name: str,
) -> List[Proposal]:
    """Parse and validate an LLM response into holdout-safe proposals."""
    data = json.loads(_strip_fences(payload))
    raw_claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(raw_claims, list):
        raise ValueError("response does not contain a 'claims' list")
    proposals: List[Proposal] = []
    for index, raw in enumerate(raw_claims[:3]):
        if not isinstance(raw, dict):
            continue
        weights = raw.get("evidence_type_weights") or {}
        if not isinstance(weights, dict):
            continue
        algorithm = str(raw.get("network_algorithm") or "ppr").strip().lower()
        if algorithm not in SUPPORTED_ALGORITHMS:
            algorithm = "ppr"
        params = raw.get("network_params") or {}
        if not isinstance(params, dict):
            params = {}
        try:
            proposal = make_claim(
                claim_id=f"llm-{context.round}-{index}",
                agent_name=agent_name,
                title=str(raw.get("title") or "LLM claim"),
                hypothesis=str(raw.get("hypothesis") or ""),
                weights={str(k): float(v) for k, v in weights.items()},
                holdout=context.holdout,
                algorithm=algorithm,
                network_params=params,
                cost=1.0,
                risk_tags=[str(t) for t in (raw.get("risk_tags") or [])],
                falsification_test=str(raw.get("falsification_test") or ""),
                predicted_auprc=float(raw.get("expected_auprc_gain") or 0.0),
                predicted_recall=float(raw.get("expected_recall_gain") or 0.0),
            )
        except (TypeError, ValueError) as exc:
            LOG.warning("dropping invalid LLM claim %d: %s", index, exc)
            continue
        proposals.append(proposal)
    return proposals


class LLMAgent(Agent):
    """Wrap an LLM in front of a heuristic agent used as fallback."""

    def __init__(
        self,
        name: str,
        persona: str,
        *,
        client: Optional[LLMClient] = None,
        fallback: Optional[Agent] = None,
    ) -> None:
        self.name = name
        self.persona = persona
        self.client = client or LLMClient()
        self.fallback = fallback

    def _user_prompt(self, context: DebateContext) -> str:
        best = context.best()
        leaderboard = [
            {
                "claim_id": entry.claim.claim_id,
                "agent": entry.claim.agent_name,
                "auprc": round(entry.measured_auprc, 5),
                "accepted": entry.accepted,
            }
            for entry in context.leaderboard[-8:]
        ]
        return json.dumps(
            {
                "persona": self.persona,
                "round": context.round,
                "holdout": context.holdout,
                "diseases": [d.name for d in context.diseases],
                "leaderboard": leaderboard,
                "best_claim": best.claim.claim_id if best else None,
                "feedback": context.feedback,
                "instructions": "Propose at most 3 new claims differing from the leaderboard.",
            },
            ensure_ascii=False,
            indent=2,
        )

    def propose(self, context: DebateContext) -> List[Proposal]:
        try:
            text = self.client.chat(SYSTEM_PROMPT, self._user_prompt(context))
            proposals = parse_claims(text, context=context, agent_name=self.name)
            if proposals:
                return proposals
            LOG.warning("%s returned no usable claims; using heuristic fallback", self.name)
        except Exception as exc:  # noqa: BLE001 - never break the debate
            LOG.warning("%s LLM call failed (%s); using heuristic fallback", self.name, exc)
        if self.fallback is not None:
            return self.fallback.propose(context)
        return []
