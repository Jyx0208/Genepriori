"""Multi-agent claim debate: heuristic and LLM agents under a Governor."""

from __future__ import annotations

from .base import Agent, DebateContext, Proposal, ScoredClaim, make_claim, sanitize_weights
from .debate import config_for_claim, run_debate
from .governor import Decision, Governor, GovernorConfig
from .heuristic import (
    CausalGeneticsAgent,
    ConsensusDesignerAgent,
    NetworkMechanismAgent,
    TranslationalAgent,
    default_agents,
)
from .llm import LLMAgent, LLMClient, parse_claims

__all__ = [
    "Agent",
    "DebateContext",
    "Proposal",
    "ScoredClaim",
    "make_claim",
    "sanitize_weights",
    "Governor",
    "GovernorConfig",
    "Decision",
    "CausalGeneticsAgent",
    "NetworkMechanismAgent",
    "TranslationalAgent",
    "ConsensusDesignerAgent",
    "default_agents",
    "LLMAgent",
    "LLMClient",
    "parse_claims",
    "run_debate",
    "config_for_claim",
]
