"""Context graph and agent node definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from prover.schemas import KBEntry


class AgentType(str, Enum):
    """Supported agent types."""

    ORCHESTRATOR = "orchestrator"
    EXPLORATION = "exploration"
    WORKER = "worker"
    PROVER = "prover"
    VERIFIER = "verifier"
    PARSER = "parser"


@dataclass
class SpawnRequest:
    """Request to create a child agent."""

    agent_type: AgentType
    task: dict[str, Any]
    edge_from_parent: bool = True


@dataclass
class AgentOutput:
    """Result produced by an agent execution."""

    agent_type: AgentType
    raw_text: str
    normalized: Any
    kb_writes: list[KBEntry]
    spawn_requests: list[SpawnRequest]


@dataclass
class AgentNode:
    """Node in the context graph."""

    id: int
    agent_type: AgentType
    inputs: dict[str, Any]
    outputs: list[AgentOutput] = field(default_factory=list)
    status: str = "pending"  # pending | running | done
    parents: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    pending_children: set[int] = field(default_factory=set)
    waiting: bool = False
