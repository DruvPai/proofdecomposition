"""Typed schemas for normalized agent outputs and KB entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class KBKind(str, Enum):
    """Kinds of knowledge base entries."""

    DEFINITION = "Definition"
    NOTATION = "Notation"
    RESULT = "Result"
    ALGORITHM = "Algorithm"
    EXAMPLE = "Example"
    COUNTEREXAMPLE = "Counterexample"


@dataclass
class KBEntry:
    """Knowledge base entry."""

    id: str
    kind: KBKind
    title: str
    content_md: str
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class SolutionAttempt:
    """Structured solution produced by a prover."""

    final_answer_md: str
    outline_steps: list[str] = field(default_factory=list)
    kb_updates: list[KBEntry] = field(default_factory=list)
    claims_incorrect_conclusion: bool = False


@dataclass
class VerificationReport:
    """Verifier's judgment on prover attempts."""

    accepted: bool
    best_attempt_index: int | None
    attempt_scores: list[int]
    attempt_critiques_md: list[str]
    global_feedback_md: str


@dataclass
class ExplorationQuestions:
    """Questions proposed during exploration."""

    questions: list[str]
    rationales_md: list[str]


@dataclass
class OrchestratorStatus:
    """Structured status update produced by the orchestrator.

    Attributes:
        phase: str. High-level execution phase (e.g., "exploration", "solve").
        round_index: int | None. Exploration round index when applicable.
        message: str. Human-readable status message for tracing.
    """

    phase: str
    round_index: int | None
    message: str


@dataclass
class WorkerStatus:
    """Structured status update produced by a worker agent.

    Attributes:
        phase: str. Current loop phase ("prover_generation", "verification", etc.).
        round_index: int. Number of verification rounds completed so far.
        provers_spawned: int. Count of prover agents spawned in this transition.
        verifier_spawned: bool. Whether a verifier spawn was requested.
        decomposition_triggered: bool. Whether decomposition sub-workers were spawned.
        feedback_md: str | None. Verifier feedback propagated to the next round.
        notes: str | None. Optional human-readable detail for debugging.
    """

    phase: str
    round_index: int
    provers_spawned: int
    verifier_spawned: bool
    decomposition_triggered: bool
    feedback_md: str | None = None
    notes: str | None = None
