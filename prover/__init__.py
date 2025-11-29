"""ProofDecomposition multi-agent prover package."""

from prover.config import (
    LLMConfig,
    OrchestratorConfig,
    ExplorationConfig,
    WorkerConfig,
    ProverConfig,
    VerifierConfig,
    ParserConfig,
    KBSummarizerConfig,
    RunConfig,
    get_config,
    list_configs,
)
from prover.runtime import run_problem

__all__ = [
    "LLMConfig",
    "OrchestratorConfig",
    "ExplorationConfig",
    "WorkerConfig",
    "ProverConfig",
    "VerifierConfig",
    "ParserConfig",
    "KBSummarizerConfig",
    "RunConfig",
    "get_config",
    "list_configs",
    "run_problem",
]
