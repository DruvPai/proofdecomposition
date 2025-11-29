"""Integration-style tests that exercise the runtime with mocked LLMs."""

from __future__ import annotations

import asyncio
from typing import Iterable

from prover.config import get_config
from prover.graph import AgentType
from prover.runtime import Runtime
from prover.schemas import SolutionAttempt
from tests.mock_llm import MockLLMRouter


def _execute(
    problem: str, config_name: str = "default"
) -> tuple[Runtime, SolutionAttempt]:
    """Convenience wrapper to execute the runtime and return normalized output."""

    runtime = Runtime(get_config(config_name))
    result = asyncio.run(runtime.run(problem))
    return runtime, result.normalized


def test_runtime_produces_single_round_solution(
    mock_llm_router: MockLLMRouter,
) -> None:
    """The mocked run should finish with an accepted solution without extra rounds."""

    runtime, normalized = _execute("Prove that 1 + 1 = 2.")
    assert isinstance(normalized, SolutionAttempt)
    assert normalized.final_answer_md, "Expected non-empty final answer."

    worker_nodes = [
        node
        for node in runtime.nodes.values()
        if node.agent_type == AgentType.WORKER
        and node.inputs.get("task", {}).get("goal") == "solve"
    ]
    assert worker_nodes, "Expected at least one solve worker."
    worker_node = worker_nodes[0]
    assert worker_node.status == "done"

    # Worker should have emitted status updates then the final solution attempt.
    assert len(worker_node.outputs) == 3
    assert isinstance(worker_node.outputs[-1].normalized, SolutionAttempt)

    # Ensure only the configured number of prover nodes (single round) were created.
    prover_children = [
        runtime.nodes[child_id]
        for child_id in worker_node.children
        if runtime.nodes[child_id].agent_type == AgentType.PROVER
    ]
    assert len(prover_children) == runtime.config.worker.num_provers

    verifier_nodes = [
        node
        for node in worker_node.children
        if runtime.nodes[node].agent_type == AgentType.VERIFIER
    ]
    assert len(verifier_nodes) == 1

    assert mock_llm_router.history, "Mock LLM should record at least one call."


def test_worker_collects_prover_attempts(
    mock_llm_router: MockLLMRouter,
) -> None:
    """Ensure prover attempts reach the worker even though orchestration stalls."""

    runtime, _ = _execute("Prove that 2 + 2 = 4.")
    worker_nodes = [
        node
        for node in runtime.nodes.values()
        if node.agent_type == AgentType.WORKER
        and node.inputs.get("task", {}).get("goal") == "solve"
    ]
    assert worker_nodes, "Expected at least one solve worker."
    worker_node = worker_nodes[0]
    prover_attempts: Iterable[SolutionAttempt] = (
        ctx.normalized
        for ctx in worker_node.inputs["local_context"]
        if ctx.agent_type == AgentType.PROVER
        and isinstance(ctx.normalized, SolutionAttempt)
    )
    attempts = list(prover_attempts)
    assert attempts, "Expected at least one SolutionAttempt routed to the worker."
    assert all(attempt.final_answer_md for attempt in attempts)


def test_mock_records_all_prompts(mock_llm_router: MockLLMRouter) -> None:
    """The router should capture every system prompt we rely on for debugging."""

    _execute("Establish that 3 + 3 = 6.")
    system_prompts = {call.system_prompt for call in mock_llm_router.history}
    prompts_map = mock_llm_router.system_prompts
    expected_prompts = {
        prompts_map["prover"],
        prompts_map["verifier"],
        prompts_map["parser"],
    }
    if get_config("default").orchestrator.exploration_rounds > 0:
        expected_prompts.add(prompts_map["exploration"])
    missing = expected_prompts - system_prompts
    assert not missing, f"Missing mocked prompts: {missing}"


def test_children_receive_parent_context(
    mock_llm_router: MockLLMRouter,
) -> None:
    """Every spawned node should receive parent outputs as local context."""

    runtime, _ = _execute("Show that 1 + 0 = 1.")
    for node in runtime.nodes.values():
        if not node.parents:
            continue
        local_context = node.inputs.get("local_context", [])
        assert len(local_context) >= len(node.parents), (
            f"Expected >= {len(node.parents)} parent context items for node {node.id}, "
            f"got {len(local_context)}."
        )
