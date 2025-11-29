"""Orchestrator agent implementation."""

from __future__ import annotations

from prover.agents.base import BaseAgent
from prover.graph import AgentNode, AgentOutput, AgentType, SpawnRequest
from prover.kb import KnowledgeBase
from prover.schemas import OrchestratorStatus, SolutionAttempt
from prover.trace import TraceLogger


class OrchestratorAgent(BaseAgent):
    """Top-level orchestrator.

    The orchestrator is a minimal state machine driven by routed `local_context`.
    It spawns exploration agents (if configured) and then a single solve worker,
    returning the worker's best accepted solution.
    """

    async def run(
        self, node: AgentNode, kb: KnowledgeBase, trace: TraceLogger
    ) -> AgentOutput:
        """Run exploration rounds (if configured) then solve via a worker.

        Args:
            node: Agent node wrapper.
            kb: Knowledge base for this run (unused directly by orchestrator).
            trace: Trace logger (unused directly by orchestrator).

        Returns:
            `AgentOutput` with either spawn requests or the final solution.
        """

        problem = node.inputs["problem"]
        local_context: list[AgentOutput] = node.inputs.get("local_context", [])

        exploration_done = [
            ctx for ctx in local_context if ctx.agent_type == AgentType.EXPLORATION
        ]
        if len(exploration_done) < self.config.orchestrator.exploration_rounds:
            next_round = len(exploration_done) + 1
            return AgentOutput(
                agent_type=AgentType.ORCHESTRATOR,
                raw_text=f"spawning exploration round {next_round}",
                normalized=OrchestratorStatus(
                    phase="exploration",
                    round_index=next_round,
                    message=f"spawning exploration round {next_round}",
                ),
                kb_writes=[],
                spawn_requests=[
                    SpawnRequest(
                        agent_type=AgentType.EXPLORATION,
                        task={"problem": problem, "round": next_round},
                        edge_from_parent=True,
                    )
                ],
            )

        worker_outputs = [
            ctx for ctx in local_context if ctx.agent_type == AgentType.WORKER
        ]
        if not worker_outputs:
            return AgentOutput(
                agent_type=AgentType.ORCHESTRATOR,
                raw_text="spawning solve worker",
                normalized=OrchestratorStatus(
                    phase="solve",
                    round_index=None,
                    message="spawning solve worker",
                ),
                kb_writes=[],
                spawn_requests=[
                    SpawnRequest(
                        agent_type=AgentType.WORKER,
                        task={"problem": problem, "goal": "solve"},
                        edge_from_parent=True,
                    )
                ],
            )

        last_worker = worker_outputs[-1]
        normalized = last_worker.normalized
        if isinstance(normalized, SolutionAttempt):
            solution = normalized
        else:
            solution = SolutionAttempt(
                final_answer_md=str(last_worker.raw_text).strip()
            )
        return AgentOutput(
            agent_type=AgentType.ORCHESTRATOR,
            raw_text=solution.final_answer_md,
            normalized=solution,
            kb_writes=[],
            spawn_requests=[],
        )
