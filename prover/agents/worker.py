"""Worker agent implementation."""

from __future__ import annotations

from prover.agents.base import BaseAgent
from prover.agents.text import make_kb_entry, prepare_kb_entries
from prover.graph import AgentNode, AgentOutput, AgentType, SpawnRequest
from prover.kb import KnowledgeBase
from prover.schemas import KBEntry, SolutionAttempt, VerificationReport, WorkerStatus
from prover.trace import TraceLogger


class WorkerAgent(BaseAgent):
    """Worker agent orchestrating prover and verifier subagents."""

    @staticmethod
    def _status(
        phase: str,
        round_index: int,
        provers_spawned: int,
        *,
        verifier_spawned: bool = False,
        decomposition_triggered: bool = False,
        feedback_md: str | None = None,
        notes: str | None = None,
    ) -> WorkerStatus:
        """Build a standardized worker status payload."""

        return WorkerStatus(
            phase=phase,
            round_index=round_index,
            provers_spawned=provers_spawned,
            verifier_spawned=verifier_spawned,
            decomposition_triggered=decomposition_triggered,
            feedback_md=feedback_md,
            notes=notes,
        )

    async def run(
        self, node: AgentNode, kb: KnowledgeBase, trace: TraceLogger
    ) -> AgentOutput:
        """Solve a (sub-)problem using a generation→verification loop."""

        local_context: list[AgentOutput] = node.inputs.get("local_context", [])
        task = node.inputs.get("task", {})
        problem = task.get("problem", node.inputs.get("problem", ""))
        goal = task.get("goal", "solve")

        decomposition_depth_raw = task.get("decomposition_depth", 0)
        try:
            decomposition_depth = int(decomposition_depth_raw)
        except (TypeError, ValueError):
            decomposition_depth = 0

        pv_outputs = [
            ctx
            for ctx in local_context
            if ctx.agent_type in (AgentType.PROVER, AgentType.VERIFIER)
        ]
        verifier_positions = [
            i
            for i, ctx in enumerate(pv_outputs)
            if ctx.agent_type == AgentType.VERIFIER
        ]
        verifier_outputs = [pv_outputs[i] for i in verifier_positions]
        rounds_completed = len(verifier_outputs)

        last_verifier_pos = verifier_positions[-1] if verifier_positions else -1
        in_progress_provers = [
            ctx
            for ctx in pv_outputs[last_verifier_pos + 1 :]
            if ctx.agent_type == AgentType.PROVER
            and isinstance(ctx.normalized, SolutionAttempt)
        ]

        if pv_outputs and pv_outputs[-1].agent_type == AgentType.VERIFIER:
            prev_verifier_pos = (
                verifier_positions[-2] if len(verifier_positions) >= 2 else -1
            )
            just_verified_provers = [
                ctx
                for ctx in pv_outputs[prev_verifier_pos + 1 : last_verifier_pos]
                if ctx.agent_type == AgentType.PROVER
                and isinstance(ctx.normalized, SolutionAttempt)
            ]
        else:
            just_verified_provers = []

        latest_is_verifier = (
            bool(pv_outputs) and pv_outputs[-1].agent_type == AgentType.VERIFIER
        )
        verifier_report = verifier_outputs[-1].normalized if verifier_outputs else None

        if latest_is_verifier and isinstance(verifier_report, VerificationReport):
            if (
                verifier_report.accepted
                and verifier_report.best_attempt_index is not None
            ):
                best_attempt = just_verified_provers[
                    verifier_report.best_attempt_index
                ].normalized

                kb_writes: list[KBEntry] = []
                for entry in best_attempt.kb_updates:
                    kb_writes.append(entry)
                if not kb_writes:
                    kb_writes.append(
                        make_kb_entry(
                            node.id, "Auto-generated fact", best_attempt.final_answer_md
                        )
                    )

                kb_writes = prepare_kb_entries(kb_writes)
                best_attempt.kb_updates = kb_writes
                return AgentOutput(
                    agent_type=AgentType.WORKER,
                    raw_text=best_attempt.final_answer_md,
                    normalized=best_attempt,
                    kb_writes=kb_writes,
                    spawn_requests=[],
                )

            if rounds_completed >= self.config.worker.max_verify_rounds:
                failure_md = f"Verifier could not confirm a solution for: {problem}"
                return AgentOutput(
                    agent_type=AgentType.WORKER,
                    raw_text=failure_md,
                    normalized=SolutionAttempt(final_answer_md=failure_md),
                    kb_writes=[],
                    spawn_requests=[],
                )

            can_decompose = (
                self.config.worker.allow_decomposition
                and decomposition_depth < self.config.worker.max_decomposition_depth
                and goal == "solve"
            )
            last_verifier_idx_lc = max(
                (
                    i
                    for i, ctx in enumerate(local_context)
                    if ctx.agent_type == AgentType.VERIFIER
                ),
                default=-1,
            )
            decomp_outputs = [
                ctx
                for ctx in local_context[last_verifier_idx_lc + 1 :]
                if ctx.agent_type == AgentType.WORKER
            ]
            if can_decompose and not decomp_outputs:
                plan_attempt = None
                if verifier_report.best_attempt_index is not None:
                    candidate = just_verified_provers[
                        verifier_report.best_attempt_index
                    ].normalized
                    if candidate.outline_steps:
                        plan_attempt = candidate
                if plan_attempt is None:
                    for prov in just_verified_provers:
                        if prov.normalized.outline_steps:
                            plan_attempt = prov.normalized
                            break
                if plan_attempt is not None and plan_attempt.outline_steps:
                    step_spawns = [
                        SpawnRequest(
                            agent_type=AgentType.WORKER,
                            task={
                                "problem": step,
                                "goal": "decompose_step",
                                "step": step,
                                "parent_problem": problem,
                                "decomposition_depth": decomposition_depth + 1,
                            },
                        )
                        for step in plan_attempt.outline_steps[
                            : self.config.worker.max_plan_steps
                        ]
                    ]
                    return AgentOutput(
                        agent_type=AgentType.WORKER,
                        raw_text="spawning decomposition steps",
                        normalized=self._status(
                            phase="decomposition",
                            round_index=rounds_completed,
                            provers_spawned=len(just_verified_provers),
                            decomposition_triggered=True,
                            feedback_md=verifier_report.global_feedback_md,
                            notes="spawning decomposition steps",
                        ),
                        kb_writes=[],
                        spawn_requests=step_spawns,
                    )

            feedback_md = verifier_report.global_feedback_md
            spawns = [
                SpawnRequest(
                    agent_type=AgentType.PROVER,
                    task={"problem": problem, "goal": goal, "feedback_md": feedback_md},
                )
                for _ in range(self.config.worker.num_provers)
            ]
            return AgentOutput(
                agent_type=AgentType.WORKER,
                raw_text="starting next prover round",
                normalized=self._status(
                    phase="prover_generation",
                    round_index=rounds_completed,
                    provers_spawned=self.config.worker.num_provers,
                    feedback_md=feedback_md,
                    notes="starting next prover round",
                ),
                kb_writes=[],
                spawn_requests=spawns,
            )

        expected_provers = self.config.worker.num_provers
        if len(in_progress_provers) < expected_provers:
            feedback_md = None
            if verifier_outputs:
                last_report = verifier_outputs[-1].normalized
                if isinstance(last_report, VerificationReport):
                    feedback_md = last_report.global_feedback_md
            missing = expected_provers - len(in_progress_provers)
            spawns = [
                SpawnRequest(
                    agent_type=AgentType.PROVER,
                    task={"problem": problem, "goal": goal, "feedback_md": feedback_md},
                )
                for _ in range(missing)
            ]
            return AgentOutput(
                agent_type=AgentType.WORKER,
                raw_text="spawning provers",
                normalized=self._status(
                    phase="prover_generation",
                    round_index=rounds_completed,
                    provers_spawned=missing,
                    feedback_md=feedback_md,
                    notes="spawning provers",
                ),
                kb_writes=[],
                spawn_requests=spawns,
            )

        if (
            pv_outputs
            and pv_outputs[-1].agent_type == AgentType.PROVER
            and len(in_progress_provers) == expected_provers
        ):
            verifier_task = {"problem": problem, "prover_outputs": in_progress_provers}
            return AgentOutput(
                agent_type=AgentType.WORKER,
                raw_text="spawning verifier",
                normalized=self._status(
                    phase="verification",
                    round_index=rounds_completed,
                    provers_spawned=len(in_progress_provers),
                    verifier_spawned=True,
                    notes="spawning verifier",
                ),
                kb_writes=[],
                spawn_requests=[
                    SpawnRequest(agent_type=AgentType.VERIFIER, task=verifier_task)
                ],
            )

        failure_md = f"Worker internal error (no verifier report) for: {problem}"
        return AgentOutput(
            agent_type=AgentType.WORKER,
            raw_text=failure_md,
            normalized=SolutionAttempt(final_answer_md=failure_md),
            kb_writes=[],
            spawn_requests=[],
        )
