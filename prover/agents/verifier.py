"""Verifier agent implementation."""

from __future__ import annotations

import asyncio
import re

from prover.agents.base import BaseAgent
from prover.graph import AgentNode, AgentOutput, AgentType
from prover.kb import KnowledgeBase
from prover.schemas import SolutionAttempt, VerificationReport
from prover.trace import TraceLogger


class VerifierAgent(BaseAgent):
    """Verifier agent applying majority vote acceptance."""

    @staticmethod
    def parse_verifier_vote(text: str) -> tuple[bool | None, str]:
        """Parse a verifier vote response into (verdict, reason).

        Args:
            text: Raw verifier output.

        Returns:
            A tuple `(verdict, reason)` where verdict is True/False/None.
        """

        verdict_match = re.search(
            r"verdict:\s*(correct|incorrect)", text, re.IGNORECASE
        )
        verdict: bool | None = None
        if verdict_match:
            verdict = verdict_match.group(1).lower() == "correct"
        reason_match = re.search(r"reason\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else text.strip()
        return verdict, reason

    async def run(
        self, node: AgentNode, kb: KnowledgeBase, trace: TraceLogger
    ) -> AgentOutput:
        """Score prover attempts using an ensemble of LLM votes; majority acceptance."""

        local_context: list[AgentOutput] = node.inputs.get("local_context", [])
        task_provers = node.inputs.get("task", {}).get("prover_outputs")
        prover_outputs: list[AgentOutput] = (
            task_provers
            if isinstance(task_provers, list) and task_provers
            else [ctx for ctx in local_context if ctx.agent_type == AgentType.PROVER]
        )

        attempts: list[SolutionAttempt] = [
            ctx.normalized
            for ctx in prover_outputs
            if isinstance(ctx.normalized, SolutionAttempt)
        ]

        ensemble_size = max(1, self.config.verifier.ensemble_size)
        scores: list[int] = []
        critiques: list[str] = []
        verifier_llms = self.config.verifier.llms
        if not verifier_llms:
            raise RuntimeError("Verifier LLM configurations must be provided.")

        problem_text = node.inputs.get("task", {}).get(
            "problem", node.inputs.get("problem", "")
        )
        for attempt in attempts:
            vote_yes = 0
            critique_parts: list[str] = []
            user_prompt = f"""
Problem:
{problem_text}

Context hierarchy:
{node.inputs.get("context_hierarchy_md") or "None."}

KB:
{kb.render_prompt_md()}

Solution attempt:
{attempt.final_answer_md}""".strip()
            vote_tasks = [
                self._chat_with_trace(
                    llm=conf,
                    trace=trace,
                    node_id=node.id,
                    system_prompt=self.config.verifier.system_prompt,
                    user_prompt=user_prompt,
                )
                for conf in verifier_llms[:ensemble_size]
            ]
            responses = await asyncio.gather(*vote_tasks)
            for resp in responses:
                content_raw = resp["choices"][0]["message"].get("content", "")
                verdict, reason = self.parse_verifier_vote(content_raw)
                if verdict is True:
                    vote_yes += 1
                critique_parts.append(reason.strip() if reason else content_raw.strip())
            scores.append(vote_yes)
            critiques.append(" | ".join(critique_parts))

        if attempts:
            best_idx = max(range(len(attempts)), key=lambda i: (scores[i], -i))
            accepted = scores[best_idx] > (ensemble_size / 2)
        else:
            best_idx = None
            accepted = False

        report = VerificationReport(
            accepted=accepted,
            best_attempt_index=best_idx,
            attempt_scores=scores,
            attempt_critiques_md=critiques,
            global_feedback_md="Deterministic verifier majority rule.",
        )
        return AgentOutput(
            agent_type=AgentType.VERIFIER,
            raw_text="accepted" if accepted else "rejected",
            normalized=report,
            kb_writes=[],
            spawn_requests=[],
        )
