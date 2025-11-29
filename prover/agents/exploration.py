"""Exploration agent implementation."""

from __future__ import annotations

from typing import Any

from prover.agents.base import BaseAgent
from prover.agents.parser import ParserAgent
from prover.agents.text import (
    extract_result_snippet,
    format_local_context,
    prepare_kb_entries,
)
from prover.graph import AgentNode, AgentOutput, AgentType, SpawnRequest
from prover.kb import KnowledgeBase
from prover.schemas import ExplorationQuestions, KBEntry, KBKind, SolutionAttempt
from prover.trace import TraceLogger


class ExplorationAgent(BaseAgent):
    """Exploration agent that proposes questions and delegates them to workers."""

    @staticmethod
    def _exploration_schema() -> dict[str, Any]:
        """Return the JSON schema used to parse exploration questions."""

        return {
            "type": "object",
            "properties": {
                "questions": {"type": "array", "items": {"type": "string"}},
                "rationales_md": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["questions", "rationales_md"],
        }

    @staticmethod
    def parse_exploration_questions(
        text: str, *, max_questions: int
    ) -> ExplorationQuestions:
        """Parse a Markdown-ish bullet list into `ExplorationQuestions`.

        Args:
            text: Model output.
            max_questions: Maximum number of questions to return.

        Returns:
            Parsed `ExplorationQuestions`.
        """

        questions: list[str] = []
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned[0] in "-*":
                cleaned = cleaned[1:].strip()
            questions.append(cleaned)
        if not questions and text.strip():
            questions = [text.strip()]

        questions = questions[:max_questions]
        rationales_md = ["" for _ in range(len(questions))]
        return ExplorationQuestions(questions=questions, rationales_md=rationales_md)

    @staticmethod
    def coerce_exploration_questions(
        data: dict[str, Any], *, max_questions: int
    ) -> ExplorationQuestions:
        """Convert generic JSON data into `ExplorationQuestions`."""

        questions_raw = list(data.get("questions", []) or [])
        rationales_raw = list(data.get("rationales_md", []) or [])
        questions: list[str] = [str(q).strip() for q in questions_raw if str(q).strip()]
        questions = questions[:max_questions]
        rationales: list[str] = []
        for idx, question in enumerate(questions):
            if idx < len(rationales_raw) and str(rationales_raw[idx]).strip():
                rationales.append(str(rationales_raw[idx]).strip())
            else:
                rationales.append(f"Rationale for: {question}")
        return ExplorationQuestions(questions=questions, rationales_md=rationales)

    @staticmethod
    def _build_worker_spawns(
        questions: list[str],
        rationales: list[str],
        problem: str,
        *,
        round_index: int | None,
    ) -> list[SpawnRequest]:
        """Create worker spawn requests for each exploration question.

        Args:
            questions: Exploration questions to delegate.
            rationales: Rationales for each question.
            problem: Root problem statement for context propagation.
            round_index: Optional exploration round index (1-based).

        Returns:
            Spawn requests ordered so that the earliest question executes first.
        """

        total_questions = len(questions)
        spawns: list[SpawnRequest] = []
        for idx, question in enumerate(questions, start=1):
            rationale = (
                rationales[idx - 1]
                if idx - 1 < len(rationales)
                else "LLM-proposed question"
            )
            task: dict[str, Any] = {
                "problem": question,
                "goal": "explore",
                "parent_problem": problem,
                "question_index": idx,
                "total_questions": total_questions,
                "question_rationale_md": rationale,
            }
            if round_index is not None:
                task["exploration_round"] = round_index
            spawns.append(SpawnRequest(agent_type=AgentType.WORKER, task=task))

        spawns.reverse()
        return spawns

    async def run(
        self,
        node: AgentNode,
        kb: KnowledgeBase,
        trace: TraceLogger,
    ) -> AgentOutput:
        """Propose exploration questions, then wait for worker answers.

        Args:
            node: Agent node wrapper.
            kb: Knowledge base for context.
            trace: Trace logger.

        Returns:
            Either (a) spawn requests for worker agents, or (b) a finalized set of
            KB writes encoding the question/answer pairs.
        """

        problem = node.inputs["task"].get("problem", node.inputs["problem"])
        local_context: list[AgentOutput] = node.inputs.get("local_context", [])
        worker_outputs = [
            ctx for ctx in local_context if ctx.agent_type == AgentType.WORKER
        ]

        # Phase 2: finalize once workers have completed.
        if worker_outputs:
            metadata = node.inputs.get("exploration_metadata")
            questions = (
                list(metadata.get("questions", []))
                if isinstance(metadata, dict)
                else []
            )
            rationales = (
                list(metadata.get("rationales_md", []))
                if isinstance(metadata, dict)
                else []
            )
            if not questions:
                for out in node.outputs:
                    if isinstance(out.normalized, ExplorationQuestions):
                        questions = list(out.normalized.questions)
                        rationales = list(out.normalized.rationales_md)
                        break

            kb_writes: list[KBEntry] = []
            for idx, question in enumerate(questions, start=1):
                rationale = (
                    rationales[idx - 1]
                    if idx - 1 < len(rationales)
                    else "LLM-proposed question"
                )
                worker_out = (
                    worker_outputs[idx - 1] if idx - 1 < len(worker_outputs) else None
                )
                if worker_out and isinstance(worker_out.normalized, SolutionAttempt):
                    answer_md = extract_result_snippet(
                        worker_out.normalized.final_answer_md
                    )
                elif worker_out:
                    answer_md = worker_out.raw_text.strip()
                else:
                    answer_md = ""
                if not answer_md:
                    answer_md = "No worker response was produced."

                clean_question = question.strip()
                rationale_section = (
                    f"\n\n**Rationale:** {rationale.strip()}"
                    if rationale and rationale.strip()
                    else ""
                )
                content_md = (
                    f"**Question {idx}:** {clean_question}"
                    f"{rationale_section}\n\n**Answer:**\n{answer_md.strip()}"
                ).strip()
                kb_writes.append(
                    KBEntry(
                        id=f"Exploration {node.id}.{idx}",
                        kind=KBKind.RESULT,
                        title=f"Exploration Q{idx}: {clean_question}",
                        content_md=content_md,
                        tags=["exploration"],
                        sources=[f"agent-{node.id}"],
                    )
                )

            normalized = ExplorationQuestions(
                questions=questions, rationales_md=rationales
            )
            if len(worker_outputs) > len(questions):
                for extra_idx, worker_out in enumerate(
                    worker_outputs[len(questions) :], start=len(questions) + 1
                ):
                    if isinstance(worker_out.normalized, SolutionAttempt):
                        extra_answer = extract_result_snippet(
                            worker_out.normalized.final_answer_md
                        )
                    else:
                        extra_answer = worker_out.raw_text.strip()
                    if not extra_answer:
                        extra_answer = "No worker response was produced."
                    kb_writes.append(
                        KBEntry(
                            id=f"Exploration {node.id}.{extra_idx}",
                            kind=KBKind.RESULT,
                            title=f"Exploration extra answer {extra_idx}",
                            content_md=extra_answer,
                            tags=["exploration"],
                            sources=[f"agent-{node.id}"],
                        )
                    )

            return AgentOutput(
                agent_type=AgentType.EXPLORATION,
                raw_text="exploration complete",
                normalized=normalized,
                kb_writes=prepare_kb_entries(kb_writes),
                spawn_requests=[],
            )

        # Phase 1: propose questions then spawn workers.
        exploration_llm = self.config.exploration.llm
        if exploration_llm is None:
            raise RuntimeError("Exploration LLM configuration must be provided.")

        response = await self._chat_with_trace(
            llm=exploration_llm,
            trace=trace,
            node_id=node.id,
            system_prompt=self.config.exploration.system_prompt.format(
                max_questions=self.config.exploration.max_questions
            ),
            user_prompt=f"""
Problem:
{problem}

Context hierarchy:
{node.inputs.get("context_hierarchy_md") or "None."}

Local context:
{format_local_context(local_context) or "None."}

KB:
{kb.render_prompt_md()}""".strip(),
            tools=self.tool_defs(),
        )

        choice = response["choices"][0]["message"]
        tool_calls = choice.get("tool_calls", [])
        _, _, finish_text = self.parse_tool_calls(tool_calls)

        content_text = finish_text or choice.get("content", "") or ""
        parsed = await ParserAgent.llm_parse_text(
            self.config,
            trace=trace,
            node_id=node.id,
            target="exploration_questions",
            text=content_text,
            schema=self._exploration_schema(),
            max_questions=self.config.exploration.max_questions,
        )
        if parsed is not None:
            normalized = self.coerce_exploration_questions(
                parsed, max_questions=self.config.exploration.max_questions
            )
        else:
            normalized = self.parse_exploration_questions(
                content_text, max_questions=self.config.exploration.max_questions
            )

        node.inputs["exploration_metadata"] = {
            "questions": list(normalized.questions),
            "rationales_md": list(normalized.rationales_md),
        }

        task_payload = node.inputs.get("task", {})
        raw_round = task_payload.get("round")
        try:
            round_index = int(raw_round) if raw_round is not None else None
        except (TypeError, ValueError):
            round_index = None

        spawn_requests = self._build_worker_spawns(
            normalized.questions,
            normalized.rationales_md,
            problem,
            round_index=round_index,
        )

        return AgentOutput(
            agent_type=AgentType.EXPLORATION,
            raw_text=content_text,
            normalized=normalized,
            kb_writes=[],
            spawn_requests=spawn_requests,
        )
