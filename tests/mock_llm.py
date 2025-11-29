"""Test-only LLM mocks keyed by configuration system prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from prover.config import RunConfig

Message = dict[str, Any]
ToolDef = dict[str, Any]


@dataclass
class RecordedCall:
    """Record of a single mock LLM invocation."""

    system_prompt: str
    user_prompt: str
    model: str
    tools_requested: bool
    response_format: dict[str, Any] | None


class MockLLMRouter:
    """Router that fabricates deterministic responses for each agent type."""

    def __init__(self, config: RunConfig) -> None:
        """Initialize router with formatted prompts."""

        self.config = config
        self.history: list[RecordedCall] = []
        self._exploration_prompt = config.exploration.system_prompt.format(
            max_questions=config.exploration.max_questions
        )
        self._prover_prompt = config.prover.system_prompt
        self._verifier_prompt = config.verifier.system_prompt
        self._parser_prompt = config.parser.system_prompt

    @property
    def system_prompts(self) -> dict[str, str]:
        """Return the formatted system prompts used for routing."""

        return {
            "exploration": self._exploration_prompt,
            "prover": self._prover_prompt,
            "verifier": self._verifier_prompt,
            "parser": self._parser_prompt,
        }

    def respond(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolDef] | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return a mocked OpenAI-style response."""

        system_prompt = messages[0]["content"] if messages else ""
        user_prompt = messages[1]["content"] if len(messages) >= 2 else ""
        self.history.append(
            RecordedCall(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                tools_requested=bool(tools),
                response_format=response_format,
            )
        )

        if response_format and response_format.get("type") == "json_object":
            return self._json_response(user_prompt)

        if system_prompt == self._exploration_prompt:
            return self._exploration_response()

        if system_prompt == self._prover_prompt:
            return self._prover_response(user_prompt)

        if system_prompt == self._verifier_prompt:
            return self._verifier_response()

        if system_prompt == self._parser_prompt:
            return self._json_response(user_prompt)

        raise ValueError(f"Unrecognized system prompt: {system_prompt!r}")

    def _json_response(self, user_prompt: str) -> dict[str, Any]:
        """Produce JSON parsing output for parser invocations."""

        markdown_match = re.search(r"```(.*?)```", user_prompt, re.DOTALL)
        markdown = markdown_match.group(1).strip() if markdown_match else ""

        if "statement_md" in user_prompt:
            data = self._summarize_kb_entry(markdown)
        elif "questions" in user_prompt:
            data = self._parse_questions(markdown)
        elif "final_answer_md" in user_prompt:
            data = self._parse_solution(markdown)
        else:
            data = {}

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(data),
                        "tool_calls": [],
                    },
                }
            ]
        }

    def _summarize_kb_entry(self, markdown: str) -> dict[str, Any]:
        """Summarize a draft KB entry into a title + statement."""

        cleaned = markdown.strip()
        if not cleaned:
            cleaned = "Mock statement."
        first_line = cleaned.splitlines()[0].strip() if cleaned else "Mock result"
        title = first_line[:80] if first_line else "Mock result"
        statement = cleaned if len(cleaned) <= 200 else f"{cleaned[:199].rstrip()}…"
        return {"title": title, "statement_md": statement}

    def _parse_questions(self, markdown: str) -> dict[str, Any]:
        """Coerce bullet list Markdown into exploration questions."""

        questions: list[str] = []
        for line in markdown.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned[0] in "-*":
                cleaned = cleaned[1:].strip()
            questions.append(cleaned)
        if not questions and markdown.strip():
            questions = [markdown.strip()]
        rationales = [f"Mock rationale {i + 1}" for i in range(len(questions))]
        return {"questions": questions, "rationales_md": rationales}

    def _parse_solution(self, markdown: str) -> dict[str, Any]:
        """Wrap markdown into a solution attempt payload."""

        outline = [
            line.strip("-* ").strip()
            for line in markdown.splitlines()
            if line.lower().startswith("step")
        ]
        return {
            "final_answer_md": markdown.strip(),
            "outline_steps": outline,
            "kb_updates": [],
            "claims_incorrect_conclusion": False,
        }

    def _exploration_response(self) -> dict[str, Any]:
        """Return bullet list questions for exploration agent."""

        content = (
            "- Consider the foundational axioms needed for the statement.\n"
            "- Compare the problem to known lemmas about successor functions."
        )
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [],
                    },
                }
            ]
        }

    def _prover_response(self, user_prompt: str) -> dict[str, Any]:
        """Emit a finish tool call with a deterministic solution."""

        problem_match = re.search(r"Problem:\s*(.*?)\n\n", user_prompt, re.DOTALL)
        problem_text = (
            problem_match.group(1).strip()
            if problem_match
            else "Solve the stated problem."
        )
        solution = (
            "Output type: Solution\n\n"
            "Solution:\n"
            f"The problem states: {problem_text}. "
            "A mock proof verifies the conclusion directly."
        )
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "finish_mock",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps({"output_text": solution}),
                                },
                            }
                        ],
                    },
                }
            ]
        }

    def _verifier_response(self) -> dict[str, Any]:
        """Provide a deterministic verifier acceptance."""

        content = "Verdict: Correct\n\nReason:\nMock verifier accepted the proof."
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [],
                    },
                }
            ]
        }


class MockLLMClient:
    """Client wrapper that delegates to the shared router."""

    def __init__(self, router: MockLLMRouter, model: str) -> None:
        """Initialize the client for a specific model."""

        self._router = router
        self._model = model

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the router's crafted response."""

        return self._router.respond(self._model, messages, tools, response_format)

    async def aclose(self) -> None:  # pragma: no cover - trivial
        """No-op close for compatibility with real clients."""

        return
