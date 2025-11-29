"""Configuration registry and dataclasses for the prover system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from prover.constants import (
    DEFAULT_EXPLORATION_MAX_QUESTIONS,
    DEFAULT_EXPLORATION_ROUNDS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_P,
    DEFAULT_MAX_TOTAL_STEPS,
    DEFAULT_VERIFIER_ENSEMBLE_SIZE,
    DEFAULT_WORKER_MAX_DECOMPOSITION_DEPTH,
    DEFAULT_WORKER_MAX_PLAN_STEPS,
    DEFAULT_WORKER_MAX_VERIFY_ROUNDS,
    DEFAULT_WORKER_NUM_PROVERS,
)


@dataclass
class LLMConfig:
    """LLM configuration for OpenAI-compatible chat API."""

    model: str
    api_key_env: str
    base_url: str | None = None
    temperature: float = DEFAULT_LLM_TEMPERATURE
    top_p: float = DEFAULT_LLM_TOP_P
    max_tokens: int | None = None


@dataclass
class OrchestratorConfig:
    """Orchestrator parameters."""

    exploration_rounds: int = DEFAULT_EXPLORATION_ROUNDS
    max_total_steps: int = DEFAULT_MAX_TOTAL_STEPS


@dataclass
class ExplorationConfig:
    """Exploration agent parameters."""

    max_questions: int = DEFAULT_EXPLORATION_MAX_QUESTIONS
    llm: LLMConfig | None = None
    system_prompt: str = """
You are a mathematical genius. You are given a problem and a (possibly empty) knowledge base of things you have already discovered about this problem. Your job is to understand more about the problem by proving intermediate results related to the setting of the problem. To this end, please propose up to {max_questions} independent helpful questions or conjectures that could make some progress towards a complete understanding of the setting of the problem.

The format of your response should be as follows (omitting the ``` markers):

```
Questions:
1) ...
...
N) ...
```
    """.strip()


@dataclass
class WorkerConfig:
    """Worker agent parameters."""

    num_provers: int = DEFAULT_WORKER_NUM_PROVERS
    max_verify_rounds: int = DEFAULT_WORKER_MAX_VERIFY_ROUNDS
    allow_decomposition: bool = True
    max_plan_steps: int = DEFAULT_WORKER_MAX_PLAN_STEPS
    max_decomposition_depth: int = DEFAULT_WORKER_MAX_DECOMPOSITION_DEPTH


@dataclass
class ProverConfig:
    """Prover agent parameters."""

    llm: LLMConfig | None = None
    system_prompt: str = """
You are a mathematical genius. You are given a problem and a (possibly empty) knowledge base of things you have already discovered about this problem. If you have any uncertainty about the problem, propose a plan to solve it that can be delegated to your PhD students. If you are confident about the problem, solve it and write a complete, rigorous proof. If the problem asks you to prove a conclusion that is false, state it and give a proof for why it is false, or a counterexample. Your output should be a Markdown document with the following format (omitting the ``` markers).

If you will give a PLAN, the format of your response should be as follows:
```
Output type: Plan

Plan:
- Step 1: ...
...
- Step N: ...
```

If you will give a SOLUTION, the format of your response should be as follows:
```
Output type: Solution

Solution:
...
```

If you believe the problem conclusion is false, the format of your response should be as follows:
```
Output type: Error

Reason:
...
```
    """.strip()


@dataclass
class VerifierConfig:
    """Verifier agent parameters."""

    ensemble_size: int = DEFAULT_VERIFIER_ENSEMBLE_SIZE
    llms: list[LLMConfig] | None = None
    system_prompt: str = """
You are a mathematical genius. You are given a proof attempt and a (possibly empty) knowledge base of things you have already discovered about this problem. Your job is to determine if the proof is correct and (reasonably) complete, and provide thorough and complete reasoning for your verdict. The format of your response is as follows (omitting the ``` markers):

```
Verdict: [Correct, Incorrect]

Reason:
...
```

If any essential detail is missing, answer Incorrect (regardless of whether or not _you_ can fill in the missing detail), but do not nit-pick on non-essential details.    
    """.strip()


@dataclass
class ParserConfig:
    """Parser agent parameters."""

    llm: LLMConfig | None = None
    system_prompt: str = """
You are a parser agent. You are given a semi-structured Markdown document and a schema. Your job is to parse the document into the schema. You should copy-and-paste the original Markdown verbatim into the output as much as possible (the best case is that you copy-paste the entire document, minus the headers which indicate the structure). ONLY output the JSON string, nothing else.
    """.strip()


@dataclass
class KBSummarizerConfig:
    """KB summarizer configuration.

    The KB should store succinct *statements* rather than full proofs. This
    config controls an optional LLM call that distills a proof attempt or draft
    KB entry into a compact statement suitable for prompting.
    """

    llm: LLMConfig | None = None
    system_prompt: str = """
You are a mathematical knowledge-base summarizer.

Given an input that may contain a proof or long explanation, produce a concise
KB entry capturing the *statement/result* (not the proof).

Return a JSON object with keys:
- "title": short title (no more than ~1 line)
- "statement_md": Markdown statement of the result (1-3 sentences; may include LaTeX)

Only output JSON.
    """.strip()


@dataclass
class RunConfig:
    """Top-level run configuration."""

    name: str
    orchestrator: OrchestratorConfig
    exploration: ExplorationConfig
    worker: WorkerConfig
    prover: ProverConfig
    verifier: VerifierConfig
    parser: ParserConfig
    kb_summarizer: KBSummarizerConfig
    trace_enabled: bool = False


class ConfigFactory(Protocol):
    """Callable that constructs a RunConfig."""

    __name__: str

    def __call__(self) -> RunConfig: ...


_CONFIG_REGISTRY: dict[str, ConfigFactory] = {}


def register_config(func: ConfigFactory) -> ConfigFactory:
    """Decorator to register a config factory by function name."""

    _CONFIG_REGISTRY[func.__name__] = func
    return func


def get_config(name: str) -> RunConfig:
    """Retrieve a configuration by name."""

    if name not in _CONFIG_REGISTRY:
        available = ", ".join(sorted(_CONFIG_REGISTRY))
        raise KeyError(f"Config '{name}' not found. Available: {available}")
    return _CONFIG_REGISTRY[name]()


def list_configs() -> list[str]:
    """List available configuration names."""

    return sorted(_CONFIG_REGISTRY)


def _make_openrouter_llm(model: str, temperature: float) -> LLMConfig:
    """Convenience helper for OpenRouter (OpenAI-compatible) configs."""

    return LLMConfig(
        model=model,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


@register_config
def default() -> RunConfig:
    """Default configuration using deterministic placeholder LLM settings."""

    exploration_temperature = 0.7
    prover_temperature = 0.4
    verifier_temperature = 0.2
    parser_temperature = 0.0

    exploration_llm = _make_openrouter_llm(
        "openai/gpt-5.2-pro", exploration_temperature
    )
    prover_llm = _make_openrouter_llm("openai/gpt-5.2", prover_temperature)
    verifier_llms = [
        # _make_openrouter_llm("openai/gpt-5.2-pro", verifier_temperature),
        _make_openrouter_llm("google/gemini-3-pro-preview", verifier_temperature),
        # _make_openrouter_llm("anthropic/claude-4.5-sonnet", verifier_temperature),
    ]
    parser_llm = _make_openrouter_llm("openai/gpt-5-mini", parser_temperature)

    exploration_rounds = 2
    max_total_steps = 128
    max_questions = 2
    num_provers = 2
    max_verify_rounds = 1
    return RunConfig(
        name="default",
        orchestrator=OrchestratorConfig(
            exploration_rounds=exploration_rounds,
            max_total_steps=max_total_steps,
        ),
        exploration=ExplorationConfig(max_questions=max_questions, llm=exploration_llm),
        worker=WorkerConfig(
            num_provers=num_provers,
            max_verify_rounds=max_verify_rounds,
            allow_decomposition=True,
        ),
        prover=ProverConfig(llm=prover_llm),
        verifier=VerifierConfig(ensemble_size=len(verifier_llms), llms=verifier_llms),
        parser=ParserConfig(llm=parser_llm),
        kb_summarizer=KBSummarizerConfig(llm=parser_llm),
        trace_enabled=False,
    )
