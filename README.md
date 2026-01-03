# ProofDecomposition (`prover`)

Multi-agent, LLM-powered mathematical problem solver built around a sequential runtime, a shared in-memory knowledge base, and a small set of cooperating “agents” (orchestrator/exploration/worker/prover/verifier/parser).

`SPEC.md` is the source of truth for intended behavior; `docs/ARCHITECTURE.md` summarizes the current module layout.

## Installation

This project uses `uv` and requires Python 3.13+ (see `pyproject.toml`).

```bash
uv sync
```

## Quickstart (CLI)

1) Write a problem file (see “Problem format” below), e.g. `problem.md`.
2) Run:

```bash
uv run prover run --config default --input problem.md --output solution.md
```

Optional: record a JSONL trace and override the runtime step cap for that run:

```bash
uv run prover run --config default --input problem.md --output solution.md --trace trace.jsonl --max-steps 64
```

List available configs (registered in Python code):

```bash
uv run prover list-configs
```

## API keys / LLM providers

Runs that call real LLMs require the API key environment variable referenced by the chosen config.

- `default` uses OpenRouter and expects `OPENROUTER_API_KEY` to be set.
- You can define your own config to use OpenAI (or any OpenAI-compatible provider) by setting `LLMConfig.base_url` and `LLMConfig.api_key_env`.

Example:

```bash
export OPENROUTER_API_KEY="..."
```

If the key is missing, the run fails with a `RuntimeError` from `BaseAgent._client_for`.

## Problem format

The input problem is a UTF-8 Markdown file. There is no required schema: the runtime treats the entire file content as the problem statement and feeds it to agents verbatim (including LaTeX).

Recommended minimal `problem.md`:

```md
Prove the Pythagorean theorem: for a right triangle with legs $a,b$ and hypotenuse $c$,
we have $a^2+b^2=c^2$.
```

Notes:
- Keep the file focused on the statement; extra commentary becomes part of the prompt.
- If you want the model to assume conventions/definitions, include them explicitly in the Markdown.

## Output format

The CLI writes a Markdown report to the `--output` path with three sections:

1) `# Problem` (the original input file content)
2) `# Solution` (the accepted prover attempt, as Markdown)
3) `# KB Appendix` (a bullet list of knowledge base entries accumulated during the run)

This is implemented in `prover/runtime.py` in `run_problem`.

## Configs: how to create a new one

Configs are Python factories registered via `@register_config` in `prover/config.py`. A “config” is a `RunConfig` dataclass that wires together:
- runtime caps (orchestrator),
- agent counts/loop parameters (worker/verifier),
- LLM connections + sampling params (exploration/prover/verifier/parser/kb_summarizer),
- agent system prompts.

To add a new config:

1) Add a new factory function decorated with `@register_config` (by convention in `prover/config.py`).
2) The CLI config key is the *factory function name*; `RunConfig.name` is used for tracing (it’s convenient to keep them the same).
3) Point each `LLMConfig.api_key_env` to the environment variable you’ll set in your shell.

Example config factory (OpenAI-compatible base URL):

```py
from prover.config import (
    ExplorationConfig,
    KBSummarizerConfig,
    LLMConfig,
    OrchestratorConfig,
    ParserConfig,
    ProverConfig,
    RunConfig,
    VerifierConfig,
    WorkerConfig,
    register_config,
)

MAX_TOTAL_STEPS = 256
NUM_PROVERS = 4
VERIFY_ROUNDS = 2
MODEL = "gpt-4o-mini"  # pick any model supported by your provider

@register_config
def openai_example() -> RunConfig:
    openai_base_url = "https://api.openai.com/v1"
    prover_llm = LLMConfig(
        model=MODEL,
        api_key_env="OPENAI_API_KEY",
        base_url=openai_base_url,
        temperature=0.4,
    )
    verifier_llm = LLMConfig(
        model=MODEL,
        api_key_env="OPENAI_API_KEY",
        base_url=openai_base_url,
        temperature=0.2,
    )
    parser_llm = LLMConfig(
        model=MODEL,
        api_key_env="OPENAI_API_KEY",
        base_url=openai_base_url,
        temperature=0.0,
    )
    return RunConfig(
        name="openai_example",
        orchestrator=OrchestratorConfig(max_total_steps=MAX_TOTAL_STEPS),
        exploration=ExplorationConfig(llm=prover_llm),
        worker=WorkerConfig(num_provers=NUM_PROVERS, max_verify_rounds=VERIFY_ROUNDS),
        prover=ProverConfig(llm=prover_llm),
        verifier=VerifierConfig(ensemble_size=1, llms=[verifier_llm]),
        parser=ParserConfig(llm=parser_llm),
        kb_summarizer=KBSummarizerConfig(llm=parser_llm),
        trace_enabled=False,
    )
```

Then run:

```bash
export OPENAI_API_KEY="..."
uv run prover run --config openai_example --input problem.md --output solution.md
```

## Examples

Run the included Pythagorean example:

```bash
uv run prover run --config default --input examples/pythagorean/problem.md --output examples/pythagorean/solution.md
```

Run with tracing:

```bash
uv run prover run --config default --input examples/pythagorean/problem.md --output examples/pythagorean/solution.md --trace examples/pythagorean/trace.jsonl
```

## Using the runtime from Python

For programmatic use (e.g., notebooks), you can call `Runtime` directly:

```py
import asyncio

from prover.config import get_config
from prover.runtime import Runtime

runtime = Runtime(get_config("default"))
result = asyncio.run(runtime.run("Prove that 1 + 1 = 2."))
print(result.normalized.final_answer_md)
```

## Tracing

`--trace` writes a JSONL file with:
- `run_start` / `run_end`,
- per-agent lifecycle events,
- LLM request/response payloads (prompts and outputs).

The trace is intended for debugging; it may contain the full problem text and model outputs. API keys are not included in the recorded request payloads by default, but you should still avoid sharing traces that include sensitive content.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check prover tests
```
