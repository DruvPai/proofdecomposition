# ProofDecomposition (`prover`)

Multi-agent, LLM-powered mathematical problem solver.

## Quickstart

```
uv run prover run --config default --input problem.md --output solution.md
```

With tracing + a custom step cap:

```
uv run prover run --config default --input problem.md --output solution.md --trace trace.jsonl --max-steps 64
```

List configs:

```
uv run prover list-configs
```

## Development

```
uv run pytest
uv run ruff check .
uv run ruff format .
uv run ty check prover tests
```

See `docs/ARCHITECTURE.md` for module layout.

## Features (MVP)
- Sequential LIFO scheduler with a context graph.
- Agents: orchestrator, exploration, worker, prover, verifier, parser.
- In-memory knowledge base shared across agents.
- OpenRouter-ready LLM client (OpenAI-compatible HTTP via `httpx`); deterministic fake client for offline use.
- Parser agent supports schema-driven LLM parsing: calling agents supply the JSON Schema and receive typed outputs with deterministic fallbacks.
- Majority-vote verifier per SPEC; final output includes KB appendix.
- JSONL tracing via `--trace` (prompts/responses recorded, API keys redacted).
