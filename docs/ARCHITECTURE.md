## Architecture overview

This project implements the MVP described in `SPEC.md`: a sequential, multi-agent runtime that solves a single math problem via a context graph + shared in-memory knowledge base.

### Key modules

- `prover/runtime.py`: Sequential LIFO scheduler + context routing + orchestration entrypoint (`run_problem`).
- `prover/graph.py`: Lightweight graph/node data structures (`AgentNode`, `AgentOutput`, `SpawnRequest`).
- `prover/kb.py`: In-memory knowledge base with prompt/appendix renderers.
- `prover/agents/`: Agent implementations:
  - `orchestrator.py`: Spawns exploration rounds then a solve worker.
  - `exploration.py`: Proposes exploration questions and spawns workers.
  - `worker.py`: Prover→verifier loop + optional decomposition via plan steps.
  - `prover_agent.py`: Produces candidate solutions (LLM + parser fallback).
  - `verifier.py`: Majority-vote verifier (LLM ensemble).
  - `parser.py`: Schema-driven parsing via a parser LLM (mockable in tests).
- `prover/llm.py`: OpenAI-compatible `httpx` client (OpenRouter-ready).
- `prover/trace.py`: JSONL trace logger.

### Execution model (MVP)

- A run begins by spawning an orchestrator node.
- The runtime maintains a LIFO stack of runnable nodes and executes exactly one node at a time.
- Child outputs are routed back to parents through `local_context`; parents resume once all dependent children are done.
- The worker agent runs the generation→verification loop and returns a `SolutionAttempt` when accepted.

