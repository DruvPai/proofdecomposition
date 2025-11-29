# ProofDecomposition / `prover` — Specification

This document defines the target behavior, core abstractions, and interfaces for a **multi-agent** LLM-powered system that decomposes and solves mathematical problems.

The repo includes this SPEC and an MVP implementation; ongoing implementation work should continue to follow this SPEC.

## 1. Scope

### 1.1 Goals (MVP)

1) Provide a Python package (`prover`) with a CLI to:
   - read a problem from a Markdown file,
   - run a multi-agent solve pipeline under a named configuration,
   - write a final solution/proof to an output Markdown file,
   - optionally write a machine-readable trace for debugging/analysis.
2) Implement a multi-agent runtime with:
   - a mutable **context graph** (agents as vertices, context edges),
   - a **knowledge base** shared by all agents,
   - deterministic scheduling semantics (single “active agent” at a time),
   - support for “sub-agents” spawned by parent agents.
3) Implement the agent types described in §4, with the generation→verification loop in §5.
4) Support LLM-backed agents through an **OpenAI-compatible** chat/completions API so the same code can target OpenAI, OpenRouter, etc. via configuration.

### 1.2 Non-goals (initially)

- Formal proof checking (Lean, Coq, Isabelle) is out of scope for MVP.
- Long-term persistence of the knowledge base (DB) is out of scope for MVP.
- A UI/webapp is out of scope for MVP.
- Tooling beyond “LLM + internal graph editing + optional math libs” is out of scope for MVP.

## 2. Terminology

- **Run**: One end-to-end execution of the system on a single problem input.
- **Agent**: A unit of reasoning that consumes context and produces an output (often via an LLM).
- **Context graph**: A directed graph whose nodes are agents and whose edges define context flow.
- **Knowledge base (KB)**: An in-memory, globally shared store of structured math facts created during a run.
- **Tool call**: A structured request produced by an LLM output that asks the runtime to do something (spawn an agent, write KB entries, etc.).
- **Normalized output**: A structured (typed) representation of an agent’s output used internally, derived from raw LLM text by a parser step.

## 3. High-level architecture

### 3.1 Context graph

The fundamental data structure is a **context graph** that evolves over time:

- Vertices: agents, each with (a) type, (b) configuration, (c) input, (d) output(s), (e) status.
- Directed edges: “send output of A to B as part of B’s context”.
- Only one agent is “active” at a time (i.e., mutating graph/KB), ensuring no races.

### 3.2 Scheduling model

- The runtime maintains a **LIFO** stack of runnable agent nodes.
- The runtime executes nodes **sequentially** (no concurrency) for debuggability.
- A node becomes runnable when:
  - it is first created (spawned), or
  - it is re-queued by its parent after receiving sub-agent outputs, or
  - it is re-queued by the verifier loop (§5).
- Nodes run to completion (produce a final normalized output) before another node runs.

### 3.3 Knowledge base

- The KB is global to the run, accessible read/write to all agents.
- KB entries are structured objects with stable IDs and types (Definition/Result/Notation/etc.).
- The KB is used as:
  - shared memory between agents,
  - a source of context injected into prompts,
  - a substrate for incremental, compositional reasoning.

## 4. Agents

### 4.1 Common agent contract

Each agent type must support:

- **Inputs**
  - `problem`: the original problem statement (Markdown/plain text)
  - `local_context`: a list of upstream agent outputs selected by the runtime via graph edges
  - `kb_snapshot`: a view of the KB (possibly truncated/filtered)
  - `task`: a typed “what to do” payload (varies per agent type)
- **Outputs**
  - `raw_text`: the model’s raw completion (for trace/debug)
  - `normalized`: structured output parsed from `raw_text` (see §6)
  - `kb_writes`: KB entries added/updated (may be empty)
  - `spawn_requests`: requests to spawn sub-agents (may be empty)

The runtime is responsible for:
- executing spawn requests,
- routing outputs along graph edges,
- applying KB writes,
- enqueuing next runnable agents.

### 4.2 Orchestrator agent

Purpose: top-level controller for the run.

Responsibilities:
- Spawn exploration (§4.3) for `M` rounds (possibly `M=0`).
- Spawn a solve attempt via a worker agent (§4.4).
- Produce the final run output (proof + metadata).

Config:
- `exploration_rounds: int` (default 0)
- `max_total_steps: int` (hard cap on runnable-node executions)

### 4.3 Exploration agent

Purpose: broaden understanding of the problem setting by generating questions/conjectures and delegating them.

Responsibilities:
- Propose up to `Q` exploration questions based on the problem statement + KB.
- Spawn worker agents to address each question.
- Incorporate results into KB (via KB writes).

Config:
- `max_questions: int`
- `llm: LLMConfig`

### 4.4 Worker agent

Purpose: solve a given subproblem using an internal generation→verification loop.

Responsibilities:
- Spawn `N` prover agents (§4.5) that attempt independent solutions.
- Spawn a verifier agent (§4.6) that scores/ranks attempts.
- If no acceptable solution, optionally iterate for `K` rounds:
  - re-prompt provers with verifier feedback and updated KB,
  - potentially decompose via plan steps (see below).

Config:
- `num_provers: int` (default 1)
- `max_verify_rounds: int` (default 1)
- `allow_decomposition: bool` (default true)

Decomposition semantics:
- A prover may output a **plan** as an ordered list of steps.
- The worker may execute steps sequentially by spawning sub-worker agents, each tasked with a step.
- Step outputs are written into the KB and become context for subsequent steps.

### 4.5 Prover agent

Purpose: generate a candidate solution/proof attempt.

Responsibilities:
- Produce a solution attempt for the given task.
- Optionally produce:
  - a plan (list of steps) to solve the problem,
  - a claim that the requested conclusion is false, with a counterexample/argument.

Config:
- `llm: LLMConfig`

### 4.6 Verifier agent

Purpose: evaluate and rank prover attempts.

Responsibilities:
- Consume `N` candidate attempts.
- Produce:
  - a ranking (best→worst),
  - per-attempt scores and critiques,
  - a decision: “accept attempt i” vs “reject all”.

Verifier model:
- Verifier may be an ensemble of `E` LLM calls; aggregate to a final verdict.

Config:
- `ensemble_size: int` (default 1)
- `llms: list[LLMConfig]` (length 1 or `ensemble_size`, depending on implementation choice)
- Acceptance is by majority vote: accept iff the best attempt has strictly more than half the votes (§9).

### 4.7 Parser agent

Motivation: LLMs frequently produce brittle Markdown. Internally we need typed, machine-usable outputs.

Purpose:
- Convert an agent’s raw Markdown-ish text into a well-typed normalized schema, preserving verbatim content as much as possible.

Usage:
- Prover outputs are parsed into `SolutionAttempt` (§6.2).
- Verifier outputs are parsed into `VerificationReport` (§6.3).
- (Optionally) Exploration outputs are parsed into `ExplorationQuestions` (§6.4).

Config:
- `llm: LLMConfig` (prefer a cheap/fast model)

## 5. Solve pipeline (run flow)

Given an input problem, the run proceeds:

1) Create a new run with empty context graph and KB.
2) Spawn an orchestrator agent node with the problem as input.
3) Orchestrator performs:
   - **Exploration phase** (optional):
     - for `M` rounds:
       - spawn exploration agent,
       - exploration agent spawns worker agents for each question,
       - apply KB writes from completed workers.
   - **Solve phase**:
     - spawn a worker agent on the original problem.
4) Worker performs up to `K` verify rounds:
   - spawn `N` provers (executed sequentially in MVP),
   - parse each attempt into normalized schema,
   - spawn verifier; parse report; accept best attempt if acceptable,
   - else re-run provers with feedback (and/or decomposition) until `K` exhausted.
5) Orchestrator writes final output Markdown:
   - final proof (or explicit failure + reason),
   - include a KB appendix (key KB items used).

## 6. Schemas (normalized outputs)

All normalized schemas should be represented as Python `@dataclass`es (MVP), and be serializable to JSON for tracing.

### 6.1 Knowledge base entry

KB entry kinds (extensible):
- `Definition`, `Notation`, `Result` (lemma/theorem/proposition), `Algorithm`, `Example`, `Counterexample`.

Minimum required fields:
- `id: str` (stable within run; e.g. `"Result 2.1"`)
- `kind: KBKind`
- `title: str`
- `content_md: str` (verbatim)
- `tags: list[str]`
- `sources: list[str]` (free-form citations such as “derived from attempt 1”)

### 6.2 `SolutionAttempt`

Fields:
- `final_answer_md: str` (verbatim solution/proof/counterexample)
- `outline_steps: list[str]` (optional)
- `kb_updates: list[KBEntry]` (optional)
- `claims_incorrect_conclusion: bool`

### 6.3 `VerificationReport`

Fields:
- `accepted: bool`
- `best_attempt_index: int | None`
- `attempt_scores: list[int]` (number of verifier votes for correctness)
- `attempt_critiques_md: list[str]`
- `global_feedback_md: str`

### 6.4 `ExplorationQuestions`

Fields:
- `questions: list[str]`
- `rationales_md: list[str]` (same length as `questions`)

## 7. LLM integration

### 7.1 LLM interface + OpenRouter client

The system must define an LLM interface (e.g., `LLMClient`) and implement an **OpenRouter** client using `httpx`.

OpenRouter exposes an OpenAI-compatible chat API, so the request/response format must follow OpenAI-style chat completions with function calling enabled.

- `base_url`: e.g. OpenAI default, OpenRouter, or other compatible provider.
- `api_key_env`: env var name used to load the API key.
- `model`: model identifier.
- Sampling params: temperature, top_p, max_tokens, etc.

This avoids hard-coding “OpenRouter-only” while still supporting OpenRouter.

### 7.2 Tool calling

The system should support OpenAI **function-calling** tool calls emitted by the LLM. MVP tool surface area:

- `spawn_agent(agent_type, task, edge_from_parent: bool = true)`
- `kb_write(entries: list[KBEntry])`
- `finish(output_text)` (agent declares completion)

Tool calls must be passed through the runtime to mutate the context graph/KB as appropriate.

## 8. Configuration system

Requirements:

- All runtime behavior is driven by a top-level run config dataclass.
- Configs live in a registry: `dict[str, RunConfig]`.
- Preferred style:
  - a `@register_config` decorator applied to a function that returns a `RunConfig`,
  - the function name is the config name.
 - Configs are drawn from the registry in code only (no config file loading in MVP).

Config must cover:
- all agent configs (LLMs + numeric parameters),
- global runtime limits (max steps, timeouts, trace toggles),
- default prompts (system + per-agent instructions), overridable per config.

## 9. Concrete decisions (MVP)

These decisions are fixed for MVP:

1) **LLM client**: implement an `LLMClient` interface with an OpenRouter `httpx` implementation.
2) **Tool calling**: use OpenAI function-calling schemas.
3) **Verifier acceptance + scoring**:
   - Each verifier LLM produces a boolean verdict “correct” / “incorrect” for each attempt.
   - The score of an attempt is the number of “correct” votes it receives.
   - The best attempt is the one with maximum score (ties broken deterministically, e.g. lowest index).
   - Accept iff the best attempt has strictly more than half the verifier votes (`score > ensemble_size / 2`); otherwise reject and report “unverified”.
4) **Final output**: `output.md` includes only the final proof plus a KB appendix; it does not include attempts or verifier feedback.
5) **Execution**: keep everything sequential (no async parallelism) in MVP.
6) **Configs**: registry-in-code only.
7) **Tracing**: if `--trace` is provided, store full prompts/responses and tool calls; redact API keys (never write secrets).

## 10. CLI

Minimum CLI:

- `prover run --config <name> --input <path> --output <path> [--trace <path>]`
- `prover list-configs`

Flags (MVP):
- `--config`: config name in registry
- `--input`: Markdown input file
- `--output`: Markdown output file
- `--trace`: optional JSONL trace file path
- `--max-steps`: override global step cap

## 11. Tracing & reproducibility

If `--trace` is provided, write JSONL events such as:

- run_start/run_end (timestamps, config name, versions)
- agent_spawn/agent_start/agent_end
- llm_request/llm_response (with API keys redacted; store full prompts/responses in MVP)
- kb_write (IDs and kinds)

## 12. Implementation milestones (suggested)

1) Core runtime: context graph + scheduler + KB + trace.
2) Config registry + CLI wired to run orchestrator.
3) Fake LLM backend + unit tests for runtime determinism.
4) Prover/verifier/parser agents with real LLM backend.
5) Exploration + decomposition planning.
