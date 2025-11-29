"""Sequential runtime and scheduler for the prover system."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from prover.constants import (
    CONTEXT_HIERARCHY_MAX_DEPTH,
    CONTEXT_HIERARCHY_PROBLEM_CHARS,
)
from prover.agents import (
    ExplorationAgent,
    OrchestratorAgent,
    ParserAgent,
    ProverAgent,
    VerifierAgent,
    WorkerAgent,
)
from prover.config import RunConfig, get_config
from prover.graph import (
    AgentNode,
    AgentOutput,
    AgentType,
    SpawnRequest,
)
from prover.kb import KnowledgeBase
from prover.schemas import KBEntry
from prover.trace import TraceLogger


class AgentImpl(Protocol):
    """Minimal interface implemented by all agent implementations."""

    async def run(
        self, node: AgentNode, kb: KnowledgeBase, trace: TraceLogger
    ) -> AgentOutput:
        """Execute the agent for one step and return an output."""


class Runtime:
    """Main runtime managing the context graph and sequential scheduler."""

    def __init__(self, config: RunConfig, trace_path: Path | None = None) -> None:
        """Initialize runtime state."""
        self.config = config
        self.kb = KnowledgeBase()
        self.nodes: dict[int, AgentNode] = {}
        self.stack: list[int] = []
        self.next_id = 0
        self.trace = TraceLogger(trace_path if trace_path else None)
        self.trace.run_start(config.name)

    def _new_id(self) -> int:
        """Return a fresh node id."""
        self.next_id += 1
        return self.next_id

    def _register_node(self, node: AgentNode) -> None:
        """Insert node and push to runnable stack."""
        self.nodes[node.id] = node
        self.stack.append(node.id)
        self.trace.agent_event(node.id, node.agent_type.value, "created")

    def _get_agent_impl(self, agent_type: AgentType) -> AgentImpl:
        """Return the agent implementation for a type."""
        return {
            AgentType.ORCHESTRATOR: OrchestratorAgent(self.config),
            AgentType.EXPLORATION: ExplorationAgent(self.config),
            AgentType.WORKER: WorkerAgent(self.config),
            AgentType.PROVER: ProverAgent(self.config),
            AgentType.VERIFIER: VerifierAgent(self.config),
            AgentType.PARSER: ParserAgent(self.config),
        }[agent_type]

    def _apply_kb_writes(self, entries: list[KBEntry]) -> None:
        """Apply KB updates."""
        self.kb.extend(entries)

    def _spawn_children(self, parent: AgentNode, spawns: list[SpawnRequest]) -> None:
        """Create child nodes requested by an agent."""
        dependent_child_ids: list[int] = []
        for req in spawns:
            child_id = self._new_id()
            parent_ids = [parent.id] if req.edge_from_parent else []
            local_context: list[AgentOutput] = []
            for parent_id in parent_ids:
                parent_node = self.nodes.get(parent_id)
                if parent_node is None or not parent_node.outputs:
                    continue
                local_context.append(parent_node.outputs[-1])

            context_hierarchy_md = self._build_context_hierarchy_md(parent_ids)
            child_inputs = {
                "problem": parent.inputs.get("problem"),
                "local_context": local_context,
                "task": req.task,
                "context_hierarchy_md": context_hierarchy_md,
            }
            node = AgentNode(
                id=child_id,
                agent_type=req.agent_type,
                inputs=child_inputs,
                parents=parent_ids,
            )
            parent.children.append(child_id)
            self._register_node(node)
            if req.edge_from_parent:
                dependent_child_ids.append(child_id)
        if dependent_child_ids:
            parent.pending_children.update(dependent_child_ids)
            parent.waiting = True
            parent.status = "waiting"

    def _build_context_hierarchy_md(self, parent_ids: list[int]) -> str:
        """Build a Markdown hierarchy of parent problem statements.

        Args:
            parent_ids: Immediate parent node IDs for the child being spawned.

        Returns:
            A Markdown string describing parent chains from root to each parent.
        """

        if not parent_ids:
            return "None."

        def _clip(text: str, limit: int) -> str:
            cleaned = text.strip()
            if len(cleaned) <= limit:
                return cleaned
            return f"{cleaned[: limit - 1].rstrip()}…"

        lines: list[str] = []
        for parent_id in parent_ids:
            chain: list[AgentNode] = []
            current = self.nodes.get(parent_id)
            steps = 0
            while current is not None and steps < CONTEXT_HIERARCHY_MAX_DEPTH:
                chain.append(current)
                if not current.parents:
                    break
                next_parent_id = min(current.parents)
                current = self.nodes.get(next_parent_id)
                steps += 1
            chain.reverse()

            for depth, node in enumerate(chain):
                task = node.inputs.get("task", {})
                goal = task.get("goal")
                problem = task.get("problem", node.inputs.get("problem", ""))
                label = f"{node.agent_type.value}#{node.id}"
                if goal:
                    label = f"{label} ({goal})"
                indent = "  " * depth
                problem_text = _clip(str(problem), CONTEXT_HIERARCHY_PROBLEM_CHARS)
                lines.append(f"{indent}- **{label}**: {problem_text}")
        return "\n".join(lines) if lines else "None."

    def _handle_child_completion(self, child: AgentNode, output: AgentOutput) -> None:
        """Propagate child outputs to parents and requeue if ready."""
        # If the child is still waiting on its own children, do not propagate a
        # partial/placeholder output up the graph. We only requeue parents once
        # the child has fully completed its work (waiting == False).
        if child.waiting:
            return
        for parent_id in child.parents:
            parent = self.nodes[parent_id]
            local_context: list[AgentOutput] = parent.inputs.setdefault(
                "local_context", []
            )
            local_context.append(output)
            if child.id in parent.pending_children:
                parent.pending_children.remove(child.id)
            if not parent.pending_children and parent.waiting:
                parent.waiting = False
                parent.status = "pending"
                self.stack.append(parent.id)

    async def _run_once(self) -> bool:
        """Execute one runnable node if available."""
        if not self.stack:
            return False
        node_id = self.stack.pop()
        node = self.nodes[node_id]
        if node.status == "done" or node.waiting:
            return True
        node.status = "running"
        self.trace.agent_event(node.id, node.agent_type.value, "start")
        agent = self._get_agent_impl(node.agent_type)
        output = await agent.run(node, self.kb, self.trace)
        node.outputs.append(output)
        node.status = "done"
        self._apply_kb_writes(output.kb_writes)
        self._spawn_children(node, output.spawn_requests)
        self.trace.agent_event(node.id, node.agent_type.value, "end")
        self._handle_child_completion(node, output)
        return True

    async def run(self, problem_text: str) -> AgentOutput:
        """Run the orchestrator on the given problem and return its final output."""
        orchestrator_task = {"problem": problem_text}
        root = AgentNode(
            id=self._new_id(),
            agent_type=AgentType.ORCHESTRATOR,
            inputs={
                "problem": problem_text,
                "local_context": [],
                "task": orchestrator_task,
            },
        )
        self._register_node(root)

        steps = 0
        try:
            while self.stack and steps < self.config.orchestrator.max_total_steps:
                steps += 1
                await self._run_once()
        finally:
            # Consider the run successful only if the graph has quiesced.
            success = (
                bool(self.nodes[root.id].outputs)
                and not self.stack
                and not any(n.waiting for n in self.nodes.values())
            )
            self.trace.run_end(success=success)
            self.trace.close()

        return self.nodes[root.id].outputs[-1]


def run_problem(
    input_path: Path,
    output_path: Path,
    config_name: str,
    trace_path: Path | None = None,
    *,
    max_steps: int | None = None,
) -> None:
    """Entry point to run the solver from CLI."""

    problem = input_path.read_text(encoding="utf-8")
    config = get_config(config_name)
    if max_steps is not None:
        config.orchestrator.max_total_steps = max_steps
    runtime = Runtime(config, trace_path)
    output = asyncio.run(runtime.run(problem))
    # Produce final Markdown.
    appendix_lines = runtime.kb.render_appendix_lines()
    kb_section = "\n".join(appendix_lines) if appendix_lines else "No KB entries."
    final_md = f"""
# Problem
{problem}

# Solution
{output.normalized.final_answer_md}

# KB Appendix
{kb_section}
    """.strip()
    output_path.write_text(final_md, encoding="utf-8")
