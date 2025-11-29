"""Project-wide constants and default limits.

This module centralizes numeric limits so that code stays readable and avoids
"magic numbers" sprinkled across the codebase.
"""

from __future__ import annotations

# Core runtime caps
DEFAULT_MAX_TOTAL_STEPS: int = 256

# Configuration defaults
DEFAULT_EXPLORATION_ROUNDS: int = 0
DEFAULT_EXPLORATION_MAX_QUESTIONS: int = 3

DEFAULT_WORKER_NUM_PROVERS: int = 1
DEFAULT_WORKER_MAX_VERIFY_ROUNDS: int = 1
DEFAULT_WORKER_MAX_PLAN_STEPS: int = 8
DEFAULT_WORKER_MAX_DECOMPOSITION_DEPTH: int = 1

DEFAULT_VERIFIER_ENSEMBLE_SIZE: int = 1

# LLM sampling defaults
DEFAULT_LLM_TEMPERATURE: float = 0.0
DEFAULT_LLM_TOP_P: float = 1.0

# Text / prompt limits
DEFAULT_MAX_TEXT_CHARS: int = 1_000_000
DEFAULT_LOCAL_CONTEXT_LIMIT: int = 8

# Runtime display helpers
RUNTIME_SUMMARY_CHARS: int = 1_000_000
RUNTIME_STATUS_MESSAGE_CHARS: int = 1_000_000
RUNTIME_FEEDBACK_CHARS: int = 1_000_000
RUNTIME_WORKER_FEEDBACK_INLINE_CHARS: int = 1_000_000

RUNTIME_TASK_METADATA_MAX_KEYS: int = 1_000_000
RUNTIME_TASK_METADATA_VALUE_CHARS: int = 1_000_000
RUNTIME_METADATA_LINE_MAX_PARTS: int = 4
RUNTIME_METADATA_LIST_MAX_ITEMS: int = 3
RUNTIME_METADATA_LIST_ITEM_CHARS: int = 1_000_000
RUNTIME_METADATA_SCALAR_CHARS: int = 1_000_000

# Default caps when a task omits them.
DEFAULT_MAX_QUESTIONS: int = 3

# Prompt formatting
CONTEXT_HIERARCHY_MAX_DEPTH: int = 8
CONTEXT_HIERARCHY_PROBLEM_CHARS: int = 600

# KB prompt rendering
KB_PROMPT_MAX_CONTENT_CHARS: int = 320

# KB summarization
KB_SUMMARY_TITLE_CHARS: int = 120
KB_SUMMARY_STATEMENT_CHARS: int = 320

# Local context display
LOCAL_CONTEXT_SNIPPET_CHARS: int = 240

# HTTP / networking
HTTP_TIMEOUT_SECONDS: float = 60.0
