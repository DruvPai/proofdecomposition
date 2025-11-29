"""Shared pytest fixtures for the prover test suite."""

from __future__ import annotations

import pytest

from prover.agents import BaseAgent
from prover.config import LLMConfig, get_config
from tests.mock_llm import MockLLMClient, MockLLMRouter


@pytest.fixture
def mock_llm_router(monkeypatch: pytest.MonkeyPatch) -> MockLLMRouter:
    """Patch agent client creation with deterministic mocks."""

    config = get_config("default")
    router = MockLLMRouter(config)

    def _client_for(self: BaseAgent, llm_config: LLMConfig) -> MockLLMClient:
        if llm_config is None:  # pragma: no cover - defensive
            raise AssertionError("LLM configuration must not be None.")
        return MockLLMClient(router, llm_config.model)

    monkeypatch.setattr(BaseAgent, "_client_for", _client_for, raising=False)
    return router
