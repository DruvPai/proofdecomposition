"""Structured JSONL tracing utility."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class TraceLogger:
    """Simple JSONL tracer."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Best-effort JSON default serializer for trace payloads."""

        if is_dataclass(value):
            return asdict(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    def _write(self, event: dict[str, Any]) -> None:
        if self._handle is None:
            return
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        self._handle.write(json.dumps(payload, default=self._json_default) + "\n")
        self._handle.flush()

    def run_start(self, config_name: str) -> None:
        self._write({"event": "run_start", "config": config_name})

    def run_end(self, success: bool) -> None:
        self._write({"event": "run_end", "success": success})

    def agent_event(
        self,
        agent_id: int,
        agent_type: str,
        status: str,
        info: dict[str, Any] | None = None,
    ) -> None:
        self._write(
            {
                "event": "agent",
                "id": agent_id,
                "type": agent_type,
                "status": status,
                "info": info or {},
            }
        )

    def llm_request(self, agent_id: int, request: dict[str, Any]) -> None:
        redacted = {**request}
        if "api_key" in redacted:
            redacted["api_key"] = "***"
        self._write({"event": "llm_request", "id": agent_id, "request": redacted})

    def llm_response(self, agent_id: int, response: dict[str, Any]) -> None:
        self._write({"event": "llm_response", "id": agent_id, "response": response})

    def close(self) -> None:
        if self._handle:
            self._handle.close()
