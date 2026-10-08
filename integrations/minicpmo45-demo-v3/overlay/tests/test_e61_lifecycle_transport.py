"""Offline lifecycle transport checks; no socket, model, or audible-stop claim."""

import ast
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from runtime.backend_client import RemoteBackendSession
from runtime.session import backend_event_to_runtime_event


def test_runtime_and_worker_payload_preserve_lifecycle_fields():
    # Extract the production serializer without importing GPU worker startup.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "worker.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_event_payload")
    namespace = {"Any": Any, "Dict": Dict}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "worker.py", "exec"), namespace)
    cases = [("text", "output.text", False, False),
             ("audio", "output.audio", False, False),
             ("listen", "model.state", False, False),
             ("listen", "model.state", True, False),
             ("listen", "model.state", True, True)]
    for kind, channel, terminated, cancel in cases:
        event = {"type": "response.output.delta", "kind": kind,
                 "response_id": "r-fixed", "reason": "force_listen" if cancel else "model_listen",
                 "metrics": {"tfd_lifecycle_enabled": True,
                             "tfd_response_terminated": terminated,
                             "tfd_client_force_listen": cancel}}
        runtime = backend_event_to_runtime_event(event)
        assert runtime.channel == channel
        payload = namespace["_event_payload"](runtime)
        assert json.loads(json.dumps(payload)) == event
        assert payload["metrics"]["tfd_response_terminated"] is terminated
        assert payload["metrics"]["tfd_client_force_listen"] is cancel


def test_remote_backend_pull_preserves_boolean_metrics_and_response_id():
    event = {"type": "response.output.delta", "kind": "listen", "response_id": "r-fixed",
             "metrics": {"tfd_lifecycle_enabled": True, "tfd_response_terminated": False,
                         "tfd_client_force_listen": False}}

    class Wire:
        async def recv(self):
            return json.dumps(event).encode("utf-8")

    session = RemoteBackendSession(base_url="http://127.0.0.1:1", mode="audio_duplex")
    session._ws = Wire()
    assert asyncio.run(session.pull()) == event
