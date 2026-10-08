"""One prepared controlled case; caller must supply an outer process watchdog.

No model imports or default-demo changes. The factory must construct a fresh
BackendProtocolSession, so controller/gate/usage/response state is per case.
This adapter neither owns the GPU process nor promises to interrupt a blocked
reset/init thread. Never launch an evaluation queue without the outer watchdog.
"""
import asyncio
import math
import time

from .controlled_prefix_audio import replay_packets
from .prefix_session_executor import execute_prefix_session


async def execute_isolated_prefix_case(model, vad, session_factory, recorder,
                                      prefix, near_speech, init_params, *,
                                      max_seconds=300, on_prepared=None, on_completed=None):
    """Reset model/TTS cache and VAD, construct/init a fresh session, then replay.

max_seconds bounds replay only, not preparation. Failed preparation is terminal
for this case; there are no retries or implicit cleanup/reset of a live worker.
The owner must discard a failed worker and reset even after a successful case.
"""
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    if not callable(near_speech) or not isinstance(init_params, dict):
        raise ValueError("near_speech callable and init_params dict required")
    if any(hook is not None and not callable(hook) for hook in (on_prepared, on_completed)):
        raise ValueError("progress hooks must be synchronous callables")
    # Reject malformed inputs before touching model or VAD state.
    list(replay_packets(prefix))
    started = time.monotonic()
    stage = "model_reset"
    try:
        await asyncio.to_thread(model.reset_session, reset_token2wav_cache=True)
        stage = "vad_reset"
        vad.reset()
        stage = "session_construct"
        session = session_factory()
        if session.initialized or session.closed or session._active_response_id is not None:
            raise ValueError("factory must return a fresh uninitialized session")
        stage = "session_init"
        await session.init(dict(init_params))
        preparation_seconds = time.monotonic() - started
        stage = "progress_ready"
        if on_prepared is not None:
            on_prepared()
        stage = "replay"
        result = await execute_prefix_session(session, recorder, prefix, near_speech,
                                             max_seconds=max_seconds, on_completed=on_completed)
        result["preparation_complete"] = True
        result["preparation_seconds"] = preparation_seconds
        return result
    except Exception as exc:
        # Do not expose diagnostic text (paths/tokens may appear in backend errors).
        return {"schema": "tfd.controlled_prefix_preparation_failure.v1",
                "status": "runtime_failure", "stage": stage,
                "error_type": type(exc).__name__, "preparation_complete": False,
                "preparation_seconds": time.monotonic() - started,
                "input_complete": False, "session_reuse_safe": False,
                "final_response_naturally_complete": None, "chunks": []}
