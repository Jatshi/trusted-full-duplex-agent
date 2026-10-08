"""In-process controlled replay adapter for an already initialized session.

Caller must reset model/VAD/gate/audio caches per case. Cooperative timeout
cannot stop blocking GPU/thread work: an outer process watchdog is required,
and a failed session MUST NOT be reused. No external data or model is loaded.
"""
import asyncio
import base64
import copy
from contextlib import suppress
import math
import time

from .controlled_prefix_audio import replay_packets


async def execute_prefix_session(session, recorder, prefix, near_speech, *, max_seconds=300,
                                 on_completed=None):
    """Execute fixed packets without passing EOF/expected outcomes to policy."""
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Expected positive finite session deadline")
    if on_completed is not None and not callable(on_completed):
        raise ValueError("on_completed must be a synchronous callable")
    # Validate all input before invoking the session or VAD.
    packets = list(replay_packets(prefix))
    chunks = []
    started = time.monotonic()

    async def replay():
        for index, packet in enumerate(packets):
            offset = len(recorder.events)
            record = dict(input_id=str(index), start_sample=packet.start_sample,
                          real_samples=packet.real_samples,
                          artificial_samples=packet.artificial_samples,
                          input_eof=packet.eof_reached, completed=False,
                          response_before=session._active_response_id, events=[])
            chunks.append(record)
            t0 = time.monotonic()
            try:
                near = bool(near_speech(packet.audio.copy()))
                record['near_speech'] = near
                await session._push_full_duplex(dict(
                    audio=base64.b64encode(packet.audio.tobytes()).decode('ascii'),
                    input_id=str(index), force_listen=near,
                    hints=dict(source='controlled_near_end_replay')))
                await session._wait_finalize()
                record['completed'] = True
            finally:
                record['events'] = copy.deepcopy(recorder.events[offset:])
                record['response_after'] = session._active_response_id
                record['seconds'] = time.monotonic() - t0
            if on_completed is not None:
                # Completed only after backend + finalize + evidence capture.
                # Give sinks their own copy; sink failure terminates the case.
                on_completed(copy.deepcopy(record))

    status, error_type = 'input_exhausted', None
    task = asyncio.create_task(replay())
    try:
        done, pending = await asyncio.wait({task}, timeout=max_seconds)
        if pending:
            status, error_type = 'deadline_timeout', 'TimeoutError'
        else:
            await task  # Backend TimeoutError is a runtime failure, not our deadline.
    except Exception as exc:
        status, error_type = 'runtime_failure', type(exc).__name__
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    complete = status == 'input_exhausted'
    return dict(schema='tfd.controlled_prefix_session.v1', status=status,
                error_type=error_type, input_complete=complete,
                session_reuse_safe=complete,
                final_response_naturally_complete=None,
                completed_real_samples=sum(c['real_samples'] for c in chunks if c['completed']),
                completed_artificial_samples=sum(c['artificial_samples'] for c in chunks if c['completed']),
                elapsed_seconds=time.monotonic() - started,
                deadline_seconds=max_seconds, chunks=chunks)
