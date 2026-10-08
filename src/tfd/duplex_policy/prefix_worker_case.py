"""One case inside an owned worker, with evidence-derived progress.

Caller owns the concrete model loader/resource audit and parent watchdog. No
model library imports or instance control here. No retry of failed cases.
"""
import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .isolated_prefix_case import execute_isolated_prefix_case
from .owned_worker_watchdog import _accept_progress
from .controlled_prefix_audio import replay_packets


async def run_prefix_worker_case(model, vad, session_factory, recorder, prefix,
                                 near_speech, init_params, *, progress_path,
                                 result_path, max_seconds=300, load_components=None,
                                 load_stack=None):
    """Write complete only after the result is exclusively saved and fsynced.

progress_path is an empty file reserved by run_owned_worker. This single-case
protocol expects the fixed 45-packet prefix. Complete means input processing,
not natural answering, semantic validity or a successful independent evaluation.
Optional synchronous load_components runs in this owned worker's thread after
starting is recorded, returning (model, vad, fresh session factory). It must not
spawn subprocesses. Alternatively load_stack returns compose_controlled_stack's
components/near_speech/init_params mapping, loaded after starting. Supply no
preloaded components or replay controls in that mode. Neither hook imports a
concrete GPU runtime or substitutes for the parent's preparation watchdog.
"""
    progress_path, result_path = Path(progress_path), Path(result_path)
    if load_stack is not None and (not callable(load_stack) or load_components is not None
            or any(v is not None for v in (model, vad, session_factory, near_speech, init_params))):
        raise ValueError('stack loader requires no mixed components, controls or loaders')
    if load_components is not None and (not callable(load_components) or
                                        any(v is not None for v in (model, vad, session_factory))):
        raise ValueError('loader requires callable and no preloaded components')
    if (not math.isfinite(max_seconds) or max_seconds <= 0
            or (load_stack is None and (not callable(near_speech) or not isinstance(init_params, dict)))):
        raise ValueError('valid replay controls required before loading')
    list(replay_packets(prefix))
    if result_path.resolve() == progress_path.resolve():
        raise ValueError("separate evidence paths required")
    if result_path.exists():
        raise FileExistsError("result already exists")
    if not progress_path.is_file() or progress_path.stat().st_size != 0:
        raise ValueError("parent must reserve empty progress file")
    previous, completed = None, 0

    def emit(stage, cases=0):
        nonlocal previous
        message = dict(schema="tfd.worker_progress.v1", seq=0 if previous is None else previous["seq"] + 1,
                       stage=stage, completed_packets=completed, completed_cases=cases)
        accepted, _ = _accept_progress(message, previous, 45, 1)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(accepted) + "\n")
            stream.flush()
        previous = accepted

    def packet_done(record):
        nonlocal completed
        if not record["completed"] or record["input_id"] != str(completed):
            raise ValueError("nonconsecutive completed packet")
        completed += 1
        emit("running")

    emit("starting")
    outcome = None
    loading_seconds = None
    runtime_audit, runtime_audit_sha256 = None, None
    feature_audit = None
    if load_components is not None or load_stack is not None:
        started = time.monotonic()
        try:
            if load_stack is not None:
                stack = await asyncio.to_thread(load_stack)
                if not isinstance(stack, dict) or not {'components','near_speech','init_params'}.issubset(stack):
                    raise ValueError('complete controlled stack mapping required')
                if not callable(stack['near_speech']) or not isinstance(stack['init_params'], dict):
                    raise ValueError('loaded replay controls invalid')
                if 'feature_audit' in stack:
                    feature_audit = stack['feature_audit']
                    if not callable(feature_audit):
                        raise ValueError('feature audit hook must be callable')
                if 'runtime_audit' in stack:
                    audit = stack['runtime_audit']
                    if (not isinstance(audit, dict) or audit.get('schema') != 'tfd.selected_runtime_audit.v1'
                            or audit.get('external_launch_ready') is not False):
                        raise ValueError('selected non-launch runtime audit required')
                    # Detach before reset/init can mutate a loader-owned mapping.
                    # Hash canonical JSON, not a claim of complete runtime truth.
                    canonical = json.dumps(audit, sort_keys=True, separators=(',', ':'), allow_nan=False)
                    runtime_audit = json.loads(canonical)
                    runtime_audit_sha256 = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
                components = stack['components']
                near_speech, init_params = stack['near_speech'], dict(stack['init_params'])
            else:
                components = await asyncio.to_thread(load_components)
            if not isinstance(components, tuple) or len(components) != 3:
                raise ValueError('loader must return component triple')
            model, vad, session_factory = components
            if not callable(getattr(model, 'reset_session', None)) or not callable(getattr(vad, 'reset', None)) or not callable(session_factory):
                raise ValueError('invalid component interfaces')
            emit('model_loaded')
        except Exception as exc:
            outcome = dict(schema='tfd.controlled_prefix_preparation_failure.v1',
                status='runtime_failure', stage='component_load', error_type=type(exc).__name__,
                preparation_complete=False, preparation_seconds=time.monotonic() - started,
                input_complete=False, session_reuse_safe=False,
                final_response_naturally_complete=None, chunks=[])
        loading_seconds = time.monotonic() - started
    if outcome is None:
        outcome = await execute_isolated_prefix_case(model, vad, session_factory, recorder,
            prefix, near_speech, init_params, max_seconds=max_seconds,
            on_prepared=lambda: emit("ready"), on_completed=packet_done)
    if loading_seconds is not None:
        outcome['component_loading_seconds'] = loading_seconds
    if runtime_audit is not None:
        outcome['runtime_audit'] = runtime_audit
        outcome['runtime_audit_sha256'] = runtime_audit_sha256
    if outcome['input_complete'] and feature_audit is not None:
        try:
            audit = feature_audit()
            if (not isinstance(audit, dict) or audit.get('schema') != 'tfd.controlled_feature_audit.v1'
                or audit.get('external_launch_ready') is not False):
                raise ValueError('selected non-launch feature audit required')
            canonical = json.dumps(audit, sort_keys=True, separators=(',', ':'), allow_nan=False)
            outcome['feature_audit'] = json.loads(canonical)
            outcome['feature_audit_sha256'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        except Exception as exc:
            outcome.update(status='runtime_failure',stage='feature_audit',
                error_type=type(exc).__name__,input_complete=False,session_reuse_safe=False)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(outcome, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if outcome["input_complete"]:
        if completed != 45:
            raise ValueError("fixed packet plan incomplete")
        emit("complete", cases=1)
    return outcome
