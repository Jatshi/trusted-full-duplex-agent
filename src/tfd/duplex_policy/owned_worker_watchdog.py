"""Supervise one newly created direct child, never a service or existing PID.

Windows direct-child semantics have local test evidence. Opt-in POSIX group
abort has only injected test evidence: detached/leader-first-exit descendants
are not owned or verified gone. Workers must not spawn subprocesses until the
complete ownership policy is independently tested on the remote platform.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import time


STAGES = ("starting", "model_loaded", "ready", "running", "complete")


def _worker_platform():
    return os.name


class SharedDeadlineBudget:
    """One owner/host clock, no launch, retry, progress renewal or persistence.

    Caller must stop its queue after a failed leg and pass the returned absolute
    deadline to every worker in that leg. This primitive does not enforce that
    orchestration or authorize an independent evaluation launcher.
    """
    def __init__(self, queue_seconds, *, clock=time.monotonic):
        self._positive(queue_seconds)
        self._clock=clock
        self._last=self._now()
        self._queue_deadline=self._last+queue_seconds
        if not math.isfinite(self._queue_deadline):
            raise ValueError('finite queue deadline required')
        self._legs=set()

    @staticmethod
    def _positive(value):
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('positive finite budget required')

    def _now(self):
        now=self._clock()
        if type(now) not in (int,float) or not math.isfinite(now) or now<getattr(self,'_last',now):
            raise ValueError('finite nondecreasing owner clock required')
        self._last=now
        return now

    @property
    def queue_deadline(self):
        return self._queue_deadline

    def begin_leg(self, leg_id, leg_seconds):
        self._positive(leg_seconds)
        if not isinstance(leg_id,str) or not leg_id.strip() or leg_id in self._legs:
            raise ValueError('nonempty single-use leg id required')
        now=self._now()
        if now>=self._queue_deadline:
            raise TimeoutError('queue budget exhausted before leg launch')
        self._legs.add(leg_id)
        return min(self._queue_deadline,now+leg_seconds)

    def remaining(self, deadline):
        """Observe without renewing; deadline belongs to this owner's clock."""
        if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline>self._queue_deadline:
            raise ValueError('finite deadline within queue required')
        return max(0.0,deadline-self._now())


def _accept_progress(value, previous, expected_packets, expected_cases):
    if not isinstance(value, dict) or value.get("schema") != "tfd.worker_progress.v1":
        raise ValueError("invalid progress schema")
    for key in ("seq", "completed_packets", "completed_cases"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError("invalid progress integer")
    if value.get("stage") not in STAGES:
        raise ValueError("unknown stage")
    if value["seq"] != (previous["seq"] + 1 if previous else 0):
        raise ValueError("nonconsecutive sequence")
    rank = STAGES.index(value["stage"])
    if previous:
        if previous["stage"] == "complete" or rank < STAGES.index(previous["stage"]):
            raise ValueError("stage regression or progress after completion")
        if any(value[k] < previous[k] for k in ("completed_packets", "completed_cases")):
            raise ValueError("counter regression")
    if value["completed_packets"] > expected_packets or value["completed_cases"] > expected_cases:
        raise ValueError("counter exceeds fixed plan")
    if value["stage"] == "complete" and (value["completed_packets"] != expected_packets or
                                         value["completed_cases"] != expected_cases):
        raise ValueError("premature completion")
    effective = previous is None or rank > STAGES.index(previous["stage"]) or any(
        value[k] > previous[k] for k in ("completed_packets", "completed_cases"))
    # Keep only the fixed protocol, not arbitrary fields that may carry secrets.
    return {k: value[k] for k in ("schema", "seq", "stage", "completed_packets", "completed_cases")}, effective


def run_owned_worker(argv, *, log_path, progress_path, expected_packets,
                     expected_cases, deadline_seconds, no_progress_seconds,
                     poll_seconds=.05, terminate_grace_seconds=2, prefix_result_path=None,
                     phase_deadlines=None, checkpoint_path=None, checkpoint_seconds=60,
                     process_scope='new_direct_child_only', absolute_deadline=None):
    """Exclusive new logs + fixed JSONL progress; exit0 needs complete counters.

This is a supervisor primitive, not a GPU queue launcher. Resource checks,
per-init/case/leg/queue policy belong to the owner. Optional prefix_result_path
requires fixed single-case artifact validation; generic protocol-only callers
remain unchanged and do not establish artifact validity.
No retries, shell, instance operations, PID lookup or broad process-name kills.
Optional phase_deadlines bounds prepare (spawn through model loading/init) and
replay (first observed ready/running through exit). Parent monotonic clocks;
neither counter progress nor model_loaded resets a phase deadline. Single
worker phases only, not per-case/leg/queue deadlines or process-tree ownership.
Optional parent JSONL checkpoints: initial, periodic (default60s), terminal.
These are observations, not model/KV snapshots and never authorize resumption.
Write/flush/fsync failure stops the worker; blocking storage I/O itself is not
hard-bounded by this synchronous supervisor.
"""
    if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(a, str) for a in argv):
        raise ValueError("explicit nonempty argv required")
    if absolute_deadline is not None:
        if type(absolute_deadline) not in (int,float) or not math.isfinite(absolute_deadline):
            raise ValueError('finite owner-host monotonic deadline required')
        if time.monotonic()>=absolute_deadline:
            raise TimeoutError('shared budget exhausted before worker launch')
    if process_scope not in ('new_direct_child_only','new_posix_process_group'):
        raise ValueError('unknown owned process scope')
    group_abort=process_scope=='new_posix_process_group'
    if group_abort and (_worker_platform()!='posix' or not callable(getattr(os,'killpg',None))):
        raise ValueError('POSIX process-group support required before launch')
    for value in (deadline_seconds, no_progress_seconds, poll_seconds, terminate_grace_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite time controls required")
    for value in (expected_packets, expected_cases):
        if type(value) is not int or value <= 0:
            raise ValueError("positive fixed expected counters required")
    if phase_deadlines is not None:
        if not isinstance(phase_deadlines, dict) or set(phase_deadlines) != {'prepare', 'replay'}:
            raise ValueError('exact prepare/replay budgets required')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in phase_deadlines.values()):
            raise ValueError('positive finite phase budgets required')
        phase_deadlines = dict(phase_deadlines)
    log_path, progress_path = Path(log_path), Path(progress_path)
    if type(checkpoint_seconds) not in (int, float) or not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
        raise ValueError('positive finite checkpoint interval required')
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        other_paths = [log_path, progress_path]
        if prefix_result_path is not None:
            other_paths.append(Path(prefix_result_path))
        if checkpoint_path.resolve() in [p.resolve() for p in other_paths]:
            raise ValueError('separate checkpoint path required')
    if prefix_result_path is not None:
        prefix_result_path = Path(prefix_result_path)
        if expected_packets != 45 or expected_cases != 1:
            raise ValueError('prefix validation requires one fixed case')
        if prefix_result_path.resolve() in (log_path.resolve(), progress_path.resolve()):
            raise ValueError('separate result path required')
        if prefix_result_path.exists():
            raise FileExistsError('result already exists')
    if log_path.resolve() == progress_path.resolve():
        raise ValueError("separate log and progress paths required")
    if checkpoint_path is not None:
        # Refuse old evidence before creating logs or starting any child.
        with checkpoint_path.open('x', encoding='utf-8'):
            pass
    started = last_effective = time.monotonic()
    previous, pending, status = None, b"", None
    terminated = killed = False
    phase, phase_started, deadline_phase = 'prepare', started, None
    checkpoint_seq, checkpoint_error_type, last_checkpoint = 0, None, started

    def save_checkpoint(observed_status, reaped=False, evidence=None):
        nonlocal checkpoint_seq, checkpoint_error_type, last_checkpoint, status
        if checkpoint_path is None or checkpoint_error_type is not None:
            return
        now = time.monotonic()
        record = dict(schema='tfd.worker_checkpoint.v1', seq=checkpoint_seq,
                      status=observed_status, pid=process.pid, reaped=reaped,
                      exit_code=process.returncode if reaped else None,
                      elapsed_seconds=now - started, phase=phase,
                      phase_elapsed_seconds=now - phase_started,
                      deadline_phase=deadline_phase, last_progress=previous,
                      result_validation=evidence, process_scope=process_scope,
                      absolute_deadline=absolute_deadline,
                      descendants_verified_gone=False, external_launch_ready=False)
        try:
            with checkpoint_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            checkpoint_seq += 1
            last_checkpoint = now
        except OSError as exc:
            checkpoint_error_type = type(exc).__name__
            status = 'checkpoint_failure'

    def stop_owned(process):
        nonlocal terminated, killed
        if process.poll() is None:
            terminated = True
            if group_abort:
                # Popen(start_new_session=True) creates this group's leader.
                # Keep leader unreaped until BOTH signals have been attempted:
                # even a zombie reserves its PID against group-id reuse here.
                # Never signal a leader already observed/reaped by poll().
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(terminate_grace_seconds)
                killed=True
                try:
                    os.killpg(process.pid,getattr(signal,'SIGKILL',9))
                except ProcessLookupError:
                    pass
                process.wait(timeout=terminate_grace_seconds)
                return
            process.terminate()
            try:
                process.wait(timeout=terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                killed = True
                process.kill()
                process.wait(timeout=terminate_grace_seconds)

    # Exclusive creation avoids overwriting prior evidence; no child before both.
    with log_path.open("xb") as log:
        with progress_path.open("xb"):
            pass
        with progress_path.open("rb") as progress:
            if absolute_deadline is not None and time.monotonic()>=absolute_deadline:
                raise TimeoutError('shared budget exhausted during launch preparation')
            process = subprocess.Popen(argv, shell=False, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if _worker_platform() == "nt" else 0,
                **({'start_new_session':True} if group_abort else {}))
            try:
                save_checkpoint('running')
                while status is None:
                    # Check the old phase before processing a late ready record:
                    # a late stage transition must not erase an expired budget.
                    now = time.monotonic()
                    if absolute_deadline is not None and now>=absolute_deadline:
                        status='shared_deadline_timeout'
                        break
                    if phase_deadlines and now - phase_started >= phase_deadlines[phase]:
                        status, deadline_phase = 'phase_deadline_timeout', phase
                        break
                    # Observe exit before reading: all child writes precede an
                    # observed exit. Otherwise the final record can race a read.
                    exit_code = process.poll()
                    raw = progress.read(65536)
                    pending += raw
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    try:
                        if len(pending) > 65536:
                            raise ValueError("overlong progress record")
                        for line in lines:
                            if len(line) > 65536:
                                raise ValueError("overlong progress record")
                            previous, effective = _accept_progress(json.loads(line), previous,
                                                                  expected_packets, expected_cases)
                            if phase == 'prepare' and previous['stage'] in ('ready', 'running', 'complete'):
                                phase, phase_started = 'replay', time.monotonic()
                            if effective:
                                last_effective = time.monotonic()
                    except (ValueError, TypeError, UnicodeError):
                        status = "progress_invalid"
                        break
                    now = time.monotonic()
                    if absolute_deadline is not None and now>=absolute_deadline:
                        status='shared_deadline_timeout'
                    elif now - started >= deadline_seconds:
                        status = "deadline_timeout"
                    elif exit_code is not None and len(raw) < 65536:
                        if pending:
                            status = "progress_invalid"
                        elif exit_code != 0:
                            status = "worker_nonzero"
                        elif previous and previous["stage"] == "complete":
                            status = "complete"
                        else:
                            status = "completion_missing"
                    elif now - last_effective >= no_progress_seconds:
                        status = "no_progress_timeout"
                    if status is None and checkpoint_path is not None and now - last_checkpoint >= checkpoint_seconds:
                        save_checkpoint('running')
                    if status is None:
                        remaining = max(0, deadline_seconds - (now - started))
                        if absolute_deadline is not None:
                            remaining=min(remaining,max(0,absolute_deadline-now))
                        if phase_deadlines:
                            remaining = min(remaining, max(0, phase_deadlines[phase] - (now - phase_started)))
                        if checkpoint_path is not None:
                            remaining = min(remaining, max(0, checkpoint_seconds - (now - last_checkpoint)))
                        time.sleep(min(poll_seconds, remaining))
            finally:
                stop_owned(process)
                # wait returns already-finished children immediately; always reap.
                process.wait(timeout=terminate_grace_seconds)
    evidence = None
    if status == 'complete' and prefix_result_path is not None:
        from .prefix_result_validation import validate_prefix_result
        try:
            evidence = validate_prefix_result(prefix_result_path)
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            status = 'result_invalid'
    save_checkpoint(status, reaped=True, evidence=evidence)
    return {"schema": "tfd.owned_worker_result.v1", "status": status,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_error_type": checkpoint_error_type,
            "deadline_phase": deadline_phase,
            "result_validation": evidence,
            "pid": process.pid, "exit_code": process.returncode, "reaped": True,
            "termination_requested": terminated, "kill_requested": killed,
            "elapsed_seconds": time.monotonic() - started, "last_progress": previous,
            "log_path": str(log_path), "progress_path": str(progress_path),
            "process_scope": process_scope, "descendants_verified_gone": False,
            "absolute_deadline": absolute_deadline,
            "external_launch_ready": False}
