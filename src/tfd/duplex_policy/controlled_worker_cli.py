"""Opt-in single development prefix; not an independent evaluation launcher.

Run this file with --help. No arbitrary loader/command, retries, data downloads,
or default Demo changes. Direct-child supervision does not own descendants;
POSIX runtime and complete ASR/TTS dependency closure remain unverified.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tfd.duplex_policy.controlled_prefix_audio import replay_packets
from tfd.duplex_policy.controlled_stack_composition import load_controlled_runtime
from tfd.duplex_policy.owned_worker_watchdog import SharedDeadlineBudget, run_owned_worker
from tfd.duplex_policy.prefix_worker_case import run_prefix_worker_case


def run_development_pair(*, case_ids, run_case, scope, queue_seconds,
                         leg_seconds, clock=time.monotonic):
    """Injected development sequencing, not a concrete CLI/GPU launcher.

    A synchronous callback must itself use owned-worker supervision with the
    supplied deadline. This boundary cannot interrupt a blocking callback.
    Only fixed completion fields are accepted; arbitrary payloads are omitted.
    """
    if scope != 'controlled_development' or not callable(run_case):
        raise ValueError('controlled development callback required')
    if (not isinstance(case_ids,(list,tuple)) or not case_ids or
        any(not isinstance(case,str) or not case.strip() for case in case_ids) or
        len(set(case_ids)) != len(case_ids)):
        raise ValueError('nonempty unique fixed case ids required')
    if type(leg_seconds) not in (int,float) or not math.isfinite(leg_seconds) or leg_seconds<=0:
        raise ValueError('positive finite leg budget required')
    cases=tuple(case_ids)
    budget=SharedDeadlineBudget(queue_seconds,clock=clock)
    report={'schema':'tfd.development_pair.v1','scope':scope,
            'external_launch_ready':False,'status':'pending','cases':[]}
    for leg_id in ('baseline','candidate'):
        try:
            deadline=budget.begin_leg(leg_id,leg_seconds)
            for case_id in cases:
                if budget.remaining(deadline)<=0:
                    report['status']='shared_deadline_timeout'
                    return report
                value=run_case(leg_id=leg_id,case_id=case_id,absolute_deadline=deadline)
                row={'leg_id':leg_id,'case_id':case_id,'absolute_deadline':deadline}
                report['cases'].append(row)
                if budget.remaining(deadline)<=0:
                    report['status']='shared_deadline_timeout'
                    return report
                valid=(isinstance(value,dict) and value.get('schema')=='tfd.owned_worker_result.v1'
                       and value.get('status')=='complete' and value.get('reaped') is True
                       and type(value.get('exit_code')) is int and value['exit_code']==0
                       and value.get('external_launch_ready') is False
                       and value.get('scope')==scope)
                row['status']='complete' if valid else 'worker_result_rejected'
                if not valid:
                    report['status']='worker_result_rejected'
                    return report
        except TimeoutError:
            report['status']='shared_deadline_timeout'
            return report
        except Exception:
            report['status']='callback_or_clock_failure'
            return report
    report['status']='complete'
    return report


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_config(path, expected_sha256=None):
    path = Path(path)
    # Hash and parse the same bytes; child refuses parent/child config drift.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('config changed since parent preflight')
    value = json.loads(raw)
    fields = {'scope', 'prefix_path', 'prefix_sha256', 'model_path', 'adapter_path',
              'reference_path', 'session_id', 'system_prompt', 'runtime_manifest'}
    declared_fields=fields|{'controller_environment'}
    if not isinstance(value, dict) or set(value) not in (fields,declared_fields) or value['scope'] != 'controlled_development':
        raise ValueError('exact controlled development config required; independent pool forbidden')
    if any(not isinstance(value[k], str) or not value[k].strip() for k in fields - {'runtime_manifest'}):
        raise ValueError('nonempty development config strings required')
    if not isinstance(value['runtime_manifest'], dict):
        raise ValueError('runtime manifest mapping required')
    if 'controller_environment' in value:
        env=value['controller_environment']
        common=dict(TFD_CONTROLLED_NEAR_END_REPLAY='1',TFD_TURN_MLP_ENABLED='1',
            TFD_TRUST_GATE_ENABLED='1',TFD_TURN_WAIT_MODE='off',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',TFD_NATIVE_ACTION_MODE='residual')
        keys=set(common)|{'TFD_NATIVE_ACTION_HEAD','TFD_RESPONSE_LIFECYCLE'}
        if (not isinstance(env,dict) or set(env)!=keys or
            any(env.get(k)!=v for k,v in common.items()) or
            not isinstance(env['TFD_NATIVE_ACTION_HEAD'],str) or
            env['TFD_RESPONSE_LIFECYCLE'] not in {'0','1'}):
            raise ValueError('exact declared controlled feature environment required')
    # Explicit absolute paths avoid parent/child current-directory ambiguity.
    prefix = Path(value['prefix_path'])
    if not prefix.is_absolute() or not prefix.is_file() or sha256(prefix) != value['prefix_sha256']:
        raise ValueError('absolute pinned local development prefix required')
    return value, digest


def validate_development_pair_configs(*, baseline_configs, candidate_configs):
    """Read-only fixed paired development preflight; no launch or decoding.

    Pins/checksums are not proof of a trained seed, semantic asset completeness,
    successful runtime, or permission to decode an independent dataset.
    """
    if (not isinstance(baseline_configs,(list,tuple)) or not baseline_configs or
        not isinstance(candidate_configs,(list,tuple)) or
        len(baseline_configs)!=len(candidate_configs)):
        raise ValueError('same nonempty fixed case count required')
    report=dict(scope='controlled_development',external_launch_ready=False,cases=[])
    seen=set()
    for left,right in zip(baseline_configs,candidate_configs):
        values=[]
        refs=[]
        for entry in (left,right):
            if (not isinstance(entry,(list,tuple)) or len(entry)!=2 or
                not isinstance(entry[1],str) or len(entry[1])!=64):
                raise ValueError('explicit config path and SHA required')
            path=Path(entry[0]).resolve()
            value,digest=read_config(path,entry[1])
            if 'controller_environment' not in value:
                raise ValueError('paired configs must declare feature environment')
            values.append(value)
            refs.append(dict(config_path=str(path),config_sha256=digest))
        baseline,candidate=values
        if ({k:v for k,v in baseline.items() if k!='controller_environment'} !=
            {k:v for k,v in candidate.items() if k!='controller_environment'}):
            raise ValueError('both legs require identical input, identity, prompt and assets')
        a,b=baseline['controller_environment'],candidate['controller_environment']
        if a['TFD_NATIVE_ACTION_HEAD']!='' or a['TFD_RESPONSE_LIFECYCLE']!='0' or b['TFD_RESPONSE_LIFECYCLE']!='1':
            raise ValueError('fixed baseline off and candidate lifecycle on required')
        head=Path(b['TFD_NATIVE_ACTION_HEAD'])
        if not head.is_absolute() or not head.is_file():
            raise ValueError('candidate requires absolute local head')
        assets=candidate['runtime_manifest'].get('artifacts')
        if not isinstance(assets,list):
            raise ValueError('candidate head must be declared in shared asset inventory')
        matches=[row for row in assets if isinstance(row,dict) and
                 isinstance(row.get('path'),str) and Path(row['path']).resolve()==head.resolve()]
        digest=sha256(head)
        if len(matches)!=1 or matches[0].get('expected_sha256')!=digest:
            raise ValueError('one predeclared matching candidate head SHA required')
        case_id=baseline['session_id']
        if case_id in seen:
            raise ValueError('unique case identity required')
        seen.add(case_id)
        report['cases'].append(dict(case_id=case_id,baseline=refs[0],candidate=refs[1],head_sha256=digest))
    return report


def run_owned_development_pair(*, baseline_configs, candidate_configs,
                               expected_cli_sha256, output, queue_seconds,
                               leg_seconds, prepare_seconds, replay_seconds,
                               deadline_seconds, no_progress_seconds):
    """Fixed supervised development pair, never an independent-pool launcher.

    Read-only initial preflight precedes the queue clock. Rechecks and owned
    children share that clock; synchronous filesystem IO is not interruptible.
    Declared environment binding does not attest actual runtime feature use.
    """
    for value in (queue_seconds, leg_seconds, prepare_seconds, replay_seconds,
                  deadline_seconds, no_progress_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('positive finite budgets required')
    if (not isinstance(expected_cli_sha256, str) or len(expected_cli_sha256) != 64
        or any(c not in '0123456789abcdef' for c in expected_cli_sha256)
        or sha256(__file__) != expected_cli_sha256):
        raise ValueError('pinned current CLI source required')
    plan = validate_development_pair_configs(
        baseline_configs=baseline_configs, candidate_configs=candidate_configs)
    frozen = {leg: tuple((row[leg]['config_path'], row[leg]['config_sha256'])
                         for row in plan['cases'])
              for leg in ('baseline', 'candidate')}
    case_ids = tuple(row['case_id'] for row in plan['cases'])
    indices = {case: index for index, case in enumerate(case_ids)}
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)

    def run_case(*, leg_id, case_id, absolute_deadline):
        validate_development_pair_configs(baseline_configs=frozen['baseline'],
                                          candidate_configs=frozen['candidate'])
        if sha256(__file__) != expected_cli_sha256:
            raise ValueError('CLI drift before child')
        index = indices[case_id]
        path, digest = frozen[leg_id][index]
        config, _ = read_config(path, digest)
        value = run_owned_development_case(
            config_path=path, expected_config_sha256=digest,
            expected_cli_sha256=expected_cli_sha256,
            output=root / leg_id / f'{index:06d}',
            prepare_seconds=prepare_seconds, replay_seconds=replay_seconds,
            deadline_seconds=deadline_seconds, no_progress_seconds=no_progress_seconds,
            absolute_deadline=absolute_deadline)
        if isinstance(value, dict) and value.get('status') == 'complete':
            if (value.get('config_sha256') != digest or
                value.get('cli_sha256') != expected_cli_sha256 or
                value.get('declared_controller_environment') != config['controller_environment']):
                value = dict(value, status='worker_binding_rejected')
        return value

    report = run_development_pair(case_ids=case_ids, run_case=run_case,
        scope='controlled_development', queue_seconds=queue_seconds,
        leg_seconds=leg_seconds)
    report['preflight'] = plan
    report['cli_sha256'] = expected_cli_sha256
    with (root / 'pair_result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return report


class Recorder:
    def __init__(self):
        self.events = []

    async def send_json(self, event):
        self.events.append(event)


def run_worker(config_path, config_sha256, progress_path, result_path, replay_seconds):
    config, _ = read_config(config_path, config_sha256)
    # Only preprepared 30s/16k mono development NPY; never decode the external pool.
    with Path(config['prefix_path']).open('rb') as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != config['prefix_sha256']:
        raise ValueError('development prefix changed before decode')
    import io
    prefix = np.load(io.BytesIO(raw), allow_pickle=False)
    list(replay_packets(prefix))
    # This entry executes inside one owned child; clear inherited head/lifecycle
    # via the exact allowlisted declaration, never modify the parent Demo.
    if 'controller_environment' in config:
        os.environ.update(config['controller_environment'])
    recorder = Recorder()
    def load_stack():
        return load_controlled_runtime(**{k: config[k] for k in (
            'model_path', 'adapter_path', 'reference_path', 'session_id',
            'system_prompt', 'runtime_manifest')}, recorder=recorder)
    outcome = asyncio.run(run_prefix_worker_case(None, None, None, recorder, prefix,
        None, None, progress_path=progress_path, result_path=result_path,
        max_seconds=replay_seconds, load_stack=load_stack))
    return 0 if outcome['input_complete'] else 1


def run_owned_development_case(*, config_path, expected_config_sha256,
                               expected_cli_sha256, output, prepare_seconds,
                               replay_seconds, deadline_seconds, no_progress_seconds,
                               absolute_deadline=None):
    """Pinned local development child; no arbitrary command or independent pool.

    CLI-file pin is not an attestation of imported/native dependency closure.
    Absolute deadlines use the same host monotonic clock as the queue owner.
    """
    budgets=[prepare_seconds,replay_seconds,deadline_seconds,no_progress_seconds]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in budgets):
        raise ValueError('positive finite budgets required before output creation')
    for pin in (expected_config_sha256,expected_cli_sha256):
        if not isinstance(pin,str) or len(pin)!=64 or any(c not in '0123456789abcdefABCDEF' for c in pin):
            raise ValueError('explicit SHA256 pins required')
    config_path,output=Path(config_path).resolve(),Path(output).resolve()
    config,digest=read_config(config_path,expected_config_sha256)
    cli_path=Path(__file__).resolve()
    source_digest=sha256(cli_path)
    if source_digest!=expected_cli_sha256:
        raise ValueError('CLI source changed since preflight')
    if absolute_deadline is not None:
        if type(absolute_deadline) not in (int,float) or not math.isfinite(absolute_deadline):
            raise ValueError('finite owner absolute deadline required')
        if time.monotonic()>=absolute_deadline:
            raise TimeoutError('shared deadline exhausted before output creation')
    output.mkdir(parents=True,exist_ok=False)
    child=[sys.executable,str(cli_path),'--config',str(config_path),'--output',str(output),
           '--worker-config-sha256',digest,'--worker-source-sha256',source_digest]
    for name,value in zip(('prepare','replay','deadline','no-progress'),budgets):
        child.extend(['--'+name+'-seconds',str(value)])
    result=run_owned_worker(child,log_path=output/'worker.log',
        progress_path=output/'progress.jsonl',expected_packets=45,expected_cases=1,
        deadline_seconds=deadline_seconds,no_progress_seconds=no_progress_seconds,
        phase_deadlines={'prepare':prepare_seconds,'replay':replay_seconds},
        checkpoint_path=output/'checkpoint.jsonl',prefix_result_path=output/'prefix_result.json',
        absolute_deadline=absolute_deadline)
    result['config_sha256']=digest
    result['cli_sha256']=source_digest
    result['scope']='controlled_development'
    if 'controller_environment' in config:
        result['declared_controller_environment']=dict(config['controller_environment'])
    with (output/'parent_result.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,allow_nan=False,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    for name in ('prepare', 'replay', 'deadline', 'no-progress'):
        parser.add_argument('--'+name+'-seconds', required=True, type=float)
    parser.add_argument('--worker-config-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--worker-source-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    budgets = [args.prepare_seconds, args.replay_seconds, args.deadline_seconds, args.no_progress_seconds]
    if any(not math.isfinite(value) or value <= 0 for value in budgets):
        raise ValueError('positive finite budgets required before output creation')
    config_path, output = args.config.resolve(), args.output.resolve()
    _, digest = read_config(config_path, args.worker_config_sha256)
    if args.worker_source_sha256 is not None and sha256(__file__)!=args.worker_source_sha256:
        raise ValueError('CLI source changed before worker execution')
    if args.worker_config_sha256 is not None:
        return run_worker(config_path, digest, output/'progress.jsonl',
                          output/'prefix_result.json', args.replay_seconds)
    result=run_owned_development_case(config_path=config_path,
        expected_config_sha256=digest,expected_cli_sha256=sha256(__file__),output=output,
        prepare_seconds=args.prepare_seconds,replay_seconds=args.replay_seconds,
        deadline_seconds=args.deadline_seconds,no_progress_seconds=args.no_progress_seconds)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
