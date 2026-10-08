from tfd.duplex_policy.response_lifecycle import listen_transition, allow_client_cancel


def test_ordinary_listen_preserves_response_and_never_cancels_playback():
    result = listen_transition('r1', enabled=True)
    assert result['active_response_id'] == 'r1'
    assert result['reason'] == 'model_listen'
    assert result['cancel_playback'] is False


def test_native_end_releases_response_without_cancelling_queued_audio():
    result = listen_transition('r1', enabled=True, natural_end=True)
    assert result['active_response_id'] is None
    assert result['reason'] == 'turn_end'
    assert result['cancel_playback'] is False


def test_verified_cancel_releases_id_but_mlp_listen_does_not():
    result = listen_transition('r1', enabled=True, client_cancel=True)
    assert result['cancel_playback'] is True
    assert result['active_response_id'] is None
    assert result['reason'] == 'force_listen'
    assert listen_transition('r1', enabled=True, force_listen=True)['active_response_id'] == 'r1'


def test_legacy_default_preserved_and_raw_mic_cancel_not_verified():
    assert listen_transition('r1', enabled=False)['active_response_id'] is None
    assert allow_client_cancel(True, enabled=False, active=True, controlled_replay=False) is True
    assert allow_client_cancel(True, enabled=True, active=True, controlled_replay=False) is False
    assert allow_client_cancel(True, enabled=True, active=True, controlled_replay=True) is True


def test_actual_server_push_keeps_listen_id_and_rejects_raw_cancel():
    """Execute the exact server method with only transport/model boundaries faked."""
    import ast
    import asyncio
    import time
    import uuid
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Any, Dict
    source = Path(__file__).resolve().parents[1] / 'tests/fixtures/e58_push_full_duplex.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == '_push_full_duplex')
    calls, outputs = [], []
    result = SimpleNamespace(is_listen=True, end_of_turn=False)
    backend = SimpleNamespace(duplex_prefill=lambda **kw: {}, duplex_generate=lambda **kw: (calls.append(kw) or result))
    env = dict(asyncio=asyncio, time=time, uuid=uuid, Any=Any, Dict=Dict,
               logger=SimpleNamespace(info=lambda *args: None),
               _extract_audio_base64=lambda p: 'audio', _extract_frame_base64_list=lambda p: [],
               decode_audio_base64=lambda p: [0.], decode_frame_base64_list=lambda p: SimpleNamespace(frame_list=[]),
               _first_dict=lambda p: p or {}, _coalesce=lambda *args, default=None: next((v for v in args if v is not None), default),
               _result_metrics=lambda *args: {}, allow_client_cancel=allow_client_cancel,
               listen_transition=listen_transition,
               TokenUsage=SimpleNamespace(from_duplex_chunk=lambda *args: SimpleNamespace(to_dict=lambda: {})))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), env)
    async def no_wait():
        pass
    async def send(*args, **kw):
        outputs.append(kw)
    session = SimpleNamespace(_op_lock=asyncio.Lock(), _wait_finalize=no_wait,
        _active_response_id='r1', _response_lifecycle_enabled=True, _controlled_near_end_replay=False,
        _turn_controller=None, _trust_gate=None, backend=backend, _safe_metrics=lambda: {},
        _usage_chunk_index=0, _usage=SimpleNamespace(add=lambda x: {}), send_output_delta=send,
        session_id='s1', _schedule_finalize=lambda: None)
    async def exercise():
        await env['_push_full_duplex'](session, {})
        assert session._active_response_id == 'r1'
        assert outputs[-1]['metrics']['tfd_gate_evaluated_this_chunk'] is False
        await env['_push_full_duplex'](session, {'force_listen': True})
        assert calls[-1]['force_listen'] is False
        assert outputs[-1]['metrics']['tfd_unverified_cancel_blocked'] is True
        assert session._active_response_id == 'r1'
        result.end_of_turn = True
        await env['_push_full_duplex'](session, {})
        assert outputs[-1]['reason'] == 'turn_end'
        assert session._active_response_id is None
        # Cached gate metrics must not masquerade as another ASR evaluation.
        evaluations = []
        turn = SimpleNamespace(allow_speak=True, voiced_frames=5, trailing_silence_frames=3, action='speak',
                               metrics=lambda: {})
        gate = SimpleNamespace(should_intercept=False, allow_execute=True, listen_only=False,
                               metrics=lambda: {'tfd_gate_action': 'execute'},
                               action='execute', text='hello', confidence=1., risk=0., asr_ms=2.)
        session._active_response_id = None
        session._turn_controller = SimpleNamespace(feed=lambda audio: turn, reset=lambda: None)
        session._trust_gate = SimpleNamespace(feed=lambda audio: None, reset=lambda: None,
                        evaluate=lambda: (evaluations.append(1) or gate))
        session._gate_released_voiced_frames = -1
        session._wait_mode = 'off'
        session._wait_gate_config = None
        session._safe_metrics = lambda: {'tfd_gate_action': 'execute'}
        env['_result_metrics'] = lambda result, metrics: dict(metrics)
        env['evaluate_wait_gate'] = lambda **kwargs: SimpleNamespace(hold=False, metrics=lambda: {})
        await env['_push_full_duplex'](session, {})
        assert outputs[-1]['metrics']['tfd_gate_evaluated_this_chunk'] is True
        session._active_response_id = 'r3'
        await env['_push_full_duplex'](session, {})
        assert outputs[-1]['metrics']['tfd_gate_action'] == 'execute'
        assert outputs[-1]['metrics']['tfd_gate_evaluated_this_chunk'] is False
        assert len(evaluations) == 1
        result.end_of_turn = False
        session._active_response_id = 'r2'
        session._controlled_near_end_replay = True
        await env['_push_full_duplex'](session, {'force_listen': True})
        assert calls[-1]['force_listen'] is True
        assert outputs[-1]['response_id'] == 'r2'
        assert session._active_response_id is None
        # Branch coverage only: decisions are injected here, not ASR quality.
        resets = []
        session._turn_controller.reset = lambda: resets.append('turn')
        session._trust_gate.reset = lambda: resets.append('gate')
        env['_guardrail_response'] = lambda action: ('guard:' + action, '')
        for action in ['clarify', 'stop', 'listen']:
            result.end_of_turn = False
            session._active_response_id = None
            session._gate_released_voiced_frames = -1
            gate.action = action
            gate.should_intercept = action in {'clarify', 'stop'}
            gate.allow_execute = False
            gate.listen_only = action == 'listen'
            gate.metrics = lambda: {'tfd_gate_action': gate.action}
            resets.clear()
            offset = len(outputs)
            await env['_push_full_duplex'](session, {})
            assert outputs[-1]['metrics']['tfd_gate_evaluated_this_chunk'] is True
            assert calls[-1]['force_listen'] is True
            assert resets == ['turn', 'gate']
            assert session._gate_released_voiced_frames == -1
            assert session._active_response_id is None
            if action != 'listen':
                assert outputs[offset]['text'] == 'guard:' + action
                assert outputs[-1]['reason'] == 'guardrail_intercept'
            else:
                assert len(outputs) == offset + 1  # noise emits no canned clarification
    asyncio.run(exercise())
