"""Controlled-only backend/session wiring derived from 174.

Includes a selected-file concrete runtime entry and injected composition.
Neither is a complete dependency audit or permission to use raw VAD on a
microphone. Session construction still loads the actual controllers.
"""
import base64
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import numpy as np
from .duplex_model_loader import load_duplex_model_core
from .stack_artifact_manifest import (collect_artifacts, collect_runtime_versions,
                                      collect_loaded_module_origins)


RUNTIME_MODULES = ('torch', 'librosa', 'MiniCPMO45.modeling_minicpmo_unified',
    'core.processors.unified', 'core.schemas.duplex', 'py_backend.tfd_runtime',
    'core.processors.pytorch_backend', 'py_backend.server', 'silero_vad',
    'tfd.duplex_policy.causal_vad_gate', 'faster_whisper', 'faster_whisper.transcribe',
    'stepaudio2', 'ctranslate2', 'tokenizers', 'MiniCPMO45.processing_minicpmo',
    'py_backend.tfd_native_action', 'py_backend.tfd_response_lifecycle',
    'tfd.duplex_policy.native_action_bridge', 'tfd.duplex_policy.action_residual')
RUNTIME_DISTRIBUTIONS = ('torch', 'transformers', 'accelerate', 'peft', 'librosa',
    'silero-vad', 'onnxruntime', 'faster-whisper', 'numpy', 'scipy', 'soundfile',
    'PyYAML', 'opencc-python-reimplemented', 'ctranslate2', 'tokenizers',
    'stepaudio2-minicpmo', 'Pillow')


def load_controlled_runtime(*, model_path, adapter_path, reference_path,
                            recorder, session_id, system_prompt, runtime_manifest):
    """Concrete imports/bindings for load_stack, only in an owned worker.

    Caller configures the pinned module search path before entry. No sys.path
    edits/downloads/auto-locking. Imports execute code (including parents), so
    these selected-file checks are not a sandbox or recursive dependency audit.
    """
    required_env = dict(TFD_CONTROLLED_NEAR_END_REPLAY='1', TFD_TURN_MLP_ENABLED='1',
        TFD_TRUST_GATE_ENABLED='1', TFD_TURN_WAIT_MODE='off',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if any(os.environ.get(k)!=v for k,v in required_env.items()) or os.environ.get('TFD_RESPONSE_LIFECYCLE') not in {'0','1'}:
        raise ValueError('controlled explicit leg required before runtime import')
    # These known library caches can outlive an environment change. Do not
    # import to inspect them; refuse an observed conflict before runtime loads.
    # Unknown installed-version internals still require the runtime canary.
    for name, key in (('huggingface_hub.constants','HF_HUB_OFFLINE'),
                      ('transformers.utils.hub','_is_offline_mode')):
        module = sys.modules.get(name)
        if module is not None and key in vars(module) and vars(module)[key] is not True:
            raise ValueError('already-loaded offline cache conflicts with controlled environment')
    if not isinstance(runtime_manifest, dict) or set(runtime_manifest)!={'artifacts','versions','modules'}:
        raise ValueError('explicit runtime manifest required')
    modules = list(runtime_manifest['modules'])
    # Validate all pin syntax/duplicates before importing any runtime module.
    collect_loaded_module_origins(modules, module_provider=lambda n:None)
    pins = {r['module']:r for r in modules}
    if not set(RUNTIME_MODULES).issubset(pins):
        raise ValueError('all concrete runtime module pins required')
    artifacts = list(runtime_manifest['artifacts'])
    if not artifacts or any(not isinstance(r,dict) or r.get('expected_sha256') is None for r in artifacts):
        raise ValueError('predeclared asset SHA required')
    # Match the actual Demo controller path resolution, including its defaults.
    # ASR/model/TTS file closure is not established by these two asset checks.
    controller_keys = ('TFD_PROJECT_ROOT','TFD_TURN_MLP_WEIGHTS','TFD_GATE_CONFIG','TFD_ASR_MODEL',
                       'TFD_NATIVE_ACTION_HEAD','TFD_NATIVE_ACTION_MODE',
                       'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')
    controller_env = {key:os.environ.get(key) for key in controller_keys}
    project_root = Path(os.environ.get('TFD_PROJECT_ROOT',
        '/root/autodl-tmp/trusted-full-duplex-agent')).expanduser().resolve()
    mlp_path = Path(os.environ.get('TFD_TURN_MLP_WEIGHTS',
        str(project_root/'outputs/turntaking/learned_mlp.pt'))).expanduser().resolve()
    gate_path = Path(os.environ.get('TFD_GATE_CONFIG',
        str(project_root/'configs/gate.yaml'))).expanduser().resolve()
    asr_root = Path(os.environ.get('TFD_ASR_MODEL',
        str(project_root/'weights/faster-whisper-small'))).expanduser().resolve()
    required_paths = {(Path(model_path)/'config.json').resolve(),
        (Path(adapter_path)/'adapter_model.safetensors').resolve(), Path(reference_path).resolve(),
        (Path(pins['silero_vad']['expected_path']).parent/'data/silero_vad.onnx').resolve(),
        mlp_path, gate_path, asr_root/'model.bin', asr_root/'config.json',
        asr_root/'tokenizer.json'}
    head = os.environ.get('TFD_NATIVE_ACTION_HEAD','').strip()
    if head:
        required_paths.add(Path(head).expanduser().resolve())
    if not required_paths.issubset({Path(r['path']).resolve() for r in artifacts}):
        raise ValueError('config, adapter, reference, Silero ONNX, MLP and gate must be pinned')
    sources = [dict(role='module:'+r['module'],path=r['expected_path'],
                    expected_sha256=r['expected_sha256']) for r in modules]
    assets = collect_artifacts(artifacts)
    source_check = collect_artifacts(sources)
    if not assets['artifacts_complete'] or not source_check['artifacts_complete']:
        raise ValueError('runtime asset/source gate failed')
    pinned_paths = {Path(r['path']).resolve() for r in artifacts}
    model_root = Path(model_path).resolve()
    # Cover all currently present local ASR/TTS files, not merely a directory
    # marker. This inventory gate does not certify missing semantic assets.
    # Include every present model-root file, so tokenizer/processor artifacts
    # cannot silently escape the inventory. Presence is not semantic validity
    # or proof that the installed processor needs no additional files/cache.
    for root in (asr_root,model_root/'assets/token2wav',model_root):
        if not root.is_dir():
            raise ValueError('local ASR and token2wav directories required')
        files = []
        for path in root.rglob('*'):
            if path.is_symlink() and path.is_dir():
                raise ValueError('asset directory symlinks are not supported')
            if path.is_file():
                resolved=path.resolve()
                if not resolved.is_relative_to(root) or resolved not in pinned_paths:
                    raise ValueError('every local ASR/TTS file must be pinned within its root')
                files.append(resolved)
        if not files:
            raise ValueError('nonempty local ASR/TTS inventory required')
    with (model_root/'config.json').open(encoding='utf-8') as handle:
        model_config = json.load(handle)
    if not isinstance(model_config,dict) or model_config.get('transformers_weights') is not None:
        raise ValueError('custom model checkpoint selection is not supported by this asset gate')
    single = model_root/'model.safetensors'
    index = model_root/'model.safetensors.index.json'
    # Only this experimental concrete entry accepts local safetensors.
    # Preserve the injected/legacy loader's existing selection behavior.
    if single.is_file():
        if single.resolve() not in pinned_paths:
            raise ValueError('base safetensors must be pinned')
    else:
        if not index.is_file() or index.resolve() not in pinned_paths:
            raise ValueError('local pinned safetensors index required')
        with index.open(encoding='utf-8') as handle:
            checkpoint_index = json.load(handle)
        weight_map = checkpoint_index.get('weight_map') if isinstance(checkpoint_index,dict) else None
        if not isinstance(weight_map,dict) or not weight_map:
            raise ValueError('nonempty checkpoint weight_map required')
        for filename in weight_map.values():
            if (not isinstance(filename,str) or not filename or '\\' in filename
                    or PurePosixPath(filename).is_absolute() or PureWindowsPath(filename).drive
                    or any(part in {'..','.'} for part in filename.split('/'))
                    or not filename.endswith('.safetensors')):
                raise ValueError('safe relative safetensors shard path required')
            shard = (model_root/filename).resolve()
            if not shard.is_relative_to(model_root) or shard not in pinned_paths or not shard.is_file():
                raise ValueError('every local model shard must be pinned within model root')
    versions = runtime_manifest['versions']
    if not isinstance(versions,dict) or not set(RUNTIME_DISTRIBUTIONS).issubset(versions):
        raise ValueError('all concrete runtime distribution versions required')
    version_check = collect_runtime_versions(versions,expected_versions=versions)
    if not assets['artifacts_complete'] or not source_check['artifacts_complete'] or not version_check['versions_match']:
        raise ValueError('runtime pre-import gate failed')
    loaded = {name:importlib.import_module(name) for name in RUNTIME_MODULES}
    origins = collect_loaded_module_origins(modules)
    if not origins['origins_match']:
        raise ValueError('imported runtime origin gate failed')
    omni = loaded['MiniCPMO45.modeling_minicpmo_unified']
    # Package-root SHA does not identify a re-export's defining module. Check
    # the selected classes after origin verification, before constructing any
    # model/controller. Extra implementation module pins must be predeclared;
    # package imports must already have loaded them (never discover/auto-pin).
    exports = (
        ('MiniCPMO45.modeling_minicpmo_unified', 'MiniCPMO'),
        ('MiniCPMO45.modeling_minicpmo_unified', 'MiniCPMOProcessor'),
        ('faster_whisper', 'WhisperModel'), ('stepaudio2', 'Token2wav'))
    class_exports = []
    for package, attribute in exports:
        value = getattr(loaded[package], attribute, None)
        implementation = getattr(value, '__module__', None)
        if not inspect.isclass(value) or implementation not in pins:
            raise ValueError('selected class implementation must be predeclared and pinned')
        try:
            path = Path(inspect.getfile(value)).resolve()
        except (TypeError, OSError) as exc:
            raise ValueError('selected class must have a verified file origin') from exc
        if path != Path(pins[implementation]['expected_path']).resolve():
            raise ValueError('selected class implementation path mismatch')
        class_exports.append(dict(package=package, attribute=attribute,
            implementation_module=implementation, implementation_path=str(path)))
    if (omni.MiniCPMOProcessor is not loaded['MiniCPMO45.processing_minicpmo'].MiniCPMOProcessor
            or loaded['faster_whisper'].WhisperModel is not loaded['faster_whisper.transcribe'].WhisperModel):
        raise ValueError('selected class aliases do not match verified implementations')
    # A pinned module may re-export a different function. Check the exact
    # selected feature definitions and the server's already-bound aliases.
    # File origins are metadata, not authentication of in-memory bytecode.
    feature_exports = []
    for package, attribute in (
            ('py_backend.tfd_native_action','build_native_action_bridge_from_env'),
            ('py_backend.tfd_response_lifecycle','allow_client_cancel'),
            ('py_backend.tfd_response_lifecycle','listen_transition'),
            ('tfd.duplex_policy.native_action_bridge','load_native_action_bridge')):
        value = getattr(loaded[package], attribute, None)
        if not inspect.isfunction(value) or value.__module__ != package:
            raise ValueError('selected feature function must be defined in its pinned module')
        path = Path(inspect.getfile(value)).resolve()
        if path != Path(pins[package]['expected_path']).resolve():
            raise ValueError('selected feature function path mismatch')
        feature_exports.append(dict(package=package, attribute=attribute,
            implementation_module=package, implementation_path=str(path)))
    residual = loaded['tfd.duplex_policy.action_residual']
    bridge = loaded['tfd.duplex_policy.native_action_bridge']
    value = getattr(residual, 'ActionResidual', None)
    if (not inspect.isclass(value) or value.__module__ != 'tfd.duplex_policy.action_residual'
            or Path(inspect.getfile(value)).resolve() != Path(pins['tfd.duplex_policy.action_residual']['expected_path']).resolve()
            or getattr(bridge, 'ActionResidual', None) is not value):
        raise ValueError('selected residual class binding mismatch')
    feature_exports.append(dict(package='tfd.duplex_policy.action_residual',attribute='ActionResidual',
        implementation_module=value.__module__,implementation_path=str(Path(inspect.getfile(value)).resolve())))
    residual_class = value
    bridge_class = getattr(bridge, 'NativeActionBridge', None)
    bridge_package = 'tfd.duplex_policy.native_action_bridge'
    if (not inspect.isclass(bridge_class) or bridge_class.__module__ != bridge_package
            or Path(inspect.getfile(bridge_class)).resolve() != Path(pins[bridge_package]['expected_path']).resolve()):
        raise ValueError('selected bridge class binding mismatch')
    feature_exports.append(dict(package=bridge_package,attribute='NativeActionBridge',
        implementation_module=bridge_package,implementation_path=str(Path(inspect.getfile(bridge_class)).resolve())))
    for attribute in ('allow_client_cancel','listen_transition'):
        if getattr(loaded['py_backend.server'], attribute, None) is not getattr(loaded['py_backend.tfd_response_lifecycle'], attribute):
            raise ValueError('selected lifecycle server alias mismatch')
    feature_bindings = [(row['package'],row['attribute'],
                         getattr(loaded[row['package']],row['attribute'])) for row in feature_exports]
    feature_bindings.extend((package,attribute,getattr(loaded[package],attribute))
        for package,attribute in (('tfd.duplex_policy.native_action_bridge','ActionResidual'),
            ('py_backend.server','allow_client_cancel'),('py_backend.server','listen_transition')))
    def check_feature_bindings():
        # Detect persistent late import/rebinding at trusted boundary checks.
        # This is not a sandbox: in-place code/globals changes or temporary
        # swaps restored between checks are outside this identity guarantee.
        for package,attribute,value in feature_bindings:
            if (sys.modules.get(package) is not loaded[package]
                    or getattr(loaded[package],attribute,None) is not value):
                raise ValueError('verified selected feature binding changed')
    result = compose_controlled_stack(
        load_core=lambda:load_duplex_model_core(model_class=omni.MiniCPMO,
            processor_mode=omni.ProcessorMode, duplex_view=loaded['core.processors.unified'].DuplexView,
            duplex_config=loaded['core.schemas.duplex'].DuplexConfig,
            load_adapter=loaded['py_backend.tfd_runtime'].load_grpo_adapter,
            model_path=model_path,adapter_path=adapter_path,dtype=loaded['torch'].bfloat16,
            runtime_manifest=runtime_manifest,safetensors_only=True,local_asset_root=model_root,
            processor_class=omni.MiniCPMOProcessor),
        load_vad=lambda:loaded['tfd.duplex_policy.causal_vad_gate'].CausalVadGate(
            loaded['silero_vad'].load_silero_vad(onnx=True)),
        backend_class=loaded['core.processors.pytorch_backend'].PyTorchBackend,
        server=loaded['py_backend.server'],recorder=recorder,session_id=session_id,
        model_path=model_path,reference_path=reference_path,load_audio=loaded['librosa'].load,
        system_prompt=system_prompt)
    result['runtime_audit'] = dict(schema='tfd.selected_runtime_audit.v1',artifacts=assets,source_files=source_check,
        versions=version_check,modules=origins,class_exports=class_exports,feature_exports=feature_exports,controller_environment=controller_env,
        external_launch_ready=False)
    model, vad, factory = result['components']
    selected_sessions = []
    def checked_session():
        if any(os.environ.get(key)!=value for key,value in controller_env.items()):
            raise ValueError('verified controller asset environment changed')
        check_feature_bindings()
        session = factory()
        check_feature_bindings()
        selected_sessions.append(session)
        return session
    result['components'] = (model, vad, checked_session)
    audit = result['feature_audit']
    def checked_feature_audit():
        check_feature_bindings()
        evidence = audit()
        if evidence['native_head_enabled']:
            # Exact types, not duck typing/subclasses. Identity does not
            # authenticate object internals or prove checkpoint semantics.
            if len(selected_sessions) != 1:
                raise ValueError('one selected feature instance session required')
            instance = getattr(selected_sessions[0].backend, 'tfd_native_action_bridge', None)
            if type(instance) is not bridge_class or type(getattr(instance, 'head', None)) is not residual_class:
                raise ValueError('selected bridge/head instance type mismatch')
        check_feature_bindings()
        return dict(evidence, selected_feature_bindings_unchanged=True,
                    selected_feature_instance_types_match=True)
    result['feature_audit'] = checked_feature_audit
    return result


def compose_controlled_stack(*, load_core, load_vad, backend_class, server,
                            recorder, session_id, model_path, reference_path,
                            load_audio, system_prompt):
    required = dict(TFD_CONTROLLED_NEAR_END_REPLAY='1', TFD_TURN_MLP_ENABLED='1',
                    TFD_TRUST_GATE_ENABLED='1', TFD_TURN_WAIT_MODE='off')
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError('explicit controlled full stack configuration required')
    lifecycle = os.environ.get('TFD_RESPONSE_LIFECYCLE')
    if lifecycle not in {'0', '1'}:
        raise ValueError('explicit lifecycle leg required')
    feature_env = {key: os.environ.get(key) for key in
                   ('TFD_NATIVE_ACTION_HEAD', 'TFD_NATIVE_ACTION_MODE')}
    head_path = (feature_env['TFD_NATIVE_ACTION_HEAD'] or '').strip()
    head_sha = hashlib.sha256(Path(head_path).read_bytes()).hexdigest() if head_path else None
    head_mode = feature_env['TFD_NATIVE_ACTION_MODE'] or 'residual'
    if not isinstance(session_id, str) or not session_id or not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError('case identity and frozen prompt required')
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference, rate = load_audio(reference_path, sr=16000, mono=True)
    reference = np.asarray(reference, dtype=np.float32)
    if rate != 16000 or reference.ndim != 1 or not reference.size or not np.isfinite(reference).all():
        raise ValueError('finite nonempty mono 16k reference required')
    voice = base64.b64encode(reference.tobytes()).decode('ascii')

    core = load_core()
    model, view = core['model'], core['view']
    if not callable(getattr(model, 'reset_session', None)) or view is None:
        raise ValueError('valid model core required')
    vad = load_vad()
    if not callable(getattr(vad, 'reset', None)) or not callable(getattr(vad, 'feed', None)):
        raise ValueError('causal VAD interfaces required')
    backend = backend_class(str(model_path), 0)
    backend.processor = SimpleNamespace(model=model, set_duplex_mode=lambda: view,
                                        kv_cache_length=0)
    created = []

    def fresh_session():
        # Fail on environment drift instead of silently changing either leg.
        if any(os.environ.get(k) != v for k, v in required.items()) or os.environ.get('TFD_RESPONSE_LIFECYCLE') != lifecycle:
            raise ValueError('controlled configuration changed')
        session = server.BackendProtocolSession(session_id=session_id,
            mode='full_duplex', backend=backend, ws=recorder,
            state=server.BackendServerState(backend))
        if getattr(session, '_turn_controller', None) is None or getattr(session, '_trust_gate', None) is None:
            raise ValueError('required controller missing')
        if not getattr(session, '_controlled_near_end_replay', False) or getattr(session, '_response_lifecycle_enabled', None) is not (lifecycle == '1'):
            raise ValueError('session control flags do not match selected leg')
        bridge = getattr(backend, 'tfd_native_action_bridge', None)
        initial = getattr(bridge, 'applied_chunks', 0)
        if type(initial) is not int or initial < 0:
            raise ValueError('invalid initial head counter')
        created.append((session, initial))
        return session

    def feature_audit():
        # Observe selected objects after replay, not just their environment.
        # Counters prove this bridge reports fresh use, not semantic benefit.
        if (len(created) != 1 or any(os.environ.get(k) != v for k,v in feature_env.items())
            or any(os.environ.get(k) != v for k,v in required.items())
            or os.environ.get('TFD_RESPONSE_LIFECYCLE') != lifecycle):
            raise ValueError('one unchanged selected feature session required')
        session, initial = created[0]
        if (session._response_lifecycle_enabled is not (lifecycle == '1') or
            session._controlled_near_end_replay is not True):
            raise ValueError('observed session flags changed')
        bridge = getattr(backend, 'tfd_native_action_bridge', None)
        delta = 0
        if head_path:
            count = getattr(bridge, 'applied_chunks', None)
            if (bridge is None or type(count) is not int or count <= initial or
                getattr(bridge, 'mode', None) != head_mode or
                getattr(bridge, 'checkpoint_sha256', None) != head_sha or
                getattr(model, 'duplex', None) is None or
                getattr(bridge, 'duplex', None) is not model.duplex or
                hashlib.sha256(Path(head_path).read_bytes()).hexdigest() != head_sha):
                raise ValueError('selected head must report fresh matching application')
            delta = count - initial
        elif bridge is not None:
            raise ValueError('baseline must have no native head bridge')
        return dict(schema='tfd.controlled_feature_audit.v1',external_launch_ready=False,
            native_head_enabled=bool(head_path),head_sha256=head_sha,head_mode=head_mode,
            applied_chunks_delta=delta,lifecycle_enabled=session._response_lifecycle_enabled,
            controlled_near_end_replay=True)

    return dict(components=(model, vad, fresh_session), near_speech=vad.feed,
        feature_audit=feature_audit,
        init_params=dict(system_prompt=system_prompt, tts_ref_audio_base64=voice,
                         config=dict(decode_mode='greedy', force_listen_count=3)))
