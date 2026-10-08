"""Injectable extraction of E60's model core recipe, not a full stack loader.

Call only inside an owned worker. Factories must come from the pinned runtime;
this module neither imports GPU libraries nor downloads missing model assets.
"""
from pathlib import Path
from .stack_artifact_manifest import (collect_artifacts, collect_runtime_versions,
                                      collect_loaded_module_origins)


def load_duplex_model_core(*, model_class, processor_mode, duplex_view,
                           duplex_config, load_adapter, model_path,
                           adapter_path, dtype, runtime_manifest=None, safetensors_only=False,
                           local_asset_root=None, processor_class=None):
    """Optional experimental preflight, preserving the legacy recipe by default.

    Selected pins are caller-declared, not automatically generated or a full
    dependency closure. Imported factories must still be bound to those pins.
    Execute within the owned worker's preparation deadline, including hashing.
    """
    model_path, adapter_path = Path(model_path), Path(adapter_path)
    if processor_class is not None:
        if (local_asset_root is None
                or Path(local_asset_root).resolve()!=model_path.resolve()
                or not callable(getattr(processor_class,'from_pretrained',None))):
            raise ValueError('explicit processor requires matching local model root and factory')
    for asset in (model_path / 'config.json',
                  adapter_path / 'adapter_model.safetensors'):
        if not asset.is_file():
            raise FileNotFoundError(asset)

    preflight = None
    if runtime_manifest is not None:
        if not isinstance(runtime_manifest, dict) or set(runtime_manifest) != {'artifacts','versions','modules'}:
            raise ValueError('explicit artifacts, exact versions and loaded modules required')
        artifacts = list(runtime_manifest['artifacts'])
        if not artifacts or any(not isinstance(r, dict) or r.get('expected_sha256') is None for r in artifacts):
            raise ValueError('every selected artifact requires a predeclared SHA256')
        required = {(model_path/'config.json').resolve(),
                    (adapter_path/'adapter_model.safetensors').resolve()}
        if not required.issubset({Path(r['path']).resolve() for r in artifacts}):
            raise ValueError('model config and adapter weight must be covered by artifact pins')
        versions = runtime_manifest['versions']
        if not isinstance(versions, dict) or not versions:
            raise ValueError('nonempty exact distribution versions required')
        preflight = dict(artifacts=collect_artifacts(artifacts),
            versions=collect_runtime_versions(versions, expected_versions=versions),
            modules=collect_loaded_module_origins(runtime_manifest['modules']),
            external_launch_ready=False)
        preflight['selected_runtime_verified'] = (preflight['artifacts']['artifacts_complete']
            and preflight['versions']['versions_match'] and preflight['modules']['origins_match'])
        if not preflight['selected_runtime_verified']:
            raise ValueError('selected runtime preflight failed; model loading refused')

    model = model_class.from_pretrained(
        str(model_path), trust_remote_code=True, attn_implementation='sdpa',
        init_vision=False, init_tts=True, torch_dtype=dtype,
        **({'use_safetensors':True,'local_files_only':True} if safetensors_only else {})).eval().cuda()
    load_adapter(model, str(adapter_path))
    model.eval().requires_grad_(False)
    if local_asset_root is not None:
        asset_root=Path(local_asset_root).resolve()
        if asset_root!=model_path.resolve() or not callable(getattr(model,'_ensure_asset_dir',None)):
            raise ValueError('local model asset root and compatible resolver required')
        def local_assets(asset_subpath,model_dir=None):
            # Replace only the newly owned experimental model instance's
            # download-capable resolver; never change the class/default Demo.
            if asset_subpath!='assets/token2wav':
                raise ValueError('only local token2wav assets are allowed')
            local_dir=asset_root/'assets/token2wav'
            if model_dir is not None and Path(model_dir).expanduser().resolve()!=local_dir:
                raise ValueError('token2wav asset override refused')
            if not local_dir.is_dir():
                raise FileNotFoundError(local_dir)
            return str(local_dir)
        model._ensure_asset_dir=local_assets
    processor = None
    if processor_class is not None:
        # Bind before unified initialization reaches its lazy processor loader.
        # This opt-in affects only the newly owned experimental model instance.
        processor=processor_class.from_pretrained(str(model_path.resolve()),
            trust_remote_code=True,local_files_only=True)
        if processor is None:
            raise ValueError('local processor factory returned no processor')
        model.processor=processor
    model.init_unified(duplex_config=dict(generate_audio=True,
        ls_mode='explicit', force_listen_count=3), device='cuda')
    model.set_mode(processor_mode.DUPLEX)
    if processor_class is not None and model.processor is not processor:
        raise ValueError('selected local processor was replaced during initialization')

    llm = model.llm
    active = list(llm.active_adapters()) if getattr(llm, 'peft_config', None) else []
    layers = [m for _, m in llm.named_modules() if hasattr(m, 'lora_A')]
    identity = dict(decoder_is_loaded_llm=model.duplex.decoder.m is llm,
        active_adapters=active, lora_layers=len(layers),
        enabled_lora_layers=sum(not getattr(m, 'disable_adapters', True)
                               for m in layers))
    if (not identity['decoder_is_loaded_llm'] or 'tfd_grpo' not in active
            or not identity['enabled_lora_layers']):
        raise ValueError('adapter runtime identity failed')
    view = duplex_view(model, ref_audio_path=None,
        config=duplex_config(decode_mode='greedy', force_listen_count=3))
    result = dict(model=model, view=view, identity=identity)
    if preflight is not None:
        result['preflight'] = preflight
    return result
