"""Exclusive per-session experimental LoRA, never stacking with release adapter."""
import hashlib
import json
from pathlib import Path
import threading

LORA_PROFILES = tuple(f'v3_lora_e{experiment}_s{seed}' for experiment in (127, 134) for seed in (42, 43, 44))
BASE_PROFILES = ('v2_reference', 'v3_engineering', 'v3_shadow', 'v3_components')


def require_profile_assets(profile, controller):
    """A requested learned mode must have its real checkpoint controller."""
    if profile not in BASE_PROFILES + LORA_PROFILES:
        raise ValueError('Unknown adapter profile')
    if profile in LORA_PROFILES and controller is None:
        raise RuntimeError('Requested LoRA mode requires provisioned checkpoints')


def verify_asset(entry):
    path = Path(entry['path']).resolve(strict=True)
    for filename, field in [('adapter_config.json', 'config_sha256'), ('adapter_model.safetensors', 'weight_sha256')]:
        digest = hashlib.sha256((path/filename).read_bytes()).hexdigest()
        if digest != entry[field]:
            raise ValueError(f'Adapter SHA mismatch: {filename}')
    config = json.loads((path/'adapter_config.json').read_text())
    if config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM':
        raise ValueError('Expected causal LoRA asset')
    return path


class AdapterProfiles:
    def __init__(self, llm, entries, *, default, reset):
        if set(entries) != set(LORA_PROFILES):
            raise ValueError('All six experimental adapters required')
        self.llm, self.entries, self.default, self.reset = llm, entries, default, reset
        self.owner = None
        self.profile = None
        self.reset_generation = 0
        self.lock = threading.RLock()

    def _select(self, name):
        self.llm.set_adapter(name)
        if list(self.llm.active_adapters()) != [name]:
            raise RuntimeError('Adapter not exclusively active')

    def acquire(self, owner, profile):
        with self.lock:
            if self.owner is not None:
                raise RuntimeError('Adapter already leased')
            if profile not in BASE_PROFILES + LORA_PROFILES:
                raise ValueError('Unknown adapter profile')
            self.reset()
            self.reset_generation += 1
            try:
                self._select(self.entries[profile]['adapter_name'] if profile in self.entries else self.default)
            except Exception:
                # Initialization failure must not leave the next session on a candidate.
                self._select(self.default)
                raise
            self.owner, self.profile = owner, profile

    def release(self, owner):
        with self.lock:
            if self.owner is None:
                return
            if self.owner != owner:
                raise RuntimeError('Adapter lease owner mismatch')
            # Reset KV/audio/TTS state before changing the active weights.
            # If reset/restore fails, retain lease and fail closed.
            self.reset()
            self.reset_generation += 1
            self._select(self.default)
            self.owner, self.profile = None, None

    def metrics(self):
        with self.lock:
            entry = self.entries.get(self.profile, {})
            return {'tfd_lora_profile': self.profile, 'tfd_lora_active_adapters': list(self.llm.active_adapters()),
                    'tfd_lora_controls_answer': self.profile in self.entries,
                    'tfd_lora_weight_sha256': entry.get('weight_sha256'),
                    'tfd_lora_config_sha256': entry.get('config_sha256'),
                    'tfd_lora_exact_loaded_tensors': entry.get('verified_tensors'),
                    'tfd_lora_cache_reset_generation': self.reset_generation,
                    'tfd_lora_scope': 'exclusive_existing_text_SFT_not_audio_training_or_quality_promotion'}


def load_adapter_profiles(model, manifest_path, *, default, reset):
    """Insert existing weights before TTS-only compile, verify every tensor loaded."""
    import torch
    from safetensors.torch import load_file
    entries = json.loads(Path(manifest_path).read_text())
    if set(entries) != set(LORA_PROFILES):
        raise ValueError('Incomplete adapter manifest')
    llm = model.llm
    for profile, entry in entries.items():
        path = verify_asset(entry)
        name = entry['adapter_name']
        if name != profile:
            raise ValueError('Adapter name must match frozen profile')
        llm.load_adapter(str(path), adapter_name=name, is_trainable=False)
        # Transformers warns (rather than raises) on incompatible keys. Verify
        # the exact tensors against the checkpoint; a quiet partial load is forbidden.
        expected = load_file(str(path/'adapter_model.safetensors'), device='cpu')
        actual = dict(llm.named_parameters())
        for key, value in expected.items():
            canonical = key.removeprefix('base_model.model.')
            if not canonical.endswith(('.lora_A.weight', '.lora_B.weight')):
                raise ValueError('Unexpected non-LoRA checkpoint tensor')
            target = canonical[:-len('.weight')] + f'.{name}.weight'
            if target not in actual or not torch.equal(actual[target].detach().cpu(), value.to(dtype=actual[target].dtype)):
                raise ValueError(f'Adapter tensor failed exact load check: {target}')
        if len(expected) != 504:
            raise ValueError('Expected complete 36-layer seven-projection LoRA')
        entry['verified_tensors'] = len(expected)
        del expected, actual
    llm.set_adapter(default)
    if list(llm.active_adapters()) != [default]:
        raise RuntimeError('Release adapter restoration failed during loading')
    return AdapterProfiles(llm, entries, default=default, reset=reset)
