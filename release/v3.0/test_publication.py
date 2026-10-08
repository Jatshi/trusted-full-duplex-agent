import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('publication', ROOT/'publication.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_private_runtime_material_never_enters_public_tree():
    for name, text in [('recordings/session.json', '{}'), ('weights/head.pt','weights'),
                       ('src/x.py','connect.westb.'+'seetacloud.com'),
                       ('docs/x.md','C:/Users/'+'jat_s/WorkBuddy'),
                       ('src/x.py','ghp_'+'a'*36)]:
        with pytest.raises(ValueError):
            publication.check_public_text(name, text)
    publication.check_public_text('src/tfd/x.py', 'import os\n')


def test_readme_describes_whole_stack_and_actual_mode_boundaries():
    readme = (ROOT.parents[1]/'README.md').read_text(encoding='utf-8')
    for name in ['3.0','MLP','TrustGate','GRPO','Faster-Whisper','EasyTurn','Smart Turn','LoRA','生命周期','旁路','记忆']:
        assert name in readme
    assert 'E124最新增量' not in readme


def test_build_rejects_wrong_upstream_revision():
    with pytest.raises(ValueError, match='revision'):
        publication.require_revision('0000000')
    publication.require_revision('47709a9210dfd71afa76c058e017fc8c4db5c8d2')


def test_integration_manifest_prevents_overlay_traversal(tmp_path):
    with pytest.raises(ValueError):
        publication.validate_relative('../outside.py')
    assert publication.validate_relative('py_backend/tfd_runtime.py') == 'py_backend/tfd_runtime.py'


def test_unprovisioned_lora_mode_does_not_silently_use_default():
    from tfd.duplex_policy.demo_lora import require_profile_assets
    with pytest.raises(RuntimeError, match='checkpoint'):
        require_profile_assets('v3_lora_e127_s42', None)
    require_profile_assets('v3_engineering', None)
    require_profile_assets('v3_lora_e127_s42', object())
