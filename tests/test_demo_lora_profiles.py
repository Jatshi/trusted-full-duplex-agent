import hashlib
import json
import pytest
from tfd.duplex_policy.demo_lora import AdapterProfiles, LORA_PROFILES, verify_asset


class LLM:
    def __init__(self):
        self.active = ['tfd_grpo']
        self.history = []
        self.fail = False

    def set_adapter(self, name):
        self.history.append(name)
        self.active = [name]
        if self.fail and name != 'tfd_grpo':
            raise RuntimeError('injected selection failure')

    def active_adapters(self):
        return self.active


def make():
    llm = LLM()
    events = []
    entries = {p: {'adapter_name': p, 'weight_sha256': 'a'*64} for p in LORA_PROFILES}
    return AdapterProfiles(llm, entries, default='tfd_grpo', reset=lambda: events.append('reset')), llm, events


def test_all_six_candidates_not_best_seed_selection():
    c, llm, events = make()
    assert len(LORA_PROFILES) == 6
    for p in LORA_PROFILES:
        c.acquire('s', p)
        assert llm.active == [p]
        assert c.metrics()['tfd_lora_controls_answer'] is True
        c.release('s')
        assert llm.active == ['tfd_grpo']
    assert len(events) == 12


def test_no_cross_session_or_mid_session_switch():
    c, llm, events = make()
    c.acquire('a', LORA_PROFILES[0])
    for owner in ['a', 'b']:
        with pytest.raises(RuntimeError, match='leased'):
            c.acquire(owner, LORA_PROFILES[1])
    with pytest.raises(RuntimeError, match='owner'):
        c.release('b')
    assert llm.active == [LORA_PROFILES[0]] and events == ['reset']


def test_error_restores_default_and_clears_lease():
    c, llm, events = make()
    llm.fail = True
    with pytest.raises(RuntimeError, match='selection'):
        c.acquire('s', LORA_PROFILES[0])
    assert llm.active == ['tfd_grpo'] and c.owner is None
    c.acquire('b', 'v2_reference')
    c.release('b')


def test_unknown_profile_rejected_before_reset():
    c, llm, events = make()
    with pytest.raises(ValueError):
        c.acquire('s', '../../evil')
    assert events == [] and llm.history == []


def test_reset_happens_before_every_adapter_selection():
    c, llm, events = make()
    original = llm.set_adapter
    def select(name):
        assert len(events) > len(llm.history)
        original(name)
    llm.set_adapter = select
    c.acquire('s', LORA_PROFILES[0])
    c.release('s')


def test_hash_pin_rejects_changed_or_missing_asset(tmp_path):
    config = tmp_path/'adapter_config.json'
    weight = tmp_path/'adapter_model.safetensors'
    config.write_text(json.dumps({'peft_type': 'LORA', 'task_type': 'CAUSAL_LM'}))
    weight.write_bytes(b'fixture')
    entry = {'path': str(tmp_path), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
             'weight_sha256': hashlib.sha256(weight.read_bytes()).hexdigest()}
    assert verify_asset(entry) == tmp_path
    weight.write_bytes(b'drift')
    with pytest.raises(ValueError, match='SHA'):
        verify_asset(entry)
    weight.unlink()
    with pytest.raises(FileNotFoundError):
        verify_asset(entry)
