import importlib.util
from pathlib import Path
import sys
import pytest
import torch
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from lab import events_from_controls, tensor_observation


def test_explicit_protocol_inputs_no_labels_or_asr_guess():
    ev = events_from_controls([{'op':'set','source':0,'value':2}, {'op':'revoke','source':0}, {'op':'query','source':0}])
    assert ev[0]['text'] == '请把当前代号设为绿色。'
    assert ev[1]['revocations'] == [0]
    assert ev[2]['source'] == 2
    assert not any(k in e for e in ev for k in ('target','value','action'))


def test_sticky_revocation_and_exact_cached_features():
    ev = events_from_controls([{'op':'set','source':0,'value':2}, {'op':'revoke','source':0}, {'op':'query','source':0}])
    cache = {e['text']: torch.arange(4).float()+i for i,e in enumerate(ev)}
    x,s,r = tensor_observation(ev,cache)
    assert x.shape == (1,3,4) and s.tolist() == [[0,2,2]]
    assert r[0,:,0].tolist() == [False,True,True]
    assert torch.equal(x[0,0],cache[ev[0]['text']])
    with pytest.raises(KeyError):
        tensor_observation(ev,{})


@pytest.mark.parametrize('row', [{'op':'set','source':3,'value':1}, {'op':'query','source':True}, {'op':'set','source':0,'value':99}, {'op':'query','source':0,'target':1}, {'op':'query','source':0,'template':3}])
def test_reject_invalid_and_heldout_template(row):
    with pytest.raises(ValueError):
        events_from_controls([row])
