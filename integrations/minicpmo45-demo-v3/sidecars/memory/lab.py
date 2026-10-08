"""Actual trained memory on its exact cached encoder contract, not speech control."""
import hashlib
import json
from pathlib import Path
import re
import torch
from controlled import StudyModel, VALUES, SET, QUERY, latest_rule


def events_from_controls(controls):
    if not isinstance(controls, list) or not 1 <= len(controls) <= 64:
        raise ValueError('Expected 1..64 observed controls')
    events = []
    for c in controls:
        if not isinstance(c,dict) or set(c)-{'op','source','value','template'}:
            raise ValueError('Unknown control fields/labels forbidden')
        source, template, op = c.get('source'), c.get('template',0), c.get('op')
        if type(source) is not int or source not in (0,1) or type(template) is not int or template not in (0,1):
            raise ValueError('Only explicit symbolic source0/1 and training expressions0/1')
        event = {'source':source, 'revocations':[], 'kind':op}
        if op == 'set':
            value = c.get('value')
            if type(value) is not int or not 0 <= value < len(VALUES):
                raise ValueError('Invalid color input')
            event['text'] = SET[template].format(VALUES[value])
        elif op == 'revoke':
            event.update(source=2, revocations=[source], text='协议事件已到达。', kind='protocol')
        elif op == 'query':
            event.update(source=2, text=QUERY[template].format(source))
        else:
            raise ValueError('Unknown operation')
        events.append(event)
    return events


def tensor_observation(events, cache):
    x = torch.stack([cache[e['text']] for e in events]).unsqueeze(0)
    sources = torch.tensor([[e['source'] for e in events]],dtype=torch.long)
    revoke = torch.zeros(1,len(events),3,dtype=torch.bool)
    active = set()
    for j,e in enumerate(events):
        active.update(e['revocations'])
        for s in active:
            revoke[0,j,s] = True
    return x,sources,revoke


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MemoryLab:
    def __init__(self, manifest):
        pins = json.loads(Path(manifest).read_text())
        for path, digest in pins['files'].items():
            if sha(path) != digest:
                raise ValueError('Memory provenance SHA mismatch')
        parent = Path(pins['parent'])
        raw = torch.load(parent/'encode/features.pt',map_location='cpu',weights_only=True)
        rows = json.loads((parent/'data.json').read_text())
        texts = {e['text'] for r in rows if r['split']=='train' for e in r['events']}
        scale = torch.stack([raw[t] for t in sorted(texts)]).square().mean().sqrt().clamp_min(1e-6)
        self.cache = {t:raw[t]/scale for t in texts}  # only training expressions, no heldout rescoring
        self.models = {}
        for name, info in pins['checkpoints'].items():
            model = StudyModel(len(next(iter(self.cache.values()))),info['arm']).cpu().eval()
            model.load_state_dict(torch.load(info['path'],map_location='cpu',weights_only=True),strict=True)
            self.models[name] = (model, pins['files'][info['path']])
        if len(self.models) != 18:
            raise ValueError('All eighteen checkpoints required, no best-seed selection')
        self.health = {'ready':True,'device':'cpu','checkpoints':len(self.models),'feature_dim':len(next(iter(self.cache.values()))),
                       'scale':float(scale),'feature_sha256':pins['files'][str(parent/'encode/features.pt')],
                       'scope':'cached_independent_MiniCPM_event_texts_explicit_protocol_not_audio_KV_forgetting',
                       'controls_main_answer':False,'heldout_scoring':False,'new_training':False}

    def predict(self, controls):
        events = events_from_controls(controls)
        if controls[-1]['op'] != 'query' or any(c['op']=='query' for c in controls[:-1]):
            raise ValueError('Exactly one final query per fresh observation scope')
        x,s,r = tensor_observation(events,self.cache)
        results = {}
        with torch.inference_mode():
            for name,(model,digest) in self.models.items():
                v,a = model(x,s,r)
                if not torch.isfinite(v).all() or not torch.isfinite(a).all():
                    raise ValueError('Nonfinite memory output')
                results[name] = {'value':(VALUES+['无有效记录'])[int(v.argmax())],
                                 'action':('WAIT','RESPOND','CANCEL')[int(a.argmax())],
                                 'value_probabilities':v.softmax(-1)[0].tolist(),
                                 'action_probabilities':a.softmax(-1)[0].tolist(),'checkpoint_sha256':digest}
        # Exact parser is a strong control; input protocol is shared by all arms.
        rule_value,rule_action = latest_rule(events)
        return {'health':self.health,'observed_events':events,'predictions':results,
                'rule':{'value':(VALUES+['无有效记录'])[rule_value],'action':('WAIT','RESPOND','CANCEL')[rule_action]}}
