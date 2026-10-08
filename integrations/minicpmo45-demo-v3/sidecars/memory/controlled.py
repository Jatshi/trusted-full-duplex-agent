"""Controlled symbolic protocol study, NOT real assistant/audio supervision.

Texts are authored; targets are deterministic state retrieval. Revocation is an
observed protocol signal shared by every arm, not a learned semantic detector.
"""
import random
import re
import torch
from torch import nn
from memory import RevocableEvidenceMemory

VALUES = ['红色', '蓝色', '绿色', '黄色', '紫色', '白色', '黑色', '橙色']
SET = ['请把当前代号设为{}。', '更正：现在使用{}这个代号。',
       '此前内容作废，本次代号是{}。', '更新记录，代号换为{}。']
QUERY = ['现在读取来源{}的有效代号。', '请报告来源{}当前可用的代号。',
         '核对来源{}：有效代号是什么？', '最后查询来源{}的代号；若无有效记录则不作答。']


def reference(events):
    values = {}; revoked = set()
    for e in events:
        revoked.update(e['revocations'])
        if e['kind'] == 'set' and e['source'] not in revoked:
            values[e['source']] = e['value']
        if e['kind'] == 'query':
            s = e['query_source']
            return (8, 2) if s in revoked else ((values[s], 1) if s in values else (8, 0))
    raise ValueError('query missing')


def latest_rule(events):
    # Upper bound with exact text parser; no gold value field is consulted.
    values = {}; revoked = set()
    for e in events:
        revoked.update(e['revocations'])
        if e['kind'] == 'set' and e['source'] not in revoked:
            found = [i for i, v in enumerate(VALUES) if v in e['text']]
            if len(found) != 1:
                raise ValueError('ambiguous authored text')
            values[e['source']] = found[0]
        if e['kind'] == 'query':
            s = int(re.search(r'来源([01])', e['text']).group(1))
            if s in revoked:
                return 8, 2
            return (values[s], 1) if s in values else (8, 0)
    raise ValueError('query missing')


def observation(row):
    return {'events': [{k: e[k] for k in ('text', 'source', 'revocations')}
                       for e in row['events']]}


def build(families=2048):
    if families < 4 or families % 4:
        raise ValueError('family count must be divisible by4 and >=4')
    rows = []
    for i in range(families):
        split = 'train' if i < families * 3 // 4 else 'dev' if i < families * 7 // 8 else 'test'
        template = i % 2 if split == 'train' else 2 if split == 'dev' else 3
        for condition in range(4):
            rng = random.Random(800000 + i * 4 + condition)
            q = rng.randrange(2); other = 1 - q
            old, new, distract = rng.sample(range(8), 3)
            def event(kind, source, text, **kw):
                return dict(kind=kind, source=source, text=text, revocations=[], **kw)
            ev = [event('set', other, SET[template].format(VALUES[distract]), value=distract)]
            if condition != 3:
                ev += [event('set', q, SET[template].format(VALUES[old]), value=old)]
            else:
                ev += [event('filler', 2, '记录尚未提供。')]
            for _ in range(3 + (24 if split == 'test' else 0)):
                v = rng.randrange(8)
                ev.append(event('set', other, SET[template].format(VALUES[v]), value=v))
            ev.append(event('set', q, SET[template].format(VALUES[new]), value=new)
                      if condition != 3 else event('filler', 2, '等待有效记录。'))
            # Same prefix/query across pair; only observed revocation differs.
            for variant in range(2):
                rev = [q if condition in (0, 2, 3) else other] if variant else []
                tail = event('protocol', 2, '协议事件已到达。')
                tail['revocations'] = rev
                events = ev + [tail, event('query', 2, QUERY[template].format(q), query_source=q)]
                value, action = reference(events)
                rows.append({'id': f'{i:05d}_{condition}_{variant}',
                             'family': f'f{i:05d}_c{condition}', 'split': split,
                             'condition': condition, 'variant': variant,
                             'events': events, 'value': value, 'action': action})
    return rows


class StudyModel(nn.Module):
    def __init__(self, feature_dim, arm):
        super().__init__()
        if arm not in ('memory', 'gru', 'no_clear'):
            raise ValueError('unknown arm')
        self.arm = arm
        self.project = nn.Linear(feature_dim, 64)
        if arm == 'gru':
            self.cell = nn.GRUCell(70, 32)
            width = 70 + 32
            self.action = nn.Linear(width, 3)
        else:
            self.cell = RevocableEvidenceMemory(70, 32, 3)
            width = 70 + 96
        self.value = nn.Linear(width, 9)

    def forward(self, features, source, revoke, lengths=None):
        b, t, _ = features.shape
        state = (features.new_zeros(b, 32) if self.arm == 'gru'
                 else self.cell.initial_state(b))
        values = []; actions = []
        for j in range(t):
            x = torch.cat((torch.tanh(self.project(features[:, j])),
                           torch.nn.functional.one_hot(source[:, j], 3).float(),
                           revoke[:, j].float()), -1)
            if self.arm == 'gru':
                state = self.cell(x, state)
                context = torch.cat((x, state), -1)
                action = self.action(context)
            else:
                clear = revoke[:, j] if self.arm == 'memory' else torch.zeros_like(revoke[:, j])
                state, action = self.cell.step(x, source[:, j], state, clear)
                context = torch.cat((x, state.flatten(1)), -1)
            values.append(self.value(context)); actions.append(action)
        last = features.new_full((b,), t - 1, dtype=torch.long) if lengths is None else lengths - 1
        idx = torch.arange(b, device=features.device)
        return torch.stack(values, 1)[idx, last], torch.stack(actions, 1)[idx, last]
