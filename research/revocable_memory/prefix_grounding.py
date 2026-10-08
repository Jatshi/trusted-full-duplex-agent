"""Train/dev-only causal prefix-grounding diagnostic after E136 underfitting.

No old test scoring, no full-system or novelty claim. New supervision is an
earlier prefix-state read, not duplication of the final reference.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import time
import torch
from controlled import QUERY, reference, observation, StudyModel
from study import dump, sha, tensor_batch, evaluate


def prefix_example(row, cut, query_source):
    if row['split'] != 'train' or query_source not in (0, 1):
        raise ValueError('prefix supervision only for train with observed source0/1')
    if not 1 <= cut <= len(row['events']):
        raise ValueError('invalid prefix boundary')
    events = copy.deepcopy(row['events'][:cut])
    if any(e['kind'] == 'query' for e in events):
        raise ValueError('prefix cannot include original final query')
    template = int(row['id'].split('_')[0]) % 2
    events.append({'kind': 'query', 'source': 2, 'revocations': [],
                   'text': QUERY[template].format(query_source), 'query_source': query_source})
    result = {k: v for k, v in row.items() if k not in ('events', 'value', 'action')}
    result['id'] = f"{row['id']}_prefix{cut}_query{query_source}"
    result['events'] = events
    result['value'], result['action'] = reference(events)
    return result


def main(run, output):
    import fcntl
    lock = Path('/root/tfd-long-runs/tfd_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(v) >= 500 for v in used.splitlines()):
        raise RuntimeError('GPU occupied')
    output.mkdir(exist_ok=False); started = time.monotonic(); torch.set_num_threads(4)
    rows = [r for r in json.loads((run / 'data.json').read_text(encoding='utf-8')) if r['split'] != 'test']
    train = [r for r in rows if r['split'] == 'train']; dev = [r for r in rows if r['split'] == 'dev']
    cache = torch.load(run / 'encode/features.pt', map_location='cuda', weights_only=True)
    texts = {e['text'] for r in train for e in observation(r)['events']}
    scale = torch.stack([cache[t] for t in sorted(texts)]).square().mean().sqrt().clamp_min(1e-6)
    cache = {t: v / scale for t, v in cache.items()}
    pins = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    dump(output / 'protocol.json', {'source_sha': pins, 'data_sha': sha(run / 'data.json'),
         'feature_sha': sha(run / 'encode/features.pt'), 'train_rows': len(train), 'dev_rows': len(dev),
         'test_scoring': False, 'same_observed_protocol': True,
         'routes': ['memory_endpoint', 'memory_prefix', 'gru_prefix'], 'seeds': [42, 43, 44],
         'batch': 64, 'max_updates': 800, 'dev_interval': 100, 'plateau_evaluations': 3,
         'prefix_fraction': .5, 'lr': .001, 'canary_updates': 3,
         'budget_note': 'Same updates and trajectories/update; prefix event lengths differ, not exact frame budget.',
         'scope': 'seen development corpus optimization diagnosis, NOT new validation/independent/full2.0',
         'deadline_seconds': 3600, 'promotion': False})
    results = []
    for seed in (42, 43, 44):
        for route in ('memory_endpoint', 'memory_prefix', 'gru_prefix'):
            if any(sha(Path(__file__).parent / n) != h for n, h in pins.items()):
                raise RuntimeError('source drift')
            torch.manual_seed(seed); rng = random.Random(seed)
            model = StudyModel(len(next(iter(cache.values()))), 'gru' if route == 'gru_prefix' else 'memory').cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            leg = output / f'{route}_{seed}'; leg.mkdir()
            best = -1; stale = 0; checkpoint = None; leg_start = time.monotonic()
            sampled_examples = []; event_count = 0
            with (leg / 'train.jsonl').open('x', encoding='utf-8', buffering=1) as log:
                for step in range(1, 801):
                    batch = [train[rng.randrange(len(train))] for _ in range(64)]
                    if route != 'memory_endpoint':
                        for i in range(32):
                            row = batch[i]
                            batch[i] = prefix_example(row, rng.randrange(1, len(row['events'])), rng.randrange(2))
                    if step == 1:
                        sampled_examples = batch[:4]
                    event_count += sum(len(r['events']) for r in batch)
                    x, s, r, l, y, a = tensor_batch(batch, cache, 'cuda')
                    model.train(); opt.zero_grad(set_to_none=True)
                    v, z = model(x, s, r, l)
                    loss = torch.nn.functional.cross_entropy(v, y) + torch.nn.functional.cross_entropy(z, a)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('invalid gradients')
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    if step == 3:
                        dump(leg / 'canary.json', {'finite_updates': 3, 'continuation': True,
                             'sampled_prefixes': sampled_examples, 'future_labels_in_inputs': False})
                    if step % 10 == 0 or step <= 3:
                        log.write(json.dumps({'step': step, 'loss': loss.item()}) + '\n')
                        dump(output / 'status.json', {'state': 'training', 'route': route, 'seed': seed,
                             'step': step, 'completed_runs': len(results), 'pid': os.getpid(),
                             'elapsed_seconds': time.monotonic() - started})
                    if step % 100 == 0:
                        score = evaluate(model, dev, cache)
                        log.write(json.dumps({'step': step, 'dev_joint': score}) + '\n')
                        if score > best + .001:
                            best = score; stale = 0
                            checkpoint = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            torch.save(checkpoint, leg / 'best.pt')
                        else:
                            stale += 1
                        if stale >= 3 or best == 1:
                            break
                    if time.monotonic() - leg_start > 900 or time.monotonic() - started > 3600:
                        raise TimeoutError('prefix diagnostic deadline')
            model.load_state_dict(checkpoint)
            measured = evaluate(model, dev, cache, leg / 'dev_outputs.jsonl')
            result = {'route': route, 'seed': seed, 'updates': step, 'dev_joint': measured,
                      'event_count': event_count, 'elapsed_seconds': time.monotonic() - leg_start,
                      'parameters': sum(p.numel() for p in model.parameters()), 'promotion': False}
            dump(leg / 'result.json', result); results.append(result)
    dump(output / 'result.json', {'results': results, 'test_scoring': False,
         'elapsed_seconds': time.monotonic() - started, 'promotion': False,
         'next_action': 'Analyze grounding objective and representation failures across all seeds; no old test rerun, no promotion. Transition to real/partial observations only after basic learning and licensing gates.'})
    dump(output / 'status.json', {'state': 'complete', 'runs': 9})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.run, a.output)
