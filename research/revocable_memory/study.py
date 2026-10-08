"""Finite frozen-encoder mechanism falsification; no Demo or audio modifications."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from controlled import build, observation, latest_rule, StudyModel


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def last_token(hidden, mask):
    if not (mask.sum(1) > 0).all():
        raise ValueError('empty encoding')
    # Tokenizer right padding is required, verified at extraction.
    return hidden[torch.arange(len(hidden), device=hidden.device), mask.sum(1) - 1]


def tensor_batch(rows, cache, device):
    obs = [observation(r)['events'] for r in rows]
    length = max(map(len, obs)); dim = len(next(iter(cache.values())))
    x = torch.zeros(len(rows), length, dim, device=device)
    sources = torch.full((len(rows), length), 2, dtype=torch.long, device=device)
    revoke = torch.zeros(len(rows), length, 3, dtype=torch.bool, device=device)
    for i, events in enumerate(obs):
        active = set()
        for j, e in enumerate(events):
            x[i, j] = cache[e['text']].to(device)
            sources[i, j] = e['source']
            active.update(e['revocations'])
            for s in active:
                revoke[i, j, s] = True
    lengths = torch.tensor([len(e) for e in obs], device=device)
    value = torch.tensor([r['value'] for r in rows], device=device)
    action = torch.tensor([r['action'] for r in rows], device=device)
    return x, sources, revoke, lengths, value, action


def summarize(results):
    by_arm = {a: [r['test_joint'] for r in results if r['arm'] == a]
              for a in ('memory', 'gru', 'no_clear')}
    return {'all_runs': results, 'joint_by_arm': by_arm, 'rule_joint': 1.0,
            'product_promotion': False, 'method_superiority_over_rule': False,
            'scope': 'authored symbolic protocol, frozen text features; NOT audio/full2.0/independent',
            'next_action': 'Inspect all-seed memory/GRU and no-clear outcomes. Rule100% prevents unique-method claim. Do not extend this task to fill GPU; move to a genuinely partial/noisy-observation problem only after review.'}


def encode(a, rows):
    out = a.output / 'encode'; out.mkdir()
    started = time.monotonic()
    dump(out / 'status.json', {'state': 'asset_hash', 'pid': os.getpid()})
    # No base weights are changed; full source/checkpoint provenance before use.
    assets = sorted(a.model.glob('*.safetensors'))
    assets += sorted(a.model.glob('*.json'))
    dump(out / 'assets.json', {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in assets})
    sys.path.insert(0, str(a.source / 'src'))
    from tfd.rl.grpo_trainer import load_causallm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=True)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causallm(str(a.model), 'cuda', False).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    texts = sorted({e['text'] for r in rows for e in observation(r)['events']})
    cache = {}
    for i in range(0, len(texts), 8):
        encoded = tok(texts[i:i + 8], padding=True, return_tensors='pt', add_special_tokens=True)
        if encoded['input_ids'].shape[1] > 128:
            raise ValueError('no truncation permitted')
        encoded = {k: v.to('cuda') for k, v in encoded.items()}
        with torch.inference_mode():
            # Each text independently encoded with zero previous conversation KV.
            hidden = model.model(**encoded, use_cache=False, return_dict=True).last_hidden_state
            features = last_token(hidden, encoded['attention_mask']).float().cpu()
        if not torch.isfinite(features).all() or not (features.norm(dim=1) > 0).all():
            raise RuntimeError('invalid canary features')
        cache.update(zip(texts[i:i + 8], features))
        dump(out / 'status.json', {'state': 'encoding', 'texts': len(cache), 'total': len(texts),
                                   'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started})
        if i == 0:
            dump(out / 'canary.json', {'finite_features': True, 'rows': len(features),
                 'separate_event_contexts': True, 'semantic_validation': False})
    torch.save(cache, out / 'features.pt')
    dump(out / 'status.json', {'state': 'complete', 'texts': len(cache),
                               'elapsed_seconds': time.monotonic() - started})


def evaluate(model, rows, cache, path=None):
    model.eval(); records = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            batch = rows[i:i + 64]
            x, s, r, l, y, a = tensor_batch(batch, cache, 'cuda')
            v, z = model(x, s, r, l)
            vs = v.argmax(-1).cpu().tolist(); zs = z.argmax(-1).cpu().tolist()
            records += [{'id': row['id'], 'family': row['family'], 'condition': row['condition'],
                         'variant': row['variant'], 'value': row['value'], 'action': row['action'],
                         'pred_value': pv, 'pred_action': pa,
                         'joint': pv == row['value'] and pa == row['action']}
                        for row, pv, pa in zip(batch, vs, zs)]
    if path:
        with path.open('x', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
    return sum(r['joint'] for r in records) / len(records)


def train(a, rows):
    out = a.output / f'{a.arm}_{a.seed}'; out.mkdir()
    started = time.monotonic(); torch.manual_seed(a.seed); torch.set_num_threads(4)
    cache = torch.load(a.output / 'encode/features.pt', map_location='cuda', weights_only=True)
    # Train-vocabulary-only scalar normalization; do not fit held-out features.
    texts = {e['text'] for r in rows if r['split'] == 'train' for e in observation(r)['events']}
    scale = torch.stack([cache[t] for t in sorted(texts)]).square().mean().sqrt().clamp_min(1e-6)
    cache = {t: v / scale for t, v in cache.items()}
    model = StudyModel(len(next(iter(cache.values()))), a.arm).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    parts = {s: [r for r in rows if r['split'] == s] for s in ('train', 'dev', 'test')}
    generator = torch.Generator().manual_seed(a.seed)
    best = -1; stale = 0; checkpoint = None; final_step = 0
    dump(out / 'launch.json', {'arm': a.arm, 'seed': a.seed,
         'parameters': sum(p.numel() for p in model.parameters()), 'max_steps': 800,
         'batch': 64, 'lr': .001, 'dev_interval': 100, 'plateau_evaluations': 3,
         'selection': 'best dev joint; same rule all arms; test once', 'independent': False})
    with (out / 'train.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        for step in range(1, 801):
            model.train()
            idx = torch.randint(len(parts['train']), (64,), generator=generator).tolist()
            x, s, r, l, y, action = tensor_batch([parts['train'][i] for i in idx], cache, 'cuda')
            opt.zero_grad(set_to_none=True); v, z = model(x, s, r, l)
            loss = torch.nn.functional.cross_entropy(v, y) + torch.nn.functional.cross_entropy(z, action)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('missing/nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); final_step = step
            if step == 3:
                dump(out / 'canary.json', {'finite_optimization_steps': 3, 'full_continuation': True})
            if step % 10 == 0 or step <= 3:
                log.write(json.dumps({'step': step, 'loss': loss.item()}) + '\n')
                dump(out / 'status.json', {'state': 'training', 'step': step,
                     'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started})
            if step % 100 == 0:
                score = evaluate(model, parts['dev'], cache)
                log.write(json.dumps({'step': step, 'dev_joint': score}) + '\n')
                if score > best + .001:
                    best = score; stale = 0
                    checkpoint = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(checkpoint, out / 'best.pt')
                else:
                    stale += 1
                if stale >= 3 or best == 1.0:
                    break
            if time.monotonic() - started > 2400:
                raise TimeoutError('training deadline')
    model.load_state_dict(checkpoint)
    test = evaluate(model, parts['test'], cache, out / 'test_outputs.jsonl')
    dev = evaluate(model, parts['dev'], cache, out / 'dev_outputs.jsonl')
    result = {'state': 'complete', 'arm': a.arm, 'seed': a.seed, 'steps': final_step,
              'dev_joint': dev, 'test_joint': test, 'test_rows': len(parts['test']),
              'elapsed_seconds': time.monotonic() - started,
              'parameters': sum(p.numel() for p in model.parameters()), 'promotion': False}
    dump(out / 'result.json', result); dump(out / 'status.json', result)


def supervise(a, rows):
    import fcntl
    lock = Path('/root/tfd-long-runs/tfd_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(x.strip()) >= 500 for x in used.splitlines()):
        raise RuntimeError('GPU occupied; no overlap')
    a.output.mkdir(exist_ok=False)
    started = time.monotonic()
    pins = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    dump(a.output / 'source_hashes.json', pins)
    dump(a.output / 'data.json', rows)
    if any(latest_rule(r['events']) != (r['value'], r['action']) for r in rows):
        raise RuntimeError('data/reference gate failed')
    dump(a.output / 'protocol.json', {'rule_joint': 1.0, 'families': 2048, 'rows': len(rows),
         'split_rows': {s: sum(r['split'] == s for r in rows) for s in ('train', 'dev', 'test')},
         'raw_text_parser_upper_bound': True, 'revocation': 'observed protocol shared all arms',
         'synthetic_test_extra_distractors': 24, 'max_queue_seconds': 14400,
         'per_worker_seconds': 2400, 'stall_seconds': 300, 'canary_steps': 3,
         'originality_claim': False, 'full2_comparison': False})
    jobs = [('encode', None, None)] + [('train', arm, seed) for seed in (42, 43, 44)
                                      for arm in ('memory', 'gru', 'no_clear')]
    results = []
    for stage, arm, seed in jobs:
        if any(sha(Path(__file__).parent / n) != h for n, h in pins.items()):
            raise RuntimeError('source drift')
        job = 'encode' if stage == 'encode' else f'{arm}_{seed}'
        command = [sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                   '--source', str(a.source), '--model', str(a.model), '--output', str(a.output)]
        if arm:
            command += ['--arm', arm, '--seed', str(seed)]
        dump(a.output / 'queue_status.json', {'state': 'running', 'job': job,
             'completed_jobs': len(results), 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started})
        with (a.output / f'{job}.log').open('x', encoding='utf-8') as log:
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            began = time.monotonic(); worker_wall = time.time(); status = a.output / job / 'status.json'
            while p.poll() is None:
                last_progress = status.stat().st_mtime if status.exists() else worker_wall
                if time.monotonic() - began > 2400 or time.monotonic() - started > 14400 or time.time() - last_progress > 300:
                    import signal
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL); p.wait()
                    dump(a.output / f'{job}_exit.json', {'exit': p.returncode, 'failure': 'deadline_or_stall'})
                    raise TimeoutError(job)
                time.sleep(2)
            dump(a.output / f'{job}_exit.json', {'exit': p.returncode})
            if p.returncode:
                raise RuntimeError(f'{job} exit{p.returncode}; stop remaining jobs')
        if stage == 'train':
            results.append(json.loads((a.output / job / 'result.json').read_text()))
    dump(a.output / 'analysis.json', summarize(results))
    dump(a.output / 'queue_status.json', {'state': 'complete', 'runs': len(results),
         'elapsed_seconds': time.monotonic() - started, 'automatic_analysis': True, 'promotion': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['supervise', 'encode', 'train'], default='supervise')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['memory', 'gru', 'no_clear'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(); began_wall = time.time()
    try:
        rows = build()
        {'supervise': supervise, 'encode': encode, 'train': train}[args.stage](args, rows)
    except Exception:
        if args.output.exists():
            dump(args.output / (f'failure_{args.stage}_{args.arm}_{args.seed}.json'),
                 {'traceback': traceback.format_exc(), 'stage': args.stage, 'arm': args.arm, 'seed': args.seed})
        raise
