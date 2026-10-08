"""Publish a reviewed source commit to a new branch through GitHub Git API.

For environments where the Git smart-HTTP endpoint is unavailable. Does not
update master, create a release, upload model weights or access recordings.
"""
import argparse
import json
from pathlib import Path
import subprocess
from publication import check_public_text

ROOT = Path(__file__).resolve().parents[2]
REPO = 'repos/Jatshi/trusted-full-duplex-agent'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def api(gh, endpoint, payload=None):
    command = [gh, 'api', REPO+'/'+endpoint]
    if payload is not None:
        command += ['--method', 'POST', '--input', '-']
    result = subprocess.run(command, input=json.dumps(payload).encode() if payload is not None else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return json.loads(result.stdout)


def main(gh, base, branch):
    if api(gh, 'git/ref/heads/master')['object']['sha'] != base:
        raise RuntimeError('Remote base changed; review before publication')
    parent = api(gh, 'git/commits/'+base)
    rows = []
    for raw in git('diff', '--name-only', base, 'HEAD').decode().splitlines():
        # Binary historical assets are inherited from the existing public tree.
        value = git('show', 'HEAD:'+raw).decode('utf-8')
        check_public_text(raw, value)
        if raw.startswith(('outputs/','data/')):
            raise ValueError('New runtime outputs are outside this source release')
        rows.append({'path':raw,'mode':'100644','type':'blob','content':value})
    tree = api(gh, 'git/trees', {'base_tree':parent['tree']['sha'], 'tree':rows})
    commit = api(gh, 'git/commits', {'message':'Release TFD-STAR 3.0 source integration',
                                  'tree':tree['sha'],'parents':[base]})
    ref = api(gh, 'git/refs', {'ref':'refs/heads/'+branch,'sha':commit['sha']})
    print(json.dumps({'branch':ref['ref'],'commit':commit['sha'],
                      'tree':tree['sha'],'files_changed':len(rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gh', default='gh')
    parser.add_argument('--base', required=True)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    main(args.gh, args.base, args.branch)
