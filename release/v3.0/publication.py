"""Build the source-only 3.0 integration from an explicit publication allowlist."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

UPSTREAM_REVISION = '47709a9210dfd71afa76c058e017fc8c4db5c8d2'
ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT/'integrations/minicpmo45-demo-v3'
TRACKED_DEMO = [
    'core/processors/pytorch_backend.py', 'py_backend/server.py',
    'runtime/backend_client.py', 'static/audio-duplex/audio-duplex-app.js',
    'static/audio-duplex/audio-duplex.css', 'static/audio-duplex/audio_duplex.html',
    'static/duplex/lib/audio-player.js', 'static/duplex/lib/realtime-session.js',
    'static/duplex/ui/duplex-ui.js', 'static/lib/audio-device-selector.js',
]
OWNED_DEMO = [
    'py_backend/tfd_native_action.py','py_backend/tfd_response_context.py',
    'py_backend/tfd_response_lifecycle.py','py_backend/tfd_runtime.py',
    'py_backend/tfd_wait_gate.py','static/duplex/lib/audible-stop-measurement.js',
    'tests/test_tfd_runtime.py','tests/test_tfd_native_action.py',
    'tests/test_tfd_response_context.py','tests/test_e61_lifecycle_transport.py',
    'tests/browser/audio-clock-ready.js',
]
NEW_TESTS = [
    'test_demo_lora_profiles.py','test_demo_component_observers.py',
    'test_response_lifecycle.py','test_e48_action_residual.py',
    'test_duplex_policy_v3.py','test_smart_turn_pretrained.py',
]


def validate_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe relative publication path')
    return name


def check_public_text(name, text):
    validate_relative(name)
    if any(part in {'recordings','weights','profile_output','autonomy','.ssh'} for part in PurePosixPath(name).parts):
        raise ValueError('Private or binary runtime material')
    if PurePosixPath(name).suffix in {'.pt','.npz','.wav','.safetensors','.bin','.env'}:
        raise ValueError('Weights, recordings and credentials stay outside source release')
    markers = ('connect.westb.'+'seetacloud.com', 'C:/Users/'+'jat_s/WorkBuddy', 'C:\\Users\\'+'jat_s',
               'BEGIN OPENSSH '+'PRIVATE KEY', 'BEGIN RSA '+'PRIVATE KEY')
    if any(token in text for token in markers) or re.search(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',text):
        raise ValueError('Local operations or credential content')


def require_revision(value):
    if value != UPSTREAM_REVISION:
        raise ValueError('Expected pinned upstream revision')


def put(relative, text):
    check_public_text(relative, text)
    if not relative.endswith('.patch'):
        text = text.rstrip()+'\n'
    target = ROOT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(text.replace('\r\n','\n'),encoding='utf-8',newline='\n')


def copy(source, relative):
    put(relative,source.read_text(encoding='utf-8'))


def build(source, demo):
    require_revision(subprocess.check_output(['git','rev-parse','HEAD'],cwd=demo,text=True).strip())
    copy(source/'configs/duplex_policy.yaml','configs/duplex_policy.yaml')
    for path in sorted((source/'src/tfd').rglob('*.py')):
        value = path.read_text(encoding='utf-8')
        if path == source/'src/tfd/__init__.py':
            value = re.sub(r'__version__ = "[^"]+"', '__version__ = "3.0.0"', value)
        put(path.relative_to(source).as_posix(),value)
    for name in NEW_TESTS:
        value = (source/'tests'/name).read_text(encoding='utf-8')
        if name == 'test_response_lifecycle.py':
            value = value.replace('integrations/minicpmo45-demo/experimental_e58/server.py',
                                  'tests/fixtures/e58_push_full_duplex.py')
        put('tests/'+name,value)
    # Keep the historical method-level test reproducible without vendoring the
    # entire upstream experimental server. This fixture preserves its AST.
    legacy = (source/'integrations/minicpmo45-demo/experimental_e58/server.py').read_text(encoding='utf-8')
    method = next(node for node in ast.walk(ast.parse(legacy))
                  if isinstance(node,ast.AsyncFunctionDef) and node.name=='_push_full_duplex')
    put('tests/fixtures/e58_push_full_duplex.py',
        '"""Historical E58 method AST fixture, not a runnable server."""\n'+ast.unparse(method)+'\n')
    for script in sorted((source/'scripts').glob('*.py')):
        if re.match(r'(7[0-9]|8[0-3])_',script.name):
            copy(script,'scripts/'+script.name)
    for name in OWNED_DEMO:
        copy(demo/name,'integrations/minicpmo45-demo-v3/overlay/'+name)
    for path in sorted((demo/'tests/js').glob('*.test.js')):
        # Only project-added tests; existing upstream tests remain upstream.
        if subprocess.run(['git','ls-files','--error-unmatch',str(path.relative_to(demo))],
                          cwd=demo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            copy(path,'integrations/minicpmo45-demo-v3/overlay/'+path.relative_to(demo).as_posix())
    changes = []
    for name in TRACKED_DEMO:
        old = subprocess.check_output(['git','show',UPSTREAM_REVISION+':'+name],cwd=demo)
        new = (demo/name).read_bytes().replace(b'\r\n',b'\n')
        # Public release identifies the same integrated feature set as version3.0.
        text = new.decode('utf-8').replace('3.0.0-rc.5','3.0.0').replace('TFD-STAR 3.0 rc.5','TFD-STAR 3.0')
        if name.endswith('audio-duplex.css'):
            text = re.sub(r'@font-face\s*\{[^}]*\}\s*','',text)
        check_public_text(name,text)
        # Generate a Git-compatible full-file replacement. No upstream tree is vendored.
        import difflib
        diff = ''.join(difflib.unified_diff(old.decode('utf-8').splitlines(True),
                        text.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        if diff:
            changes.append('diff --git a/'+name+' b/'+name+'\n'+diff)
    put('integrations/minicpmo45-demo-v3/tfd-star-v3.patch',''.join(changes))
    copy(source/'experiments/demo_v3_components_20261008/observer_service.py',
         'integrations/minicpmo45-demo-v3/sidecars/observer_service.py')
    for name in ['lab.py','service.py','panel.html','controlled.py','memory.py']:
        copy(source/'experiments/demo_v3_memory_lab_20261008'/name,
             'integrations/minicpmo45-demo-v3/sidecars/memory/'+name)
    copy(source/'experiments/demo_v3_memory_lab_20261008/tests/test_lab.py',
         'integrations/minicpmo45-demo-v3/sidecars/memory/tests/test_lab.py')
    for name in ['memory','controlled','study','prefix_grounding']:
        copy(source/'experiments/revocable_evidence'/(name+'.py'),
             'research/revocable_memory/'+name+'.py')
    files = []
    for path in sorted((INTEGRATION/'overlay').rglob('*')):
        if path.is_file():
            data=path.read_bytes()
            files.append({'path':path.relative_to(INTEGRATION/'overlay').as_posix(),
                          'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    patch=(INTEGRATION/'tfd-star-v3.patch').read_bytes()
    manifest={'version':'3.0.0','upstream_revision':UPSTREAM_REVISION,
              'patch_sha256':hashlib.sha256(patch).hexdigest(),'overlay':files,
              'weights_included':False,'private_recordings_included':False}
    put('integrations/minicpmo45-demo-v3/MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    return {'version':'3.0.0','overlay_files':len(files),'patch_bytes':len(patch)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--demo-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source_root.resolve(),args.demo_root.resolve())))
