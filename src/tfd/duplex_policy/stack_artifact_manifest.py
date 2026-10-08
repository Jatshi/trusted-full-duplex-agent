"""Explicit file identities only; never infer a complete execution dependency closure."""
import hashlib
import importlib.metadata
import re
import sys
from pathlib import Path


def collect_artifacts(specification):
    """Hash an explicit nonempty set, reporting missing/mismatched required files.

    This does not execute imports, deserialize weights, capture secrets/environment,
    or authorize model launch. Callers still need the independent runtime gates.
    """
    rows = list(specification)
    roles = [row["role"] for row in rows]
    if not rows or any(not isinstance(role, str) or not role.strip() for role in roles):
        raise ValueError("nonempty explicit roles required")
    if len(set(roles)) != len(roles):
        raise ValueError("duplicate artifact role")
    files = []
    for row in sorted(rows, key=lambda item: item["role"]):
        path = Path(row["path"]).resolve()
        expected = row.get("expected_sha256")
        if expected is not None:
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
                raise ValueError("expected SHA256 must be 64 hex characters")
            expected = expected.lower()
        record = {"role": row["role"], "path": str(path), "expected_sha256": expected}
        if not path.is_file():
            record["status"] = "missing"
        else:
            digest = hashlib.sha256()
            before = path.stat()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
            sha = digest.hexdigest()
            record.update(bytes=after.st_size, sha256=sha)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                record["status"] = "changed_during_read"
            else:
                record["status"] = "ok" if expected is None or expected == sha else "sha256_mismatch"
        files.append(record)
    return {"schema": "tfd.explicit_artifact_manifest.v1", "files": files,
            "artifacts_complete": all(row["status"] == "ok" for row in files),
            "external_launch_ready": False,
            "scope": "explicit current files; no automatic dependency closure or historical identity"}


def collect_runtime_versions(package_names, *, expected_versions=None, version_provider=None):
    """Read distribution metadata, optionally compare exact declared versions.

    Does not import packages or prove that sys.path loads these distributions.
    Absence of an expected map is a snapshot, never a verified version lock.
    """
    names = list(package_names)
    if not names or any(not isinstance(n, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', n) for n in names):
        raise ValueError('explicit distribution names required')
    canonical = [re.sub(r'[-_.]+', '-', n).lower() for n in names]
    if len(set(canonical)) != len(names):
        raise ValueError('duplicate distribution name')
    if expected_versions is not None:
        if not isinstance(expected_versions, dict) or set(expected_versions) != set(names) or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in expected_versions.values()):
            raise ValueError('exact expected versions required for every selected distribution')
    provider = importlib.metadata.version if version_provider is None else version_provider
    records = []
    for name in sorted(names):
        expected = None if expected_versions is None else expected_versions[name]
        try:
            version = provider(name)
        except importlib.metadata.PackageNotFoundError:
            version, status = None, 'missing'
        else:
            if not isinstance(version, str) or not version.strip() or version != version.strip():
                raise ValueError('invalid distribution version metadata')
            status = 'observed' if expected is None else ('match' if version == expected else 'version_mismatch')
        records.append(dict(name=name, version=version, expected_version=expected, status=status))
    return dict(schema='tfd.runtime_distribution_versions.v1', python_version=sys.version.split()[0],
        packages=records, versions_match=expected_versions is not None and all(r['status']=='match' for r in records),
        external_launch_ready=False, scope='distribution metadata only;not imported module origins or remote execution closure')


def collect_loaded_module_origins(specification, *, module_provider=None):
    """Verify selected already-loaded modules; never import absent modules.

    Pins must be declared before loading. Disk SHA after import cannot prove the
    exact in-memory code bytes or a recursive dependency/weight closure.
    """
    rows = list(specification)
    prepared = []
    if not rows:
        raise ValueError('explicit loaded module expectations required')
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('module expectation must be a mapping')
        name, path, sha = row.get('module'), row.get('expected_path'), row.get('expected_sha256')
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*', name)
                or not isinstance(path, (str, Path)) or not Path(path).is_absolute()
                or not isinstance(sha, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', sha)):
            raise ValueError('module name, absolute expected path and SHA256 required')
        prepared.append(dict(module=name, expected_path=str(Path(path).resolve()), expected_sha256=sha.lower()))
    if len({r['module'] for r in prepared}) != len(prepared):
        raise ValueError('duplicate module expectation')
    provider = sys.modules.get if module_provider is None else module_provider
    records = []
    for row in sorted(prepared, key=lambda r:r['module']):
        record = dict(row)
        module = provider(row['module'])
        if module is None:
            record['status'] = 'not_loaded'
        else:
            file = getattr(module, '__file__', None)
            origin = getattr(getattr(module, '__spec__', None), 'origin', None)
            if not isinstance(file, str) or not isinstance(origin, str) or origin in {'built-in','frozen'}:
                record['status'] = 'no_file_origin'
            else:
                actual = Path(file).resolve()
                record['actual_path'] = str(actual)
                if Path(origin).resolve() != actual:
                    record['status'] = 'spec_file_mismatch'
                elif actual != Path(row['expected_path']):
                    record['status'] = 'path_mismatch'
                else:
                    hashed = collect_artifacts([dict(role=row['module'], path=actual,
                                                    expected_sha256=row['expected_sha256'])])['files'][0]
                    record['status'] = hashed['status']
                    for key in ('bytes','sha256'):
                        if key in hashed:
                            record[key] = hashed[key]
        records.append(record)
    return dict(schema='tfd.loaded_module_origins.v1', modules=records,
        origins_match=all(r['status']=='ok' for r in records), external_launch_ready=False,
        scope='selected loaded module file/spec and current disk SHA;not in-memory bytes or recursive execution closure')
