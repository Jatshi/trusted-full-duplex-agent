"""Parent-side fixed E72 input-processing evidence check, not model scoring."""
import hashlib
import json
from pathlib import Path


def validate_prefix_result(path):
    """Hash exactly the bytes parsed. No natural/semantic completion inferred.

    Validates structure and fixed packet accounting, not honesty of a backend,
    source identity or durable storage. The caller must close those separately.
    """
    with Path(path).open('rb') as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError('result exceeds single-case limit')
    def reject_constant(value):
        raise ValueError('nonfinite JSON')
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in (
        ('schema', 'tfd.controlled_prefix_session.v1'), ('status', 'input_exhausted'))):
        raise ValueError('invalid result schema/status')
    if value.get('input_complete') is not True or value.get('preparation_complete') is not True:
        raise ValueError('incomplete case')
    if 'final_response_naturally_complete' not in value or value['final_response_naturally_complete'] is not None:
        raise ValueError('unsupported natural completion claim')
    if value.get('error_type') is not None:
        raise ValueError('case error')
    chunks = value.get('chunks')
    if not isinstance(chunks, list) or len(chunks) != 45:
        raise ValueError('incomplete packet plan')
    cursor = 0
    for index, chunk in enumerate(chunks):
        end = min(cursor + (16560 if index == 0 else 16000), 720000)
        real = max(0, min(end, 480000) - min(cursor, 480000))
        if not isinstance(chunk, dict) or chunk.get('input_id') != str(index) or chunk.get('completed') is not True:
            raise ValueError('incomplete/nonconsecutive packet')
        for key, expected in (('start_sample', cursor), ('real_samples', real),
                              ('artificial_samples', end - cursor - real)):
            if type(chunk.get(key)) is not int or chunk[key] != expected:
                raise ValueError('packet accounting mismatch')
        if chunk.get('input_eof') is not (end >= 480000) or not isinstance(chunk.get('events'), list):
            raise ValueError('invalid packet evidence')
        cursor = end
    for key, expected in (('completed_real_samples', 480000), ('completed_artificial_samples', 240000)):
        if type(value.get(key)) is not int or value[key] != expected:
            raise ValueError('total mismatch')
    return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
                completed_packets=45, final_response_naturally_complete=None)
