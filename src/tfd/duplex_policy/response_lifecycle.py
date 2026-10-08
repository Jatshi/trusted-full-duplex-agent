"""Separate decoder LISTEN, natural completion, and verified cancellation.

Does not determine speaker identity: controlled_replay is an operator-level
input contract, never evidence supplied by an untrusted microphone client.
"""


def allow_client_cancel(requested, *, enabled, active, controlled_replay):
    if not enabled or not active:
        return bool(requested)
    return bool(requested and controlled_replay)


def listen_transition(active_response_id, *, enabled=False, natural_end=False,
                      client_cancel=False, force_listen=False):
    reason = 'force_listen' if force_listen or client_cancel else 'model_listen'
    if enabled and natural_end and not client_cancel:
        reason = 'turn_end'
    release = not enabled or natural_end or client_cancel
    return dict(active_response_id=None if release else active_response_id,
                reason=reason, cancel_playback=bool(client_cancel),
                lifecycle_enabled=bool(enabled), natural_end=bool(natural_end))
