import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const source = fs.readFileSync(path.resolve('runtime/backend_client.py'), 'utf8');
const backendSource = fs.readFileSync(path.resolve('py_backend/server.py'), 'utf8');
const lifecycleSource = fs.readFileSync(path.resolve('py_backend/tfd_response_lifecycle.py'), 'utf8');

describe('backend compile keepalive contract', () => {
    it('does not kill loopback WebSocket sessions during first-use torch.compile', () => {
        expect(source).toContain('ping_interval=None');
        expect(source).toContain('torch.compile can block');
    });

    it('labels natural turn completion separately from a real force-listen interrupt', () => {
        expect(backendSource).toContain('reason="turn_end"');
        expect(backendSource).toContain('transition = listen_transition(');
        expect(backendSource).toContain('natural_end=result.end_of_turn, client_cancel=client_force_listen');
        expect(lifecycleSource).toContain("reason = 'force_listen' if force_listen or client_cancel else 'model_listen'");
        expect(lifecycleSource).toContain("reason = 'turn_end'");
    });
});
