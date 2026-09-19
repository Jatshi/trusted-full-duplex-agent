import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const source = fs.readFileSync(path.resolve('runtime/backend_client.py'), 'utf8');
const backendSource = fs.readFileSync(path.resolve('py_backend/server.py'), 'utf8');

describe('backend compile keepalive contract', () => {
    it('does not kill loopback WebSocket sessions during first-use torch.compile', () => {
        expect(source).toContain('ping_interval=None');
        expect(source).toContain('torch.compile can block');
    });

    it('labels natural turn completion separately from a real force-listen interrupt', () => {
        expect(backendSource).toContain('reason="turn_end"');
        expect(backendSource).toContain('reason="force_listen" if force_listen else "model_listen"');
    });
});
