import {describe, expect, it, vi} from 'vitest';
import {withClockReady} from '../browser/audio-clock-ready.js';

function fixture(times) {
    let wall = 0, index = 0;
    const context = {state: 'running', currentTime: times[0], resume: vi.fn(async () => {}),
        close: vi.fn(async () => {})};
    return {context, options: {now: () => wall, sleep: async ms => {
        wall += ms; context.currentTime = times[Math.min(++index, times.length - 1)];
    }}};
}
describe('bounded clock readiness, test harness only', () => {
    it('rejects a running but permanently zero clock without invoking the scenario', async () => {
        const {context, options} = fixture([0]);
        const run = vi.fn();
        await expect(withClockReady(context, run, options)).rejects.toThrow('failed_readiness');
        expect(run).not.toHaveBeenCalled();
        expect(context.close).toHaveBeenCalledOnce();
    });
    it('requires two consecutive advances and records startup zeros', async () => {
        const {context, options} = fixture([0, 0, 0.1, 0.2]);
        const report = await withClockReady(context, async readiness => readiness, options);
        expect(report.wait_ms).toBe(300);
        expect(report.samples.map(row => row.context_time)).toEqual([0, 0, 0.1, 0.2]);
        expect(context.close).toHaveBeenCalledOnce();
    });
    it('does not accept a second advance at the deadline', async () => {
        const {context, options} = fixture([0, 0.1, 0.2]);
        await expect(withClockReady(context, vi.fn(), {...options, deadlineMs: 200}))
            .rejects.toThrow('failed_readiness');
        expect(context.close).toHaveBeenCalledOnce();
    });
    it('closes the context when a ready scenario throws', async () => {
        const {context, options} = fixture([0, 0.1, 0.2]);
        await expect(withClockReady(context, async () => {throw new Error('scenario failure');}, options))
            .rejects.toThrow('scenario failure');
        expect(context.close).toHaveBeenCalledOnce();
    });
});
