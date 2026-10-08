// Test harness only: readiness is an observed advancing clock, never just state.
export async function withClockReady(context, run, {
    now = () => performance.now(),
    sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
    deadlineMs = 5000,
    onSample = () => {},
} = {}) {
    const began = now();
    const samples = [];
    let timer, expired = false;
    try {
        const readiness = await Promise.race([(async () => {
            await context.resume();
            let previous = context.currentTime, advances = 0;
            const sample = () => {
                const row = {wall_ms: now() - began, state: context.state,
                    context_time: context.currentTime};
                samples.push(row); onSample(row);
            };
            sample();
            while (!expired && now() - began < deadlineMs) {
                await sleep(100);
                if (expired || now() - began >= deadlineMs) break;
                sample();
                advances = context.state === 'running' && context.currentTime > previous
                    ? advances + 1 : 0;
                previous = context.currentTime;
                if (advances >= 2) return {wait_ms: now() - began, samples};
            }
            throw new Error('failed_readiness: clock did not advance twice before deadline');
        })(), new Promise((_, reject) => {
            timer = setTimeout(() => {
                expired = true;
                reject(new Error('failed_readiness: resume/clock deadline'));
            }, deadlineMs);
        })]);
        clearTimeout(timer);
        return await run(readiness);
    } finally {
        expired = true;
        clearTimeout(timer);
        await context.close();
    }
}
