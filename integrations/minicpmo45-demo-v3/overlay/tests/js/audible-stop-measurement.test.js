import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {
    AudibleStopRecorder,
    MEASUREMENT_EVENTS,
    MeasurementTimeline,
    canClaimAudibleP95,
} from '../../static/duplex/lib/audible-stop-measurement.js';

let fakeClock;
let tick;

beforeEach(() => {
    fakeClock = 1000;
    tick = ms => { fakeClock += ms; };
    vi.stubGlobal('performance', {now: () => fakeClock});
});

afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

function sessionStub() {
    return {
        onPlaybackCancel: null,
        audioPlayer: {turnActive: true, turnIdx: 7, playing: true},
    };
}

function recorderFor({probeAvailable = false} = {}) {
    const session = sessionStub();
    const timeline = new MeasurementTimeline({clock: () => fakeClock});
    const recorder = new AudibleStopRecorder(session, {timeline, probeAvailable});
    recorder.start();
    return {session, recorder, timeline};
}

describe('measurement timeline schema', () => {
    it('keeps one clock domain and rejects unknown event kinds', () => {
        const timeline = new MeasurementTimeline({clock: () => fakeClock});
        expect(timeline.clockId).toBe('performance.now');
        expect(() => timeline.record('made.up.event')).toThrow(/unknown measurement event/);
    });

    it('derives onset->cancel and cancel->dispatch deltas in ms', () => {
        const {recorder} = recorderFor();
        recorder.noteNearendOnset({rms: 0.12, peak: 0.4});
        tick(30);
        recorder.noteCancelRequested({source: 'client_vad'});
        tick(5);
        // Simulate the session dispatching the cancel synchronously.
        recorder.session.onPlaybackCancel({
            source: 'client_vad', canceledClientTurn: 7, canceledResponseId: 'r-1',
            scheduledAheadMs: 1400, stopDispatchMs: 5,
        });
        const report = recorder.report();
        expect(report.schema).toBe('cfdc.audible_stop.v1');
        expect(report.derived.onsetToCancelRequestMs).toBe(30);
        expect(report.derived.cancelRequestToDispatchMs).toBe(5);
        expect(report.eventCount).toBe(3);
    });

    it('never derives an audible stop without a physical probe', () => {
        const {recorder} = recorderFor();
        recorder.noteNearendOnset({rms: 0.1, peak: 0.3});
        tick(10);
        recorder.noteCancelRequested();
        tick(4);
        recorder.session.onPlaybackCancel({
            source: 'client_vad', canceledClientTurn: 7, canceledResponseId: 'r-1',
            scheduledAheadMs: 900, stopDispatchMs: 4,
        });
        const report = recorder.report();
        expect(report.derived.audibleStopMs).toBeNull();
        expect(report.derived.onsetToAudibleStopMs).toBeNull();
        expect(report.provenance.audibleStopMeasured).toBe(false);
        expect(report.provenance.dispatchIsPhysical).toBe(false);
        expect(canClaimAudibleP95(report)).toBe(false);
    });

    it('refuses to fabricate an audible stop in a headless run', () => {
        const {recorder} = recorderFor({probeAvailable: false});
        expect(() => recorder.markAudibleStopFromLoopback({tMs: 1234}))
            .toThrow(/refusing to mark audible stop without a physical loopback probe/);
    });

    it('accepts a physical stop only with a probe and a finite instant', () => {
        const {recorder} = recorderFor({probeAvailable: true});
        recorder.noteNearendOnset({rms: 0.1, peak: 0.3});
        tick(48);
        recorder.noteCancelRequested();
        tick(6);
        recorder.session.onPlaybackCancel({
            source: 'client_vad', canceledClientTurn: 7, canceledResponseId: 'r-1',
            scheduledAheadMs: 900, stopDispatchMs: 6,
        });
        tick(120);
        recorder.markAudibleStopFromLoopback({tMs: fakeClock, probeId: 'loop-1'});
        const report = recorder.report();
        expect(report.derived.onsetToAudibleStopMs).toBe(174);
        expect(report.provenance.audibleStopMeasured).toBe(true);
        expect(report.provenance.audibleStopSource).toBe('loopback_probe');
        expect(canClaimAudibleP95(report)).toBe(true);
        expect(() => recorder.markAudibleStopFromLoopback({}))
            .toThrow(/tMs from the loopback probe is required/);
    });
});

describe('negative control: no barge-in means no cancel', () => {
    it('produces no cancel events when the user never interrupts', () => {
        const {recorder, session} = recorderFor();
        // Playback runs to completion; no onset, no cancel.
        expect(session.onPlaybackCancel).toBeTypeOf('function');
        const report = recorder.report();
        expect(report.eventCount).toBe(0);
        expect(report.derived.onsetToCancelRequestMs).toBeNull();
        expect(report.derived.cancelRequestToDispatchMs).toBeNull();
        expect(report.derived.latePacketsDropped).toBe(0);
        expect(report.provenance.audibleStopMeasured).toBe(false);
    });

    it('counts dropped late packets so a false truncation is visible', () => {
        const {recorder} = recorderFor();
        recorder.noteNearendOnset({rms: 0.2, peak: 0.6});
        tick(20);
        recorder.noteCancelRequested();
        recorder.session.onPlaybackCancel({
            source: 'client_vad', canceledClientTurn: 3, canceledResponseId: 'old',
            scheduledAheadMs: 500, stopDispatchMs: 2,
        });
        tick(15);
        recorder.noteLatePacketDropped({responseId: 'old'});
        tick(5);
        recorder.noteLatePacketDropped({responseId: 'old'});
        const report = recorder.report();
        expect(report.derived.latePacketsDropped).toBe(2);
        expect(report.events.filter(e => e.kind === MEASUREMENT_EVENTS.FIRST_LATE_PACKET_DROPPED))
            .toHaveLength(2);
    });
});

describe('recorder lifecycle', () => {
    it('restores the original hook on stop and is idempotent on start', () => {
        const original = vi.fn();
        const session = sessionStub();
        session.onPlaybackCancel = original;
        const recorder = new AudibleStopRecorder(session, {probeAvailable: false});
        recorder.start();
        recorder.start();
        const wrapped = session.onPlaybackCancel;
        expect(wrapped).not.toBe(original);
        wrapped({source: 'client_vad', canceledClientTurn: 1});
        expect(original).toHaveBeenCalledTimes(1);
        expect(recorder.report().eventCount).toBe(1);
        recorder.stop();
        expect(session.onPlaybackCancel).toBe(original);
    });

    it('caps the event log so long sessions cannot grow unbounded', () => {
        const timeline = new MeasurementTimeline({clock: () => fakeClock});
        for (let i = 0; i < 2100; i++) {
            timeline.record(MEASUREMENT_EVENTS.NEAREND_ONSET, {rms: 0.1});
        }
        expect(timeline.events.length).toBe(2000);
        timeline.clear();
        expect(timeline.events).toHaveLength(0);
    });
});
