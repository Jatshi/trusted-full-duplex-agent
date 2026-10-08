/**
 * End-to-end timeline regression: real RealtimeSession driven by simulated
 * server events, observed through AudibleStopRecorder.
 *
 * The contract test only checks source strings. This one drives the actual
 * code paths so the recorded timeline is what the shipping code produces, not
 * what a mock does. It still asserts NO physical latency anywhere.
 */
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {RealtimeSession} from '../../static/duplex/lib/realtime-session.js';
import {
    AudibleStopRecorder,
    MEASUREMENT_EVENTS,
    canClaimAudibleP95,
} from '../../static/duplex/lib/audible-stop-measurement.js';

let clock;

beforeEach(() => {
    clock = 5000;
    vi.stubGlobal('performance', {now: () => clock});
    vi.stubGlobal('requestAnimationFrame', fn => fn());
    vi.stubGlobal('WebSocket', {OPEN: 1});
});

afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

function makeSession() {
    const session = new RealtimeSession('e2e');
    const player = {
        turnActive: true,
        playing: true,
        turnIdx: 1,
        _snapshot: {scheduledAheadMs: 1800, activeSources: 3, pendingChunks: 0,
            outputLatencyEstimateMs: 55},
        playbackSnapshot: vi.fn(function () { return {...this._snapshot}; }),
        stopAll: vi.fn(function () { this.playing = false; }),
        endTurn: vi.fn(function () { this.turnActive = false; }),
        beginTurn: vi.fn(function () { this.turnActive = true; this.turnIdx++; }),
        playChunk: vi.fn(),
    };
    session.audioPlayer = player;
    session.ws = {readyState: 1, send: vi.fn()};
    session.onMetrics = vi.fn();
    session.onSystemLog = vi.fn();
    session.onListenResult = vi.fn();
    session.onSpeakStart = vi.fn();
    session.onSpeakEnd = vi.fn();
    session.onSpeakUpdate = vi.fn();
    session.onExtraResult = vi.fn();
    session.onForceListenChange = vi.fn();
    session.onPauseStateChange = vi.fn();
    session.onRunningChange = vi.fn();
    return {session, player};
}

function withRecorder(session, {probeAvailable = false} = {}) {
    const recorder = new AudibleStopRecorder(session, {probeAvailable});
    const prevCancel = session.onPlaybackCancel;
    const prevAck = session.onCancelAck;
    const prevDrop = session.onLatePacketDropped;
    recorder.start();
    const recordedCancel = session.onPlaybackCancel;
    session.onPlaybackCancel = event => {
        recordedCancel(event);
        if (typeof prevCancel === 'function') prevCancel(event);
    };
    session.onCancelAck = event => {
        recorder.noteServerAck({
            responseId: event?.canceledResponseId ?? null,
            ackDelayMs: event?.ackDelayMs ?? null,
        });
        if (typeof prevAck === 'function') prevAck(event);
    };
    session.onLatePacketDropped = responseId => {
        recorder.noteLatePacketDropped({responseId: responseId ?? null});
        if (typeof prevDrop === 'function') prevDrop(responseId);
    };
    return recorder;
}

describe('E2E: explicit client barge-in produces a complete timeline', () => {
    it('records onset -> request -> dispatch -> ack, and drops the old tail', () => {
        const {session, player} = makeSession();
        const recorder = withRecorder(session);

        // 1. assistant is speaking
        session._handleSpeak({response_id: 'r-old', audio: 'pkt-1'});
        expect(player.playChunk).toHaveBeenCalledTimes(1);

        // 2. user starts speaking -> VAD fires
        clock += 120;
        recorder.noteNearendOnset({rms: 0.18, peak: 0.52});

        // 3. client requests barge-in
        clock += 8;
        recorder.noteCancelRequested({source: 'client_vad'});
        expect(session.requestBargeIn({source: 'client_vad'})).toBe(true);
        expect(player.stopAll).toHaveBeenCalledTimes(1);

        // 4. late packet from the canceled answer arrives before the ack
        clock += 30;
        session._handleSpeak({response_id: 'r-old', audio: 'late'});
        expect(player.playChunk).toHaveBeenCalledTimes(1); // still 1

        // 5. server acks the explicit client cancel
        clock += 40;
        session._handleListen({
            response_id: 'r-old', reason: 'force_listen',
            metrics: {tfd_client_force_listen: true},
        });

        // 6. a genuinely new answer is allowed through
        clock += 25;
        session._handleSpeak({response_id: 'r-new', audio: 'fresh'});
        expect(player.playChunk).toHaveBeenCalledTimes(2);

        const report = recorder.report();
        expect(report.schema).toBe('cfdc.audible_stop.v1');
        expect(report.derived.onsetToCancelRequestMs).toBe(8);
        expect(report.derived.cancelRequestToServerAckMs).toBe(70);
        expect(report.derived.latePacketsDropped).toBe(1);
        expect(report.events.map(e => e.kind)).toEqual([
            MEASUREMENT_EVENTS.NEAREND_ONSET,
            MEASUREMENT_EVENTS.CANCEL_REQUESTED,
            MEASUREMENT_EVENTS.CANCEL_DISPATCHED,
            MEASUREMENT_EVENTS.FIRST_LATE_PACKET_DROPPED,
            MEASUREMENT_EVENTS.SERVER_ACK,
        ]);
        // The sentinel: still not a physical measurement.
        expect(report.derived.audibleStopMs).toBeNull();
        expect(report.provenance.dispatchIsPhysical).toBe(false);
        expect(canClaimAudibleP95(report)).toBe(false);
    });

    it('captures dispatch evidence without ever becoming a physical claim', () => {
        const {session} = makeSession();
        const recorder = withRecorder(session);
        recorder.noteNearendOnset({rms: 0.2, peak: 0.6});
        recorder.noteCancelRequested({source: 'client_vad'});
        session.requestBargeIn({source: 'client_vad'});
        const dispatched = recorder.report().events
            .find(e => e.kind === MEASUREMENT_EVENTS.CANCEL_DISPATCHED);
        expect(dispatched).toBeTruthy();
        expect(dispatched.physical).toBe(false);
        expect(dispatched.scheduledAheadMs).toBe(1800);
        expect(dispatched.stopDispatchMs).toBeGreaterThanOrEqual(0);
    });
});

describe('E2E: negative control — a plain MLP listen must not truncate', () => {
    it('keeps the answer audible and records no cancel at all', () => {
        const {session, player} = makeSession();
        const recorder = withRecorder(session);

        session._handleSpeak({response_id: 'r-1', audio: 'pkt-1'});
        session._handleSpeak({response_id: 'r-1', audio: 'pkt-2'});
        expect(player.playChunk).toHaveBeenCalledTimes(2);

        // The turn MLP decides "listen" on its own — no client barge-in.
        session._handleListen({
            response_id: 'r-1', reason: 'force_listen',
            metrics: {tfd_client_force_listen: false},
        });

        // must NOT stop pre-scheduled audio
        expect(player.stopAll).not.toHaveBeenCalled();
        // and the next chunk of the SAME answer still plays
        session._handleSpeak({response_id: 'r-1', audio: 'pkt-3'});
        expect(player.playChunk).toHaveBeenCalledTimes(3);

        const report = recorder.report();
        expect(report.eventCount).toBe(0);
        expect(report.derived.onsetToCancelRequestMs).toBeNull();
        expect(report.derived.latePacketsDropped).toBe(0);
        expect(report.provenance.audibleStopMeasured).toBe(false);
    });

    it('does not treat ordinary listen/turn_end as a cancel either', () => {
        const {session, player} = makeSession();
        const recorder = withRecorder(session);
        session._handleSpeak({response_id: 'r-9', audio: 'pkt-1'});
        session._handleListen({reason: 'ordinary_listen'});
        session._handleSpeak({response_id: 'r-9', audio: 'pkt-2'});
        expect(player.stopAll).not.toHaveBeenCalled();
        expect(player.playChunk).toHaveBeenCalledTimes(2);
        expect(recorder.report().eventCount).toBe(0);
    });
});

describe('E2E: repeated barge-in is throttled and each run is recorded', () => {
    it('ignores marked termination for an old response without clearing the current answer', () => {
        const {session, player} = makeSession();
        session.onSpeakStart.mockReturnValue('handle-new');
        session.onLatePacketDropped = vi.fn();
        session._handleSpeak({response_id: 'r-new', text: 'Current', audio: 'pkt'});
        session._handleListen({response_id: 'r-old', reason: 'turn_end', metrics: {
            tfd_lifecycle_enabled: true, tfd_response_terminated: true,
        }});
        expect(player.endTurn).not.toHaveBeenCalled();
        expect(session.onSpeakEnd).not.toHaveBeenCalled();
        expect(session.currentSpeakText).toBe('Current');
        expect(session.onLatePacketDropped).toHaveBeenCalledWith('r-old');
        session._handleListen({response_id: 'r-new', reason: 'turn_end', metrics: {
            tfd_lifecycle_enabled: true, tfd_response_terminated: true,
        }});
        expect(player.endTurn).toHaveBeenCalledTimes(1);
    });

    it('does not acknowledge marked client cancellation for the wrong response ID', () => {
        const {session} = makeSession();
        session.onCancelAck = vi.fn();
        session._handleSpeak({response_id: 'r-target', audio: 'pkt'});
        session.requestBargeIn({source: 'manual'});
        const pending = session._pendingExplicitCancel;
        session._handleListen({response_id: 'r-other', reason: 'force_listen', metrics: {
            tfd_lifecycle_enabled: true, tfd_response_terminated: true,
            tfd_client_force_listen: true,
        }});
        expect(session._pendingExplicitCancel).toBe(pending);
        expect(session.onCancelAck).not.toHaveBeenCalled();
        session._handleListen({response_id: 'r-target', reason: 'force_listen', metrics: {
            tfd_lifecycle_enabled: true, tfd_response_terminated: true,
            tfd_client_force_listen: true,
        }});
        expect(session._pendingExplicitCancel).toBeNull();
        expect(session.onCancelAck).toHaveBeenCalledTimes(1);
    });

    it('keeps marked ordinary LISTEN in the same response and closes only on termination', () => {
        const {session, player} = makeSession();
        session.onSpeakStart.mockReturnValue('text-handle');
        session._handleSpeak({response_id: 'r-life', text: 'Hello', audio: 'pkt-1'});
        session._handleListen({response_id: 'r-life', reason: 'model_listen', metrics: {
            tfd_lifecycle_enabled: true, tfd_response_terminated: false,
        }});
        expect(player.endTurn).not.toHaveBeenCalled();
        expect(session.onSpeakEnd).not.toHaveBeenCalled();
        session._handleSpeak({response_id: 'r-life', text: ' world', audio: 'pkt-2'});
        expect(session.currentSpeakText).toBe('Hello world');
        expect(session.onSpeakStart).toHaveBeenCalledTimes(1);
        expect(player.stopAll).not.toHaveBeenCalled();
        session._handleListen({response_id: 'r-life', reason: 'turn_end', metrics: {
            tfd_lifecycle_enabled: true, tfd_response_terminated: true,
        }});
        expect(player.endTurn).toHaveBeenCalledTimes(1);
        expect(session.onSpeakEnd).toHaveBeenCalledTimes(1);
        expect(player.stopAll).not.toHaveBeenCalled();
    });

    it('rejects a second barge-in inside the debounce window', () => {
        const {session} = makeSession();
        const recorder = withRecorder(session);
        recorder.noteCancelRequested({source: 'client_vad'});
        expect(session.requestBargeIn({source: 'client_vad'})).toBe(true);
        clock += 300; // inside the 1200 ms guard
        expect(session.requestBargeIn({source: 'client_vad'})).toBe(false);
        clock += 1500; // outside
        expect(session.requestBargeIn({source: 'client_vad'})).toBe(true);
        const dispatched = recorder.report().events
            .filter(e => e.kind === MEASUREMENT_EVENTS.CANCEL_DISPATCHED);
        expect(dispatched).toHaveLength(2);
    });
});
