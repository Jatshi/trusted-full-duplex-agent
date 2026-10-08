/**
 * CFDC audible-stop measurement protocol (local / deterministic part).
 *
 * WHY THIS EXISTS
 * ---------------
 * `cancelTrace.stopDispatchMs` is the duration of a synchronous JS call. It is
 * NOT how long the speaker keeps making sound. `playbackSnapshot.
 * outputLatencyEstimateMs` is an AudioContext guess, not a physical
 * measurement. The handoff forbids reporting either as a human-audible stop
 * latency (HANDOFF §4).
 *
 * This module defines the *event timeline contract* and the browser-side
 * recorder that fills it, so that a later real-microphone capture can attach a
 * physical stop instant to the same schema. Nothing here measures sound.
 *
 * `AudibleStopRecorder` therefore keeps `audibleStopMeasured = false` and
 * `audibleStopMs = null` until an external loopback probe explicitly supplies a
 * physical mark. `markAudibleStopFromLoopback()` is the only way to flip it,
 * and it refuses to run in a headless/no-probe environment.
 */

/** Event kinds, kept as a closed set so downstream analysis cannot drift. */
export const MEASUREMENT_EVENTS = Object.freeze({
    NEAREND_ONSET: 'nearend.onset',
    CANCEL_REQUESTED: 'cancel.requested',
    CANCEL_DISPATCHED: 'cancel.dispatched',
    SERVER_ACK: 'cancel.server_ack',
    FIRST_LATE_PACKET_DROPPED: 'late_packet.dropped',
    AUDIBLE_STOP: 'audible.stop',
});

function nowMs(clock) {
    if (typeof clock === 'function') return clock();
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return 0;
}

/**
 * Monotonic event timeline. All entries share one clock domain; the header
 * records which domain, because mixing `performance.now()` with `Date.now()`
 * silently produces meaningless deltas.
 */
export class MeasurementTimeline {
    constructor({clock = null, clockId = 'performance.now'} = {}) {
        this._clock = clock;
        this.clockId = clockId;
        this._events = [];
        this._sessionId = null;
    }

    get events() { return this._events.map(entry => ({...entry})); }

    setSession(sessionId) {
        if (typeof sessionId === 'string' && sessionId) this._sessionId = sessionId;
    }

    record(kind, detail = {}) {
        if (!Object.values(MEASUREMENT_EVENTS).includes(kind)) {
            throw new Error(`unknown measurement event: ${kind}`);
        }
        const entry = {
            kind,
            tMs: nowMs(this._clock),
            ...detail,
        };
        this._events.push(entry);
        if (this._events.length > 2000) this._events.shift();
        return entry;
    }

    /** First event of a kind, or null. */
    first(kind) {
        return this._events.find(entry => entry.kind === kind) || null;
    }

    all(kind) {
        return this._events.filter(entry => entry.kind === kind).map(e => ({...e}));
    }

    clear() {
        this._events = [];
        this._sessionId = null;
    }

    /**
     * Machine-readable export. `audibleStopMs` stays null unless a physical
     * probe marked it, and `provenance` states exactly which clocks were in
     * play so no reader can mistake dispatch time for audible time.
     */
    toReport() {
        const onset = this.first(MEASUREMENT_EVENTS.NEAREND_ONSET);
        const requested = this.first(MEASUREMENT_EVENTS.CANCEL_REQUESTED);
        const dispatched = this.first(MEASUREMENT_EVENTS.CANCEL_DISPATCHED);
        const ack = this.first(MEASUREMENT_EVENTS.SERVER_ACK);
        const audible = this.first(MEASUREMENT_EVENTS.AUDIBLE_STOP);
        const dropped = this.all(MEASUREMENT_EVENTS.FIRST_LATE_PACKET_DROPPED);
        const delta = (a, b) => (a && b ? Math.max(0, a.tMs - b.tMs) : null);
        return {
            schema: 'cfdc.audible_stop.v1',
            sessionId: this._sessionId,
            clockId: this.clockId,
            eventCount: this._events.length,
            events: this.events,
            derived: {
                onsetToCancelRequestMs: delta(requested, onset),
                cancelRequestToDispatchMs: delta(dispatched, requested),
                cancelRequestToServerAckMs: delta(ack, requested),
                onsetToAudibleStopMs: delta(audible, onset),
                audibleStopMs: audible ? audible.tMs - (onset ? onset.tMs : audible.tMs) : null,
                latePacketsDropped: dropped.length,
            },
            provenance: {
                audibleStopMeasured: Boolean(audible),
                audibleStopSource: audible ? (audible.source || null) : null,
                dispatchIsPhysical: false,
                note: 'stopDispatchMs / outputLatencyEstimateMs are dispatch and '
                    + 'estimator values, not speaker-side measurements.',
            },
        };
    }
}

/**
 * Wraps a RealtimeSession to record the cancel timeline. Deliberately does not
 * compute audible latency; it only sequences observable events.
 */
export class AudibleStopRecorder {
    constructor(session, {timeline = null, probeAvailable = false} = {}) {
        if (!session) throw new Error('session is required');
        this.session = session;
        this.timeline = timeline || new MeasurementTimeline();
        this.probeAvailable = Boolean(probeAvailable);
        this._originals = null;
    }

    start() {
        if (this._originals) return this;
        const session = this.session;
        const timeline = this.timeline;
        const originals = {};
        this._originals = originals;

        originals.onPlaybackCancel = session.onPlaybackCancel;
        session.onPlaybackCancel = event => {
            timeline.record(MEASUREMENT_EVENTS.CANCEL_DISPATCHED, {
                source: event.source,
                canceledClientTurn: event.canceledClientTurn,
                canceledResponseId: event.canceledResponseId,
                scheduledAheadMs: event.scheduledAheadMs,
                stopDispatchMs: event.stopDispatchMs,
                physical: false,
            });
            if (typeof originals.onPlaybackCancel === 'function') {
                originals.onPlaybackCancel(event);
            }
        };
        return this;
    }

    stop() {
        if (!this._originals) return;
        this.session.onPlaybackCancel = this._originals.onPlaybackCancel;
        this._originals = null;
    }

    /** Call from the client VAD when playback is active and speech is detected. */
    noteNearendOnset({rms = null, peak = null} = {}) {
        return this.timeline.record(MEASUREMENT_EVENTS.NEAREND_ONSET, {rms, peak});
    }

    /** Call immediately before invoking `session.requestBargeIn`. */
    noteCancelRequested({source = 'client_vad'} = {}) {
        return this.timeline.record(MEASUREMENT_EVENTS.CANCEL_REQUESTED, {source});
    }

    /** Call when `_handleListen` acknowledges a client cancel. */
    noteServerAck({responseId = null, ackDelayMs = null} = {}) {
        return this.timeline.record(MEASUREMENT_EVENTS.SERVER_ACK, {responseId, ackDelayMs});
    }

    /** Call when a late packet is rejected by response_id / pending-cancel guard. */
    noteLatePacketDropped({responseId = null} = {}) {
        return this.timeline.record(MEASUREMENT_EVENTS.FIRST_LATE_PACKET_DROPPED, {responseId});
    }

    /**
     * The ONLY way to set an audible stop instant. Requires a physical probe
     * (loopback capture) to have been declared available; refuses otherwise so
     * a headless run can never manufacture a p95.
     */
    markAudibleStopFromLoopback({tMs = null, probeId = null} = {}) {
        if (!this.probeAvailable) {
            throw new Error(
                'refusing to mark audible stop without a physical loopback probe; '
                + 'local dispatch timing is not an audible measurement',
            );
        }
        if (typeof tMs !== 'number' || !Number.isFinite(tMs)) {
            throw new Error('tMs from the loopback probe is required');
        }
        return this.timeline.record(MEASUREMENT_EVENTS.AUDIBLE_STOP, {
            source: 'loopback_probe',
            probeId,
            physical: true,
        });
    }

    report() {
        return this.timeline.toReport();
    }
}

/** Convenience: does this report support a human-audible p95 claim? */
export function canClaimAudibleP95(report) {
    return Boolean(report && report.provenance && report.provenance.audibleStopMeasured);
}
