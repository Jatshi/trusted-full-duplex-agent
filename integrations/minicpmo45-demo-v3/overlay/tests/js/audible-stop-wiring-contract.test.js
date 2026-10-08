import {describe, expect, it} from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const appSource = fs.readFileSync(
    path.resolve('static/audio-duplex/audio-duplex-app.js'),
    'utf8',
);
const sessionSource = fs.readFileSync(
    path.resolve('static/duplex/lib/realtime-session.js'),
    'utf8',
);
const measurementSource = fs.readFileSync(
    path.resolve('static/duplex/lib/audible-stop-measurement.js'),
    'utf8',
);

describe('audible-stop measurement wiring contract', () => {
    it('imports the measurement module and keeps a single recorder handle', () => {
        expect(appSource).toContain("from '../duplex/lib/audible-stop-measurement.js'");
        expect(appSource).toContain('let audibleStopRecorder = null');
        expect(appSource).toContain('new AudibleStopRecorder(session, {');
    });

    it('gates the physical probe behind an explicit flag, defaulting to off', () => {
        expect(appSource).toContain('probeAvailable: Boolean(window.__TFD_LOOPBACK_PROBE__)');
        // The flag must never be set true by the module itself.
        expect(appSource).not.toContain('__TFD_LOOPBACK_PROBE__ = true');
        expect(measurementSource).toContain('refusing to mark audible stop without a physical loopback probe');
    });

    it('notes onset and cancel request from the local VAD path', () => {
        expect(appSource).toContain('audibleStopRecorder.noteNearendOnset({');
        expect(appSource).toContain("audibleStopRecorder.noteCancelRequested({source: 'client_vad'})");
        expect(appSource).toContain("source: 'client_vad',");
    });

    it('exposes a report handle without claiming a speaker-side latency', () => {
        expect(appSource).toContain('window.tfdAudibleStopReport = () =>');
        expect(appSource).toContain('audibleStopRecorder.report()');
        expect(appSource).not.toContain('audibleStopMeasured: true');
    });

    it('chains the original onPlaybackCancel instead of replacing it', () => {
        expect(appSource).toContain('const _prevPlaybackCancel = session.onPlaybackCancel');
        expect(appSource).toContain('if (typeof _prevPlaybackCancel === \'function\') _prevPlaybackCancel(event)');
    });

    it('emits ack and late-packet hooks from the session', () => {
        expect(sessionSource).toContain('this.onCancelAck?.(ackEvent)');
        expect(sessionSource).toContain('this.onLatePacketDropped?.(msg.response_id)');
        expect(sessionSource).toContain('onCancelAck(event) {}');
        expect(sessionSource).toContain('onLatePacketDropped(responseId) {}');
    });

    it('keeps the ack hook fired only on an explicit client cancel', () => {
        expect(sessionSource).toContain(
            'const acknowledgesCancel = Boolean(this._pendingExplicitCancel && explicitClientStop)');
        expect(sessionSource).toContain('if (acknowledgesCancel) {');
    });
});
