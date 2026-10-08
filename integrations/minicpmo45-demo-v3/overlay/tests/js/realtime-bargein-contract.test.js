import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const source = fs.readFileSync(
    path.resolve('static/duplex/lib/realtime-session.js'),
    'utf8',
);
const appSource = fs.readFileSync(
    path.resolve('static/audio-duplex/audio-duplex-app.js'),
    'utf8',
);

describe('realtime hard barge-in contract', () => {
    it('cancels pre-scheduled audio only on client-confirmed listen', () => {
        expect(source).toContain('msg.metrics?.tfd_client_force_listen === true');
        expect(source).toContain('if (acknowledgesCancel)');
        expect(source).toContain('this.audioPlayer.stopAll()');
    });

    it('does not classify an MLP-selected force_listen as client barge-in', () => {
        expect(source).toContain('const explicitClientStop = msg.metrics?.tfd_client_force_listen === true');
        expect(source).not.toContain("reason === 'force_listen'");
        expect(source).not.toContain('!this._lastSpeakEndedTurn');
    });

    it('adapts the jitter buffer for localhost and public relays', () => {
        expect(appSource).toContain('isLocalForward ? 700 : 1200');
        expect(appSource).toContain('delayEl.value = String(recommendedDelayMs)');
        expect(appSource).toContain("audio_duplex_playback_buffer_v5");
        expect(source).toContain('this.audioPlayer.stopAll()');
    });

    it('supports one-shot client VAD barge-in without a sticky force-listen mode', () => {
        expect(source).toContain('requestBargeIn(meta = {})');
        expect(source).toContain('this._oneShotForceListen = true');
        expect(appSource).toContain('processLocalBargeInVad');
        expect(appSource).toContain("source: 'client_vad'");
    });
});
