// Production player, deterministic context; no browser/device acoustic evidence.
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {AudioPlayer} from '../../static/duplex/lib/audio-player.js';
import {RealtimeSession} from '../../static/duplex/lib/realtime-session.js';

beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('requestAnimationFrame', fn => fn());
});
afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

function fixture() {
    const sources = [];
    const player = new AudioPlayer({outputSampleRate: 24000, getPlaybackDelayMs: () => 0});
    player._outputSR = 24000;
    player._ctx = {
        currentTime: 10, state: 'running', destination: {},
        createBuffer: (_channels, length, rate) => ({
            duration: length / rate, getChannelData: () => new Float32Array(length),
        }),
        createBufferSource: () => {
            const source = {connect: vi.fn(), start: vi.fn(), stop: vi.fn(), disconnect: vi.fn()};
            sources.push(source);
            return source;
        },
    };
    const pcm = Buffer.from(new Float32Array(24000).fill(0.1).buffer).toString('base64');
    return {player, sources, pcm};
}

describe('natural-end queued tail policy', () => {
    it('routes serialized marked lifecycle and cancellation frames into the real player without cross-ID release', () => {
        vi.stubGlobal('WebSocket', {OPEN: 1});
        const {player, sources, pcm} = fixture();
        const session = new RealtimeSession('message-route');
        session.audioPlayer = player;
        session.ws = {readyState: 1, send: vi.fn()};
        session._checkKvCache = vi.fn();
        session._emitMetrics = vi.fn();
        session.onSpeakStart = vi.fn(() => 'text-handle');
        session.onSpeakEnd = vi.fn();
        session.onCancelAck = vi.fn();
        session.onLatePacketDropped = vi.fn();
        const active = {tfd_lifecycle_enabled: true, tfd_response_terminated: false};
        const terminated = {...active, tfd_response_terminated: true};
        const ack = {...terminated, tfd_client_force_listen: true};
        const deliver = frame => session._handleMessage(JSON.parse(JSON.stringify(frame)));
        const speak = (id, text) => deliver({type: 'response.output_audio.delta',
            response_id: id, audio: pcm, text, metrics: active});
        const listen = (id, metrics) => deliver({type: 'response.listen',
            response_id: id, metrics});
        speak('r1', 'First');
        listen('r1', active);
        expect(player.turnActive).toBe(true);
        expect(session.currentSpeakText).toBe('First');
        expect(session.onSpeakEnd).not.toHaveBeenCalled();
        listen('r1', terminated);
        expect(player.turnActive).toBe(false);
        expect(session.currentSpeakText).toBe('');
        expect(sources[0].stop).not.toHaveBeenCalled();
        speak('r2', 'Second');
        expect(sources[1].start).toHaveBeenCalledWith(11);
        expect(session._playbackResponseId).toBe('r2');
        expect(session.requestBargeIn({source: 'test'})).toBe(true);
        expect(player.playbackSnapshot().activeSources).toBe(0);
        expect(sources[0].stop).toHaveBeenCalledOnce();
        expect(sources[1].stop).toHaveBeenCalledOnce();
        const pending = session._pendingExplicitCancel;
        speak('r2', 'Stale');
        listen('wrong', ack);
        expect(session._pendingExplicitCancel).toBe(pending);
        expect(session.onCancelAck).not.toHaveBeenCalled();
        expect(sources).toHaveLength(2);
        listen('r2', ack);
        expect(session._pendingExplicitCancel).toBeNull();
        expect(session.onCancelAck).toHaveBeenCalledOnce();
        expect(session.currentSpeakText).toBe('');
        speak('r2', 'Still stale');
        expect(sources).toHaveLength(2);
        speak('r3', 'Third');
        expect(session._playbackResponseId).toBe('r3');
        expect(session.currentSpeakText).toBe('Third');
        expect(sources[2].start).toHaveBeenCalledWith(10);
        expect(player.turnActive).toBe(true);
        expect(session.onLatePacketDropped.mock.calls.map(call => call[0]))
            .toEqual(['r2', 'wrong', 'r2']);
        player.stopAll(); player.endTurn();
    });
    it('consumes natural ended sources without stopping or rescheduling either tail', () => {
        const {player, sources, pcm} = fixture();
        player.beginTurn(); player.playChunk(pcm); player.endTurn();
        player.beginTurn({preserveQueuedAudio: true}); player.playChunk(pcm); player.endTurn();
        sources[0].onended();
        expect(player.playbackSnapshot().activeSources).toBe(1);
        sources[1].onended();
        expect(player.playbackSnapshot().activeSources).toBe(0);
        expect(sources[0].stop).not.toHaveBeenCalled();
        expect(sources[1].stop).not.toHaveBeenCalled();
        expect(sources[1].start).toHaveBeenCalledOnce();
        expect(player.nextTime).toBe(12);
    });

    it('late canceled-source ended callback cannot consume a newer response source', () => {
        const {player, sources, pcm} = fixture();
        player.beginTurn(); player.playChunk(pcm);
        player.stopAll(); player.endTurn();
        player.beginTurn(); player.playChunk(pcm);
        sources[0].onended();
        expect(sources[0].stop).toHaveBeenCalledOnce();
        expect(player.playbackSnapshot().activeSources).toBe(1);
        expect(sources[1].stop).not.toHaveBeenCalled();
        sources[1].onended();
        expect(player.playbackSnapshot().activeSources).toBe(0);
        player.endTurn();
    });
    it('reports zero queued duration immediately after cancel while retaining natural-end tail evidence', () => {
        const {player, sources, pcm} = fixture();
        player.beginTurn();
        player.playChunk(pcm);
        player.endTurn();
        player.beginTurn({preserveQueuedAudio: true});
        player.playChunk(pcm);
        player.endTurn();
        expect(player.playbackSnapshot().scheduledAheadMs).toBe(2000);
        expect(player.lastAheadMs).toBe(2000);
        expect(sources[0].stop).not.toHaveBeenCalled();
        player.stopAll();
        expect(player.playbackSnapshot()).toMatchObject({
            scheduledAheadMs: 0, activeSources: 0, pendingChunks: 0,
        });
        expect(player.lastAheadMs).toBe(0);
        player.stopAll();
        expect(sources[0].stop).toHaveBeenCalledOnce();
        expect(sources[1].stop).toHaveBeenCalledOnce();
    });

    it('retains legacy replacement of an unfinished tail by default', () => {
        const {player, sources, pcm} = fixture();
        player.beginTurn();
        player.playChunk(pcm);
        player.endTurn();
        expect(sources[0].stop).not.toHaveBeenCalled();
        player.beginTurn();
        player.playChunk(pcm);
        expect(sources[0].stop).toHaveBeenCalledOnce();
        expect(sources[1].start).toHaveBeenCalledWith(10);
    });

    it('opt-in queues the next answer after the natural-end tail without stopping or overlapping it', () => {
        const {player, sources, pcm} = fixture();
        player.beginTurn();
        player.playChunk(pcm);
        player.endTurn();
        player.beginTurn({preserveQueuedAudio: true});
        player.playChunk(pcm);
        expect(sources[0].stop).not.toHaveBeenCalled();
        expect(sources[1].start).toHaveBeenCalledWith(11);
        expect(player.nextTime).toBe(12);
    });

    it('explicit stop clears both tails and does not leave a phantom delay on the next answer', () => {
        const {player, sources, pcm} = fixture();
        player.beginTurn();
        player.playChunk(pcm);
        player.endTurn();
        player.beginTurn({preserveQueuedAudio: true});
        player.playChunk(pcm);
        player.stopAll();
        player.endTurn();
        expect(sources[0].stop).toHaveBeenCalledOnce();
        expect(sources[1].stop).toHaveBeenCalledOnce();
        player.beginTurn({preserveQueuedAudio: true});
        player.playChunk(pcm);
        expect(sources[2].start).toHaveBeenCalledWith(10);
    });

    it('RealtimeSession opts into tail preservation only for marked identity-aware events', () => {
        const {player, sources, pcm} = fixture();
        const session = new RealtimeSession('tail');
        session.audioPlayer = player;
        session._checkKvCache = vi.fn();
        session._emitMetrics = vi.fn();
        session.onExtraResult = vi.fn();
        session.onListenResult = vi.fn();
        const metrics = {tfd_lifecycle_enabled: true, tfd_response_terminated: false};
        session._handleSpeak({response_id: 'r1', audio: pcm, metrics});
        session._handleListen({response_id: 'r1', metrics: {...metrics, tfd_response_terminated: true}});
        session._handleSpeak({response_id: 'r2', audio: pcm, metrics});
        expect(sources[0].stop).not.toHaveBeenCalled();
        expect(sources[1].start).toHaveBeenCalledWith(11);
        session._handleListen({response_id: 'r2', metrics: {...metrics, tfd_response_terminated: true}});
        session._handleSpeak({response_id: 'legacy', audio: pcm});
        expect(sources[0].stop).toHaveBeenCalledOnce();
        expect(sources[1].stop).toHaveBeenCalledOnce();
        expect(sources[2].start).toHaveBeenCalledWith(10);
    });
});
