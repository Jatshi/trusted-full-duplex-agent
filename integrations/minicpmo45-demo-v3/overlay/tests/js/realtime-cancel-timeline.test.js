import {afterEach, describe, expect, it, vi} from 'vitest';
import {RealtimeSession} from '../../static/duplex/lib/realtime-session.js';
import {AudioPlayer} from '../../static/duplex/lib/audio-player.js';

afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

function sessionFixture() {
    const session = new RealtimeSession('test');
    const player = {
        turnActive: true,
        playing: true,
        turnIdx: 3,
        playbackSnapshot: vi.fn(() => ({scheduledAheadMs: 1200, outputLatencyEstimateMs: 45})),
        stopAll: vi.fn(),
        endTurn: vi.fn(),
        beginTurn: vi.fn(),
        playChunk: vi.fn(),
    };
    session.audioPlayer = player;
    session.ws = {readyState: 1};
    session.onMetrics = vi.fn();
    session.onPlaybackCancel = vi.fn();
    vi.stubGlobal('WebSocket', {OPEN: 1});
    vi.stubGlobal('requestAnimationFrame', fn => fn());
    return {session, player};
}

describe('local cancel is scoped to one pending assistant turn', () => {
    it('suppresses late audio until explicit force-listen acknowledgment', () => {
        const {session, player} = sessionFixture();
        const clock = vi.spyOn(performance, 'now').mockReturnValue(500);
        expect(session.requestBargeIn({source: 'client_vad'})).toBe(true);
        expect(player.stopAll).toHaveBeenCalledTimes(1);
        expect(session.onPlaybackCancel).toHaveBeenCalledWith(expect.objectContaining({
            canceledClientTurn: 3,
            scheduledAheadMs: 1200,
            stopDispatchMs: expect.any(Number),
            audibleStopMeasured: false,
        }));
        expect(session.cancelTrace).toHaveLength(1);

        session._handleSpeak({audio: 'late-old-answer'});
        expect(player.playChunk).not.toHaveBeenCalled();
        session._handleListen({reason: 'ordinary_listen'});
        session._handleSpeak({audio: 'still-old-answer'});
        expect(player.playChunk).not.toHaveBeenCalled();
        session._handleListen({reason: 'force_listen', metrics: {tfd_client_force_listen: false}});
        session._handleSpeak({audio: 'not-our-ack'});
        expect(player.playChunk).not.toHaveBeenCalled();
        expect(session.cancelTrace[0].ackDelayMs).toBeUndefined();
        session._handleListen({reason: 'force_listen', metrics: {tfd_client_force_listen: true}});
        expect(session.cancelTrace[0].ackDelayMs).toBe(0);
        session._handleSpeak({audio: 'new-answer'});
        expect(player.playChunk).toHaveBeenCalledTimes(1);
        expect(player.playChunk).toHaveBeenCalledWith('new-answer', expect.any(Number));
        clock.mockRestore();
    });

    it('does not suppress a normal answer when no explicit client cancel occurred', () => {
        const {session, player} = sessionFixture();
        session._handleListen({reason: 'ordinary_listen'});
        session._handleSpeak({audio: 'normal-answer'});
        expect(player.playChunk).toHaveBeenCalledTimes(1);
        expect(session.onPlaybackCancel).not.toHaveBeenCalled();
    });

    it('does not cut a scheduled answer for an MLP-selected force_listen', () => {
        const {session, player} = sessionFixture();
        session._handleListen({reason: 'force_listen', metrics: {tfd_client_force_listen: false}});
        expect(player.stopAll).not.toHaveBeenCalled();
        expect(player.endTurn).toHaveBeenCalledTimes(1);
        expect(session.onPlaybackCancel).not.toHaveBeenCalled();
    });

    it('drops an old response_id even after the client acknowledgment', () => {
        const {session, player} = sessionFixture();
        session._handleSpeak({response_id: 'old-answer', audio: 'first-packet'});
        expect(session.requestBargeIn({source: 'client_vad'})).toBe(true);
        session._handleListen({
            response_id: 'old-answer', reason: 'force_listen',
            metrics: {tfd_client_force_listen: true},
        });
        player.playChunk.mockClear();
        session._handleSpeak({response_id: 'old-answer', audio: 'late-packet'});
        expect(player.playChunk).not.toHaveBeenCalled();
        session._handleSpeak({response_id: 'new-answer', audio: 'fresh-packet'});
        expect(player.playChunk).toHaveBeenCalledOnce();
        expect(player.playChunk).toHaveBeenCalledWith('fresh-packet', expect.any(Number));
        expect(session.cancelTrace[0].canceledResponseId).toBe('old-answer');
    });
});

describe('playback cancellation telemetry', () => {
    it('reports only an estimated output latency and queued browser audio', () => {
        const player = new AudioPlayer();
        player._ctx = {currentTime: 2, baseLatency: 0.02, outputLatency: 0.03};
        player._nextTime = 3.2;
        player._sources = [{}, {}];
        player._pendingChunks = [{}];
        expect(player.playbackSnapshot()).toEqual({
            scheduledAheadMs: 1200,
            activeSources: 2,
            pendingChunks: 1,
            outputLatencyEstimateMs: 50,
        });
    });
});
