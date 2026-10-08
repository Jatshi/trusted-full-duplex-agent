import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const html = fs.readFileSync(
    path.resolve('static/audio-duplex/audio_duplex.html'),
    'utf8',
);
const appSource = fs.readFileSync(
    path.resolve('static/audio-duplex/audio-duplex-app.js'),
    'utf8',
);

describe('TFD-STAR audio duplex presentation contract', () => {
    it('uses project-owned branding without the upstream model title', () => {
        expect(html).toContain('TFD-STAR');
        expect(html).toContain('Trusted Full-Duplex Speech Agent');
        expect(html).not.toContain('MiniCPM-o 4.5');
    });

    it('does not load the FAQ sidebar', () => {
        expect(html).not.toContain('faq-sidebar.js');
        expect(html).not.toContain('initFaqSidebar');
    });

    it('keeps every functional control required by the realtime client', () => {
        for (const id of [
            'btnStart', 'btnStop', 'btnPause', 'btnForceListen',
            'playbackDelay', 'adxMicDevice', 'adxSpeakerDevice',
            'chatLog', 'panelStatus', 'waveformCanvas',
        ]) {
            expect(html).toContain(`id="${id}"`);
        }
    });

    it('presents readable Chinese telemetry while preserving metric IDs', () => {
        expect(appSource).toContain('实时性能');
        expect(appSource).toContain('单片推理');
        expect(appSource).toContain('首次开口');
        expect(appSource).toContain('播放缓冲');
        expect(appSource).toContain('链路漂移');
        expect(appSource).toContain('id="ttfsDisplay"');
        expect(appSource).toContain('id="driftDisplay"');
    });
});
