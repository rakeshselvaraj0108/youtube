import { describe, expect, it } from 'vitest';
import { buildFfmpegCommand, compileRemediation } from '@/lib/ffmpeg';
import type { RemediationOp } from '@/types/analysis';
import { beforeReport } from '@/data/fixture';

const bleep: RemediationOp = {
  index: 1,
  op: 'BLEEP',
  startMs: 252_400,
  endMs: 254_100,
  details: 'strong profanity',
  findingId: 'f_02',
  freqHz: 1000,
};

const mute: RemediationOp = {
  index: 2,
  op: 'MUTE',
  startMs: 910_000,
  endMs: 916_000,
  details: 'casualty figure',
  findingId: 'f_04',
};

const blur: RemediationOp = {
  index: 3,
  op: 'BLUR_REGION',
  startMs: 454_000,
  endMs: 457_000,
  details: 'wound region',
  findingId: 'f_01',
  box: [0.29, 0.35, 0.42, 0.3],
};

describe('the -c:v copy fast path', () => {
  it('stream-copies the video when the EDL contains no video ops', () => {
    const result = compileRemediation([bleep, mute]);
    expect(result.videoStreamCopied).toBe(true);
    expect(result.command).toContain('-c:v copy');
    expect(result.command).not.toContain('libx264');
  });

  it('re-encodes as soon as a video op is present', () => {
    const result = compileRemediation([bleep, mute, blur]);
    expect(result.videoStreamCopied).toBe(false);
    expect(result.command).toContain('libx264');
    expect(result.command).not.toContain('-c:v copy');
  });

  it('passes an empty EDL straight through', () => {
    const result = compileRemediation([]);
    expect(result.videoStreamCopied).toBe(true);
    expect(result.command).toBe('ffmpeg -y -i input.mp4 -c copy output.safe.mp4');
  });
});

describe('codegen', () => {
  it('emits a sine input whose duration matches the bleep span', () => {
    // 254100 - 252400 = 1700ms
    expect(buildFfmpegCommand([bleep])).toContain('sine=frequency=1000:duration=1.700');
  });

  it('delays the tone to the op start in milliseconds', () => {
    expect(buildFfmpegCommand([bleep])).toContain('adelay=252400|252400');
  });

  it('silences the source underneath every audio op', () => {
    const cmd = buildFfmpegCommand([bleep, mute]);
    expect(cmd).toContain("volume=enable='between(t,252.4,254.1)':volume=0");
    expect(cmd).toContain("volume=enable='between(t,910.0,916.0)':volume=0");
  });

  it('derives crop and overlay geometry from the box, not a template', () => {
    const cmd = buildFfmpegCommand([blur]);
    expect(cmd).toContain('crop=iw*0.42:ih*0.3:iw*0.29:ih*0.35');
    expect(cmd).toContain("overlay=iw*0.29:ih*0.35:enable='between(t,454.0,457.0)'");
    expect(cmd).toContain('boxblur=20:2');
  });

  it('mixes one amix input per audio source', () => {
    expect(buildFfmpegCommand([mute])).not.toContain('amix');
    expect(buildFfmpegCommand([bleep, mute])).toContain('amix=inputs=2');
    expect(
      buildFfmpegCommand([
        bleep,
        mute,
        { ...bleep, index: 4, startMs: 10_000, endMs: 10_800, findingId: 'f_x' },
      ]),
    ).toContain('amix=inputs=3');
  });

  it('splits the video once per blur region', () => {
    expect(buildFfmpegCommand([blur])).toContain('[0:v]split[base][tmp0]');
    expect(
      buildFfmpegCommand([blur, { ...blur, index: 5, startMs: 600_000, endMs: 602_000 }]),
    ).toContain('[0:v]split=3[base][tmp0][tmp1]');
  });

  it('is generated, not stored — moving an op moves the command', () => {
    const a = buildFfmpegCommand([mute]);
    const b = buildFfmpegCommand([{ ...mute, startMs: 12_000, endMs: 15_000 }]);
    expect(a).not.toBe(b);
    expect(b).toContain('between(t,12.0,15.0)');
  });

  it('orders ops by start time regardless of EDL order', () => {
    const forward = buildFfmpegCommand([mute, bleep]);
    const reversed = buildFfmpegCommand([bleep, mute]);
    expect(forward).toBe(reversed);
  });
});

describe('fixture remediation', () => {
  it('compiles the demo EDL to the command the report renders', () => {
    const recompiled = compileRemediation(
      beforeReport.remediation.ops,
      'documentary.mp4',
      'documentary.safe.mp4',
    );
    expect(recompiled.command).toBe(beforeReport.remediation.ffmpegCommand);
  });

  it('reports honestly that the demo re-encodes, because it contains a blur', () => {
    expect(beforeReport.remediation.videoStreamCopied).toBe(false);
    expect(beforeReport.remediation.ffmpegCommand).toContain('libx264');
  });

  it('references every op back to a real finding', () => {
    const ids = new Set(beforeReport.findings.map((f) => f.id));
    for (const op of beforeReport.remediation.ops) {
      expect(ids.has(op.findingId)).toBe(true);
    }
  });
});
