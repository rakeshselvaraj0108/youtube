import type { RemediationOp } from '@/types/analysis';

/**
 * The remediation code generator.
 *
 * Findings lower into an Edit Decision List; this file is the back end that
 * turns that EDL into an executable ffmpeg program. Nothing here is a template
 * string with holes punched in it — the filter graph is assembled from the ops,
 * so changing a finding changes the command.
 *
 * The one optimisation worth naming out loud: if the EDL contains no video ops,
 * the video stream is never re-encoded (`-c:v copy`). Audio-only remediation on
 * a 15-minute file is a stream copy plus an audio pass, not a full transcode.
 */

const BLEEP_TONE_GAIN = 0.35;
const REPLACE_BED_GAIN = 0.9;
const BOXBLUR = 'boxblur=20:2';

export interface CompiledRemediation {
  command: string;
  /** True when the EDL is audio-only and the video stream is passed through. */
  videoStreamCopied: boolean;
  /** Extra `-i` inputs beyond the source, in input-index order. */
  auxInputs: string[];
  filterGraph: string;
}

/** Format milliseconds as seconds for an ffmpeg `enable=` expression. */
function sec(ms: number, dp = 1): string {
  return (ms / 1000).toFixed(dp);
}

function between(startMs: number, endMs: number): string {
  return `between(t,${sec(startMs)},${sec(endMs)})`;
}

const isVideoOp = (op: RemediationOp) => op.op === 'BLUR_REGION' || op.op === 'CUT';

/** Ops that silence the source audio underneath them before anything is mixed in. */
const silencesSource = (op: RemediationOp) =>
  op.op === 'MUTE' || op.op === 'BLEEP' || op.op === 'REPLACE_AUDIO';

export function compileRemediation(
  ops: RemediationOp[],
  srcPath = 'input.mp4',
  outPath = 'output.safe.mp4',
): CompiledRemediation {
  const sorted = [...ops].sort((a, b) => a.startMs - b.startMs || a.index - b.index);

  if (sorted.length === 0) {
    return {
      command: `ffmpeg -y -i ${srcPath} -c copy ${outPath}`,
      videoStreamCopied: true,
      auxInputs: [],
      filterGraph: '',
    };
  }

  const blurs = sorted.filter((o) => o.op === 'BLUR_REGION');
  const cuts = sorted.filter((o) => o.op === 'CUT');
  const bleeps = sorted.filter((o) => o.op === 'BLEEP');
  const replaces = sorted.filter((o) => o.op === 'REPLACE_AUDIO');
  const muteSpans = sorted.filter(silencesSource);

  const videoStreamCopied = !sorted.some(isVideoOp);

  /* ---------------- aux inputs ---------------- */
  // Input 0 is always the source. Bleep tones are synthesised with lavfi;
  // replacement beds are real files on disk.
  const auxInputs: string[] = [];
  const auxFlags: string[] = [];
  const inputIndex = new Map<RemediationOp, number>();

  for (const op of bleeps) {
    const durationS = ((op.endMs - op.startMs) / 1000).toFixed(3);
    const freq = op.freqHz ?? 1000;
    const spec = `sine=frequency=${freq}:duration=${durationS}`;
    inputIndex.set(op, auxInputs.length + 1);
    auxInputs.push(spec);
    auxFlags.push(`-f lavfi -i "${spec}"`);
  }
  for (const op of replaces) {
    const asset = op.asset ?? 'assets/cc_music/calm_01.mp3';
    inputIndex.set(op, auxInputs.length + 1);
    auxInputs.push(asset);
    auxFlags.push(`-i ${asset}`);
  }

  /* ---------------- video chain ---------------- */
  const chain: string[] = [];
  let videoOut = '0:v';

  if (blurs.length > 0) {
    const tmp = blurs.map((_, i) => `[tmp${i}]`).join('');
    const splitArg = blurs.length > 1 ? `split=${blurs.length + 1}` : 'split';
    chain.push(`[0:v]${splitArg}[base]${tmp}`);

    blurs.forEach((op, i) => {
      const [x, y, w, h] = op.box ?? [0.3, 0.3, 0.4, 0.4];
      chain.push(
        `[tmp${i}]crop=iw*${w}:ih*${h}:iw*${x}:ih*${y},${BOXBLUR}[bl${i}]`,
      );
    });

    let base = '[base]';
    blurs.forEach((op, i) => {
      const [x, y] = op.box ?? [0.3, 0.3, 0.4, 0.4];
      const label = i === blurs.length - 1 && cuts.length === 0 ? '[vout]' : `[v${i}]`;
      chain.push(
        `${base}[bl${i}]overlay=iw*${x}:ih*${y}:enable='${between(op.startMs, op.endMs)}'${label}`,
      );
      base = label;
    });
    videoOut = base.slice(1, -1);
  }

  // Cuts run last: every `enable=` above is evaluated against source timestamps,
  // so dropping frames afterwards keeps every other op's window correct.
  if (cuts.length > 0) {
    const keep = cuts.map((op) => `not(${between(op.startMs, op.endMs)})`).join('*');
    chain.push(`[${videoOut}]select='${keep}',setpts=N/FRAME_RATE/TB[vout]`);
    videoOut = 'vout';
  }

  /* ---------------- audio chain ---------------- */
  const audioChain: string[] = [];
  const mixLabels: string[] = [];

  if (muteSpans.length > 0) {
    const volumes = muteSpans
      .map((op) => `volume=enable='${between(op.startMs, op.endMs)}':volume=0`)
      .join(',');
    audioChain.push(`[0:a]${volumes}[a0]`);
  } else {
    audioChain.push('[0:a]anull[a0]');
  }
  mixLabels.push('[a0]');

  bleeps.forEach((op, i) => {
    const idx = inputIndex.get(op)!;
    audioChain.push(
      `[${idx}:a]adelay=${op.startMs}|${op.startMs},volume=${BLEEP_TONE_GAIN}[bp${i}]`,
    );
    mixLabels.push(`[bp${i}]`);
  });

  replaces.forEach((op, i) => {
    const idx = inputIndex.get(op)!;
    const durationS = ((op.endMs - op.startMs) / 1000).toFixed(3);
    audioChain.push(
      `[${idx}:a]atrim=0:${durationS},asetpts=PTS-STARTPTS,` +
        `adelay=${op.startMs}|${op.startMs},volume=${REPLACE_BED_GAIN}[rp${i}]`,
    );
    mixLabels.push(`[rp${i}]`);
  });

  let audioOut = 'a0';
  if (mixLabels.length > 1) {
    audioChain.push(
      `${mixLabels.join('')}amix=inputs=${mixLabels.length}:duration=first:normalize=0[aout]`,
    );
    audioOut = 'aout';
  }

  if (cuts.length > 0) {
    const keep = cuts.map((op) => `not(${between(op.startMs, op.endMs)})`).join('*');
    audioChain.push(`[${audioOut}]aselect='${keep}',asetpts=N/SR/TB[afin]`);
    audioOut = 'afin';
  }

  const filterGraph = [...chain, ...audioChain].join(';');

  /* ---------------- assembly ---------------- */
  const lines: string[] = [`ffmpeg -y -i ${srcPath}`];
  for (const flag of auxFlags) lines.push(`  ${flag}`);

  const graphLines = [...chain, ...audioChain].map((s, i, arr) =>
    i === arr.length - 1 ? s : `${s};\\`,
  );
  lines.push('  -filter_complex "\\');
  for (const g of graphLines) lines.push(g);
  lines[lines.length - 1] += '"';

  const mapVideo = videoStreamCopied ? '-map 0:v' : `-map "[${videoOut}]"`;
  lines.push(`  ${mapVideo} -map "[${audioOut}]"`);
  lines.push(
    videoStreamCopied
      ? '  -c:v copy -c:a aac -b:a 192k'
      : '  -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 192k',
  );
  lines.push(`  ${outPath}`);

  return {
    command: lines.join(' \\\n'),
    videoStreamCopied,
    auxInputs,
    filterGraph,
  };
}

/** Convenience wrapper — the signature the report pipeline calls. */
export function buildFfmpegCommand(
  ops: RemediationOp[],
  srcPath = 'input.mp4',
  outPath = 'output.safe.mp4',
): string {
  return compileRemediation(ops, srcPath, outPath).command;
}

/** Human labels for the remediation table's ACTION column. */
export const OP_LABELS: Readonly<Record<RemediationOp['op'], string>> = Object.freeze({
  MUTE: 'Mute Audio',
  BLEEP: 'Bleep Audio',
  BLUR_REGION: 'Blur Region',
  REPLACE_AUDIO: 'Replace Audio',
  CUT: 'Cut Segment',
});
