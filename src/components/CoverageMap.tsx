import { Panel } from '@/components/ui';
import { useAnalysis, useReport } from '@/store/analysis';
import { SIGNAL_HEX } from '@/lib/scoring';
import type { CoverageBand } from '@/types/analysis';

/**
 * COVERAGE MAP — which minutes were actually examined, and by what.
 *
 * The readiness score answers "how risky is this video". This answers the
 * question underneath it: *how much of the video does that number even
 * describe?* A run can report 62% coverage while never once looking at
 * minutes nine through twelve, and every "nothing found" over that stretch
 * would be an artefact of the audit rather than a fact about the video.
 *
 * So UNEXAMINED is rendered as a hole, not as a pass. It is deliberately the
 * most visually distinct state here — hatched, labelled, and never green —
 * because the failure this panel exists to prevent is a reader glancing at a
 * mostly-filled row and concluding the video is mostly clean.
 *
 * Every cell seeks the player. A coverage report you cannot navigate is a
 * table; this is meant to be an instrument.
 */

const STATE_TONE: Record<string, string> = {
  EXAMINED: SIGNAL_HEX.clear,
  THIN: SIGNAL_HEX.medium,
  UNEXAMINED: SIGNAL_HEX.critical,
};

const STATE_LABEL: Record<string, string> = {
  EXAMINED: 'examined',
  THIN: 'thin',
  UNEXAMINED: 'not examined',
};

function Cell({
  band,
  modality,
  onSeek,
}: {
  band: CoverageBand;
  modality: string;
  onSeek: (ms: number) => void;
}) {
  const state = band.states[modality] ?? 'UNEXAMINED';
  const count = band.samples[modality] ?? 0;
  const tone = STATE_TONE[state] ?? '#4E5A70';

  return (
    <button
      type="button"
      onClick={() => onSeek(band.startMs)}
      title={
        `${band.label} · ${modality}\n` +
        `${STATE_LABEL[state] ?? state} — ${count} sample${count === 1 ? '' : 's'}\n` +
        (state === 'UNEXAMINED'
          ? 'Nothing looked here. This is a hole in the audit, not a pass.'
          : state === 'THIN'
            ? 'Sampled too thinly to support an absence claim.'
            : 'Absence of findings here is supported by evidence.') +
        '\nClick to seek the player.'
      }
      className="group relative h-6 flex-1 rounded-[2px] border transition-colors duration-instant"
      style={{
        borderColor: `${tone}66`,
        // Unexamined is hatched rather than merely a different fill, so it
        // reads as a gap even in a screenshot or to a colour-blind reader.
        background:
          state === 'UNEXAMINED'
            ? `repeating-linear-gradient(45deg, ${tone}22 0 4px, transparent 4px 8px)`
            : state === 'THIN'
              ? `${tone}33`
              : `${tone}55`,
      }}
    >
      <span className="num absolute inset-0 flex items-center justify-center text-[8px] text-ink/70 opacity-0 transition-opacity duration-instant group-hover:opacity-100">
        {count}
      </span>
    </button>
  );
}

export function CoverageMap() {
  const report = useReport();
  const seekTo = useAnalysis((s) => s.seekTo);
  const coverage = report.meta.temporalCoverage;

  if (!coverage || coverage.bands.length === 0) {
    return (
      <Panel title="Coverage Map" className="min-w-0">
        <p className="text-[11px] text-inkFaint">
          NOT MEASURED — this report carries no per-minute coverage. It was
          emitted before the engine recorded which minutes each modality
          examined.
        </p>
      </Panel>
    );
  }

  const { bands, modalities, shareExamined, blindSpots } = coverage;
  const totalBlind = new Set(
    Object.values(blindSpots ?? {}).flat(),
  ).size;

  return (
    <Panel
      title="Coverage Map"
      className="min-w-0"
      aside={
        <span className="num text-[9px] uppercase tracking-[0.06em] text-inkFaint">
          {bands.length} min · {modalities.length} modalities
        </span>
      }
    >
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
        {/* Minute ruler */}
        <div className="flex items-center gap-1 pl-[86px]">
          {bands.map((b) => (
            <span
              key={b.index}
              className="num flex-1 text-center text-[8px] text-inkFaint"
            >
              {b.index}
            </span>
          ))}
        </div>

        {modalities.map((m) => (
          <div key={m} className="flex items-center gap-1">
            <span className="w-[80px] shrink-0 truncate text-[10px] text-inkDim" title={m}>
              {m}
            </span>
            <div className="flex flex-1 items-center gap-1">
              {bands.map((b) => (
                <Cell key={b.index} band={b} modality={m} onSeek={seekTo} />
              ))}
            </div>
            <span className="num w-[38px] shrink-0 text-right text-[10px] text-inkFaint">
              {Math.round((shareExamined?.[m] ?? 0) * 100)}%
            </span>
          </div>
        ))}

        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-edge pt-2">
          {(['EXAMINED', 'THIN', 'UNEXAMINED'] as const).map((state) => (
            <span key={state} className="flex items-center gap-1.5">
              <span
                className="h-2.5 w-4 rounded-[2px] border"
                style={{
                  borderColor: `${STATE_TONE[state]}66`,
                  background:
                    state === 'UNEXAMINED'
                      ? `repeating-linear-gradient(45deg, ${STATE_TONE[state]}22 0 4px, transparent 4px 8px)`
                      : state === 'THIN'
                        ? `${STATE_TONE[state]}33`
                        : `${STATE_TONE[state]}55`,
                }}
              />
              <span className="text-[9px] uppercase tracking-[0.06em] text-inkFaint">
                {STATE_LABEL[state]}
              </span>
            </span>
          ))}
        </div>

        {totalBlind > 0 ? (
          <p className="text-[10px] leading-relaxed" style={{ color: SIGNAL_HEX.critical }}>
            {totalBlind} minute{totalBlind === 1 ? '' : 's'} were not examined by at
            least one modality. Findings are absent there because nothing
            looked — not because the video is clean.
          </p>
        ) : (
          <p className="text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
            every minute examined by every modality that ran · click any cell to
            seek
          </p>
        )}
      </div>
    </Panel>
  );
}
