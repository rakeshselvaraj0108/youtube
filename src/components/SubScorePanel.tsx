import { Bar, Panel } from '@/components/ui';
import { useAnalysis, useReport } from '@/store/analysis';
import {
  CLAMP_HEADROOM,
  computeReadiness,
  readinessHex,
  SUB_SCORE_LABELS,
  SUB_SCORE_ORDER,
  WEIGHTS,
} from '@/lib/scoring';
import type { SubScores } from '@/types/analysis';

/**
 * Five dimensions, one direction. 100 is always good — a bar that is 90% full
 * means the same thing whether it measures copyright exposure or captions.
 *
 * The footnote states the clamp out loud. Showing the scoring logic is worth
 * more than hiding it: it is the reason the headline number can be trusted.
 */
/**
 * Clause prefix to scoring dimension. Mirrors `readiness.CLAUSE_DIMENSION`
 * on the Python side; the two have drifted apart twice in this project, so
 * anything reading it here is a view of that table rather than a rival to
 * it — used only to count which findings a dimension covers.
 */
const DIMENSION_OF: Record<string, keyof SubScores> = {
  AF: 'policy',
  COPY: 'copyright',
  CID: 'copyright',
  META: 'metadata',
  DISC: 'metadata',
  ACC: 'accessibility',
  VID: 'accessibility',
  AUD: 'audio',
};

export function SubScorePanel() {
  const setCategoryFilter = useAnalysis((s) => s.setCategoryFilter);
  const findings = useReport().findings;

  // Clicking a dimension filters the findings list to the categories that
  // score against it — the "show me what made this number" move.
  function onPick(key: keyof SubScores) {
    const categories = findings
      .filter((f) => DIMENSION_OF[f.clauseId.split('-')[0]] === key)
      .map((f) => f.category);
    setCategoryFilter(categories[0] ?? null);
  }

  const report = useReport();
  const readiness = computeReadiness(report.scores.sub);

  return (
    <Panel title="Dimensions" className="min-w-0">
      <div className="flex h-full flex-col justify-between gap-3">
        <div className="flex flex-col gap-2.5">
          {SUB_SCORE_ORDER.map((key) => {
            const value = report.scores.sub[key];
            const tone = readinessHex(value);
            const isWeakest = key === readiness.weakest;

            const contribution = (value * WEIGHTS[key]).toFixed(1);
            const counted = report.findings.filter(
              (f) => DIMENSION_OF[f.clauseId.split('-')[0]] === key,
            );

            return (
              <div
                key={key}
                role="button"
                tabIndex={0}
                onClick={() => onPick(key)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onPick(key);
                  }
                }}
                // The calculation, in the place the number is read. A weight
                // shown as "×0.40" beside a score invites the question this
                // answers, and the answer is arithmetic the reader can check.
                title={
                  `${SUB_SCORE_LABELS[key]}: ${value} × ${WEIGHTS[key].toFixed(2)} = ` +
                  `${contribution} of the weighted mean\n` +
                  `${counted.length} finding(s) score against this dimension` +
                  (isWeakest
                    ? '\nWeakest dimension — the overall is clamped to this + 15'
                    : '') +
                  '\nClick to filter the findings list to it'
                }
                className={`-mx-2 flex cursor-pointer items-center gap-2.5 rounded-chip px-2 py-1.5 transition-colors duration-instant hover:bg-panelHi ${
                  isWeakest ? 'bg-panelHi' : ''
                }`}
              >
                <span className="flex w-[92px] shrink-0 items-center gap-1.5">
                  <span className="truncate text-label uppercase text-inkDim">
                    {SUB_SCORE_LABELS[key]}
                  </span>
                </span>

                <Bar value={value / 100} tone={tone} className="flex-1" />

                <span className="num w-7 shrink-0 text-right text-data" style={{ color: tone }}>
                  {value}
                </span>

                <span className="num w-9 shrink-0 text-right text-[9px] text-inkFaint">
                  ×{WEIGHTS[key].toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>

        <div className="flex flex-col gap-2 border-t border-edge pt-3">
          {readiness.capped && (
            <div className="num flex items-baseline justify-between text-[10px]">
              <span className="text-inkFaint">weighted mean</span>
              <span className="text-inkDim">
                {readiness.weighted.toFixed(1)}{' '}
                <span className="text-inkFaint">→</span> {readiness.overall}
              </span>
            </div>
          )}
          <p className="text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
            overall capped at weakest + {CLAMP_HEADROOM}
            <br />
            one fatal flaw is never averaged away
          </p>
        </div>
      </div>
    </Panel>
  );
}
