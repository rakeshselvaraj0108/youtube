import { Bar, Panel } from '@/components/ui';
import { useReport } from '@/store/analysis';
import {
  CLAMP_HEADROOM,
  computeReadiness,
  readinessHex,
  SUB_SCORE_LABELS,
  SUB_SCORE_ORDER,
  WEIGHTS,
} from '@/lib/scoring';

/**
 * Five dimensions, one direction. 100 is always good — a bar that is 90% full
 * means the same thing whether it measures copyright exposure or captions.
 *
 * The footnote states the clamp out loud. Showing the scoring logic is worth
 * more than hiding it: it is the reason the headline number can be trusted.
 */
export function SubScorePanel() {
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

            return (
              <div
                key={key}
                className={`-mx-2 flex items-center gap-2.5 rounded-chip px-2 py-1.5 ${
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
