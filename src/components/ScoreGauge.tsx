import { Panel } from '@/components/ui';
import { useReport } from '@/store/analysis';
import { readinessHex, VERDICT_META } from '@/lib/scoring';

/**
 * Release Readiness gauge.
 *
 * 270° sweep starting at −225° (7:30) and ending at 45° (4:30), so the gap sits
 * at the bottom where the caption lives. The arc is a stroked circle rotated
 * into place — no path arithmetic, which means no rounding artefacts at the
 * endpoints when Phase 4 animates the sweep.
 */

const SIZE = 200;
const STROKE = 10;
const R = (SIZE - STROKE) / 2 - 8;
const C = 2 * Math.PI * R;
const SWEEP = 0.75; // 270° of 360°

export function ScoreGauge() {
  const report = useReport();
  const score = report.scores.overall;
  const tone = readinessHex(score);
  const verdict = VERDICT_META[report.scores.verdict];

  const fraction = Math.max(0, Math.min(100, score)) / 100;

  return (
    <Panel className="min-w-0">
      <div className="flex h-full flex-col items-center justify-center gap-1">
        <span className="text-panel uppercase text-inkDim">Release Readiness</span>

        <div className="relative w-full max-w-[200px]">
          {/* One glow. Not three. */}
          <div
            className="pointer-events-none absolute inset-0 rounded-full blur-2xl"
            style={{ background: tone, opacity: 0.18 }}
            aria-hidden="true"
          />

          <svg
            viewBox={`0 0 ${SIZE} ${SIZE}`}
            className="relative w-full"
            role="img"
            aria-label={`Release readiness ${score} out of 100 — ${verdict.label}`}
          >
            <g transform={`rotate(-225 ${SIZE / 2} ${SIZE / 2})`}>
              <circle
                cx={SIZE / 2}
                cy={SIZE / 2}
                r={R}
                fill="none"
                stroke="#1A2233"
                strokeWidth={STROKE}
                strokeLinecap="butt"
                strokeDasharray={`${C * SWEEP} ${C}`}
              />
              <circle
                cx={SIZE / 2}
                cy={SIZE / 2}
                r={R}
                fill="none"
                stroke={tone}
                strokeWidth={STROKE}
                strokeLinecap="butt"
                strokeDasharray={`${C * SWEEP * fraction} ${C}`}
              />
              {/* Leading edge — a bright 2° cap that reads as the needle. */}
              {fraction > 0.01 && (
                <circle
                  cx={SIZE / 2}
                  cy={SIZE / 2}
                  r={R}
                  fill="none"
                  stroke="#E8EDF7"
                  strokeWidth={STROKE}
                  strokeLinecap="butt"
                  strokeDasharray={`${C * 0.004} ${C}`}
                  strokeDashoffset={-(C * SWEEP * fraction - C * 0.004)}
                  opacity={0.9}
                />
              )}
            </g>

            <text
              x={SIZE / 2}
              y={SIZE / 2 + 6}
              textAnchor="middle"
              className="num"
              style={{ fontSize: 56, fontWeight: 300, fill: tone }}
            >
              {score}
            </text>
            <text
              x={SIZE / 2}
              y={SIZE / 2 + 30}
              textAnchor="middle"
              className="num"
              style={{ fontSize: 12, fill: '#8A97AE' }}
            >
              / 100
            </text>
          </svg>
        </div>

        <span className="num text-[9px] uppercase tracking-[0.1em] text-inkFaint">
          0–100 · higher is better
        </span>

        <div
          className="mt-2 flex items-center gap-2 rounded-chip border px-2.5 py-1.5"
          style={{
            color: tone,
            borderColor: `${tone}66`,
            backgroundColor: `${tone}1F`,
          }}
        >
          <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: tone }} />
          <span className="text-micro uppercase">{verdict.label}</span>
        </div>
      </div>
    </Panel>
  );
}
