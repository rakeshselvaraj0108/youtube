import { afterReport, beforeReport } from '@/data/fixture';
import { computeReadiness, SUB_SCORE_LABELS, SUB_SCORE_ORDER, readinessHex, VERDICT_META } from '@/lib/scoring';
import { formatDuration, formatRange, truncateHash } from '@/lib/time';
import { degradedAgents } from '@/lib/coverage';

/**
 * PHASE 1 SCAFFOLD.
 *
 * This is not the Command Deck — it is a foundation probe that renders the
 * computed layer so the numbers can be checked before any pixels are designed.
 * Phase 2 replaces this entire file with the panel grid.
 */
export default function App() {
  const report = beforeReport;
  const readiness = computeReadiness(report.scores.sub);
  const after = computeReadiness(afterReport.scores.sub);
  const verdict = VERDICT_META[readiness.verdict];

  return (
    <div className="h-full overflow-auto bg-void p-8">
      <div className="mx-auto flex max-w-4xl flex-col gap-gutter">
        <header className="flex items-baseline justify-between">
          <h1 className="text-h1">PREFLIGHT — Phase 1 foundation</h1>
          <span className="num text-data text-inkFaint">
            {report.video.filename} · {formatDuration(report.video.durationMs)}
          </span>
        </header>

        <section className="panel-surface p-panel">
          <div className="panel-title mb-3">Release readiness</div>
          <div className="flex items-end gap-6">
            <div
              className="num text-display"
              style={{ color: readinessHex(readiness.overall) }}
            >
              {readiness.overall}
            </div>
            <div className="flex flex-col gap-1 pb-2">
              <span
                className="text-micro uppercase"
                style={{ color: readinessHex(readiness.overall) }}
              >
                {verdict.label}
              </span>
              <span className="num text-data text-inkDim">
                weighted {readiness.weighted.toFixed(2)} · weakest {readiness.weakest} (
                {readiness.worst}) · capped {String(readiness.capped)}
              </span>
              <span className="num text-data text-inkFaint">
                after fix → {after.overall} · {VERDICT_META[after.verdict].label}
              </span>
            </div>
          </div>

          <div className="mt-4 flex flex-col gap-2">
            {SUB_SCORE_ORDER.map((key) => {
              const value = report.scores.sub[key];
              return (
                <div key={key} className="flex items-center gap-3">
                  <span className="field-label w-28">{SUB_SCORE_LABELS[key]}</span>
                  <div className="h-1 flex-1 rounded-bar bg-edge">
                    <div
                      className="h-full rounded-bar"
                      style={{ width: `${value}%`, background: readinessHex(value) }}
                    />
                  </div>
                  <span className="num w-8 text-right text-data">{value}</span>
                </div>
              );
            })}
          </div>
        </section>

        <section className="panel-surface p-panel">
          <div className="panel-title mb-3">
            Findings ({report.findings.length}) · coverage{' '}
            {Math.round(report.meta.coverage * 100)}% ·{' '}
            {degradedAgents(report.agents)
              .map((a) => a.name)
              .join(', ')}{' '}
            degraded
          </div>
          <ul className="flex flex-col gap-2">
            {report.findings.map((f) => (
              <li key={f.id} className="flex items-baseline gap-3 text-body">
                <span className="num w-32 text-data text-inkFaint">
                  {formatRange(f.startMs, f.endMs)}
                </span>
                <span className="num w-16 text-micro uppercase">{f.severity}</span>
                <span className="num w-16 text-data text-inkDim">{f.clauseId}</span>
                <span className="flex-1">{f.title}</span>
                <span className="num text-data text-inkFaint">{f.suggestedFix}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel-surface p-panel">
          <div className="panel-title mb-3">
            Generated ffmpeg command · {report.remediation.ops.length} operations ·{' '}
            {report.remediation.videoStreamCopied ? 'stream copy' : 're-encode'}
          </div>
          <pre className="overflow-x-auto whitespace-pre text-code text-inkDim">
            {report.remediation.ffmpegCommand}
          </pre>
          <div className="mt-3 num text-data text-inkFaint">
            attestation {truncateHash(report.meta.attestationHash)}
          </div>
        </section>
      </div>
    </div>
  );
}
