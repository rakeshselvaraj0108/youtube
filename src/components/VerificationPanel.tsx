import { useState } from 'react';
import { ChevronRight } from 'lucide-react';

import { LifecycleStrip } from '@/components/LifecycleStrip';
import { Panel } from '@/components/ui';
import { useAnalysis } from '@/store/analysis';
import { STATUS_LABEL, statusHex } from '@/lib/verification';
import type { IncidentChange, VerificationCertificate } from '@/types/analysis';

/**
 * VERIFICATION — what the remediation actually did, on evidence.
 *
 * The one panel on the deck that is not about the video. Everything else here
 * answers "what is wrong with this file"; this answers "what changed between
 * two files, and how much of that is known". Those are different questions
 * and the second one is the product's whole argument.
 *
 * Every number is read from the verification object the backend persisted.
 * Nothing is recomputed: a rollup on this side would eventually disagree with
 * the certificate rendered three rows below it, and a page that contradicts
 * its own certificate is worse than one that shows neither.
 *
 * Absent by design when no remediation has run. A verdict must never appear
 * because a plan exists or because ffmpeg exited zero.
 */

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number;
  hint?: string;
}) {
  return (
    <span className="flex flex-col gap-0.5" title={hint}>
      <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">{label}</span>
      <span className="num text-[12px] text-ink">{value}</span>
    </span>
  );
}

/** A status chip that never relies on colour alone. */
function StatusChip({ status }: { status: string }) {
  return (
    <span
      className="num shrink-0 rounded-chip border px-1.5 py-0.5 text-[8px] uppercase tracking-[0.06em]"
      style={{ color: statusHex(status), borderColor: `${statusHex(status)}55` }}
    >
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}

function IncidentRow({ change }: { change: IncidentChange }) {
  const selectChange = useAnalysis((s) => s.selectChange);
  const anchor =
    change.resolvedFindings[0] ??
    change.persistingFindings[0] ??
    change.newFindings[0] ??
    change.inconclusiveFindings[0] ??
    null;

  return (
    <button
      type="button"
      onClick={() => anchor && selectChange(anchor)}
      title={`${change.detail}\nclauses ${change.clauses.join(', ') || '—'}`}
      className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-chip px-2 py-1.5 text-left transition-colors duration-instant hover:bg-panelHi/60"
    >
      <span className="flex min-w-0 flex-col">
        <span className="truncate text-[11px] text-inkDim">
          {change.originalId ?? change.remediatedId} · {change.category}
        </span>
        <span className="truncate text-[10px] text-inkFaint">{change.detail}</span>
      </span>
      <StatusChip status={change.status} />
    </button>
  );
}

function CertificateBlock({ certificate }: { certificate: VerificationCertificate }) {
  const [open, setOpen] = useState(false);
  const { lineage, artifacts, scores, coverage } = certificate;

  return (
    <div className="flex flex-col gap-1.5 border-t border-edge pt-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex items-center gap-1 text-left text-[9px] uppercase tracking-[0.08em] text-inkFaint transition-colors duration-instant hover:text-ink"
      >
        <ChevronRight
          className="h-3 w-3 transition-transform duration-instant"
          style={{ transform: open ? 'rotate(90deg)' : 'none' }}
        />
        verification certificate {certificate.certificateId}
      </button>

      {open && (
        <div className="flex flex-col gap-2 pl-4">
          <div className="flex flex-wrap gap-x-5 gap-y-2">
            <Stat label="original run" value={lineage.originalRunId.slice(0, 18)} />
            <Stat label="remediation" value={lineage.remediationId} />
            <Stat
              label="verification run"
              value={lineage.verificationRunId?.slice(0, 18) ?? 'NOT MEASURED'}
            />
            <Stat label="simulation" value={lineage.simulationId ?? 'none'} />
          </div>

          <div className="flex flex-wrap gap-x-5 gap-y-2">
            <Stat
              label="original hash"
              value={String(artifacts.original.contentHash).slice(0, 14)}
              hint={String(artifacts.original.contentHash)}
            />
            <Stat
              label="remediated hash"
              value={String(artifacts.remediated.contentHash).slice(0, 14)}
              hint={String(artifacts.remediated.contentHash)}
            />
            <Stat label="original duration" value={`${artifacts.original.durationMs}`} />
            <Stat label="remediated duration" value={`${artifacts.remediated.durationMs}`} />
          </div>

          <div className="flex flex-wrap gap-x-5 gap-y-2">
            <Stat label="structural" value={certificate.verification.structural} />
            <Stat label="post-analysis" value={certificate.verification.postAnalysis} />
            <Stat label="coverage" value={`${coverage.overall}`} />
            <Stat label="prediction" value={scores.predictionOutcome} />
          </div>

          {coverage.belowFloor.length > 0 && (
            <span className="text-[10px] text-inkFaint">
              below the {Math.round(coverage.absenceFloor * 100)}% absence floor:{' '}
              {coverage.belowFloor.join(', ')} — findings missing from these are
              reported inconclusive, never resolved
            </span>
          )}

          <span
            className="num break-all text-[9px] text-inkFaint"
            title="An integrity digest over the certificate payload. Not a signature — it attests to no identity."
          >
            {certificate.certificateHash}
          </span>

          <ul className="flex flex-col gap-0.5">
            {certificate.limitations.map((line) => (
              <li key={line} className="text-[9px] leading-relaxed text-inkFaint">
                · {line}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function VerificationPanel() {
  const verification = useAnalysis((s) => s.verification);
  const certificate = useAnalysis((s) => s.certificate);
  const record = useAnalysis((s) => s.remediationRecord);
  const interrupted = useAnalysis((s) => s.interrupted);
  const running = useAnalysis((s) => s.fixRunning);

  // Nothing has been verified. Say that, rather than showing a hopeful empty
  // scaffold that reads like a pending success.
  if (!verification) {
    return (
      <Panel
        title="Verification"
        className="min-w-0"
        aside={
          running ? (
            <span className="num text-[9px] uppercase tracking-[0.06em] text-inkFaint">
              running
            </span>
          ) : null
        }
      >
        <div className="flex flex-col gap-2">
          {/* While a remediation is in flight this is the live state machine,
              not a spinner: each step is a state the engine persisted. */}
          <LifecycleStrip />
          {!running && (
            <p className="text-[11px] text-inkFaint">
              NOT VERIFIED — no remediation has been rendered and re-analysed
              for this run. A verdict appears only after the rendered file has
              been put back through the pipeline.
            </p>
          )}
          {interrupted.length > 0 && (
            <p className="text-[10px] text-inkFaint">
              {interrupted.map((r) => r.describe).join(' · ')}. Applying the fix
              again resumes it at {interrupted[0].resumesAt}.
            </p>
          )}
        </div>
      </Panel>
    );
  }

  const {
    verdict,
    originalScore,
    remediatedScore,
    predictedScore,
    resolved,
    persisting,
    new: appeared,
    inconclusive,
    incidentChanges,
  } = verification;

  return (
    <Panel
      title="Verification"
      className="min-w-0"
      aside={
        <span
          className="num rounded-chip border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.06em]"
          style={{ color: statusHex(verdict), borderColor: `${statusHex(verdict)}55` }}
        >
          {verdict.replace(/_/g, ' ')}
        </span>
      }
    >
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
        {/* The route the remediation actually took. Kept after completion so
            a reader can see it passed through structural verification and
            re-analysis rather than having to trust that it did. */}
        <LifecycleStrip />

        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-2">
          <Stat
            label="readiness"
            value={`${originalScore} → ${
              verification.reanalysisOk ? remediatedScore : 'NOT MEASURED'
            }`}
            hint="Measured on the original, then measured again on the rendered file"
          />
          <Stat
            label="predicted"
            // Null is not zero. A run with no simulation made no prediction,
            // and rendering that as 0 would be a forecast nobody made.
            value={
              predictedScore === null
                ? 'NOT COMPUTED'
                : verification.predictionIsForThisEdit === false
                  ? `${predictedScore}*`
                  : predictedScore
            }
            hint={
              verification.predictedScenario
                ? `From scenario “${verification.predictedScenario}”` +
                  (verification.predictionIsForThisEdit === false
                    ? ' — a different operation set than the one rendered, so only the scores are comparable'
                    : ' — the same operation set that was rendered')
                : 'What the simulation said before anything was rendered'
            }
          />
          <Stat label="resolved" value={resolved} />
          <Stat label="persisting" value={persisting} />
          <Stat label="new" value={appeared} />
          {inconclusive > 0 && <Stat label="inconclusive" value={inconclusive} />}
        </div>

        {appeared > 0 && (
          <p className="text-[10px] leading-relaxed text-inkFaint">
            The remediation resolved what it targeted and {appeared} finding
            {appeared === 1 ? '' : 's'} appeared that {appeared === 1 ? 'was' : 'were'} not
            in the original. A check that stopped at the render would have
            reported success.
          </p>
        )}

        {incidentChanges.length > 0 && (
          <div className="flex flex-col gap-0.5 border-t border-edge pt-1.5">
            <span className="px-2 text-[9px] uppercase tracking-[0.08em] text-inkFaint">
              incidents
            </span>
            {incidentChanges.map((change, index) => (
              <IncidentRow
                key={`${change.originalId ?? change.remediatedId}-${index}`}
                change={change}
              />
            ))}
          </div>
        )}

        {record && (
          <span className="text-[9px] uppercase tracking-[0.06em] text-inkFaint">
            {record.remediationId} · {record.state.replace(/_/g, ' ').toLowerCase()} ·{' '}
            {record.transitions.length} recorded transitions
          </span>
        )}

        {certificate && <CertificateBlock certificate={certificate} />}
      </div>
    </Panel>
  );
}
