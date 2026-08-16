import { useEffect, useState } from 'react';

import { useAnalysis } from '@/store/analysis';
import { SIGNAL_HEX } from '@/lib/scoring';
import { LIFECYCLE_STEPS, stepStatus } from '@/lib/verification';

/**
 * The remediation lifecycle, while it happens.
 *
 * A remediation takes minutes, and for all of them the deck previously showed
 * one word. Minutes of a single word is indistinguishable from a hang, and
 * the thing a reader most wants to know during the wait — *has it got past
 * the render yet* — was exactly what it would not say.
 *
 * Every step here is a real persisted state. The backend sends the state it
 * just wrote to the transition table, so this strip and the stored history
 * are the same account of the run; nothing is inferred from a stage word on
 * this side. That matters because the alternative — mapping progress labels
 * to states in the frontend — produces a second, unbacked narrative that
 * drifts the first time either end is edited.
 *
 * The elapsed clock is wall time since the request. It is measured, not
 * estimated, and there is deliberately no progress percentage or ETA: the
 * engine does not know how long re-analysis will take, and a bar that fills
 * at an invented rate is the most convincing lie a UI can tell.
 */

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const seconds = Math.max(0, Math.round((now - since) / 1000));
  const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
  const ss = String(seconds % 60).padStart(2, '0');
  return <span className="num text-[10px] text-inkFaint">{mm}:{ss}</span>;
}

export function LifecycleStrip() {
  const running = useAnalysis((s) => s.fixRunning);
  const current = useAnalysis((s) => s.fixState);
  const seen = useAnalysis((s) => s.fixSeen);
  const detail = useAnalysis((s) => s.fixDetail);
  const stage = useAnalysis((s) => s.fixStage);
  const error = useAnalysis((s) => s.fixError);
  const startedAt = useAnalysis((s) => s.fixStartedAt);
  const record = useAnalysis((s) => s.remediationRecord);

  // After a completed run the persisted record is authoritative; during one,
  // the live events are all that exist yet.
  const finalState = record?.state ?? null;
  const active = running ? current : finalState;

  if (!running && !finalState && !error) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
          {running ? 'remediation in progress' : 'remediation lifecycle'}
        </span>
        <span className="flex items-baseline gap-2">
          {record && (
            <span className="num text-[10px] text-inkDim">{record.remediationId}</span>
          )}
          {startedAt !== null && running && <Elapsed since={startedAt} />}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-x-1 gap-y-1">
        {LIFECYCLE_STEPS.map((step, index) => {
          const status = stepStatus(step.state, active, seen, record);
          const tone =
            status === 'done'
              ? SIGNAL_HEX.clear
              : status === 'active'
                ? '#E3C77B'
                : status === 'failed'
                  ? SIGNAL_HEX.critical
                  : '#58647A';
          return (
            <span key={step.state} className="flex items-center gap-1">
              <span
                title={`${step.state} — ${step.hint}`}
                className="num rounded-chip border px-1.5 py-0.5 text-[8px] uppercase tracking-[0.06em] transition-colors duration-fast"
                style={{
                  color: status === 'pending' ? '#58647A' : tone,
                  borderColor: status === 'pending' ? '#26324A' : `${tone}66`,
                  background: status === 'active' ? `${tone}18` : 'transparent',
                }}
              >
                {step.label}
              </span>
              {index < LIFECYCLE_STEPS.length - 1 && (
                <span className="text-[8px] text-inkFaint/50">→</span>
              )}
            </span>
          );
        })}
      </div>

      <span className="text-[10px] text-inkFaint">
        {error
          ? `FAILED — ${error}`
          : running
            ? `${stage ?? ''}${detail ? ` · ${detail}` : ''}`
            : record
              ? record.stateDetail
              : ''}
      </span>
    </div>
  );
}
