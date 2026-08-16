import { Panel } from '@/components/ui';
import { useAnalysis } from '@/store/analysis';
import { STATUS_LABEL, statusHex } from '@/lib/verification';
import { formatTimecode } from '@/lib/time';
import type { EvidencePair } from '@/types/analysis';

/**
 * BEFORE / REMEDIATION / AFTER — the evidence a verdict rests on.
 *
 * Three panes, and the middle one is the point: it names the operation that
 * ran between the two stills, so the pair is a claim about cause rather than
 * two pictures side by side.
 *
 * The rule the backend enforces and this panel must not undermine: an after
 * frame comes out of the rendered file or there is no after frame. Where a
 * cut removed the span there is nothing to photograph, and this says
 * EVIDENCE REMOVED BY REMEDIATION instead of showing the original still under
 * an "after" label — which would be a claim about the output made from the
 * input, and the exact thing the closed loop exists to disprove.
 */

function Still({
  pair,
  side,
}: {
  pair: EvidencePair;
  side: 'before' | 'after';
}) {
  const seekTo = useAnalysis((s) => s.seekTo);
  const setApplied = useAnalysis((s) => s.setApplied);
  const hasVerifiedAfter = useAnalysis((s) => s.hasVerifiedAfter);

  const block = side === 'before' ? pair.before : pair.after;
  const frame = block.frame;
  const ts = side === 'before' ? pair.before.tsMs : pair.after.tsMs;

  const missing =
    side === 'after' && pair.after.removedByRemediation
      ? 'EVIDENCE REMOVED BY REMEDIATION'
      : side === 'after' && !frame
        ? pair.after.unavailable || 'NOT MEASURED'
        : !frame
          ? 'NOT MEASURED — frame extraction failed'
          : null;

  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1">
      <span className="flex items-baseline justify-between text-[9px] uppercase tracking-[0.08em] text-inkFaint">
        <span>{side}</span>
        <span className="num">{ts === null ? '—' : formatTimecode(ts)}</span>
      </span>

      {missing ? (
        <div className="flex aspect-video items-center justify-center rounded-chip border border-dashed border-edge px-2 text-center">
          <span className="num text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
            {missing}
          </span>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => {
            // Seek in the timeline this still belongs to, switching the
            // player if needed. Seeking the remediated player to an original
            // timestamp would land on unrelated material after a cut.
            if (side === 'after' && hasVerifiedAfter) setApplied(true);
            if (side === 'before') setApplied(false);
            if (ts !== null) seekTo(ts);
          }}
          title={`Seek the ${side} player to ${ts === null ? '—' : formatTimecode(ts)}`}
          className="overflow-hidden rounded-chip border border-edge transition-colors duration-instant hover:border-inkFaint"
        >
          <img
            src={frame?.image}
            alt={`${side} frame at ${ts === null ? 'unknown time' : formatTimecode(ts)}`}
            className="aspect-video w-full object-cover"
          />
        </button>
      )}

      <span className="num text-[9px] text-inkFaint">
        {frame ? `${frame.source} · ${frame.runId?.slice(0, 14) ?? 'run unknown'}` : '—'}
      </span>
    </div>
  );
}

export function EvidencePanel() {
  const evidence = useAnalysis((s) => s.evidence);
  const selectedChangeId = useAnalysis((s) => s.selectedChangeId);
  const selectChange = useAnalysis((s) => s.selectChange);
  const verification = useAnalysis((s) => s.verification);

  if (!verification || evidence.length === 0) {
    return (
      <Panel title="Before / After Evidence" className="min-w-0">
        <p className="text-[11px] text-inkFaint">
          NOT MEASURED — evidence pairs are extracted from both artifacts after
          a remediation has been rendered and re-analysed.
        </p>
      </Panel>
    );
  }

  const pair =
    evidence.find((p) => p.findingId === selectedChangeId) ?? evidence[0];

  return (
    <Panel
      title="Before / After Evidence"
      className="min-w-0"
      aside={
        <span className="num text-[9px] uppercase tracking-[0.06em] text-inkFaint">
          {verification.evidence?.afterFramesExtracted ?? 0}/
          {verification.evidence?.pairs ?? evidence.length} after frames extracted
        </span>
      }
    >
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
        {/* The findings carrying the verdict, most interesting first. */}
        <div className="flex flex-wrap gap-1">
          {evidence.map((candidate) => (
            <button
              key={candidate.findingId}
              type="button"
              onClick={() => selectChange(candidate.findingId)}
              title={`${candidate.clauseId} — ${STATUS_LABEL[candidate.status] ?? candidate.status}`}
              className={`num rounded-chip border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.06em] transition-colors duration-instant ${
                candidate.findingId === pair.findingId ? 'bg-panelHi' : ''
              }`}
              style={{
                color: statusHex(candidate.status),
                borderColor: `${statusHex(candidate.status)}55`,
              }}
            >
              {candidate.clauseId} {STATUS_LABEL[candidate.status] ?? candidate.status}
            </button>
          ))}
        </div>

        <div className="flex flex-col gap-2 border-t border-edge pt-2 lg:flex-row lg:items-start">
          <Still pair={pair} side="before" />

          <div className="flex min-w-0 shrink-0 flex-col items-center justify-center gap-1 px-2 lg:w-[150px]">
            <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
              remediation
            </span>
            {pair.remediation ? (
              <>
                <span className="num text-[11px] text-ink">{pair.remediation.op}</span>
                <span className="num text-[9px] text-inkFaint">
                  {formatTimecode(pair.remediation.startMs)} → {formatTimecode(pair.remediation.endMs)}
                </span>
                <span className="num text-[9px] text-inkFaint">
                  {pair.remediation.remediationId}
                </span>
              </>
            ) : (
              <span className="num text-center text-[9px] uppercase leading-relaxed text-inkFaint">
                no operation
                <br />
                targeted this span
              </span>
            )}
          </div>

          <Still pair={pair} side="after" />
        </div>

        <div className="flex flex-col gap-1 border-t border-edge pt-2">
          <span className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="num text-[10px] text-inkDim">{pair.clauseId}</span>
            <span className="text-[10px] text-inkFaint">{pair.category}</span>
            <span
              className="num rounded-chip border px-1.5 text-[8px] uppercase tracking-[0.06em]"
              style={{
                color: statusHex(pair.status),
                borderColor: `${statusHex(pair.status)}55`,
              }}
            >
              {STATUS_LABEL[pair.status] ?? pair.status}
            </span>
            <span className="num text-[9px] text-inkFaint">
              confidence {pair.before.confidence.toFixed(2)}
            </span>
            <span className="num text-[9px] text-inkFaint">
              {/* Null coverage is "not recorded", not zero percent. */}
              coverage{' '}
              {pair.before.coverage === null
                ? 'NOT MEASURED'
                : `${Math.round(pair.before.coverage * 100)}%`}
            </span>
          </span>

          {pair.before.transcript && (
            <p className="text-[10px] leading-relaxed text-inkFaint">
              “{pair.before.transcript}”
            </p>
          )}

          {pair.notes.map((note) => (
            <span key={note} className="text-[9px] text-inkFaint">
              · {note}
            </span>
          ))}
        </div>
      </div>
    </Panel>
  );
}
