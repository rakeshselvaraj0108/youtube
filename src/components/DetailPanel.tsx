import { useState } from 'react';
import { ArrowRight, ImageOff } from 'lucide-react';
import { Bar, Panel, SeverityChip } from '@/components/ui';
import { useAnalysis, useReport, useSelectedFinding } from '@/store/analysis';
import type { DetailTab } from '@/store/analysis';
import { severityHex, SIGNAL_HEX } from '@/lib/scoring';
import { AGENT_HEX } from '@/lib/agents';
import { formatPrecise } from '@/lib/time';
import type { Finding } from '@/types/analysis';

const TABS: { key: DetailTab; label: string }[] = [
  { key: 'EVIDENCE', label: 'Evidence' },
  { key: 'POLICY', label: 'Policy' },
  { key: 'ADVERSARIAL', label: 'Adversarial Record' },
];

/* ------------------------------------------------------------------ */
/* Evidence                                                            */
/* ------------------------------------------------------------------ */

function FrameStrip({ frames, onSeek }: { frames: string[]; onSeek: () => void }) {
  const [index, setIndex] = useState(0);
  const [failed, setFailed] = useState<Record<number, boolean>>({});

  if (frames.length === 0) {
    return (
      <div className="flex h-full min-h-[96px] items-center justify-center rounded-panel border border-edge bg-abyss">
        <span className="num text-[9px] uppercase tracking-[0.1em] text-inkFaint">
          no visual evidence
        </span>
      </div>
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="relative aspect-video w-full overflow-hidden rounded-panel border border-edge bg-abyss">
        {failed[index] ? (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 text-inkFaint">
            <ImageOff className="h-4 w-4" strokeWidth={1.5} />
            <span className="num text-[9px] uppercase tracking-[0.1em]">frame unavailable</span>
          </div>
        ) : (
          <img
            src={frames[index]}
            alt={`Keyframe ${index + 1} of ${frames.length}`}
            onClick={onSeek}
            title="Play from this frame"
            className="h-full w-full cursor-pointer object-cover"
            onError={() => setFailed((f) => ({ ...f, [index]: true }))}
          />
        )}
      </div>

      {frames.length > 1 && (
        <div className="flex items-center justify-center gap-1.5">
          {frames.map((frame, i) => (
            <button
              key={frame}
              type="button"
              onClick={() => setIndex(i)}
              aria-label={`Frame ${i + 1}`}
              className="h-1.5 w-1.5 rounded-full transition-colors duration-instant"
              style={{ background: i === index ? '#E8EDF7' : '#26324A' }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function EvidenceTab({ finding }: { finding: Finding }) {
  const [start, end] = finding.evidence.highlightSpan;
  const text = finding.evidence.transcript;
  const hasHighlight = end > start;
  const tone = severityHex(finding.severity);
  const seekTo = useAnalysis((s) => s.seekTo);

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-y-auto lg:grid-cols-[1.3fr_1fr]">
      <div className="flex min-w-0 flex-col gap-2">
        <span className="text-label uppercase text-inkFaint">Transcript / Evidence</span>
        {/* The transcript is a seek control. A reader looking at the words
            wants to hear them, and asking them to find the same moment on a
            scrub bar is the step that makes a report feel like a document
            rather than a workspace. */}
        <p
          role="button"
          tabIndex={0}
          onClick={() => seekTo(finding.startMs)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              seekTo(finding.startMs);
            }
          }}
          title="Play from here"
          className="num cursor-pointer rounded-panel border border-edge bg-abyss p-3 text-code leading-relaxed text-inkDim transition-colors duration-instant hover:border-inkFaint">
          {hasHighlight ? (
            <>
              {text.slice(0, start)}
              <mark
                className="rounded-[2px] px-0.5"
                style={{
                  background: `${tone}26`,
                  color: '#E8EDF7',
                  borderBottom: `1px solid ${tone}`,
                }}
              >
                {text.slice(start, end)}
              </mark>
              {text.slice(end)}
            </>
          ) : (
            text
          )}
        </p>
        <span className="num text-[10px] text-inkFaint">
          [{formatPrecise(finding.startMs)}]
        </span>

        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1.5 border-t border-edge pt-2.5">
          {Object.entries(finding.modalities).map(([modality, value]) => (
            <span key={modality} className="flex items-center gap-1.5">
              <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
                {modality}
              </span>
              <span className="num text-[10px] text-inkDim">{value.toFixed(2)}</span>
            </span>
          ))}
          <span
            className="flex cursor-help items-center gap-1.5"
            // The fused number is the one a reader is asked to trust, so the
            // calculation behind it belongs where it is read rather than in
            // documentation nobody opens.
            title={
              'Noisy-or over the per-modality confidences, each scaled by ' +
              "that agent's actual coverage.\n" +
              Object.entries(finding.modalities)
                .map(([m, v]) => `  ${m}: ${v.toFixed(2)}`)
                .join('\n') +
              `\n  fused: ${finding.fusedConfidence.toFixed(2)}\n` +
              'Independent agents agreeing raises it; one agent repeating ' +
              'itself does not.'
            }
          >
            <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">fused</span>
            <span className="num text-[10px] text-ink">{finding.fusedConfidence.toFixed(2)}</span>
          </span>
        </div>
      </div>

      <div className="flex min-w-0 flex-col gap-2">
        <span className="text-label uppercase text-inkFaint">Visual Evidence</span>
        <FrameStrip
          frames={finding.evidence.frames}
          onSeek={() => seekTo(finding.startMs)}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Policy                                                              */
/* ------------------------------------------------------------------ */

/**
 * Every other finding judged under the same clause.
 *
 * Replaces a "View full clause" button that had no handler. The clause text
 * is already printed directly above it, so the useful move is not showing
 * that text again — it is showing what else in this video was judged the
 * same way, which is the question a reader has once they have read it.
 * Clicking one selects it, which seeks the video to that moment.
 */
function ClauseSiblings({ clauseId }: { clauseId: string }) {
  const report = useReport();
  const selectedId = useAnalysis((s) => s.selectedFindingId);
  const select = useAnalysis((s) => s.select);

  const siblings = report.findings.filter(
    (f) => f.policy.clauseId === clauseId && f.id !== selectedId,
  );

  if (siblings.length === 0) {
    return (
      <p className="text-[11px] text-inkFaint">
        No other finding in this video was judged under {clauseId}.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
        {siblings.length} other finding{siblings.length === 1 ? '' : 's'} under {clauseId}
      </span>
      {siblings.map((f) => (
        <button
          key={f.id}
          type="button"
          onClick={() => select(f.id)}
          title="Select and seek the video to this moment"
          className="flex w-full items-center gap-2 rounded border border-edge px-2 py-1 text-left text-[11px] text-inkDim transition-colors duration-instant hover:border-inkFaint hover:text-ink"
        >
          <span className="num text-[10px] text-inkFaint">
            {formatPrecise(f.startMs)}
          </span>
          <span className="min-w-0 flex-1 truncate">{f.title}</span>
          <ArrowRight className="h-3 w-3 shrink-0" />
        </button>
      ))}
    </div>
  );
}

function PolicyTab({ finding }: { finding: Finding }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
      <div className="flex items-baseline gap-2">
        <span className="num rounded-chip border border-edge bg-panelHi px-1.5 py-1 text-[10px] text-ink">
          {finding.policy.clauseId}
        </span>
        <span className="text-[13px] font-semibold text-ink">{finding.policy.title}</span>
      </div>

      <span className="num text-[10px] text-inkFaint">{finding.policy.section}</span>

      <blockquote className="rounded-panel border border-edge bg-abyss p-3 text-body leading-relaxed text-inkDim">
        {finding.policy.text}
      </blockquote>

      <ClauseSiblings clauseId={finding.policy.clauseId} />

      <p className="mt-auto border-t border-edge pt-2.5 text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
        retrieved by RRF fusion of dense + BM25 over the policy corpus · every finding cites the
        clause it was judged against
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Adversarial record                                                  */
/* ------------------------------------------------------------------ */

function Column({
  name,
  role,
  tone,
  body,
  strength,
}: {
  name: string;
  role: string;
  tone: string;
  body: string;
  strength?: number;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-2 rounded-panel border border-edge bg-abyss p-3">
      <div className="-mx-3 -mt-3 mb-1 h-0.5" style={{ background: tone }} />
      <div className="flex flex-col gap-0.5">
        <span className="text-[11px] font-semibold" style={{ color: tone }}>
          {name}
        </span>
        <span className="text-[9px] uppercase tracking-[0.1em] text-inkFaint">{role}</span>
      </div>
      <p className="flex-1 text-[11px] leading-relaxed text-inkDim">{body}</p>
      {strength !== undefined && (
        <div className="flex items-center gap-2 border-t border-edge pt-2">
          <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">strength</span>
          <Bar value={strength} tone={tone} height={3} className="flex-1" />
          <span className="num text-[10px] text-inkDim">{strength.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

function AdversarialTab({ finding }: { finding: Finding }) {
  const { auditor, advocate, adjudicator } = finding.adversarial;
  const upheld = adjudicator.verdict === 'UPHELD';
  const verdictTone = upheld ? SIGNAL_HEX.critical : SIGNAL_HEX.clear;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
      <div className="grid grid-cols-1 gap-2.5 md:grid-cols-3">
        <Column
          name="AUDITOR"
          role="Prosecute"
          tone={SIGNAL_HEX.high}
          body={auditor.charge}
        />
        <Column
          name="ADVOCATE"
          role="Defend"
          tone={AGENT_HEX.speech}
          body={
            advocate.defense ??
            'No defence available. The clause text offers no exemption that covers this evidence.'
          }
          strength={advocate.strength}
        />
        <Column
          name="ADJUDICATOR"
          role="Rule"
          tone={SIGNAL_HEX.clear}
          body={adjudicator.rationale}
        />
      </div>

      <div
        className="flex items-center justify-between gap-3 rounded-panel border px-3 py-2.5"
        style={{ borderColor: `${verdictTone}66`, background: `${verdictTone}14` }}
      >
        <span className="text-micro uppercase" style={{ color: verdictTone }}>
          Decision · Violation {adjudicator.verdict}
        </span>
        <span className="flex items-center gap-2">
          <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">confidence</span>
          <span className="num text-data" style={{ color: verdictTone }}>
            {adjudicator.confidence.toFixed(2)}
          </span>
        </span>
      </div>

      <p className="text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
        three passes, not one · the advocate exists so the linter does not cry wolf
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Panel                                                               */
/* ------------------------------------------------------------------ */

export function DetailPanel() {
  const finding = useSelectedFinding();
  const tab = useAnalysis((s) => s.detailTab);
  const setTab = useAnalysis((s) => s.setDetailTab);

  if (!finding) {
    return (
      <Panel title="Detail" className="min-w-0">
        <div className="flex flex-1 items-center justify-center">
          <span className="num text-[10px] uppercase tracking-[0.1em] text-inkFaint">
            no finding selected
          </span>
        </div>
      </Panel>
    );
  }

  const tone = severityHex(finding.severity);

  return (
    <Panel className="min-w-0" flush>
      <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-edge px-4 py-2.5">
        <span className="num text-data text-ink">
          {formatPrecise(finding.startMs)} – {formatPrecise(finding.endMs)}
        </span>
        <SeverityChip severity={finding.severity} />
        <span className="ml-auto flex items-center gap-2">
          <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">confidence</span>
          <Bar value={finding.confidence} tone={tone} height={3} className="w-16" />
          <span className="num text-data" style={{ color: tone }}>
            {Math.round(finding.confidence * 100)}%
          </span>
        </span>
      </header>

      <nav className="flex shrink-0 gap-1 border-b border-edge px-2">
        {TABS.map((t) => {
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              className={`relative px-2.5 py-2 text-[10px] uppercase tracking-[0.1em] transition-colors duration-instant ${
                active ? 'text-ink' : 'text-inkFaint hover:text-inkDim'
              }`}
            >
              {t.label}
              {active && (
                <span
                  className="absolute inset-x-2 -bottom-px h-px"
                  style={{ background: '#E8EDF7' }}
                />
              )}
            </button>
          );
        })}
      </nav>

      <div className="flex min-h-0 flex-1 flex-col p-4">
        {tab === 'EVIDENCE' && <EvidenceTab finding={finding} />}
        {tab === 'POLICY' && <PolicyTab finding={finding} />}
        {tab === 'ADVERSARIAL' && <AdversarialTab finding={finding} />}
      </div>
    </Panel>
  );
}
