import { ChevronRight, ImageOff } from 'lucide-react';

import { Bar, Panel, SeverityChip } from '@/components/ui';
import { ReasoningChainView } from '@/components/ReasoningChainView';
import { useAnalysis, useReport } from '@/store/analysis';
import { formatPrecise, formatTimecode } from '@/lib/time';
import { severityHex } from '@/lib/scoring';
import { STATUS_LABEL, statusHex } from '@/lib/verification';
import type { Finding, Incident, RemediationOp } from '@/types/analysis';

/**
 * INCIDENTS — the investigation layer, between the timeline and the findings.
 *
 * Findings are what each agent observed. An incident is what happened: the
 * correlation layer's answer to "these four observations are one event". The
 * order on the page is deliberate — a reader should learn *what happened*
 * before *what each agent saw*, which is why this sits above the findings
 * workspace rather than beside it.
 *
 * The grouping is never computed here. `scoring/incidents.py` owns the rules
 * — temporal overlap, category compatibility, independent-agent
 * corroboration, file-scoped exclusion — and this renders that decision. A
 * second grouping implementation in the frontend would eventually disagree
 * with the backend about what an incident is, and the reader would have no
 * way to tell which was right.
 *
 * Several fields the brief asks for are deliberately absent rather than
 * invented: per-incident coverage, evidence counts and a remediation status
 * are not in the contract, and deriving something plausible-looking from
 * adjacent data would be exactly the fabrication the brief forbids.
 */

function agentTone(count: number): string {
  // Corroboration is the property worth seeing at a glance: one agent is an
  // observation, three agreeing is a finding a reviewer will act on.
  if (count >= 3) return '#7BE3A8';
  if (count === 2) return '#E3C77B';
  return '#8896B0';
}

/** The op the compiler actually wrote for this finding, if any.
 *
 * An incident's `suggestedFix` names what the lead finding's own agent
 * proposed; it is not a promise the compiler acted on it. The renderable-
 * ceiling filter in `scoring/simulation.py` and the EDL compiler both refuse
 * edits above their own cost budget, so a suggestion and a compiled op are
 * different facts and this panel must not conflate them.
 */
function opFor(ops: RemediationOp[], findingId: string): RemediationOp | undefined {
  return ops.find((op) => op.findingId === findingId);
}

function MemberRow({
  finding,
  op,
}: {
  finding: Finding;
  op: RemediationOp | undefined;
}) {
  const select = useAnalysis((s) => s.select);
  const setHoveredOp = useAnalysis((s) => s.setHoveredOp);
  const comparison = useAnalysis((s) =>
    s.verification?.changes.find(
      (c) => c.originalId === finding.id || c.remediatedId === finding.id,
    ) ?? null,
  );
  const upheld = finding.adversarial.adjudicator.verdict === 'UPHELD';

  return (
    <button
      type="button"
      onClick={() => select(finding.id)}
      onMouseEnter={() => op && setHoveredOp(op.index)}
      onMouseLeave={() => op && setHoveredOp(null)}
      title={`${finding.title}\n${finding.description}\nselect and seek to this finding`}
      className="grid w-full grid-cols-[auto_minmax(0,1fr)_50px_auto_auto] items-center gap-2 rounded-chip px-1.5 py-1 text-left transition-colors duration-instant hover:bg-panelHi/60"
    >
      <SeverityChip severity={finding.severity} />
      <span className="min-w-0 truncate text-[10px] text-inkDim">{finding.title}</span>
      <Bar value={finding.fusedConfidence ?? finding.confidence} tone={severityHex(finding.severity)} height={3} />
      <span
        className="num shrink-0 rounded-chip border px-1 text-[8px] uppercase tracking-[0.04em]"
        style={{
          color: upheld ? '#E38B7B' : '#7BE3A8',
          borderColor: upheld ? '#E38B7B55' : '#7BE3A855',
        }}
        title={
          upheld
            ? `ADJUDICATOR upheld the charge — ${finding.adversarial.adjudicator.rationale}`
            : `ADJUDICATOR dismissed the charge — ${finding.adversarial.adjudicator.rationale}`
        }
      >
        {upheld ? 'upheld' : 'dismissed'}
      </span>
      {comparison ? (
        <span
          className="num shrink-0 rounded-chip border px-1 text-[8px] uppercase tracking-[0.04em]"
          style={{ color: statusHex(comparison.status), borderColor: `${statusHex(comparison.status)}55` }}
          title={comparison.detail}
        >
          {STATUS_LABEL[comparison.status] ?? comparison.status}
        </span>
      ) : op ? (
        <span
          className="num shrink-0 rounded-chip border border-edge px-1 text-[8px] uppercase tracking-[0.04em] text-inkFaint"
          title={`Compiled into the remediation plan: ${op.op} at ${formatTimecode(op.startMs)}`}
        >
          {op.op}
        </span>
      ) : (
        <span className="w-1" />
      )}
    </button>
  );
}

/** The best still frame among an incident's members — the highest-severity
 * finding that actually carries one, so the thumbnail matches what the
 * incident is most severely about rather than whichever member sorted
 * first. */
function IncidentThumbnail({ members }: { members: Finding[] }) {
  const select = useAnalysis((s) => s.select);
  const ranked = [...members].sort(
    (a, b) => severityRank(b.severity) - severityRank(a.severity),
  );
  const withFrame = ranked.find((f) => f.evidence.frames.length > 0);

  if (!withFrame) {
    return (
      <div className="flex aspect-video w-full max-w-[160px] shrink-0 flex-col items-center justify-center gap-1 rounded-panel border border-dashed border-edge text-inkFaint">
        <ImageOff className="h-4 w-4" strokeWidth={1.5} />
        <span className="num text-[8px] uppercase tracking-[0.06em]">no visual evidence</span>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => select(withFrame.id)}
      title={`${withFrame.id} — select and seek`}
      className="aspect-video w-full max-w-[160px] shrink-0 overflow-hidden rounded-panel border border-edge transition-colors duration-instant hover:border-inkFaint"
    >
      <img
        src={withFrame.evidence.frames[0]}
        alt={`Evidence for ${withFrame.id}`}
        className="h-full w-full object-cover"
      />
    </button>
  );
}

function severityRank(severity: string): number {
  return { CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0 }[severity] ?? 0;
}

function IncidentRow({ incident }: { incident: Incident }) {
  const report = useReport();
  const selectedId = useAnalysis((s) => s.selectedIncidentId);
  const selectIncident = useAnalysis((s) => s.selectIncident);
  const expandedId = useAnalysis((s) => s.expandedIncidentId);
  const toggleIncident = useAnalysis((s) => s.toggleIncident);

  // Matched by original id, which is what the backend's comparison keys on.
  // The frontend never re-derives which incident is which: the second run
  // renumbers them, and a second matching implementation here would
  // eventually disagree with the certificate.
  const comparison = useAnalysis((s) =>
    s.verification?.incidentChanges.find((c) => c.originalId === incident.id) ?? null,
  );

  const isSelected = selectedId === incident.id;
  const isExpanded = expandedId === incident.id;
  const members = report.findings.filter((f) => incident.findingIds.includes(f.id));
  const durationMs = incident.endMs - incident.startMs;
  const fileScoped = durationMs >= (report.video.durationMs || 0) * 0.9;
  const ops = report.remediation.ops;
  // The engine's cited argument for this incident, keyed by the same id the
  // comparison and the certificate use — never re-derived here.
  const chain = (report.reasoning ?? []).find((c) => c.incidentId === incident.id);

  return (
    <div
      className={`rounded-panel border transition-colors duration-instant ${
        isSelected ? 'border-inkFaint bg-panelHi' : 'border-edge hover:border-inkFaint/60'
      }`}
    >
      <div className="flex items-center gap-2 px-2.5 py-2">
        <button
          type="button"
          onClick={() => toggleIncident(incident.id)}
          aria-label={isExpanded ? 'Collapse incident' : 'Expand incident'}
          aria-expanded={isExpanded}
          className="shrink-0 text-inkFaint transition-transform duration-instant hover:text-ink"
          style={{ transform: isExpanded ? 'rotate(90deg)' : 'none' }}
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>

        <button
          type="button"
          onClick={() => selectIncident(incident.id)}
          title={
            `${incident.id} · ${incident.category}\n` +
            `${formatPrecise(incident.startMs)} → ${formatPrecise(incident.endMs)} ` +
            `(${(durationMs / 1000).toFixed(1)}s)\n` +
            `${incident.findingIds.length} finding(s) · ${incident.agents.length} agent(s) · ` +
            `${incident.clauses.length} clause(s)\n` +
            `${incident.reasoning}` +
            (fileScoped ? '\nFile-scoped — describes the upload, not a moment in it' : '')
          }
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <span className="num shrink-0 text-[10px] text-inkFaint">{incident.id}</span>
          <SeverityChip severity={incident.severity} />
          <span className="min-w-0 flex-1 truncate text-[11px] text-inkDim">
            {incident.category}
          </span>
          <span className="num shrink-0 text-[10px] text-inkFaint">
            {fileScoped ? 'file-scoped' : formatPrecise(incident.startMs)}
          </span>
          {/* What the verification concluded about this incident, when one
              has run. A text label, never colour alone — and absent entirely
              before a remediation, rather than defaulting to a state that
              would read as a result. */}
          {comparison && (
            <span
              className="num shrink-0 rounded-chip border px-1.5 text-[8px] uppercase tracking-[0.06em]"
              style={{
                color: statusHex(comparison.status),
                borderColor: `${statusHex(comparison.status)}55`,
              }}
              title={comparison.detail}
            >
              {STATUS_LABEL[comparison.status] ?? comparison.status}
            </span>
          )}
        </button>

        <span className="flex shrink-0 items-center gap-2">
          <span
            className="num text-[10px]"
            style={{ color: agentTone(incident.agents.length) }}
            title={
              incident.corroborated
                ? `Corroborated by ${incident.agents.length} independent agents: ${incident.agents.join(', ')}`
                : `Only ${incident.agents[0] ?? 'one agent'} observed this — nothing independent corroborates it`
            }
          >
            {incident.agents.length}a
          </span>
          <span className="num text-[10px] text-inkFaint">
            {incident.findingIds.length}f
          </span>
          <span
            className="num cursor-help text-[10px] text-inkDim"
            title="Best single observation plus a bounded step per additional independent agent. Never reaches certainty."
          >
            {(incident.confidence * 100).toFixed(0)}%
          </span>
        </span>
      </div>

      {isExpanded && (
        <div className="flex flex-col gap-2.5 border-t border-edge px-2.5 py-2.5">
          <div className="flex gap-2.5">
            <IncidentThumbnail members={members} />

            <div className="flex min-w-0 flex-1 flex-col gap-2">
              <Section label="Observations">
                {incident.agents.map((agent) => (
                  <Token key={agent}>{agent}</Token>
                ))}
              </Section>

              <Section label="Policy">
                {incident.clauses.map((c) => (
                  <Token key={c} title={memberPolicy(members, c)}>
                    {c}
                  </Token>
                ))}
              </Section>

              <Section label="Remediation">
                {remediationSummary(incident, members, ops)}
              </Section>

              {comparison && (
                <Section label="Verification">
                  <span
                    className="num rounded-chip border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.04em]"
                    style={{
                      color: statusHex(comparison.status),
                      borderColor: `${statusHex(comparison.status)}55`,
                    }}
                  >
                    {STATUS_LABEL[comparison.status] ?? comparison.status}
                  </span>
                  <span className="text-[10px] text-inkFaint">{comparison.detail}</span>
                </Section>
              )}
            </div>
          </div>

          {/* Every member finding, richer than an id chip: severity,
              confidence, the adversarial verdict that actually decided
              whether it counted, and — when a fix was rendered — the
              verification outcome or the compiled operation. */}
          <div className="flex flex-col gap-0.5 border-t border-edge pt-2">
            <span className="px-1.5 text-[9px] uppercase tracking-[0.08em] text-inkFaint">
              findings ({members.length})
            </span>
            {members.map((f) => (
              <MemberRow key={f.id} finding={f} op={opFor(ops, f.id)} />
            ))}
          </div>

          {/* The cited reasoning chain — the engine's actual argument for
              this incident, not a one-line summary of it. Falls back to the
              flat summary only for a report emitted before chains existed. */}
          {chain ? (
            <div className="border-t border-edge pt-2">
              <ReasoningChainView chain={chain} compact />
            </div>
          ) : (
            <p className="border-t border-edge pt-1.5 text-[10px] leading-relaxed text-inkFaint">
              {incident.reasoning}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/** What actually happened to this incident's remediation.
 *
 * Three distinct facts, and the panel used to collapse them into one token:
 * a lead finding can *suggest* a fix that the compiler never selects — the
 * renderable-ceiling filter in `scoring/simulation.py` refuses edits above
 * its own cost budget — so "suggested" and "compiled" are different claims.
 */
function remediationSummary(
  incident: Incident,
  members: Finding[],
  ops: RemediationOp[],
): React.ReactNode {
  const compiled = members
    .map((f) => opFor(ops, f.id))
    .filter((op): op is RemediationOp => op !== undefined);

  if (compiled.length > 0) {
    return (
      <>
        {compiled.map((op) => (
          <Token key={op.index} title={op.details}>
            {op.op} @ {formatTimecode(op.startMs)}
          </Token>
        ))}
      </>
    );
  }

  if (incident.suggestedFix && incident.suggestedFix !== 'NONE') {
    return (
      <span className="text-[10px] text-inkFaint">
        <Token>{incident.suggestedFix}</Token> suggested by the lead finding, not
        selected by the compiler
      </span>
    );
  }

  return (
    <span className="text-[10px] text-inkFaint">
      no executable remediation for this incident
    </span>
  );
}

function memberPolicy(members: Finding[], clauseId: string): string {
  const owner = members.find((f) => f.policy.clauseId === clauseId);
  return owner ? `${owner.policy.title} — ${owner.policy.section}` : clauseId;
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2">
      <span className="w-[92px] shrink-0 text-[9px] uppercase tracking-[0.08em] text-inkFaint">
        {label}
      </span>
      <div className="flex min-w-0 flex-1 flex-wrap gap-1">{children}</div>
    </div>
  );
}

function Token({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className="num rounded-chip border border-edge px-1.5 py-0.5 text-[9px] text-inkDim"
    >
      {children}
    </span>
  );
}

export function IncidentsPanel() {
  const report = useReport();
  const incidents = report.incidents ?? [];
  const severityFilter = useAnalysis((s) => s.incidentSeverity);
  const setSeverityFilter = useAnalysis((s) => s.setIncidentSeverity);

  const shown = severityFilter
    ? incidents.filter((i) => i.severity === severityFilter)
    : incidents;

  const corroborated = incidents.filter((i) => i.corroborated).length;

  return (
    <Panel
      title="Incidents"
      className="min-w-0"
      aside={
        <span className="flex items-center gap-1.5">
          {(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as const).map((level) => (
            <button
              key={level}
              type="button"
              onClick={() => setSeverityFilter(severityFilter === level ? null : level)}
              title={`Show only ${level} incidents`}
              className={`num rounded-chip border px-1.5 py-0.5 text-[9px] transition-colors duration-instant ${
                severityFilter === level
                  ? 'border-inkFaint text-ink'
                  : 'border-edge text-inkFaint hover:text-inkDim'
              }`}
            >
              {level[0]}
            </button>
          ))}
        </span>
      }
    >
      {incidents.length === 0 ? (
        <p className="text-[11px] text-inkFaint">
          No incidents. Correlation runs over the findings this report carries;
          a report emitted before that existed carries none.
        </p>
      ) : (
        <div className="flex min-h-0 flex-col gap-1.5 overflow-y-auto">
          {shown.map((incident) => (
            <IncidentRow key={incident.id} incident={incident} />
          ))}
          {shown.length === 0 && (
            <p className="text-[11px] text-inkFaint">
              No {severityFilter} incidents in this run.
            </p>
          )}
          <p className="mt-1 border-t border-edge pt-2 text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
            {incidents.length} incident{incidents.length === 1 ? '' : 's'} ·{' '}
            {corroborated} corroborated by more than one agent · grouped by the
            engine, not by proximity
          </p>
        </div>
      )}
    </Panel>
  );
}
