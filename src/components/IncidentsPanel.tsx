import { ChevronRight } from 'lucide-react';

import { Panel, SeverityChip } from '@/components/ui';
import { useAnalysis, useReport } from '@/store/analysis';
import { formatPrecise } from '@/lib/time';
import type { Incident } from '@/types/analysis';

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

function IncidentRow({ incident }: { incident: Incident }) {
  const report = useReport();
  const selectedId = useAnalysis((s) => s.selectedIncidentId);
  const selectIncident = useAnalysis((s) => s.selectIncident);
  const expandedId = useAnalysis((s) => s.expandedIncidentId);
  const toggleIncident = useAnalysis((s) => s.toggleIncident);

  const isSelected = selectedId === incident.id;
  const isExpanded = expandedId === incident.id;
  const members = report.findings.filter((f) => incident.findingIds.includes(f.id));
  const durationMs = incident.endMs - incident.startMs;
  const fileScoped = durationMs >= (report.video.durationMs || 0) * 0.9;

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
        <div className="flex flex-col gap-2 border-t border-edge px-2.5 py-2">
          <Section label="Observations">
            {incident.agents.map((agent) => (
              <Token key={agent}>{agent}</Token>
            ))}
          </Section>

          <Section label="Findings">
            {members.map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => useAnalysis.getState().select(f.id)}
                title={`${f.title} — select and seek`}
                className="num rounded-chip border border-edge px-1.5 py-0.5 text-[9px] text-inkDim hover:border-inkFaint hover:text-ink"
              >
                {f.id}
              </button>
            ))}
          </Section>

          <Section label="Policy">
            {incident.clauses.map((c) => (
              <Token key={c}>{c}</Token>
            ))}
          </Section>

          <Section label="Remediation">
            {incident.suggestedFix && incident.suggestedFix !== 'NONE' ? (
              <Token>{incident.suggestedFix}</Token>
            ) : (
              <span className="text-[10px] text-inkFaint">
                no executable remediation for this incident
              </span>
            )}
          </Section>

          <p className="border-t border-edge pt-1.5 text-[10px] leading-relaxed text-inkFaint">
            {incident.reasoning}
          </p>
        </div>
      )}
    </div>
  );
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

function Token({ children }: { children: React.ReactNode }) {
  return (
    <span className="num rounded-chip border border-edge px-1.5 py-0.5 text-[9px] text-inkDim">
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
