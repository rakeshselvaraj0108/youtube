import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { Bar, Panel, SeverityChip } from '@/components/ui';
import { useAnalysis, useSelectedFinding, useVisibleFindings } from '@/store/analysis';
import { severityHex } from '@/lib/scoring';
import type { FindingSort } from '@/lib/findings';
import { formatTimecode } from '@/lib/time';
import type { Finding } from '@/types/analysis';

const SORTS: { key: FindingSort; label: string }[] = [
  { key: 'severity', label: 'Severity' },
  { key: 'time', label: 'Time' },
  { key: 'confidence', label: 'Confidence' },
];

function FindingCard({ finding, selected }: { finding: Finding; selected: boolean }) {
  const select = useAnalysis((s) => s.select);
  const tone = severityHex(finding.severity);

  return (
    <li>
      <button
        type="button"
        data-finding={finding.id}
        onClick={() => select(finding.id)}
        aria-current={selected}
        className={`flex w-full gap-2.5 border-b border-edge/60 px-3 py-2.5 text-left transition-colors duration-instant ${
          selected ? 'bg-panelHi' : 'hover:bg-panelHi/50'
        }`}
        style={selected ? { boxShadow: 'var(--elev-2)' } : undefined}
      >
        <span
          className="shrink-0 self-stretch rounded-[1px]"
          style={{ width: selected ? 4 : 3, background: tone }}
        />

        <span className="flex min-w-0 flex-1 flex-col gap-1.5">
          <span className="flex items-center justify-between gap-2">
            <span className="num text-[10px] text-inkDim">
              {formatTimecode(finding.startMs)} – {formatTimecode(finding.endMs)}
            </span>
            <SeverityChip severity={finding.severity} />
          </span>

          <span className="truncate text-[13px] font-semibold text-ink">{finding.title}</span>
          <span className="line-clamp-2 text-[11px] leading-snug text-inkDim">
            {finding.description}
          </span>

          <span className="flex items-center gap-2 pt-0.5">
            <Bar value={finding.confidence} tone={tone} height={3} className="flex-1" />
            <span className="num shrink-0 text-[10px] text-inkFaint">
              {Math.round(finding.confidence * 100)}%
            </span>
          </span>
        </span>
      </button>
    </li>
  );
}

export function FindingsList() {
  const findings = useVisibleFindings();
  const selected = useSelectedFinding();
  const sortBy = useAnalysis((s) => s.sortBy);
  const setSortBy = useAnalysis((s) => s.setSortBy);
  const categoryFilter = useAnalysis((s) => s.categoryFilter);
  const setCategoryFilter = useAnalysis((s) => s.setCategoryFilter);
  const select = useAnalysis((s) => s.select);
  const listRef = useRef<HTMLUListElement>(null);

  /** Arrow keys walk the list in whatever order it is currently sorted. */
  const onKeyDown = (event: React.KeyboardEvent<HTMLUListElement>) => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    const index = findings.findIndex((f) => f.id === selected?.id);
    const next = event.key === 'ArrowDown' ? index + 1 : index - 1;
    const target = findings[Math.max(0, Math.min(findings.length - 1, next))];
    if (!target) return;
    select(target.id);
    listRef.current
      ?.querySelector<HTMLButtonElement>(`[data-finding="${target.id}"]`)
      ?.focus();
  };

  // Keep the selected card in view when selection arrives from the timeline or
  // the breakdown panel rather than from this list.
  useEffect(() => {
    if (!selected) return;
    listRef.current
      ?.querySelector(`[data-finding="${selected.id}"]`)
      ?.scrollIntoView({ block: 'nearest' });
  }, [selected]);

  return (
    <Panel
      title={`Findings (${findings.length})`}
      aside={
        <div className="flex items-center gap-1">
          {SORTS.map((s) => (
            <button
              key={s.key}
              type="button"
              onClick={() => setSortBy(s.key)}
              className={`rounded-chip px-1.5 py-1 text-[9px] uppercase tracking-[0.08em] transition-colors duration-instant ${
                sortBy === s.key ? 'bg-panelHi text-ink' : 'text-inkFaint hover:text-inkDim'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      }
      className="min-w-0"
      flush
    >
      {categoryFilter && (
        <button
          type="button"
          onClick={() => setCategoryFilter(null)}
          className="flex shrink-0 items-center justify-between gap-2 border-b border-edge bg-panelHi px-3 py-2 text-left"
        >
          <span className="truncate text-[10px] uppercase tracking-[0.08em] text-inkDim">
            filtered · {categoryFilter}
          </span>
          <X className="h-3 w-3 shrink-0 text-inkFaint" />
        </button>
      )}

      <ul
        ref={listRef}
        onKeyDown={onKeyDown}
        className="min-h-0 flex-1 overflow-y-auto focus:outline-none"
      >
        {findings.map((f) => (
          <FindingCard key={f.id} finding={f} selected={selected?.id === f.id} />
        ))}
      </ul>
    </Panel>
  );
}
