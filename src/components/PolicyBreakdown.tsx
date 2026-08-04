import {
  AlertTriangle,
  BadgeDollarSign,
  Captions,
  Copyright,
  FileText,
  Gavel,
  MessageSquareWarning,
  Mountain,
  Music4,
  Siren,
  Wine,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Chip, Panel } from '@/components/ui';
import { useAnalysis, useReport } from '@/store/analysis';
import { severityHex } from '@/lib/scoring';

const CATEGORY_ICON: Record<string, LucideIcon> = {
  Violence: Siren,
  Language: MessageSquareWarning,
  Copyright: Copyright,
  'Sensitive Events': AlertTriangle,
  'Dangerous Acts': Mountain,
  'Controversial Issues': Gavel,
  'Regulated Goods': Wine,
  Accessibility: Captions,
  Metadata: FileText,
  'Audio Delivery': Music4,
};

export function PolicyBreakdown() {
  const report = useReport();
  const categoryFilter = useAnalysis((s) => s.categoryFilter);
  const setCategoryFilter = useAnalysis((s) => s.setCategoryFilter);
  const select = useAnalysis((s) => s.select);
  const total = report.findings.length;

  /** Clicking a category filters the list and jumps to its worst finding. */
  const onPick = (category: string) => {
    const next = categoryFilter === category ? null : category;
    setCategoryFilter(next);
    if (next) {
      const first = report.findings.find((f) => f.category === next);
      if (first) select(first.id);
    }
  };

  return (
    <Panel title="Policy Breakdown" className="min-w-0" flush>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <ul className="flex flex-col">
          {report.breakdown.map((row) => {
            const Icon = CATEGORY_ICON[row.category] ?? BadgeDollarSign;
            const active = categoryFilter === row.category;
            return (
              <li key={row.category}>
                <button
                  type="button"
                  onClick={() => onPick(row.category)}
                  aria-pressed={active}
                  className={`flex w-full items-center gap-2.5 border-b border-edge/60 px-4 py-2 text-left transition-colors duration-instant ${
                    active ? 'bg-panelHi' : 'hover:bg-panelHi/50'
                  }`}
                >
                  <Icon
                    className="h-3.5 w-3.5 shrink-0"
                    strokeWidth={1.6}
                    style={{ color: active ? severityHex(row.severity) : '#4E5A70' }}
                  />
                  <span className="min-w-0 flex-1 truncate text-body text-ink">{row.category}</span>
                  <span className="num w-4 shrink-0 text-right text-data text-inkDim">
                    {row.count}
                  </span>
                  <Chip tone={severityHex(row.severity)} className="w-[68px] justify-center">
                    {row.severity}
                  </Chip>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="flex shrink-0 items-center justify-between border-t border-edge px-4 py-2.5">
        <span className="text-label uppercase text-inkFaint">Total findings</span>
        <span className="num text-data text-ink">{total}</span>
      </div>
    </Panel>
  );
}
