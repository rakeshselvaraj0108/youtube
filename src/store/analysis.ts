import { create } from 'zustand';
import type { AnalysisReport, Finding } from '@/types/analysis';
import { afterReport, beforeReport } from '@/data/fixture';
import { sortFindings, type FindingSort } from '@/lib/findings';

export type DetailTab = 'EVIDENCE' | 'POLICY' | 'ADVERSARIAL';

interface AnalysisState {
  /** Which render the deck is showing. The fix transition flips this. */
  applied: boolean;
  selectedFindingId: string | null;
  categoryFilter: string | null;
  sortBy: FindingSort;
  detailTab: DetailTab;
  /** Remediation row under the cursor — highlights its pin on the timeline. */
  hoveredOpIndex: number | null;

  setApplied: (applied: boolean) => void;
  select: (id: string | null) => void;
  setCategoryFilter: (category: string | null) => void;
  setSortBy: (sort: FindingSort) => void;
  setDetailTab: (tab: DetailTab) => void;
  setHoveredOp: (index: number | null) => void;
}

export const useAnalysis = create<AnalysisState>((set) => ({
  applied: false,
  selectedFindingId: beforeReport.findings[0]?.id ?? null,
  categoryFilter: null,
  sortBy: 'severity',
  detailTab: 'EVIDENCE',
  hoveredOpIndex: null,

  setApplied: (applied) =>
    set(() => {
      const next = applied ? afterReport : beforeReport;
      return { applied, selectedFindingId: next.findings[0]?.id ?? null, categoryFilter: null };
    }),
  select: (selectedFindingId) => set({ selectedFindingId }),
  setCategoryFilter: (categoryFilter) => set({ categoryFilter }),
  setSortBy: (sortBy) => set({ sortBy }),
  setDetailTab: (detailTab) => set({ detailTab }),
  setHoveredOp: (hoveredOpIndex) => set({ hoveredOpIndex }),
}));

/* ------------------------------------------------------------------ */
/* Selectors                                                           */
/* ------------------------------------------------------------------ */

export function useReport(): AnalysisReport {
  return useAnalysis((s) => (s.applied ? afterReport : beforeReport));
}

/** Findings after the category filter and the active sort. */
export function useVisibleFindings(): Finding[] {
  const report = useReport();
  const categoryFilter = useAnalysis((s) => s.categoryFilter);
  const sortBy = useAnalysis((s) => s.sortBy);

  const filtered = categoryFilter
    ? report.findings.filter((f) => f.category === categoryFilter)
    : report.findings;
  return sortFindings(filtered, sortBy);
}

export function useSelectedFinding(): Finding | null {
  const report = useReport();
  const id = useAnalysis((s) => s.selectedFindingId);
  return report.findings.find((f) => f.id === id) ?? report.findings[0] ?? null;
}
