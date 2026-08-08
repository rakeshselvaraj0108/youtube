import { create } from 'zustand';
import type { AnalysisReport, Finding } from '@/types/analysis';
import { afterReport as fixtureAfter, beforeReport as fixtureBefore } from '@/data/fixture';
import { fetchLatestRun, type StageEvent } from '@/lib/api';
import { sortFindings, type FindingSort } from '@/lib/findings';
import { injectedAfterReport, injectedReport } from '@/lib/reportSource';

/**
 * The report the deck renders. Real CLI output when report.html injected it,
 * the demo fixture otherwise. Resolved once at module load so every selector
 * agrees on which report it is reading.
 */
export const BEFORE: AnalysisReport = injectedReport() ?? fixtureBefore;
export const AFTER: AnalysisReport = injectedAfterReport() ?? fixtureAfter;

export type DetailTab = 'EVIDENCE' | 'POLICY' | 'ADVERSARIAL' | 'REASONING';

/**
 * Where the report on screen came from. Shown in the header, because a deck
 * rendering demo data and a deck rendering a real run must never be
 * mistakable for one another — that confusion is how a fixture ends up in a
 * screenshot presented as a result.
 */
export type ReportSource = 'injected' | 'api' | 'fixture';

/** One agent's state during a live run. */
export interface LiveStage {
  stage: string;
  name: string;
  status: 'RUNNING' | 'OK' | 'DEGRADED' | 'SKIPPED' | 'FAILED';
  coverage?: number;
  elapsedMs?: number;
  findings?: number;
  detail?: string;
}

interface AnalysisState {
  before: AnalysisReport;
  after: AnalysisReport;
  source: ReportSource;
  /** True while the API is being polled at startup. */
  loading: boolean;

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
  /**
   * Per-agent state while a run is in flight, keyed by pipeline stage id.
   * Empty when nothing is running, so the deck falls back to the report's
   * own agent rows.
   */
  live: Record<string, LiveStage>;
  /** True between the first event and the terminal one. */
  running: boolean;

  applyEvent: (event: StageEvent) => void;
  resetLive: () => void;

  /**
   * Playhead position, in milliseconds.
   *
   * Lifted out of the player because it was the thing stopping the deck
   * being an investigation surface: while it lived in local state, no other
   * panel could move the video, so clicking a finding could highlight it
   * and nothing else. Every panel that knows a timestamp can now seek.
   */
  playheadMs: number;
  /** Which clause the reader is following, if any. Filters the findings
   * list to the incidents citing it. */
  clauseFilter: string | null;

  seekTo: (ms: number) => void;
  setClauseFilter: (clause: string | null) => void;

  /** Adopt a report the engine produced. */
  setReport: (report: AnalysisReport, source: ReportSource) => void;
  /** Ask the API for its newest run. No-op when nothing answers. */
  hydrate: () => Promise<void>;
}

export const useAnalysis = create<AnalysisState>((set, get) => ({
  before: BEFORE,
  after: AFTER,
  source: injectedReport() ? 'injected' : 'fixture',
  loading: false,

  applied: false,
  selectedFindingId: BEFORE.findings[0]?.id ?? null,
  categoryFilter: null,
  sortBy: 'severity',
  detailTab: 'EVIDENCE',
  hoveredOpIndex: null,

  setApplied: (applied) =>
    set((state) => {
      const next = applied ? state.after : state.before;
      return { applied, selectedFindingId: next.findings[0]?.id ?? null, categoryFilter: null };
    }),
  // Selecting a finding moves the playhead to it. This is the whole
  // investigative loop in one line: every panel keyed on the selection
  // highlights, and the video arrives at the moment being discussed
  // instead of leaving the reader to scrub for it.
  select: (selectedFindingId) =>
    set((state) => {
      const report = state.applied ? state.after : state.before;
      const finding = report.findings.find((f) => f.id === selectedFindingId);
      const duration = report.video.durationMs || 0;
      if (!finding) return { selectedFindingId };
      // A file-scoped finding has no moment to seek to — jumping to 0 would
      // claim the problem is at the start, which is a different and false
      // statement about the video.
      const scoped = duration > 0 && finding.endMs - finding.startMs >= duration * 0.9;
      return scoped
        ? { selectedFindingId }
        : { selectedFindingId, playheadMs: finding.startMs };
    }),
  setCategoryFilter: (categoryFilter) => set({ categoryFilter }),
  setSortBy: (sortBy) => set({ sortBy }),
  setDetailTab: (detailTab) => set({ detailTab }),
  setHoveredOp: (hoveredOpIndex) => set({ hoveredOpIndex }),

  live: {},
  running: false,
  playheadMs: 0,
  clauseFilter: null,

  seekTo: (ms) => set({ playheadMs: Math.max(0, Math.round(ms)) }),
  setClauseFilter: (clauseFilter) => set({ clauseFilter }),

  resetLive: () => set({ live: {}, running: false }),

  applyEvent: (event) =>
    set((state) => {
      switch (event.type) {
        case 'run.start':
          return { live: {}, running: true };

        case 'stage.start':
          if (!event.stage) return state;
          return {
            running: true,
            live: {
              ...state.live,
              [event.stage]: {
                stage: event.stage,
                name: event.name ?? event.stage,
                status: 'RUNNING',
              },
            },
          };

        case 'stage.end':
          if (!event.stage) return state;
          return {
            live: {
              ...state.live,
              [event.stage]: {
                stage: event.stage,
                name: event.name ?? event.stage,
                status: (event.status as LiveStage['status']) ?? 'OK',
                coverage: event.coverage,
                elapsedMs: event.elapsedMs,
                findings: event.findings,
                detail: event.detail,
              },
            },
          };

        case 'run.complete':
        case 'run.error':
          // The per-agent states are kept, not cleared. They are the record
          // of what just happened, and blanking the graph the instant a run
          // finishes throws away the thing the viewer was watching.
          return { running: false };

        default:
          return state;
      }
    }),

  setReport: (report, source) =>
    set({
      before: report,
      // A run carries one report. Until a remediated render is analysed in
      // its own right there is no honest "after" to show, so the toggle
      // holds the same data rather than implying a second measurement that
      // was never taken.
      after: report,
      source,
      applied: false,
      selectedFindingId: report.findings[0]?.id ?? null,
      categoryFilter: null,
      loading: false,
    }),

  hydrate: async () => {
    // An injected report is the CLI's own output for *this* page. It already
    // outranks anything the API might be serving from another run, so do not
    // go looking.
    if (get().source === 'injected') return;
    set({ loading: true });
    const report = await fetchLatestRun();
    if (report) get().setReport(report, 'api');
    else set({ loading: false });
  },
}));

/* ------------------------------------------------------------------ */
/* Selectors                                                           */
/* ------------------------------------------------------------------ */

export function useReport(): AnalysisReport {
  return useAnalysis((s) => (s.applied ? s.after : s.before));
}

/** Where the rendered report came from — surfaced so demo data is never
 * mistaken for a real run. */
export function useReportSource(): ReportSource {
  return useAnalysis((s) => s.source);
}

export function useLiveStages(): Record<string, LiveStage> {
  return useAnalysis((s) => s.live);
}

export function useRunning(): boolean {
  return useAnalysis((s) => s.running);
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

