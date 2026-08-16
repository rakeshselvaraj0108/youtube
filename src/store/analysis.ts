import { create } from 'zustand';
import type { AnalysisReport, Finding } from '@/types/analysis';
import { beforeReport as fixtureBefore } from '@/data/fixture';
import {
  fetchLatestRun,
  fetchRemediations,
  type InterruptedRemediation,
  type StageEvent,
} from '@/lib/api';
import type {
  EvidencePair,
  RemediationRecord,
  Verification,
  VerificationCertificate,
} from '@/types/analysis';
import { sortFindings, type FindingSort } from '@/lib/findings';
import { injectedAfterReport, injectedReport } from '@/lib/reportSource';

/**
 * The report the deck renders. Real CLI output when report.html injected it,
 * the demo fixture otherwise. Resolved once at module load so every selector
 * agrees on which report it is reading.
 */
export const BEFORE: AnalysisReport = injectedReport() ?? fixtureBefore;
// A remediated report exists only after a separate analysis of the rendered
// artifact. The fallback preserves the layout without manufacturing an
// unverified improvement from fixture data.
export const AFTER: AnalysisReport = injectedAfterReport() ?? BEFORE;

export type DetailTab = 'EVIDENCE' | 'POLICY' | 'ADVERSARIAL' | 'REASONING';

/**
 * Where the report on screen came from. Shown in the header, because a deck
 * rendering demo data and a deck rendering a real run must never be
 * mistakable for one another — that confusion is how a fixture ends up in a
 * screenshot presented as a result.
 */
export type ReportSource = 'injected' | 'api' | 'fixture';
export type AnalysisStatus = 'IDLE' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';

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
  analysisStatus: AnalysisStatus;
  analysisError: string | null;
  /** True while the API is being polled at startup. */
  loading: boolean;

  /** Which render the deck is showing. The fix transition flips this. */
  applied: boolean;
  /** Set only for an embedded or freshly measured remediation report. */
  hasVerifiedAfter: boolean;
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

  /**
   * The incident under investigation. Distinct from `selectedFindingId`:
   * a finding is one agent's observation, an incident is the event those
   * observations describe, and a reader moves between the two constantly.
   */
  selectedIncidentId: string | null;
  expandedIncidentId: string | null;
  incidentSeverity: string | null;

  /** Simulator is collapsed by default — it is the decision layer, not
   * the reporting layer, and should not consume the deck by default. */
  simulatorOpen: boolean;
  selectedScenario: string | null;
  toggleSimulator: () => void;
  selectScenario: (name: string | null) => void;

  selectIncident: (id: string | null) => void;
  toggleIncident: (id: string) => void;
  setIncidentSeverity: (severity: string | null) => void;

  /* ---- the closed loop --------------------------------------------- */

  /**
   * The comparison between the original and the rendered artifact, exactly as
   * the backend computed it. Null until a remediation has actually reached a
   * verdict — never derived from a plan, a prediction or a successful ffmpeg
   * exit, because each of those is a claim about what *should* happen and
   * this is the record of what did.
   */
  verification: Verification | null;
  certificate: VerificationCertificate | null;
  evidence: EvidencePair[];
  telemetry: Record<string, unknown> | null;
  /** The persisted lifecycle row, so the deck shows a state the engine
   * actually stored rather than one inferred from the last event seen. */
  remediationRecord: RemediationRecord | null;
  /** Remediations a previous process left unfinished, found at startup. */
  interrupted: InterruptedRemediation[];
  /** Which comparison row the reader is inspecting — a finding change or an
   * incident change. Drives the evidence panel and the players. */
  selectedChangeId: string | null;

  /**
   * The remediation in flight, as the engine reports it.
   *
   * `fixState` is the lifecycle state the backend says it is in — never one
   * inferred here from a stage word. `fixSeen` is the ordered set of states
   * already passed, which is what lets the strip show a completed step as
   * completed rather than merely not-current.
   */
  fixRunning: boolean;
  fixStage: string | null;
  fixDetail: string;
  fixState: string | null;
  fixSeen: string[];
  fixError: string | null;
  fixStartedAt: number | null;

  applyFixEvent: (event: StageEvent) => void;
  beginFix: () => void;

  adoptVerification: (event: StageEvent) => void;
  selectChange: (findingId: string | null) => void;
  /** Ask the engine what it has on disk. Surfaces interrupted work. */
  loadRemediations: () => Promise<void>;

  /** Adopt a report the engine produced. */
  setReport: (report: AnalysisReport, source: ReportSource) => void;
  /** Adopt the independently analysed rendered artifact. This is deliberately
   * separate from `setApplied`: a toggle must never manufacture an "after"
   * result from a plan or a successful ffmpeg exit. */
  setRemediatedReport: (report: AnalysisReport) => void;
  /** Ask the API for its newest run. No-op when nothing answers. */
  hydrate: () => Promise<void>;
}

export const useAnalysis = create<AnalysisState>((set, get) => ({
  before: BEFORE,
  after: AFTER,
  source: injectedReport() ? 'injected' : 'fixture',
  analysisStatus: injectedReport() ? 'COMPLETED' : 'IDLE',
  analysisError: null,
  loading: false,

  applied: false,
  hasVerifiedAfter: Boolean(injectedAfterReport()),
  selectedFindingId: BEFORE.findings[0]?.id ?? null,
  categoryFilter: null,
  sortBy: 'severity',
  detailTab: 'EVIDENCE',
  hoveredOpIndex: null,

  setApplied: (applied) =>
    set((state) => {
      if (applied && !state.hasVerifiedAfter) return state;
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
      const owning = (report.incidents ?? []).find((i) =>
        i.findingIds.includes(finding.id),
      );
      const base = { selectedFindingId, selectedIncidentId: owning?.id ?? null };
      return scoped ? base : { ...base, playheadMs: finding.startMs };
    }),
  setCategoryFilter: (categoryFilter) => set({ categoryFilter }),
  setSortBy: (sortBy) => set({ sortBy }),
  setDetailTab: (detailTab) => set({ detailTab }),
  setHoveredOp: (hoveredOpIndex) => set({ hoveredOpIndex }),

  live: {},
  running: false,
  playheadMs: 0,
  clauseFilter: null,

  simulatorOpen: false,
  selectedScenario: null,
  toggleSimulator: () => set((st) => ({ simulatorOpen: !st.simulatorOpen })),
  selectScenario: (selectedScenario) => set({ selectedScenario }),

  selectedIncidentId: null,
  expandedIncidentId: null,
  incidentSeverity: null,

  // Selecting an incident seeks to where it starts and selects its first
  // finding, so the detail panel, the evidence and the policy tab all follow
  // without the reader having to click again in three more places.
  selectIncident: (id) =>
    set((state) => {
      const report = state.applied ? state.after : state.before;
      const incident = (report.incidents ?? []).find((i) => i.id === id);
      if (!incident) return { selectedIncidentId: id };
      const duration = report.video.durationMs || 0;
      const scoped =
        duration > 0 && incident.endMs - incident.startMs >= duration * 0.9;
      return {
        selectedIncidentId: id,
        expandedIncidentId: id,
        selectedFindingId: incident.findingIds[0] ?? state.selectedFindingId,
        // A file-scoped incident describes the upload, not a moment in it —
        // seeking to 0 would assert the problem is at the start.
        ...(scoped ? {} : { playheadMs: incident.startMs }),
      };
    }),

  toggleIncident: (id) =>
    set((state) => ({
      expandedIncidentId: state.expandedIncidentId === id ? null : id,
    })),

  setIncidentSeverity: (incidentSeverity) => set({ incidentSeverity }),

  seekTo: (ms) => set({ playheadMs: Math.max(0, Math.round(ms)) }),
  setClauseFilter: (clauseFilter) => set({ clauseFilter }),

  resetLive: () => set({ live: {}, running: false }),

  applyEvent: (event) =>
    set((state) => {
      switch (event.type) {
        case 'run.start':
          return { live: {}, running: true, analysisStatus: 'RUNNING', analysisError: null };

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
          // The per-agent states are kept, not cleared. They are the record
          // of what just happened, and blanking the graph the instant a run
          // finishes throws away the thing the viewer was watching.
          return { running: false, analysisStatus: 'COMPLETED' };

        case 'run.error':
          return { running: false, analysisStatus: 'FAILED', analysisError: event.error ?? 'analysis failed' };

        default:
          return state;
      }
    }),

  verification: null,
  certificate: null,
  evidence: [],
  telemetry: null,
  remediationRecord: null,
  interrupted: [],
  selectedChangeId: null,

  fixRunning: false,
  fixStage: null,
  fixDetail: '',
  fixState: null,
  fixSeen: [],
  fixError: null,
  fixStartedAt: null,

  beginFix: () =>
    set({
      fixRunning: true,
      fixStage: 'starting',
      fixDetail: 'requesting a remediation',
      fixState: 'REMEDIATION_REQUESTED',
      fixSeen: [],
      fixError: null,
      fixStartedAt: Date.now(),
      // The previous verdict belongs to the previous render. Keeping it on
      // screen while a new remediation runs would attribute an old result to
      // work still in progress.
      verification: null,
      certificate: null,
      evidence: [],
    }),

  applyFixEvent: (event) =>
    set((state) => {
      if (event.type === 'run.error') {
        return { fixRunning: false, fixError: event.error ?? 'remediation failed' };
      }
      if (event.type === 'run.complete') {
        return { fixRunning: false, fixStage: 'complete', fixDetail: '' };
      }
      if (event.type !== 'fix.progress') return state;
      const seen =
        event.state && !state.fixSeen.includes(event.state)
          ? [...state.fixSeen, event.state]
          : state.fixSeen;
      return {
        fixRunning: true,
        fixStage: event.stage ?? state.fixStage,
        fixDetail: event.detail ?? '',
        fixState: event.state ?? state.fixState,
        fixSeen: seen,
      };
    }),

  // Adopted wholesale from the terminal event. Nothing is recomputed here:
  // a rollup on this side would eventually disagree with the certificate,
  // and then the page and the document it displays would be making different
  // claims about the same run.
  adoptVerification: (event) =>
    set(() => ({
      verification: event.verification ?? null,
      certificate: event.certificate ?? null,
      evidence: event.evidence ?? [],
      telemetry: event.telemetry ?? null,
      remediationRecord: event.lifecycle ?? null,
      selectedChangeId:
        // Open on what appeared, if anything did. A new finding is the one
        // outcome a reader must not have to go looking for.
        event.verification?.changes.find((c) => c.status === 'NEW')?.remediatedId ??
        event.verification?.changes[0]?.originalId ??
        null,
    })),

  selectChange: (selectedChangeId) =>
    set((state) => {
      const pair = state.evidence.find((p) => p.findingId === selectedChangeId);
      if (!pair) return { selectedChangeId };
      // Seek to the moment being discussed — in whichever timeline the reader
      // is looking at. Seeking the remediated player to the original
      // timestamp would land on unrelated material after a cut.
      const ms = state.applied ? pair.after.tsMs : pair.before.tsMs;
      return {
        selectedChangeId,
        selectedFindingId: pair.findingId,
        selectedIncidentId: pair.incidentId ?? state.selectedIncidentId,
        // A removed span has no counterpart to seek to, so the playhead stays
        // where it is rather than jumping somewhere that means nothing.
        ...(ms === null || ms === undefined ? {} : { playheadMs: ms }),
      };
    }),

  loadRemediations: async () => {
    const { interrupted } = await fetchRemediations();
    set({ interrupted });
  },

  setReport: (report, source) =>
    set({
      before: report,
      // A run carries one report. Until a remediated render is analysed in
      // its own right there is no honest "after" to show, so the toggle
      // holds the same data rather than implying a second measurement that
      // was never taken.
      after: report,
      source,
      analysisStatus: 'COMPLETED',
      analysisError: null,
      applied: false,
      hasVerifiedAfter: false,
      selectedFindingId: report.findings[0]?.id ?? null,
      categoryFilter: null,
      loading: false,
    }),

  setRemediatedReport: (report) =>
    set({
      after: report,
      applied: true,
      hasVerifiedAfter: true,
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
    else set({ loading: false, analysisStatus: 'IDLE' });
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

