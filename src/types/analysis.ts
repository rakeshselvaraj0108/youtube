/**
 * The PREFLIGHT data contract.
 *
 * Every value rendered on the Command Deck derives from this shape. When the
 * Python engine replaces the fixture with a real response, nothing in
 * `src/components` should need to change.
 */

export type Severity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export type Verdict =
  | 'READY_TO_PUBLISH'
  | 'PUBLISH_WITH_FIXES'
  | 'NOT_READY'
  | 'DO_NOT_PUBLISH';

export type AgentStatus = 'OK' | 'DEGRADED' | 'FAILED' | 'SKIPPED' | 'RUNNING' | 'PENDING';

export type OpKind = 'MUTE' | 'BLEEP' | 'BLUR_REGION' | 'REPLACE_AUDIO' | 'CUT';

/** Keys of the agent identity palette in tailwind.config.ts. */
export type AgentId =
  | 'orchestrator'
  | 'ingest'
  | 'speech'
  | 'vision'
  | 'ocr'
  | 'audio'
  | 'access'
  | 'meta'
  | 'policy'
  | 'score'
  | 'remedy'
  | 'report';

export interface AgentRun {
  id: AgentId;
  name: string;
  /** Depth in the DAG. Tier 0 is the orchestrator. */
  tier: number;
  parents: AgentId[];
  status: AgentStatus;
  detail: string;
  /** 0..1 — the fraction of the input this agent actually managed to inspect. */
  coverage: number;
  elapsedMs: number;
  /** Offset from run start, used to sequence the terminal log and graph pulses. */
  tsMs: number;
  /** Remote model invocations made by this agent. Free-tier budget evidence. */
  calls: number;
}

export interface AdversarialRecord {
  auditor: { charge: string };
  advocate: { defense: string | null; strength: number };
  adjudicator: {
    verdict: 'UPHELD' | 'DISMISSED';
    rationale: string;
    confidence: number;
  };
}

export interface Finding {
  id: string;
  clauseId: string;
  category: string;
  title: string;
  description: string;
  startMs: number;
  endMs: number;
  severity: Severity;
  confidence: number;
  /** Per-modality confidence before fusion, e.g. { speech: 0.94, vision: 0.92 }. */
  modalities: Record<string, number>;
  fusedConfidence: number;
  evidence: {
    transcript: string;
    /** [start, end] character offsets into `transcript`. */
    highlightSpan: [number, number];
    frames: string[];
  };
  policy: {
    clauseId: string;
    title: string;
    section: string;
    text: string;
  };
  adversarial: AdversarialRecord;
  suggestedFix: OpKind | 'NONE';
}

export interface RemediationOp {
  index: number;
  op: OpKind;
  startMs: number;
  endMs: number;
  details: string;
  findingId: string;
  /** BLUR_REGION only — normalised [x, y, w, h] in the 0..1 frame space. */
  box?: [number, number, number, number];
  /** REPLACE_AUDIO only — path to the CC-licensed replacement bed. */
  asset?: string;
  /** BLEEP only — tone frequency in Hz. */
  freqHz?: number;
}

export interface SubScores {
  policy: number;
  copyright: number;
  metadata: number;
  accessibility: number;
  audio: number;
}

export type SubScoreKey = keyof SubScores;

export interface RiskBand {
  startMs: number;
  endMs: number;
  /** 0..1 */
  risk: number;
}

export interface BreakdownRow {
  category: string;
  count: number;
  severity: Severity;
}

/**
 * One fixed-length slice of a long video, ranked by how much of the total risk
 * it carries. A 90-minute upload with 200 findings is not usefully read as a
 * list of 200 findings; it is usefully read as "segments 4 and 7 carry 82% of
 * the risk". Emitted only above the rollup threshold — short videos are their
 * own segment and the grouping tells you nothing.
 */
export interface Segment {
  index: number;
  startMs: number;
  endMs: number;
  findingCount: number;
  /** 0..1 — share of the run's total risk sitting in this segment. */
  riskShare: number;
  /** Most frequent clause in this segment, or null when it is clean. */
  dominantClause: string | null;
  worstSeverity: Severity | null;
}

export interface VideoMeta {
  filename: string;
  durationMs: number;
  width: number;
  height: number;
  fps: number;
  sizeBytes: number;
  audioCodec: string;
  sampleRate: number;
  posterUrl: string;
  srcUrl: string;
  /**
   * Base64 poster extracted at 10% of duration. Present when the report was
   * emitted by the CLI, absent in dev. Preferred over `posterUrl` so
   * report.html stays a single self-contained file.
   */
  posterDataUri?: string;
}

export interface RunMeta {
  analyzedAt: string;
  policyVersion: string;
  engineVersion: string;
  attestationHash: string;
  /** 0..1 — weighted mean of per-agent coverage. Reported, never hidden. */
  coverage: number;
}

export interface Scores {
  overall: number;
  sub: SubScores;
  verdict: Verdict;
  weakest: SubScoreKey;
}

export interface Remediation {
  ops: RemediationOp[];
  ffmpegCommand: string;
  renderMs: number;
  videoStreamCopied: boolean;
  /** Strategy applied, if any — conservative|balanced|aggressive. Absent means
   * each finding's own suggested fix was trusted directly. */
  strategy?: string;
  /** Human-readable compiler decisions, in order — snap-to-word, coalesce,
   * cut-budget demotions, and (when a strategy is set) "chose BLEEP over CUT —
   * same risk reduction, 0.65 less viewer impact" style overrides. */
  log: string[];
}

/** One stage the run gave up to stay inside its call budget. */
export interface ShedRecord {
  stage: string;
  reason: string;
  /** Windows the AUDITOR never examined because of this shed. */
  windowsLost: number;
}

/**
 * What the run predicted it would cost, against what it actually spent.
 *
 * `estimatedCalls` comes from the decomposition plan, computed before any
 * work started, and is an upper bound by construction — so `actualCalls`
 * exceeding it means the plan is wrong, not merely pessimistic. That makes
 * this block a check on PREFLIGHT rather than a decoration.
 */
export interface CostRecord {
  estimatedCalls: number;
  actualCalls: number;
  /** Ceiling in force, or null when the run was uncapped. */
  ceiling: number | null;
  shed: ShedRecord[];
}

/**
 * One event, as every agent that saw it described it.
 *
 * Findings are what each agent reported; incidents are what happened. Four
 * agents noticing the same moment is one problem observed four times, and a
 * creator deciding what to fix needs the second view — the first tells them
 * how many detectors fired, which is a fact about PREFLIGHT rather than
 * about their video.
 */
export interface Incident {
  id: string;
  startMs: number;
  endMs: number;
  category: string;
  severity: Severity;
  /** 0..1 — the best single observation, plus a bounded corroboration step
   * per additional independent agent. Never reaches certainty. */
  confidence: number;
  findingIds: string[];
  /** Independent agents that observed this. One agent reporting twice does
   * not appear twice, because an observer repeating itself is not
   * corroboration. */
  agents: string[];
  clauses: string[];
  suggestedFix: string;
  reasoning: string;
  /** True when more than one independent agent saw it. */
  corroborated: boolean;
}

export interface AnalysisReport {
  video: VideoMeta;
  meta: RunMeta;
  scores: Scores;
  riskBands: RiskBand[];
  findings: Finding[];
  incidents: Incident[];
  breakdown: BreakdownRow[];
  remediation: Remediation;
  agents: AgentRun[];
  /**
   * Present only for videos long enough to roll up (see the decomposition
   * plan's threshold). Absent, not empty, on short videos — an empty array
   * would read as "rolled up and found nothing".
   */
  segments?: Segment[];
  cost: CostRecord;
}
