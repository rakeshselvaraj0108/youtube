import type { Severity, SubScoreKey, SubScores, Verdict } from '@/types/analysis';

/**
 * Release Readiness scoring.
 *
 * Two rules govern this file and both exist to stop the headline number lying:
 *
 *  1. Every dimension runs in the SAME direction. 100 is always good. A bar that
 *     is 90% full always means "this dimension is fine", whether it is measuring
 *     copyright exposure or caption coverage.
 *
 *  2. The overall score is capped at `worst + 15`. A confirmed Content ID match
 *     scores 19/100 on copyright; without the cap, four healthy dimensions would
 *     average that away into a passing grade. One fatal flaw is never averaged
 *     away — that is the entire reason a creator would trust this number.
 */

export const WEIGHTS: Readonly<Record<SubScoreKey, number>> = Object.freeze({
  policy: 0.4,
  copyright: 0.3,
  metadata: 0.12,
  accessibility: 0.1,
  audio: 0.08,
});

/** How far above the weakest dimension the overall score is allowed to float. */
export const CLAMP_HEADROOM = 15;

export const SUB_SCORE_ORDER: readonly SubScoreKey[] = [
  'policy',
  'copyright',
  'metadata',
  'accessibility',
  'audio',
] as const;

export const SUB_SCORE_LABELS: Readonly<Record<SubScoreKey, string>> = Object.freeze({
  policy: 'Policy',
  copyright: 'Copyright',
  metadata: 'Metadata',
  accessibility: 'Accessibility',
  audio: 'Audio',
});

export interface ReadinessResult {
  /** The headline number, 0–100, rounded. Verdict is derived from this value. */
  overall: number;
  /** Weighted mean before the clamp — exposed so the UI can show its own working. */
  weighted: number;
  /** Lowest sub-score. */
  worst: number;
  weakest: SubScoreKey;
  verdict: Verdict;
  /** True when the clamp actually bound, i.e. one dimension dragged the score down. */
  capped: boolean;
}

export function computeReadiness(sub: SubScores): ReadinessResult {
  const weighted = SUB_SCORE_ORDER.reduce((acc, key) => acc + WEIGHTS[key] * sub[key], 0);

  // Ties resolve to the earlier key in SUB_SCORE_ORDER, so `weakest` is stable.
  let weakest: SubScoreKey = SUB_SCORE_ORDER[0];
  for (const key of SUB_SCORE_ORDER) {
    if (sub[key] < sub[weakest]) weakest = key;
  }
  const worst = sub[weakest];

  const clamped = Math.min(weighted, worst + CLAMP_HEADROOM);
  const overall = Math.round(clamp01to100(clamped));

  return {
    overall,
    weighted,
    worst,
    weakest,
    verdict: verdictFor(overall, worst),
    capped: worst + CLAMP_HEADROOM < weighted,
  };
}

export function verdictFor(overall: number, worst: number): Verdict {
  if (overall >= 85 && worst >= 70) return 'READY_TO_PUBLISH';
  if (overall >= 70 && worst >= 50) return 'PUBLISH_WITH_FIXES';
  if (overall >= 50) return 'NOT_READY';
  return 'DO_NOT_PUBLISH';
}

function clamp01to100(n: number): number {
  return Math.max(0, Math.min(100, n));
}

/* ------------------------------------------------------------------ */
/* Presentation ramps                                                  */
/* ------------------------------------------------------------------ */

/** Tokens in the `sig` ramp. The only saturated colour allowed in data. */
export type SignalTone = 'critical' | 'high' | 'medium' | 'low' | 'clear';

export const SIGNAL_HEX: Readonly<Record<SignalTone, string>> = Object.freeze({
  critical: '#FF3B5C',
  high: '#FF7A45',
  medium: '#FFB020',
  low: '#A3E635',
  clear: '#34D399',
});

/**
 * Readiness ramp — higher is better, so it runs the opposite way to severity.
 * red < 50 · amber 50–69 · lime 70–84 · green 85+
 */
export function readinessTone(score: number): SignalTone {
  if (score >= 85) return 'clear';
  if (score >= 70) return 'low';
  if (score >= 50) return 'medium';
  return 'critical';
}

export function readinessHex(score: number): string {
  return SIGNAL_HEX[readinessTone(score)];
}

/** Risk ramp — 0..1 where higher is worse. Used by the timeline terrain. */
export function riskTone(risk: number): SignalTone {
  if (risk >= 0.8) return 'critical';
  if (risk >= 0.6) return 'high';
  if (risk >= 0.4) return 'medium';
  if (risk >= 0.2) return 'low';
  return 'clear';
}

export function riskHex(risk: number): string {
  return SIGNAL_HEX[riskTone(risk)];
}

export const SEVERITY_TONE: Readonly<Record<Severity, SignalTone>> = Object.freeze({
  CRITICAL: 'critical',
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low',
});

export function severityHex(severity: Severity): string {
  return SIGNAL_HEX[SEVERITY_TONE[severity]];
}

export const SEVERITY_RANK: Readonly<Record<Severity, number>> = Object.freeze({
  CRITICAL: 0,
  HIGH: 1,
  MEDIUM: 2,
  LOW: 3,
});

export interface VerdictMeta {
  label: string;
  tone: SignalTone;
}

export const VERDICT_META: Readonly<Record<Verdict, VerdictMeta>> = Object.freeze({
  READY_TO_PUBLISH: { label: 'READY TO PUBLISH', tone: 'clear' },
  PUBLISH_WITH_FIXES: { label: 'PUBLISH WITH FIXES', tone: 'medium' },
  NOT_READY: { label: 'NOT READY', tone: 'high' },
  DO_NOT_PUBLISH: { label: 'DO NOT PUBLISH', tone: 'critical' },
});
