import { SIGNAL_HEX } from '@/lib/scoring';
import type {
  EvidencePair,
  LifecycleState,
  RemediationRecord,
  Verification,
} from '@/types/analysis';

/**
 * The lifecycle, as the deck displays it.
 *
 * These are the persisted states from `preflight/lifecycle.py`, in the order
 * the machine visits them. The two housekeeping states either side —
 * REANALYSIS_QUEUED and RENDERED — are folded into their neighbours because
 * they are instantaneous bookkeeping rather than work a reader waits on;
 * every state that can hold for a perceptible time has its own step.
 *
 * Terminal verdicts are deliberately not steps. A verdict is the outcome of
 * the strip, not a stage within it, and the panel above renders it.
 */
export const LIFECYCLE_STEPS: {
  state: LifecycleState;
  label: string;
  hint: string;
}[] = [
  {
    state: 'REMEDIATION_REQUESTED',
    label: 'compile',
    hint: 'lowering findings into an edit list',
  },
  { state: 'RENDERING', label: 'render', hint: 'ffmpeg writing the output' },
  {
    state: 'STRUCTURAL_VERIFYING',
    label: 'structure',
    hint: 'checking the output against the edit list',
  },
  {
    state: 'STRUCTURALLY_VALID',
    label: 'valid',
    hint: 'the output matches what the edit list predicted',
  },
  {
    state: 'REANALYSING',
    label: 're-inspect',
    hint: 'running the full pipeline against the rendered file',
  },
  {
    state: 'COMPARING',
    label: 'compare',
    hint: 'matching findings and incidents across the two runs',
  },
];

export type StepStatus = 'done' | 'active' | 'pending' | 'failed';

/**
 * Where a step stands.
 *
 * `seen` is the ordered set of states the engine reported passing through, so
 * a completed step reads as completed rather than merely not-current. Falling
 * back to positional ordering covers a reload mid-run, where the events that
 * would have populated `seen` were delivered to a page that no longer exists.
 */
export function stepStatus(
  step: LifecycleState,
  active: string | null,
  seen: string[],
  record: RemediationRecord | null,
): StepStatus {
  if (record?.state === 'FAILED') {
    // The lifecycle records where it failed. Steps before it genuinely ran.
    const failedAt = record.previousState;
    const order = LIFECYCLE_STEPS.map((s) => s.state);
    const at = failedAt ? order.indexOf(failedAt as LifecycleState) : -1;
    const here = order.indexOf(step);
    if (at >= 0 && here === at) return 'failed';
    if (at >= 0 && here < at) return 'done';
    return 'pending';
  }

  // A finished remediation passed every step, whatever the verdict.
  if (record?.terminal) return 'done';

  if (step === active) return 'active';
  if (seen.includes(step)) return 'done';

  const order = LIFECYCLE_STEPS.map((s) => s.state);
  const here = order.indexOf(step);
  const now = active ? order.indexOf(active as LifecycleState) : -1;
  if (now >= 0 && here < now) return 'done';
  return 'pending';
}

/**
 * One vocabulary for comparison states, shared by every panel that shows one.
 *
 * Colour is never the only signal. Every state has a text label here and the
 * components render it — a reader who cannot distinguish the greens still
 * reads RESOLVED and PERSISTING, and a screenshot in a monochrome deck still
 * carries its meaning.
 *
 * These map onto the same palette the readiness scoring already uses rather
 * than introducing a second one, so "amber means attention" holds across the
 * whole page instead of only within this feature.
 */

export const STATUS_LABEL: Record<string, string> = {
  RESOLVED: 'resolved',
  PERSISTING: 'persisting',
  PARTIALLY_REMEDIATED: 'partly fixed',
  CHANGED: 'changed',
  NEW: 'new',
  INCONCLUSIVE: 'inconclusive',

  VERIFIED_SAFE: 'verified safe',
  REMEDIATION_FAILED: 'failed',
  NEW_RISK_DETECTED: 'new risk',
  NO_CHANGE: 'no change',
};

export function statusHex(status: string): string {
  switch (status) {
    case 'RESOLVED':
    case 'VERIFIED_SAFE':
      return SIGNAL_HEX.clear;
    case 'NEW':
    case 'NEW_RISK_DETECTED':
    case 'REMEDIATION_FAILED':
      return SIGNAL_HEX.critical;
    case 'PERSISTING':
    case 'PARTIALLY_REMEDIATED':
      return SIGNAL_HEX.medium;
    case 'CHANGED':
      return SIGNAL_HEX.medium;
    default:
      // Inconclusive is deliberately not a warning colour. "Nobody looked" is
      // not a milder problem — it is an absence of information, and dressing
      // it as a mild problem invites a reader to treat it as nearly fine.
      return '#8A97AE';
  }
}

/** One line explaining what the verdict rests on. */
export function verdictRationale(verification: Verification): string {
  if (!verification.structuralOk) {
    return 'The rendered file did not match the edit list, so nothing was concluded about its content.';
  }
  if (!verification.reanalysisOk) {
    return 'The rendered file could not be re-analysed. Whether the findings were resolved is unknown.';
  }
  const parts: string[] = [];
  if (verification.resolved) parts.push(`${verification.resolved} resolved`);
  if (verification.persisting) parts.push(`${verification.persisting} still detected`);
  if (verification.new) parts.push(`${verification.new} newly detected`);
  if (verification.inconclusive) {
    parts.push(`${verification.inconclusive} not checkable at the coverage reached`);
  }
  return parts.length ? parts.join(', ') : 'Nothing changed between the two runs.';
}

/** The evidence pair for a finding, or null. Never a fallback to another. */
export function pairFor(
  evidence: EvidencePair[],
  findingId: string | null,
): EvidencePair | null {
  if (!findingId) return null;
  return evidence.find((pair) => pair.findingId === findingId) ?? null;
}

/**
 * Where to seek for a pair, in the timeline currently on screen.
 *
 * Null means there is nowhere to go — the span was cut, so the remediated
 * timeline has no counterpart. Returning 0 instead would seek to the start of
 * the video and quietly assert the evidence is there.
 */
export function seekTarget(pair: EvidencePair, applied: boolean): number | null {
  if (!applied) return pair.before.frame ? pair.before.tsMs : null;
  if (pair.after.removedByRemediation) return null;
  return pair.after.tsMs;
}
