import type { AgentId, AgentRun } from '@/types/analysis';

/**
 * Agent identity colours.
 *
 * These appear in exactly two places — the terminal log and the agent graph —
 * because that is where an agent is the subject. Everywhere else on the deck,
 * colour means severity, and mixing the two vocabularies would make both
 * meaningless.
 */
export const AGENT_HEX: Readonly<Record<AgentId, string>> = Object.freeze({
  orchestrator: '#94A3B8',
  ingest: '#38BDF8',
  speech: '#818CF8',
  vision: '#E879F9',
  ocr: '#F472B6',
  audio: '#2DD4BF',
  access: '#FBBF24',
  meta: '#A78BFA',
  policy: '#22D3EE',
  score: '#4ADE80',
  remedy: '#FB7185',
  report: '#94A3B8',
});

export const STATUS_GLYPH: Readonly<Record<AgentRun['status'], string>> = Object.freeze({
  OK: '✓',
  DEGRADED: '!',
  FAILED: '✗',
  SKIPPED: '–',
  RUNNING: '▸',
  PENDING: '·',
});

/** `[hh:mm:ss]` elapsed from run start — this is offset time, not wall clock. */
export function elapsedStamp(ms: number): string {
  const total = Math.floor(ms / 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(Math.floor(total / 3600))}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
}
