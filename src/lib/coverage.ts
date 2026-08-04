import type { AgentId, AgentRun } from '@/types/analysis';

/**
 * Coverage honesty.
 *
 * A compliance tool that reports what it could NOT see is more trustworthy than
 * one that quietly returns a clean bill of health. Coverage is a weighted mean
 * over agents, weighted by each agent's share of the analysis surface — the
 * vision agent inspecting 42% of keyframes costs far more coverage than the
 * report writer, so the two cannot count equally.
 */
export const AGENT_SURFACE_WEIGHT: Readonly<Record<AgentId, number>> = Object.freeze({
  orchestrator: 0,
  ingest: 0.05,
  speech: 0.2,
  vision: 0.22,
  ocr: 0.13,
  audio: 0.15,
  access: 0.06,
  meta: 0.04,
  policy: 0.1,
  score: 0.02,
  remedy: 0.02,
  report: 0.01,
});

export function computeCoverage(agents: AgentRun[]): number {
  let weightSum = 0;
  let acc = 0;
  for (const agent of agents) {
    const w = AGENT_SURFACE_WEIGHT[agent.id] ?? 0;
    weightSum += w;
    acc += w * agent.coverage;
  }
  return weightSum === 0 ? 1 : acc / weightSum;
}

/** Agents that did not fully complete — the tooltip behind the coverage chip. */
export function degradedAgents(agents: AgentRun[]): AgentRun[] {
  return agents.filter(
    (a) => a.status === 'DEGRADED' || a.status === 'FAILED' || a.status === 'SKIPPED',
  );
}
