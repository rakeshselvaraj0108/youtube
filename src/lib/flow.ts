import type { AgentId, AgentRun, AgentStatus } from '@/types/analysis';

/**
 * Layout and clock for the agent flow.
 *
 * The graph is a single plane tilted in CSS 3D. Depth comes from perspective on
 * that plane, not from per-node translateZ — that keeps the SVG edge layer and
 * the DOM node layer in the same coordinate space, so edges actually land on
 * the nodes they connect instead of drifting as the plane rotates.
 *
 * Rows compress the 8-tier DAG into 4 visual ranks: orchestration, perception,
 * extraction, synthesis. The edges still encode the real parent relationships,
 * so cross-rank links are visible rather than hidden by a tidy layout.
 */

export const FLOW_ROWS: AgentId[][] = [
  ['orchestrator'],
  ['ingest', 'speech', 'vision', 'audio'],
  ['ocr', 'meta', 'access', 'policy'],
  ['score', 'remedy', 'report'],
];

export const ROW_LABELS = ['Orchestration', 'Perception', 'Extraction', 'Synthesis'];

export interface FlowNode {
  id: AgentId;
  row: number;
  /** Percentage across the plane, 0..100. */
  x: number;
  /** Percentage down the plane, 0..100. */
  y: number;
}

const ROW_Y = [8, 34, 62, 90];

export function layoutFlow(): Map<AgentId, FlowNode> {
  const nodes = new Map<AgentId, FlowNode>();
  FLOW_ROWS.forEach((row, rowIndex) => {
    const gap = 100 / (row.length + 1);
    row.forEach((id, col) => {
      nodes.set(id, {
        id,
        row: rowIndex,
        x: gap * (col + 1),
        y: ROW_Y[rowIndex] ?? 50,
      });
    });
  });
  return nodes;
}

/**
 * A cubic bezier with vertical control handles. Vertical handles make every
 * edge leave its parent downward and arrive at its child from above, which is
 * what makes a dense graph readable — the eye follows direction, not just line.
 */
export function edgePath(from: FlowNode, to: FlowNode): string {
  const dy = to.y - from.y;
  const bend = Math.max(6, Math.abs(dy) * 0.45);
  return `M ${from.x} ${from.y} C ${from.x} ${from.y + bend}, ${to.x} ${to.y - bend}, ${to.x} ${to.y}`;
}

/* ------------------------------------------------------------------ */
/* Replay clock                                                        */
/* ------------------------------------------------------------------ */

/** How much faster the replay runs than the real pipeline did. */
export const REPLAY_SPEED = 4.6;

export function totalRunMs(agents: AgentRun[]): number {
  return agents.reduce((max, a) => Math.max(max, a.tsMs + a.elapsedMs), 0);
}

/**
 * Agent status at a point in the replay. `clock` is elapsed run time in ms, or
 * null once the replay has finished and the report's real statuses apply.
 */
export function statusAt(agent: AgentRun, clock: number | null): AgentStatus {
  if (clock === null) return agent.status;
  if (clock < agent.tsMs) return 'PENDING';
  if (clock < agent.tsMs + agent.elapsedMs) return 'RUNNING';
  return agent.status;
}

/** An edge carries a pulse while its child agent is working. */
export function edgeActive(child: AgentRun, clock: number | null): boolean {
  return statusAt(child, clock) === 'RUNNING';
}
