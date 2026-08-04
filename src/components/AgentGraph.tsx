import { useMemo } from 'react';
import { useReport } from '@/store/analysis';
import { AGENT_HEX } from '@/lib/agents';
import { SIGNAL_HEX } from '@/lib/scoring';
import type { AgentId } from '@/types/analysis';

/**
 * The agent DAG.
 *
 * 2D SVG for now. Phase 5 attempts the WebGL version against a 3ms frame
 * budget; if it misses, this stays — the page is better with a fast 2D graph
 * than a slow 3D one. Node positions are derived from tier and fan-out, never
 * hand-placed.
 */

const W = 300;
const H = 232;
const PAD_X = 26;
const PAD_Y = 18;

interface Node {
  id: AgentId;
  name: string;
  x: number;
  y: number;
  color: string;
  degraded: boolean;
}

export function AgentGraph() {
  const report = useReport();

  const { nodes, edges } = useMemo(() => {
    const tiers = new Map<number, typeof report.agents>();
    for (const agent of report.agents) {
      const bucket = tiers.get(agent.tier);
      if (bucket) bucket.push(agent);
      else tiers.set(agent.tier, [agent]);
    }

    const tierKeys = [...tiers.keys()].sort((a, b) => a - b);
    const rowGap = (H - PAD_Y * 2) / Math.max(1, tierKeys.length - 1);

    const placed = new Map<AgentId, Node>();
    tierKeys.forEach((tier, row) => {
      const members = tiers.get(tier)!;
      const colGap = (W - PAD_X * 2) / (members.length + 1);
      members.forEach((agent, col) => {
        placed.set(agent.id, {
          id: agent.id,
          name: agent.name,
          x: PAD_X + colGap * (col + 1),
          y: PAD_Y + rowGap * row,
          color: AGENT_HEX[agent.id],
          degraded: agent.status !== 'OK',
        });
      });
    });

    const links: { from: Node; to: Node; key: string }[] = [];
    for (const agent of report.agents) {
      const to = placed.get(agent.id);
      if (!to) continue;
      for (const parentId of agent.parents) {
        const from = placed.get(parentId);
        if (from) links.push({ from, to, key: `${parentId}->${agent.id}` });
      }
    }

    return { nodes: [...placed.values()], edges: links };
  }, [report.agents]);

  return (
    <div className="shrink-0 border-t border-edge px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-panel uppercase text-inkDim">Agent Graph</span>
        <span className="num text-[9px] text-inkFaint">{nodes.length} nodes</span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Agent dependency graph">
        {edges.map((edge) => (
          <line
            key={edge.key}
            x1={edge.from.x}
            y1={edge.from.y}
            x2={edge.to.x}
            y2={edge.to.y}
            stroke="#1A2233"
            strokeWidth={1}
          />
        ))}

        {nodes.map((node) => (
          <g key={node.id}>
            {node.degraded && (
              <circle
                cx={node.x}
                cy={node.y}
                r={7}
                fill="none"
                stroke={SIGNAL_HEX.medium}
                strokeWidth={1}
                opacity={0.7}
              />
            )}
            <circle cx={node.x} cy={node.y} r={4} fill={node.color} />
            <circle
              cx={node.x}
              cy={node.y}
              r={4}
              fill="none"
              stroke={node.color}
              strokeWidth={1}
              opacity={0.35}
            />
            <title>{`${node.name}${node.degraded ? ' — degraded' : ''}`}</title>
          </g>
        ))}
      </svg>
    </div>
  );
}
