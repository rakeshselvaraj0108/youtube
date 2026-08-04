import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AudioLines,
  Captions,
  Expand,
  Eye,
  FileCheck,
  Film,
  Gauge,
  Mic,
  RotateCw,
  Scale,
  ScanText,
  Tags,
  Wrench,
  X,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useReport } from '@/store/analysis';
import { AGENT_HEX } from '@/lib/agents';
import { SIGNAL_HEX } from '@/lib/scoring';
import {
  edgeActive,
  edgePath,
  layoutFlow,
  REPLAY_SPEED,
  statusAt,
  totalRunMs,
  type FlowNode,
} from '@/lib/flow';
import type { AgentId, AgentRun, AgentStatus } from '@/types/analysis';

/**
 * MULTI-AGENT FLOW
 *
 * The pipeline, live. Twelve agents on a plane tilted in CSS 3D; each edge
 * carries a travelling pulse from parent to child while that child is working,
 * timed to the same tsMs values the terminal log prints. The graph and the log
 * narrate the same run in two languages.
 *
 * Why CSS 3D rather than WebGL: the nodes need crisp 16px icons and small caps
 * labels. Those are pixel-perfect and nearly free in DOM, and blurry and
 * expensive as WebGL textures. Depth is doing real work here — pipeline order
 * reads as recession — so it is bought with one perspective transform rather
 * than a renderer, a frame budget, and 140KB of dependency.
 */

const ICONS: Record<AgentId, LucideIcon> = {
  orchestrator: Scale,
  ingest: Film,
  speech: Mic,
  vision: Eye,
  audio: AudioLines,
  ocr: ScanText,
  meta: Tags,
  access: Captions,
  policy: Scale,
  score: Gauge,
  remedy: Wrench,
  report: FileCheck,
};

const SHORT_NAME: Record<AgentId, string> = {
  orchestrator: 'Orchestrator',
  ingest: 'Video\nProcessing',
  speech: 'Speech\nAgent',
  vision: 'Vision\nAgent',
  audio: 'Audio\nAgent',
  ocr: 'OCR\nAgent',
  meta: 'Metadata\nAgent',
  access: 'Access\nAgent',
  policy: 'Policy\nAgents ×6',
  score: 'Risk\nScoring',
  remedy: 'Remediation\nAgent',
  report: 'Report\nAgent',
};

function Falcon({ size }: { size: number }) {
  return (
    <svg viewBox="0 0 24 16" width={size} height={size * 0.66} fill="none" aria-hidden="true">
      <path
        d="M1 4.5 L11.2 7.4 L12 3 L12.8 7.4 L23 4.5 L14.4 9.6 L12 15 L9.6 9.6 Z"
        fill="currentColor"
      />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Replay clock                                                        */
/* ------------------------------------------------------------------ */

function useReplay(agents: AgentRun[]) {
  const runMs = useMemo(() => totalRunMs(agents), [agents]);
  const [clock, setClock] = useState<number | null>(0);
  const rafRef = useRef<number | undefined>(undefined);

  const play = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setClock(null);
      return;
    }
    const start = performance.now();
    const step = (now: number) => {
      const elapsed = (now - start) * REPLAY_SPEED;
      if (elapsed >= runMs) {
        setClock(null); // settle on the report's real statuses
        return;
      }
      setClock(elapsed);
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
  }, [runMs]);

  useEffect(() => {
    play();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [play]);

  return { clock, play, runMs };
}

/* ------------------------------------------------------------------ */
/* Node                                                                */
/* ------------------------------------------------------------------ */

interface Scale2 {
  node: number;
  icon: number;
  label: string;
  gap: string;
}

const COMPACT: Scale2 = { node: 34, icon: 14, label: 'text-[7.5px]', gap: 'gap-1' };
const FULL: Scale2 = { node: 62, icon: 24, label: 'text-[10px]', gap: 'gap-2' };

function Node({
  agent,
  node,
  status,
  scale,
  dimmed,
  onHover,
}: {
  agent: AgentRun;
  node: FlowNode;
  status: AgentStatus;
  scale: Scale2;
  dimmed: boolean;
  onHover: (id: AgentId | null) => void;
}) {
  const color = AGENT_HEX[agent.id];
  const Icon = ICONS[agent.id];
  const isOrchestrator = agent.id === 'orchestrator';
  const size = isOrchestrator ? scale.node * 1.25 : scale.node;

  const running = status === 'RUNNING';
  const pending = status === 'PENDING';
  const degraded = status === 'DEGRADED';
  const done = status === 'OK' || degraded;

  return (
    <div
      className={`absolute flex -translate-x-1/2 -translate-y-1/2 flex-col items-center transition-opacity duration-fast ${scale.gap}`}
      style={{ left: `${node.x}%`, top: `${node.y}%`, opacity: dimmed ? 0.3 : 1 }}
      onMouseEnter={() => onHover(agent.id)}
      onMouseLeave={() => onHover(null)}
    >
      <div
        className="relative flex items-center justify-center"
        style={{ width: size, height: size }}
      >
        {(running || done) && (
          <span
            className={`absolute inset-0 rounded-full blur-md ${running ? 'node-halo' : ''}`}
            style={{ background: color, opacity: running ? 0.35 : 0.16 }}
            aria-hidden="true"
          />
        )}

        {done && (
          <svg
            className="absolute inset-0"
            viewBox="0 0 40 40"
            style={{ width: size, height: size }}
            aria-hidden="true"
          >
            <circle
              cx="20"
              cy="20"
              r="18.5"
              fill="none"
              stroke={degraded ? SIGNAL_HEX.medium : color}
              strokeWidth="1"
              strokeDasharray={degraded ? '3 3' : undefined}
              className={degraded ? 'ring-degraded' : ''}
              opacity={degraded ? 0.85 : 0.55}
            />
          </svg>
        )}

        <span
          className={`relative flex items-center justify-center rounded-full border ${
            running ? 'node-running' : ''
          }`}
          style={{
            width: size - scale.node * 0.24,
            height: size - scale.node * 0.24,
            borderColor: pending ? '#26324A' : `${color}99`,
            background: pending
              ? '#0B0F17'
              : `radial-gradient(circle at 50% 30%, ${color}33 0%, #070A11 72%)`,
            color: pending ? '#4E5A70' : color,
            opacity: pending ? 0.45 : 1,
          }}
        >
          {isOrchestrator ? (
            <Falcon size={scale.icon * 1.15} />
          ) : (
            <Icon size={scale.icon} strokeWidth={1.8} />
          )}
        </span>
      </div>

      <span
        className={`whitespace-pre text-center uppercase leading-[1.35] tracking-[0.06em] ${scale.label}`}
        style={{ color: pending ? '#4E5A70' : '#8A97AE', opacity: pending ? 0.5 : 1 }}
      >
        {SHORT_NAME[agent.id]}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The plane                                                           */
/* ------------------------------------------------------------------ */

function FlowPlane({
  agents,
  clock,
  scale,
  height,
  hovered,
  setHovered,
}: {
  agents: AgentRun[];
  clock: number | null;
  scale: Scale2;
  height: string;
  hovered: AgentId | null;
  setHovered: (id: AgentId | null) => void;
}) {
  const nodes = useMemo(() => layoutFlow(), []);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const planeRef = useRef<HTMLDivElement>(null);

  const edges = useMemo(() => {
    const list: { key: string; d: string; child: AgentRun; color: string }[] = [];
    for (const agent of agents) {
      const to = nodes.get(agent.id);
      if (!to) continue;
      for (const parentId of agent.parents) {
        const from = nodes.get(parentId);
        if (!from) continue;
        list.push({
          key: `${parentId}->${agent.id}`,
          d: edgePath(from, to),
          child: agent,
          color: AGENT_HEX[agent.id],
        });
      }
    }
    return list;
  }, [agents, nodes]);

  const onMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = planeRef.current?.getBoundingClientRect();
    if (!rect) return;
    setTilt({
      x: -((event.clientY - rect.top) / rect.height - 0.5) * 8,
      y: ((event.clientX - rect.left) / rect.width - 0.5) * 10,
    });
  };

  /** A node is dimmed when another node is hovered and this one is not adjacent. */
  const adjacency = useMemo(() => {
    if (!hovered) return null;
    const set = new Set<AgentId>([hovered]);
    for (const agent of agents) {
      if (agent.id === hovered) agent.parents.forEach((p) => set.add(p));
      if (agent.parents.includes(hovered)) set.add(agent.id);
    }
    return set;
  }, [hovered, agents]);

  return (
    <div
      className="relative"
      style={{ perspective: '900px' }}
      onMouseMove={onMove}
      onMouseLeave={() => {
        setTilt({ x: 0, y: 0 });
        setHovered(null);
      }}
    >
      <div
        ref={planeRef}
        className="relative w-full transition-transform duration-slow ease-expo-out"
        style={{
          height,
          transform: `rotateX(${14 + tilt.x}deg) rotateY(${tilt.y}deg)`,
          transformStyle: 'preserve-3d',
        }}
      >
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          className="absolute inset-0 h-full w-full overflow-visible"
          aria-hidden="true"
        >
          {edges.map((edge) => {
            const isActive = edgeActive(edge.child, clock);
            const dim = adjacency !== null && !adjacency.has(edge.child.id);
            return (
              <g key={edge.key} opacity={dim ? 0.2 : 1}>
                <path
                  d={edge.d}
                  fill="none"
                  stroke={isActive ? edge.color : '#1A2233'}
                  strokeWidth={isActive ? 1 : 0.8}
                  opacity={isActive ? 0.5 : 1}
                  vectorEffect="non-scaling-stroke"
                />
                {isActive && (
                  <path
                    d={edge.d}
                    fill="none"
                    stroke={edge.color}
                    strokeWidth={2.2}
                    strokeLinecap="round"
                    className="edge-flow"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </g>
            );
          })}
        </svg>

        {agents.map((agent) => {
          const node = nodes.get(agent.id);
          if (!node) return null;
          return (
            <Node
              key={agent.id}
              agent={agent}
              node={node}
              status={statusAt(agent, clock)}
              scale={scale}
              dimmed={adjacency !== null && !adjacency.has(agent.id)}
              onHover={setHovered}
            />
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Telemetry                                                           */
/* ------------------------------------------------------------------ */

/**
 * Pipeline telemetry, not hardware telemetry.
 *
 * CPU / GPU / RAM / temperature would be four numbers this page cannot measure
 * and cannot defend under questioning. These four it computed itself, and the
 * RPM cap in particular is the number that explains how a whole run fits inside
 * a free tier.
 */
function Telemetry({ clock, columns }: { clock: number | null; columns: string }) {
  const report = useReport();

  const calls = report.agents.reduce(
    (sum, a) => sum + (statusAt(a, clock) === 'PENDING' ? 0 : a.calls),
    0,
  );
  const elapsed = clock ?? totalRunMs(report.agents);
  const coverage = Math.round(report.meta.coverage * 100);
  const done = report.agents.filter((a) => statusAt(a, clock) !== 'PENDING').length;

  const cells = [
    { label: 'Agents', value: `${done}/${report.agents.length}` },
    { label: 'LLM Calls', value: String(calls) },
    { label: 'RPM Cap', value: '30' },
    {
      label: 'Coverage',
      value: `${coverage}%`,
      tone: coverage >= 90 ? SIGNAL_HEX.clear : SIGNAL_HEX.medium,
    },
    { label: 'Elapsed', value: `${(elapsed / 1000).toFixed(1)}s` },
  ];

  return (
    <div className={`grid gap-1.5 ${columns}`}>
      {cells.map((cell) => (
        <div
          key={cell.label}
          className="flex flex-col gap-0.5 rounded-chip border border-edge bg-panel px-2 py-1.5"
        >
          <span className="text-[7px] uppercase tracking-[0.08em] text-inkFaint">{cell.label}</span>
          <span className="num text-[11px]" style={{ color: cell.tone ?? '#E8EDF7' }}>
            {cell.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Expanded overlay                                                    */
/* ------------------------------------------------------------------ */

function ExpandedFlow({ onClose }: { onClose: () => void }) {
  const report = useReport();
  const { clock, play } = useReplay(report.agents);
  const [hovered, setHovered] = useState<AgentId | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const active = hovered ? report.agents.find((a) => a.id === hovered) : null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-void/95 backdrop-blur-sm">
      <header className="flex h-11 shrink-0 items-center justify-between border-b border-edge px-4">
        <div className="flex items-center gap-3">
          <span className="text-panel uppercase text-ink">Multi-Agent Flow</span>
          <span className="num text-[10px] text-inkFaint">
            {report.agents.length} agents · {report.video.filename}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={play}
            className="flex h-8 items-center gap-1.5 rounded-chip border border-edge px-2.5 text-[10px] uppercase tracking-[0.08em] text-inkDim transition-colors duration-instant hover:border-edgeHi hover:text-ink"
          >
            <RotateCw className="h-3 w-3" />
            replay analysis
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex h-8 w-8 items-center justify-center rounded-chip border border-edge text-inkDim transition-colors duration-instant hover:border-edgeHi hover:text-ink"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 gap-gutter p-gutter">
        <div className="flex min-w-0 flex-1 flex-col gap-gutter">
          <div className="flex min-h-0 flex-1 items-center justify-center rounded-panel border border-edge bg-panel px-8">
            <div className="w-full max-w-3xl">
              <FlowPlane
                agents={report.agents}
                clock={clock}
                scale={FULL}
                height="min(62vh, 560px)"
                hovered={hovered}
                setHovered={setHovered}
              />
            </div>
          </div>
          <Telemetry clock={clock} columns="grid-cols-5" />
        </div>

        <aside className="flex w-[320px] shrink-0 flex-col rounded-panel border border-edge bg-panel">
          <header className="flex h-9 shrink-0 items-center border-b border-edge px-4">
            <span className="text-panel uppercase text-inkDim">
              {active ? 'Agent' : 'Roster'}
            </span>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {report.agents.map((agent) => {
              const status = statusAt(agent, clock);
              const color = AGENT_HEX[agent.id];
              const isActive = hovered === agent.id;
              return (
                <div
                  key={agent.id}
                  onMouseEnter={() => setHovered(agent.id)}
                  onMouseLeave={() => setHovered(null)}
                  className={`border-b border-edge/60 px-4 py-2.5 transition-colors duration-instant ${
                    isActive ? 'bg-panelHi' : ''
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{
                        background: status === 'PENDING' ? '#26324A' : color,
                      }}
                    />
                    <span className="min-w-0 flex-1 truncate text-[12px] text-ink">
                      {agent.name}
                    </span>
                    <span
                      className="num shrink-0 text-[9px] uppercase"
                      style={{
                        color:
                          status === 'DEGRADED'
                            ? SIGNAL_HEX.medium
                            : status === 'RUNNING'
                              ? color
                              : '#4E5A70',
                      }}
                    >
                      {status}
                    </span>
                  </div>
                  <p className="mt-1 pl-3.5 text-[10px] leading-snug text-inkDim">{agent.detail}</p>
                  <div className="mt-1.5 flex gap-3 pl-3.5">
                    <span className="num text-[9px] text-inkFaint">
                      cov {Math.round(agent.coverage * 100)}%
                    </span>
                    <span className="num text-[9px] text-inkFaint">
                      {(agent.elapsedMs / 1000).toFixed(1)}s
                    </span>
                    <span className="num text-[9px] text-inkFaint">{agent.calls} calls</span>
                  </div>
                </div>
              );
            })}
          </div>
        </aside>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Rail panel                                                          */
/* ------------------------------------------------------------------ */

export function AgentFlow() {
  const report = useReport();
  const { clock, runMs } = useReplay(report.agents);
  const [hovered, setHovered] = useState<AgentId | null>(null);
  const [expanded, setExpanded] = useState(false);

  const active = hovered ? report.agents.find((a) => a.id === hovered) : null;
  const running = report.agents.filter((a) => statusAt(a, clock) === 'RUNNING');
  const completed = report.agents.filter((a) => {
    const s = statusAt(a, clock);
    return s === 'OK' || s === 'DEGRADED';
  }).length;

  return (
    <div className="shrink-0 border-t border-edge">
      <div className="flex items-center justify-between px-4 pb-1 pt-3">
        <span className="text-panel uppercase text-inkDim">Multi-Agent Flow</span>
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="flex items-center gap-1 text-[9px] uppercase tracking-[0.08em] text-inkFaint transition-colors duration-instant hover:text-ink"
        >
          <Expand className="h-2.5 w-2.5" />
          expand
        </button>
      </div>

      <div className="px-3">
        <FlowPlane
          agents={report.agents}
          clock={clock}
          scale={COMPACT}
          height="248px"
          hovered={hovered}
          setHovered={setHovered}
        />
      </div>

      <div className="mx-3 mt-1 min-h-[34px] rounded-panel border border-edge bg-abyss px-2.5 py-1.5">
        {active ? (
          <>
            <span className="num text-[9px]" style={{ color: AGENT_HEX[active.id] }}>
              {active.name}
            </span>
            <p className="num truncate text-[8.5px] text-inkFaint">{active.detail}</p>
          </>
        ) : running.length > 0 ? (
          <>
            <span className="num text-[9px]" style={{ color: AGENT_HEX[running[0]!.id] }}>
              {running.map((a) => a.name.replace(' Agent', '')).join(' · ')}
            </span>
            <p className="num truncate text-[8.5px] text-inkFaint">{running[0]!.detail}</p>
          </>
        ) : (
          <>
            <span className="num text-[9px]" style={{ color: SIGNAL_HEX.clear }}>
              Workflow complete
            </span>
            <p className="num truncate text-[8.5px] text-inkFaint">
              {completed}/{report.agents.length} agents · {(runMs / 1000).toFixed(1)}s
            </p>
          </>
        )}
      </div>

      <div className="border-t border-edge px-3 py-2.5">
        <div className="mb-1.5 text-[8px] uppercase tracking-[0.14em] text-inkFaint">
          Pipeline Telemetry
        </div>
        <Telemetry clock={clock} columns="grid-cols-5" />
      </div>

      {expanded && <ExpandedFlow onClose={() => setExpanded(false)} />}
    </div>
  );
}
