import { useLiveStages, useReport, useRunning, type LiveStage } from '@/store/analysis';
import type { AgentRun } from '@/types/analysis';
import { SIGNAL_HEX } from '@/lib/scoring';
import { AGENT_HEX, elapsedStamp, STATUS_GLYPH } from '@/lib/agents';

/**
 * The terminal column.
 *
 * The log and the agent graph narrate the same twelve events in two languages —
 * the graph shows the dependency structure, the log shows the sequence. Phase 4
 * types these entries in; Phase 2 renders the finished stream.
 */

function LogEntry({ agent }: { agent: AgentRun }) {
  const tone = agent.status === 'OK' ? AGENT_HEX[agent.id] : SIGNAL_HEX.medium;
  return (
    <li className="flex gap-2 leading-relaxed">
      <span className="num shrink-0 text-code text-inkFaint">[{elapsedStamp(agent.tsMs)}]</span>
      <span className="num w-3 shrink-0 text-code" style={{ color: tone }}>
        {STATUS_GLYPH[agent.status]}
      </span>
      <span className="min-w-0 flex-1">
        <span className="num block text-code font-medium text-ink">{agent.name}</span>
        <span className="num block text-code text-inkDim">{agent.detail}</span>
      </span>
    </li>
  );
}

function LiveLogEntry({ stage }: { stage: LiveStage }) {
  const tone = stage.status === 'FAILED' ? SIGNAL_HEX.critical : stage.status === 'DEGRADED' ? SIGNAL_HEX.medium : SIGNAL_HEX.clear;
  return (
    <li className="flex gap-2 leading-relaxed">
      <span className="num shrink-0 text-code text-inkFaint">{stage.elapsedMs === undefined ? 'LIVE' : `${(stage.elapsedMs / 1000).toFixed(1)}s`}</span>
      <span className="num w-3 shrink-0 text-code" style={{ color: tone }}>{stage.status === 'RUNNING' ? '…' : stage.status === 'FAILED' ? '!' : '✓'}</span>
      <span className="min-w-0 flex-1">
        <span className="num block text-code font-medium text-ink">{stage.name}</span>
        <span className="num block text-code text-inkDim">{stage.detail ?? stage.status}</span>
      </span>
    </li>
  );
}

export function TerminalColumn() {
  const report = useReport();
  const running = useRunning();
  const live = useLiveStages();
  const findings = report.findings.length;
  const bySeverity = (s: string) => report.findings.filter((f) => f.severity === s).length;
  const failed = report.agents.filter((agent) => agent.status === 'FAILED').length;
  const observedStages = Object.keys(live).length;
  const stateLabel = running
    ? `Workflow running${observedStages ? ` · ${observedStages} stage${observedStages === 1 ? '' : 's'} observed` : ''}`
    : failed > 0
      ? `Workflow completed with ${failed} failed agent${failed === 1 ? '' : 's'}`
      : 'Workflow completed';

  return (
    <aside className="flex h-full min-h-0 w-full flex-col border-r border-edge bg-abyss">
      {/* z-10 keeps the header above the scrolling log — without it the first
          log line rides up over the title and reads as clipped. */}
      <header className="relative z-10 flex h-11 shrink-0 items-center gap-2 border-b border-edge bg-abyss px-4">
        <span className="relative flex h-1.5 w-1.5">
          <span
            className="absolute inline-flex h-full w-full rounded-full opacity-60"
            style={{ background: running ? SIGNAL_HEX.medium : failed > 0 ? SIGNAL_HEX.critical : SIGNAL_HEX.clear }}
          />
        </span>
        <span className="text-panel uppercase text-inkDim">Terminal</span>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        <p className="num mb-4 text-code">
          <span className="text-inkFaint">$ </span>
          <span style={{ color: AGENT_HEX.policy }}>preflight</span>
          <span className="text-ink"> check {report.video.filename} </span>
          <span style={{ color: SIGNAL_HEX.medium }}>--html</span>
        </p>

        <p className="num mb-3 text-code text-inkFaint">
          {running ? 'Live server events are being recorded' : 'Recorded report state'}
        </p>

        <ul className="flex flex-col gap-3">
          {running
            ? Object.values(live).length > 0
              ? Object.values(live).map((stage) => <LiveLogEntry key={stage.stage} stage={stage} />)
              : <li className="num text-code text-inkFaint">Waiting for the first server stage…</li>
            : report.agents.map((agent) => <LogEntry key={agent.id} agent={agent} />)}
        </ul>

        <p className="num mt-4 text-code" style={{ color: running ? SIGNAL_HEX.medium : failed > 0 ? SIGNAL_HEX.critical : SIGNAL_HEX.clear }}>
          [{running ? '…' : failed > 0 ? '!' : '✓'}] {stateLabel}
        </p>

        <div className="mt-4 rounded-panel border border-edge p-3">
          <div className="text-label mb-2 uppercase text-inkFaint">Result</div>
          {running ? (
            <dl className="num flex flex-col gap-1 text-code">
              <Row k="STAGES" v={`${Object.keys(live).length} observed`} />
              <Row k="COVERAGE" v="NOT MEASURED" />
              <Row k="FINDINGS" v="NOT MEASURED" />
              <Row k="SCORE" v="NOT MEASURED" />
            </dl>
          ) : (
            <dl className="num flex flex-col gap-1 text-code">
              <Row k="READINESS" v={`${report.scores.overall} / 100`} />
              <Row k="VERDICT" v={report.scores.verdict.replace(/_/g, ' ')} />
              <Row k="FINDINGS" v={`${findings} (${bySeverity('CRITICAL')}C ${bySeverity('HIGH')}H ${bySeverity('MEDIUM')}M ${bySeverity('LOW')}L)`} />
              <Row k="COVERAGE" v={`${Math.round(report.meta.coverage * 100)}%`} />
              <Row k="OPS" v={String(report.remediation.ops.length)} />
            </dl>
          )}
        </div>

        <p className="num mt-4 text-code text-inkFaint">
          $ <span className="text-ink">_</span>
        </p>
      </div>
    </aside>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex gap-2">
      <dt className="w-24 shrink-0 text-inkFaint">{k}</dt>
      <dd className="min-w-0 flex-1 truncate text-inkDim">: {v}</dd>
    </div>
  );
}
