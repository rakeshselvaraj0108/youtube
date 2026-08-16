import { useEffect } from 'react';

import { AgentFlow } from '@/components/AgentFlow';
import { BeforeAfterPlayers } from '@/components/BeforeAfterPlayers';
import { CoverageMap } from '@/components/CoverageMap';
import { DetailPanel } from '@/components/DetailPanel';
import { EvidencePanel } from '@/components/EvidencePanel';
import { FfmpegBlock } from '@/components/FfmpegBlock';
import { FileCard } from '@/components/FileCard';
import { FindingsList } from '@/components/FindingsList';
import { HeaderBar } from '@/components/HeaderBar';
import { PolicyBreakdown } from '@/components/PolicyBreakdown';
import { RemediationPlan } from '@/components/RemediationPlan';
import { DecisionSimulator } from '@/components/DecisionSimulator';
import { IncidentsPanel } from '@/components/IncidentsPanel';
import { RiskTimeline } from '@/components/RiskTimeline';
import { RunBar } from '@/components/RunBar';
import { ScoreGauge } from '@/components/ScoreGauge';
import { SubScorePanel } from '@/components/SubScorePanel';
import { TerminalColumn } from '@/components/TerminalColumn';
import { VerificationPanel } from '@/components/VerificationPanel';
import { useAnalysis } from '@/store/analysis';
import { Panel } from '@/components/ui';

/**
 * The Command Deck.
 *
 * Full viewport, no page scroll — the right column scrolls internally. Track
 * sizes are fractional so every panel reflows rather than clipping; nothing is
 * positioned in absolute pixels.
 */
export default function App() {
  // Ask the engine for its newest run once, at startup. Fails soft: when
  // `preflight serve` is not running — the normal case for a report.html
  // opened from an email — the deck keeps whatever it already had.
  useEffect(() => {
    void useAnalysis.getState().hydrate();
    // Ask what the engine has on disk, so a remediation a previous process
    // left half-finished is surfaced rather than silently forgotten.
    void useAnalysis.getState().loadRemediations();
  }, []);

  const analysisStatus = useAnalysis((s) => s.analysisStatus);
  const analysisError = useAnalysis((s) => s.analysisError);
  const source = useAnalysis((s) => s.source);
  const hasAnalysis = source !== 'fixture' && analysisStatus === 'COMPLETED';

  if (analysisStatus === 'QUEUED' || analysisStatus === 'RUNNING') {
    return <LiveWorkspace />;
  }

  if (!hasAnalysis) {
    return <IdleWorkspace status={analysisStatus} error={analysisError} />;
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-void lg:flex-row">
      {/* Terminal rail — a collapsible strip above the deck under 1024px. */}
      <div className="flex max-h-[38vh] shrink-0 flex-col overflow-hidden border-b border-edge lg:h-full lg:max-h-none lg:w-[340px] lg:border-b-0">
        <div className="flex min-h-0 flex-1 flex-col">
          <TerminalColumn />
        </div>
        <div className="hidden lg:block">
          <AgentFlow />
        </div>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <HeaderBar />
        <RunBar />

        <main className="flex min-h-0 flex-1 flex-col gap-gutter overflow-y-auto p-gutter">
          {/* Row 1 — file, gauge, dimensions */}
          <div className="grid shrink-0 grid-cols-1 gap-gutter md:grid-cols-2 xl:grid-cols-[minmax(0,1.55fr)_minmax(0,0.62fr)_minmax(0,0.72fr)]">
            <FileCard />
            <ScoreGauge />
            <SubScorePanel />
          </div>

          {/* Row 2 — timeline, breakdown */}
          <div className="grid shrink-0 grid-cols-1 gap-gutter xl:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
            <RiskTimeline />
            <PolicyBreakdown />
          </div>

          {/* Row 2a — coverage. Directly under the risk timeline on purpose:
              the timeline shows where risk was found, and this shows where
              anything actually looked. Reading the first without the second
              is how an unexamined stretch gets mistaken for a clean one. */}
          <div className="grid shrink-0 grid-cols-1 gap-gutter">
            <CoverageMap />
          </div>

          {/* Row 2b — incidents: the correlation layer. Placed between the
              timeline and the findings so the page reads what-happened before
              what-each-agent-observed. */}
          <div className="grid shrink-0 grid-cols-1 gap-gutter">
            <IncidentsPanel />
          </div>

          {/* Row 3 — findings, detail, remediation */}
          <div className="grid min-h-[420px] shrink-0 grid-cols-1 gap-gutter lg:grid-cols-2 2xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.35fr)_minmax(0,1.05fr)]">
            <FindingsList />
            <DetailPanel />
            <div className="flex min-h-0 min-w-0 flex-col gap-gutter lg:col-span-2 2xl:col-span-1">
              <RemediationPlan />
              <DecisionSimulator />
              <FfmpegBlock />
            </div>
          </div>

          {/* Row 4 — the proof */}
          <div className="grid shrink-0 grid-cols-1 gap-gutter lg:grid-cols-[minmax(0,1fr)_200px_minmax(0,1fr)]">
            <BeforeAfterPlayers />
          </div>

          {/* Row 5 — the closed loop's own record. Last because it is the
              conclusion: everything above describes the video, and this
              describes what changed after acting on it. Both panels render an
              explicit not-verified state until a remediation has actually
              been rendered and re-analysed, so the deck never implies a
              verdict it does not have. */}
          <div className="grid shrink-0 grid-cols-1 gap-gutter xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
            <VerificationPanel />
            <EvidencePanel />
          </div>
        </main>
      </div>
    </div>
  );
}

function LiveWorkspace() {
  const live = useAnalysis((s) => s.live);
  const completed = Object.values(live).filter((stage) => stage.status !== 'RUNNING').length;
  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-void lg:flex-row">
      <div className="flex max-h-[52vh] shrink-0 flex-col overflow-hidden border-b border-edge lg:h-full lg:max-h-none lg:w-[340px] lg:border-b-0">
        <div className="flex min-h-0 flex-1 flex-col"><TerminalColumn /></div>
        <div className="hidden lg:block"><AgentFlow /></div>
      </div>
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <HeaderBar />
        <RunBar />
        <main className="flex min-h-0 flex-1 items-center justify-center p-gutter">
          <Panel className="w-full max-w-2xl">
            <div className="flex flex-col items-center gap-3 py-10 text-center">
              <span className="text-panel uppercase tracking-[0.14em] text-ink">Live agent execution</span>
              <span className="num text-[11px] uppercase tracking-[0.1em] text-inkFaint">
                {Object.keys(live).length === 0 ? 'Waiting for server stages' : `${completed}/${Object.keys(live).length} observed stages settled`}
              </span>
              <p className="max-w-md text-[11px] leading-relaxed text-inkDim">
                The terminal and graph are fed by the current job’s SSE events. Scores, findings, and remediation remain unavailable until the backend emits a completed report.
              </p>
            </div>
          </Panel>
        </main>
      </div>
    </div>
  );
}

/** Pre-analysis is a real product state, not an analysis report with zeros.
 * Keep the selected/default video visible while making the required user
 * action unambiguous; no agent, score, finding, or coverage panel is mounted
 * until the backend has accepted an explicit ANALYZE request. */
function IdleWorkspace({
  status,
  error,
}: {
  status: 'IDLE' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  error: string | null;
}) {
  const active = status === 'QUEUED' || status === 'RUNNING';
  const failed = status === 'FAILED';
  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-void">
      <HeaderBar />
      <RunBar />
      <main className="flex min-h-0 flex-1 items-center justify-center p-gutter">
        <Panel className="w-full max-w-2xl">
          <div className="flex flex-col items-center gap-4 py-10 text-center">
            <span className="text-panel uppercase tracking-[0.14em] text-ink">
              {active ? 'PREFLIGHT ANALYSIS IN PROGRESS' : failed ? 'PREFLIGHT ANALYSIS FAILED' : 'PREFLIGHT READY'}
            </span>
            <p className="num text-[11px] uppercase tracking-[0.1em] text-inkFaint">
              {active
                ? status === 'QUEUED' ? 'Request accepted — connecting to the analysis stream.' : 'Agents are executing — waiting for measured report data.'
                : failed ? error ?? 'The server did not provide an error detail.' : 'Video ready — analysis has not been requested.'}
            </p>
            <p className="max-w-md text-[11px] leading-relaxed text-inkDim">
              {active
                ? 'Scores, findings, and coverage remain unavailable until the backend produces a report.'
                : 'Choose a video or enter a path above, then press ANALYSE. Loading, playing, or selecting a video never starts the policy pipeline.'}
            </p>
            <span className="num rounded-chip border border-edge px-3 py-1.5 text-[9px] uppercase tracking-[0.1em] text-inkFaint">
              {active ? `${status} · NOT MEASURED` : failed ? 'FAILED · NO REPORT PRODUCED' : 'IDLE · NOT MEASURED'}
            </span>
          </div>
        </Panel>
      </main>
    </div>
  );
}
