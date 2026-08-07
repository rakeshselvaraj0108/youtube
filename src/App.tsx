import { useEffect } from 'react';

import { AgentFlow } from '@/components/AgentFlow';
import { BeforeAfterPlayers } from '@/components/BeforeAfterPlayers';
import { DetailPanel } from '@/components/DetailPanel';
import { FfmpegBlock } from '@/components/FfmpegBlock';
import { FileCard } from '@/components/FileCard';
import { FindingsList } from '@/components/FindingsList';
import { HeaderBar } from '@/components/HeaderBar';
import { PolicyBreakdown } from '@/components/PolicyBreakdown';
import { RemediationPlan } from '@/components/RemediationPlan';
import { RiskTimeline } from '@/components/RiskTimeline';
import { RunBar } from '@/components/RunBar';
import { ScoreGauge } from '@/components/ScoreGauge';
import { SubScorePanel } from '@/components/SubScorePanel';
import { TerminalColumn } from '@/components/TerminalColumn';
import { useAnalysis } from '@/store/analysis';

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
  }, []);

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

          {/* Row 3 — findings, detail, remediation */}
          <div className="grid min-h-[420px] shrink-0 grid-cols-1 gap-gutter lg:grid-cols-2 2xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.35fr)_minmax(0,1.05fr)]">
            <FindingsList />
            <DetailPanel />
            <div className="flex min-h-0 min-w-0 flex-col gap-gutter lg:col-span-2 2xl:col-span-1">
              <RemediationPlan />
              <FfmpegBlock />
            </div>
          </div>

          {/* Row 4 — the proof */}
          <div className="grid shrink-0 grid-cols-1 gap-gutter lg:grid-cols-[minmax(0,1fr)_200px_minmax(0,1fr)]">
            <BeforeAfterPlayers />
          </div>
        </main>
      </div>
    </div>
  );
}
