import { Download, ShieldCheck } from 'lucide-react';
import { Chip } from '@/components/ui';
import { useReport } from '@/store/analysis';
import { degradedAgents } from '@/lib/coverage';
import { SIGNAL_HEX } from '@/lib/scoring';
import { buildSarif, exitCode } from '@/lib/sarif';
import { buildCertificate } from '@/lib/certificate';
import { downloadJson } from '@/lib/download';

/** The falcon mark. Drawn, not an emoji, not an imported asset. */
function Falcon({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 16" className={className} fill="none" aria-hidden="true">
      <path d="M1 4.5 L11.2 7.4 L12 3 L12.8 7.4 L23 4.5 L14.4 9.6 L12 15 L9.6 9.6 Z" fill="currentColor" />
    </svg>
  );
}

export function HeaderBar() {
  const report = useReport();
  const degraded = degradedAgents(report.agents);
  const coveragePct = Math.round(report.meta.coverage * 100);

  const coverageTitle = [
    `Analysis surface covered: ${coveragePct}%`,
    ...report.agents.map(
      (a) => `${a.status === 'OK' ? '·' : '!'} ${a.name} — ${Math.round(a.coverage * 100)}% (${a.status})`,
    ),
  ].join('\n');

  return (
    <header className="flex h-11 shrink-0 items-center justify-between gap-4 border-b border-edge px-4">
      <div className="flex items-center gap-3">
        <Falcon className="h-4 w-6 text-ink" />
        <span
          className="text-[13px] font-semibold text-ink"
          style={{ letterSpacing: '0.14em' }}
        >
          PREFLIGHT
        </span>
        <Chip>CLI</Chip>
        <span className="ml-1 hidden text-micro uppercase tracking-[0.18em] text-inkFaint lg:inline">
          Analysis Report
        </span>
      </div>

      <div className="flex items-center gap-2">
        <Chip
          tone={coveragePct >= 90 ? undefined : SIGNAL_HEX.medium}
          title={coverageTitle}
          className="cursor-help"
        >
          Coverage {coveragePct}%
          {degraded.length > 0 && (
            <span className="normal-case tracking-normal opacity-80">
              · {degraded[0]!.name.replace(' Agent', '').toLowerCase()} degraded
            </span>
          )}
        </Chip>

        <Chip title={`Policy corpus version ${report.meta.policyVersion}`}>
          Policy {report.meta.policyVersion}
        </Chip>

        {/* Predicted before the run, measured after it. The estimate is an
            upper bound by construction, so actual exceeding it would mean
            the plan is wrong rather than the run unlucky — which makes this
            a check on PREFLIGHT rather than a statistic about the video. */}
        {report.cost && (
          <Chip
            title={
              `Plan estimated at most ${report.cost.estimatedCalls} hosted calls; ` +
              `the run made ${report.cost.actualCalls}.` +
              (report.cost.ceiling !== null
                ? `\nBudget ceiling: ${report.cost.ceiling}`
                : '\nNo budget ceiling was set.') +
              (report.cost.shed.length > 0
                ? `\nShed: ${report.cost.shed.map((s) => s.stage).join(', ')}`
                : '')
            }
          >
            {report.cost.actualCalls}/{report.cost.estimatedCalls} calls
          </Chip>
        )}

        <Chip
          tone={exitCode(report) === 0 ? SIGNAL_HEX.clear : SIGNAL_HEX.critical}
          title="Exit code a CI run would take from this report"
        >
          exit {exitCode(report)}
        </Chip>

        <div className="mx-1 h-4 w-px bg-edge" />

        <GhostButton
          icon={<Download className="h-3 w-3" />}
          label="report.sarif"
          title="Download SARIF 2.1.0 — renders natively in GitHub's Security tab"
          onClick={() => downloadJson('preflight.sarif', buildSarif(report))}
        />
        <GhostButton
          icon={<ShieldCheck className="h-3 w-3" />}
          label="certificate.json"
          title="Download the release certificate — what was checked, against which rules, with which models"
          onClick={() => downloadJson('preflight-certificate.json', buildCertificate(report))}
        />
      </div>
    </header>
  );
}

function GhostButton({
  icon,
  label,
  title,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  title: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="flex h-8 items-center gap-1.5 rounded-chip border border-edge px-2.5 text-data text-inkDim transition-colors duration-instant hover:border-edgeHi hover:bg-panelHi hover:text-ink"
    >
      {icon}
      <span className="num">{label}</span>
    </button>
  );
}
