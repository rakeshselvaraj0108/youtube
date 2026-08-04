import { ArrowRight, AudioLines, Music4, Scissors, SquareDashedBottom, VolumeX } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Panel } from '@/components/ui';
import { useAnalysis, useReport } from '@/store/analysis';
import { OP_LABELS } from '@/lib/ffmpeg';
import { formatTimecode } from '@/lib/time';
import type { OpKind } from '@/types/analysis';

const OP_ICON: Record<OpKind, LucideIcon> = {
  BLEEP: AudioLines,
  MUTE: VolumeX,
  BLUR_REGION: SquareDashedBottom,
  REPLACE_AUDIO: Music4,
  CUT: Scissors,
};

export function RemediationPlan() {
  const report = useReport();
  const setHoveredOp = useAnalysis((s) => s.setHoveredOp);
  const ops = report.remediation.ops;

  return (
    <Panel
      title="Remediation Plan"
      aside={
        <span className="num text-[10px] text-inkFaint">
          {ops.length} {ops.length === 1 ? 'operation' : 'operations'}
        </span>
      }
      className="min-w-0"
      flush
    >
      {ops.length === 0 ? (
        <div className="flex flex-1 items-center justify-center px-4 py-6">
          <span className="num text-center text-[10px] uppercase leading-relaxed tracking-[0.1em] text-inkFaint">
            plan applied
            <br />
            no outstanding operations
          </span>
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-edge">
                {['#', 'Start', 'End', 'Action', 'Details'].map((h) => (
                  <th
                    key={h}
                    className="px-2 py-2 text-label uppercase text-inkFaint first:pl-4 last:pr-4"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ops.map((op) => {
                const Icon = OP_ICON[op.op];
                return (
                  <tr
                    key={op.index}
                    onMouseEnter={() => setHoveredOp(op.index)}
                    onMouseLeave={() => setHoveredOp(null)}
                    className="border-b border-edge/60 transition-colors duration-instant last:border-b-0 hover:bg-panelHi"
                  >
                    <td className="num px-2 py-2 pl-4 text-[10px] text-inkFaint">{op.index}</td>
                    <td className="num px-2 py-2 text-data text-inkDim">
                      {formatTimecode(op.startMs)}
                    </td>
                    <td className="num px-2 py-2 text-data text-inkDim">
                      {formatTimecode(op.endMs)}
                    </td>
                    <td className="px-2 py-2">
                      <span className="flex items-center gap-1.5 whitespace-nowrap text-[11px] text-ink">
                        <Icon className="h-3 w-3 shrink-0 text-inkFaint" strokeWidth={1.6} />
                        {OP_LABELS[op.op]}
                      </span>
                    </td>
                    <td className="px-2 py-2 pr-4 text-[11px] text-inkDim">{op.details}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex shrink-0 items-center justify-between border-t border-edge px-4 py-2">
        <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
          lowered from {ops.length} of {report.findings.length} findings
        </span>
        <button
          type="button"
          className="flex items-center gap-1 text-[10px] text-inkDim transition-colors duration-instant hover:text-ink"
        >
          edl.json
          <ArrowRight className="h-3 w-3" />
        </button>
      </div>
    </Panel>
  );
}
