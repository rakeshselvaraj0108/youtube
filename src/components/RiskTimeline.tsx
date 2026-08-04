import { Panel, Dot, TONE_LABELS } from '@/components/ui';
import { useAnalysis, useReport, useSelectedFinding } from '@/store/analysis';
import { riskHex, severityHex, SIGNAL_HEX } from '@/lib/scoring';
import { localisedFindings } from '@/lib/risk';
import { formatTimecode, timelineTicks } from '@/lib/time';

/**
 * Risk timeline.
 *
 * Every segment height is max(severity × confidence) over the findings that
 * overlap it — the terrain is computed, never drawn. Ticks are generated from
 * the runtime and never run past it.
 *
 * Phase 5 extrudes this into 3D topography; the DOM structure here is already
 * the one that will carry the translateZ.
 */
export function RiskTimeline() {
  const report = useReport();
  const selected = useSelectedFinding();
  const hoveredOpIndex = useAnalysis((s) => s.hoveredOpIndex);
  const select = useAnalysis((s) => s.select);

  const duration = report.video.durationMs;
  const ticks = timelineTicks(duration);
  const pins = localisedFindings(report.findings, duration);

  const hoveredOp = report.remediation.ops.find((o) => o.index === hoveredOpIndex);

  /** Select whichever finding is nearest the clicked point on the timeline. */
  const selectNearest = (t: number) => {
    if (pins.length === 0) return;
    const ms = t * duration;
    let best = pins[0]!;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const finding of pins) {
      // Zero distance anywhere inside the finding's own span.
      const distance =
        ms < finding.startMs
          ? finding.startMs - ms
          : ms > finding.endMs
            ? ms - finding.endMs
            : 0;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = finding;
      }
    }
    select(best.id);
  };

  return (
    <Panel
      title="Risk Timeline"
      aside={
        <span className="num text-[10px] text-inkFaint">
          {report.riskBands.length} segments · {formatTimecode(duration)}
        </span>
      }
      className="min-w-0"
    >
      <div className="flex h-full flex-col justify-between gap-3">
        {/* pins */}
        <div className="relative h-6 shrink-0">
          {pins.map((f) => {
            const t = ((f.startMs + f.endMs) / 2 / duration) * 100;
            const isSelected = selected?.id === f.id;
            const tone = severityHex(f.severity);
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => select(f.id)}
                aria-label={`${formatTimecode(f.startMs)} — ${f.title}`}
                title={`${formatTimecode(f.startMs)} — ${f.title}`}
                className="absolute bottom-0 flex h-full w-8 -translate-x-1/2 cursor-pointer flex-col items-center justify-end"
                style={{ left: `${t}%` }}
              >
                {isSelected && (
                  <span className="num mb-0.5 whitespace-nowrap text-[9px]" style={{ color: tone }}>
                    {formatTimecode(f.startMs)}
                  </span>
                )}
                <span
                  className="rounded-[1px] transition-all duration-instant"
                  style={{
                    width: isSelected ? 3 : 2,
                    height: isSelected ? 12 : 7,
                    background: tone,
                  }}
                />
              </button>
            );
          })}
        </div>

        {/* terrain */}
        <div
          className="relative cursor-pointer"
          onClick={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            selectNearest((event.clientX - rect.left) / rect.width);
          }}
        >
          <div className="flex h-12 w-full items-end gap-px overflow-hidden rounded-bar bg-abyss">
            {report.riskBands.map((band) => {
              const height = 22 + band.risk * 78; // never zero — the track stays readable
              return (
                <div
                  key={band.startMs}
                  className="flex-1"
                  style={{
                    height: `${height}%`,
                    background: band.risk === 0 ? '#141B29' : riskHex(band.risk),
                    opacity: band.risk === 0 ? 1 : 0.55 + band.risk * 0.45,
                  }}
                  title={`${formatTimecode(band.startMs)} · risk ${(band.risk * 100).toFixed(0)}%`}
                />
              );
            })}
          </div>

          {/* selected-finding guide line */}
          {selected && selected.endMs - selected.startMs < duration * 0.5 && (
            <div
              className="pointer-events-none absolute inset-y-0 w-px"
              style={{
                left: `${((selected.startMs + selected.endMs) / 2 / duration) * 100}%`,
                background: severityHex(selected.severity),
                opacity: 0.7,
              }}
            />
          )}

          {/* hovered remediation op span */}
          {hoveredOp && (
            <div
              className="pointer-events-none absolute inset-y-0 border-x"
              style={{
                left: `${(hoveredOp.startMs / duration) * 100}%`,
                width: `${Math.max(0.4, ((hoveredOp.endMs - hoveredOp.startMs) / duration) * 100)}%`,
                borderColor: SIGNAL_HEX.clear,
                background: `${SIGNAL_HEX.clear}26`,
              }}
            />
          )}
        </div>

        {/* axis */}
        <div className="relative h-3 shrink-0">
          {ticks.map((tick) => (
            <span
              key={tick.ms}
              className="num absolute top-0 -translate-x-1/2 text-[9px] text-inkFaint"
              style={{ left: `${tick.t * 100}%` }}
            >
              {tick.label}
            </span>
          ))}
          <span className="num absolute right-0 top-0 text-[9px] text-inkFaint">
            {formatTimecode(duration)}
          </span>
        </div>

        {/* legend */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-edge pt-2.5">
          {TONE_LABELS.map(([tone, label]) => (
            <span key={tone} className="flex items-center gap-1.5">
              <Dot tone={SIGNAL_HEX[tone]} />
              <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">{label}</span>
            </span>
          ))}
        </div>
      </div>
    </Panel>
  );
}
