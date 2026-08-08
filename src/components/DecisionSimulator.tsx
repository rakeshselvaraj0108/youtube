import { ChevronRight } from 'lucide-react';

import { Panel } from '@/components/ui';
import { useAnalysis, useReport } from '@/store/analysis';
import type { Scenario } from '@/types/analysis';

/**
 * DECISION SIMULATOR — what each repair is worth, before committing to one.
 *
 * Collapsed by default. This is the decision layer, not the reporting layer:
 * a reader consults it when choosing what to fix, and it should not consume
 * the deck the rest of the time.
 *
 * Everything shown is PREDICTED. The engine computes each scenario by
 * removing the evidence an edit destroys and re-scoring with the same
 * `sub_scores` and `compute_readiness` that produced the current number, so
 * the prediction and the headline cannot drift — but a prediction is still
 * not a render. The state column says so, and only says RENDERED after
 * ffmpeg has actually written a file.
 *
 * Deliberately absent, because the backend does not produce them: prediction
 * confidence, assumptions, unknowns, and prediction accuracy. Those read as
 * NOT COMPUTABLE rather than as numbers, which is the honest answer — a
 * fabricated confidence would be the single most misleading thing this panel
 * could show.
 */

function Delta({ value }: { value: number }) {
  if (value === 0) return <span className="num text-[10px] text-inkFaint">±0</span>;
  return (
    <span
      className="num text-[10px]"
      style={{ color: value > 0 ? '#7BE3A8' : '#E38B7B' }}
    >
      {value > 0 ? '+' : ''}
      {value}
    </span>
  );
}

function ScenarioRow({
  scenario,
  baseline,
  isBest,
}: {
  scenario: Scenario;
  baseline: Scenario;
  isBest: boolean;
}) {
  const selected = useAnalysis((s) => s.selectedScenario);
  const selectScenario = useAnalysis((s) => s.selectScenario);
  const isSelected = selected === scenario.name;

  // A scenario the compiler would refuse to render is not a recommendation.
  // Showing it without saying so would offer advice the rest of the system
  // cannot carry out.
  const overCeiling = scenario.impact > 0.45;

  return (
    <button
      type="button"
      onClick={() => selectScenario(isSelected ? null : scenario.name)}
      title={
        `${scenario.name}\n` +
        `Predicted readiness ${baseline.overall} → ${scenario.overall}\n` +
        `Viewer impact ${scenario.impact.toFixed(2)} on the remediation ` +
        `compiler's own scale` +
        (overCeiling
          ? '\nAbove the renderable ceiling — the compiler would refuse this set'
          : '') +
        (scenario.gatedBy
          ? `\nRepaired something real but the score is held by the ${scenario.gatedBy} dimension`
          : '')
      }
      className={`grid w-full grid-cols-[minmax(0,1fr)_44px_38px_46px_58px] items-center gap-2 rounded-chip px-2 py-1.5 text-left transition-colors duration-instant ${
        isSelected ? 'bg-panelHi' : 'hover:bg-panelHi/60'
      }`}
    >
      <span className="flex min-w-0 items-center gap-1.5">
        <span className="min-w-0 truncate text-[11px] text-inkDim">{scenario.name}</span>
        {isBest && (
          <span className="num shrink-0 rounded-chip border border-edge px-1 text-[8px] text-inkFaint">
            best
          </span>
        )}
      </span>

      <span className="num text-right text-[11px] text-ink">{scenario.overall}</span>
      <span className="text-right">
        <Delta value={scenario.delta} />
      </span>
      <span
        className="num text-right text-[10px] text-inkFaint"
        title="Cumulative viewer impact — what the edit costs the video"
      >
        {scenario.impact.toFixed(2)}
      </span>
      <span
        className={`num text-right text-[9px] uppercase tracking-[0.06em] ${
          overCeiling ? 'text-inkFaint/60' : 'text-inkFaint'
        }`}
      >
        {overCeiling ? 'refused' : 'predicted'}
      </span>
    </button>
  );
}

export function DecisionSimulator() {
  const report = useReport();
  const open = useAnalysis((s) => s.simulatorOpen);
  const toggle = useAnalysis((s) => s.toggleSimulator);
  const selected = useAnalysis((s) => s.selectedScenario);

  const simulation = report.simulation;
  const scenarios = simulation?.scenarios ?? [];
  const baseline = simulation?.baseline;
  const chosen = scenarios.find((s) => s.name === selected);

  return (
    <Panel
      title="Decision Simulator"
      className="min-w-0"
      aside={
        <button
          type="button"
          onClick={toggle}
          aria-expanded={open}
          title={open ? 'Collapse' : 'Explore what each repair is worth'}
          className="flex items-center gap-1 text-[10px] text-inkFaint transition-colors duration-instant hover:text-ink"
        >
          {open ? 'collapse' : `${scenarios.length} scenario${scenarios.length === 1 ? '' : 's'}`}
          <ChevronRight
            className="h-3 w-3 transition-transform duration-instant"
            style={{ transform: open ? 'rotate(90deg)' : 'none' }}
          />
        </button>
      }
    >
      {!open ? (
        <p className="text-[11px] text-inkFaint">
          {baseline
            ? `Current readiness ${baseline.overall}. Expand to see what each repair would be worth before committing to one.`
            : 'No simulation in this report.'}
        </p>
      ) : !simulation || !baseline ? (
        <p className="text-[11px] text-inkFaint">
          PREDICTION UNAVAILABLE — this report carries no simulation. It was
          emitted before the engine computed scenarios.
        </p>
      ) : (
        <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
          <div className="grid grid-cols-[minmax(0,1fr)_44px_38px_46px_58px] gap-2 border-b border-edge px-2 pb-1 text-[9px] uppercase tracking-[0.06em] text-inkFaint">
            <span>scenario</span>
            <span className="text-right">score</span>
            <span className="text-right">Δ</span>
            <span className="text-right">impact</span>
            <span className="text-right">state</span>
          </div>

          <div className="grid grid-cols-[minmax(0,1fr)_44px_38px_46px_58px] items-center gap-2 px-2 py-1.5">
            <span className="text-[11px] text-inkDim">current video</span>
            <span className="num text-right text-[11px] text-ink">{baseline.overall}</span>
            <span className="num text-right text-[10px] text-inkFaint">—</span>
            <span className="num text-right text-[10px] text-inkFaint">0.00</span>
            <span className="num text-right text-[9px] uppercase tracking-[0.06em] text-inkFaint">
              measured
            </span>
          </div>

          {scenarios.length === 0 ? (
            <p className="px-2 text-[11px] text-inkFaint">
              No remediable findings, so there is nothing to simulate.
            </p>
          ) : (
            scenarios.map((s) => (
              <ScenarioRow
                key={s.name}
                scenario={s}
                baseline={baseline}
                isBest={s.name === simulation.best}
              />
            ))
          )}

          {chosen && (
            <div className="mt-1 flex flex-col gap-1.5 border-t border-edge pt-2">
              <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
                current vs predicted
              </span>
              <div className="flex flex-wrap gap-x-5 gap-y-1">
                <Compare
                  label="readiness"
                  from={baseline.overall}
                  to={chosen.overall}
                />
                <Compare
                  label="findings"
                  from={baseline.survivingFindings}
                  to={chosen.survivingFindings}
                />
              </div>
              {chosen.removedFindingIds.length > 0 && (
                <span className="text-[10px] text-inkFaint">
                  clears: {chosen.removedFindingIds.join(', ')}
                </span>
              )}
              {chosen.gatedBy && (
                <span className="text-[10px] text-inkFaint">
                  repairs something real, but the score is held at the{' '}
                  {chosen.gatedBy} dimension until that is addressed
                </span>
              )}
              <span className="text-[10px] text-inkFaint">
                prediction confidence: NOT COMPUTABLE — the engine scores the
                outcome but does not estimate its own reliability
              </span>
            </div>
          )}

          <p className="mt-1 border-t border-edge pt-2 text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
            predicted by removing the evidence each edit destroys and
            re-scoring with the same scorer · nothing here has been rendered
          </p>
        </div>
      )}
    </Panel>
  );
}

function Compare({ label, from, to }: { label: string; from: number; to: number }) {
  const delta = to - from;
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">{label}</span>
      <span className="num text-[11px] text-inkFaint">{from}</span>
      <span className="text-[10px] text-inkFaint">→</span>
      <span className="num text-[11px] text-ink">{to}</span>
      <Delta value={delta} />
    </span>
  );
}
