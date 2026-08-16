import type { ReasoningChain } from '@/types/analysis';

/**
 * The cited reasoning chain, rendered wherever an incident needs to explain
 * itself.
 *
 * Extracted rather than left inside `DetailPanel`: the incidents panel needs
 * the identical rendering — same step grouping, same citation format, same
 * dismissed/unresolved sections — and a second copy of this logic would be
 * exactly the duplicate-architecture risk the engine already paid once to
 * avoid (the reasoning engine itself was rewritten to stop three separate
 * modules from re-deriving "what does this claim cite").
 *
 * Every claim shown here was constructed with a source at build time —
 * `preflight.scoring.reasoning.Claim` raises `UnsourcedClaim` without one —
 * so there is nothing in this component that invents an explanation; it only
 * lays out what the engine already refused to emit unsourced.
 */

export const STEP_LABELS: Record<string, string> = {
  observation: 'Observation',
  evidence: 'Evidence',
  policy: 'Policy',
  risk_argument: 'Argument for risk',
  counter_argument: 'Argument against',
  decision: 'Decision',
  uncertainty: 'Remaining uncertainty',
};

export function ReasoningChainView({
  chain,
  compact = false,
}: {
  chain: ReasoningChain;
  compact?: boolean;
}) {
  const ordered = Object.keys(STEP_LABELS);
  const grouped = ordered
    .map((step) => ({ step, claims: chain.claims.filter((c) => c.step === step) }))
    .filter((g) => g.claims.length > 0);

  return (
    <div className={`flex flex-col ${compact ? 'gap-2' : 'gap-3'}`}>
      {!compact && (
        <div className="flex items-baseline gap-2">
          <span className="num rounded-chip border border-edge bg-panelHi px-1.5 py-1 text-[10px] text-ink">
            {chain.incidentId}
          </span>
          <span className="text-[13px] font-semibold text-ink">{chain.decision}</span>
          <span
            className="num ml-auto cursor-help text-[10px] text-inkFaint"
            title="Best single observation, plus a bounded step per additional independent agent. Never reaches certainty."
          >
            {(chain.confidence * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {grouped.map(({ step, claims }) => (
        <div key={step} className="flex flex-col gap-1">
          <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
            {STEP_LABELS[step]}
          </span>
          {claims.map((claim, i) => (
            <div
              key={`${step}-${i}`}
              className={`rounded-panel border border-edge bg-abyss ${compact ? 'px-2 py-1.5' : 'px-2.5 py-2'}`}
            >
              <p className={`leading-relaxed text-inkDim ${compact ? 'text-[10px]' : 'text-body'}`}>
                {claim.text}
              </p>
              <span className="num mt-1 block text-[9px] text-inkFaint">
                {claim.source.kind}:{claim.source.ref}
                {claim.source.detail ? ` · ${claim.source.detail}` : ''}
              </span>
            </div>
          ))}
        </div>
      ))}

      {chain.dismissed.length > 0 && (
        <div className="flex flex-col gap-1 border-t border-edge pt-2">
          <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
            Considered and rejected
          </span>
          {chain.dismissed.map((claim, i) => (
            <p key={i} className="text-[10px] leading-relaxed text-inkFaint">
              {claim.text}
            </p>
          ))}
        </div>
      )}

      {chain.unresolved.length > 0 && (
        <div className="flex flex-col gap-1 border-t border-edge pt-2">
          <span className="text-[9px] uppercase tracking-[0.08em] text-inkFaint">
            Unresolved
          </span>
          {chain.unresolved.map((claim, i) => (
            <p key={i} className="text-[10px] leading-relaxed text-inkFaint">
              {claim.text}
            </p>
          ))}
        </div>
      )}

      {!compact && (
        <p className="mt-auto border-t border-edge pt-2.5 text-[9px] uppercase leading-relaxed tracking-[0.06em] text-inkFaint">
          every claim cites a finding, a clause, an agent or a measurement · no
          statement here was generated at report time
        </p>
      )}
    </div>
  );
}
