import type { Finding, RiskBand, Severity } from '@/types/analysis';

/**
 * Risk terrain.
 *
 * The timeline is not decorative — every segment's height is the maximum
 * severity×confidence of the findings that overlap it. Findings that span the
 * whole file (accessibility, metadata, loudness) are excluded: they are true of
 * every second and would flatten the terrain into a plateau, hiding the spikes
 * that actually matter.
 */

const SEVERITY_WEIGHT: Record<Severity, number> = {
  CRITICAL: 1.0,
  HIGH: 0.76,
  MEDIUM: 0.52,
  LOW: 0.28,
};

/** A finding covering more than this fraction of the runtime is file-scoped. */
const FILE_SCOPE_RATIO = 0.5;

export function isFileScoped(finding: Finding, durationMs: number): boolean {
  return (finding.endMs - finding.startMs) / durationMs > FILE_SCOPE_RATIO;
}

export function localisedFindings(findings: Finding[], durationMs: number): Finding[] {
  return findings.filter((f) => !isFileScoped(f, durationMs));
}

export function buildRiskBands(
  findings: Finding[],
  durationMs: number,
  segments = 96,
): RiskBand[] {
  const width = durationMs / segments;
  const scoped = localisedFindings(findings, durationMs);

  const bands: RiskBand[] = [];
  for (let i = 0; i < segments; i++) {
    const startMs = Math.round(i * width);
    const endMs = Math.round((i + 1) * width);

    let risk = 0;
    for (const f of scoped) {
      if (f.endMs <= startMs || f.startMs >= endMs) continue;
      risk = Math.max(risk, SEVERITY_WEIGHT[f.severity] * f.confidence);
    }
    bands.push({ startMs, endMs, risk });
  }

  // One-segment neighbour bleed so a 3-second spike still reads at this scale
  // without inventing risk where there is none.
  return bands.map((band, i) => {
    const prev = bands[i - 1]?.risk ?? 0;
    const next = bands[i + 1]?.risk ?? 0;
    const bled = Math.max(band.risk, prev * 0.45, next * 0.45);
    return { ...band, risk: Number(bled.toFixed(3)) };
  });
}

export function findingRisk(finding: Finding): number {
  return SEVERITY_WEIGHT[finding.severity] * finding.confidence;
}
