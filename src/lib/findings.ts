import type { BreakdownRow, Finding, Severity } from '@/types/analysis';
import { SEVERITY_RANK } from '@/lib/scoring';

/** Category rollup for the policy breakdown panel. Derived, never authored. */
export function buildBreakdown(findings: Finding[]): BreakdownRow[] {
  const byCategory = new Map<string, Finding[]>();
  for (const f of findings) {
    const bucket = byCategory.get(f.category);
    if (bucket) bucket.push(f);
    else byCategory.set(f.category, [f]);
  }

  const rows: BreakdownRow[] = [];
  for (const [category, group] of byCategory) {
    const severity = group.reduce<Severity>(
      (worst, f) => (SEVERITY_RANK[f.severity] < SEVERITY_RANK[worst] ? f.severity : worst),
      'LOW',
    );
    rows.push({ category, count: group.length, severity });
  }

  return rows.sort(
    (a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || b.count - a.count,
  );
}

export type FindingSort = 'severity' | 'time' | 'confidence';

export function sortFindings(findings: Finding[], by: FindingSort): Finding[] {
  const copy = [...findings];
  switch (by) {
    case 'time':
      return copy.sort((a, b) => a.startMs - b.startMs);
    case 'confidence':
      return copy.sort((a, b) => b.confidence - a.confidence);
    case 'severity':
    default:
      return copy.sort(
        (a, b) =>
          SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] ||
          b.confidence - a.confidence ||
          a.startMs - b.startMs,
      );
  }
}

/** Findings the remediation compiler can actually lower into an ffmpeg op. */
export function remediableFindings(findings: Finding[]): Finding[] {
  return findings.filter((f) => f.suggestedFix !== 'NONE');
}
