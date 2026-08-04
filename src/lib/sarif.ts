import type { AnalysisReport, Finding, Severity } from '@/types/analysis';
import { formatPrecise } from '@/lib/time';

/**
 * SARIF 2.1.0 emitter.
 *
 * SARIF is the interchange format CodeQL, Semgrep and every serious static
 * analyser speaks, which means these findings render natively in GitHub's
 * Security tab and a video repository can literally fail CI.
 *
 * SARIF has no time axis. Rather than invent one, seconds are encoded as
 * `startLine` so GitHub still renders a position, and the true millisecond
 * spans travel in `properties` where a consumer that understands video can
 * read them. That is a deliberate mapping, not a coincidence of copying the
 * schema.
 */

const SARIF_LEVEL: Record<Severity, 'error' | 'warning' | 'note'> = {
  CRITICAL: 'error',
  HIGH: 'error',
  MEDIUM: 'warning',
  LOW: 'note',
};

export const SARIF_SCHEMA = 'https://json.schemastore.org/sarif-2.1.0.json';
export const INFORMATION_URI = 'https://github.com/rakeshselvaraj0108/youtube';

/** PascalCase rule name from a clause title, e.g. "Inappropriate language". */
function ruleName(title: string): string {
  return title
    .replace(/[^A-Za-z0-9 ]/g, '')
    .split(/\s+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join('');
}

function resultMessage(finding: Finding): string {
  const fix =
    finding.suggestedFix === 'NONE'
      ? 'No automated fix available.'
      : `Fix: ${finding.suggestedFix} ${finding.startMs}-${finding.endMs}ms.`;
  return (
    `${finding.title} at ${formatPrecise(finding.startMs)} — ${finding.severity} ` +
    `(conf ${finding.confidence.toFixed(2)}). ${fix}`
  );
}

export function buildSarif(report: AnalysisReport) {
  // One rule per distinct clause, deduplicated — SARIF rules are a set, and
  // two Language findings must not produce two AF-01 entries.
  const rules = new Map<string, Finding>();
  for (const finding of report.findings) {
    if (!rules.has(finding.clauseId)) rules.set(finding.clauseId, finding);
  }

  return {
    $schema: SARIF_SCHEMA,
    version: '2.1.0',
    runs: [
      {
        tool: {
          driver: {
            name: 'PREFLIGHT',
            version: report.meta.engineVersion,
            informationUri: INFORMATION_URI,
            rules: [...rules.values()].map((finding) => ({
              id: finding.clauseId,
              name: ruleName(finding.policy.title),
              shortDescription: { text: finding.policy.title },
              fullDescription: { text: finding.policy.text },
              defaultConfiguration: { level: SARIF_LEVEL[finding.severity] },
              properties: {
                tags: ['monetization', 'advertiser-friendly', finding.category.toLowerCase()],
                section: finding.policy.section,
              },
            })),
          },
        },
        results: report.findings.map((finding) => ({
          ruleId: finding.clauseId,
          level: SARIF_LEVEL[finding.severity],
          message: { text: resultMessage(finding) },
          locations: [
            {
              physicalLocation: {
                artifactLocation: { uri: report.video.filename },
                // No time axis in SARIF: seconds stand in for lines so GitHub
                // renders a position. True timings live in properties below.
                region: {
                  startLine: Math.max(1, Math.floor(finding.startMs / 1000)),
                  endLine: Math.max(1, Math.floor(finding.endMs / 1000)),
                  startColumn: 1,
                  snippet: { text: finding.evidence.transcript },
                },
              },
            },
          ],
          properties: {
            startMs: finding.startMs,
            endMs: finding.endMs,
            confidence: finding.confidence,
            fusedConfidence: finding.fusedConfidence,
            modalities: finding.modalities,
            severity: finding.severity,
            category: finding.category,
            suggestedFix: finding.suggestedFix,
            policySection: finding.policy.section,
            auditorCharge: finding.adversarial.auditor.charge,
            advocateDefense: finding.adversarial.advocate.defense,
            advocateStrength: finding.adversarial.advocate.strength,
            adjudicatorVerdict: finding.adversarial.adjudicator.verdict,
            adjudicatorRationale: finding.adversarial.adjudicator.rationale,
          },
        })),
        invocations: [
          {
            executionSuccessful: true,
            endTimeUtc: report.meta.analyzedAt,
            // Coverage is reported, never hidden — a consumer can see this run
            // did not inspect everything.
            properties: {
              coverage: report.meta.coverage,
              policyVersion: report.meta.policyVersion,
              degradedAgents: report.agents
                .filter((a) => a.status !== 'OK')
                .map((a) => ({ id: a.id, status: a.status, coverage: a.coverage })),
            },
          },
        ],
      },
    ],
  };
}

/** Exit code a CI run would take from this report. */
export function exitCode(report: AnalysisReport): 0 | 1 {
  return report.scores.verdict === 'READY_TO_PUBLISH' ||
    report.scores.verdict === 'PUBLISH_WITH_FIXES'
    ? 0
    : 1;
}
