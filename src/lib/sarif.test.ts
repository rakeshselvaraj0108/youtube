import { describe, expect, it } from 'vitest';
import { buildSarif, exitCode, SARIF_SCHEMA } from '@/lib/sarif';
import { buildCertificate } from '@/lib/certificate';
import { afterReport, beforeReport } from '@/data/fixture';

const sarif = buildSarif(beforeReport);
const run = sarif.runs[0]!;

describe('SARIF envelope', () => {
  it('declares the 2.1.0 schema and version', () => {
    expect(sarif.$schema).toBe(SARIF_SCHEMA);
    expect(sarif.version).toBe('2.1.0');
    expect(sarif.runs).toHaveLength(1);
  });

  it('identifies the tool', () => {
    expect(run.tool.driver.name).toBe('PREFLIGHT');
    expect(run.tool.driver.version).toBe(beforeReport.meta.engineVersion);
  });
});

describe('SARIF rules', () => {
  it('emits one rule per distinct clause, not one per finding', () => {
    const clauses = new Set(beforeReport.findings.map((f) => f.clauseId));
    expect(run.tool.driver.rules).toHaveLength(clauses.size);
    // AF-01 is cited by two findings and must appear exactly once.
    expect(run.tool.driver.rules.filter((r) => r.id === 'AF-01')).toHaveLength(1);
  });

  it('gives every rule a PascalCase name and full clause text', () => {
    for (const rule of run.tool.driver.rules) {
      expect(rule.name).toMatch(/^[A-Za-z0-9]+$/);
      expect(rule.fullDescription.text.length).toBeGreaterThan(60);
    }
  });

  it('maps severity onto SARIF levels', () => {
    const violence = run.tool.driver.rules.find((r) => r.id === 'AF-04');
    expect(violence?.defaultConfiguration.level).toBe('error');
  });
});

describe('SARIF results', () => {
  it('emits one result per finding', () => {
    expect(run.results).toHaveLength(beforeReport.findings.length);
  });

  it('references only rules it declared', () => {
    const declared = new Set(run.tool.driver.rules.map((r) => r.id));
    for (const result of run.results) {
      expect(declared.has(result.ruleId)).toBe(true);
    }
  });

  it('encodes seconds as lines and keeps true timings in properties', () => {
    const graphic = run.results.find((r) => r.properties.startMs === 454_000)!;
    const region = graphic.locations[0]!.physicalLocation.region;
    expect(region.startLine).toBe(454);
    expect(region.endLine).toBe(457);
    expect(graphic.properties.endMs).toBe(457_000);
  });

  it('never emits startLine 0 — SARIF lines are 1-indexed', () => {
    for (const result of run.results) {
      expect(result.locations[0]!.physicalLocation.region.startLine).toBeGreaterThanOrEqual(1);
    }
  });

  it('carries the full adversarial record into properties', () => {
    for (const result of run.results) {
      expect(typeof result.properties.auditorCharge).toBe('string');
      expect(['UPHELD', 'DISMISSED']).toContain(result.properties.adjudicatorVerdict);
    }
  });

  it('reports coverage and degraded agents in the invocation', () => {
    const invocation = run.invocations[0]!;
    expect(invocation.properties.coverage).toBeCloseTo(beforeReport.meta.coverage, 10);
    expect(invocation.properties.degradedAgents.length).toBeGreaterThan(0);
  });
});

describe('exit code', () => {
  it('fails CI on the unremediated video', () => {
    expect(exitCode(beforeReport)).toBe(1);
  });

  it('passes CI once the fix is applied', () => {
    expect(exitCode(afterReport)).toBe(0);
  });
});

describe('release certificate', () => {
  const cert = buildCertificate(beforeReport);

  it('carries the scoring rule alongside the score so it can be recomputed', () => {
    expect(cert.readiness.clamp).toContain('worst + 15');
    expect(cert.readiness.overall).toBe(beforeReport.scores.overall);
    expect(cert.readiness.weights.policy).toBe(0.4);
  });

  it('states coverage per agent rather than a single reassuring number', () => {
    expect(cert.coverage.agents).toHaveLength(beforeReport.agents.length);
    expect(cert.coverage.overall).toBeLessThan(1);
  });

  it('ships its own limitations', () => {
    expect(cert.limitations.length).toBeGreaterThanOrEqual(3);
    expect(cert.limitations.join(' ')).toContain('does not prove safety');
  });

  it('counts findings by severity consistently with the report', () => {
    const total = Object.values(cert.findings.bySeverity).reduce((a, b) => a + b, 0);
    expect(total).toBe(beforeReport.findings.length);
  });
});
