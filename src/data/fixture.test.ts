import { describe, expect, it } from 'vitest';
import { afterReport, beforeReport, DEMO_DURATION_MS } from '@/data/fixture';
import { computeCoverage, degradedAgents } from '@/lib/coverage';
import { remediableFindings } from '@/lib/findings';
import { timelineTicks } from '@/lib/time';

const report = beforeReport;

describe('findings', () => {
  it('has 11 findings, each with a unique id', () => {
    expect(report.findings).toHaveLength(11);
    expect(new Set(report.findings.map((f) => f.id)).size).toBe(11);
  });

  it('bounds every span inside the runtime', () => {
    for (const f of report.findings) {
      expect(f.startMs).toBeGreaterThanOrEqual(0);
      expect(f.endMs).toBeGreaterThan(f.startMs);
      expect(f.endMs).toBeLessThanOrEqual(DEMO_DURATION_MS);
    }
  });

  it('cites a policy clause matching the finding clauseId', () => {
    for (const f of report.findings) {
      expect(f.policy.clauseId).toBe(f.clauseId);
      expect(f.policy.text.length).toBeGreaterThan(60);
      expect(f.policy.section).toMatch(/§/);
    }
  });

  it('carries a complete adversarial record on every finding', () => {
    for (const f of report.findings) {
      expect(f.adversarial.auditor.charge.length).toBeGreaterThan(20);
      expect(f.adversarial.adjudicator.rationale.length).toBeGreaterThan(20);
      expect(f.adversarial.adjudicator.confidence).toBeGreaterThan(0);
      expect(f.adversarial.adjudicator.confidence).toBeLessThanOrEqual(1);
    }
  });

  it('resolves every highlight span inside its own transcript', () => {
    for (const f of report.findings) {
      const [start, end] = f.evidence.highlightSpan;
      expect(end).toBeLessThanOrEqual(f.evidence.transcript.length);
      expect(start).toBeLessThanOrEqual(end);
    }
  });

  it('contains no placeholder copy', () => {
    const blob = JSON.stringify(report).toLowerCase();
    for (const banned of ['lorem', 'ipsum', 'sample finding', 'placeholder', 'todo', 'foo bar']) {
      expect(blob).not.toContain(banned);
    }
  });
});

describe('breakdown', () => {
  it('accounts for every finding exactly once', () => {
    const total = report.breakdown.reduce((a, row) => a + row.count, 0);
    expect(total).toBe(report.findings.length);
  });

  it('reports each category at its worst severity', () => {
    const language = report.breakdown.find((r) => r.category === 'Language');
    expect(language).toEqual({ category: 'Language', count: 2, severity: 'HIGH' });
  });
});

describe('remediation', () => {
  it('emits exactly one op per remediable finding', () => {
    const remediable = remediableFindings(report.findings);
    expect(report.remediation.ops).toHaveLength(remediable.length);
    expect(report.remediation.ops).toHaveLength(4);
  });

  it('matches each op kind to the finding that suggested it', () => {
    for (const op of report.remediation.ops) {
      const finding = report.findings.find((f) => f.id === op.findingId);
      expect(finding?.suggestedFix).toBe(op.op);
    }
  });

  it('keeps every op inside the runtime and inside its finding span', () => {
    for (const op of report.remediation.ops) {
      const finding = report.findings.find((f) => f.id === op.findingId)!;
      expect(op.endMs).toBeGreaterThan(op.startMs);
      expect(op.endMs).toBeLessThanOrEqual(DEMO_DURATION_MS);
      expect(op.startMs).toBeGreaterThanOrEqual(finding.startMs - 200);
      expect(op.endMs).toBeLessThanOrEqual(finding.endMs + 200);
    }
  });
});

describe('agents', () => {
  it('is a 12-node DAG with unique ids', () => {
    expect(report.agents).toHaveLength(12);
    expect(new Set(report.agents.map((a) => a.id)).size).toBe(12);
  });

  it('resolves every parent, and every parent sits in a lower tier', () => {
    const byId = new Map(report.agents.map((a) => [a.id, a]));
    for (const agent of report.agents) {
      for (const parentId of agent.parents) {
        const parent = byId.get(parentId);
        expect(parent, `${agent.id} -> ${parentId}`).toBeDefined();
        expect(parent!.tier).toBeLessThan(agent.tier);
      }
    }
  });

  it('sequences the terminal log in ascending time', () => {
    const stamps = report.agents.map((a) => a.tsMs);
    expect([...stamps].sort((a, b) => a - b)).toEqual(stamps);
  });
});

describe('coverage honesty', () => {
  it('reports 83% coverage', () => {
    expect(Math.round(report.meta.coverage * 100)).toBe(83);
    expect(report.meta.coverage).toBeCloseTo(computeCoverage(report.agents), 10);
  });

  it('names the degraded vision agent at 42% coverage', () => {
    const degraded = degradedAgents(report.agents);
    const vision = degraded.find((a) => a.id === 'vision');
    expect(vision?.status).toBe('DEGRADED');
    expect(vision?.coverage).toBe(0.42);
  });

  it('never claims full coverage while an agent is degraded', () => {
    expect(report.meta.coverage).toBeLessThan(1);
  });
});

describe('risk terrain', () => {
  it('tiles the whole runtime with no gaps', () => {
    const bands = report.riskBands;
    expect(bands[0]!.startMs).toBe(0);
    expect(bands.at(-1)!.endMs).toBe(DEMO_DURATION_MS);
    for (let i = 1; i < bands.length; i++) {
      expect(bands[i]!.startMs).toBe(bands[i - 1]!.endMs);
    }
  });

  it('keeps every band inside 0..1', () => {
    for (const band of report.riskBands) {
      expect(band.risk).toBeGreaterThanOrEqual(0);
      expect(band.risk).toBeLessThanOrEqual(1);
    }
  });

  it('peaks over the CRITICAL finding', () => {
    const critical = report.findings.find((f) => f.severity === 'CRITICAL')!;
    const overlapping = report.riskBands.filter(
      (b) => b.endMs > critical.startMs && b.startMs < critical.endMs,
    );
    expect(Math.max(...overlapping.map((b) => b.risk))).toBeGreaterThan(0.9);
  });

  it('is not flattened by file-scoped findings', () => {
    const quiet = report.riskBands.filter((b) => b.risk === 0);
    expect(quiet.length).toBeGreaterThan(report.riskBands.length / 2);
  });
});

describe('timeline axis', () => {
  it('never emits a tick past the runtime', () => {
    for (const tick of timelineTicks(DEMO_DURATION_MS)) {
      expect(tick.ms).toBeLessThanOrEqual(DEMO_DURATION_MS);
      expect(tick.t).toBeLessThanOrEqual(1);
    }
  });

  it('emits ascending, evenly spaced ticks', () => {
    const ticks = timelineTicks(DEMO_DURATION_MS);
    expect(ticks.length).toBeGreaterThanOrEqual(6);
    const step = ticks[1]!.ms - ticks[0]!.ms;
    for (let i = 1; i < ticks.length; i++) {
      expect(ticks[i]!.ms - ticks[i - 1]!.ms).toBe(step);
    }
  });
});

describe('after report', () => {
  it('clears every remediated finding and keeps the residue', () => {
    expect(afterReport.findings).toHaveLength(4);
    expect(afterReport.findings.every((f) => f.suggestedFix === 'NONE')).toBe(true);
  });

  it('carries no CRITICAL or HIGH findings', () => {
    expect(afterReport.findings.some((f) => f.severity === 'CRITICAL')).toBe(false);
    expect(afterReport.findings.some((f) => f.severity === 'HIGH')).toBe(false);
  });

  it('has a lower risk terrain than the before state', () => {
    const peak = (r: typeof afterReport) => Math.max(...r.riskBands.map((b) => b.risk));
    expect(peak(afterReport)).toBeLessThan(peak(beforeReport));
  });

  it('flips the verdict', () => {
    expect(beforeReport.scores.verdict).toBe('DO_NOT_PUBLISH');
    expect(afterReport.scores.verdict).toBe('READY_TO_PUBLISH');
  });
});
