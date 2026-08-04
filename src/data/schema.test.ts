import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { describe, expect, it } from 'vitest';
import { afterReport, beforeReport } from '@/data/fixture';
import type { AnalysisReport } from '@/types/analysis';

/**
 * The contract gate.
 *
 * `schema/analysis-report.schema.json` is generated from src/types/analysis.ts
 * and is what the Python engine validates its output against. If this test goes
 * red, the two sides of the wire have drifted and the page will render numbers
 * the engine cannot produce.
 *
 * Regenerate with: npm run schema
 */

const schema = JSON.parse(
  readFileSync(resolve(process.cwd(), 'schema/analysis-report.schema.json'), 'utf8'),
);

const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);
const validate = ajv.compile(schema);

function check(report: AnalysisReport, label: string) {
  const ok = validate(report);
  if (!ok) {
    const detail = (validate.errors ?? [])
      .map((e) => `${e.instancePath || '/'} ${e.message}`)
      .join('\n  ');
    throw new Error(`${label} failed schema validation:\n  ${detail}`);
  }
  expect(ok).toBe(true);
}

describe('AnalysisReport schema', () => {
  it('declares the top-level report shape', () => {
    expect(schema.$ref).toBe('#/definitions/AnalysisReport');
    expect(Object.keys(schema.definitions)).toContain('Finding');
    expect(Object.keys(schema.definitions)).toContain('RemediationOp');
  });

  it('validates the before fixture', () => {
    check(beforeReport, 'beforeReport');
  });

  it('validates the after fixture', () => {
    check(afterReport, 'afterReport');
  });

  it('rejects a report missing a required section', () => {
    const { scores: _scores, ...broken } = beforeReport;
    expect(validate(broken)).toBe(false);
  });

  it('rejects an out-of-contract severity', () => {
    const broken = structuredClone(beforeReport) as AnalysisReport;
    // @ts-expect-error deliberately violating the union to prove the gate bites
    broken.findings[0]!.severity = 'CATASTROPHIC';
    expect(validate(broken)).toBe(false);
  });

  it('rejects an out-of-contract op kind', () => {
    const broken = structuredClone(beforeReport) as AnalysisReport;
    // @ts-expect-error deliberately violating the union
    broken.remediation.ops[0]!.op = 'ENHANCE';
    expect(validate(broken)).toBe(false);
  });
});
