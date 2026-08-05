import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { computeReadiness } from '@/lib/scoring';
import type { SubScores } from '@/types/analysis';

/**
 * The cross-language scoring contract.
 *
 * The page renders this TypeScript implementation; `report.json` carries the
 * Python one. If they disagree by even a point, the headline number on screen
 * contradicts the data underneath it — and the certificate that ships the
 * scoring rule alongside the score becomes a lie.
 *
 * Both suites assert against the same vectors, emitted by
 * `python scripts/emit_scoring_vectors.py`.
 */

interface Vector {
  sub: SubScores;
  overall: number;
  verdict: string;
  weakest: string;
  capped: boolean;
}

const path = resolve(process.cwd(), 'tests/fixtures/scoring_vectors.json');
const vectors: { cases: Vector[] } = JSON.parse(readFileSync(path, 'utf8'));

describe('Python ≡ TypeScript scoring', () => {
  it('has vectors to check', () => {
    expect(vectors.cases.length).toBeGreaterThanOrEqual(50);
  });

  it('agrees on the overall score for every vector', () => {
    const disagreements: string[] = [];
    for (const vector of vectors.cases) {
      const actual = computeReadiness(vector.sub);
      if (actual.overall !== vector.overall) {
        disagreements.push(
          `${JSON.stringify(vector.sub)} → ts ${actual.overall} vs py ${vector.overall}`,
        );
      }
    }
    expect(disagreements).toEqual([]);
  });

  it('agrees on the verdict for every vector', () => {
    for (const vector of vectors.cases) {
      expect(computeReadiness(vector.sub).verdict, JSON.stringify(vector.sub)).toBe(
        vector.verdict,
      );
    }
  });

  it('agrees on which dimension is weakest', () => {
    for (const vector of vectors.cases) {
      expect(computeReadiness(vector.sub).weakest, JSON.stringify(vector.sub)).toBe(
        vector.weakest,
      );
    }
  });

  it('agrees on whether the clamp bound', () => {
    for (const vector of vectors.cases) {
      expect(computeReadiness(vector.sub).capped, JSON.stringify(vector.sub)).toBe(
        vector.capped,
      );
    }
  });

  it('agrees at the rounding boundary, where the two languages differ', () => {
    // Python's round() is banker's rounding and JavaScript's Math.round is not.
    // The Python side compensates explicitly; these vectors pin that it works.
    const halfway = vectors.cases.filter((v) => {
      const weighted =
        0.4 * v.sub.policy +
        0.3 * v.sub.copyright +
        0.12 * v.sub.metadata +
        0.1 * v.sub.accessibility +
        0.08 * v.sub.audio;
      return Math.abs(weighted - Math.floor(weighted) - 0.5) < 1e-9;
    });
    for (const vector of halfway) {
      expect(computeReadiness(vector.sub).overall).toBe(vector.overall);
    }
  });
});
