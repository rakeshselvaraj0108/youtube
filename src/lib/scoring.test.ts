import { describe, expect, it } from 'vitest';
import { computeReadiness, SUB_SCORE_ORDER, verdictFor, WEIGHTS } from '@/lib/scoring';
import type { SubScores } from '@/types/analysis';
import { beforeReport, afterReport } from '@/data/fixture';

const flat = (n: number): SubScores => ({
  policy: n,
  copyright: n,
  metadata: n,
  accessibility: n,
  audio: n,
});

/** Everything at 95 except one dimension, used to drive the clamp. */
const weakOn = (key: keyof SubScores, value: number): SubScores => ({
  ...flat(95),
  [key]: value,
});

describe('WEIGHTS', () => {
  it('sums to exactly 1', () => {
    const total = SUB_SCORE_ORDER.reduce((a, k) => a + WEIGHTS[k], 0);
    expect(total).toBeCloseTo(1, 10);
  });
});

describe('computeReadiness — monotonicity', () => {
  it('never decreases when any single dimension improves', () => {
    for (const key of SUB_SCORE_ORDER) {
      for (let v = 0; v < 100; v += 5) {
        const lower = computeReadiness(weakOn(key, v)).overall;
        const higher = computeReadiness(weakOn(key, v + 5)).overall;
        expect(higher).toBeGreaterThanOrEqual(lower);
      }
    }
  });

  it('is monotonic across a uniform sweep', () => {
    let prev = -1;
    for (let v = 0; v <= 100; v += 1) {
      const { overall } = computeReadiness(flat(v));
      expect(overall).toBeGreaterThanOrEqual(prev);
      prev = overall;
    }
  });
});

describe('computeReadiness — the anti-masking clamp', () => {
  it('caps the overall score at weakest + 15', () => {
    // Four strong dimensions would otherwise average a fatal copyright score away.
    const sub = weakOn('copyright', 19);
    const result = computeReadiness(sub);
    expect(result.weighted).toBeGreaterThan(70); // a plain average would pass this video
    expect(result.overall).toBe(34);
    expect(result.capped).toBe(true);
  });

  it('does not bind when every dimension is close together', () => {
    const result = computeReadiness(flat(90));
    expect(result.overall).toBe(90);
    expect(result.capped).toBe(false);
  });

  it('identifies the weakest dimension', () => {
    expect(computeReadiness(weakOn('accessibility', 12)).weakest).toBe('accessibility');
    expect(computeReadiness(weakOn('audio', 3)).weakest).toBe('audio');
  });

  it('resolves weakest ties to the highest-weighted dimension', () => {
    const sub: SubScores = { ...flat(95), policy: 40, audio: 40 };
    expect(computeReadiness(sub).weakest).toBe('policy');
  });

  it('never returns a score outside 0–100', () => {
    expect(computeReadiness(flat(0)).overall).toBe(0);
    expect(computeReadiness(flat(100)).overall).toBe(100);
  });
});

describe('verdict boundaries', () => {
  it('READY_TO_PUBLISH requires overall >= 85 AND worst >= 70', () => {
    expect(verdictFor(85, 70)).toBe('READY_TO_PUBLISH');
    expect(verdictFor(84, 70)).toBe('PUBLISH_WITH_FIXES');
    expect(verdictFor(99, 69)).toBe('PUBLISH_WITH_FIXES');
  });

  it('PUBLISH_WITH_FIXES requires overall >= 70 AND worst >= 50', () => {
    expect(verdictFor(70, 50)).toBe('PUBLISH_WITH_FIXES');
    expect(verdictFor(69, 50)).toBe('NOT_READY');
    expect(verdictFor(80, 49)).toBe('NOT_READY');
  });

  it('NOT_READY covers overall >= 50 regardless of worst', () => {
    expect(verdictFor(50, 0)).toBe('NOT_READY');
    expect(verdictFor(69, 49)).toBe('NOT_READY');
  });

  it('DO_NOT_PUBLISH covers everything below 50', () => {
    expect(verdictFor(49, 49)).toBe('DO_NOT_PUBLISH');
    expect(verdictFor(0, 0)).toBe('DO_NOT_PUBLISH');
  });

  it('agrees with computeReadiness at each boundary', () => {
    expect(computeReadiness(weakOn('accessibility', 70)).verdict).toBe('READY_TO_PUBLISH');
    expect(computeReadiness(weakOn('accessibility', 69)).verdict).toBe('PUBLISH_WITH_FIXES');
    expect(computeReadiness(weakOn('accessibility', 55)).verdict).toBe('PUBLISH_WITH_FIXES');
    expect(computeReadiness(weakOn('accessibility', 45)).verdict).toBe('NOT_READY');
    expect(computeReadiness(weakOn('accessibility', 19)).verdict).toBe('DO_NOT_PUBLISH');
  });

  it('never displays a score that contradicts its verdict', () => {
    // Verdict is derived from the same rounded integer the gauge renders.
    for (let v = 0; v <= 100; v += 1) {
      const r = computeReadiness(flat(v));
      expect(r.verdict).toBe(verdictFor(r.overall, r.worst));
    }
  });
});

describe('demo fixture', () => {
  it('scores the before state as DO_NOT_PUBLISH', () => {
    const result = computeReadiness(beforeReport.scores.sub);
    expect(result.verdict).toBe('DO_NOT_PUBLISH');
    expect(result.overall).toBe(34);
    expect(beforeReport.scores.verdict).toBe('DO_NOT_PUBLISH');
    expect(beforeReport.scores.overall).toBe(34);
  });

  it('scores the after state as READY_TO_PUBLISH', () => {
    const result = computeReadiness(afterReport.scores.sub);
    expect(result.verdict).toBe('READY_TO_PUBLISH');
    expect(afterReport.scores.verdict).toBe('READY_TO_PUBLISH');
    expect(afterReport.scores.overall).toBeGreaterThanOrEqual(85);
  });

  it('is dragged down by copyright, not by an average', () => {
    const before = computeReadiness(beforeReport.scores.sub);
    expect(before.weakest).toBe('copyright');
    expect(before.capped).toBe(true);
  });
});
