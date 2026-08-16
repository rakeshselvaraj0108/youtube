import { describe, expect, it } from 'vitest';

import {
  LIFECYCLE_STEPS,
  STATUS_LABEL,
  pairFor,
  seekTarget,
  statusHex,
  stepStatus,
  verdictRationale,
} from '@/lib/verification';
import type {
  EvidencePair,
  LifecycleState,
  RemediationRecord,
  Verification,
} from '@/types/analysis';

function record(overrides: Partial<RemediationRecord> = {}): RemediationRecord {
  return {
    remediationId: 'REM-0001',
    sourceRunId: 'run-a',
    simulationId: 'SIM-0001',
    verificationRunId: 'run-b',
    verificationId: 'VER-0001',
    artifactId: 'ART-abc',
    sourcePath: 'clip.mp4',
    outputPath: 'clip.safe.mp4',
    findingIds: [],
    incidentIds: [],
    ops: [],
    state: 'PARTIALLY_REMEDIATED',
    previousState: 'COMPARING',
    stateDetail: 'some findings resolved, others remain',
    terminal: true,
    verdict: 'PARTIALLY_REMEDIATED',
    error: null,
    createdAt: '',
    updatedAt: '',
    transitions: [],
    ...overrides,
  };
}

/**
 * The deck's side of the closed loop.
 *
 * Two rules are under test, and both are about refusing to fill a gap:
 * a state must never be conveyed by colour alone, and a missing measurement
 * must never be rendered as a plausible number or a plausible seek target.
 */

function verification(overrides: Partial<Verification> = {}): Verification {
  return {
    verdict: 'PARTIALLY_REMEDIATED',
    changes: [],
    incidentChanges: [],
    originalScore: 42,
    remediatedScore: 42,
    scoreDelta: 0,
    predictedScore: 43,
    predictionOutcome: 'MATCHED',
    resolved: 2,
    persisting: 3,
    new: 1,
    inconclusive: 0,
    incidentsResolved: 1,
    incidentsPersisting: 1,
    incidentsPartial: 0,
    incidentsChanged: 0,
    incidentsNew: 1,
    incidentsInconclusive: 0,
    structuralOk: true,
    reanalysisOk: true,
    notes: [],
    ...overrides,
  };
}

function pair(overrides: Partial<EvidencePair> = {}): EvidencePair {
  return {
    findingId: 'f1',
    incidentId: 'INC-001',
    clauseId: 'AF-09',
    category: 'Language',
    severity: 'HIGH',
    status: 'RESOLVED',
    before: {
      runId: 'run-a',
      tsMs: 5_000,
      frame: { tsMs: 5_000, source: 'original', runId: 'run-a', width: 720, image: 'data:,' },
      transcript: 'words',
      highlightSpan: [0, 5],
      confidence: 0.8,
      coverage: 0.9,
    },
    remediation: { remediationId: 'REM-0001', op: 'CUT', startMs: 2_000, endMs: 4_000 },
    after: {
      runId: 'run-b',
      tsMs: 3_000,
      frame: { tsMs: 3_000, source: 'remediated', runId: 'run-b', width: 720, image: 'data:,' },
      removedByRemediation: false,
      unavailable: '',
    },
    notes: [],
    ...overrides,
  };
}

describe('status vocabulary', () => {
  it('gives every comparison state a text label', () => {
    for (const status of [
      'RESOLVED',
      'PERSISTING',
      'PARTIALLY_REMEDIATED',
      'CHANGED',
      'NEW',
      'INCONCLUSIVE',
    ]) {
      expect(STATUS_LABEL[status]).toBeTruthy();
    }
  });

  it('gives every verdict a text label', () => {
    for (const verdict of [
      'VERIFIED_SAFE',
      'PARTIALLY_REMEDIATED',
      'REMEDIATION_FAILED',
      'NEW_RISK_DETECTED',
      'NO_CHANGE',
      'INCONCLUSIVE',
    ]) {
      expect(STATUS_LABEL[verdict]).toBeTruthy();
    }
  });

  it('separates resolved from new by more than shade', () => {
    expect(statusHex('RESOLVED')).not.toBe(statusHex('NEW'));
  });

  it('does not dress inconclusive as a mild warning', () => {
    // "Nobody looked" is an absence of information, not a smaller problem.
    // Sharing the caution colour would invite reading it as nearly fine.
    expect(statusHex('INCONCLUSIVE')).not.toBe(statusHex('PERSISTING'));
    expect(statusHex('INCONCLUSIVE')).not.toBe(statusHex('RESOLVED'));
  });

  it('falls back rather than throwing on an unknown state', () => {
    expect(statusHex('SOMETHING_ELSE')).toBeTruthy();
  });
});

describe('verdict rationale', () => {
  it('leads with the structural failure when there is one', () => {
    expect(verdictRationale(verification({ structuralOk: false }))).toContain(
      'did not match the edit list',
    );
  });

  it('says nothing is known when re-analysis did not complete', () => {
    const text = verdictRationale(verification({ reanalysisOk: false }));
    expect(text).toContain('unknown');
  });

  it('names each category that has a count', () => {
    const text = verdictRationale(verification({ inconclusive: 2 }));
    expect(text).toContain('2 resolved');
    expect(text).toContain('3 still detected');
    expect(text).toContain('1 newly detected');
    expect(text).toContain('2 not checkable');
  });

  it('reports an unchanged run plainly', () => {
    const text = verdictRationale(
      verification({ resolved: 0, persisting: 0, new: 0, inconclusive: 0 }),
    );
    expect(text).toContain('Nothing changed');
  });
});

describe('evidence lookup', () => {
  it('finds the pair for a finding', () => {
    expect(pairFor([pair()], 'f1')?.clauseId).toBe('AF-09');
  });

  it('returns null rather than falling back to another finding', () => {
    // Showing a neighbouring finding's frames under this finding's label
    // would be evidence for a claim nobody made.
    expect(pairFor([pair()], 'f-other')).toBeNull();
    expect(pairFor([pair()], null)).toBeNull();
  });
});

describe('seek targets', () => {
  it('seeks the original timeline before the fix is applied', () => {
    expect(seekTarget(pair(), false)).toBe(5_000);
  });

  it('seeks the mapped timestamp after the fix is applied', () => {
    // Not 5000. The cut moved this moment, and seeking the remediated player
    // to the original timestamp lands on unrelated material.
    expect(seekTarget(pair(), true)).toBe(3_000);
  });

  it('refuses to seek to removed evidence', () => {
    const removed = pair({
      after: {
        runId: 'run-b',
        tsMs: null,
        frame: null,
        removedByRemediation: true,
        unavailable: 'EVIDENCE REMOVED BY REMEDIATION',
      },
    });
    // Null, not 0. Returning 0 would seek to the start of the video and
    // quietly assert the evidence is there.
    expect(seekTarget(removed, true)).toBeNull();
  });

  it('refuses to seek when no before frame was extracted', () => {
    const unmeasured = pair({
      before: { ...pair().before, frame: null },
    });
    expect(seekTarget(unmeasured, false)).toBeNull();
  });
});

describe('lifecycle steps', () => {
  it('only lists states the backend actually persists', () => {
    // A step the engine never enters would sit pending forever, reading as a
    // stalled pipeline on every successful run.
    const real: LifecycleState[] = [
      'REMEDIATION_REQUESTED',
      'RENDERING',
      'STRUCTURAL_VERIFYING',
      'STRUCTURALLY_VALID',
      'REANALYSING',
      'COMPARING',
    ];
    expect(LIFECYCLE_STEPS.map((s) => s.state)).toEqual(real);
  });

  it('puts the render before the checks that judge it', () => {
    const order = LIFECYCLE_STEPS.map((s) => s.state);
    expect(order.indexOf('RENDERING')).toBeLessThan(
      order.indexOf('STRUCTURAL_VERIFYING'),
    );
    expect(order.indexOf('STRUCTURAL_VERIFYING')).toBeLessThan(
      order.indexOf('REANALYSING'),
    );
    expect(order.indexOf('REANALYSING')).toBeLessThan(order.indexOf('COMPARING'));
  });

  it('marks the current state active and earlier ones done', () => {
    const seen = ['REMEDIATION_REQUESTED', 'RENDERING'];
    expect(stepStatus('RENDERING', 'REANALYSING', seen, null)).toBe('done');
    expect(stepStatus('REANALYSING', 'REANALYSING', seen, null)).toBe('active');
    expect(stepStatus('COMPARING', 'REANALYSING', seen, null)).toBe('pending');
  });

  it('infers completion positionally after a reload mid-run', () => {
    // A page reloaded mid-remediation never received the earlier events, so
    // `seen` is empty. The steps before the current one still ran.
    expect(stepStatus('RENDERING', 'COMPARING', [], null)).toBe('done');
  });

  it('shows every step done once the remediation is terminal', () => {
    for (const step of LIFECYCLE_STEPS) {
      expect(stepStatus(step.state, null, [], record())).toBe('done');
    }
  });

  it('marks the failing step failed and keeps the earlier ones done', () => {
    const failed = record({
      state: 'FAILED',
      previousState: 'RENDERING',
      terminal: true,
    });
    expect(stepStatus('REMEDIATION_REQUESTED', null, [], failed)).toBe('done');
    expect(stepStatus('RENDERING', null, [], failed)).toBe('failed');
    // Nothing after the failure ran, and must not read as if it had.
    expect(stepStatus('REANALYSING', null, [], failed)).toBe('pending');
    expect(stepStatus('COMPARING', null, [], failed)).toBe('pending');
  });

  it('never reports a step done before anything has started', () => {
    for (const step of LIFECYCLE_STEPS) {
      expect(stepStatus(step.state, null, [], null)).toBe('pending');
    }
  });
});
