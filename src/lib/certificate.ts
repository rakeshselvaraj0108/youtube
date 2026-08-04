import type { AnalysisReport } from '@/types/analysis';
import { computeReadiness, WEIGHTS } from '@/lib/scoring';
import { exitCode } from '@/lib/sarif';

/**
 * The release certificate.
 *
 * An editor hands this to a client to prove exactly what was checked, against
 * which version of the rules, with which models. That is why it carries the
 * policy corpus version and the coverage figure alongside the score: a
 * certificate that only says "94/100" is marketing, one that says "94/100,
 * against policy 2026.08.01, having inspected 83% of the analysis surface" is
 * evidence.
 */

export const MODELS = {
  asr: 'faster-whisper base.en (local)',
  auditor: 'meta/llama-3.3-70b-instruct',
  advocate: 'nvidia/llama-3.3-nemotron-super-49b-v1',
  adjudicator: 'nvidia/llama-3.3-nemotron-super-49b-v1',
  embed: 'nvidia/nv-embedqa-e5-v5',
  rerank: 'nvidia/nv-rerankqa-mistral-4b-v3',
  ocr: 'baidu/paddleocr',
} as const;

export function buildCertificate(report: AnalysisReport) {
  const readiness = computeReadiness(report.scores.sub);

  return {
    certificateVersion: '1.0',
    generatedAt: report.meta.analyzedAt,

    subject: {
      filename: report.video.filename,
      durationMs: report.video.durationMs,
      resolution: `${report.video.width}x${report.video.height}`,
      fps: report.video.fps,
      sizeBytes: report.video.sizeBytes,
    },

    attestation: {
      attestationHash: report.meta.attestationHash,
      policyVersion: report.meta.policyVersion,
      engineVersion: report.meta.engineVersion,
      models: MODELS,
    },

    readiness: {
      overall: readiness.overall,
      verdict: report.scores.verdict,
      weakestDimension: readiness.weakest,
      subScores: report.scores.sub,
      weights: WEIGHTS,
      // The scoring rule travels with the score, so a reader can recompute it.
      clamp: 'overall = min(weightedMean, worst + 15)',
      weightedMean: Number(readiness.weighted.toFixed(2)),
      clampBound: readiness.capped,
    },

    coverage: {
      overall: Number(report.meta.coverage.toFixed(4)),
      agents: report.agents.map((agent) => ({
        id: agent.id,
        name: agent.name,
        status: agent.status,
        coverage: agent.coverage,
        elapsedMs: agent.elapsedMs,
        calls: agent.calls,
      })),
    },

    findings: {
      total: report.findings.length,
      bySeverity: {
        CRITICAL: report.findings.filter((f) => f.severity === 'CRITICAL').length,
        HIGH: report.findings.filter((f) => f.severity === 'HIGH').length,
        MEDIUM: report.findings.filter((f) => f.severity === 'MEDIUM').length,
        LOW: report.findings.filter((f) => f.severity === 'LOW').length,
      },
      clauses: [...new Set(report.findings.map((f) => f.clauseId))].sort(),
    },

    remediation: {
      operations: report.remediation.ops.length,
      renderMs: report.remediation.renderMs,
      videoStreamCopied: report.remediation.videoStreamCopied,
      ffmpegCommand: report.remediation.ffmpegCommand,
    },

    ci: { exitCode: exitCode(report) },

    /* PREFLIGHT predicts risk against published policy. It is not YouTube's
       classifier and cannot be. A fingerprint match to a commercial recording
       predicts a Content ID claim; the absence of one does not prove safety. */
    limitations: [
      'PREFLIGHT predicts risk against published policy. It is not YouTube classifier output.',
      'A public fingerprint match predicts a Content ID claim. No match does not prove safety.',
      `Analysis covered ${Math.round(report.meta.coverage * 100)}% of the surface; see coverage.agents.`,
    ],
  };
}
