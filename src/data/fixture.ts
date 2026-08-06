import type {
  AgentRun,
  AnalysisReport,
  Finding,
  RemediationOp,
  SubScores,
  VideoMeta,
} from '@/types/analysis';
import { computeReadiness } from '@/lib/scoring';
import { compileRemediation } from '@/lib/ffmpeg';
import { buildRiskBands } from '@/lib/risk';
import { buildBreakdown } from '@/lib/findings';
import { computeCoverage } from '@/lib/coverage';

/**
 * The demo fixture.
 *
 * This is the shape the Python engine will POST. Everything downstream of it is
 * computed: scores run through `computeReadiness`, the risk terrain through
 * `buildRiskBands`, the ffmpeg program through `compileRemediation`, coverage
 * through `computeCoverage`. Nothing on the page is a typed-in number.
 */

const DURATION_MS = 1_122_000; // 18:42

/** Resolve a highlight span by searching the transcript, so offsets cannot drift. */
function mark(text: string, needle: string): [number, number] {
  const start = text.indexOf(needle);
  if (start < 0) throw new Error(`fixture: highlight "${needle}" not found in transcript`);
  return [start, start + needle.length];
}

/* ------------------------------------------------------------------ */
/* Video + run metadata                                                */
/* ------------------------------------------------------------------ */

const video: VideoMeta = {
  filename: 'documentary.mp4',
  durationMs: DURATION_MS,
  width: 1920,
  height: 1080,
  fps: 30,
  sizeBytes: 1_331_439_862, // 1.24 GB
  audioCodec: 'AAC',
  sampleRate: 48_000,
  posterUrl: '/media/documentary-poster.jpg',
  srcUrl: '/media/documentary.mp4',
};

/* ------------------------------------------------------------------ */
/* Agents — the 12-node DAG                                            */
/* ------------------------------------------------------------------ */

const agents: AgentRun[] = [
  {
    id: 'orchestrator',
    name: 'Orchestrator',
    tier: 0,
    parents: [],
    status: 'OK',
    detail: '12 agents scheduled · 6 lanes · 30 rpm token bucket',
    coverage: 1,
    elapsedMs: 34_600,
    tsMs: 0,
    calls: 0,
  },
  {
    id: 'ingest',
    name: 'Ingest Agent',
    tier: 1,
    parents: ['orchestrator'],
    status: 'OK',
    detail: 'Demuxed 18:42 · 16 kHz mono WAV + 44.1 kHz stereo fingerprint track',
    coverage: 1,
    elapsedMs: 2_140,
    tsMs: 300,
    calls: 0,
  },
  {
    id: 'speech',
    name: 'Speech Agent',
    tier: 2,
    parents: ['ingest'],
    status: 'OK',
    detail: 'faster-whisper base.en · 4,318 words · word-level timestamps',
    coverage: 1,
    elapsedMs: 5_820,
    tsMs: 2_400,
    calls: 0,
  },
  {
    id: 'vision',
    name: 'Vision Agent',
    tier: 2,
    parents: ['ingest'],
    status: 'DEGRADED',
    detail: 'Rate limited at frame 774/1,842 — sampled remainder uniformly',
    coverage: 0.42,
    elapsedMs: 9_310,
    tsMs: 8_200,
    calls: 31,
  },
  {
    id: 'ocr',
    name: 'OCR Agent',
    tier: 3,
    parents: ['vision'],
    status: 'DEGRADED',
    detail: 'Read 1,253 of 1,842 keyframes — limited by upstream vision coverage',
    coverage: 0.68,
    elapsedMs: 4_460,
    tsMs: 14_000,
    calls: 14,
  },
  {
    id: 'audio',
    name: 'Audio Agent',
    tier: 2,
    parents: ['ingest'],
    status: 'OK',
    detail: 'Chromaprint fingerprint → AcoustID lookup · 74 windows · 1 match',
    coverage: 1,
    elapsedMs: 3_900,
    tsMs: 18_000,
    calls: 74,
  },
  {
    id: 'access',
    name: 'Accessibility Agent',
    tier: 3,
    parents: ['speech'],
    status: 'OK',
    detail: 'No caption track · contrast pass on 96 sampled frames',
    coverage: 1,
    elapsedMs: 1_180,
    tsMs: 20_500,
    calls: 0,
  },
  {
    id: 'meta',
    name: 'Metadata Agent',
    tier: 2,
    parents: ['ingest'],
    status: 'OK',
    detail: 'Title, description, tags, chapter markers linted',
    coverage: 1,
    elapsedMs: 640,
    tsMs: 21_400,
    calls: 0,
  },
  {
    id: 'policy',
    name: 'Policy Agent',
    tier: 4,
    parents: ['speech', 'vision', 'ocr', 'audio'],
    status: 'OK',
    detail: 'RRF retrieval over 14 clauses → AUDITOR ▸ ADVOCATE ▸ ADJUDICATOR',
    coverage: 1,
    elapsedMs: 7_900,
    tsMs: 23_000,
    calls: 17,
  },
  {
    id: 'score',
    name: 'Scoring Agent',
    tier: 5,
    parents: ['policy', 'access', 'meta', 'audio'],
    status: 'OK',
    detail: 'Cross-modal fusion → release readiness · clamped at weakest + 15',
    coverage: 1,
    elapsedMs: 210,
    tsMs: 31_000,
    calls: 0,
  },
  {
    id: 'remedy',
    name: 'Remediation Agent',
    tier: 6,
    parents: ['score'],
    status: 'OK',
    detail: 'Lowered 4 findings to EDL · 6 optimiser passes · emitted ffmpeg',
    coverage: 1,
    elapsedMs: 480,
    tsMs: 32_000,
    calls: 0,
  },
  {
    id: 'report',
    name: 'Report Agent',
    tier: 7,
    parents: ['remedy'],
    status: 'OK',
    detail: 'report.sarif · report.html · attestation signed',
    coverage: 1,
    elapsedMs: 390,
    tsMs: 34_000,
    calls: 0,
  },
];

/* ------------------------------------------------------------------ */
/* Findings                                                            */
/* ------------------------------------------------------------------ */

const T_VIOLENCE =
  "…he slipped and went straight down onto the rocks. Look at the blood — that's a bad one. " +
  'Get the kit open, get pressure on it now.';

const T_PROFANITY =
  'The anchor pulled clean out of the ice and the whole shelf just went. This is fucked, ' +
  'we need to be off this face in ten minutes.';

const T_MUSIC =
  '[no speech · sustained music bed under drone footage · 25.0 s]';

const T_SENSITIVE =
  'Everyone here still talks about the 2019 slide on the north col. Eleven people went out ' +
  "that morning and four of them didn't come back down.";

const T_ALCOHOL =
  "We cracked the whisky we'd been hauling since base camp — one finger each, purely medicinal.";

const T_OCR =
  '[on-screen text overlay · burned-in group chat capture]  "no way in hell im going back up ' +
  'that ridge, that guide is a f***ing liability"';

const T_DANGEROUS =
  'We skipped the fixed line entirely here. No rope, no anchor — just downclimbing the gully ' +
  'on wet rock because we were losing light.';

const T_CONTROVERSIAL =
  'Half the permits get handed to operators who have never set foot on the mountain, and the ' +
  'ministry knows exactly what it is doing when it signs them.';

const T_FILE_SCOPE = '[file-scoped finding · no single evidence span]';

const findings: Finding[] = [
  {
    id: 'f_01',
    clauseId: 'AF-04',
    category: 'Violence',
    title: 'Graphic injury with visible blood',
    description:
      'Close, in-focus shot of an open wound with audible commentary drawing attention to it.',
    startMs: 454_000,
    endMs: 457_000,
    severity: 'CRITICAL',
    confidence: 0.94,
    modalities: { vision: 0.92, speech: 0.88, ocr: 0.11 },
    fusedConfidence: 0.94,
    evidence: {
      transcript: T_VIOLENCE,
      highlightSpan: mark(T_VIOLENCE, "Look at the blood — that's a bad one."),
      frames: ['/media/frames/f01_454200.jpg', '/media/frames/f01_455600.jpg', '/media/frames/f01_456800.jpg'],
    },
    policy: {
      clauseId: 'AF-04',
      title: 'Violence',
      section: 'Advertiser-friendly guidelines § 2.2.2',
      text:
        'Content showing real or dramatised violence, injury, or blood in a way that is likely ' +
        'to shock viewers is not suitable for most advertisers. Brief, non-focal, or clearly ' +
        'educational depictions may remain eligible for limited ads where the injury is not the ' +
        'subject of the shot.',
    },
    adversarial: {
      auditor: {
        charge:
          'Wound occupies roughly a third of the frame for 2.8 s and is held in focus. Narration ' +
          'directs the viewer to it, which removes any argument that the shot is incidental.',
      },
      advocate: {
        defense:
          'Expedition documentary with a clear safety-education throughline; the sequence is ' +
          'brief and immediately cuts to first-aid procedure rather than lingering.',
        strength: 0.41,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'The clause exempts non-focal depictions. Here the injury is the subject of the shot ' +
          'and is verbally emphasised, so the educational framing does not reach the exemption.',
        confidence: 0.94,
      },
    },
    suggestedFix: 'BLUR_REGION',
  },
  {
    id: 'f_02',
    clauseId: 'AF-01',
    category: 'Language',
    title: 'Strong profanity in the first half',
    description:
      'Unbleeped strong profanity at 04:12, inside the window the classifier weights most heavily.',
    startMs: 252_400,
    endMs: 254_100,
    severity: 'HIGH',
    confidence: 0.91,
    modalities: { speech: 0.96, vision: 0.04 },
    fusedConfidence: 0.91,
    evidence: {
      transcript: T_PROFANITY,
      highlightSpan: mark(T_PROFANITY, 'This is fucked'),
      frames: ['/media/frames/f02_252600.jpg'],
    },
    policy: {
      clauseId: 'AF-01',
      title: 'Inappropriate language',
      section: 'Advertiser-friendly guidelines § 2.1.1',
      text:
        'Frequent use of strong profanity, or use of strong profanity in the first seven seconds ' +
        'or in the title or thumbnail, limits advertising. Isolated strong profanity later in a ' +
        'video may remain eligible for full monetisation depending on context and frequency.',
    },
    adversarial: {
      auditor: {
        charge:
          'Strong profanity, unbleeped, clearly intelligible in the mix at 04:12.4. Word-level ' +
          'ASR confidence 0.96.',
      },
      advocate: {
        defense:
          'Single occurrence, well past the opening window, spoken under genuine duress rather ' +
          'than gratuitously. The clause treats isolated use as potentially eligible.',
        strength: 0.55,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'Isolated use is not automatically exempt under this clause and the term is fully ' +
          'intelligible. A 1.7 s bleep removes the exposure entirely at no narrative cost.',
        confidence: 0.91,
      },
    },
    suggestedFix: 'BLEEP',
  },
  {
    id: 'f_03',
    clauseId: 'CID-01',
    category: 'Copyright',
    title: 'Commercial recording matched under drone sequence',
    description:
      'Chromaprint fingerprint resolved to a commercially released recording across 25 s of B-roll.',
    startMs: 775_000,
    endMs: 800_000,
    severity: 'HIGH',
    confidence: 0.89,
    modalities: { audio: 0.89 },
    fusedConfidence: 0.89,
    evidence: {
      transcript: T_MUSIC,
      highlightSpan: [0, 0],
      frames: ['/media/frames/f03_778000.jpg', '/media/frames/f03_790000.jpg'],
    },
    policy: {
      clauseId: 'CID-01',
      title: 'Third-party content',
      section: 'Copyright policy § 1.4 — Content ID',
      text:
        'Uploading a commercially released sound recording without a licence permits the rights ' +
        'holder to claim the video, redirecting revenue for its full duration or blocking it in ' +
        'some territories. A claim may be applied at any time after upload.',
    },
    adversarial: {
      auditor: {
        charge:
          'Fingerprint match across 5 consecutive 30 s windows, mean AcoustID score 0.89. Match ' +
          'covers 25 s of continuous audio.',
      },
      advocate: {
        defense: null,
        strength: 0,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'A public fingerprint match to a commercial recording predicts a Content ID claim. ' +
          'Absence of a match would not prove safety, but a match is decisive in this direction.',
        confidence: 0.89,
      },
    },
    suggestedFix: 'REPLACE_AUDIO',
  },
  {
    id: 'f_04',
    clauseId: 'AF-10',
    category: 'Sensitive Events',
    title: 'Fatalities from a named real-world incident',
    description:
      'Specific casualty count attached to an identifiable recent event, stated without framing.',
    startMs: 910_000,
    endMs: 916_000,
    severity: 'MEDIUM',
    confidence: 0.76,
    modalities: { speech: 0.81, vision: 0.22 },
    fusedConfidence: 0.76,
    evidence: {
      transcript: T_SENSITIVE,
      highlightSpan: mark(T_SENSITIVE, "four of them didn't come back down"),
      frames: ['/media/frames/f04_911500.jpg'],
    },
    policy: {
      clauseId: 'AF-10',
      title: 'Sensitive events',
      section: 'Advertiser-friendly guidelines § 2.4.1',
      text:
        'Content discussing a tragedy, death, or disaster in which people were harmed may be ' +
        'unsuitable for advertising where the discussion is graphic or dwells on the loss. ' +
        'News reporting and factual commentary that does not sensationalise remain eligible.',
    },
    adversarial: {
      auditor: {
        charge:
          'Explicit casualty figure for a named event. The clause covers discussion of deaths ' +
          'in a specific disaster.',
      },
      advocate: {
        defense:
          'Factual, non-graphic, six seconds long, and delivered as context for the route rather ' +
          'than dwelt upon — squarely inside the factual-commentary exemption.',
        strength: 0.62,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'The defence is credible and the severity is reduced accordingly, but the casualty ' +
          'figure is stated without surrounding framing, which the clause treats as dwelling.',
        confidence: 0.76,
      },
    },
    suggestedFix: 'MUTE',
  },
  {
    id: 'f_05',
    clauseId: 'AF-12',
    category: 'Regulated Goods',
    title: 'Brief alcohol reference',
    description:
      'Passing mention of drinking spirits, no consumption shown. Advisory only — no fix generated.',
    startMs: 131_000,
    endMs: 138_000,
    severity: 'LOW',
    confidence: 0.6,
    modalities: { speech: 0.6, vision: 0.18 },
    fusedConfidence: 0.6,
    evidence: {
      transcript: T_ALCOHOL,
      highlightSpan: mark(T_ALCOHOL, "cracked the whisky"),
      frames: ['/media/frames/f05_133000.jpg'],
    },
    policy: {
      clauseId: 'AF-12',
      title: 'Regulated goods — alcohol',
      section: 'Advertiser-friendly guidelines § 2.7.2',
      text:
        'Incidental references to alcohol in content aimed at adult audiences generally remain ' +
        'eligible for monetisation. Content that promotes excessive consumption, or features ' +
        'alcohol in content made for kids, is not.',
    },
    adversarial: {
      auditor: {
        charge: 'Named spirit brand category referenced and consumption implied on camera.',
      },
      advocate: {
        defense:
          'Incidental, adult-audience content, no promotion of excess, no consumption shown. ' +
          'The clause explicitly permits incidental adult-audience references.',
        strength: 0.78,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'Retained at ADVISORY only. The exemption very nearly applies; flagged so the creator ' +
          'can decide, not because the clause is breached.',
        confidence: 0.6,
      },
    },
    suggestedFix: 'NONE',
  },
  {
    id: 'f_06',
    clauseId: 'AF-01',
    category: 'Language',
    title: 'Masked profanity in burned-in overlay',
    description:
      'Text is composited into the source render — cannot be removed downstream. Re-export required.',
    startMs: 362_000,
    endMs: 365_000,
    severity: 'MEDIUM',
    confidence: 0.71,
    modalities: { ocr: 0.71, vision: 0.44 },
    fusedConfidence: 0.71,
    evidence: {
      transcript: T_OCR,
      highlightSpan: mark(T_OCR, 'f***ing liability'),
      frames: ['/media/frames/f06_363000.jpg', '/media/frames/f06_364200.jpg'],
    },
    policy: {
      clauseId: 'AF-01',
      title: 'Inappropriate language',
      section: 'Advertiser-friendly guidelines § 2.1.3',
      text:
        'Profanity appearing in on-screen text, thumbnails, or titles is assessed the same way as ' +
        'spoken profanity. Censored or partially masked terms are weighted lower than uncensored ' +
        'terms but are still assessed.',
    },
    adversarial: {
      auditor: {
        charge: 'Partially masked strong profanity held on screen for 3.0 s at readable size.',
      },
      advocate: {
        defense:
          'Term is masked, quoted from a third party, and never spoken aloud. The clause weights ' +
          'masked terms lower.',
        strength: 0.58,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'Masking reduces but does not remove exposure under this clause. Held at MEDIUM. No ' +
          'automated fix: the overlay is baked into the video stream.',
        confidence: 0.71,
      },
    },
    suggestedFix: 'NONE',
  },
  {
    id: 'f_07',
    clauseId: 'AF-05',
    category: 'Dangerous Acts',
    title: 'Unroped descent presented without warning',
    description:
      'Imitable high-risk technique shown approvingly. Recommend an on-screen disclaimer card.',
    startMs: 588_000,
    endMs: 604_000,
    severity: 'MEDIUM',
    confidence: 0.68,
    modalities: { vision: 0.7, speech: 0.66 },
    fusedConfidence: 0.68,
    evidence: {
      transcript: T_DANGEROUS,
      highlightSpan: mark(T_DANGEROUS, 'No rope, no anchor'),
      frames: ['/media/frames/f07_590000.jpg', '/media/frames/f07_598000.jpg'],
    },
    policy: {
      clauseId: 'AF-05',
      title: 'Harmful or dangerous acts',
      section: 'Advertiser-friendly guidelines § 2.3.1',
      text:
        'Content depicting dangerous acts that viewers could imitate and be seriously injured by ' +
        'is not suitable for most advertisers, unless it carries sufficient educational or ' +
        'documentary context, including warnings against imitation.',
    },
    adversarial: {
      auditor: {
        charge:
          'Sixteen seconds of unprotected downclimbing on wet rock, narrated matter-of-factly ' +
          'with no warning against imitation.',
      },
      advocate: {
        defense:
          'Documentary context with professional subjects; the narration frames the decision as ' +
          'a mistake forced by fading light rather than as advice.',
        strength: 0.6,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'The clause requires an explicit warning against imitation for the documentary ' +
          'exemption to apply. None is present. A disclaimer card would likely clear this.',
        confidence: 0.68,
      },
    },
    suggestedFix: 'NONE',
  },
  {
    id: 'f_08',
    clauseId: 'AF-09',
    category: 'Controversial Issues',
    title: 'Allegation against a named public body',
    description:
      'Unsourced claim of institutional misconduct. Advisory — add attribution to clear it.',
    startMs: 680_000,
    endMs: 712_000,
    severity: 'LOW',
    confidence: 0.55,
    modalities: { speech: 0.55 },
    fusedConfidence: 0.55,
    evidence: {
      transcript: T_CONTROVERSIAL,
      highlightSpan: mark(T_CONTROVERSIAL, 'the ministry knows exactly what it is doing'),
      frames: ['/media/frames/f08_690000.jpg'],
    },
    policy: {
      clauseId: 'AF-09',
      title: 'Controversial issues',
      section: 'Advertiser-friendly guidelines § 2.5.1',
      text:
        'Discussion of contentious political or social topics may limit advertising where the ' +
        'treatment is inflammatory or one-sided. Balanced, sourced reporting generally remains ' +
        'eligible.',
    },
    adversarial: {
      auditor: {
        charge: 'Direct allegation of corruption against a named government body, unsourced.',
      },
      advocate: {
        defense:
          'Thirty-two seconds inside an 18-minute film, delivered as first-hand observation ' +
          'rather than as an inflammatory appeal.',
        strength: 0.71,
      },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'Held at ADVISORY. The clause turns on whether the treatment is sourced; adding an ' +
          'on-screen citation would likely move this to DISMISSED on a re-run.',
        confidence: 0.55,
      },
    },
    suggestedFix: 'NONE',
  },
  {
    id: 'f_09',
    clauseId: 'ACC-01',
    category: 'Accessibility',
    title: 'No caption track present',
    description:
      'Word-level transcript already exists in this run — captions can be emitted directly from it.',
    startMs: 0,
    endMs: DURATION_MS,
    severity: 'HIGH',
    confidence: 0.99,
    modalities: { access: 0.99 },
    fusedConfidence: 0.99,
    evidence: { transcript: T_FILE_SCOPE, highlightSpan: [0, 0], frames: [] },
    policy: {
      clauseId: 'ACC-01',
      title: 'Caption availability',
      section: 'PREFLIGHT accessibility ruleset § 1.1',
      text:
        'A published video should ship with a caption track. Automatic captions are not a ' +
        'substitute where the audio contains technical vocabulary, accents, or wind noise, all ' +
        'of which are present in this file.',
    },
    adversarial: {
      auditor: {
        charge: 'No sidecar caption track and no embedded timed-text stream in the container.',
      },
      advocate: { defense: null, strength: 0 },
      adjudicator: {
        verdict: 'UPHELD',
        rationale:
          'Deterministic check, not a judgement call. The speech agent already produced ' +
          'word-level timings, so the remediation cost is effectively zero.',
        confidence: 0.99,
      },
    },
    suggestedFix: 'NONE',
  },
  {
    id: 'f_10',
    clauseId: 'META-01',
    category: 'Metadata',
    title: 'Description missing chapters and affiliate disclosure',
    description:
      'No chapter markers across 18:42, and two affiliate links carry no disclosure line.',
    startMs: 0,
    endMs: DURATION_MS,
    severity: 'MEDIUM',
    confidence: 0.83,
    modalities: { meta: 0.83 },
    fusedConfidence: 0.83,
    evidence: { transcript: T_FILE_SCOPE, highlightSpan: [0, 0], frames: [] },
    policy: {
      clauseId: 'META-01',
      title: 'Paid promotion disclosure',
      section: 'PREFLIGHT metadata ruleset § 2.3',
      text:
        'Descriptions containing affiliate or sponsored links must carry a disclosure. Videos ' +
        'over ten minutes without chapter markers lose retention on Suggested surfaces.',
    },
    adversarial: {
      auditor: {
        charge: 'Two shortened outbound links matched known affiliate domains; no disclosure text.',
      },
      advocate: { defense: null, strength: 0 },
      adjudicator: {
        verdict: 'UPHELD',
        rationale: 'Deterministic string check against the description. No interpretation involved.',
        confidence: 0.83,
      },
    },
    suggestedFix: 'NONE',
  },
  {
    id: 'f_11',
    clauseId: 'AUD-01',
    category: 'Audio Delivery',
    title: 'Integrated loudness 5.2 LU above target',
    description:
      'Measured −8.8 LUFS against a −14 LUFS target; playback will attenuate and flatten dynamics.',
    startMs: 0,
    endMs: DURATION_MS,
    severity: 'LOW',
    confidence: 0.95,
    modalities: { audio: 0.95 },
    fusedConfidence: 0.95,
    evidence: { transcript: T_FILE_SCOPE, highlightSpan: [0, 0], frames: [] },
    policy: {
      clauseId: 'AUD-01',
      title: 'Loudness normalisation',
      section: 'PREFLIGHT audio ruleset § 3.1',
      text:
        'Playback normalises loud uploads downward. Delivering above target does not increase ' +
        'perceived volume; it only reduces headroom and dynamic range after normalisation.',
    },
    adversarial: {
      auditor: {
        charge: 'EBU R128 integrated loudness −8.8 LUFS, true peak −0.2 dBTP across the programme.',
      },
      advocate: { defense: null, strength: 0 },
      adjudicator: {
        verdict: 'UPHELD',
        rationale: 'Measured value, not a classification. Correctable with a single loudnorm pass.',
        confidence: 0.95,
      },
    },
    suggestedFix: 'NONE',
  },
];

/* ------------------------------------------------------------------ */
/* Remediation EDL                                                     */
/* ------------------------------------------------------------------ */

const ops: RemediationOp[] = [
  {
    index: 1,
    op: 'BLEEP',
    startMs: 252_400,
    endMs: 254_100,
    details: 'Strong profanity · 1 kHz tone · snapped to word boundaries',
    findingId: 'f_02',
    freqHz: 1000,
  },
  {
    index: 2,
    op: 'BLUR_REGION',
    startMs: 454_000,
    endMs: 457_000,
    details: 'Wound region · boxblur 20:2 · 42% × 30% of frame',
    findingId: 'f_01',
    box: [0.29, 0.35, 0.42, 0.3],
  },
  {
    index: 3,
    op: 'REPLACE_AUDIO',
    startMs: 775_000,
    endMs: 800_000,
    details: 'Copyrighted bed → CC0 replacement, level-matched',
    findingId: 'f_03',
    asset: 'assets/cc_music/glacier_calm.mp3',
  },
  {
    index: 4,
    op: 'MUTE',
    startMs: 910_000,
    endMs: 916_000,
    details: 'Casualty figure · 60 ms head / 80 ms tail padding',
    findingId: 'f_04',
  },
];

const compiled = compileRemediation(ops, 'documentary.mp4', 'documentary.safe.mp4');

/* ------------------------------------------------------------------ */
/* Reports                                                             */
/* ------------------------------------------------------------------ */

const beforeSub: SubScores = {
  policy: 31,
  copyright: 19,
  metadata: 78,
  accessibility: 62,
  audio: 88,
};

const afterSub: SubScores = {
  policy: 96,
  copyright: 100,
  metadata: 94,
  accessibility: 88,
  audio: 95,
};

const beforeReadiness = computeReadiness(beforeSub);
const afterReadiness = computeReadiness(afterSub);

const coverage = computeCoverage(agents);

/** Findings the remediation plan does not clear — carried into the after report. */
const RESIDUAL_IDS = new Set(['f_05', 'f_06', 'f_07', 'f_08']);

export const beforeReport: AnalysisReport = {
  video,
  meta: {
    analyzedAt: '2026-08-04T14:32:34Z',
    policyVersion: '2026.08.01',
    engineVersion: '1.0.0',
    attestationHash: 'b3:a7c9f2e6d8b0c4f98e1b2d3c4a5f6071829304b5c6d7e8f90a1b2c3d4e5f6071',
    coverage,
  },
  scores: {
    overall: beforeReadiness.overall,
    sub: beforeSub,
    verdict: beforeReadiness.verdict,
    weakest: beforeReadiness.weakest,
  },
  riskBands: buildRiskBands(findings, DURATION_MS),
  findings,
  breakdown: buildBreakdown(findings),
  remediation: {
    ops,
    ffmpegCommand: compiled.command,
    renderMs: 4_200,
    videoStreamCopied: compiled.videoStreamCopied,
    log: [],
  },
  agents,
};

const afterFindings = findings.filter((f) => RESIDUAL_IDS.has(f.id));

export const afterReport: AnalysisReport = {
  ...beforeReport,
  video: { ...video, filename: 'documentary.safe.mp4', srcUrl: '/media/documentary-safe.mp4' },
  meta: {
    ...beforeReport.meta,
    attestationHash: 'b3:5e1d0c9b8a7f6e5d4c3b2a1908f7e6d5c4b3a29180f7e6d5c4b3a2918077f6e5',
  },
  scores: {
    overall: afterReadiness.overall,
    sub: afterSub,
    verdict: afterReadiness.verdict,
    weakest: afterReadiness.weakest,
  },
  riskBands: buildRiskBands(afterFindings, DURATION_MS),
  findings: afterFindings,
  breakdown: buildBreakdown(afterFindings),
  // The safe render carries no outstanding ops — the plan has already been applied.
  remediation: { ops: [], ffmpegCommand: '', renderMs: 0, videoStreamCopied: false, log: [] },
};

export const DEMO_DURATION_MS = DURATION_MS;
