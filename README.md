# PREFLIGHT — Command Deck

**Static analysis and CI for video.** *Ship your video like you ship your code.*

A creator uploads to YouTube and only afterwards learns whether the video will
be monetized — by which point the first 48 hours, where most of a video's
revenue lives, are gone. PREFLIGHT analyses the file *before* upload: it cites
the specific policy clause behind every finding, shows the adversarial record
that produced the verdict, and compiles an executable ffmpeg program that
repairs the video.

This repository is the **Command Deck** — the single-page analysis report that
is the entire user-facing surface of the product.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation — tokens, data contract, scoring, ffmpeg codegen, fixtures | **done** |
| 2 | Static layout — every panel in final position with real data | pending |
| 3 | Interaction — selection, tabs, sorting, downloads, linked players | pending |
| 4 | Motion — load orchestration, terminal typing, the fix transition | pending |
| 5 | The three 3D moments — risk terrain, gauge depth ring, agent graph | pending |
| 6 | Final pass — self-critique, remove what is doing the least work | pending |

`src/App.tsx` is currently a Phase 1 foundation probe, not the deck. It renders
the computed layer so the numbers can be verified before any pixels are designed.

---

## Quick start

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # 58 tests
npm run build      # typecheck + production bundle
```

---

## The two things the UI has to make visible

**1. Findings are grounded, not guessed.** Every finding cites a policy clause
verbatim and carries the three-agent adversarial record — AUDITOR prosecutes,
ADVOCATE defends, ADJUDICATOR rules. The reasoning is auditable.

**2. Findings are compiled, not suggested.** They lower into an Edit Decision
List, get optimised, and emit real ffmpeg. `src/lib/ffmpeg.ts` is the code
generator; it is not a template string. Moving an op moves the command, and the
test suite proves it.

---

## Scoring — the anti-masking clamp

Release Readiness is a weighted mean of five dimensions, **capped at
`weakest + 15`**:

```ts
overall = min(weighted, worst + 15)
```

The demo video scores 72.2 on a plain weighted average — a passing grade — while
carrying a confirmed Content ID match that would redirect 100% of its revenue.
The clamp brings it to **34 / DO NOT PUBLISH**. One fatal flaw is never averaged
away; that property is the reason the number is worth trusting, and it is
unit-tested at every verdict boundary.

Every dimension runs the same direction. **100 is always good** — a bar that is
90% full means the same thing whether it is measuring copyright exposure or
caption coverage.

| Condition | Verdict |
|---|---|
| `overall ≥ 85 && worst ≥ 70` | READY TO PUBLISH |
| `overall ≥ 70 && worst ≥ 50` | PUBLISH WITH FIXES |
| `overall ≥ 50` | NOT READY |
| else | DO NOT PUBLISH |

---

## Coverage honesty

The report header carries a `COVERAGE 83%` chip naming the degraded vision
agent. A compliance tool that reports what it *could not* see is more
trustworthy than one that quietly returns a clean bill of health. Coverage is a
weighted mean over agents, weighted by each agent's share of the analysis
surface — the vision agent reaching 42% of keyframes costs far more coverage
than the report writer.

---

## Layout

```
src/
├── types/analysis.ts    the data contract — everything derives from this
├── lib/
│   ├── scoring.ts       readiness, the clamp, verdicts, signal ramps
│   ├── ffmpeg.ts        EDL → filter_complex code generator
│   ├── risk.ts          findings → risk terrain
│   ├── findings.ts      category rollup, sorting
│   ├── coverage.ts      weighted coverage + degraded-agent reporting
│   └── time.ts          timecodes, programmatic timeline ticks
└── data/fixture.ts      before/after demo reports
```

Zero hardcoded values live in components. When the Python engine replaces the
fixture with a real response, nothing under `src/components` should need to
change.

---

## Stack

Vite · React 18 · TypeScript (strict) · Tailwind (tokens in
`tailwind.config.ts`) · Framer Motion · React Three Fiber · Zustand ·
lucide-react. No UI kit — every panel is built.

---

## Honest limits

PREFLIGHT predicts risk against published policy. It is not YouTube's
classifier and cannot be. A fingerprint match to a commercial recording predicts
a Content ID claim; the absence of a match does **not** prove safety, because
Content ID's reference database is private and larger than any public one. The
tool reports `CLAIM_LIKELY` and `NO_PUBLIC_MATCH` — never `SAFE`.

---

MIT
