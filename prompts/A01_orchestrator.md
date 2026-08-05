---
agent_id: A01
codename: ORCHESTRATOR
kind: deterministic
implementation: preflight/orchestrator.py
model: none
tier: 0
parents: []
produces: ExecutionReport
---

# A01 — ORCHESTRATOR

## Identity

You are A01, the workflow controller of PREFLIGHT.

You never perform policy analysis. You never classify videos. You never infer
violations. Your sole responsibility is to coordinate specialised agents and
guarantee deterministic execution.

## Why this agent is code and not a prompt

This file is a specification, not a system prompt. Nothing sends it to a model.

The reason is in the sentence above: *guarantee deterministic execution*. A
language model cannot. Asked to orchestrate, it would produce an
`agents_completed` list that is indistinguishable from a real one whether or
not the agents ran, and "ensure every stage executes exactly once" would
degrade from an invariant into a preference. Orchestration is bookkeeping —
dispatch, collect, validate, retry — and bookkeeping is what code is for.

What this file *does* do is bind the implementation. `preflight/agents/roster.py`
parses the frontmatter of every prompt in this directory, and
`tests/test_orchestrator.py` asserts the running pipeline matches what is
declared here: the same agents, the same DAG, the same contracts. If the code
drifts from the spec, a test goes red.

That is the same move PREFLIGHT makes on video. A specification can be compiled
into a linter; this one lints the pipeline against its own description.

## Mission

Transform one input video into a fully analysed release package. Coordinate the
complete pipeline from ingestion to report generation. Ensure every stage
executes exactly once, outputs are validated before continuing, and no
downstream agent receives incomplete data.

Responsible for workflow integrity, not content judgement.

## Responsibilities

1. Validate input
2. Create the working directory
3. Read configuration
4. Initialise logging
5. Start preprocessing
6. Dispatch agents
7. Collect outputs
8. Validate JSON
9. Handle retries
10. Build the execution timeline
11. Trigger report generation
12. Exit successfully or fail gracefully

## Prohibitions

- Never classify content
- Never guess
- Never invent missing data
- Never change another agent's output
- Never edit transcripts
- Never modify timestamps
- Never overwrite evidence

The last four are enforced structurally rather than by instruction: agent
outputs are frozen dataclasses, and the orchestrator holds references it cannot
mutate in place.

## Input contract

```json
{
  "video_path": "uploads/video.mp4",
  "thumbnail_path": "uploads/thumb.jpg",
  "config": {
    "report_format": "html",
    "policy_version": "current"
  }
}
```

## Output contract

Execution metadata only. No findings, no scores, no judgements — those belong
to the agents that produced them.

```json
{
  "status": "SUCCESS",
  "pipeline_id": "PF-2026-000421",
  "agents_completed": ["A02", "A03", "A04", "A05", "A06",
                       "A07", "A08", "A09", "A10", "A11", "A12"],
  "execution_time_seconds": 94.6,
  "report_path": "reports/report.html"
}
```

## Failure handling

| Condition | Action |
|---|---|
| Transient failure (timeout, 429, 5xx) | Retry with backoff, up to the stage's retry budget |
| Permanent failure (401, malformed input) | Do not retry. Record and continue if the stage is optional |
| Required stage fails | Stop the pipeline cleanly with a recorded reason |
| Optional stage fails | Record, mark the agent FAILED, reduce coverage, continue |

Never fabricate a missing output. An agent that did not run is reported as not
having run, and coverage falls accordingly.

**Required stages are A02 ingest/speech only.** Everything else is optional by
design: a report that is honest about what it could not inspect is worth more
than one that refuses to exist.

## Communication rules

- Structured JSON between agents; never natural-language summaries as machine input
- Preserve timestamps and evidence identifiers exactly as produced
- Treat every downstream agent as an independent service

## Quality checklist

Verified programmatically before a run is marked complete:

- [ ] Input validated
- [ ] All required agents completed
- [ ] Output schemas validated
- [ ] Report generated
- [ ] Execution log written
- [ ] Exit status recorded
