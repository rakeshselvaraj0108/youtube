"""Report emission — the contract, SARIF, the certificate, and the single file."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from preflight.models import Adversarial, Evidence, Finding, PolicyRef
from preflight.report.build import build_breakdown, build_risk_bands
from preflight.report.html import BundleMissing, emit_fixture, emit_html
from preflight.report.sarif import build_certificate, build_sarif, exit_code

SCHEMA = Path("schema/analysis-report.schema.json")


def finding(
    fid="f1",
    clause="AF-01",
    category="Language",
    severity="MEDIUM",
    confidence=0.9,
    start=10_000,
    end=12_000,
    fix="BLEEP",
) -> Finding:
    return Finding(
        id=fid,
        clauseId=clause,
        category=category,
        title="Inappropriate language",
        description="Strong profanity",
        startMs=start,
        endMs=end,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,
        modalities={"speech": confidence},
        evidence=Evidence(transcript="this is fucked"),
        policy=PolicyRef(
            clauseId=clause, title="Inappropriate language", section="§ AF-01", text="x" * 90
        ),
        adversarial=Adversarial(charge="charge", rationale="rationale", confidence=confidence),
        suggestedFix=fix,  # type: ignore[arg-type]
    )


@pytest.fixture
def report() -> dict:
    findings = [
        finding(),
        finding(fid="f2", clause="AF-02", category="Violence", severity="CRITICAL", start=40_000, end=43_000),
        finding(fid="f3", clause="AF-01", category="Language", severity="LOW", start=70_000, end=71_000),
    ]
    return {
        "video": {
            "filename": "demo.mp4",
            "durationMs": 90_000,
            "width": 1280,
            "height": 720,
            "fps": 30.0,
            "sizeBytes": 1234,
            "audioCodec": "AAC",
            "sampleRate": 44_100,
            "posterUrl": "./demo.poster.jpg",
            "srcUrl": "./demo.mp4",
        },
        "meta": {
            "analyzedAt": "2026-08-05T11:00:00Z",
            "policyVersion": "2026-08",
            "engineVersion": "0.1.0",
            "attestationHash": "b3:abc",
            "coverage": 0.83,
        },
        "scores": {
            "overall": 45,
            "sub": {
                "policy": 31.0,
                "copyright": 19.0,
                "metadata": 78.0,
                "accessibility": 62.0,
                "audio": 88.0,
            },
            "verdict": "DO_NOT_PUBLISH",
            "weakest": "copyright",
        },
        "riskBands": build_risk_bands(findings, 90_000),
        "findings": [f.to_json() for f in findings],
        "breakdown": build_breakdown(findings),
        "remediation": {
            "ops": [],
            "ffmpegCommand": "ffmpeg -y -i demo.mp4 -c copy out.mp4",
            "renderMs": 4200,
            "videoStreamCopied": True,
            "log": [],
        },
        "incidents": [],
        "reasoning": [],
        "cost": {
            "estimatedCalls": 5,
            "actualCalls": 3,
            "ceiling": None,
            "shed": [],
        },
        "agents": [
            {
                "id": "vision",
                "name": "Vision Agent",
                "tier": 2,
                "parents": ["ingest"],
                "status": "DEGRADED",
                "detail": "rate limited",
                "coverage": 0.42,
                "elapsedMs": 900,
                "tsMs": 100,
                "calls": 31,
            }
        ],
    }


class TestSegmentRollupReachesTheReport:
    """`scoring/rollup.py` shipped fully written and fully unit-tested with
    no caller at all — the same orphan pattern as the vision agent, the
    orchestrator and `rerank.text` before it. Its own tests passed the whole
    time. This is the assertion that would have failed."""

    def test_a_short_video_has_no_segments_key_at_all(self, report):
        """Absent, not empty — an empty array reads as 'rolled up and found
        nothing', which is a different and false claim."""
        assert "segments" not in report

    def test_the_schema_accepts_a_report_carrying_segments(self, report):
        from preflight.plan import SEGMENT_MS
        from preflight.scoring.rollup import rollup

        findings = [
            finding(fid="f1", start=100_000, end=110_000),
            finding(fid="f2", start=1_900_000, end=1_910_000, severity="CRITICAL"),
        ]
        duration = 2_400_000
        report["video"]["durationMs"] = duration
        report["findings"] = [f.to_json() for f in findings]
        report["segments"] = [
            s.to_json() for s in rollup(findings, duration, SEGMENT_MS)
        ]

        from preflight.report.build import validate

        schema = Path("schema/analysis-report.schema.json")
        if not schema.is_file():
            pytest.skip("run npm run schema")
        validate(report, schema)

    def test_shares_sum_to_one_so_the_page_can_show_percentages(self):
        from preflight.plan import SEGMENT_MS
        from preflight.scoring.rollup import rollup

        findings = [
            finding(fid="f1", start=100_000, end=110_000),
            finding(fid="f2", start=1_900_000, end=1_910_000, severity="CRITICAL"),
        ]
        segments = rollup(findings, 2_400_000, SEGMENT_MS)
        assert sum(s.risk_share for s in segments) == pytest.approx(1.0, abs=0.001)


class TestRiskBands:
    def test_tiles_the_whole_runtime_without_gaps(self):
        bands = build_risk_bands([finding()], 90_000)
        assert bands[0]["startMs"] == 0
        assert bands[-1]["endMs"] == 90_000
        for earlier, later in zip(bands, bands[1:]):
            assert earlier["endMs"] == later["startMs"]

    def test_peaks_over_the_finding(self):
        bands = build_risk_bands(
            [finding(severity="CRITICAL", confidence=0.95, start=40_000, end=43_000)], 90_000
        )
        overlapping = [b for b in bands if b["endMs"] > 40_000 and b["startMs"] < 43_000]
        assert max(b["risk"] for b in overlapping) > 0.9

    def test_stays_inside_zero_to_one(self):
        for band in build_risk_bands([finding()], 90_000):
            assert 0.0 <= band["risk"] <= 1.0

    def test_file_scoped_findings_do_not_flatten_the_terrain(self):
        """A finding true of every second would plateau the terrain and hide
        the spikes that are the point of looking at it."""
        bands = build_risk_bands([finding(start=0, end=90_000)], 90_000)
        assert all(b["risk"] == 0.0 for b in bands)

    def test_zero_duration_is_empty(self):
        assert build_risk_bands([finding()], 0) == []


class TestBreakdown:
    def test_accounts_for_every_finding(self, report):
        total = sum(row["count"] for row in report["breakdown"])
        assert total == len(report["findings"])

    def test_reports_each_category_at_its_worst_severity(self):
        rows = build_breakdown([finding(severity="LOW"), finding(fid="f2", severity="HIGH")])
        assert rows[0]["category"] == "Language"
        assert rows[0]["severity"] == "HIGH"
        assert rows[0]["count"] == 2


@pytest.mark.skipif(not SCHEMA.is_file(), reason="run npm run schema")
class TestSchemaContract:
    def test_emitted_shape_validates(self, report):
        import jsonschema

        jsonschema.validate(report, json.loads(SCHEMA.read_text(encoding="utf-8")))

    def test_a_bad_severity_is_rejected(self, report):
        import jsonschema

        report["findings"][0]["severity"] = "CATASTROPHIC"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(report, json.loads(SCHEMA.read_text(encoding="utf-8")))


class TestSarif:
    def test_declares_the_schema_and_tool(self, report):
        sarif = build_sarif(report)
        assert sarif["version"] == "2.1.0"
        assert sarif["runs"][0]["tool"]["driver"]["name"] == "PREFLIGHT"

    def test_one_rule_per_clause_not_per_finding(self, report):
        rules = build_sarif(report)["runs"][0]["tool"]["driver"]["rules"]
        assert len(rules) == 2  # AF-01 twice, AF-02 once
        assert len([r for r in rules if r["id"] == "AF-01"]) == 1

    def test_every_result_references_a_declared_rule(self, report):
        run = build_sarif(report)["runs"][0]
        declared = {r["id"] for r in run["tool"]["driver"]["rules"]}
        assert all(result["ruleId"] in declared for result in run["results"])

    def test_encodes_seconds_as_lines_and_keeps_true_timings(self, report):
        run = build_sarif(report)["runs"][0]
        result = next(r for r in run["results"] if r["properties"]["startMs"] == 40_000)
        region = result["locations"][0]["physicalLocation"]["region"]
        assert region["startLine"] == 40
        assert result["properties"]["endMs"] == 43_000

    def test_never_emits_line_zero(self, report):
        """SARIF lines are 1-indexed; a finding at 0.4s must not become line 0."""
        report["findings"][0]["startMs"] = 400
        report["findings"][0]["endMs"] = 900
        run = build_sarif(report)["runs"][0]
        for result in run["results"]:
            assert result["locations"][0]["physicalLocation"]["region"]["startLine"] >= 1

    def test_carries_the_adversarial_record(self, report):
        for result in build_sarif(report)["runs"][0]["results"]:
            assert result["properties"]["auditorCharge"]
            assert result["properties"]["adjudicatorVerdict"] in {"UPHELD", "DISMISSED"}

    def test_reports_degraded_agents_rather_than_hiding_them(self, report):
        invocation = build_sarif(report)["runs"][0]["invocations"][0]
        assert invocation["properties"]["coverage"] == 0.83
        assert invocation["properties"]["degradedAgents"][0]["id"] == "vision"

    def test_exit_code_follows_the_verdict(self, report):
        assert exit_code(report) == 1
        report["scores"]["verdict"] = "READY_TO_PUBLISH"
        assert exit_code(report) == 0


class TestCertificate:
    def _cert(self, report):
        return build_certificate(
            report,
            models={"auditor": "meta/llama-3.3-70b-instruct"},
            policy_digest="sha:abc",
            video_hash="b3:def",
            retrieval_backend="nim",
        )

    def test_ships_the_scoring_rule_so_it_can_be_recomputed(self, report):
        cert = self._cert(report)
        assert "worst + 15" in cert["readiness"]["clamp"]
        assert cert["readiness"]["weights"]["policy"] == 0.40

    def test_states_per_agent_coverage(self, report):
        cert = self._cert(report)
        assert cert["coverage"]["overall"] == 0.83
        assert cert["coverage"]["agents"][0]["coverage"] == 0.42

    def test_ships_its_own_limitations(self, report):
        limits = " ".join(self._cert(report)["limitations"])
        assert "does not prove safety" in limits
        assert "not affiliated" in limits

    def test_counts_findings_by_severity_consistently(self, report):
        cert = self._cert(report)
        assert sum(cert["findings"]["bySeverity"].values()) == len(report["findings"])


class TestSingleFileHtml:
    def _dist(self, tmp_path: Path) -> Path:
        dist = tmp_path / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "assets" / "app.js").write_text("console.log('app');", encoding="utf-8")
        (dist / "assets" / "app.css").write_text("body{color:red}", encoding="utf-8")
        (dist / "index.html").write_text(
            '<!doctype html><html><head>'
            '<link rel="preconnect" href="https://fonts.googleapis.com" />'
            '<link rel="stylesheet" href="/assets/app.css">'
            "</head><body><div id=root></div>"
            '<script type="module" src="/assets/app.js"></script>'
            "</body></html>",
            encoding="utf-8",
        )
        return dist

    def test_inlines_everything_and_injects_the_report(self, report, tmp_path):
        out = emit_html(report, self._dist(tmp_path), tmp_path / "report.html")
        html = out.read_text(encoding="utf-8")

        assert "console.log('app');" in html
        assert "body{color:red}" in html
        assert "__PREFLIGHT_REPORT__" in html

    def test_leaves_no_external_references(self, report, tmp_path):
        """The page has to open offline. A single remote font link breaks that."""
        html = emit_html(report, self._dist(tmp_path), tmp_path / "r.html").read_text(
            encoding="utf-8"
        )
        assert not re.search(r'(?:src|href)="https?://', html)

    def test_injects_before_the_bundle_runs(self, report, tmp_path):
        html = emit_html(report, self._dist(tmp_path), tmp_path / "r.html").read_text(
            encoding="utf-8"
        )
        assert html.index("__PREFLIGHT_REPORT__") < html.index("console.log('app');")

    def test_payload_round_trips(self, report, tmp_path):
        html = emit_html(report, self._dist(tmp_path), tmp_path / "r.html").read_text(
            encoding="utf-8"
        )
        match = re.search(r"window\.__PREFLIGHT_REPORT__=(\{.*?\});", html, re.DOTALL)
        assert match
        assert json.loads(match.group(1))["scores"]["overall"] == 45

    def test_missing_bundle_is_a_clear_error_not_a_traceback(self, report, tmp_path):
        with pytest.raises(BundleMissing, match="npm run build"):
            emit_html(report, tmp_path / "absent", tmp_path / "r.html")


class TestFixtureEmission:
    def test_writes_typescript_the_ui_can_import(self, report, tmp_path):
        out = emit_fixture(report, tmp_path / "fixture.ts")
        body = out.read_text(encoding="utf-8")
        assert "export const beforeReport: AnalysisReport" in body
        assert "GENERATED FILE" in body
        assert json.loads(re.search(r"= (\{.*\}) as AnalysisReport", body, re.DOTALL).group(1))
