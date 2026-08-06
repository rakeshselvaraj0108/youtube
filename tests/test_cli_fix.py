"""`preflight fix --apply` and `preflight check`, invoked as a real CLI
process would invoke them.

Every other test in this suite calls pipeline functions directly. Nothing
called the CLI command itself end to end, and it was the only place that
would have caught this: `fix --apply` wrote its temp file as
`dead_air.safe.mp4.tmp1234` — a suffix ffmpeg's muxer cannot infer a
container from, since the extension it actually reads is `.tmp1234`, not
`.mp4`. Every unit test of the underlying pieces passed; the real process
failed on the first real render. Atomicity, and everything that depends on
temp-file naming being right, is only provable by actually running the
command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from preflight import cli, ffmpeg
from preflight.cli import app

pytestmark = pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg is not installed")

runner = CliRunner()


@pytest.fixture(scope="module")
def dead_air_clip(tmp_path_factory) -> Path:
    """10s clip: 2s tone, 4s true silence, 4s tone.

    The silence sits well clear of both the file start and the "trailing
    silence is normal" exclusion near the end, so `_dead_air_findings`
    reliably fires — this fixture exists specifically to produce a real,
    fixable AUD-03 finding offline, with no API key and no LLM involved.
    """
    out = tmp_path_factory.mktemp("media") / "dead_air.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-f", "lavfi", "-i", "color=c=blue:size=320x180:rate=30:duration=10",
            "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[aout]",
            "-map", "3:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


class TestQualityDelta:
    """SSIM between the source and the rendered output — 'the remediation
    changed 0.4% of frames, SSIM 0.998' is the answer to the question a
    creator actually has about an automated fix: will this wreck my video?"""

    @pytest.fixture(scope="class")
    def moving_clip(self, tmp_path_factory) -> Path:
        out = tmp_path_factory.mktemp("ssim") / "moving.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1",
                str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    @pytest.fixture(scope="class")
    def blurred_clip(self, tmp_path_factory, moving_clip) -> Path:
        out = tmp_path_factory.mktemp("ssim2") / "blurred.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(moving_clip), "-vf", "boxblur=5",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    def test_a_file_compared_to_itself_is_perfect(self, moving_clip):
        assert ffmpeg.quality_delta(moving_clip, moving_clip) == pytest.approx(1.0, abs=0.001)

    def test_a_visibly_altered_file_scores_below_one(self, moving_clip, blurred_clip):
        score = ffmpeg.quality_delta(moving_clip, blurred_clip)
        assert score is not None
        assert score < 0.95

    def test_a_nonexistent_file_returns_none_rather_than_raising(self, moving_clip, tmp_path):
        """A failed quality check must not undo a render that otherwise
        succeeded and verified correctly on duration."""
        assert ffmpeg.quality_delta(moving_clip, tmp_path / "does_not_exist.mp4") is None


class TestModelsPull:
    """`doctor` and both local providers point an unresolved capability at
    `preflight models pull` in their fix hints — a command that did not
    exist. `_hf_cached` is monkeypatched here rather than relying on either
    package actually being installed, so this proves the command's own
    control flow (rejection, the cached fast-path, the exit codes) without
    depending on what happens to be downloaded on the machine running the
    suite."""

    def test_an_unknown_target_is_rejected_before_any_work_happens(self):
        result = runner.invoke(app, ["models", "pull", "bogus"])
        assert result.exit_code != 0
        assert "asr, embed or all" in result.output

    def test_an_already_cached_model_is_reported_without_importing_anything(
        self, monkeypatch
    ):
        monkeypatch.setattr(cli, "_hf_cached", lambda fragment: True)
        result = runner.invoke(app, ["models", "pull", "asr"])
        assert result.exit_code == 0, result.output
        assert "already cached" in result.output

    def test_pull_all_reports_both_models(self, monkeypatch):
        monkeypatch.setattr(cli, "_hf_cached", lambda fragment: True)
        result = runner.invoke(app, ["models", "pull"])
        assert result.exit_code == 0, result.output
        assert "faster-whisper" in result.output
        assert "all-MiniLM-L6-v2" in result.output

    def test_a_missing_optional_dependency_names_the_fix_not_a_crash(self, monkeypatch):
        """Forces the ImportError branch regardless of what happens to be
        installed on the machine running the suite — asserting on real
        absence would let this test attempt a genuine network download on a
        machine where sentence-transformers is present, which is exactly the
        slow, flaky pattern this project's own tests elsewhere refuse to
        risk."""
        monkeypatch.setattr(cli, "_hf_cached", lambda fragment: False)
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        result = runner.invoke(app, ["models", "pull", "embed"])
        assert "Traceback" not in result.output
        assert result.exit_code != 0
        assert "pip install" in result.output


class TestFixApply:
    def test_apply_writes_the_real_destination(self, dead_air_clip, tmp_path):
        destination = tmp_path / "dead_air.safe.mp4"
        result = runner.invoke(
            app,
            ["fix", str(dead_air_clip), "--offline", "--apply", "--out", str(destination)],
        )
        assert result.exit_code == 0, result.output
        assert destination.is_file()
        assert destination.stat().st_size > 0

    def test_no_temp_file_survives_a_successful_render(self, dead_air_clip, tmp_path):
        """The exact regression: a temp file with a broken extension left
        behind, or the real destination never appearing at all."""
        destination = tmp_path / "dead_air.safe.mp4"
        runner.invoke(
            app,
            ["fix", str(dead_air_clip), "--offline", "--apply", "--out", str(destination)],
        )
        leftovers = [p for p in tmp_path.iterdir() if ".tmp" in p.name]
        assert leftovers == []

    def test_the_rendered_output_is_a_valid_playable_container(self, dead_air_clip, tmp_path):
        """Proof the temp filename kept its extension where ffmpeg's muxer
        could see it — a broken temp name fails the render outright before
        this can even be checked, which is exactly how the bug first showed."""
        destination = tmp_path / "dead_air.safe.mp4"
        runner.invoke(
            app,
            ["fix", str(dead_air_clip), "--offline", "--apply", "--out", str(destination)],
        )
        meta = ffmpeg.probe(destination)
        assert float(meta["format"]["duration"]) > 0

    def test_dry_run_does_not_touch_the_filesystem(self, dead_air_clip, tmp_path):
        destination = tmp_path / "dead_air.safe.mp4"
        result = runner.invoke(
            app, ["fix", str(dead_air_clip), "--offline", "--out", str(destination)]
        )
        assert result.exit_code == 0, result.output
        assert not destination.exists()

    def test_an_unknown_strategy_is_rejected_before_any_work_happens(self, dead_air_clip):
        result = runner.invoke(
            app, ["fix", str(dead_air_clip), "--offline", "--strategy", "reckless"]
        )
        assert result.exit_code != 0

    def test_a_missing_video_fails_cleanly(self, tmp_path):
        result = runner.invoke(
            app, ["fix", str(tmp_path / "does_not_exist.mp4"), "--offline"]
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_output_duration_matches_the_source_when_nothing_is_cut(
        self, dead_air_clip, tmp_path
    ):
        """The verification path itself: MUTE preserves duration, and the
        render must confirm that rather than merely trusting ffmpeg's exit
        code — a demoted-from-CUT-to-MUTE op is exactly the case where a
        naive 'exit 0 means it worked' check would miss a length mismatch."""
        destination = tmp_path / "dead_air.safe.mp4"
        result = runner.invoke(
            app,
            ["fix", str(dead_air_clip), "--offline", "--apply", "--out", str(destination)],
        )
        assert result.exit_code == 0, result.output
        meta = ffmpeg.probe(destination)
        assert float(meta["format"]["duration"]) == pytest.approx(10.0, abs=0.5)


# A key shaped realistically enough to exercise the real redaction regex, and
# fake enough that leaking it anywhere would be embarrassing rather than
# dangerous — pragma marker so a secrets scanner does not flag this file.
FAKE_KEY = "nvapi-FAKEKEY1234567890abcdefghijklmnopqrstuvwx"  # pragma: allowlist secret


class TestNoCredentialLeak:
    """`redact()`, `fingerprint()` and `RedactFilter` are unit-tested in
    test_providers.py — proven correct in isolation. This is the different,
    missing claim: that a REAL run, with a key actually sitting in the
    environment, never lets it reach anything written to disk or printed to
    the terminal. `Settings.load()` reads `NVIDIA_API_KEY` directly rather
    than through `preflight/providers/secrets.py`'s `Secret` wrapper — a real
    architectural inconsistency with that module's stated claim to be the
    only credential reader in the project — so this test does not trust that
    the two paths were kept in sync by inspection. It runs the process and
    greps the output.
    """

    def test_the_key_never_appears_in_any_written_artifact(
        self, dead_air_clip, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        out_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "check", str(dead_air_clip), "--offline",
                "--format", "all", "--out", str(out_dir),
            ],
        )
        assert "Traceback" not in result.output

        assert FAKE_KEY not in result.output, "leaked into terminal output"
        for path in out_dir.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                assert FAKE_KEY not in text, f"leaked into {path}"

    def test_the_key_never_appears_in_a_fix_dry_run(
        self, dead_air_clip, monkeypatch
    ):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        result = runner.invoke(app, ["fix", str(dead_air_clip), "--offline"])
        assert FAKE_KEY not in result.output

    def test_the_key_never_appears_in_the_doctor_report(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        result = runner.invoke(app, ["doctor", "--offline"])
        assert FAKE_KEY not in result.output

    def test_a_truncated_fingerprint_is_the_most_that_ever_appears(self, monkeypatch):
        """Proves the negative isn't vacuous — the key was genuinely present
        and genuinely read, just never printed whole. If this assertion ever
        started failing because NOTHING related to the key appears at all,
        that would itself be worth knowing, not just a passing test."""
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        result = runner.invoke(app, ["doctor", "--offline"])
        assert FAKE_KEY[:9] in result.output or "nvapi" in result.output.lower()
