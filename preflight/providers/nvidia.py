"""NVIDIA NIM providers. All six share one governor.

Sharing the governor is the whole point: chat, embeddings, ASR, reranking,
vision and OCR are one rate limit, not six. Each class below calls
`governor("nvidia")`, which returns the same object every time.

Uses urllib rather than the openai package. The endpoint is a plain
OpenAI-compatible HTTP surface, and one fewer dependency is one fewer way a
clean clone fails.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from preflight.agents.nim import extract_json
from preflight.providers.base import (
    BaseProvider,
    BudgetExhausted,
    CircuitOpen,
    Result,
    Served,
    Unavailable,
)
from preflight.providers.governor import governor
from preflight.providers.secrets import Secret, redact

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_ATTEMPTS = 5
VENDOR = "nvidia"

# Per socket operation — how long a single read may stall with no bytes.
SOCKET_TIMEOUT_S = 60

# Hard wall-clock ceiling on one complete request/response exchange.
#
# This is the one that actually bounds a run. The socket timeout above only
# governs a single read, so a far end that trickles bytes indefinitely resets
# it forever and the call never returns — observed live, one request held the
# pipeline for eight minutes against a nominal 120s setting, with no error
# raised and nothing to distinguish it from a slow model.
#
# Every measured working call — vision, reasoning, everything — has answered
# in under 15s. 90s is still 6x that, generous headroom for one genuinely
# slow-but-working call, while halving what a failing one costs against
# TRANSPORT_MAX_ATTEMPTS below.
REQUEST_DEADLINE_S = 90

# Retry budget specifically for "nothing answered at all" — a URLError or our
# own deadline firing — kept separate from and smaller than MAX_ATTEMPTS,
# which governs HTTP 429/5xx where the vendor is reachable and retrying is
# the correct move. Two attempts is enough to rule out one transient blip;
# five, at up to REQUEST_DEADLINE_S each, is most of a run's budget spent
# re-asking a host that was never going to answer.
TRANSPORT_MAX_ATTEMPTS = 2


class NvidiaProvider(BaseProvider):
    """Shared plumbing: auth check, HTTP, retry policy, ledger accounting."""

    id = VENDOR
    tier_label = "hosted"

    def __init__(
        self,
        capability: str,
        model: str,
        secret: Secret,
        *,
        base_url: str = DEFAULT_BASE_URL,
        rpm: int = 30,
        call_budget: int = 200,
    ) -> None:
        super().__init__(capability)
        self.model = model
        self.secret = secret
        self.base_url = base_url.rstrip("/")
        self.gov = governor(VENDOR, rpm=rpm, call_budget=call_budget)

    def available(self) -> tuple[bool, str]:
        if not self.secret.present:
            return False, "no NVIDIA_API_KEY"
        if not self.secret.shape_ok:
            return False, self.secret.problem or "NVIDIA_API_KEY malformed"
        return True, "ready"

    def healthcheck(self) -> tuple[bool, str, int]:
        ok, reason = self.available()
        if not ok:
            return False, reason, 0
        started = time.monotonic()
        try:
            self._post(
                "/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                    "max_tokens": 5,
                    "temperature": 0,
                },
            )
            return True, "200", int((time.monotonic() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            return False, redact(str(exc))[:160], int((time.monotonic() - started) * 1000)

    # ---------------------------------------------------------------- #

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.secret.value}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        # `timeout=` is per socket operation, not for the call as a whole: a
        # response whose body trickles in resets it on every chunk, so a
        # nominal 120s ceiling can block for as long as the far end keeps
        # dribbling bytes. Measured on a real run, one vision call held the
        # pipeline for over eight minutes against that 120s setting — the
        # whole analysis stopped, with no error and no way to tell a slow
        # call from a dead one.
        #
        # `read(deadline)` below enforces a wall-clock ceiling on the entire
        # exchange, which is the guarantee a long-video run actually needs:
        # bounded, whatever the far end does.
        deadline = time.monotonic() + REQUEST_DEADLINE_S
        with urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT_S) as response:
            chunks: list[bytes] = []
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"response exceeded the {REQUEST_DEADLINE_S:.0f}s deadline "
                        f"after {sum(len(c) for c in chunks)} bytes"
                    )
                # `read1`, not `read`. `read(n)` blocks until it has filled n
                # bytes or hit EOF, so against a trickling server a single
                # call can block for hours and the deadline check above never
                # gets to run — which is the very bug this loop exists to
                # fix. `read1` returns whatever one underlying read yields,
                # so control comes back on every packet.
                chunk = response.read1(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        return json.loads(b"".join(chunks).decode("utf-8"))

    def _call(self, path: str, payload: dict[str, Any]) -> Result:
        """One governed call with the full retry policy."""
        for attempt in range(MAX_ATTEMPTS):
            try:
                self.gov.acquire(self.capability)
            except CircuitOpen as exc:
                return Unavailable(str(exc), VENDOR, retryable=True)
            except BudgetExhausted as exc:
                return Unavailable(str(exc), VENDOR)

            started = time.monotonic()
            try:
                raw = self._post(path, payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:200]

                # Never retry an auth rejection. It will never succeed, and
                # each attempt burns a slot the rest of the run needs.
                if exc.code in (401, 403):
                    self.gov.on_failure(terminal=True)
                    return Unavailable(
                        f"auth rejected ({exc.code}) — check NVIDIA_API_KEY", VENDOR
                    )
                if exc.code == 402:
                    self.gov.on_failure(terminal=True)
                    return Unavailable(
                        "free-tier limit reached — re-run with --offline", VENDOR
                    )
                if exc.code == 429 or exc.code >= 500:
                    self.gov.on_failure()
                    if attempt < MAX_ATTEMPTS - 1:
                        self.gov.backoff(attempt)
                        continue
                    return Unavailable(f"HTTP {exc.code} after retries", VENDOR, True)

                self.gov.on_failure()
                return Unavailable(f"HTTP {exc.code}: {redact(detail)}", VENDOR)
            except (urllib.error.URLError, TimeoutError) as exc:
                self.gov.on_failure()
                # A smaller budget than the HTTP-status branch above, and on
                # purpose. HTTP 429/5xx means the vendor answered and said
                # "try again" — retrying the full budget is the right thing
                # to do, because the same endpoint moments later often
                # succeeds. A URLError or a deadline timeout means nothing
                # answered at all; retrying the identical request five times
                # against a host that is not responding does not test
                # anything a second attempt did not already establish, and
                # at up to REQUEST_DEADLINE_S per attempt it is the single
                # most expensive way to fail. Measured live: one `_call()`
                # against an unreachable vendor cost most of a 25-minute
                # stage on this branch alone before this existed.
                if attempt < TRANSPORT_MAX_ATTEMPTS - 1:
                    self.gov.backoff(attempt, cap=8.0)
                    continue
                return Unavailable(f"unreachable: {redact(str(exc))}", VENDOR, True)

            latency = int((time.monotonic() - started) * 1000)
            usage = raw.get("usage") or {}
            self.gov.on_success(
                self.capability,
                tokens_in=int(usage.get("prompt_tokens") or 0),
                tokens_out=int(usage.get("completion_tokens") or 0),
                latency_ms=latency,
            )
            return Served(
                value=raw,
                provider=f"{VENDOR}:{self.model}",
                tier=0,
                tokens=int(usage.get("total_tokens") or 0),
                latency_ms=latency,
            )

        return Unavailable("exhausted retries", VENDOR, retryable=True)


class NvidiaChat(NvidiaProvider):
    def invoke(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Result:
        result = self._call(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        if not result:
            return result

        content = (
            result.value.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        try:
            parsed = extract_json(content)
        except ValueError as exc:
            return Unavailable(f"unparseable response: {exc}", VENDOR, retryable=True)

        result.value = parsed
        result.meta["raw_content"] = content
        return result


class NvidiaEmbed(NvidiaProvider):
    def invoke(
        self, *, texts: list[str], input_type: str = "passage", **kwargs: Any
    ) -> Result:
        if not texts:
            return Unavailable("no texts supplied", VENDOR)

        result = self._call(
            "/embeddings",
            {
                "input": texts,
                "model": self.model,
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END",
            },
        )
        if not result:
            return result

        vectors = [item["embedding"] for item in result.value.get("data", [])]
        if not vectors:
            return Unavailable("embedding endpoint returned no vectors", VENDOR)

        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        result.value = matrix
        return result


class NvidiaRerank(NvidiaProvider):
    def invoke(self, *, query: str, passages: list[str], **kwargs: Any) -> Result:
        if not passages:
            return Unavailable("no passages supplied", VENDOR)
        result = self._call(
            "/ranking",
            {
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": p} for p in passages],
            },
        )
        if not result:
            return result
        rankings = result.value.get("rankings", [])
        result.value = [(int(r["index"]), float(r.get("logit", 0.0))) for r in rankings]
        return result


class NvidiaVision(NvidiaProvider):
    # What the model is told after it answers a JSON request with prose. It
    # is shown its own output because "return JSON" alone is the instruction
    # it already ignored — the correction has to name what went wrong.
    _REPAIR = (
        "That response was prose, not JSON, and could not be parsed. "
        "Reply again with ONLY the JSON object described above. "
        "No preamble, no explanation, no markdown fence — the first "
        "character must be { and the last must be }."
    )

    def _describe(self, messages: list[dict[str, Any]], max_tokens: int) -> Result:
        return self._call(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
        )

    @staticmethod
    def _content(result: Result) -> str:
        return result.value.get("choices", [{}])[0].get("message", {}).get("content", "")

    def invoke(
        self, *, prompt: str, image_b64: str, max_tokens: int = 512, **kwargs: Any
    ) -> Result:
        """One frame described as JSON, with one corrective attempt.

        Vision-language models intermittently answer a JSON request with a
        paragraph of description. Measured live, that cost seven of eight
        frames and left the vision agent at 1% coverage against its 22%
        share of the analysis surface.

        Retrying the identical request does not help: temperature is 0, so
        the same prompt returns the same paragraph. The retry has to *say
        something different* — hence a real conversational turn that shows
        the model its own prose and names the failure. That is the whole
        difference between a retry and a repair.
        """
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ]

        result = self._describe(messages, max_tokens)
        if not result:
            return result

        content = self._content(result)
        try:
            result.value = extract_json(content)
            return result
        except ValueError:
            pass

        repaired = self._describe(
            [
                *messages,
                # Truncated: the point is to identify the answer, and echoing
                # a full paragraph back costs tokens on every repair.
                {"role": "assistant", "content": content[:400]},
                {"role": "user", "content": self._REPAIR},
            ],
            max_tokens,
        )
        if not repaired:
            return repaired

        try:
            repaired.value = extract_json(self._content(repaired))
        except ValueError as exc:
            # Two paragraphs means this frame is not going to produce JSON.
            # Still retryable at the registry for a transient cause, but the
            # reason now says the repair was tried and refused.
            return Unavailable(
                f"unparseable vision response after repair: {exc}", VENDOR, True
            )
        repaired.calls = getattr(result, "calls", 1) + getattr(repaired, "calls", 1)
        return repaired
