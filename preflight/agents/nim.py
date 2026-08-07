"""NVIDIA NIM client, built for a rate-limited free tier.

Four things this has to survive, all of which happen:

1. **429s.** The community tier sits around 40 rpm. A token bucket at 30 keeps
   headroom so a burst never trips one mid-demo, and exponential backoff with
   jitter recovers when one lands anyway.
2. **Prose around JSON.** Instruction-tuned models emit "Here is the JSON:" and
   trailing commas regardless of how firmly the prompt forbids it. The repair
   cascade recovers rather than failing the window.
3. **402 / quota exhaustion.** Raised as a clear message pointing at --offline,
   never a stack trace.
4. **Repeat runs.** Every response is cached by hash of model plus prompt, so
   re-running a video costs nothing and a live demo cannot be rate-limited by
   its own rehearsal.

Uses urllib rather than the openai or requests package: the endpoint is a plain
OpenAI-compatible HTTP surface, and one less dependency is one less way a clean
clone fails.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from preflight import cas
from preflight.config import Settings

MAX_ATTEMPTS = 5
BACKOFF_CAP_S = 30.0
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class NimUnavailable(RuntimeError):
    """Upstream cannot serve the request and there is no local fallback."""


class NimQuotaExhausted(NimUnavailable):
    """HTTP 402 — the account is out of credits."""


class TokenBucket:
    """Requests-per-minute limiter."""

    def __init__(self, rpm: int) -> None:
        self.capacity = max(1, rpm)
        self.tokens = float(self.capacity)
        self.refill_per_second = self.capacity / 60.0
        self.updated = time.monotonic()

    def take(self) -> float:
        """Block until a token is available. Returns seconds waited."""
        waited = 0.0
        while True:
            now = time.monotonic()
            self.tokens = min(
                self.capacity, self.tokens + (now - self.updated) * self.refill_per_second
            )
            self.updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return waited
            sleep_for = (1.0 - self.tokens) / self.refill_per_second
            time.sleep(sleep_for)
            waited += sleep_for


def extract_json(text: str) -> Any:
    """Recover a JSON value from model output.

    Cascade: strip code fences, parse directly, then find the outermost
    balanced brace or bracket, then repair trailing commas. Each step handles a
    failure mode that instruction-tuned models actually produce.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    candidate = text.strip()

    fenced = FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Outermost balanced structure, ignoring braces inside strings.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    blob = candidate[start : index + 1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        repaired = re.sub(r",\s*([}\]])", r"\1", blob)
                        try:
                            return json.loads(repaired)
                        except json.JSONDecodeError:
                            break
    raise ValueError(f"no recoverable JSON in response: {text[:200]!r}")


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    retries: int = 0
    waited_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    log: list[str] = field(default_factory=list)


class NimClient:
    """OpenAI-compatible chat client with caching and rate limiting."""

    def __init__(self, settings: Settings, store: cas.Store) -> None:
        self.settings = settings
        self.store = store
        self.bucket = TokenBucket(settings.rpm)
        self.usage = Usage()

    @property
    def online(self) -> bool:
        return self.settings.online

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Any:
        """One chat completion, parsed as JSON. Cached by model + prompt."""
        if not self.online:
            raise NimUnavailable(
                "no API key configured — run with --offline for local-only analysis"
            )

        key = cas.hash_json(
            {
                "model": model,
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        entry = self.store.entry("r", f"nim-{key}")
        if entry.exists:
            self.usage.cached += 1
            return entry.read_json("response.json")["parsed"]

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        raw = self._post_with_retries("/chat/completions", payload)
        content = (
            raw.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        )
        parsed = extract_json(content)

        usage = raw.get("usage") or {}
        self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)

        entry.discard()
        entry.root.mkdir(parents=True, exist_ok=True)
        entry.write_json("response.json", {"parsed": parsed, "raw_content": content})
        entry.commit()
        return parsed

    def embed(self, texts: list[str], *, model: str, input_type: str = "passage"):
        """Embed a batch. Returns an L2-normalised float32 matrix, or None."""
        if not self.online or not texts:
            return None

        import numpy as np

        key = cas.hash_json({"model": model, "texts": texts, "input_type": input_type})
        entry = self.store.entry("p", f"embed-{key}")
        if entry.exists:
            self.usage.cached += 1
            return np.array(entry.read_json("vectors.json"), dtype=np.float32)

        payload = {
            "input": texts,
            "model": model,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        raw = self._post_with_retries("/embeddings", payload)
        vectors = [item["embedding"] for item in raw.get("data", [])]
        if not vectors:
            return None

        matrix = np.array(vectors, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

        entry.discard()
        entry.root.mkdir(parents=True, exist_ok=True)
        entry.write_json("vectors.json", matrix.tolist())
        entry.commit()
        return matrix

    def _post_with_retries(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            self.usage.waited_s += self.bucket.take()
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    request, timeout=self.settings.http_timeout_s
                ) as response:
                    self.usage.calls += 1
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                if exc.code == 402:
                    raise NimQuotaExhausted(
                        "NVIDIA API returned 402 (quota exhausted). "
                        "Re-run with --offline for local-only analysis."
                    ) from exc
                if exc.code in (401, 403):
                    raise NimUnavailable(
                        f"NVIDIA API rejected the key ({exc.code}). Check NVIDIA_API_KEY."
                    ) from exc
                if exc.code == 429 or exc.code >= 500:
                    last_error = exc
                    self.usage.retries += 1
                    delay = min(2**attempt + random.random(), BACKOFF_CAP_S)
                    self.usage.log.append(
                        f"HTTP {exc.code} — backing off {delay:.1f}s "
                        f"(attempt {attempt + 1}/{MAX_ATTEMPTS})"
                    )
                    time.sleep(delay)
                    continue
                raise NimUnavailable(f"NVIDIA API error {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                self.usage.retries += 1
                delay = min(2**attempt + random.random(), BACKOFF_CAP_S)
                time.sleep(delay)

        raise NimUnavailable(
            f"NVIDIA API unreachable after {MAX_ATTEMPTS} attempts: {last_error}"
        )
