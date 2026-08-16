# Cold-clone reproducibility.
#
# The models are baked into the image rather than downloaded on first run. A
# judge on a flaky connection running `docker compose run preflight` still gets
# a working tool, and the demo does not depend on Hugging Face being reachable
# at the moment it matters.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# ffmpeg is the workhorse: demux, keyframes, loudness, and the remediation
# render all route through it. espeak-ng is only needed to synthesise the demo
# clip, and costs a few megabytes. tesseract-ocr backs the on-screen text /
# credential-disclosure scanner — without it OCR degrades honestly rather
# than failing, but a deployed instance should have the real thing.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      espeak-ng \
      tesseract-ocr \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Created early, not just before USER — the weight-caching step below needs
# /home/preflight to already exist so it has somewhere to copy the cache to.
RUN useradd --create-home --uid 1000 preflight

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the model layers.
COPY pyproject.toml README.md ./
COPY preflight/__init__.py preflight/__init__.py
RUN pip install --quiet -e '.[asr,ocr]'

# Bake the ASR weights. This is the layer that makes the image self-sufficient
# — but `_hf_cached()` resolves the cache through `Path.home()` of whichever
# user is running, and this RUN still executes as root (HOME=/root) while the
# container actually serves as the unprivileged `preflight` user (created
# above), whose HOME is /home/preflight. Caching once and only chowning /app
# at the end left the weights sitting where the runtime user's own lookup
# would never find them — reported as "not cached", not as an error, so
# ASR silently ran degraded instead of visibly failing. Copying the cache
# into the runtime user's own HOME (not just chowning /app) is the fix.
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('base.en', device='cpu', compute_type='int8'); \
print('base.en cached')" \
 && mkdir -p /home/preflight/.cache \
 && cp -r /root/.cache/huggingface /home/preflight/.cache/huggingface \
 && chown -R preflight:preflight /home/preflight/.cache

COPY . .

# The corpus is generated from its authoring script, so the image always
# carries a corpus whose manifest hashes match its clause files.
RUN python scripts/build_corpus.py \
 && python scripts/make_assets.py

# Never run as root. A container that writes root-owned files into a mounted
# working directory is a small cruelty to whoever runs it.
RUN chown -R preflight:preflight /app
USER preflight

# Plain `docker run image` (no args — exactly what a PaaS does when it has
# no dockerCommand override configured) now serves by default: with a fixed
# ENTRYPOINT, Docker combines it with CMD verbatim when no args are passed.
# `docker run image check foo.mp4` still overrides only the CMD half, so the
# CLI-passthrough contract judges rely on is untouched. 10000 is a fixed
# literal, not read from $PORT — exec-form CMD can't do shell expansion —
# so whatever PORT the deploy target sets must match this, or override the
# whole command.
ENTRYPOINT ["preflight"]
CMD ["serve", "--host", "0.0.0.0", "--port", "10000"]
