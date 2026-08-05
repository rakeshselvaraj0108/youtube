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
# clip, and costs a few megabytes.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      espeak-ng \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the model layers.
COPY pyproject.toml README.md ./
COPY preflight/__init__.py preflight/__init__.py
RUN pip install --quiet -e '.[asr]'

# Bake the ASR weights. This is the layer that makes the image self-sufficient.
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('base.en', device='cpu', compute_type='int8'); \
print('base.en cached')"

COPY . .

# The corpus is generated from its authoring script, so the image always
# carries a corpus whose manifest hashes match its clause files.
RUN python scripts/build_corpus.py \
 && python scripts/make_assets.py

# Never run as root. A container that writes root-owned files into a mounted
# working directory is a small cruelty to whoever runs it.
RUN useradd --create-home --uid 1000 preflight \
 && chown -R preflight:preflight /app
USER preflight

ENTRYPOINT ["preflight"]
CMD ["--help"]
