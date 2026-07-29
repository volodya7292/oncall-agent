# syntax=docker/dockerfile:1
# Cloud-primary (ONCALL_ROLE=server) image for the oncall orchestrator.
#
# Runs the full daemon — operator, executor (Claude CLI), Telegram, broker —
# on an always-on VPS. The user's laptop runs `oncall laptop-worker` and
# reaches this container only via outbound long-poll, so the laptop needs no
# inbound exposure.
#
# Auth: the executor uses SUBSCRIPTION OAuth (no ANTHROPIC_API_KEY). After the
# first deploy, run once:
#     docker exec -it oncall claude login
# The credential persists on the mounted /root/.claude volume.
#
# Claude Code keeps a SECOND state file at ~/.claude.json — a sibling of that
# directory, not a member of it — holding the oauth account, per-directory
# trust, and model caches. Mounting only /root/.claude leaves it in the
# container's own filesystem, so every `compose up` that recreates the
# container silently reverts it to the copy baked in at build time: the
# executor comes back deauthenticated and every hand_off dies with
# "OAuth session expired and could not be refreshed".
#
# It is symlinked into the mounted directory below rather than bind-mounted as
# a file, because a file bind whose host path is missing makes Docker create a
# DIRECTORY there — which breaks the CLI differently and only on fresh hosts.
# Directory binds have no such trap. Verified that the CLI writes through the
# symlink (open+write on the resolved path) instead of renaming over it, so
# the link survives config writes and works even while dangling on first boot.
#
# Run (single container; Ollama already on the host, no compose):
#     docker run -d --name oncall --restart unless-stopped \
#       -e ONCALL_BIND_HOST=0.0.0.0 \
#       -e ONCALL_OLLAMA_HOST=http://host.docker.internal:11434 \
#       --add-host host.docker.internal:host-gateway \
#       -v oncall_state:/root/.oncall -v oncall_claude:/root/.claude \
#       --expose 8765 \
#       ghcr.io/<owner>/oncall-agent:latest
# Put a TLS reverse proxy (Caddy/nginx) in front that forwards ONLY /laptop/*
# publicly; do NOT publish :8765 raw to the internet.

FROM python:3.12-slim

# System deps:
#   ffmpeg/libopus0 → voice-call audio. py-tgcalls uses the NATIVE ntgcalls
#       backend (prebuilt cffi wheel) — no Node runtime needed.
#   git, ca-certificates, curl → install + general runtime
#
# Claude Code is installed via Anthropic's NATIVE installer (self-contained
# binary), NOT npm — this drops the entire Node/npm toolchain from the image
# and avoids the npm optional-dep bin-symlink flakiness that left `claude` off
# PATH. We symlink it onto a standard PATH dir and run `claude --version` so a
# broken/missing binary fails the build loudly instead of shipping silently.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git ffmpeg libopus0 \
    && curl -fsSL https://claude.ai/install.sh | bash \
    && ln -sf /root/.local/bin/claude /usr/local/bin/claude \
    && claude --version \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Relocate ~/.claude.json into the persisted directory (see Auth note above).
# Must run AFTER the install step, which creates a real file there via
# `claude --version`. The link target does not exist at build time and does not
# need to: the CLI creates it on first write.
RUN mkdir -p /root/.claude \
    && rm -f /root/.claude.json \
    && ln -s /root/.claude/claude.json /root/.claude.json

WORKDIR /app

# 1a) CPU-only torch. silero-vad (voice VAD) depends on torch; the default
#     PyPI torch wheel on Linux is the CUDA build and drags in ~2-3 GB of
#     unusable nvidia-cuda-*/cudnn/cublas/triton libraries. Pre-install the
#     CPU wheels from PyTorch's CPU index so the dependency resolver below
#     finds torch/torchaudio already satisfied and never pulls the GPU stack.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio

# 1b) Dependency layer — cache-keyed on pyproject.toml ONLY. Source-only
#     changes (the common case) keep this a cache hit, so the heavy dependency
#     set is NOT reinstalled every build. Extract the PEP 508 deps from
#     pyproject and install them without the project. The pip cache mount
#     reuses already-downloaded wheels when the layer does rebuild.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# 2) Project layer — only this rebuilds on a code change; deps already present.
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .

# Server role: native executor tools disabled; local work routed to the laptop.
ENV ONCALL_ROLE=server \
    ONCALL_BIND_HOST=0.0.0.0

# Persist orchestrator state (DB, Telegram sessions) and the Claude OAuth login.
VOLUME ["/root/.oncall", "/root/.claude"]

EXPOSE 8765
ENTRYPOINT ["oncall", "api"]
