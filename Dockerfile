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
#   nodejs/npm  → the Claude Code CLI + py-tgcalls' node runtime
#   ffmpeg/libopus0 → voice-call audio (opt-in feature, but deps are hard reqs)
#   git, ca-certificates, curl → install + general runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git ffmpeg libopus0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    # Claude Code 2.x ships the CLI as a platform-specific optional-dep binary;
    # npm's `claude` bin symlink is sometimes NOT created on a clean global
    # install, leaving the executor unable to spawn `claude`. Force the symlink
    # to the real binary and verify at build time so a broken link fails loudly.
    && ln -sf /usr/lib/node_modules/@anthropic-ai/claude-code/node_modules/@anthropic-ai/claude-code-linux-x64/claude /usr/local/bin/claude \
    && claude --version \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

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
