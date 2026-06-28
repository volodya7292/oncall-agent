#!/usr/bin/env bash
# Rebuild the wheel, reinstall it into the uv-tool slot, and restart the
# launchd-managed daemon so source-tree edits land in the running service.
# See CLAUDE.md "Reinstalling after source changes" for why this is needed.

set -euo pipefail

cd "$(dirname "$0")/.."

uv build
WHEEL="$(ls -t dist/oncall_agent-*-py3-none-any.whl | head -n1)"
uv tool install --force "$WHEEL"
oncall service --worker start
