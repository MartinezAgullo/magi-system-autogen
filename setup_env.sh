#!/usr/bin/env bash
# setup_env.sh — one command from a fresh clone to a working environment.
#
# There is no requirements.txt: pyproject.toml plus uv.lock are the single
# source of truth for dependencies. A second list would drift, and the drift
# would be discovered on the Pi.
#   ./setup_env.sh            core only — enough to hold and benchmark debates
#   ./setup_env.sh --voice    adds STT/TTS, which is a long install on ARM
set -euo pipefail
cd "$(dirname "$0")"

EXTRAS="--extra dev"
if [[ "${1:-}" == "--voice" ]]; then
    EXTRAS="--extra dev --extra voice"
fi

if ! command -v uv > /dev/null 2>&1; then
    echo "uv is not installed. Get it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "▸ Installing dependencies ($EXTRAS)"
# Not --all-extras: `voice` drags in ctranslate2, which takes a long time to
# install on a Pi and is not needed until the node actually speaks.
uv sync $EXTRAS

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "▸ Created .env from .env.example — check the host addresses in it"
fi

mkdir -p data

echo
echo "Done. Next:"
echo "  ./launch.sh --check-only    verify the inference backend"
echo "  ./launch.sh                 checks, then run the node"
