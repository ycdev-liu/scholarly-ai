#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

uv python install 3.12
uv sync --frozen --python 3.12

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

mkdir -p data/downloads/papers data/vector_databases

echo "Environment ready. Start everything with: ./scripts/dev.sh"
