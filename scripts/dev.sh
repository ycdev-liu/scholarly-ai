#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

cleanup() {
    if [[ -n "${SERVICE_PID:-}" ]]; then
        kill "$SERVICE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

uv run --python 3.12 python backend/app/run_service.py &
SERVICE_PID=$!

echo "API: http://127.0.0.1:8080"
echo "Web: http://127.0.0.1:8501"
uv run --python 3.12 streamlit run backend/app/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501
