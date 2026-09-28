#!/bin/sh
set -eu

cd /app
python /app/scripts/ensure_checkpoint.py

# Keep the finalized API local to this container. Only Streamlit is exposed by
# Render, so the free single-service deployment does not publish the API.
uvicorn src.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
    kill "$API_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if ! python - <<'PY'
import time
from urllib.request import urlopen

deadline = time.monotonic() + 90
while time.monotonic() < deadline:
    try:
        with urlopen("http://127.0.0.1:8000/openapi.json", timeout=2):
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("The local model API did not become ready within 90 seconds.")
PY
then
    exit 1
fi

export API_BASE_URL="http://127.0.0.1:8000/api"
streamlit run ui/app.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT:-8501}" \
    --browser.gatherUsageStats=false &
UI_PID=$!

wait "$UI_PID"
