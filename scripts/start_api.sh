#!/bin/sh
set -eu

cd /app
python /app/scripts/ensure_checkpoint.py
exec uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"
