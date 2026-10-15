#!/usr/bin/env bash
# Finds forbidden files that are tracked in Git.
set -euo pipefail

bad='(^|/)\.env$|(^|/)\.env\.[^/]+$|\.(pem|key|pt|pth|ckpt|safetensors|gguf|onnx|parquet)$|(^|/)\.venv/|(^|/)__pycache__/|(^|/)data/raw/'

hits=$(git ls-files | grep -E "$bad" | grep -v -E '(^|/)\.env\.example$' || true)

if [ -n "$hits" ]; then
echo "FORBIDDEN tracked files:" >&2
echo "$hits" >&2
exit 1
fi
echo "OK: no forbidden files tracked"

