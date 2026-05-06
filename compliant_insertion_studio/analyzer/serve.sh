#!/bin/bash
# Serve the analyzer dashboard from the compliant_insertion_studio root
# (so the HTML can fetch ../logs/<csv> via relative URLs).
#
# Usage:
#   bash compliant_insertion_studio/analyzer/serve.sh [port]
#
# Then open: http://localhost:<port>/analyzer/analyze_inserts.html
set -euo pipefail

PORT="${1:-8765}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"   # compliant_insertion_studio/

echo "Regenerating manifest.json..."
python3 - <<'PY'
import json
from pathlib import Path
analyzer = Path(__file__).resolve().parent if "__file__" in globals() else Path("compliant_insertion_studio/analyzer")
PY

# Regenerate manifest.json (cheap; ensures freshness if new episodes were added)
python3 "$HERE/_build_manifest.py" || {
  echo "[warn] manifest rebuild script missing; using existing manifest.json"
}

URL="http://localhost:${PORT}/analyzer/analyze_inserts.html"
echo ""
echo "Serving $ROOT on port $PORT"
echo "Open:  $URL"
echo ""

# Open in default browser if possible (best-effort, ignore failure)
if command -v xdg-open >/dev/null 2>&1; then
  (sleep 1; xdg-open "$URL" >/dev/null 2>&1) &
fi

cd "$ROOT"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
