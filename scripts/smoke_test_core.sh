#!/usr/bin/env bash
# smoke_test_core.sh -- Health check for all FORTHought services
# Author: Marios Adamidis (FORTHought Lab)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Docker compose config check =="
docker compose --profile core config >/dev/null
echo "OK"

echo
echo "== Core containers running? =="
docker compose --profile core ps

echo
echo "== HTTP health checks (host ports) =="

check() {
  local name="$1" url="$2"
  echo -n "$name -> $url ... "
  if curl -fsS --max-time 5 "$url" >/dev/null; then
    echo "OK"
  else
    echo "FAIL"
    return 1
  fi
}

# Fileserver FastAPI
check "fileserver" "http://127.0.0.1:8084/health"

# Docling proxy (your Open WebUI points here)
check "docling-proxy" "http://127.0.0.1:5003/health"

# Open WebUI: root should return 200 (HTML)
check "open-webui" "http://127.0.0.1:8081/"

# Jupyter: should return 200 or 302
echo -n "jupyter -> http://127.0.0.1:8888/lab ... "
code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:8888/lab")"
# 200 = page served, 302 = redirect, 403 = auth gate (still alive)
if [[ "$code" == "200" || "$code" == "302" || "$code" == "403" ]]; then
  echo "OK ($code)"
else
  echo "FAIL ($code)"
  exit 1
fi


echo
echo "== In-container service checks =="

# Qdrant is not host-mapped, so check from inside container
echo -n "qdrant /healthz ... "
docker exec -i open_webui_container python -c \
"import urllib.request; urllib.request.urlopen('http://qdrant:6333/healthz', timeout=3).read(); print('OK')"

echo
echo "== Fileserver directory sanity =="
echo -n "listing /shared_data/files inside fileserver ... "
docker exec -i fileserver_container sh -lc 'ls -la /shared_data/files | head -n 20' >/dev/null && echo "OK" || (echo "FAIL" && exit 1)

echo
echo "All core smoke tests passed."
