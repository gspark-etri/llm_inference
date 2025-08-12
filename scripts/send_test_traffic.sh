#!/usr/bin/env bash
set -euo pipefail
MODEL="meta-llama/Llama-3.1-8B-Instruct"
N=${1:-50}
CONCURRENCY=${2:-5}
run_one() {
  curl -s http://localhost:8000/v1/chat/completions     -H "Authorization: Bearer dummy" -H "Content-Type: application/json"     -d '{
      "model":"'"$MODEL"'",
      "messages":[{"role":"user","content":"Say hello and one fact."}],
      "max_tokens":64,
      "stream":true
    }' >/dev/null
}
echo "[INFO] sending ${N} streaming requests..."
i=0
while [ "$i" -lt "$N" ]; do
  j=0
  while [ "$j" -lt "$CONCURRENCY" ] && [ "$i" -lt "$N" ]; do
    run_one &
    i=$((i+1))
    j=$((j+1))
    sleep 0.05
  done
  wait
done
echo "[OK] done."
