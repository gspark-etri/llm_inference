#!/usr/bin/env bash
# Usage: ./scripts/tunnel.sh jump_user@jump_host:jump_port target_user@target_host target_port
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <jump_user@jump_host:jump_port> <target_user@target_host> <target_ssh_port>" >&2
  exit 1
fi
JUMP=$1
TARGET=$2
TPORT=$3
ssh -J "$JUMP" -p "$TPORT" "$TARGET"   -L 3000:localhost:3000   -L 9090:localhost:9090   -L 8000:localhost:8000   -L 9400:localhost:9400 -N
