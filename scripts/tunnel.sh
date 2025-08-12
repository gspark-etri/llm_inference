#!/usr/bin/env bash
# Usage: ./scripts/tunnel.sh jump_user@jump_host:jump_port target_user@target_host target_port

# If .env exists, use it to set the proxy and target
if [ ! -f .env ]; then
  if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <jump_user@jump_host:jump_port> <target_user@target_host> <target_ssh_port>" >&2
    exit 1
  fi
  echo "[INFO] Tunneling to $1:$2:$3"
  JUMP=$1
  TARGET=$2
  TPORT=$3

  ssh -J "$JUMP" -p "$TPORT" "$TARGET"   -L 3000:localhost:3000   -L 9090:localhost:9090   -L 8000:localhost:8000   -L 9400:localhost:9400 -N
else
  PROXY_ID=$(grep PROXY_ID .env | cut -d '=' -f 2 | tr -d '"')
  PROXY_IP=$(grep PROXY_IP .env | cut -d '=' -f 2 | tr -d '"')
  PROXY_PORT=$(grep PROXY_PORT .env | cut -d '=' -f 2 | tr -d '"')
  TARGET_ID=$(grep TARGET_ID .env | cut -d '=' -f 2 | tr -d '"')
  TARGET_IP=$(grep TARGET_IP .env | cut -d '=' -f 2 | tr -d '"')
  TARGET_PORT=$(grep TARGET_PORT .env | cut -d '=' -f 2 | tr -d '"')
  echo "[INFO] Tunneling to $TARGET_ID@$TARGET_IP:$TARGET_PORT via $PROXY_ID@$PROXY_IP:$PROXY_PORT"
  ssh -o ProxyCommand="ssh -p $PROXY_PORT -W %h:%p $PROXY_ID@$PROXY_IP" \
    -p $TARGET_PORT $TARGET_ID@$TARGET_IP \
    -L 3000:localhost:3000 -L 9090:localhost:9090 -L 8000:localhost:8000 -L 9400:localhost:9400 -N
fi


