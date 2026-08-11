#!/usr/bin/env bash
# compose.sh — bring the trace backend up or down from anywhere in the repo.
#
#   ./docker/compose.sh up       start Jaeger, print where to point the node
#   ./docker/compose.sh down     stop it
#   ./docker/compose.sh logs     follow
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-up}" in
    up)
        docker compose up -d
        # The node usually runs on the Pi, so localhost is the wrong address to
        # print — it needs the LAN address of whichever machine this is.
        ip=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
        echo
        echo "Jaeger UI:  http://localhost:16686"
        echo "Point the node at:  MAGI_OTEL_ENDPOINT=http://${ip:-<this-host>}:4318"
        ;;
    down) docker compose down ;;
    logs) docker compose logs -f ;;
    *)    echo "Usage: ./docker/compose.sh [up|down|logs]"; exit 1 ;;
esac
