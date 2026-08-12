#!/usr/bin/env bash
# launch.sh — MAGI node launcher
#   ./launch.sh                 pre-flight checks, then run the daemon
#   ./launch.sh --check-only    checks and exit (useful over ssh)
#   ./launch.sh --skip-checks   straight to the daemon
#   ./launch.sh --help          every setting, its value and where it came from
#
# --help delegates the configuration table to `magi.config_help`, which
# generates it from Settings itself. A list written out here would be a second
# copy of config.py, and the drift would be discovered on the Pi.
#
# Unlike latacc-edge's launcher, these checks CAN abort. That node was built to
# survive every dependency being down; this one cannot hold a debate if a model
# is missing or cannot honour the output schema, so those are hard errors. The
# checks themselves live in Python (magi.preflight) because the decisive
# one — asking each model for a schema-constrained MagiTurn — needs the same
# Pydantic models the daemon uses. Reimplementing it in bash would test a
# different thing.
set -uo pipefail
cd "$(dirname "$0")"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; DIM='\033[2m'; NC='\033[0m'

usage() {
    echo "Usage: ./launch.sh [--skip-checks|--check-only|--help]"
    echo
    echo "  (no flag)       pre-flight checks, then run the daemon"
    echo "  --check-only    run the pre-flight checks and exit"
    echo "  --skip-checks   start the daemon without checking anything"
    echo "  --help          this, plus every configuration setting"
}

SKIP_CHECKS=0
CHECK_ONLY=0
case "${1:-}" in
    --skip-checks) SKIP_CHECKS=1 ;;
    --check-only)  CHECK_ONLY=1 ;;
    "")            ;;
    -h|--help)
        usage
        echo
        uv run python -m magi.config_help
        exit 0
        ;;
    *) usage >&2; exit 1 ;;
esac

if [[ "$SKIP_CHECKS" != "1" ]]; then
    uv run python -m magi.preflight
    rc=$?
    # 1 = a check failed, 2 = the persona file is unusable. Both stop the launch;
    # --check-only reports and exits regardless, which is the point of it.
    if [[ "$CHECK_ONLY" == "1" ]]; then
        exit $rc
    fi
    if [[ $rc -ne 0 ]]; then
        exit $rc
    fi
elif [[ "$CHECK_ONLY" == "1" ]]; then
    echo "--check-only and --skip-checks are contradictory."
    exit 1
fi

# The port comes from the same place the daemon reads it, so the URL printed
# here cannot drift from the one actually being served.
UI_PORT="${MAGI_UI_PORT:-$(grep -E "^MAGI_UI_PORT=" .env 2>/dev/null | cut -d= -f2)}"
UI_PORT="${UI_PORT:-8000}"

echo -e "${CYAN}▸ Starting MAGI${NC}"
echo -e "  Console:  ${GREEN}http://localhost:${UI_PORT}${NC}"
echo -e "  ${DIM}Kiosk:    chromium-browser --kiosk --noerrdialogs http://localhost:${UI_PORT}${NC}"
echo -e "  ${DIM}The microphone needs a secure context, so open it at localhost —"
echo -e "  a LAN address gets the console but no PTT.${NC}"
echo
uv run python -m magi
