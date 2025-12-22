#!/bin/bash
# Clear resume markers to force re-run all configs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

if [ ! -d "${LOG_DIR}" ]; then
    echo "No logs directory found. Nothing to clear."
    exit 0
fi

SUCCESS_FILES=$(find "${LOG_DIR}" -name "*.success" 2>/dev/null)
COUNT=$(echo "${SUCCESS_FILES}" | grep -c ".success" || echo "0")

if [ "${COUNT}" -eq "0" ]; then
    echo "No success markers found. All configs will run."
    exit 0
fi

echo "Found ${COUNT} success marker(s):"
echo "${SUCCESS_FILES}"
echo ""
read -p "Delete all success markers? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "${LOG_DIR}"/*.success
    echo "✓ Cleared ${COUNT} success marker(s)"
    echo "All configs will run on next execution."
else
    echo "Cancelled. No markers deleted."
fi
