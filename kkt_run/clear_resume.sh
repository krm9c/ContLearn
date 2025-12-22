#!/bin/bash
# Clear all success markers to force re-run of all configs
# Usage: bash kkt_run/clear_resume.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

echo "Clearing all success markers in ${LOG_DIR}/"
echo ""

# Count and list files to be deleted
SUCCESS_FILES=$(find "${LOG_DIR}" -name "*.success" 2>/dev/null)
COUNT=$(echo "${SUCCESS_FILES}" | grep -c "\.success$" 2>/dev/null || echo 0)

if [ -z "${SUCCESS_FILES}" ] || [ "${COUNT}" -eq 0 ]; then
    echo "No success markers found. Nothing to clear."
    exit 0
fi

echo "Found ${COUNT} success marker(s):"
echo "${SUCCESS_FILES}" | sed 's/^/  /'
echo ""

# Ask for confirmation
read -p "Delete these files? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "${LOG_DIR}"/*.success
    echo "✓ All success markers cleared."
    echo "Next job submission will re-run all configs."
else
    echo "Cancelled. No files deleted."
fi
