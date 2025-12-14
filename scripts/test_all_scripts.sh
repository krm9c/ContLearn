#!/bin/bash
# Test all experimental scripts with minimal settings
#
# Usage:
#   ./scripts/test_all_scripts.sh           # Run all script tests
#   ./scripts/test_all_scripts.sh -v        # Verbose output
#   ./scripts/test_all_scripts.sh -k sine   # Run only tests matching 'sine'
#   ./scripts/test_all_scripts.sh --tb=short # Short traceback

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Testing All Experimental Scripts${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Running minimal tests with:"
echo "  - debug_mode: true"
echo "  - debug_limit: 50 samples"
echo "  - epochs_per_task: 2"
echo "  - arch_search_max_iter: 1"
echo ""

# Run tests
if pytest scripts/test_scripts.py "$@"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ All script tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Check if markdown report was generated
    if [ -f "SCRIPT_TEST_RESULTS.md" ]; then
        echo -e "${GREEN}📝 Training outputs saved to:${NC} SCRIPT_TEST_RESULTS.md"
        echo ""
        echo "The markdown report contains:"
        echo "  - Test configuration for each script"
        echo "  - Final training metrics (losses, accuracy)"
        echo "  - Categorized by problem type (Regression, Classification, etc.)"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Some script tests failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "See TODO list above for failures to fix"

    # Check if partial markdown report was generated
    if [ -f "SCRIPT_TEST_RESULTS.md" ]; then
        echo ""
        echo -e "${YELLOW}📝 Partial results saved to:${NC} SCRIPT_TEST_RESULTS.md"
    fi

    exit 1
fi