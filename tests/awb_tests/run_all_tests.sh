#!/bin/bash
#
# Run All AWB Tests
#
# This script runs all AWB pipeline tests and benchmarks.
# Tests are organized by step to verify the 5-step AWB algorithm.
#
# Usage:
#   ./awb_tests/run_all_tests.sh           # Run all tests
#   ./awb_tests/run_all_tests.sh --quick   # Run quick tests only
#   ./awb_tests/run_all_tests.sh --bench   # Run benchmarks only
#   ./awb_tests/run_all_tests.sh --verbose # Verbose output
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUICK=false
BENCH_ONLY=false
VERBOSE=""

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=true
            ;;
        --bench)
            BENCH_ONLY=true
            ;;
        --verbose)
            VERBOSE="--verbose"
            ;;
    esac
done

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}       AWB Pipeline Test Suite        ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Test directory: $SCRIPT_DIR"
echo ""

cd "$PROJECT_ROOT"

# Track results
PASSED=0
FAILED=0
SKIPPED=0
RESULTS=""

run_test() {
    local test_name=$1
    local test_script=$2
    local extra_args=$3

    echo -e "\n${YELLOW}Running: $test_name${NC}"
    echo "----------------------------------------"

    if python "$test_script" $VERBOSE $extra_args; then
        echo -e "${GREEN}[PASSED]${NC} $test_name"
        PASSED=$((PASSED + 1))
        RESULTS="${RESULTS}\n${GREEN}[PASSED]${NC} $test_name"
    else
        echo -e "${RED}[FAILED]${NC} $test_name"
        FAILED=$((FAILED + 1))
        RESULTS="${RESULTS}\n${RED}[FAILED]${NC} $test_name"
    fi
}

skip_test() {
    local test_name=$1
    echo -e "${YELLOW}[SKIPPED]${NC} $test_name"
    SKIPPED=$((SKIPPED + 1))
    RESULTS="${RESULTS}\n${YELLOW}[SKIPPED]${NC} $test_name"
}

# Benchmarks only mode
if [ "$BENCH_ONLY" = true ]; then
    echo -e "\n${BLUE}=== Running Benchmarks Only ===${NC}"
    run_test "Performance Benchmarks" "awb_tests/benchmark_performance.py" "--output awb_tests/results/benchmark_results.json"

    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}         Benchmark Complete           ${NC}"
    echo -e "${BLUE}======================================${NC}"
    exit 0
fi

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

echo -e "\n${BLUE}=== Step Tests ===${NC}"

# Step 1: Preliminary Training
run_test "Step 1: Preliminary Training" "awb_tests/test_step1_preliminary.py"

# Step 2: Architecture Change Decision
run_test "Step 2: Architecture Decision" "awb_tests/test_step2_decision.py"

# Step 3a: Architecture Search
run_test "Step 3a: Architecture Search" "awb_tests/test_step3a_arch_search.py"

# Step 3b: A/B Training
run_test "Step 3b: A/B Matrix Training" "awb_tests/test_step3b_ab_training.py"

# Step 4: V Transformation
run_test "Step 4: V Transformation" "awb_tests/test_step4_v_transform.py"

# Step 5: V Training
run_test "Step 5: V Training" "awb_tests/test_step5_v_training.py"

echo -e "\n${BLUE}=== Mathematical Correctness Tests ===${NC}"

run_test "Mathematical Correctness" "awb_tests/test_mathematical_correctness.py"

echo -e "\n${BLUE}=== Integration Tests ===${NC}"

if [ "$QUICK" = false ]; then
    run_test "Full Pipeline Integration" "awb_tests/test_full_pipeline.py"
else
    skip_test "Full Pipeline Integration (skipped in quick mode)"
fi

echo -e "\n${BLUE}=== Performance Benchmarks ===${NC}"

if [ "$QUICK" = false ]; then
    run_test "Performance Benchmarks" "awb_tests/benchmark_performance.py" "--output awb_tests/results/benchmark_results.json"
else
    skip_test "Performance Benchmarks (skipped in quick mode)"
fi

# Summary
echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}            TEST SUMMARY              ${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "$RESULTS"
echo ""
echo "----------------------------------------"
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo "----------------------------------------"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed.${NC}"
    exit 1
fi
