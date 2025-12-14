#!/bin/bash
# Test runner script for cl_framework
# Provides options for running the pytest test suite

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Display help message
show_help() {
    cat << EOF
CL Framework Test Runner

Usage: ./run_tests.sh [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -a, --all               Run all tests (default)
    -f, --fast              Run only fast tests (skip integration tests)
    -l, --layers            Run layer tests only
    -m, --models            Run model tests only
    -d, --datasets          Run dataset tests only
    -o, --losses            Run loss function tests only
    -w, --awb               Run AWB utility tests only
    -r, --recording         Run recording tests only
    -i, --integration       Run integration tests only
    -v, --verbose           Run with verbose output
    -s, --stdout            Show print statements
    -k, --keyword PATTERN   Run tests matching PATTERN
    --cov                   Run with coverage report

EXAMPLES:
    ./run_tests.sh --all                    # Run all tests
    ./run_tests.sh --fast                   # Skip slow integration tests
    ./run_tests.sh --models --verbose       # Run model tests with verbose output
    ./run_tests.sh -k regression            # Run tests with 'regression' in name

EOF
}

# # Change to project root
# cd "$(dirname "$0")/.."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jaxss

# Default values
MODE="all"
VERBOSE=""
STDOUT=""
COVERAGE=""
KEYWORD=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--all)
            MODE="all"
            shift
            ;;
        -f|--fast)
            MODE="fast"
            shift
            ;;
        -l|--layers)
            MODE="layers"
            shift
            ;;
        -m|--models)
            MODE="models"
            shift
            ;;
        -d|--datasets)
            MODE="datasets"
            shift
            ;;
        -o|--losses)
            MODE="losses"
            shift
            ;;
        -w|--awb)
            MODE="awb"
            shift
            ;;
        -r|--recording)
            MODE="recording"
            shift
            ;;
        -i|--integration)
            MODE="integration"
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -s|--stdout)
            STDOUT="-s"
            shift
            ;;
        -k|--keyword)
            KEYWORD="-k $2"
            shift 2
            ;;
        --cov)
            COVERAGE="--cov=src/cl --cov-report=term-missing"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed. Install it with: pip install pytest"
    exit 1
fi

# Run tests based on mode
case $MODE in
    all)
        print_header "Running All Tests"
        pytest tests/ $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
    fast)
        print_header "Running Fast Tests (Skipping Integration Tests)"
        pytest tests/ -k "not Integration" $VERBOSE $STDOUT $COVERAGE
        ;;
    layers)
        print_header "Running Layer Tests"
        pytest tests/test_layers.py $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
    models)
        print_header "Running Model Tests"
        pytest tests/test_models.py $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
    datasets)
        print_header "Running Dataset Tests"
        pytest tests/test_datasets.py $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
    losses)
        print_header "Running Loss Function Tests"
        pytest tests/test_losses.py $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
    awb)
        print_header "Running AWB Utility Tests"
        pytest tests/test_awb.py $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
    recording)
        print_header "Running Recording Tests"
        pytest tests/test_recording.py $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
    integration)
        print_header "Running Integration Tests"
        pytest tests/test_integration.py $VERBOSE $STDOUT $COVERAGE $KEYWORD
        ;;
esac

# Capture exit code
TEST_EXIT_CODE=$?

# Print summary
echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "All tests passed!"
else
    print_error "Some tests failed. See output above for details."
fi

exit $TEST_EXIT_CODE
