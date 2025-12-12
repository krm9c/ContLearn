#!/bin/bash
# Test runner script for ContLearn
# Provides various options for running the test suite

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
ContLearn Test Runner

Usage: ./run_tests.sh [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -a, --all               Run all tests (default)
    -f, --fast              Run only fast tests (skip integration tests)
    -m, --models            Run model tests only
    -d, --data              Run data tests only
    -t, --trainer           Run trainer tests only
    -g, --graph             Run graph model tests only
    -c, --checkpoint        Run checkpoint tests only
    -r, --runners           Run training runner tests only
    -u, --utils             Run utility tests only
    -v, --verbose           Run with verbose output
    -s, --stdout            Show print statements
    -k, --keyword PATTERN   Run tests matching PATTERN
    --cov                   Run with coverage report
    --cov-html              Generate HTML coverage report
    --parallel              Run tests in parallel (requires pytest-xdist)
    --markers               List available test markers

EXAMPLES:
    ./run_tests.sh --all                    # Run all tests
    ./run_tests.sh --fast                   # Skip slow integration tests
    ./run_tests.sh --models --verbose       # Run model tests with verbose output
    ./run_tests.sh -k regression            # Run tests with 'regression' in name
    ./run_tests.sh --cov                    # Run with coverage report
    ./run_tests.sh --parallel               # Run tests in parallel

EOF
}

# Default values
MODE="all"
VERBOSE=""
STDOUT=""
COVERAGE=""
PARALLEL=""
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
        -m|--models)
            MODE="models"
            shift
            ;;
        -d|--data)
            MODE="data"
            shift
            ;;
        -t|--trainer)
            MODE="trainer"
            shift
            ;;
        -g|--graph)
            MODE="graph"
            shift
            ;;
        -c|--checkpoint)
            MODE="checkpoint"
            shift
            ;;
        -r|--runners)
            MODE="runners"
            shift
            ;;
        -u|--utils)
            MODE="utils"
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
            COVERAGE="--cov=utils --cov=training --cov=config --cov=data --cov-report=term-missing"
            shift
            ;;
        --cov-html)
            COVERAGE="--cov=utils --cov=training --cov=config --cov=data --cov-report=html"
            shift
            ;;
        --parallel)
            PARALLEL="-n auto"
            shift
            ;;
        --markers)
            pytest --markers
            exit 0
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

# Check for parallel testing dependency
if [[ -n "$PARALLEL" ]] && ! python -c "import xdist" 2>/dev/null; then
    print_warning "pytest-xdist not installed. Install it with: pip install pytest-xdist"
    PARALLEL=""
fi

# Run tests based on mode
case $MODE in
    all)
        print_header "Running All Tests"
        pytest tests/ $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
    fast)
        print_header "Running Fast Tests (Skipping Integration Tests)"
        pytest tests/ -k "not (train_model_reg or train_model_class or train_model_graph)" $VERBOSE $STDOUT $COVERAGE $PARALLEL
        ;;
    models)
        print_header "Running Model Tests"
        pytest tests/test_models.py tests/test_cnn3d.py $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
    data)
        print_header "Running Data Tests"
        pytest tests/test_data.py $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
    trainer)
        print_header "Running Trainer Tests"
        pytest tests/test_trainer.py $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
    graph)
        print_header "Running Graph Model Tests"
        pytest tests/test_graph_models.py $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
    checkpoint)
        print_header "Running Checkpoint Tests"
        pytest tests/test_checkpoint.py tests/test_config.py $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
    runners)
        print_header "Running Training Runner Tests (Integration Tests)"
        pytest tests/test_runners.py $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
    utils)
        print_header "Running Utility Tests"
        pytest tests/test_utils.py $VERBOSE $STDOUT $COVERAGE $PARALLEL $KEYWORD
        ;;
esac

# Capture exit code
TEST_EXIT_CODE=$?

# Print summary
echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "All tests passed!"

    # Show coverage report location if HTML was generated
    if [[ $COVERAGE == *"html"* ]]; then
        echo ""
        print_success "Coverage report generated at: htmlcov/index.html"
        echo "Open it with: open htmlcov/index.html (Mac) or xdg-open htmlcov/index.html (Linux)"
    fi
else
    print_error "Some tests failed. See output above for details."
fi

exit $TEST_EXIT_CODE
