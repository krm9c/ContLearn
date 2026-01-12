#!/bin/bash
# Full GPU Profiling Suite for MNIST Condition 1
#
# Usage:
#   ./scripts/run_profiling.sh [--quick] [--skip-training]
#
# This script runs:
#   1. profile_condition1.py - Detailed profiling of training loop
#   2. compare_configurations.py - Compare different settings
#
# Output:
#   Reports are saved to runs__/profiling/reports/
#
# Added by Claude: January 2025

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/runs__/profiling/reports"

# Parse arguments
QUICK=""
SKIP_TRAINING=""
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK="--quick"
            ;;
        --skip-training)
            SKIP_TRAINING="--skip-training"
            ;;
    esac
done

echo "========================================"
echo "GPU Profiling Suite"
echo "========================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Quick mode: ${QUICK:-no}"
echo "Skip training: ${SKIP_TRAINING:-no}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Print GPU info
echo "----------------------------------------"
echo "GPU Information:"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "nvidia-smi not available"
fi
echo ""

# Print JAX info
echo "----------------------------------------"
echo "JAX Information:"
echo "----------------------------------------"
python -c "import jax; print(f'Backend: {jax.default_backend()}'); print(f'Devices: {jax.devices()}')"
echo ""

# Run profile_condition1.py
echo "========================================"
echo "Phase 1: Detailed Training Profile"
echo "========================================"
python scripts/profile_condition1.py $QUICK $SKIP_TRAINING --output-dir "$OUTPUT_DIR"

# Run compare_configurations.py
echo ""
echo "========================================"
echo "Phase 2: Configuration Comparison"
echo "========================================"
python scripts/compare_configurations.py $QUICK --output-dir "$OUTPUT_DIR"

# Summary
echo ""
echo "========================================"
echo "Profiling Complete!"
echo "========================================"
echo ""
echo "Reports saved to: $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON reports found"
echo ""
