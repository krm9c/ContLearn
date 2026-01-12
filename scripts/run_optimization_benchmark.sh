#!/bin/bash
# Comprehensive Optimization Benchmark Runner
#
# This script runs each benchmark configuration in a separate subprocess
# to ensure proper XLA flag isolation between runs.
#
# Usage:
#   ./scripts/run_optimization_benchmark.sh [--quick]
#
# Added by Claude: January 2026

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Parse arguments
QUICK=""
if [[ "$1" == "--quick" || "$1" == "-q" ]]; then
    QUICK="--quick"
    echo "Running in quick mode..."
fi

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "OPTIMIZATION BENCHMARK RUNNER"
echo "========================================"
echo "Results directory: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Function to run a single benchmark configuration
run_benchmark() {
    local name=$1
    local xla_enabled=$2
    local fused_enabled=$3
    local awb_enabled=$4

    echo ""
    echo "----------------------------------------"
    echo "Running: $name"
    echo "  XLA: $xla_enabled, Fused: $fused_enabled, AWB: $awb_enabled"
    echo "----------------------------------------"

    # Build environment
    local env_vars=""
    if [[ "$xla_enabled" == "true" ]]; then
        env_vars="XLA_FLAGS='--xla_gpu_enable_fast_min_max=true --xla_gpu_enable_async_collectives=true --xla_gpu_enable_latency_hiding_scheduler=true' TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2"
    fi

    # Build Python command
    local python_cmd="python -c \"
import sys
sys.path.insert(0, '$PROJECT_ROOT')

# Set config
config = {
    'name': '$name',
    'xla_enabled': $xla_enabled,
    'fused_enabled': $fused_enabled,
    'awb_enabled': $awb_enabled,
    'output_file': '$RESULTS_DIR/${name}.json',
}

# Import and run
from scripts.benchmark_single import run_single_config
run_single_config(config)
\""

    # Run with timing
    local start_time=$(date +%s)

    if [[ -n "$env_vars" ]]; then
        eval "$env_vars python $PROJECT_ROOT/scripts/benchmark_single.py --name '$name' --xla $xla_enabled --fused $fused_enabled --awb $awb_enabled --output '$RESULTS_DIR/${name}.json' $QUICK" || true
    else
        python "$PROJECT_ROOT/scripts/benchmark_single.py" --name "$name" --xla "$xla_enabled" --fused "$fused_enabled" --awb "$awb_enabled" --output "$RESULTS_DIR/${name}.json" $QUICK || true
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "Completed in ${elapsed}s"
}

# Record system info
echo "Recording system info..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$RESULTS_DIR/gpu_info.csv" 2>/dev/null || echo "nvidia-smi not available"

# Run all configurations
echo ""
echo "========================================"
echo "RUNNING BENCHMARKS"
echo "========================================"

# Baseline configurations (Condition 1)
run_benchmark "baseline_no_xla" "false" "false" "false"
run_benchmark "baseline_xla_only" "true" "false" "false"
run_benchmark "baseline_fused_only" "false" "true" "false"
run_benchmark "baseline_xla_and_fused" "true" "true" "false"

# AWB configurations (Condition 4)
run_benchmark "awb_no_xla" "false" "false" "true"
run_benchmark "awb_xla_only" "true" "false" "true"
run_benchmark "awb_fused_only" "false" "true" "true"
run_benchmark "awb_xla_and_fused" "true" "true" "true"

echo ""
echo "========================================"
echo "COMBINING RESULTS"
echo "========================================"

# Combine all results into single file
python -c "
import json
import os
from pathlib import Path
from datetime import datetime

results_dir = Path('$RESULTS_DIR')
all_results = []

for f in results_dir.glob('*.json'):
    if f.name != 'combined_results.json':
        try:
            with open(f) as fp:
                all_results.append(json.load(fp))
        except Exception as e:
            print(f'Error loading {f}: {e}')

# Sort by config name
all_results.sort(key=lambda x: x.get('config_name', ''))

combined = {
    'generated_at': datetime.now().isoformat(),
    'results_dir': str(results_dir),
    'num_configs': len(all_results),
    'results': all_results,
}

output_file = results_dir / 'combined_results.json'
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f'Combined {len(all_results)} results into {output_file}')
"

echo ""
echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_benchmark.py $RESULTS_DIR/combined_results.json"
