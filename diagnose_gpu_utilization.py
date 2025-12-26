#!/usr/bin/env python
"""
GPU Utilization Diagnostic Script for ContLearn H200 Setup
Run this script and report back the output to diagnose low GPU utilization.

Usage:
    python diagnose_gpu_utilization.py
"""

import os
import sys
import subprocess
import time

def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)

def run_command(cmd, description):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    print("GPU UTILIZATION DIAGNOSTIC REPORT")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # Section 1: Environment Info
    # ========================================================================
    print_section("1. Environment Configuration")

    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'JAX_PLATFORMS',
        'XLA_FLAGS',
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_ALLOCATOR',
        'JAX_ENABLE_X64',
        'TF_CPP_MIN_LOG_LEVEL'
    ]

    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var}: {value}")

    # ========================================================================
    # Section 2: JAX Configuration
    # ========================================================================
    print_section("2. JAX Configuration")

    try:
        import jax
        import jax.numpy as jnp

        print(f"  JAX version: {jax.__version__}")
        print(f"  JAX backend: {jax.default_backend()}")
        print(f"  JAX devices: {jax.devices()}")
        print(f"  Device count: {jax.device_count()}")
        print(f"  Local device count: {jax.local_device_count()}")

        # Check device memory
        for i, device in enumerate(jax.devices()):
            print(f"  Device {i}: {device}")

    except Exception as e:
        print(f"  Error importing JAX: {e}")

    # ========================================================================
    # Section 3: GPU Hardware Info
    # ========================================================================
    print_section("3. GPU Hardware Info")

    gpu_info = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,power.limit --format=csv,noheader",
                          "GPU info")
    if gpu_info and not gpu_info.startswith("Error"):
        lines = gpu_info.split('\n')
        for i, line in enumerate(lines):
            print(f"  GPU {i}: {line}")
    else:
        print(f"  {gpu_info}")

    # Current utilization snapshot
    util_info = run_command("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader",
                           "Current utilization")
    if util_info and not util_info.startswith("Error"):
        lines = util_info.split('\n')
        for i, line in enumerate(lines):
            print(f"  GPU {i} Current: {line}")

    # ========================================================================
    # Section 4: Python Environment
    # ========================================================================
    print_section("4. Python Environment")

    packages = ['jax', 'jaxlib', 'equinox', 'torch', 'torch_geometric', 'numpy']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")

    # ========================================================================
    # Section 5: Batch Processing Benchmark
    # ========================================================================
    print_section("5. GPU Compute Benchmark (Matrix Operations)")

    try:
        import jax
        import jax.numpy as jnp

        print("  Testing different batch sizes to measure GPU throughput...")
        print("  (This measures raw compute capability, not your actual training)")
        print()

        batch_sizes = [1024, 2048, 4096, 8192, 16384]
        feature_dim = 784  # MNIST-like

        # Warmup
        print("  Warming up GPU...")
        x = jax.random.normal(jax.random.PRNGKey(0), (1024, feature_dim))
        _ = jnp.matmul(x, x.T)
        jax.block_until_ready(_)

        print(f"  {'Batch Size':<12} {'Time/Batch (ms)':<18} {'Throughput (samples/sec)':<25} {'Memory Est (MB)'}")
        print("  " + "-" * 75)

        for bs in batch_sizes:
            x = jax.random.normal(jax.random.PRNGKey(0), (bs, feature_dim))

            # Compile
            result = jnp.matmul(x, x.T)
            jax.block_until_ready(result)

            # Benchmark
            n_iters = 100
            start = time.time()
            for _ in range(n_iters):
                result = jnp.matmul(x, x.T)
                jax.block_until_ready(result)
            elapsed = time.time() - start

            time_per_batch = (elapsed / n_iters) * 1000  # ms
            throughput = bs / (elapsed / n_iters)  # samples/sec
            memory_est = (bs * feature_dim * 4) / (1024**2)  # MB (float32)

            print(f"  {bs:<12} {time_per_batch:<18.2f} {throughput:<25.0f} {memory_est:<.2f}")

        print()
        print("  INTERPRETATION:")
        print("  - If time/batch is < 1ms: GPU is underutilized, increase batch size")
        print("  - If throughput scales linearly with batch size: GPU can handle more")
        print("  - Compare 'Memory Est' with 'GPU memory.free' from Section 3")

    except Exception as e:
        print(f"  Error running benchmark: {e}")

    # ========================================================================
    # Section 6: Data Loading Test
    # ========================================================================
    print_section("6. Data Loading Performance Test")

    try:
        import torch
        import numpy as np
        import jax.numpy as jnp

        print("  Testing PyTorch → NumPy → JAX conversion overhead...")

        batch_size = 1024
        img_size = 784

        # Simulate PyTorch DataLoader output
        torch_batch = torch.randn(batch_size, img_size)

        # Time conversion
        n_iters = 1000

        # Method 1: torch → numpy → jax
        start = time.time()
        for _ in range(n_iters):
            jax_batch = jnp.array(torch_batch.numpy())
        elapsed1 = time.time() - start

        print(f"  Torch→NumPy→JAX: {elapsed1/n_iters*1000:.3f} ms/batch")
        print(f"  Overhead for batch_size={batch_size}: {elapsed1/n_iters*1e6:.1f} μs/sample")

        # Estimate impact
        samples_per_epoch = 60000  # MNIST
        overhead_per_epoch = (samples_per_epoch / batch_size) * (elapsed1 / n_iters)
        print(f"  Estimated overhead per epoch: {overhead_per_epoch:.3f} sec")
        print()
        print("  INTERPRETATION:")
        print("  - If overhead > 50ms/batch: data loading is a bottleneck")
        print("  - If overhead < 1ms/batch: data loading is efficient")

    except Exception as e:
        print(f"  Error running data loading test: {e}")

    # ========================================================================
    # Section 7: Recommendations
    # ========================================================================
    print_section("7. Optimization Recommendations")

    try:
        import jax
        # Parse GPU memory from nvidia-smi
        gpu_info_raw = run_command("nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits", "")
        if gpu_info_raw and not gpu_info_raw.startswith("Error"):
            gpu_line = gpu_info_raw.split('\n')[0]  # First GPU
            mem_total, mem_used, mem_free = map(int, gpu_line.split(','))

            mem_util_pct = (mem_used / mem_total) * 100

            print(f"  Current GPU memory utilization: {mem_util_pct:.1f}%")
            print(f"  Current memory used: {mem_used} MB / {mem_total} MB")
            print()

            if mem_util_pct < 10:
                print("  ⚠️  CRITICAL: GPU memory usage < 10%")
                print("  → Batch size is WAY too small for this GPU")
                print(f"  → Current batch size: 1024 (estimated ~3 MB)")
                print(f"  → Available memory: {mem_free} MB")
                print(f"  → Could increase batch size by ~{mem_free // 3}x")
                print()
                print("  RECOMMENDATION:")
                print("  1. Increase batch_size in configs from 1024 to 8192-16384")
                print("  2. Monitor GPU utilization with: watch -n 0.5 nvidia-smi")
                print("  3. Expected result: 30-60% GPU utilization, 300-500W power")
            elif mem_util_pct < 50:
                print("  ℹ️  GPU memory usage < 50%")
                print("  → Batch size can be increased further")
                print(f"  → Recommended: double current batch size")
            else:
                print("  ✓ GPU memory usage is healthy (>50%)")

        # Check for missing XLA flags
        xla_flags = os.environ.get('XLA_FLAGS', '')
        if not xla_flags or 'triton' not in xla_flags:
            print()
            print("  ⚠️  XLA optimization flags not set")
            print("  → Add to your run scripts:")
            print('  export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_enable_latency_hiding_scheduler=true"')

    except Exception as e:
        print(f"  Error generating recommendations: {e}")

    # ========================================================================
    # Section 8: Summary
    # ========================================================================
    print_section("8. Summary & Next Steps")

    print("""
  Please share this entire output to diagnose the GPU utilization issue.

  To monitor GPU during training, run in separate terminal:
    watch -n 0.5 nvidia-smi

  Or for detailed logging:
    nvidia-smi dmon -s pucvmet -i 0 -o DT > gpu_monitor.log
    """)

    print("\nDiagnostic complete!")
    print("="*70)

if __name__ == '__main__':
    main()
