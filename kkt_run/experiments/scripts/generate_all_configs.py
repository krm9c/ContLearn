#!/usr/bin/env python
"""
Generate all experimental configs for validation experiments.
Creates 24 config files: 6 datasets × 4 conditions

Updated Dec 25, 2024: Optimized for GPU utilization
- batch_size=1024 for all datasets (8x improvement)
- epochs_per_task=250 for all except sine (stays 500)
- Outputs to shared kkt_run/configs/ folder
"""

import json
import os
from pathlib import Path

# Base directory - now outputs to shared kkt_run/configs/
SCRIPT_DIR = Path(__file__).parent
KKT_RUN_DIR = SCRIPT_DIR.parent.parent  # kkt_run/
CONFIGS_DIR = KKT_RUN_DIR / "configs"   # Shared configs folder

# Dataset specifications (GPU-optimized)
DATASETS = {
    "sine": {
        "data": "sine",
        "n_task": 10,
        "epochs_per_task": 500,  # Keep 500 for sine
        "batch_size": 1024,      # Optimized for GPU
    },
    "mnist": {
        "data": "mnist",
        "n_task": 10,
        "epochs_per_task": 250,  # Reduced from 500
        "batch_size": 1024,      # Increased from 256
    },
    "permuted_mnist": {
        "data": "permuted_mnist",
        "n_task": 10,
        "epochs_per_task": 250,  # Reduced from 500
        "batch_size": 1024,      # Increased from 256
    },
    "cifar10": {
        "data": "cifar10",
        "n_task": 10,
        "epochs_per_task": 250,  # Reduced from 500
        "batch_size": 1024,      # Increased from 128
    },
    "cifar100": {
        "data": "cifar100",
        "n_task": 20,
        "epochs_per_task": 250,  # Reduced from 500
        "batch_size": 1024,      # Increased from 128
    },
    "synthetic_graph": {
        "data": "synthetic",
        "n_task": 10,
        "epochs_per_task": 250,  # Reduced from 500
        "batch_size": 1024,      # Increased from 128
    },
}

# Condition templates
CONDITIONS = {
    "condition1_baseline": {
        "name": "CONDITION 1: Baseline Hamiltonian (No Smoothness)",
        "desc": "Fixed arch, constant LR, no warm start",
        "config": {
            "awb_enabled": False,
            "lr_schedule": "constant",
            "lr": 0.0001,
            "warmup_epochs": 0,
            "task_warmup_enabled": False,  # Disable task warmup for true baseline
        },
    },
    "condition2_heuristics": {
        "name": "CONDITION 2: Smoothness via Heuristics",
        "desc": "Fixed arch, cosine LR schedule, warm start",
        "config": {
            "awb_enabled": False,
            "lr_schedule": "cosine",
            "lr": 0.0001,
            "warmup_epochs": 50,
            "lr_warmup_factor": 0.1,
        },
    },
    "condition3_arch_no_transfer": {
        "name": "CONDITION 3: Architecture Change, No Transfer",
        "desc": "Arch search enabled, skip A/B training, random init after arch change",
        "config": {
            "awb_enabled": True,
            "awb_skip_transfer": True,
            "awb_preliminary_epochs": 50,
            "awb_loss_ratio_threshold": 1.1,
            "lr_schedule": "constant",
            "lr": 0.0001,
            "warmup_epochs": 0,
        },
    },
    "condition4_awb_full": {
        "name": "CONDITION 4: Full AWB (Architecture + Transfer)",
        "desc": "Arch search + A/B training for knowledge transfer",
        "config": {
            "awb_enabled": True,
            "awb_skip_transfer": False,
            "awb_preliminary_epochs": 50,
            "awb_ab_training_epochs": 100,
            "awb_loss_ratio_threshold": 1.1,
            "lr_schedule": "constant",
            "lr": 0.0001,
            "warmup_epochs": 0,
        },
    },
}

def generate_config(dataset_name, dataset_spec, condition_name, condition_spec):
    """Generate a single config file."""
    config = {
        "__comment": f"{condition_spec['name']}",
        "__comment_": condition_spec["desc"],
    }

    # Add dataset-specific settings
    config.update(dataset_spec)

    # Add condition-specific settings
    config.update(condition_spec["config"])

    # Add common settings
    config["model_path"] = f"kkt_run/results/{dataset_name}_{condition_name}"
    config["save_iter"] = 50
    config["grad_weights"] = [0.01, 0.98, 0.1]
    config["debug_mode"] = False
    config["per_task_eval_enabled"] = True

    return config

def main():
    """Generate all config files to shared kkt_run/configs/ folder."""
    print("=" * 70)
    print("Generating GPU-Optimized Experimental Configs")
    print("=" * 70)
    print(f"Output: {CONFIGS_DIR.relative_to(KKT_RUN_DIR.parent)}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Total configs: {len(DATASETS) * len(CONDITIONS)}")
    print()
    print("Optimizations:")
    print("  - batch_size=1024 (all datasets)")
    print("  - epochs_per_task=250 (except sine: 500)")
    print("  - Expected GPU utilization: 60-80% (up from 20-30%)")
    print()

    count = 0
    existing_count = 0

    for dataset_name, dataset_spec in DATASETS.items():
        print(f"Dataset: {dataset_name}")
        for condition_name, condition_spec in CONDITIONS.items():
            config = generate_config(dataset_name, dataset_spec, condition_name, condition_spec)

            # Flat config structure: dataset_condition.json
            config_filename = f"{dataset_name}_{condition_name}.json"
            config_path = CONFIGS_DIR / config_filename

            if config_path.exists():
                print(f"  [{count+1:2d}/24] {config_filename} (already exists, skipping)")
                existing_count += 1
            else:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"  [{count+1:2d}/24] {config_filename} (created)")

            count += 1

    print()
    print("=" * 70)
    if existing_count > 0:
        print(f"✓ {existing_count} configs already existed")
        print(f"✓ {count - existing_count} configs generated")
    else:
        print(f"✓ Generated {count} config files")
    print(f"  Location: {CONFIGS_DIR.relative_to(KKT_RUN_DIR.parent)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
