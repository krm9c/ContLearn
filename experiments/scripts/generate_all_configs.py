#!/usr/bin/env python
"""
Generate all experimental configs for validation experiments.
Creates 24 config files: 6 datasets × 4 conditions
"""

import json
import os
from pathlib import Path

# Base directory
EXPERIMENTS_DIR = Path(__file__).parent.parent
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

# Dataset specifications
DATASETS = {
    "sine": {
        "data": "sine",
        "n_task": 10,
        "epochs_per_task": 500,
        "batch_size": 128,
    },
    "mnist": {
        "data": "mnist",
        "n_task": 10,
        "epochs_per_task": 500,
        "batch_size": 256,
    },
    "permuted_mnist": {
        "data": "permuted_mnist",
        "n_task": 10,
        "epochs_per_task": 500,
        "batch_size": 256,
    },
    "cifar10": {
        "data": "cifar10",
        "n_task": 10,
        "epochs_per_task": 500,
        "batch_size": 128,
    },
    "cifar100": {
        "data": "cifar100",
        "n_task": 20,
        "epochs_per_task": 500,
        "batch_size": 128,
    },
    "synthetic_graph": {
        "data": "synthetic",
        "n_task": 10,
        "epochs_per_task": 500,
        "batch_size": 128,
    },
}

# Condition templates
CONDITIONS = {
    "condition1_baseline": {
        "name": "Baseline Hamiltonian (No Smoothness)",
        "desc": "Fixed arch, constant LR, no warm start",
        "config": {
            "awb_enabled": False,
            "lr_schedule": "constant",
            "lr": 1e-4,
            "warmup_epochs": 0,
        },
    },
    "condition2_heuristics": {
        "name": "Smoothness via Heuristics",
        "desc": "Fixed arch, cosine LR schedule, warm start",
        "config": {
            "awb_enabled": False,
            "lr_schedule": "cosine",
            "lr": 1e-4,
            "warmup_epochs": 50,
            "lr_warmup_factor": 0.1,
        },
    },
    "condition3_arch_no_transfer": {
        "name": "Architecture Change, No Transfer",
        "desc": "Arch search enabled, skip A/B training, random init after arch change",
        "config": {
            "awb_enabled": True,
            "awb_skip_transfer": True,
            "awb_preliminary_epochs": 50,
            "awb_loss_ratio_threshold": 1.1,
            "lr_schedule": "constant",
            "lr": 1e-4,
            "warmup_epochs": 0,
        },
    },
    "condition4_awb_full": {
        "name": "Full AWB (Architecture + Transfer)",
        "desc": "Arch search + A/B training for knowledge transfer",
        "config": {
            "awb_enabled": True,
            "awb_skip_transfer": False,
            "awb_preliminary_epochs": 50,
            "awb_ab_training_epochs": 100,
            "awb_loss_ratio_threshold": 1.1,
            "lr_schedule": "constant",
            "lr": 1e-4,
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
    config["model_path"] = f"experiments/results/{dataset_name}/{condition_name}"
    config["save_iter"] = 50
    config["grad_weights"] = [0.01, 0.98, 0.1]
    config["debug_mode"] = False
    config["experiment_name"] = f"{dataset_name}_{condition_name}"
    # Added by Claude: Enable per-task evaluation for CL metrics (ACC, BWT, Forgetting, FWT)
    config["per_task_eval_enabled"] = True

    return config

def main():
    """Generate all config files."""
    print("Generating validation experiment configs...")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Total configs: {len(DATASETS) * len(CONDITIONS)}")
    print()

    count = 0
    for dataset_name, dataset_spec in DATASETS.items():
        dataset_dir = CONFIGS_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for condition_name, condition_spec in CONDITIONS.items():
            config = generate_config(dataset_name, dataset_spec, condition_name, condition_spec)

            config_path = dataset_dir / f"{condition_name}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            count += 1
            print(f"  [{count:2d}/24] Created: {config_path.relative_to(EXPERIMENTS_DIR)}")

    print()
    print(f"✓ Generated {count} config files")
    print(f"  Location: {CONFIGS_DIR.relative_to(EXPERIMENTS_DIR.parent)}")

if __name__ == "__main__":
    main()
