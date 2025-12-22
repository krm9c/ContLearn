#!/usr/bin/env python
"""
Pre-download all datasets to avoid parallel download conflicts.
Run this once before launching parallel jobs.

Usage:
    python kkt_run/download_datasets.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torchvision
from torchvision import transforms

print("=" * 60)
print("Pre-downloading datasets to avoid parallel conflicts")
print("=" * 60)
print()

# Create data directory
os.makedirs('./data', exist_ok=True)

# Download MNIST
print("Downloading MNIST...")
my_transforms = transforms.Compose([transforms.ToTensor()])
torchvision.datasets.MNIST('./data', train=True, download=True, transform=my_transforms)
print("✓ MNIST downloaded")
print()

# Download CIFAR-10
print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=my_transforms)
print("✓ CIFAR-10 downloaded")
print()

# Download CIFAR-100
print("Downloading CIFAR-100...")
torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=my_transforms)
print("✓ CIFAR-100 downloaded")
print()

print("=" * 60)
print("All datasets downloaded successfully!")
print("=" * 60)
