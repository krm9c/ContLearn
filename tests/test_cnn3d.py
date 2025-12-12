"""
Unit tests for CNN3D model components.
Tests model forward pass, AWB transformation, and data handling for 3-channel images.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import torch
import equinox as eqx

from utils.model import CNN3D
from utils.data import data_return, Continual_Dataset


def test_cnn3d_forward_pass():
    """Test CNN3D forward pass with 3x32x32 input (CIFAR-style)"""
    print("\n" + "="*60)
    print("TEST 1: CNN3D Forward Pass")
    print("="*60)

    key = jax.random.PRNGKey(42)
    filter_size = 3
    # Calculate expected flatten size: 32 -> conv(3) -> 30 -> pool -> 15 -> conv(3) -> 13 -> pool -> 6
    # Output channels = 64, so flatten = 6*6*64 = 2304
    feed_sizes = [2304, 512, 256, 10]

    model = CNN3D(key, filter_size=filter_size, feed_sizes=feed_sizes,
                  channel_in=3, channel_out=32, num_classes=10)

    # Create dummy 3x32x32 input (CIFAR-style)
    x = jax.random.normal(key, (3, 32, 32))

    try:
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: (10,)")
        assert output.shape == (10,), f"Expected (10,), got {output.shape}"
        print("  PASSED: Forward pass works correctly")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_cnn3d_awbt_forward():
    """Test CNN3D AWB transformation forward pass"""
    print("\n" + "="*60)
    print("TEST 2: CNN3D get_AWBT() Forward Pass")
    print("="*60)

    key = jax.random.PRNGKey(42)
    filter_size = 3
    feed_sizes = [2304, 512, 256, 10]

    model = CNN3D(key, filter_size=filter_size, feed_sizes=feed_sizes,
                  channel_in=3, channel_out=32, num_classes=10)

    x = jax.random.normal(key, (3, 32, 32))

    try:
        output = model.get_AWBT(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        # AWB may change output dimensions based on A_feed/B_feed initialization
        print(f"  Output sample values: {output[:5]}")
        print("  PASSED: get_AWBT() works correctly")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn3d_calc_output_size():
    """Test CNN3D output size calculation"""
    print("\n" + "="*60)
    print("TEST 3: CNN3D calc_output_size()")
    print("="*60)

    key = jax.random.PRNGKey(42)
    model = CNN3D(key, filter_size=3, feed_sizes=[2304, 512, 256, 10],
                  channel_in=3, channel_out=32, num_classes=10)

    # Test calculation: 32x32 input with filter_size=3
    # After conv: (32 - 3 + 1) = 30
    # After pool: 30 // 2 = 15
    result1 = model.calc_output_size(32, 3)
    expected1 = 15

    # Second layer: 15 input with filter_size=3
    # After conv: (15 - 3 + 1) = 13
    # After pool: 13 // 2 = 6
    result2 = model.calc_output_size(15, 3)
    expected2 = 6

    print(f"  Layer 1: Input 32, filter 3 -> Expected {expected1}, Got {result1}")
    print(f"  Layer 2: Input 15, filter 3 -> Expected {expected2}, Got {result2}")

    if result1 == expected1 and result2 == expected2:
        print("  PASSED: calc_output_size() works correctly")
        return True
    else:
        print("  FAILED: calc_output_size() returned incorrect values")
        return False


def test_cnn3d_awb_matrix_shapes():
    """Test that AWB matrices have correct shapes"""
    print("\n" + "="*60)
    print("TEST 4: CNN3D AWB Matrix Shapes")
    print("="*60)

    key = jax.random.PRNGKey(42)
    filter_size = 3
    channel_in = 3
    channel_out = 32
    new_filter_size = filter_size + 2  # As per init

    model = CNN3D(key, filter_size=filter_size, feed_sizes=[2304, 512, 256, 10],
                  channel_in=channel_in, channel_out=channel_out, num_classes=10)

    passed = True

    # A_conv1: [channel_out][channel_in] each of shape (new_filter_size, filter_size)
    print(f"  A_conv1: {len(model.A_conv1)} x {len(model.A_conv1[0])} matrices")
    if len(model.A_conv1) != channel_out:
        print(f"    FAILED: Expected {channel_out} output channels, got {len(model.A_conv1)}")
        passed = False
    if len(model.A_conv1[0]) != channel_in:
        print(f"    FAILED: Expected {channel_in} input channels, got {len(model.A_conv1[0])}")
        passed = False
    if model.A_conv1[0][0].shape != (new_filter_size, filter_size):
        print(f"    FAILED: Expected shape {(new_filter_size, filter_size)}, got {model.A_conv1[0][0].shape}")
        passed = False
    else:
        print(f"    A_conv1[0][0] shape: {model.A_conv1[0][0].shape} - OK")

    # A_conv2: [channel_out*2][channel_out] each of shape (new_filter_size, filter_size)
    print(f"  A_conv2: {len(model.A_conv2)} x {len(model.A_conv2[0])} matrices")
    if len(model.A_conv2) != channel_out * 2:
        print(f"    FAILED: Expected {channel_out * 2} output channels, got {len(model.A_conv2)}")
        passed = False
    if len(model.A_conv2[0]) != channel_out:
        print(f"    FAILED: Expected {channel_out} input channels, got {len(model.A_conv2[0])}")
        passed = False
    if model.A_conv2[0][0].shape != (new_filter_size, filter_size):
        print(f"    FAILED: Expected shape {(new_filter_size, filter_size)}, got {model.A_conv2[0][0].shape}")
        passed = False
    else:
        print(f"    A_conv2[0][0] shape: {model.A_conv2[0][0].shape} - OK")

    if passed:
        print("  PASSED: All AWB matrix shapes are correct")
    return passed


def test_data_3channel_handling():
    """Test that 3-channel images are handled correctly in data.py"""
    print("\n" + "="*60)
    print("TEST 5: 3-Channel Image Handling in data.py")
    print("="*60)

    config = {
        'data_id': 'cifar10',
        'len_exp_replay': 1000,
        'batch_size': 32,
        'problem': 'classification',
        'network': 'cnn'
    }

    try:
        data = data_return(config)

        # Generate first task
        data.generate_dataset(task_id=0, batch_size=32, phase='training')

        print(f"  X_train shape before append: {data.X_train.shape}")
        print(f"  Expected: [N, 3, 32, 32] (3 channels)")

        # Append to experience
        data.append_to_experience(task_id=0)

        print(f"  exp_x_train shape after append: {data.exp_x_train.shape}")

        # Check that 3-channel images don't get an extra dimension
        if len(data.exp_x_train.shape) == 4 and data.exp_x_train.shape[1] == 3:
            print("  PASSED: 3-channel images handled correctly (no extra unsqueeze)")
            return True
        else:
            print(f"  FAILED: Expected shape [N, 3, 32, 32], got {data.exp_x_train.shape}")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_1channel_handling():
    """Test that 1-channel images still get unsqueezed correctly"""
    print("\n" + "="*60)
    print("TEST 6: 1-Channel Image Handling in data.py (MNIST)")
    print("="*60)

    config = {
        'data_id': 'mnist',
        'len_exp_replay': 1000,
        'batch_size': 32,
        'problem': 'classification',
        'network': 'cnn'
    }

    try:
        data = data_return(config)

        # Generate first task
        data.generate_dataset(task_id=0, batch_size=32, phase='training')

        print(f"  X_train shape before append: {data.X_train.shape}")

        # Append to experience
        data.append_to_experience(task_id=0)

        print(f"  exp_x_train shape after append: {data.exp_x_train.shape}")

        # Check that 1-channel images get proper channel dimension [N, 1, 28, 28]
        if len(data.exp_x_train.shape) == 4 and data.exp_x_train.shape[1] == 1:
            print("  PASSED: 1-channel images handled correctly (unsqueeze applied)")
            return True
        else:
            print(f"  FAILED: Expected shape [N, 1, 28, 28], got {data.exp_x_train.shape}")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continual_dataset_cifar():
    """Test Continual_Dataset with CIFAR data"""
    print("\n" + "="*60)
    print("TEST 7: Continual_Dataset with CIFAR")
    print("="*60)

    config = {
        'problem': 'classification',
        'network': 'cnn'
    }

    # Create dummy CIFAR-style data
    x = torch.randn(100, 3, 32, 32)
    y = np.random.randint(0, 10, 100)

    try:
        dataset = Continual_Dataset(config, data_x=x, data_y=y)
        sample_x, sample_y = dataset[0]

        print(f"  Dataset length: {len(dataset)}")
        print(f"  Sample x shape: {sample_x.shape}")
        print(f"  Sample y: {sample_y}")

        if sample_x.shape == (3, 32, 32):
            print("  PASSED: Continual_Dataset works with CIFAR data")
            return True
        else:
            print(f"  FAILED: Expected sample shape (3, 32, 32), got {sample_x.shape}")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*60)
    print("RUNNING ALL CNN3D UNIT TESTS")
    print("="*60)

    results = {
        "CNN3D Forward Pass": test_cnn3d_forward_pass(),
        "CNN3D get_AWBT()": test_cnn3d_awbt_forward(),
        "CNN3D calc_output_size()": test_cnn3d_calc_output_size(),
        "CNN3D AWB Matrix Shapes": test_cnn3d_awb_matrix_shapes(),
        "3-Channel Image Handling": test_data_3channel_handling(),
        "1-Channel Image Handling": test_data_1channel_handling(),
        "Continual_Dataset CIFAR": test_continual_dataset_cifar(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  {total - passed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
