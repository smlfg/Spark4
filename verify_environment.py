#!/usr/bin/env python3
"""
Verify Spark4 Unsloth Training Environment
This script checks all dependencies and GPU availability.
"""

import sys
from typing import Dict, Tuple

def check_import(module_name: str, display_name: str = None) -> Tuple[bool, str]:
    """Try to import a module and return status."""
    display_name = display_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    """Run all environment checks."""
    print("=" * 70)
    print("Spark4 Unsloth Training Environment Verification")
    print("=" * 70)
    print()

    # Track overall status
    all_passed = True

    # Check Python version
    print("1. Python Version")
    print(f"   Version: {sys.version}")
    if sys.version_info >= (3, 9) and sys.version_info < (3, 14):
        print("   ✓ Python version is compatible")
    else:
        print("   ✗ Python version should be 3.9-3.13")
        all_passed = False
    print()

    # Check core dependencies
    print("2. Core Dependencies")
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('trl', 'TRL'),
        ('peft', 'PEFT'),
        ('datasets', 'Datasets'),
        ('bitsandbytes', 'bitsandbytes'),
        ('flash_attn', 'Flash Attention'),
        ('unsloth', 'Unsloth'),
    ]

    for module, display in dependencies:
        success, info = check_import(module, display)
        if success:
            print(f"   ✓ {display:20s} {info}")
        else:
            print(f"   ✗ {display:20s} FAILED: {info}")
            all_passed = False
    print()

    # Check PyTorch and CUDA
    print("3. PyTorch and CUDA")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"           Compute Capability: {props.major}.{props.minor}")
                print(f"           Total Memory: {props.total_memory / 1024**3:.2f} GB")

            # Test tensor creation on GPU
            try:
                x = torch.randn(3, 3).cuda()
                print("   ✓ Successfully created tensor on GPU")
            except Exception as e:
                print(f"   ✗ Failed to create tensor on GPU: {e}")
                all_passed = False
        else:
            print("   ✗ CUDA is not available")
            all_passed = False
    except Exception as e:
        print(f"   ✗ PyTorch check failed: {e}")
        all_passed = False
    print()

    # Check Flash Attention
    print("4. Flash Attention")
    try:
        import flash_attn
        print(f"   Flash Attention version: {flash_attn.__version__}")

        # Try to use flash attention
        from flash_attn import flash_attn_func
        import torch

        if torch.cuda.is_available():
            # Create dummy tensors
            batch_size, seqlen, num_heads, head_dim = 2, 128, 8, 64
            q = torch.randn(batch_size, seqlen, num_heads, head_dim,
                           dtype=torch.float16, device='cuda')
            k = torch.randn(batch_size, seqlen, num_heads, head_dim,
                           dtype=torch.float16, device='cuda')
            v = torch.randn(batch_size, seqlen, num_heads, head_dim,
                           dtype=torch.float16, device='cuda')

            try:
                out = flash_attn_func(q, k, v)
                print("   ✓ Flash Attention is working correctly")
            except Exception as e:
                print(f"   ✗ Flash Attention test failed: {e}")
                all_passed = False
        else:
            print("   ⚠ Skipped Flash Attention test (no CUDA)")
    except Exception as e:
        print(f"   ✗ Flash Attention check failed: {e}")
        all_passed = False
    print()

    # Check Unsloth
    print("5. Unsloth")
    try:
        import unsloth
        print("   Unsloth imported successfully")

        # Try to access FastLanguageModel
        from unsloth import FastLanguageModel
        print("   ✓ FastLanguageModel available")
    except Exception as e:
        print(f"   ✗ Unsloth check failed: {e}")
        all_passed = False
    print()

    # Check optional dependencies
    print("6. Optional Dependencies")
    optional_deps = [
        ('wandb', 'Weights & Biases'),
        ('tensorboard', 'TensorBoard'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
    ]

    for module, display in optional_deps:
        success, info = check_import(module, display)
        if success:
            print(f"   ✓ {display:20s} {info}")
        else:
            print(f"   ⚠ {display:20s} not available (optional)")
    print()

    # Final summary
    print("=" * 70)
    if all_passed:
        print("✓ All critical checks passed! Environment is ready for training.")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
