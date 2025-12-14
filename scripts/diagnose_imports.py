#!/usr/bin/env python
"""
Diagnostic script to identify import issues.
Run this on the remote computer to diagnose the problem.

Usage:
    python scripts/diagnose_imports.py
"""

import os
import sys
from pathlib import Path

print("="*70)
print("IMPORT DIAGNOSTICS")
print("="*70)

# 1. Check current directory
print(f"\n1. Current directory: {os.getcwd()}")

# 2. Check if we're in the right place
project_root = Path(__file__).parent.parent
print(f"2. Project root: {project_root}")
print(f"   Exists: {project_root.exists()}")

# 3. Check if src directory exists
src_dir = project_root / 'src'
print(f"\n3. Source directory: {src_dir}")
print(f"   Exists: {src_dir.exists()}")

# 4. Check if cl package exists
cl_dir = src_dir / 'cl'
print(f"\n4. CL package directory: {cl_dir}")
print(f"   Exists: {cl_dir.exists()}")

# 5. Check critical __init__.py files
critical_files = [
    'src/cl/__init__.py',
    'src/cl/models/__init__.py',
    'src/cl/models/layers.py',
    'src/cl/datasets/__init__.py',
    'src/cl/runners/__init__.py',
]

print(f"\n5. Critical files:")
for file_path in critical_files:
    full_path = project_root / file_path
    exists = full_path.exists()
    size = full_path.stat().st_size if exists else 0
    print(f"   {'✓' if exists else '✗'} {file_path} ({size} bytes)")

# 6. Check Python path
print(f"\n6. Python sys.path:")
for p in sys.path:
    print(f"   - {p}")

# 7. Try adding src to path
sys.path.insert(0, str(src_dir))
print(f"\n7. Added to sys.path: {src_dir}")

# 8. Try importing cl
print(f"\n8. Testing imports...")
try:
    import cl
    print(f"   ✓ import cl - SUCCESS")
    print(f"     cl.__file__ = {cl.__file__}")
except Exception as e:
    print(f"   ✗ import cl - FAILED: {e}")
    sys.exit(1)

# 9. Try importing cl.models
try:
    import cl.models
    print(f"   ✓ import cl.models - SUCCESS")
except Exception as e:
    print(f"   ✗ import cl.models - FAILED: {e}")
    print(f"     Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 10. Try importing cl.models.layers directly
try:
    import cl.models.layers
    print(f"   ✓ import cl.models.layers - SUCCESS")
except Exception as e:
    print(f"   ✗ import cl.models.layers - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 11. Try importing from cl.models.layers
try:
    from cl.models.layers import Linear
    print(f"   ✓ from cl.models.layers import Linear - SUCCESS")
except Exception as e:
    print(f"   ✗ from cl.models.layers import Linear - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 12. Try importing MLP
try:
    from cl.models.mlp import MLP
    print(f"   ✓ from cl.models.mlp import MLP - SUCCESS")
except Exception as e:
    print(f"   ✗ from cl.models.mlp import MLP - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 13. Try the full import chain that's failing
try:
    from cl.runners import train_model
    print(f"   ✓ from cl.runners import train_model - SUCCESS")
except Exception as e:
    print(f"   ✗ from cl.runners import train_model - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nIf you're still having issues with run.py, try:")
print("  1. Clean Python cache: find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null")
print("  2. Reinstall package: pip install -e .")
print("="*70)
