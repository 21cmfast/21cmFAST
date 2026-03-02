#!/usr/bin/env python
"""
Wrapper to import py21cmfast with proper environment setup.

This script handles the ninja path issue and CUDA library loading.
"""

import os
import sys

# Ensure CUDA libraries are available
cuda_lib_path = "/apps/modules/software/CUDA/12.8.0/lib64"
if cuda_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

# Fix the ninja path in the editable loader
try:
    import _21cmfast_editable_loader

    for item in sys.meta_path:
        if isinstance(item, _21cmfast_editable_loader.MesonpyMetaFinder):
            # Update to use ninja from current environment
            item._build_cmd = ["ninja"]
            break
except ImportError:
    pass  # Editable loader not installed

# Import py21cmfast
import py21cmfast as p21c

print(f"Successfully imported py21cmfast version {p21c.__version__}")
print("You can now use p21c in your scripts")

# Make it available globally
__all__ = ["p21c"]
