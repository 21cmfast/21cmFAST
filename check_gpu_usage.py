#!/usr/bin/env python
"""
Script to verify GPU usage in 21cmFAST and monitor GPU activity.

This script:
1. Checks if CUDA is compiled into the binary
2. Monitors GPU usage during a test calculation
3. Provides ways to verify GPU acceleration
"""

import os
import subprocess
import sys
import threading
import time

# Fix the ninja path issue first
try:
    import _21cmfast_editable_loader

    for item in sys.meta_path:
        if isinstance(item, _21cmfast_editable_loader.MesonpyMetaFinder):
            item._build_cmd = ["ninja"]
            break
except ImportError:
    pass

# Set CUDA library path
os.environ["LD_LIBRARY_PATH"] = (
    f"/apps/modules/software/CUDA/12.8.0/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
)


def check_cuda_compilation():
    """Check if the binary was compiled with CUDA support."""
    print("=" * 60)
    print("Checking CUDA compilation...")
    print("=" * 60)

    # Check if the compiled library has CUDA symbols
    so_file = (
        "build/cp311/src/py21cmfast/src/c_21cmfast.cpython-311-x86_64-linux-gnu.so"
    )
    if os.path.exists(so_file):
        try:
            result = subprocess.run(["nm", so_file], capture_output=True, text=True)
            cuda_symbols = [
                line
                for line in result.stdout.split("\n")
                if "cuda" in line.lower() or "gpu" in line.lower()
            ]

            if cuda_symbols:
                print(f"✓ Found {len(cuda_symbols)} CUDA-related symbols in the binary")
                print("  Sample CUDA symbols found:")
                for symbol in cuda_symbols[:5]:
                    print(f"    {symbol.strip()}")
                return True
            else:
                print("✗ No CUDA symbols found in binary")
                return False
        except Exception as e:
            print(f"  Error checking symbols: {e}")
    else:
        print(f"  Binary not found at {so_file}")

    return False


def check_cuda_runtime():
    """Check if CUDA runtime is available."""
    print("\n" + "=" * 60)
    print("Checking CUDA runtime...")
    print("=" * 60)

    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU(s) detected:")
            print("  " + result.stdout.strip().replace("\n", "\n  "))
            return True
        else:
            print("✗ No NVIDIA GPUs detected or nvidia-smi not available")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found - CUDA runtime may not be installed")
        return False


def monitor_gpu(stop_event, gpu_used):
    """Monitor GPU usage in a separate thread."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                util, mem = result.stdout.strip().split(", ")
                if (
                    int(util) > 5 or int(mem) > 100
                ):  # GPU utilization > 5% or memory > 100MB
                    gpu_used[0] = True
                    gpu_used[1] = max(gpu_used[1], int(util))
                    gpu_used[2] = max(gpu_used[2], int(mem))
        except:
            pass
        time.sleep(0.1)


def test_gpu_usage():
    """Run a small test to see if GPU is being used."""
    print("\n" + "=" * 60)
    print("Testing GPU usage with 21cmFAST...")
    print("=" * 60)

    try:
        import py21cmfast as p21c

        print(f"✓ Imported py21cmfast version {p21c.__version__}")

        # Check the compiled library for USE_CUDA flag
        print("\nChecking USE_CUDA compilation flag...")
        lib_path = p21c.c_21cmfast.__file__ if hasattr(p21c, "c_21cmfast") else None
        if lib_path:
            result = subprocess.run(
                ["strings", lib_path, "|", "grep", "-i", "cuda"],
                shell=True,
                capture_output=True,
                text=True,
            )
            if "cuda" in result.stdout.lower():
                print("✓ CUDA strings found in compiled library")
            else:
                print("✗ No CUDA strings found in compiled library")

        # Run a small calculation while monitoring GPU
        print("\nRunning a small test calculation...")
        print("Monitoring GPU usage during calculation...")

        # Start GPU monitoring
        stop_event = threading.Event()
        gpu_used = [False, 0, 0]  # [was_used, max_utilization%, max_memory_mb]
        monitor_thread = threading.Thread(
            target=monitor_gpu, args=(stop_event, gpu_used)
        )
        monitor_thread.start()

        try:
            # Run a small ionization calculation
            initial_conditions = p21c.initial_conditions(
                user_params={"HII_DIM": 32, "BOX_LEN": 50}
            )

            # This should trigger GPU usage if available
            ionize_box = p21c.ionize_box(
                initial_conditions=initial_conditions, redshift=8.0
            )

            time.sleep(1)  # Give GPU monitor a chance to catch any activity

        finally:
            stop_event.set()
            monitor_thread.join()

        if gpu_used[0]:
            print("✓ GPU was used during calculation!")
            print(f"  Max GPU utilization: {gpu_used[1]}%")
            print(f"  Max GPU memory used: {gpu_used[2]} MB")
        else:
            print("✗ No GPU usage detected during calculation")
            print("  Note: The code may be running on CPU due to:")
            print("  - use_cuda flags being hardcoded to false in the C code")
            print("  - Compilation without CUDA support")
            print("  - Missing CUDA runtime libraries")

        return gpu_used[0]

    except ImportError as e:
        print(f"✗ Failed to import py21cmfast: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_code_flags():
    """Check the source code for GPU flags."""
    print("\n" + "=" * 60)
    print("Checking source code GPU flags...")
    print("=" * 60)

    # Check if use_cuda is hardcoded to false
    ionbox_file = "src/py21cmfast/src/IonisationBox.c"
    filter_file = "src/py21cmfast/src/filtering.c"

    files_to_check = [ionbox_file, filter_file]

    for file in files_to_check:
        if os.path.exists(file):
            with open(file) as f:
                content = f.read()
                if "use_cuda = false" in content:
                    print(f"✗ Found 'use_cuda = false' hardcoded in {file}")
                    # Count occurrences
                    count = content.count("use_cuda = false")
                    print(f"  Found {count} occurrences")
                elif "use_cuda = true" in content:
                    print(f"✓ Found 'use_cuda = true' in {file}")
                else:
                    print(f"  No use_cuda assignments found in {file}")


if __name__ == "__main__":
    print("21cmFAST GPU Usage Verification Script")
    print("=" * 60)

    # Run all checks
    cuda_compiled = check_cuda_compilation()
    cuda_runtime = check_cuda_runtime()
    check_code_flags()
    gpu_used = test_gpu_usage()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if cuda_compiled and cuda_runtime:
        if gpu_used:
            print("✓ GPU support is working!")
        else:
            print("⚠ GPU support is compiled in but not being used")
            print("\nTo enable GPU usage, you need to:")
            print("1. Edit the C source files to set 'use_cuda = true'")
            print("2. Recompile the package")
            print("\nFiles to edit:")
            print("  - src/py21cmfast/src/IonisationBox.c (lines ~1389, ~1552, ~1599)")
            print("  - src/py21cmfast/src/filtering.c (lines ~206, ~261)")
            print("\nChange: bool use_cuda = false;")
            print("To:     bool use_cuda = true;")
    elif cuda_compiled and not cuda_runtime:
        print("⚠ CUDA compiled but runtime not available")
        print("  Make sure you're running on a node with GPUs")
    else:
        print("✗ CUDA support not fully available")
