# GPU Usage Verification Guide for 21cmFAST

This guide explains how to verify and monitor GPU usage in 21cmFAST after enabling CUDA support.

## Quick Check: Is GPU Being Used?

### Method 1: Real-time GPU Monitoring
Monitor GPU usage while running 21cmFAST calculations:

```bash
# In Terminal 1: Start GPU monitoring
watch -n 0.5 nvidia-smi

# In Terminal 2: Run your 21cmFAST calculation
python your_21cmfast_script.py
```

When GPU is active, you should see:
- GPU utilization percentage increase (GPU-Util column)
- Memory usage increase (Memory-Usage column)
- Process name appear in the process list

### Method 2: Using the Verification Script

The `check_gpu_usage.py` script provides comprehensive GPU diagnostics:

```bash
# Run the verification script
python check_gpu_usage.py
```

The script will:
1. Check if CUDA symbols are compiled into the binary
2. Verify CUDA runtime and GPU availability
3. Check source code for GPU enable flags
4. Run a test calculation while monitoring GPU
5. Provide a summary with recommendations

## Understanding the Output

### Successful GPU Usage
```
✓ Found 280 CUDA-related symbols in the binary
✓ NVIDIA GPU(s) detected: Tesla P100-PCIE-12GB
✓ GPU was used during calculation!
  Max GPU utilization: 45%
  Max GPU memory used: 1024 MB
```

### GPU Not Being Used
```
✓ Found 280 CUDA-related symbols in the binary
✓ NVIDIA GPU(s) detected: Tesla P100-PCIE-12GB
✗ Found 'use_cuda = false' hardcoded in source files
✗ No GPU usage detected during calculation
```

## GPU Requirements

For GPU acceleration to work, you need:

### 1. Hardware Requirements
- NVIDIA GPU with CUDA support
- CUDA 10.0 or later installed

### 2. Code Configuration
GPU acceleration only activates with these specific settings:
```python
astro_params = {
    "USE_MASS_DEPENDENT_ZETA": True,  # Required
    "USE_MINI_HALOS": False,          # Must be False
}
matter_options = {
    "USE_HALO_FIELD": False,           # Must be False
}
```

### 3. Source Code Settings
The following files must have `use_cuda = true`:
- `src/py21cmfast/src/IonisationBox.c`
- `src/py21cmfast/src/filtering.c`

## Monitoring GPU Performance

### Using nvidia-smi
```bash
# Show current GPU usage
nvidia-smi

# Monitor continuously
watch -n 1 nvidia-smi

# Show only utilization
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv --loop=1
```

### Using the Script's Monitor Function
The verification script includes a GPU monitor that tracks peak usage:
```python
# From check_gpu_usage.py
def monitor_gpu(stop_event, gpu_used):
    """Monitor GPU usage in a separate thread."""
    # Tracks: [was_used, max_utilization%, max_memory_mb]
```

## Troubleshooting

### GPU Not Detected
```bash
# Check if CUDA is available
nvidia-smi -L

# Check CUDA version
nvcc --version

# Verify CUDA library path
echo $LD_LIBRARY_PATH | grep -o '/[^:]*cuda[^:]*'
```

### GPU Detected but Not Used

1. **Check source code flags:**
   ```bash
   grep "bool use_cuda" src/py21cmfast/src/*.c
   ```
   Should show `bool use_cuda = true;`

2. **Verify compilation:**
   ```bash
   nm build/cp311/src/py21cmfast/src/c_21cmfast.*.so | grep -i cuda | head -5
   ```
   Should show CUDA symbols

3. **Check runtime configuration:**
   ```python
   import py21cmfast as p21c

   # Ensure these settings for GPU:
   astro_params = p21c.AstroParams(
       USE_MASS_DEPENDENT_ZETA=True,
       USE_MINI_HALOS=False
   )
   matter_options = p21c.MatterOptions(
       USE_HALO_FIELD=False
   )
   ```

### Library Loading Issues
If you get `libcudart.so.12: cannot open shared object file`:
```bash
export LD_LIBRARY_PATH=/apps/modules/software/CUDA/12.8.0/lib64:$LD_LIBRARY_PATH
```

## Performance Expectations

When GPU is working correctly:
- Small boxes (HII_DIM=32-64): May not show significant GPU usage
- Medium boxes (HII_DIM=128-256): Should show 20-60% GPU utilization
- Large boxes (HII_DIM=512+): Should show 50-90% GPU utilization

Note: Initial runs may be slower due to:
- CUDA kernel compilation (cached after first run)
- Data transfer overhead between CPU and GPU
- FFTW wisdom generation

## Example Test

Here's a simple test to verify GPU usage:

```python
# Import with proper environment
exec(open('import_21cmfast.py').read())

# Run with GPU-compatible settings
user_params = p21c.UserParams(HII_DIM=128, BOX_LEN=100)
astro_params = p21c.AstroParams(
    USE_MASS_DEPENDENT_ZETA=True,
    USE_MINI_HALOS=False
)
matter_options = p21c.MatterOptions(USE_HALO_FIELD=False)

# This should use GPU if available
ic = p21c.initial_conditions(user_params=user_params)
ionize_box = p21c.ionize_box(
    initial_conditions=ic,
    redshift=8.0,
    astro_params=astro_params,
    matter_options=matter_options
)

print("Calculation complete - check nvidia-smi for GPU usage")
```

## Additional Resources

- [NVIDIA System Management Interface (nvidia-smi) Documentation](https://developer.nvidia.com/nvidia-system-management-interface)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [21cmFAST Documentation](https://21cmfast.readthedocs.io/)
