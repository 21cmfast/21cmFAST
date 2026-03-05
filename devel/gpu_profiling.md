# GPU Profiling Guide for 21cmFAST

This guide documents tools and techniques for profiling GPU performance in 21cmFAST.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (module load cuda/12.8.0 on OzStar)
- GPU build: `PY21C_USE_CUDA=TRUE pip install -e . --no-deps`

## NVIDIA Nsight Systems

Nsight Systems provides system-wide performance analysis, including CPU-GPU interactions.

### Basic Usage

```bash
# Profile a script
nsys profile --stats=true python script/batchC/simple-coeval-small.py

# Save to file for GUI analysis
nsys profile -o nsys_profile python script/batchC/simple-coeval-small.py
# Opens with: nsys-ui nsys_profile.qdrep
```

### Key Metrics to Capture

```bash
# Capture CUDA API calls and GPU kernels
nsys profile --trace=cuda,nvtx python script.py

# Capture memory operations
nsys profile --cuda-memory-usage=true python script.py

# Capture with sampling
nsys profile --sample=cpu python script.py
```

### Analysis Focus Areas

1. **GPU Kernel Time** - Total time spent in kernels
2. **Host-Device Transfers** - memcpy times (minimize these)
3. **Kernel Launch Overhead** - Time between kernel launches
4. **GPU Idle Time** - Periods where GPU is waiting

## NVIDIA Nsight Compute

Nsight Compute provides detailed kernel-level analysis.

**Note:** Requires V100 or newer GPU (not P100).

### Basic Usage

```bash
# Profile all kernels
ncu python script/batchC/simple-coeval-small.py

# Profile specific kernel
ncu --kernel-name "perturb_density_field_kernel" python script.py

# Full metrics set
ncu --set full python script.py
```

### Key Metrics

```bash
# Memory throughput
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python script.py

# Occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python script.py

# Memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed python script.py
```

## compute-sanitizer

Memory error detection for CUDA code.

### Memory Check

```bash
# Check for memory errors
compute-sanitizer --tool memcheck python script/batchC/simple-coeval-small.py
```

### Race Detection

```bash
# Check for race conditions
compute-sanitizer --tool racecheck python script.py
```

### Initialization Check

```bash
# Check for uninitialized memory
compute-sanitizer --tool initcheck python script.py
```

## cuda-gdb

CUDA-aware debugger for kernel debugging.

### Build with Debug Symbols

```bash
PY21C_CUDA_DEBUG=TRUE pip install -e . --no-deps
```

### Basic Usage

```bash
# Start debugger
cuda-gdb --args python script/batchC/simple-coeval-small.py

# Common commands:
# run          - start execution
# cuda thread  - switch to CUDA thread
# cuda block   - show current block
# info cuda kernels - list running kernels
# break file.cu:line - set breakpoint
```

## nvprof (Legacy)

For older CUDA versions or simpler analysis:

```bash
# Basic profiling
nvprof python script.py

# Detailed timeline
nvprof --print-gpu-trace python script.py

# Memory operations
nvprof --print-gpu-summary python script.py
```

## Performance Optimization Checklist

### Host-Device Transfers

1. Minimize data transfers between CPU and GPU
2. Use pinned memory for frequent transfers
3. Overlap transfers with computation where possible

### Kernel Optimization

1. Ensure sufficient occupancy (>50%)
2. Minimize warp divergence
3. Use shared memory for frequently accessed data
4. Coalesce global memory accesses

### Memory Patterns

1. Check for uncoalesced memory access
2. Monitor L2 cache hit rate
3. Use texture memory for read-only data with spatial locality

## Example: Full GPU Profiling Session

```bash
# 1. Load CUDA module
module load cuda/12.8.0

# 2. Build GPU version with debug symbols
cd install/21cmFAST/gpu-build-v4
PY21C_CUDA_DEBUG=TRUE PY21C_USE_CUDA=TRUE pip install -e . --no-deps

# 3. Check for memory errors first
compute-sanitizer --tool memcheck python script/batchC/simple-coeval-small.py

# 4. Profile with Nsight Systems
nsys profile --stats=true -o gpu_profile python script/batchC/simple-coeval-small.py

# 5. Detailed kernel analysis (on V100/A100)
ncu --set full --kernel-name "perturb_density_field_kernel" python script.py

# 6. Rebuild for production
PY21C_USE_CUDA=TRUE pip install -e . --no-deps
```

## OzStar-Specific Notes

### GPU Availability

- P100 GPUs: skylake-gpu partition (12GB memory)
- A100 GPUs: milan partition (40-80GB memory)

### Interactive GPU Session

```bash
# Request interactive GPU node
srun --partition=skylake-gpu --gres=gpu:1 --time=1:00:00 --pty bash

# On Milan (A100)
srun --partition=milan --gres=gpu:1 --time=1:00:00 --pty bash
```

### Batch Job Profiling

```bash
# Add to job script
#SBATCH --gres=gpu:1
#SBATCH --partition=skylake-gpu

module load cuda/12.8.0
nsys profile --stats=true python script.py
```

## Common Issues

### cuFFT INTERNAL_ERROR

- Check CUDA context state
- Ensure sufficient GPU memory
- Try explicit cuFFT workspace allocation

### Out of Memory

- Monitor with `nvidia-smi`
- Check `PY21C_CACHE` location
- Reduce box size for testing

### Low Occupancy

- Increase block size
- Reduce register usage per thread
- Use shared memory instead of local arrays

## Batch Profiling Scripts

For OzStar SLURM batch jobs, use these dedicated profiling scripts:

### Nsight Systems (all GPUs)

```bash
# Profile on skylake-gpu (P100)
sbatch --export=PY21C_SCRIPT=script/batchC/simple-coeval-tiny.py,PY21C_VENV=venv \
       script/batch-nsys.sh

# Profile on milan (A100)
sbatch --partition=milan --export=PY21C_SCRIPT=script/batchC/simple-coeval-tiny.py,PY21C_VENV=venv.milan \
       script/batch-nsys.sh
```

### Nsight Compute (A100 only)

```bash
# Note: Built-in --partition=milan, as P100 not supported
sbatch --export=PY21C_SCRIPT=script/batchC/simple-coeval-tiny.py,PY21C_VENV=venv.milan,NCU_SET=full \
       script/batch-ncu.sh
```

### nvprof (P100 preferred, legacy)

```bash
# For P100 kernel profiling where ncu doesn't work
sbatch --export=PY21C_SCRIPT=script/batchC/simple-coeval-tiny.py,PY21C_VENV=venv \
       script/batch-nvprof.sh
```

### compute-sanitizer (all GPUs)

```bash
# Memory error checking - works on P100 and A100
sbatch --export=PY21C_SCRIPT=script/batchC/simple-coeval-tiny.py,PY21C_VENV=venv,SANITIZER_TOOL=memcheck \
       script/batch-sanitizer.sh
```

## Tool Compatibility Matrix

| Tool              | P100 (skylake-gpu) | A100 (milan) | Output Format | Use Case                       |
|-------------------|:------------------:|:------------:|---------------|--------------------------------|
| nsys              | ✅                 | ✅           | .nsys-rep     | System-wide CPU-GPU profiling  |
| ncu               | ❌                 | ✅           | .ncu-rep      | Kernel-level metrics           |
| nvprof            | ✅                 | ✅           | .nvvp         | Legacy kernel profiling        |
| compute-sanitizer | ✅                 | ✅           | text log      | CUDA memory error detection    |

**Notes:**
- ncu requires Volta (V100) or newer; P100 (Pascal) not supported
- nvprof is deprecated but functional; prefer ncu on A100
- All tools available in CUDA modules (cuda/12.8.0)

## See Also

- `devel/cpu_profiling.md` - CPU profiling guide
- `script/gpu.sh` - GPU job runner
- `script/batch-nsys.sh` - Batch nsys profiling
- `script/batch-ncu.sh` - Batch ncu profiling (A100 only)
- `script/batch-nvprof.sh` - Batch nvprof profiling
- `script/batch-sanitizer.sh` - Batch memory checking
- NVIDIA Nsight documentation: https://developer.nvidia.com/nsight-systems
