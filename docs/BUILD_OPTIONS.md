# Build Environment Variables

This document describes the environment variables that control the build configuration of 21cmFAST.

## Summary

| Variable           | Values                    | Default   | Description                            |
|--------------------|---------------------------|-----------|----------------------------------------|
| `USE_CUDA`         | `TRUE` / `FALSE`          | `FALSE`   | Enable GPU build (legacy)              |
| `PY21C_USE_CUDA`   | `TRUE` / `FALSE`          | `FALSE`   | Enable GPU build                       |
| `PY21C_OPT_LEVEL`  | `debug`/`release`/`fast`  | `release` | Optimization level                     |
| `PY21C_DEBUG`      | `TRUE`                    | (unset)   | Force debug mode                       |
| `PY21C_CUDA_DEBUG` | `TRUE`                    | (unset)   | CUDA device debugging                  |
| `PY21C_PGO_PHASE`  | `generate` / `use`        | (unset)   | Profile-guided optimization            |

## Detailed Description

### PY21C_USE_CUDA / USE_CUDA

Controls whether CUDA GPU support is enabled during compilation.

- `PY21C_USE_CUDA` takes precedence over `USE_CUDA` if both are set
- `USE_CUDA` is retained for backwards compatibility
- Requires `nvcc` to be available in the PATH

**Example:**
```bash
# Enable CUDA
PY21C_USE_CUDA=TRUE pip install -e . --no-deps

# Legacy (also works)
USE_CUDA=TRUE pip install -e . --no-deps

# PY21C_USE_CUDA overrides USE_CUDA
USE_CUDA=TRUE PY21C_USE_CUDA=FALSE pip install -e .  # CUDA disabled
```

### PY21C_OPT_LEVEL

Controls the optimization level for C/C++ compilation.

| Value     | Flags                                    | Use Case                        |
|-----------|------------------------------------------|---------------------------------|
| `debug`   | `-O0 -g`                                 | Debugging with gdb/lldb         |
| `release` | `-O3`                                    | Normal builds (default)         |
| `fast`    | `-O3 -march=native -ffast-math -flto`    | Maximum performance             |

**Example:**
```bash
# Debug build
PY21C_OPT_LEVEL=debug pip install -e . --no-deps

# Maximum performance (may reduce numerical precision)
PY21C_OPT_LEVEL=fast pip install -e . --no-deps
```

**Notes:**
- `fast` uses `-ffast-math` which relaxes IEEE floating-point semantics
- `fast` uses `-march=native` which optimizes for the current CPU and may not be portable
- `fast` uses LTO (link-time optimization) which increases build time

### PY21C_DEBUG

Convenience flag that forces debug mode, overriding `PY21C_OPT_LEVEL`.

- When `PY21C_DEBUG=TRUE`, the optimization level is forced to `debug` regardless of `PY21C_OPT_LEVEL`
- For GPU builds, this also enables CUDA device debugging (`-G`)
- A warning is displayed if `PY21C_OPT_LEVEL` was set to something other than `debug`

**Example:**
```bash
# Force debug mode (overrides OPT_LEVEL=fast)
PY21C_DEBUG=TRUE PY21C_OPT_LEVEL=fast pip install -e . --no-deps
# Warning: PY21C_DEBUG=TRUE overrides PY21C_OPT_LEVEL=fast
```

### PY21C_CUDA_DEBUG

Enables CUDA-specific debugging symbols without affecting C/C++ optimization.

- Adds `-g -G` flags to nvcc for cuda-gdb support
- Useful when you need to debug GPU code but want CPU code to remain optimized
- Has no effect when CUDA is disabled (a warning is displayed)

**Example:**
```bash
# CUDA debugging with optimized CPU code
PY21C_CUDA_DEBUG=TRUE PY21C_USE_CUDA=TRUE pip install -e . --no-deps
```

### PY21C_PGO_PHASE

Enables profile-guided optimization (PGO) for improved performance.

PGO is a two-phase process:

1. **Generate phase**: Build with instrumentation, run representative workloads
2. **Use phase**: Rebuild using the collected profile data

| Value      | Flags                                 | Description                    |
|------------|---------------------------------------|--------------------------------|
| `generate` | `-fprofile-generate`                  | Build with instrumentation     |
| `use`      | `-fprofile-use -fprofile-correction`  | Build using profile data       |

**Example:**
```bash
# Phase 1: Generate profile data
PY21C_PGO_PHASE=generate pip install -e . --no-deps
python script/batchC/park19-coeval-small.py
python script/batchC/Munoz21-coeval-small.py

# Phase 2: Use profile data (keep .gcda files from phase 1)
rm -rf build/ builddir/ src/py21cmfast/*.so
PY21C_PGO_PHASE=use pip install -e . --no-deps
```

**Notes:**
- Profile data files (`.gcda`) must be retained between phases
- A warning is displayed if `use` is specified but no profile data exists
- PGO typically provides 5-15% performance improvement

## Common Configurations

### Development (Debugging)
```bash
PY21C_DEBUG=TRUE pip install -e . --no-deps
```

### Production (Default)
```bash
pip install -e . --no-deps
# or explicitly:
PY21C_OPT_LEVEL=release pip install -e . --no-deps
```

### Maximum Performance
```bash
PY21C_OPT_LEVEL=fast pip install -e . --no-deps
```

### GPU Development
```bash
PY21C_USE_CUDA=TRUE PY21C_DEBUG=TRUE pip install -e . --no-deps
```

### GPU Debugging Only (CPU optimized)
```bash
PY21C_USE_CUDA=TRUE PY21C_CUDA_DEBUG=TRUE pip install -e . --no-deps
```

## Precedence Rules

1. `PY21C_USE_CUDA` takes precedence over `USE_CUDA`
2. `PY21C_DEBUG=TRUE` overrides `PY21C_OPT_LEVEL` to `debug`
3. Environment variables take precedence over meson options
