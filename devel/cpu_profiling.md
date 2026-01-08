# CPU Profiling Guide for 21cmFAST

This guide documents tools and techniques for profiling CPU performance in 21cmFAST.

## Python Profiling with cProfile

### Basic Usage

```bash
# Profile a script and save results
python -m cProfile -o profile.pstats script/batchC/simple-coeval-small.py

# View results with pstats
python -c "import pstats; p = pstats.Stats('profile.pstats'); p.sort_stats('cumulative').print_stats(20)"
```

### Using the Profiling Script

The repository includes `script/profile.sh` for batch profiling:

```bash
# Profile with CPU build
./script/profile.sh -n 3 -t cpu-test -v venv.cpu

# Profile with GPU build
./script/profile.sh -n 3 -t gpu-test -v venv -g

# Profile main branch
./script/profile.sh -n 3 -t main-ref -v venv.main
```

### Visualizing cProfile Results

```bash
# Install visualization tools
pip install snakeviz

# Generate visualization
snakeviz profile.pstats
```

## C/C++ Profiling with perf

### Basic Sampling

```bash
# Record performance data
perf record -g python script/batchC/simple-coeval-small.py

# View report interactively
perf report

# View report as text
perf report --stdio > perf_report.txt
```

### Recording Specific Events

```bash
# CPU cycles
perf record -e cycles -g python script.py

# Cache misses
perf record -e cache-misses -g python script.py

# Branch mispredictions
perf record -e branch-misses -g python script.py
```

### Flame Graphs

```bash
# Install flamegraph tools
git clone https://github.com/brendangregg/FlameGraph.git

# Generate flame graph
perf record -g python script.py
perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > flamegraph.svg
```

## Debugging with gdb

### Basic Usage

```bash
# Start debugger
gdb --args python script/batchC/simple-coeval-small.py

# Common commands:
# run          - start execution
# bt           - backtrace
# break file.c:line - set breakpoint
# print var    - print variable
# cont         - continue execution
```

### Debug Build

For better debugging, build with debug symbols:

```bash
PY21C_DEBUG=TRUE pip install -e . --no-deps
```

### Attaching to Running Process

```bash
# Find process ID
ps aux | grep python

# Attach gdb
gdb -p <PID>
```

## Memory Profiling

### Valgrind for Memory Leaks

Use the suppression files in `devel/` to reduce noise from Python:

```bash
valgrind --suppressions=devel/valgrind-python.supp \
         --suppressions=devel/valgrind-suppress-all-but-c.supp \
         --leak-check=full \
         python script.py
```

### Python Memory Profiling

```bash
pip install memory_profiler

# Add @profile decorator to functions of interest
python -m memory_profiler script.py
```

## OpenMP Thread Analysis

### Thread Timing

```bash
# Set thread count
export OMP_NUM_THREADS=4

# Enable timing
export OMP_DISPLAY_ENV=TRUE

# Run with timing
python script.py
```

### Intel VTune (if available)

```bash
# Basic hotspot analysis
vtune -collect hotspots python script.py

# Thread analysis
vtune -collect threading python script.py
```

## Performance Metrics

### Key Metrics to Monitor

1. **Total execution time** - Compare between builds
2. **Per-function time** - Identify hot spots
3. **Memory allocation** - Check for leaks
4. **Cache performance** - L1/L2/L3 cache hit rates
5. **OpenMP scaling** - Speedup vs thread count

### Collecting Metrics

```bash
# Time execution
time python script.py

# Memory high-water mark
/usr/bin/time -v python script.py

# CPU usage
perf stat python script.py
```

## Example: Full Profiling Session

```bash
# 1. Build with debug symbols
cd install/21cmFAST/cpu-build-v4
PY21C_OPT_LEVEL=debug pip install -e . --no-deps

# 2. Run cProfile
python -m cProfile -o profile.pstats script/batchC/simple-coeval-small.py

# 3. Run perf
perf record -g python script/batchC/simple-coeval-small.py

# 4. Check memory
valgrind --leak-check=summary python script/batchC/simple-coeval-small.py

# 5. Rebuild for production
PY21C_OPT_LEVEL=release pip install -e . --no-deps
```

## See Also

- `script/profile.sh` - Batch profiling script
- `script/plot.py` - Generate timing plots
- `devel/valgrind-python.supp` - Valgrind suppression file
- `devel/performance.py` - Performance measurement utilities
