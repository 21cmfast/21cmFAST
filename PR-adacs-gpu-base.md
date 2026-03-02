# PR: ADACS GPU Base Implementation

**Branch:** `adacs-gpu-base`
**Base:** `main`
**From:** `gusgw/21cmFAST`

## Summary

This PR provides the foundation for GPU acceleration in 21cmFAST, developed as part of the ADACS optimization project. It includes:

- Complete GPU implementation of the InitialConditions path
- Support for both ZELDOVICH and 2LPT perturbation algorithms on GPU
- Verified CPU/GPU numerical parity across all physics configurations
- Build system improvements for profiling and optimization workflows

## Key Changes

### GPU Initial Conditions Implementation

- **GPU InitialConditions path**: Full CUDA implementation of initial condition generation, including density field computation, velocity field generation, and high-resolution perturbation support
- **2LPT support on GPU**: Second-order Lagrangian Perturbation Theory implementation with correct cuFFT handling and volume scaling
- **MapMass GPU kernel**: Rewritten to match CPU algorithm exactly, with proper velocity index calculation and bounds checking

### Build System Improvements

- Profile-Guided Optimization (PGO) workflow support via `PY21C_PGO_PHASE` and `PY21C_PGO_DIR` environment variables
- Debug symbols and symbol visibility controls for profiling
- Build control environment variables for flexible optimization levels

### Testing Infrastructure

- GPU-CPU parity test framework with reference data
- Comprehensive field diagnostics for CPU/GPU comparison
- Three-way comparison infrastructure (main vs cpu-optimized vs gpu)

## Validation

Extensive three-way comparison testing was performed on both Skylake/P100 and Milan/A100 architectures:

### Numerical Parity (CPU vs GPU)

| Architecture | Min Correlation | Notes                                    |
|--------------|-----------------|------------------------------------------|
| Skylake/P100 | 0.999999        | All non-discrete scripts                 |
| Milan/A100   | 0.9995          | All non-discrete scripts                 |

Discrete halo sampling scripts show expected divergence due to different random number sequences on CPU vs GPU.

### Performance (Average over 46 test scripts)

| Architecture | CPU vs Main | GPU vs CPU |
|--------------|-------------|------------|
| Skylake/P100 | +13.0%      | +8.9%      |
| Milan/A100   | +10.5%      | -7.2%      |

Note: GPU is slower on A100 for these small test workloads due to transfer overhead. Larger production runs are expected to benefit more from GPU acceleration.

### Selected Recent Bug Fixes

- Fix cuFFT first-call failure on P100/Pascal GPUs
- Fix GPU velocity displacement calculation in MapMass_gpu
- Fix GPU stochasticity: position randomization and type mismatch for discrete halo sampling
- Fix 2LPT implementation: cuFFT R2C requires tightly-packed input (not FFT-padded),
  and phi_2 needs VOLUME pre-multiplication to match velocity kernel expectations

### Test Configurations

Testing covered all major physics configurations:
- park19, Munoz21, Qin20 physics models
- Coeval and lightcone calculations
- With and without 2LPT (ZELDOVICH algorithm)
- Minihalo and discrete halo sampling modes
- Multiple random seeds for reproducibility

## Commits (75 total)

Key commits:
- `7c3a5060` Fix GPU 2LPT implementation
- `278aa749` Re-enable GPU InitialConditions path
- `0c01b8f7` Implement 2LPT support in GPU MapMass kernel
- `4433dea1` Fix cuFFT first-call failure on P100/Pascal GPUs
- `4ca51dfc` Cherry-pick InitialConditions GPU implementation
- `107c6f9f` Fix discrete halo correlation failures in GPU stochasticity sampling
- `32568104` Fix GPU stochasticity: position randomization and type mismatch
- `29a9b3d3` Fix GPU velocity displacement calculation in MapMass_gpu

## Future Work

This branch serves as the base for continued GPU optimization work:
- GPU profiling and kernel optimization
- Extended GPU coverage for additional computation stages
- Performance optimization for production workloads

## Test Plan

- [ ] CI tests pass
- [ ] GPU parity tests pass on P100 and A100
- [ ] Coeval calculations produce matching results between CPU and GPU
- [ ] Lightcone calculations produce matching results between CPU and GPU
- [ ] Both ZELDOVICH and 2LPT algorithms work correctly on GPU
