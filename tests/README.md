# 21cmFAST Test Suite

This directory contains the test suite for py21cmfast. Tests are organized into categories based on their purpose and execution time.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py

# Run tests in parallel
pytest tests/ -n auto

# Run with coverage
pytest tests/ --cov=py21cmfast --cov-report=html
```

## Test Categories

| Category       | Purpose                                             | Speed   | When to Run          |
|----------------|-----------------------------------------------------|---------|----------------------|
| Unit           | Test individual functions in isolation              | Fast    | Every commit         |
| Component      | Test major components (IC, perturb, filtering)      | Medium  | PR validation        |
| Integration    | Test multi-step workflows (IC -> Perturb -> Ion)    | Medium  | PR validation        |
| Regression     | Compare outputs to known-good references            | Slow    | Release validation   |
| I/O            | Test file reading/writing and caching               | Fast    | Every commit         |

### Unit Tests

Fast tests for individual functions and data structures:

- `test_config.py` - Configuration module
- `test_utils.py` - Utility functions
- `test_exceptions.py` - Exception handling
- `test_data_exists.py` - Package data presence
- `test_tables.py` - Data tables
- `test_input_structs.py` - Input parameter structures
- `test_output_structs.py` - Output data structures
- `test_input_serialization.py` - Input serialization/deserialization
- `test_templates.py` - Parameter templates

### Component Tests

Tests for major simulation components:

- `test_initial_conditions.py` - Initial conditions computation
- `test_perturb.py` - Perturbation field computation
- `test_filtering.py` - Filtering functions
- `test_rsds.py` - Redshift space distortions
- `test_singlefield.py` - Single field computations
- `test_cfuncs.py` - C function interfaces
- `test_c_interpolation_tables.py` - C interpolation tables (extensive)
- `test_halo_sampler.py` - Halo sampling algorithms
- `test_lightconer.py` - Lightcone generation
- `test_lightcones.py` - Lightcone output structures

### Integration Tests

End-to-end workflow tests:

- `test_drivers_coev.py` - Coeval driver workflows
- `test_drivers_lc.py` - Lightcone driver workflows

### Regression Tests

Tests comparing outputs to pre-computed reference data:

- `test_integration_features.py` - Power spectra regression tests
  - Reference data in `test_data/*.h5` (25 files)
  - Tests various physics configurations
  - Stochastic tests have higher tolerance (5%)

### I/O Tests

Tests for file operations and caching:

- `test_high_level_io.py` - High-level I/O operations
- `io/test_caching.py` - Caching system
- `io/test_h5.py` - HDF5 file operations

### Other Tests

- `test_cli.py` - Command-line interface
- `test_plotting.py` - Plotting functions
- `test_classy_interface.py` - CLASS cosmology interface

## Fixtures

Session-scoped fixtures for expensive computations (defined in `conftest.py`):

- `ic` - Cached InitialConditions
- `perturbed_field` - Cached PerturbedField
- `ionize_box` - Cached IonizedBox
- `lc` - Cached LightCone

Module-scoped fixtures:

- `module_direc` - Temporary directory for module
- `test_direc` - Temporary directory for test

## Reference Data

Reference data for regression tests is stored in `test_data/`:

- `power_spectra_*.h5` - Power spectrum reference data
- `perturb_field_data_*.h5` - Perturbation field reference data

Reference data was last regenerated August 2025 using the nanobind codebase.

## Known Issues

1. **test_cli.py** - May fail with cyclopts version mismatch
2. **sampler_ts_ir_onethread** - Higher variance (>5%) in neutral_fraction due to single-threaded execution
3. **GPU tests** - Some tests require CUDA runtime; run on compute nodes with GPU

## Adding New Tests

1. Choose the appropriate category based on test scope
2. Use existing fixtures when possible to avoid redundant computation
3. For regression tests, add reference data to `test_data/`
4. Use `@pytest.mark.parametrize` for testing multiple configurations
