#!/usr/bin/env python
import subprocess
import time

print("GPU Test for 21cmFAST")
print("="*60)

# Load py21cmfast
exec(open('import_21cmfast.py').read())

try:
    # Create parameter objects with GPU-compatible settings
    inputs = p21c.InputParameters(
        random_seed=12345,
        simulation_options=p21c.SimulationOptions(HII_DIM=64, BOX_LEN=50),
        astro_options=p21c.AstroOptions(
            USE_MASS_DEPENDENT_ZETA=True,  # Required for GPU
            USE_MINI_HALOS=False           # Must be False for GPU
        ),
        matter_options=p21c.MatterOptions(
            USE_HALO_FIELD=False,          # Must be False for GPU
            HALO_STOCHASTICITY=False       # Must be False when USE_HALO_FIELD=False
        )
    )
    
    print("✓ Parameters configured for GPU acceleration")
    print("\nTo monitor GPU usage, run in another terminal:")
    print("  watch -n 0.5 nvidia-smi")
    print("\n" + "="*60)
    
    # Compute initial conditions
    print("Computing initial conditions...")
    ic = p21c.compute_initial_conditions(inputs=inputs)
    print("✓ Initial conditions complete")
    
    # Compute ionization field (should use GPU)
    print("\nComputing ionization field (GPU should activate)...")
    start = time.time()
    
    ion = p21c.compute_ionization_field(
        inputs=inputs,
        initial_conditions=ic,
        redshift=8.0
    )
    
    elapsed = time.time() - start
    print(f"✓ Ionization complete in {elapsed:.1f} seconds")
    
    # Show some results
    if hasattr(ion, 'mean_f_ion'):
        print(f"\nResults:")
        print(f"  Mean ionized fraction: {ion.mean_f_ion:.3f}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
