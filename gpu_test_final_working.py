#!/usr/bin/env python
"""Test GPU acceleration in 21cmFAST - Final version with all conflicts resolved"""

# Load py21cmfast
exec(open("import_21cmfast.py").read())

print("\n" + "=" * 60)
print("21cmFAST GPU Acceleration Test")
print("=" * 60)

try:
    # Create GPU-compatible parameters with ALL necessary settings
    inputs = p21c.InputParameters(
        random_seed=12345,
        simulation_options=p21c.SimulationOptions(HII_DIM=64, BOX_LEN=50),
        astro_options=p21c.AstroOptions(
            USE_MASS_DEPENDENT_ZETA=True,  # Required for GPU
            USE_MINI_HALOS=False,  # Must be False for GPU
            USE_UPPER_STELLAR_TURNOVER=False,  # Must be False when USE_HALO_FIELD=False
            USE_EXP_FILTER=False,  # Must be False when USE_HALO_FIELD=False
        ),
        matter_options=p21c.MatterOptions(
            USE_HALO_FIELD=False,  # Must be False for GPU
            HALO_STOCHASTICITY=False,  # Must be False when USE_HALO_FIELD=False
        ),
        cosmo_params=p21c.CosmoParams(),  # Use defaults
    )

    print("\n✓ GPU-compatible parameters successfully configured")
    print("\nSettings for GPU acceleration:")
    print("  ✓ USE_MASS_DEPENDENT_ZETA = True")
    print("  ✓ USE_MINI_HALOS = False")
    print("  ✓ USE_HALO_FIELD = False")
    print(f"  Box: {inputs.simulation_options.HII_DIM}^3 cells")

    print("\n" + "=" * 60)
    print("IMPORTANT: Monitor GPU in another terminal with:")
    print("  watch -n 0.5 nvidia-smi")
    print("=" * 60)

    # Run the calculation
    print("\nStarting calculations...")

    # 1. Initial conditions (CPU only)
    print("\n[1/2] Initial conditions (CPU)...")
    ic = p21c.compute_initial_conditions(inputs=inputs)
    print("      ✓ Complete")

    # 2. Ionization field (GPU should activate)
    print("\n[2/2] Ionization field (GPU)...")
    print("      >>> GPU SHOULD BE ACTIVE NOW <<<")

    import time

    start = time.time()

    ion = p21c.compute_ionization_field(
        inputs=inputs, initial_conditions=ic, redshift=8.0
    )

    elapsed = time.time() - start

    print(f"      ✓ Complete in {elapsed:.2f} seconds")

    # Show results
    if hasattr(ion, "mean_f_ion"):
        print("\n📊 Results at z=8.0:")
        print(f"   Mean ionized fraction: {ion.mean_f_ion:.4f}")

    print("\n" + "=" * 60)
    print("SUCCESS! Test complete.")
    print("Check nvidia-smi to confirm GPU was used.")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nFull traceback:")
    import traceback

    traceback.print_exc()
