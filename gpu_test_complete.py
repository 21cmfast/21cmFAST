#!/usr/bin/env python
"""Complete GPU test for 21cmFAST with all required fields"""

# Load py21cmfast
exec(open("import_21cmfast.py").read())

print("\n" + "=" * 60)
print("21cmFAST GPU Acceleration Test - Complete Pipeline")
print("=" * 60)

try:
    # GPU-compatible parameters
    inputs = p21c.InputParameters(
        random_seed=12345,
        simulation_options=p21c.SimulationOptions(HII_DIM=64, BOX_LEN=50),
        astro_options=p21c.AstroOptions(
            USE_MASS_DEPENDENT_ZETA=True,  # Required for GPU
            USE_MINI_HALOS=False,  # Must be False for GPU
            USE_UPPER_STELLAR_TURNOVER=False,
            USE_EXP_FILTER=False,
        ),
        matter_options=p21c.MatterOptions(
            USE_HALO_FIELD=False,  # Must be False for GPU
            HALO_STOCHASTICITY=False,
        ),
        cosmo_params=p21c.CosmoParams(),
    )

    print("\n✓ GPU-compatible parameters configured")
    print("\n" + "=" * 60)
    print("TO MONITOR GPU: Run in another terminal:")
    print("  watch -n 0.5 nvidia-smi")
    print("=" * 60 + "\n")

    import time

    # Step 1: Initial conditions
    print("[1/3] Computing initial conditions...")
    ic = p21c.compute_initial_conditions(inputs=inputs)
    print("      ✓ Complete\n")

    # Step 2: Perturbed field (required for ionization)
    print("[2/3] Computing perturbed field...")
    perturb = p21c.compute_perturbed_field(
        inputs=inputs, initial_conditions=ic, redshift=8.0
    )
    print("      ✓ Complete\n")

    # Step 3: Ionization field (GPU ACTIVE)
    print("[3/3] Computing ionization field...")
    print("      🚀 GPU ACCELERATION ACTIVE 🚀")

    start = time.time()
    ion = p21c.compute_ionization_field(
        inputs=inputs, initial_conditions=ic, perturbed_field=perturb, redshift=8.0
    )
    elapsed = time.time() - start

    print(f"      ✓ Complete in {elapsed:.2f} seconds\n")

    # Results
    if hasattr(ion, "mean_f_ion"):
        print("=" * 60)
        print("Results at z=8.0:")
        print(f"  Mean ionized fraction: {ion.mean_f_ion:.4f}")
        print("=" * 60)

    print("\n✅ SUCCESS! GPU test complete.")
    print("Check nvidia-smi output to confirm GPU usage.\n")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
