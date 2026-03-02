#!/usr/bin/env python
"""Test GPU acceleration in 21cmFAST"""

# Load py21cmfast
exec(open("import_21cmfast.py").read())

print("\n" + "=" * 60)
print("21cmFAST GPU Acceleration Test")
print("=" * 60)

try:
    # Create GPU-compatible parameters (avoiding conflicts)
    inputs = p21c.InputParameters(
        random_seed=12345,
        simulation_options=p21c.SimulationOptions(HII_DIM=64, BOX_LEN=50),
        astro_options=p21c.AstroOptions(
            USE_MASS_DEPENDENT_ZETA=True,  # Required for GPU
            USE_MINI_HALOS=False,  # Must be False for GPU
            USE_UPPER_STELLAR_TURNOVER=False,  # Must be False when USE_HALO_FIELD=False
        ),
        matter_options=p21c.MatterOptions(
            USE_HALO_FIELD=False,  # Must be False for GPU
            HALO_STOCHASTICITY=False,  # Must be False when USE_HALO_FIELD=False
        ),
    )

    print("\n✓ GPU-compatible parameters configured:")
    print(f"  Box dimensions: {inputs.simulation_options.HII_DIM}^3")
    print(f"  USE_MASS_DEPENDENT_ZETA: {inputs.astro_options.USE_MASS_DEPENDENT_ZETA}")
    print(f"  USE_MINI_HALOS: {inputs.astro_options.USE_MINI_HALOS}")
    print(f"  USE_HALO_FIELD: {inputs.matter_options.USE_HALO_FIELD}")

    print("\n" + "-" * 60)
    print("TO MONITOR GPU: Run in another terminal:")
    print("  watch -n 0.5 nvidia-smi")
    print("-" * 60)

    # Initial conditions
    print("\n1. Computing initial conditions...")
    ic = p21c.compute_initial_conditions(inputs=inputs)
    print("   ✓ Complete")

    # Ionization field (GPU should activate here)
    print("\n2. Computing ionization field...")
    print("   >>> GPU SHOULD ACTIVATE NOW <<<")

    import time

    start = time.time()

    ion = p21c.compute_ionization_field(
        inputs=inputs, initial_conditions=ic, redshift=8.0
    )

    elapsed = time.time() - start

    print(f"   ✓ Complete in {elapsed:.2f} seconds")

    # Results
    if hasattr(ion, "mean_f_ion"):
        print(f"\nMean ionized fraction at z=8.0: {ion.mean_f_ion:.4f}")

    print("\n" + "=" * 60)
    print("Test complete! Check nvidia-smi output for GPU usage.")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback

    traceback.print_exc()
