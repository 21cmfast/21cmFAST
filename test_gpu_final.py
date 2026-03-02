#!/usr/bin/env python
# Load the import helper first
exec(open("import_21cmfast.py").read())

print(f"\nStarting GPU test with py21cmfast version {p21c.__version__}")

# Set up GPU-compatible parameters
sim_options = p21c.SimulationOptions(HII_DIM=128, BOX_LEN=100)
astro_options = p21c.AstroOptions(
    USE_MASS_DEPENDENT_ZETA=True,  # Required for GPU
    USE_MINI_HALOS=False,  # Must be False for GPU
)
matter_options = p21c.MatterOptions(
    USE_HALO_FIELD=False  # Must be False for GPU
)

print("\nParameters configured for GPU:")
print(f"  USE_MASS_DEPENDENT_ZETA = {astro_options.USE_MASS_DEPENDENT_ZETA}")
print(f"  USE_MINI_HALOS = {astro_options.USE_MINI_HALOS}")
print(f"  USE_HALO_FIELD = {matter_options.USE_HALO_FIELD}")

print("\n" + "=" * 60)
print("Running calculation... Monitor GPU with: nvidia-smi")
print("In another terminal run: watch -n 0.5 nvidia-smi")
print("=" * 60)

try:
    # Create initial conditions
    ic = p21c.compute_initial_conditions(simulation_options=sim_options)
    print("✓ Initial conditions created")

    # Run ionization calculation - should use GPU
    ionize_box = p21c.compute_ionization_field(
        initial_conditions=ic,
        redshift=8.0,
        astro_options=astro_options,
        matter_options=matter_options,
    )
    print("✓ Ionization calculation complete!")
    if hasattr(ionize_box, "mean_f_ion"):
        print(f"  Ionized fraction: {ionize_box.mean_f_ion:.3f}")

except Exception as e:
    print(f"✗ Error: {e}")
