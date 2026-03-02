#!/usr/bin/env python
"""Working GPU test for 21cmFAST"""

# Load py21cmfast
exec(open("import_21cmfast.py").read())

print("\n" + "=" * 60)
print("21cmFAST GPU Test - Monitor with: watch -n 0.5 nvidia-smi")
print("=" * 60)

try:
    import time

    # Create initial conditions
    print("\n[1/3] Initial conditions...")
    ic = p21c.InitialConditions(
        user_params={"HII_DIM": 64, "BOX_LEN": 50}, random_seed=12345
    )
    print("      ✓ Complete")

    # Perturbed field
    print("\n[2/3] Perturbed field...")
    perturb = p21c.perturb_field(redshift=8.0, init_boxes=ic)
    print("      ✓ Complete")

    # Ionization (GPU)
    print("\n[3/3] Ionization field (GPU ACTIVE)...")
    start = time.time()

    # Create parameter objects for GPU mode
    astro_params = p21c.AstroParams(
        ALPHA_STAR=0.5,
        F_STAR10=-1.3,
        t_STAR=0.5,
        USE_MASS_DEPENDENT_ZETA=True,  # GPU required
        USE_MINI_HALOS=False,  # GPU required
    )

    ion = p21c.ionize_box(
        redshift=8.0, init_boxes=ic, perturbed_field=perturb, astro_params=astro_params
    )

    elapsed = time.time() - start
    print(f"      ✓ Complete in {elapsed:.2f}s")

    # Results
    print(f"\nMean ionized fraction: {ion.global_avg_xH:.4f}")
    print("\n✅ SUCCESS! Check nvidia-smi for GPU usage.")

except Exception as e:
    print(f"\n❌ Error: {e}")
