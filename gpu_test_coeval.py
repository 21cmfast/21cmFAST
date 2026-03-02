#!/usr/bin/env python
"""GPU test using run_coeval"""

exec(open("import_21cmfast.py").read())

print("\n" + "=" * 60)
print("21cmFAST GPU Test using run_coeval")
print("Monitor GPU: watch -n 0.5 nvidia-smi")
print("=" * 60 + "\n")

try:
    import time

    # Set up parameters for GPU
    user_params = {"HII_DIM": 64, "BOX_LEN": 50}

    astro_params = {
        "USE_MASS_DEPENDENT_ZETA": True,  # Required for GPU
        "USE_MINI_HALOS": False,  # Required for GPU
    }

    print("Running coeval calculation at z=8...")
    print("GPU should activate during ionization step...\n")

    start = time.time()

    # Run the full coeval calculation
    coeval = p21c.run_coeval(
        redshift=8.0,
        user_params=user_params,
        astro_params=astro_params,
        random_seed=12345,
    )

    elapsed = time.time() - start

    print(f"\n✓ Complete in {elapsed:.1f} seconds")
    print(
        f"Ionized fraction: {coeval.brightness_temp.global_avg_brightness_temp:.2f} K"
    )
    print("\n✅ SUCCESS! Check nvidia-smi to confirm GPU was used.")

except Exception as e:
    print(f"❌ Error: {e}")
