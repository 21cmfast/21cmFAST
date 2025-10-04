#!/usr/bin/env python
import sys
import os

# Fix ninja path issue
try:
    import _21cmfast_editable_loader
    for item in sys.meta_path:
        if isinstance(item, _21cmfast_editable_loader.MesonpyMetaFinder):
            item._build_cmd = ['ninja']
            break
except:
    pass

# Import py21cmfast
sys.path.insert(0, 'src')
import py21cmfast as p21c

print(f"Imported py21cmfast version {p21c.__version__}")

# Set up GPU-compatible parameters
user_params = {"HII_DIM": 128, "BOX_LEN": 100}
astro_params = {
    "USE_MASS_DEPENDENT_ZETA": True,  # Required for GPU
    "USE_MINI_HALOS": False,          # Must be False for GPU
}

print("\nRunning test calculation with GPU-compatible settings...")
print("Monitor GPU with: watch -n 0.5 nvidia-smi")

# Run calculation
try:
    ic = p21c.initial_conditions(user_params=user_params)
    print("Initial conditions created")
    
    # This should use GPU
    ionize_box = p21c.ionize_box(
        initial_conditions=ic,
        redshift=8.0,
        astro_params=astro_params
    )
    print("Ionization calculation complete!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
