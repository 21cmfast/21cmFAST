#!/usr/bin/env python
import subprocess
import time

# Check initial GPU status
print("Checking GPU before test...")
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
    capture_output=True, text=True
)
print(f"Initial GPU status: {result.stdout.strip()}")

# Load py21cmfast
exec(open('import_21cmfast.py').read())

# Try to run a simple calculation
print("\nAttempting to run 21cmFAST calculation...")
print("Watch GPU with: watch -n 0.5 nvidia-smi")

try:
    # Use the wrapper functions
    inputs = p21c.InputParameters(
        simulation_options=p21c.SimulationOptions(HII_DIM=64, BOX_LEN=50),
        astro_options=p21c.AstroOptions(
            USE_MASS_DEPENDENT_ZETA=True,
            USE_MINI_HALOS=False
        ),
        matter_options=p21c.MatterOptions(
            USE_HALO_FIELD=False,
            HALO_STOCHASTICITY=False
        )
    )
    
    print(f"Parameters set for GPU mode")
    print(f"Computing initial conditions...")
    
    # Try the compute function
    ic = p21c.compute_initial_conditions(inputs=inputs)
    print("Initial conditions complete")
    
    # Check GPU during ionization
    print("\nComputing ionization (should use GPU)...")
    start = time.time()
    ion = p21c.compute_ionization_field(
        inputs=inputs,
        initial_conditions=ic,
        redshift=8.0
    )
    elapsed = time.time() - start
    
    print(f"✓ Calculation complete in {elapsed:.1f} seconds")
    
    # Check final GPU status
    time.sleep(1)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print(f"Final GPU status: {result.stdout.strip()}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
