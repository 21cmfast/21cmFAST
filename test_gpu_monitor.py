#!/usr/bin/env python
import subprocess
import threading
import time

# Load the import helper first
exec(open("import_21cmfast.py").read())


def monitor_gpu(stop_event):
    """Monitor GPU in background"""
    max_util = 0
    max_mem = 0
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                util, mem = result.stdout.strip().split(", ")
                max_util = max(max_util, int(util))
                max_mem = max(max_mem, int(mem))
                if int(util) > 5:
                    print(
                        f"  GPU Active: {util}% utilization, {mem}MB memory", end="\r"
                    )
        except:
            pass
        time.sleep(0.5)
    return max_util, max_mem


print(f"\nGPU Test with py21cmfast version {p21c.__version__}")

# Set up GPU-compatible parameters - fix the conflict
sim_options = p21c.SimulationOptions(HII_DIM=64, BOX_LEN=50)  # Smaller for quick test
astro_options = p21c.AstroOptions(
    USE_MASS_DEPENDENT_ZETA=True,  # Required for GPU
    USE_MINI_HALOS=False,  # Must be False for GPU
)
matter_options = p21c.MatterOptions(
    USE_HALO_FIELD=False,  # Must be False for GPU
    HALO_STOCHASTICITY=False,  # Must be False when USE_HALO_FIELD=False
)

print("\nGPU-compatible parameters:")
print(f"  USE_MASS_DEPENDENT_ZETA = {astro_options.USE_MASS_DEPENDENT_ZETA}")
print(f"  USE_MINI_HALOS = {astro_options.USE_MINI_HALOS}")
print(f"  USE_HALO_FIELD = {matter_options.USE_HALO_FIELD}")
print(f"  Box size: {sim_options.HII_DIM}^3")

print("\n" + "=" * 60)
print("Starting calculation with GPU monitoring...")
print("=" * 60)

# Start GPU monitoring
stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event,))
monitor_thread.start()

try:
    # Create initial conditions
    print("Creating initial conditions...")
    ic = p21c.compute_initial_conditions(simulation_options=sim_options)
    print("✓ Initial conditions created")

    # Run ionization calculation - should use GPU
    print("Running ionization calculation (GPU should activate)...")
    ionize_box = p21c.compute_ionization_field(
        initial_conditions=ic,
        redshift=8.0,
        astro_options=astro_options,
        matter_options=matter_options,
    )
    print("\n✓ Ionization calculation complete!")

finally:
    stop_event.set()
    monitor_thread.join(timeout=1)

print("\nGPU monitoring stopped")
