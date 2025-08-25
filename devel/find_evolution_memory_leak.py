"""Run a quick simulation to test memory leaks."""

from collections import deque

import py21cmfast as p21c

inputs = p21c.InputParameters.from_template(
    "Munoz21", random_seed=0
).evolve_input_structs(
    HII_DIM=128,
    HIRES_TO_LOWRES_FACTOR=3,
    LOWRES_CELL_SIZE_MPC=1.5,
    Z_HEAT_MAX=35.0,
    ZPRIME_STEP_FACTOR=1.2,
)
print(inputs)

coevals = deque(maxlen=2)
for coeval, _ in p21c.generate_coeval(
    inputs=inputs, cache=p21c.OutputCache("."), write=True, regenerate=True
):
    print(coeval.redshift)
    print("---------------")
    coevals.append(coeval)
    for ostruct in coevals[0].output_structs.values():
        print(ostruct.__class__.__name__)
        for name, ary in ostruct.arrays.items():
            print(f"   {name}: {ary.state}")
