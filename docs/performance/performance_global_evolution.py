"""Produce performance plots for run_global_evolution."""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

import py21cmfast as p21c

output_dir = (
    Path(__file__).parent.parent / "images" / "performance" / "global_evolution"
)

ALL_TEMPLATES = ["const-zeta", "latest", "fixed-halos", "minihalos"]

XArrayName = Literal["N_THREADS", "ZPRIME_STEP_FACTOR", "Z_HEAT_MAX"]

X_ARRAYS: dict[XArrayName, Sequence[float] | np.ndarray] = {
    "N_THREADS": np.arange(1, 11),
    "ZPRIME_STEP_FACTOR": [1.02, 1.04, 1.08, 1.1, 1.15, 1.2],
    "Z_HEAT_MAX": np.arange(10.0, 40.0, 5.0),
}


def run_global_evolution(inputs: p21c.InputParameters) -> float:
    """Run global evolution given inputs, and return runtime in seconds."""
    start = time.time()
    p21c.run_global_evolution(inputs=inputs)
    end = time.time()
    return end - start


def measure_global_evolution_runtime(template: str, **kwargs: float) -> float:
    """Measure runtime of run_global_evolution given a template and kwargs."""
    zmax = kwargs.get("Z_HEAT_MAX", 35.0)
    zstep_factor = kwargs.get("ZPRIME_STEP_FACTOR", 1.02)

    inputs = p21c.InputParameters.from_template(
        template, random_seed=1234, **kwargs
    ).with_logspaced_redshifts(
        zmax=zmax,
        step=zstep_factor,
    )

    runtime = run_global_evolution(inputs)
    return runtime


def plot_and_save_global_evolution_runtime(
    x_array: Sequence[float] | np.ndarray,
    kwarg: str,
    templates: Sequence[str] = ALL_TEMPLATES,
) -> None:
    """Plot run_global_evolution runtime as a function of a key word argument, as given by the x_array, and save the plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for template in templates:
        runtime_list = []
        for x in x_array:
            kwargs = {kwarg: x, "USE_TS_FLUCT": True}
            runtime = measure_global_evolution_runtime(template, **kwargs)
            runtime_list.append(runtime)
        ax.plot(x_array, runtime_list, label=template, marker="o")
    ax.set_xlabel(kwarg)
    ax.set_ylabel("Runtime (sec)")
    ax.legend()
    fig.savefig(output_dir / f"global_evolution_runtime_vs_{kwarg}.png")


def go(kwargs: Sequence[XArrayName] = tuple(X_ARRAYS)) -> None:
    """Plot and save global evolution runtime for each kwarg in kwargs."""
    for kwarg in kwargs:
        plot_and_save_global_evolution_runtime(
            x_array=X_ARRAYS[kwarg],
            kwarg=kwarg,
        )


if __name__ == "__main__":
    tyro.cli(go)
