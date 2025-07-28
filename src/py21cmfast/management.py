"""Tools for simulation managements."""

from . import InputParameters
from .wrapper import outputs as ostrct
from .wrapper.arrays import Array


def get_expected_outputs(inputs: InputParameters) -> dict[str, dict[str, Array]]:
    """Return a dictionary of expected output structs and their arrays for given inputs."""
    out = {
        "InitialConditions": ostrct.InitialConditions.new(inputs).arrays,
        "PerturbedField": ostrct.PerturbedField.new(inputs, redshift=6).arrays,
        "IonizedBox": ostrct.IonizedBox.new(inputs, redshift=6).arrays,
        "BrightnessTemp": ostrct.BrightnessTemp.new(inputs, redshift=6).arrays,
    }

    if inputs.matter_options.USE_HALO_FIELD:
        out |= {
            "HaloField": ostrct.HaloField.new(inputs, redshift=6).arrays,
            "HaloBox": ostrct.HaloBox.new(inputs, redshift=6).arrays,
            "PerturbHaloField": ostrct.PerturbHaloField.new(inputs, redshift=6).arrays,
        }

        if inputs.astro_options.USE_TS_FLUCT:
            out["XraySourceBox"] = ostrct.XraySourceBox.new(inputs, redshift=6).arrays

    if inputs.astro_options.USE_TS_FLUCT:
        out |= {
            "TsBox": ostrct.TsBox.new(inputs, redshift=6.0).arrays,
        }

    return out


def get_expected_sizes(
    inputs: InputParameters, redshift: float = 6.0
) -> dict[str, int]:
    """Compute the expected size of each of the relevant output structs in a sim.

    The returned dictionary holds the size in bytes of each relevant output struct
    kind, estimated as the _full_ size given the inputs.

    The peak memory usage of a simulation is *not* simply the sum of these
    numbers. Running :func:`run_lightcone` or :func:`run_coeval` is able to intelligently
    offload memory onto disk at various points, reducing peak memory, however very often
    two of each type of array must be in memory at once, and peak memory while inside
    a C compute function can be temporarily higher again.
    """
    structs = get_expected_outputs(inputs)

    out = {}
    out["InitialConditions"] = ostrct.InitialConditions.new(inputs).get_full_size()

    for name in structs:
        struct = getattr(ostrct, name)
        if name != "InitialConditions":
            out[name] = struct.new(inputs, redshift=redshift).get_full_size()

    return out


def get_total_storage_size(inputs: InputParameters):
    """Compute the total storage size of a simulation."""
    out = {}
    for i, z in enumerate(inputs.node_redshifts):
        sizes = get_expected_sizes(inputs, redshift=z)
        if i == 0:
            out["InitialConditions"] = (1, sizes["InitialConditions"])
            for name in sizes:
                if name != "InitialConditions":
                    out[name] = (0, 0)

        for name, size in sizes.items():
            if name != "InitialConditions":
                out[name] = (out[name][0] + 1, out[name][1] + size)

    return out
