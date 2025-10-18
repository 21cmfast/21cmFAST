"""Tools for simulation managements."""

from . import InputParameters
from .io.caching import CacheConfig
from .wrapper import outputs as ostrct
from .wrapper.arrays import Array


def get_expected_outputs(
    inputs: InputParameters, cache_config: CacheConfig = CacheConfig.on()
) -> dict[str, dict[str, Array]]:
    """Return a dictionary of expected output structs and their arrays for given inputs."""
    out = {
        "InitialConditions": ostrct.InitialConditions.new(inputs).arrays,
        "PerturbedField": ostrct.PerturbedField.new(inputs, redshift=6).arrays,
        "IonizedBox": ostrct.IonizedBox.new(inputs, redshift=6).arrays,
        "BrightnessTemp": ostrct.BrightnessTemp.new(inputs, redshift=6).arrays,
    }

    if inputs.matter_options.USE_HALO_FIELD:
        out |= {
            "HaloCatalog": ostrct.HaloCatalog.new(inputs, redshift=6).arrays,
            "HaloBox": ostrct.HaloBox.new(inputs, redshift=6).arrays,
        }

        if inputs.astro_options.USE_TS_FLUCT:
            out["XraySourceBox"] = ostrct.XraySourceBox.new(inputs, redshift=6).arrays

    if inputs.astro_options.USE_TS_FLUCT:
        out |= {
            "TsBox": ostrct.TsBox.new(inputs, redshift=6.0).arrays,
        }

    # Make the outputs consistent with the cache config
    if not cache_config.initial_conditions:
        del out["InitialConditions"]
    if not cache_config.perturbed_field:
        del out["PerturbedField"]
    if not cache_config.ionized_box:
        del out["IonizedBox"]
    if not cache_config.brightness_temp:
        del out["BrightnessTemp"]
    if not cache_config.halo_field and "HaloCatalog" in out:
        del out["HaloCatalog"]
    if not cache_config.halobox and "HaloBox" in out:
        del out["HaloBox"]
    if not cache_config.halo_field and "HaloCatalog" in out:
        del out["HaloCatalog"]
    if not cache_config.spin_temp and "TsBox" in out:
        del out["TsBox"]
    if not cache_config.xray_source_box and "XraySourceBox" in out:
        del out["XraySourceBox"]

    return out


def get_expected_sizes(
    inputs: InputParameters,
    cache_config: CacheConfig = CacheConfig.on(),
    redshift: float = 6.0,
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
    structs = get_expected_outputs(inputs, cache_config)

    if "InitialConditions" in structs:
        out = {
            "InitialConditions": ostrct.InitialConditions.new(inputs).get_full_size()
        }
    else:
        out = {}

    for name in structs:
        struct = getattr(ostrct, name)
        if name != "InitialConditions":
            out[name] = struct.new(inputs, redshift=redshift).get_full_size()

    return out


def get_total_storage_size(
    inputs: InputParameters, cache_config: CacheConfig = CacheConfig.on()
):
    """Compute the total storage size of a simulation."""
    out = {}
    for i, z in enumerate(inputs.node_redshifts):
        sizes = get_expected_sizes(inputs, cache_config=cache_config, redshift=z)
        if i == 0:
            if "InitialConditions" in sizes:
                out["InitialConditions"] = (1, sizes["InitialConditions"])
            for name in sizes:
                if name != "InitialConditions":
                    out[name] = (0, 0)

        for name, size in sizes.items():
            if name != "InitialConditions":
                out[name] = (out[name][0] + 1, out[name][1] + size)

    return out
