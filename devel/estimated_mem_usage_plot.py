"""Simple example script for plotting estimated memory usage."""

import matplotlib.pyplot as plt
import numpy as np

from py21cmfast import AstroParams, CosmoParams, FlagOptions, UserParams
from py21cmfast._memory import estimate_memory_lightcone

user_params = UserParams(
    {
        "HII_DIM": 250,
        "BOX_LEN": 2000.0,
        "DIM": 1000,
        "N_THREADS": 8,
        "USE_RELATIVE_VELOCITIES": True,
        "POWER_SPECTRUM": 5,
        "USE_FFTW_WISDOM": True,
        "PERTURB_ON_HIGH_RES": True,
        "USE_INTERPOLATION_TABLES": True,
        "FAST_FCOLL_TABLES": True,
    }
)
flag_options = FlagOptions(
    {
        "INHOMO_RECO": True,
        "USE_MASS_DEPENDENT_ZETA": True,
        "USE_TS_FLUCT": True,
        "USE_MINI_HALOS": True,
    },
    USE_VELS_AUX=True,
)


h2dim = np.array(list(range(200, 1500, 100)))


mems4 = []
for h2 in range(200, 1500, 100):
    user_params.update(HII_DIM=h2, DIM=4 * h2)
    mem = estimate_memory_lightcone(
        max_redshift=1420 / 50 - 1,
        redshift=1420 / 200 - 1,
        user_params=user_params,
        flag_options=flag_options,
        cosmo_params=CosmoParams(),
        astro_params=AstroParams(),
    )
    mems4.append(mem)
peaks4 = np.array([m["peak_memory"] / (1024**3) for m in mems4])

mems3 = []
for h2 in range(200, 1500, 100):
    user_params.update(HII_DIM=h2, DIM=3 * h2)
    mem = estimate_memory_lightcone(
        max_redshift=1420 / 50 - 1,
        redshift=1420 / 200 - 1,
        user_params=user_params,
        flag_options=flag_options,
        cosmo_params=CosmoParams(),
        astro_params=AstroParams(),
    )
    mems3.append(mem)
peaks3 = np.array([m["peak_memory"] / (1024**3) for m in mems3])

mems2 = []
for h2 in range(200, 1500, 100):
    user_params.update(HII_DIM=h2, DIM=2 * h2)
    mem = estimate_memory_lightcone(
        max_redshift=1420 / 50 - 1,
        redshift=1420 / 200 - 1,
        user_params=user_params,
        flag_options=flag_options,
        cosmo_params=CosmoParams(),
        astro_params=AstroParams(),
    )
    mems2.append(mem)
peaks2 = np.array([m["peak_memory"] / (1024**3) for m in mems2])


user_params.update(HII_DIM=1300, DIM=2600)  # what we actually did.
mem = estimate_memory_lightcone(
    max_redshift=1420 / 50 - 1,
    redshift=1420 / 200 - 1,
    user_params=user_params,
    flag_options=flag_options,
    cosmo_params=CosmoParams(),
    astro_params=AstroParams(),
)


plt.plot(h2dim, peaks2, label="DIM=2*HII_DIM")
plt.plot(h2dim, peaks3, label="DIM=3*HII_DIM")
plt.plot(h2dim, peaks4, label="DIM=4*HII_DIM")

plt.scatter(
    1300,
    mem["peak_memory"] / 1024**3,
    marker="*",
    color="k",
    label="Proposed Sims",
    s=100,
    zorder=3,
)

plt.scatter(
    1300, 2924.847, marker="+", color="r", label="Measured Peak RAM", s=75, zorder=3
)

theory = h2dim**3
plt.plot(
    h2dim,
    theory * (peaks3.max() / theory.max()) * 1.2,
    label="HII_DIM^3 scaling",
    color="grey",
    ls="--",
)

plt.axhline(4000, linestyle="dashed", color="k")
plt.text(h2dim[0], 4300, "Avail RAM on Bridges2-EM")
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Peak memory (GB)")
plt.xlabel("HII_DIM")

plt.savefig("peak_memory_usage.png")
