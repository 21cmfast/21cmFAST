import pytest

import numpy as np
from astropy import constants as c
from astropy import units as u

from py21cmfast import AstroParams, CosmoParams, FlagOptions, UserParams, global_params
from py21cmfast.c_21cmfast import ffi, lib

from . import produce_integration_test_data as prd
from . import test_c_interpolation_tables as cint

RELATIVE_TOLERANCE = 1e-1

options_hmf = list(cint.OPTIONS_HMF.keys())


# @pytest.mark.xfail
@pytest.mark.parametrize("name", options_hmf)
def test_sampler(name, plt):
    redshift, kwargs = cint.OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])
    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_IT(up(), cp(), ap(), fo())
    lib.Broadcast_struct_global_STOC(up(), cp(), ap(), fo())

    l10min = 8
    l10max = 11
    edges = np.logspace(l10min, l10max, num=10 * (l10max - l10min))
    bin_minima = edges[:-1]
    bin_maxima = edges[1:]
    dlnm = np.log(edges[1:]) - np.log(edges[:-1])
    # centres = (edges[:-1] * np.exp(dlnm / 2)).astype("f4")

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    n_cond = 1000

    conditions_d = np.array([-0.9, 0, 0.5, 1, 1.4])
    conditions_m = np.array([1e8, 1e9, 1e10, 1e11, 1e12])

    z = 6.0
    z_prev = 5.8
    growth_prev = lib.dicke(z_prev)
    growthf = lib.dicke(z)

    sigma_cond_m = np.vectorize(lib.sigma_z0)(conditions_m)
    delta_cond_m = (
        np.vectorize(lib.get_delta_crit)(up.HMF, sigma_cond_m, growth_prev)
        * growthf
        / growth_prev
    )

    mass_dens = cp.cosmo.Om0 * cp.cosmo.critical_density(0).to("Mpc-3 M_sun").value
    cellvol = (up.BOX_LEN / up.HII_DIM) ** 3
    cell_mass = cellvol * mass_dens

    volume_total_m = conditions_d * n_cond / mass_dens
    cond_mass_m = conditions_d

    delta_cond_d = conditions_d
    sigma_cond_d = np.full_like(delta_cond_d, lib.sigma_z0(cell_mass))
    volume_total_d = np.full_like(delta_cond_d, cellvol * n_cond)
    cond_mass_d = cell_mass

    crd_in = np.zeros(3 * n_cond).astype("i4")

    # CELL DENSITY CONDITIONS WITH FIXED SIZE
    for i, d in enumerate(conditions_d):
        cond_in = np.full(n_cond, fill_value=d).astype("f4")  # mass at z6

        nhalo_out = np.zeros(1).astype("i4")
        N_out = np.zeros(n_cond).astype("i4")
        M_out = np.zeros(n_cond).astype("f8")
        exp_M = np.zeros(n_cond).astype("f8")
        exp_N = np.zeros(n_cond).astype("f8")
        halomass_out = np.zeros(int(1e8)).astype("f4")
        halocrd_out = np.zeros(int(3e8)).astype("i4")

        print(f"starting {i} d={d}", flush=True)
        lib.single_test_sample(
            up(),
            cp(),
            ap(),
            fo(),
            12345,  # TODO: homogenize
            n_cond,
            ffi.cast("float *", cond_in.ctypes.data),
            ffi.cast("int *", crd_in.ctypes.data),
            z,
            -1,
            ffi.cast("int *", nhalo_out.ctypes.data),
            ffi.cast("int *", N_out.ctypes.data),
            ffi.cast("double *", exp_N.ctypes.data),
            ffi.cast("double *", M_out.ctypes.data),
            ffi.cast("double *", exp_M.ctypes.data),
            ffi.cast("float *", halomass_out.ctypes.data),
            ffi.cast("int *", halocrd_out.ctypes.data),
        )

        # since the tables are reallocated in the test sample function, we redo them here
        lib.initialiseSigmaMInterpTable(edges[0] / 2, edges[-1])

        # get CMF integrals in the same bins
        bin_minima = edges[:-1]
        bin_maxima = edges[1:]
        binned_cmf = np.vectorize(lib.Nhalo_Conditional)(
            growthf,
            np.log(bin_minima),
            np.log(bin_maxima),
            cell_mass,
            sigma_cond_d[i],
            delta_cond_d[i],
            up.INTEGRATION_METHOD_HALOS,
        )

        hist, _ = np.histogram(halomass_out, edges)
        mf_out = hist / volume_total_d[i] / dlnm
        binned_cmf = binned_cmf * n_cond / volume_total_d[i] / dlnm * cond_mass_d

        np.testing.assert_allclose(N_out.mean(), exp_N[0], rtol=RELATIVE_TOLERANCE)
        np.testing.assert_allclose(M_out.mean(), exp_M[0], rtol=RELATIVE_TOLERANCE)
        np.testing.assert_allclose(mf_out, binned_cmf, rtol=RELATIVE_TOLERANCE)

    # HALO MASS CONDITIONS WITH FIXED z-step
    for i, m in enumerate(conditions_m):
        cond_in = np.full(n_cond, fill_value=m).astype("f4")  # mass at z6

        nhalo_out = np.zeros(1).astype("i4")
        N_out = np.zeros(n_cond).astype("i4")
        M_out = np.zeros(n_cond).astype("f8")
        exp_M = np.zeros(n_cond).astype("f8")
        exp_N = np.zeros(n_cond).astype("f8")
        halomass_out = np.zeros(int(1e8)).astype("f4")
        halocrd_out = np.zeros(int(3e8)).astype("i4")

        print(f"starting {i} m={m:.2e}", flush=True)
        lib.single_test_sample(
            up(),
            cp(),
            ap(),
            fo(),
            12345,
            n_cond,
            ffi.cast("float *", cond_in.ctypes.data),
            ffi.cast("int *", crd_in.ctypes.data),
            z,
            z_prev,
            ffi.cast("int *", nhalo_out.ctypes.data),
            ffi.cast("int *", N_out.ctypes.data),
            ffi.cast("double *", exp_N.ctypes.data),
            ffi.cast("double *", M_out.ctypes.data),
            ffi.cast("double *", exp_M.ctypes.data),
            ffi.cast("float *", halomass_out.ctypes.data),
            ffi.cast("int *", halocrd_out.ctypes.data),
        )

        # since the tables are reallocated in the test sample function, we redo them here
        lib.initialiseSigmaMInterpTable(edges[0] / 2, edges[-1])

        # get CMF integrals in the same bins
        bin_minima = edges[:-1]
        bin_maxima = edges[1:]
        binned_cmf = np.vectorize(lib.Nhalo_Conditional)(
            growthf,
            np.log(bin_minima),
            np.log(bin_maxima),
            sigma_cond_m[i],
            delta_cond_m[i],
            up.INTEGRATION_METHOD_HALOS,
        )

        hist, _ = np.histogram(halomass_out, edges)
        mf_out = hist / volume_total_m[i] / dlnm
        binned_cmf = binned_cmf * n_cond / volume_total_m[i] / dlnm * cond_mass_m

        np.testing.assert_allclose(N_out.mean(), exp_N[0], rtol=RELATIVE_TOLERANCE)
        np.testing.assert_allclose(M_out.mean(), exp_M[0], rtol=RELATIVE_TOLERANCE)
        np.testing.assert_allclose(mf_out, binned_cmf, rtol=RELATIVE_TOLERANCE)
