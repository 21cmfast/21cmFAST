from py21cmfast.c_21cmfast import lib
from . import produce_integration_test_data as prd
import numpy as np
from itertools import repeat

RELATIVE_TOLERANCE = 1e-3

OPTIONS_PS = {
    "EH": [10, {"POWER_SPECTRUM": 0}],
    "BBKS": [10, {"POWER_SPECTRUM": 1}],
    "BE": [10, {"POWER_SPECTRUM": 2}],
    "Peebles": [10, {"POWER_SPECTRUM": 3}],
    "White": [10, {"POWER_SPECTRUM": 4}],
    "CLASS": [10, {"POWER_SPECTRUM": 5}],
}

OPTIONS_HMF = {
    "PS": [10, {"HMF": 0}],
    "ST": [10, {"HMF": 1}],
    "Watson": [10, {"HMF": 2}],
    "Watsonz": [10, {"HMF": 3}],
    "Delos": [10, {"HMF": 4}],
}

options_ps = list(OPTIONS_PS.keys())
options_hmf = list(OPTIONS_HMF.keys())

@pytest.mark.parametrize("name",options_ps)
def test_sigma_table(name):
    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift,kwargs)

    up = opts["user_params"]
    up.update(USE_INTERPOLATION_TABLES=True)
    cp = opts["cosmo_params"]
    lib.Broadcast_struct_global_PS(up(),cp())

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(p21c.global_params.M_MIN_INTEGRAL,p21c.global_params.M_MAX_INTEGRAL)

    mass_range = np.logspace(7,14,num=100)

    sigma_ref = map(lib.sigma_z0,mass_range)
    sigma_table = map(lib.EvaluateSigma,np.log10(mass_range))

    assert np.all((sigma_ref - sigma_table)/sigma_ref < RELATIVE_TOLERANCE)

@pytest.mark.parametrize("name",options_hmf)
def test_Massfunc_conditional_tables(name):
    # redshift, kwargs = OPTIONS_PS[name]
    # opts = prd.get_all_options(redshift,kwargs)

    # up = opts["user_params"]
    # cp = opts["cosmo_params"]
    # ap = opts["astro_params"]
    # fo = opts["flag_options"]
    # up.update(USE_INTERPOLATION_TABLES=True)
    
    # mass_range = np.logspace(7,14,num=100)
    # delta_range = np.linspace(-1,1.49,num=100)

    # lib.Broadcast_struct_global_PS(up(),cp())
    # lib.Broadcast_struct_global_STOC(up(),cp(),ap(),fo())

    # lib.init_ps()
    # lib.initialiseSigmaMInterpTable(p21c.global_params.M_MIN_INTEGRAL,p21c.global_params.M_MAX_INTEGRAL)

    # growthf_1 = lib.dicke(redshift)
    # cell_mass = (cp.cosmo.critical_density(0).to('M_sun Mpc-3').value * (up.BOX_LEN/up.HII_DIM)**3)
    
    # #first the cells
    # lib.initialise_dNdM_tables(delta_range.min(),delta_range.max(),mass_range.min(),mass_range.max(),growthf_1,np.log(cell_mass),False)

    # for m in delta_range:
    #     n = lib.EvaluateRGTable1D

    # assert np.all((sigma_ref - sigma_table)/sigma_ref < RELATIVE_TOLERANCE)
    assert 1

@pytest.mark.parametrize("name",options_hmf)
def test_Fcoll_conditional_table(name):
    assert 1

@pytest.mark.parametrize("name",options_hmf)
def test_Nion_conditional_table(name):
    assert 1

@pytest.mark.parametrize("name",options_hmf)
def test_SFRD_conditional_table(name):
    assert 1