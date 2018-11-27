from py21cmmc import wrapper
import numpy as np


# def test_initial_conditions():
#
#     ic = wrapper.initial_conditions(regenerate=True, user_params=wrapper.UserParams(HII_DIM=35))
#     assert len(ic.lowres_density)==ic.user_params.HII_DIM**3
#     assert np.sum(ic.lowres_density) != 0
#
# def test_initial_conditions2():
#     ic = wrapper.initial_conditions(cosmo_params=wrapper.CosmoParams(SIGMA_8=0.8),
#                                     user_params = wrapper.UserParams(HII_DIM=35), regenerate=True)
#     assert len(ic.lowres_density)==ic.user_params.HII_DIM**3
#     assert np.sum(ic.lowres_density) !=0
#
#
# def test_perturb_field_no_ic():
#     pf = wrapper.perturb_field(7.0, regenerate=True, user_params=wrapper.UserParams(HII_DIM=35))
#     assert len(pf.density)==pf.user_params.HII_DIM**3
#     assert np.sum(pf.density) !=0
#
# def test_perturb_field_ic():
#     ic = wrapper.initial_conditions(user_params=wrapper.UserParams(HII_DIM=25))
#     pf = wrapper.perturb_field(7.0, init_boxes=ic)
#     assert len(pf.density)==len(ic.lowres_density)
#     assert pf.cosmo_params == ic.cosmo_params
#     assert pf.user_params == ic.user_params
#     assert np.sum(pf.density) !=0
#
# def test_spin():
#     st = wrapper.spin_temperature(redshift=35.0, user_params=wrapper.UserParams(HII_DIM=25))
#     assert not st.astro_params.INHOMO_RECO
#
# def test_spin2():
#     st = wrapper.spin_temperature(redshift=35.0, user_params=wrapper.UserParams(HII_DIM=25))
#     st2 = wrapper.spin_temperature(redshift=18.0, previous_spin_temp=st,user_params=wrapper.UserParams(HII_DIM=25))
#     assert st.cosmo_params == st2.cosmo_params
#
# def test_spin3():
#     st = wrapper.spin_temperature(redshift=35.0, user_params=wrapper.UserParams(HII_DIM=25))
#     st2 = wrapper.spin_temperature(redshift=16.0, previous_spin_temp=35.0,
#                                    user_params=wrapper.UserParams(HII_DIM=25))
#     assert st.cosmo_params == st2.cosmo_params
#
# def test_ionize():
#     ib = wrapper.ionize_box(redshift=35.0, user_params=wrapper.UserParams(HII_DIM=25))
#     assert not ib.astro_params.INHOMO_RECO
#
# def test_ionize2():
#     ib = wrapper.ionize_box(redshift=18.0, previous_ionize_box=35.0, user_params=wrapper.UserParams(HII_DIM=25))
#     assert not ib.astro_params.INHOMO_RECO



