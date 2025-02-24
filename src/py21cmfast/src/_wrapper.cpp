#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <nanobind/stl.h>

namespace nb = nanobind;

extern "C" {
#include "InputParameters.h"
#include "OutputStructs.h"
#include "21cmFAST.h"
#include "BrightnessTemperatureBox.h"
#include "Constants.h"
#include "HaloBox.h"
#include "HaloField.h"
#include "InitialConditions.h"
#include "InputParameters.h"
#include "IonisationBox.h"
#include "LuminosityFunction.h"
#include "OutputStructs.h"
#include "PerturbField.h"
#include "PerturbHaloField.h"
#include "SpinTemperatureBox.h"
#include "Stochasticity.h"
#include "bubble_helper_progs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "dft.h"
#include "elec_interp.h"
#include "exceptions.h"
#include "filtering.h"
#include "heating_helper_progs.h"
#include "hmf.h"
#include "indexing.h"
#include "interp_tables.h"
#include "interpolation.h"
#include "logger.h"
#include "photoncons.h"
#include "recombinations.h"
#include "rng.h"
#include "subcell_rsds.h"
#include "thermochem.h"
}

NB_MODULE(wrapper_module, m) {
  // Bind input parameters

  // Bind CosmoParams
  nb::class_<CosmoParams>(m, "CosmoParams")
      .def_readwrite("SIGMA_8", &CosmoParams::SIGMA_8)
      .def_readwrite("hlittle", &CosmoParams::hlittle)
      .def_readwrite("OMm", &CosmoParams::OMm)
      .def_readwrite("OMl", &CosmoParams::OMl)
      .def_readwrite("OMb", &CosmoParams::OMb)
      .def_readwrite("POWER_INDEX", &CosmoParams::POWER_INDEX);

  // Bind UserParams
  nb::class_<UserParams>(m, "UserParams")
      .def_readwrite("HII_DIM", &UserParams::HII_DIM)
      .def_readwrite("DIM", &UserParams::DIM)
      .def_readwrite("BOX_LEN", &UserParams::BOX_LEN)
      .def_readwrite("NON_CUBIC_FACTOR", &UserParams::NON_CUBIC_FACTOR)
      .def_readwrite("USE_FFTW_WISDOM", &UserParams::USE_FFTW_WISDOM)
      .def_readwrite("HMF", &UserParams::HMF)
      .def_readwrite("USE_RELATIVE_VELOCITIES",
                     &UserParams::USE_RELATIVE_VELOCITIES)
      .def_readwrite("POWER_SPECTRUM", &UserParams::POWER_SPECTRUM)
      .def_readwrite("N_THREADS", &UserParams::N_THREADS)
      .def_readwrite("PERTURB_ON_HIGH_RES", &UserParams::PERTURB_ON_HIGH_RES)
      .def_readwrite("NO_RNG", &UserParams::NO_RNG)
      .def_readwrite("USE_INTERPOLATION_TABLES",
                     &UserParams::USE_INTERPOLATION_TABLES)
      .def_readwrite("INTEGRATION_METHOD_ATOMIC",
                     &UserParams::INTEGRATION_METHOD_ATOMIC)
      .def_readwrite("INTEGRATION_METHOD_MINI",
                     &UserParams::INTEGRATION_METHOD_MINI)
      .def_readwrite("USE_2LPT", &UserParams::USE_2LPT)
      .def_readwrite("MINIMIZE_MEMORY", &UserParams::MINIMIZE_MEMORY)
      .def_readwrite("KEEP_3D_VELOCITIES", &UserParams::KEEP_3D_VELOCITIES)
      .def_readwrite("SAMPLER_MIN_MASS", &UserParams::SAMPLER_MIN_MASS)
      .def_readwrite("SAMPLER_BUFFER_FACTOR",
                     &UserParams::SAMPLER_BUFFER_FACTOR)
      .def_readwrite("MAXHALO_FACTOR", &UserParams::MAXHALO_FACTOR)
      .def_readwrite("N_COND_INTERP", &UserParams::N_COND_INTERP)
      .def_readwrite("N_PROB_INTERP", &UserParams::N_PROB_INTERP)
      .def_readwrite("MIN_LOGPROB", &UserParams::MIN_LOGPROB)
      .def_readwrite("SAMPLE_METHOD", &UserParams::SAMPLE_METHOD)
      .def_readwrite("AVG_BELOW_SAMPLER", &UserParams::AVG_BELOW_SAMPLER)
      .def_readwrite("HALOMASS_CORRECTION", &UserParams::HALOMASS_CORRECTION)
      .def_readwrite("PARKINSON_G0", &UserParams::PARKINSON_G0)
      .def_readwrite("PARKINSON_y1", &UserParams::PARKINSON_y1)
      .def_readwrite("PARKINSON_y2", &UserParams::PARKINSON_y2);

  // Bind AstroParams
  nb::class_<AstroParams>(m, "AstroParams")
      .def_readwrite("HII_EFF_FACTOR", &AstroParams::HII_EFF_FACTOR)
      .def_readwrite("F_STAR10", &AstroParams::F_STAR10)
      .def_readwrite("ALPHA_STAR", &AstroParams::ALPHA_STAR)
      .def_readwrite("ALPHA_STAR_MINI", &AstroParams::ALPHA_STAR_MINI)
      .def_readwrite("SIGMA_STAR", &AstroParams::SIGMA_STAR)
      .def_readwrite("CORR_STAR", &AstroParams::CORR_STAR)
      .def_readwrite("UPPER_STELLAR_TURNOVER_MASS",
                     &AstroParams::UPPER_STELLAR_TURNOVER_MASS)
      .def_readwrite("UPPER_STELLAR_TURNOVER_INDEX",
                     &AstroParams::UPPER_STELLAR_TURNOVER_INDEX)
      .def_readwrite("F_STAR7_MINI", &AstroParams::F_STAR7_MINI)
      .def_readwrite("t_STAR", &AstroParams::t_STAR)
      .def_readwrite("CORR_SFR", &AstroParams::CORR_SFR)
      .def_readwrite("SIGMA_SFR_INDEX", &AstroParams::SIGMA_SFR_INDEX)
      .def_readwrite("SIGMA_SFR_LIM", &AstroParams::SIGMA_SFR_LIM)
      .def_readwrite("L_X", &AstroParams::L_X)
      .def_readwrite("L_X_MINI", &AstroParams::L_X_MINI)
      .def_readwrite("SIGMA_LX", &AstroParams::SIGMA_LX)
      .def_readwrite("CORR_LX", &AstroParams::CORR_LX)
      .def_readwrite("F_ESC10", &AstroParams::F_ESC10)
      .def_readwrite("ALPHA_ESC", &AstroParams::ALPHA_ESC)
      .def_readwrite("F_ESC7_MINI", &AstroParams::F_ESC7_MINI)
      .def_readwrite("M_TURN", &AstroParams::M_TURN)
      .def_readwrite("R_BUBBLE_MAX", &AstroParams::R_BUBBLE_MAX)
      .def_readwrite("ION_Tvir_MIN", &AstroParams::ION_Tvir_MIN);

  // Bind FlagOptions
  nb::class_<FlagOptions>(m, "FlagOptions")
      .def_readwrite("USE_HALO_FIELD", &FlagOptions::USE_HALO_FIELD)
      .def_readwrite("USE_MINI_HALOS", &FlagOptions::USE_MINI_HALOS)
      .def_readwrite("USE_CMB_HEATING", &FlagOptions::USE_CMB_HEATING)
      .def_readwrite("USE_LYA_HEATING", &FlagOptions::USE_LYA_HEATING)
      .def_readwrite("USE_MASS_DEPENDENT_ZETA",
                     &FlagOptions::USE_MASS_DEPENDENT_ZETA)
      .def_readwrite("SUBCELL_RSD", &FlagOptions::SUBCELL_RSD)
      .def_readwrite("APPLY_RSDS", &FlagOptions::APPLY_RSDS)
      .def_readwrite("INHOMO_RECO", &FlagOptions::INHOMO_RECO)
      .def_readwrite("USE_TS_FLUCT", &FlagOptions::USE_TS_FLUCT)
      .def_readwrite("M_MIN_in_Mass", &FlagOptions::M_MIN_in_Mass)
      .def_readwrite("FIX_VCB_AVG", &FlagOptions::FIX_VCB_AVG)
      .def_readwrite("HALO_STOCHASTICITY", &FlagOptions::HALO_STOCHASTICITY)
      .def_readwrite("USE_EXP_FILTER", &FlagOptions::USE_EXP_FILTER)
      .def_readwrite("FIXED_HALO_GRIDS", &FlagOptions::FIXED_HALO_GRIDS)
      .def_readwrite("CELL_RECOMB", &FlagOptions::CELL_RECOMB)
      .def_readwrite("PHOTON_CONS_TYPE", &FlagOptions::PHOTON_CONS_TYPE)
      .def_readwrite("USE_UPPER_STELLAR_TURNOVER",
                     &FlagOptions::USE_UPPER_STELLAR_TURNOVER)
      .def_readwrite("HALO_SCALING_RELATIONS_MEDIAN",
                     &FlagOptions::HALO_SCALING_RELATIONS_MEDIAN);

  nb::class_<GlobalParams>(m, "GlobalParams")
      .def_readwrite("ALPHA_UVB", &GlobalParams::ALPHA_UVB)
      .def_readwrite("EVOLVE_DENSITY_LINEARLY",
                     &GlobalParams::EVOLVE_DENSITY_LINEARLY)
      .def_readwrite("SMOOTH_EVOLVED_DENSITY_FIELD",
                     &GlobalParams::SMOOTH_EVOLVED_DENSITY_FIELD)
      .def_readwrite("R_smooth_density", &GlobalParams::R_smooth_density)
      .def_readwrite("HII_ROUND_ERR", &GlobalParams::HII_ROUND_ERR)
      .def_readwrite("FIND_BUBBLE_ALGORITHM",
                     &GlobalParams::FIND_BUBBLE_ALGORITHM)
      .def_readwrite("N_POISSON", &GlobalParams::N_POISSON)
      .def_readwrite("T_USE_VELOCITIES", &GlobalParams::T_USE_VELOCITIES)
      .def_readwrite("MAX_DVDR", &GlobalParams::MAX_DVDR)
      .def_readwrite("DELTA_R_HII_FACTOR", &GlobalParams::DELTA_R_HII_FACTOR)
      .def_readwrite("DELTA_R_FACTOR", &GlobalParams::DELTA_R_FACTOR)
      .def_readwrite("HII_FILTER", &GlobalParams::HII_FILTER)
      .def_readwrite("INITIAL_REDSHIFT", &GlobalParams::INITIAL_REDSHIFT)
      .def_readwrite("R_OVERLAP_FACTOR", &GlobalParams::R_OVERLAP_FACTOR)
      .def_readwrite("DELTA_CRIT_MODE", &GlobalParams::DELTA_CRIT_MODE)
      .def_readwrite("HALO_FILTER", &GlobalParams::HALO_FILTER)
      .def_readwrite("OPTIMIZE", &GlobalParams::OPTIMIZE)
      .def_readwrite("OPTIMIZE_MIN_MASS", &GlobalParams::OPTIMIZE_MIN_MASS)
      .def_readwrite("CRIT_DENS_TRANSITION",
                     &GlobalParams::CRIT_DENS_TRANSITION)
      .def_readwrite("MIN_DENSITY_LOW_LIMIT",
                     &GlobalParams::MIN_DENSITY_LOW_LIMIT)
      .def_readwrite("RecombPhotonCons", &GlobalParams::RecombPhotonCons)
      .def_readwrite("PhotonConsStart", &GlobalParams::PhotonConsStart)
      .def_readwrite("PhotonConsEnd", &GlobalParams::PhotonConsEnd)
      .def_readwrite("PhotonConsAsymptoteTo",
                     &GlobalParams::PhotonConsAsymptoteTo)
      .def_readwrite("PhotonConsEndCalibz", &GlobalParams::PhotonConsEndCalibz)
      .def_readwrite("PhotonConsSmoothing", &GlobalParams::PhotonConsSmoothing)
      .def_readwrite("HEAT_FILTER", &GlobalParams::HEAT_FILTER)
      .def_readwrite("CLUMPING_FACTOR", &GlobalParams::CLUMPING_FACTOR)
      .def_readwrite("Z_HEAT_MAX", &GlobalParams::Z_HEAT_MAX)
      .def_readwrite("R_XLy_MAX", &GlobalParams::R_XLy_MAX)
      .def_readwrite("NUM_FILTER_STEPS_FOR_Ts",
                     &GlobalParams::NUM_FILTER_STEPS_FOR_Ts)
      .def_readwrite("ZPRIME_STEP_FACTOR", &GlobalParams::ZPRIME_STEP_FACTOR)
      .def_readwrite("TK_at_Z_HEAT_MAX", &GlobalParams::TK_at_Z_HEAT_MAX)
      .def_readwrite("XION_at_Z_HEAT_MAX", &GlobalParams::XION_at_Z_HEAT_MAX)
      .def_readwrite("Pop", &GlobalParams::Pop)
      .def_readwrite("Pop2_ion", &GlobalParams::Pop2_ion)
      .def_readwrite("Pop3_ion", &GlobalParams::Pop3_ion)
      .def_readwrite("NU_X_BAND_MAX", &GlobalParams::NU_X_BAND_MAX)
      .def_readwrite("NU_X_MAX", &GlobalParams::NU_X_MAX)
      .def_readwrite("NBINS_LF", &GlobalParams::NBINS_LF)
      .def_readwrite("P_CUTOFF", &GlobalParams::P_CUTOFF)
      .def_readwrite("M_WDM", &GlobalParams::M_WDM)
      .def_readwrite("g_x", &GlobalParams::g_x)
      .def_readwrite("OMn", &GlobalParams::OMn)
      .def_readwrite("OMk", &GlobalParams::OMk)
      .def_readwrite("OMr", &GlobalParams::OMr)
      .def_readwrite("OMtot", &GlobalParams::OMtot)
      .def_readwrite("Y_He", &GlobalParams::Y_He)
      .def_readwrite("wl", &GlobalParams::wl)
      .def_readwrite("SHETH_b", &GlobalParams::SHETH_b)
      .def_readwrite("SHETH_c", &GlobalParams::SHETH_c)
      .def_readwrite("Zreion_HeII", &GlobalParams::Zreion_HeII)
      .def_readwrite("FILTER", &GlobalParams::FILTER)
      .def_readwrite("external_table_path", &GlobalParams::external_table_path)
      .def_readwrite("wisdoms_path", &GlobalParams::wisdoms_path)
      .def_readwrite("R_BUBBLE_MIN", &GlobalParams::R_BUBBLE_MIN)
      .def_readwrite("M_MIN_INTEGRAL", &GlobalParams::M_MIN_INTEGRAL)
      .def_readwrite("M_MAX_INTEGRAL", &GlobalParams::M_MAX_INTEGRAL)
      .def_readwrite("T_RE", &GlobalParams::T_RE)
      .def_readwrite("VAVG", &GlobalParams::VAVG)
      .def_readwrite("USE_ADIABATIC_FLUCTUATIONS",
                     &GlobalParams::USE_ADIABATIC_FLUCTUATIONS);

  // Bind output parameters
  nb::class_<InitialConditions>(m, "InitialConditions")
      .def_readwrite("lowres_density", &InitialConditions::lowres_density)
      .def_readwrite("lowres_vx", &InitialConditions::lowres_vx)
      .def_readwrite("lowres_vy", &InitialConditions::lowres_vy)
      .def_readwrite("lowres_vz", &InitialConditions::lowres_vz)
      .def_readwrite("lowres_vx_2LPT", &InitialConditions::lowres_vx_2LPT)
      .def_readwrite("lowres_vy_2LPT", &InitialConditions::lowres_vy_2LPT)
      .def_readwrite("lowres_vz_2LPT", &InitialConditions::lowres_vz_2LPT)
      .def_readwrite("hires_density", &InitialConditions::hires_density)
      .def_readwrite("hires_vx", &InitialConditions::hires_vx)
      .def_readwrite("hires_vy", &InitialConditions::hires_vy)
      .def_readwrite("hires_vz", &InitialConditions::hires_vz)
      .def_readwrite("hires_vx_2LPT", &InitialConditions::hires_vx_2LPT)
      .def_readwrite("hires_vy_2LPT", &InitialConditions::hires_vy_2LPT)
      .def_readwrite("hires_vz_2LPT", &InitialConditions::hires_vz_2LPT)
      .def_readwrite("lowres_vcb", &InitialConditions::lowres_vcb);

  nb::class_<PerturbedField>(m, "PerturbedField")
      .def_readwrite("density", &PerturbedField::density)
      .def_readwrite("velocity_x", &PerturbedField::velocity_x)
      .def_readwrite("velocity_y", &PerturbedField::velocity_y)
      .def_readwrite("velocity_z", &PerturbedField::velocity_z);

  nb::class_<HaloField>(m, "HaloField")
      .def_readwrite("n_halos", &HaloField::n_halos)
      .def_readwrite("buffer_size", &HaloField::buffer_size)
      .def_readwrite("halo_masses", &HaloField::halo_masses)
      .def_readwrite("halo_coords", &HaloField::halo_coords)
      .def_readwrite("star_rng", &HaloField::star_rng)
      .def_readwrite("sfr_rng", &HaloField::sfr_rng)
      .def_readwrite("xray_rng", &HaloField::xray_rng);

  nb::class_<PerturbHaloField>(m, "PerturbHaloField")
      .def_readwrite("n_halos", &PerturbHaloField::n_halos)
      .def_readwrite("buffer_size", &PerturbHaloField::buffer_size)
      .def_readwrite("halo_masses", &PerturbHaloField::halo_masses)
      .def_readwrite("halo_coords", &PerturbHaloField::halo_coords)
      .def_readwrite("star_rng", &PerturbHaloField::star_rng)
      .def_readwrite("sfr_rng", &PerturbHaloField::sfr_rng)
      .def_readwrite("xray_rng", &PerturbHaloField::xray_rng);

  nb::class_<HaloBox>(m, "HaloBox")
      .def_readwrite("halo_mass", &HaloBox::halo_mass)
      .def_readwrite("halo_stars", &HaloBox::halo_stars)
      .def_readwrite("halo_stars_mini", &HaloBox::halo_stars_mini)
      .def_readwrite("count", &HaloBox::count)
      .def_readwrite("n_ion", &HaloBox::n_ion)
      .def_readwrite("halo_sfr", &HaloBox::halo_sfr)
      .def_readwrite("halo_xray", &HaloBox::halo_xray)
      .def_readwrite("halo_sfr_mini", &HaloBox::halo_sfr_mini)
      .def_readwrite("whalo_sfr", &HaloBox::whalo_sfr)
      .def_readwrite("log10_Mcrit_ACG_ave", &HaloBox::log10_Mcrit_ACG_ave)
      .def_readwrite("log10_Mcrit_MCG_ave", &HaloBox::log10_Mcrit_MCG_ave);

  nb::class_<XraySourceBox>(m, "XraySourceBox")
      .def_readwrite("filtered_sfr", &XraySourceBox::filtered_sfr)
      .def_readwrite("filtered_xray", &XraySourceBox::filtered_xray)
      .def_readwrite("filtered_sfr_mini", &XraySourceBox::filtered_sfr_mini)
      .def_readwrite("mean_log10_Mcrit_LW", &XraySourceBox::mean_log10_Mcrit_LW)
      .def_readwrite("mean_sfr", &XraySourceBox::mean_sfr)
      .def_readwrite("mean_sfr_mini", &XraySourceBox::mean_sfr_mini);

  nb::class_<TsBox>(m, "TsBox")
      .def_readwrite("Ts_box", &TsBox::Ts_box)
      .def_readwrite("x_e_box", &TsBox::x_e_box)
      .def_readwrite("Tk_box", &TsBox::Tk_box)
      .def_readwrite("J_21_LW_box", &TsBox::J_21_LW_box);

  nb::class_<IonizedBox>(m, "IonizedBox")
      .def_readwrite("mean_f_coll", &IonizedBox::mean_f_coll)
      .def_readwrite("mean_f_coll_MINI", &IonizedBox::mean_f_coll_MINI)
      .def_readwrite("log10_Mturnover_ave", &IonizedBox::log10_Mturnover_ave)
      .def_readwrite("log10_Mturnover_MINI_ave",
                     &IonizedBox::log10_Mturnover_MINI_ave)
      .def_readwrite("xH_box", &IonizedBox::xH_box)
      .def_readwrite("Gamma12_box", &IonizedBox::Gamma12_box)
      .def_readwrite("MFP_box", &IonizedBox::MFP_box)
      .def_readwrite("z_re_box", &IonizedBox::z_re_box)
      .def_readwrite("dNrec_box", &IonizedBox::dNrec_box)
      .def_readwrite("temp_kinetic_all_gas", &IonizedBox::temp_kinetic_all_gas)
      .def_readwrite("Fcoll", &IonizedBox::Fcoll)
      .def_readwrite("Fcoll_MINI", &IonizedBox::Fcoll_MINI);

  nb::class_<BrightnessTemp>(m, "BrightnessTemp ")
      .def_readwrite("brightness_temp", &BrightnessTemp::brightness_temp);

  // Bind functions
  m.def("ComputeInitialConditions", &ComputeInitialConditions);
  m.def("ComputePerturbField", &ComputePerturbField);
  m.def("ComputeHaloField", &ComputeHaloField);
  m.def("ComputePerturbHaloField", &ComputePerturbHaloField);
  m.def("ComputeTsBox", &ComputeTsBox);
  m.def("ComputeIonizedBox", &ComputeIonizedBox);
  m.def("ComputeBrightnessTemp", &ComputeBrightnessTemp);
  m.def("ComputeHaloBox", &ComputeHaloBox);
  m.def("UpdateXraySourceBox", &UpdateXraySourceBox);
  m.def("InitialisePhotonCons", &InitialisePhotonCons);
  m.def("PhotonCons_Calibration", &PhotonCons_Calibration);
  m.def("ComputeZstart_PhotonCons", &ComputeZstart_PhotonCons);
  m.def("adjust_redshifts_for_photoncons", &adjust_redshifts_for_photoncons);
  m.def("determine_deltaz_for_photoncons", &determine_deltaz_for_photoncons);
  m.def("ObtainPhotonConsData", &ObtainPhotonConsData);
  m.def("FreePhotonConsMemory", &FreePhotonConsMemory);
  m.def(
      "photon_cons_allocated", []() -> bool { return photon_cons_allocated; },
      "Returns whether photon conservation memory is allocated");
  m.def("set_alphacons_params", &set_alphacons_params);
  m.def("ComputeLF", &ComputeLF);
  m.def("ComputeTau", &ComputeTau);
  m.def("init_ps", &init_ps);
  m.def("init_heat", &init_heat);
  m.def("CreateFFTWWisdoms", &CreateFFTWWisdoms);
  m.def("Broadcast_struct_global_noastro", &Broadcast_struct_global_noastro);
  m.def("Broadcast_struct_global_all", &Broadcast_struct_global_all);
  m.def("initialiseSigmaMInterpTable", &initialiseSigmaMInterpTable);
  m.def("initialise_SFRD_spline", &initialise_SFRD_spline);
  m.def("initialise_Nion_Ts_spline", &initialise_Nion_Ts_spline);
  m.def("initialise_FgtrM_delta_table", &initialise_FgtrM_delta_table);
  m.def("init_FcollTable", &init_FcollTable);
  m.def("initialise_Nion_Conditional_spline",
        &initialise_Nion_Conditional_spline);
  m.def("initialise_SFRD_Conditional_table",
        &initialise_SFRD_Conditional_table);
  m.def("initialise_dNdM_tables", &initialise_dNdM_tables);
  m.def("initialise_dNdM_inverse_table", &initialise_dNdM_inverse_table);
  m.def("EvaluateNionTs", &EvaluateNionTs);
  m.def("EvaluateNionTs_MINI", &EvaluateNionTs_MINI);
  m.def("EvaluateSFRD", &EvaluateSFRD);
  m.def("EvaluateSFRD_MINI", &EvaluateSFRD_MINI);
  m.def("EvaluateSFRD_Conditional", &EvaluateSFRD_Conditional);
  m.def("EvaluateSFRD_Conditional_MINI", &EvaluateSFRD_Conditional_MINI);
  m.def("EvaluateNion_Conditional", &EvaluateNion_Conditional);
  m.def("EvaluateNion_Conditional_MINI", &EvaluateNion_Conditional_MINI);
  m.def("EvaluateNhalo", &EvaluateNhalo);
  m.def("EvaluateMcoll", &EvaluateMcoll);
  m.def("EvaluateNhaloInv", &EvaluateNhaloInv);
  m.def("EvaluateFcoll_delta", &EvaluateFcoll_delta);
  m.def("EvaluatedFcolldz", &EvaluatedFcolldz);
  m.def("EvaluateSigma", &EvaluateSigma);
  m.def("EvaluatedSigmasqdm", &EvaluatedSigmasqdm);
  m.def("initialise_GL", &initialise_GL);
  m.def("Nhalo_Conditional", &Nhalo_Conditional);
  m.def("Mcoll_Conditional", &Mcoll_Conditional);
  m.def("Nion_ConditionalM", &Nion_ConditionalM);
  m.def("Nion_ConditionalM_MINI", &Nion_ConditionalM_MINI);
  m.def("Nion_General", &Nion_General);
  m.def("Nion_General_MINI", &Nion_General_MINI);
  m.def("Fcoll_General", &Fcoll_General);
  m.def("unconditional_mf", &unconditional_mf);
  m.def("conditional_mf", &conditional_mf);
  m.def("SomethingThatCatches", &SomethingThatCatches);
  m.def("FunctionThatCatches", &FunctionThatCatches);
  m.def("FunctionThatThrows", &FunctionThatThrows);
  m.def("single_test_sample", &single_test_sample);
  m.def("test_halo_props", &test_halo_props);
  m.def("test_filter", &test_filter);
  m.def("dicke", &dicke);
  m.def("sigma_z0", &sigma_z0);
  m.def("dsigmasqdm_z0", &dsigmasqdm_z0);
  m.def("get_delta_crit", &get_delta_crit);
  m.def("atomic_cooling_threshold", &atomic_cooling_threshold);
  m.def("expected_nhalo", &expected_nhalo);
}
