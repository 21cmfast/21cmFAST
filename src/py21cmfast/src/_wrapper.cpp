#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "InputParameters.h"
// #include <nanobind/stl.h>

namespace nb = nanobind;

extern "C" {
#include "21cmFAST.h"
#include "Constants.h"
#include "indexing.h"
}

NB_MODULE(c_21cmfast, m) {
    m.doc() = "This is the docstring for the 21cmFAST Python extension.";

    // Bind input parameters

    // Bind CosmoParams
    nb::class_<CosmoParams>(m, "CosmoParams")
        .def(nb::init<>())
        .def_rw("SIGMA_8", &CosmoParams::SIGMA_8)
        .def_rw("hlittle", &CosmoParams::hlittle)
        .def_rw("OMm", &CosmoParams::OMm)
        .def_rw("OMl", &CosmoParams::OMl)
        .def_rw("OMb", &CosmoParams::OMb)
        .def_rw("POWER_INDEX", &CosmoParams::POWER_INDEX);

    // Bind UserParams
    nb::class_<UserParams>(m, "UserParams")
        .def(nb::init<>())
        .def_rw("HII_DIM", &UserParams::HII_DIM)
        .def_rw("DIM", &UserParams::DIM)
        .def_rw("BOX_LEN", &UserParams::BOX_LEN)
        .def_rw("NON_CUBIC_FACTOR", &UserParams::NON_CUBIC_FACTOR)
        .def_rw("USE_FFTW_WISDOM", &UserParams::USE_FFTW_WISDOM)
        .def_rw("HMF", &UserParams::HMF)
        .def_rw("USE_RELATIVE_VELOCITIES", &UserParams::USE_RELATIVE_VELOCITIES)
        .def_rw("POWER_SPECTRUM", &UserParams::POWER_SPECTRUM)
        .def_rw("N_THREADS", &UserParams::N_THREADS)
        .def_rw("PERTURB_ON_HIGH_RES", &UserParams::PERTURB_ON_HIGH_RES)
        .def_rw("NO_RNG", &UserParams::NO_RNG)
        .def_rw("USE_INTERPOLATION_TABLES", &UserParams::USE_INTERPOLATION_TABLES)
        .def_rw("INTEGRATION_METHOD_ATOMIC", &UserParams::INTEGRATION_METHOD_ATOMIC)
        .def_rw("INTEGRATION_METHOD_MINI", &UserParams::INTEGRATION_METHOD_MINI)
        .def_rw("USE_2LPT", &UserParams::USE_2LPT)
        .def_rw("MINIMIZE_MEMORY", &UserParams::MINIMIZE_MEMORY)
        .def_rw("KEEP_3D_VELOCITIES", &UserParams::KEEP_3D_VELOCITIES)
        .def_rw("SAMPLER_MIN_MASS", &UserParams::SAMPLER_MIN_MASS)
        .def_rw("SAMPLER_BUFFER_FACTOR", &UserParams::SAMPLER_BUFFER_FACTOR)
        .def_rw("MAXHALO_FACTOR", &UserParams::MAXHALO_FACTOR)
        .def_rw("N_COND_INTERP", &UserParams::N_COND_INTERP)
        .def_rw("N_PROB_INTERP", &UserParams::N_PROB_INTERP)
        .def_rw("MIN_LOGPROB", &UserParams::MIN_LOGPROB)
        .def_rw("SAMPLE_METHOD", &UserParams::SAMPLE_METHOD)
        .def_rw("AVG_BELOW_SAMPLER", &UserParams::AVG_BELOW_SAMPLER)
        .def_rw("HALOMASS_CORRECTION", &UserParams::HALOMASS_CORRECTION)
        .def_rw("PARKINSON_G0", &UserParams::PARKINSON_G0)
        .def_rw("PARKINSON_y1", &UserParams::PARKINSON_y1)
        .def_rw("PARKINSON_y2", &UserParams::PARKINSON_y2);

    // Bind AstroParams
    nb::class_<AstroParams>(m, "AstroParams")
        .def(nb::init<>())
        .def_rw("HII_EFF_FACTOR", &AstroParams::HII_EFF_FACTOR)
        .def_rw("F_STAR10", &AstroParams::F_STAR10)
        .def_rw("ALPHA_STAR", &AstroParams::ALPHA_STAR)
        .def_rw("ALPHA_STAR_MINI", &AstroParams::ALPHA_STAR_MINI)
        .def_rw("SIGMA_STAR", &AstroParams::SIGMA_STAR)
        .def_rw("CORR_STAR", &AstroParams::CORR_STAR)
        .def_rw("UPPER_STELLAR_TURNOVER_MASS", &AstroParams::UPPER_STELLAR_TURNOVER_MASS)
        .def_rw("UPPER_STELLAR_TURNOVER_INDEX", &AstroParams::UPPER_STELLAR_TURNOVER_INDEX)
        .def_rw("F_STAR7_MINI", &AstroParams::F_STAR7_MINI)
        .def_rw("t_STAR", &AstroParams::t_STAR)
        .def_rw("CORR_SFR", &AstroParams::CORR_SFR)
        .def_rw("SIGMA_SFR_INDEX", &AstroParams::SIGMA_SFR_INDEX)
        .def_rw("SIGMA_SFR_LIM", &AstroParams::SIGMA_SFR_LIM)
        .def_rw("L_X", &AstroParams::L_X)
        .def_rw("L_X_MINI", &AstroParams::L_X_MINI)
        .def_rw("SIGMA_LX", &AstroParams::SIGMA_LX)
        .def_rw("CORR_LX", &AstroParams::CORR_LX)
        .def_rw("F_ESC10", &AstroParams::F_ESC10)
        .def_rw("ALPHA_ESC", &AstroParams::ALPHA_ESC)
        .def_rw("F_ESC7_MINI", &AstroParams::F_ESC7_MINI)
        .def_rw("M_TURN", &AstroParams::M_TURN)
        .def_rw("R_BUBBLE_MAX", &AstroParams::R_BUBBLE_MAX)
        .def_rw("ION_Tvir_MIN", &AstroParams::ION_Tvir_MIN);

    // Bind FlagOptions
    nb::class_<FlagOptions>(m, "FlagOptions")
        .def(nb::init<>())
        .def_rw("USE_HALO_FIELD", &FlagOptions::USE_HALO_FIELD)
        .def_rw("USE_MINI_HALOS", &FlagOptions::USE_MINI_HALOS)
        .def_rw("USE_CMB_HEATING", &FlagOptions::USE_CMB_HEATING)
        .def_rw("USE_LYA_HEATING", &FlagOptions::USE_LYA_HEATING)
        .def_rw("USE_MASS_DEPENDENT_ZETA", &FlagOptions::USE_MASS_DEPENDENT_ZETA)
        .def_rw("SUBCELL_RSD", &FlagOptions::SUBCELL_RSD)
        .def_rw("APPLY_RSDS", &FlagOptions::APPLY_RSDS)
        .def_rw("INHOMO_RECO", &FlagOptions::INHOMO_RECO)
        .def_rw("USE_TS_FLUCT", &FlagOptions::USE_TS_FLUCT)
        .def_rw("M_MIN_in_Mass", &FlagOptions::M_MIN_in_Mass)
        .def_rw("FIX_VCB_AVG", &FlagOptions::FIX_VCB_AVG)
        .def_rw("HALO_STOCHASTICITY", &FlagOptions::HALO_STOCHASTICITY)
        .def_rw("USE_EXP_FILTER", &FlagOptions::USE_EXP_FILTER)
        .def_rw("FIXED_HALO_GRIDS", &FlagOptions::FIXED_HALO_GRIDS)
        .def_rw("CELL_RECOMB", &FlagOptions::CELL_RECOMB)
        .def_rw("PHOTON_CONS_TYPE", &FlagOptions::PHOTON_CONS_TYPE)
        .def_rw("USE_UPPER_STELLAR_TURNOVER", &FlagOptions::USE_UPPER_STELLAR_TURNOVER)
        .def_rw("HALO_SCALING_RELATIONS_MEDIAN", &FlagOptions::HALO_SCALING_RELATIONS_MEDIAN);

    nb::class_<GlobalParams>(m, "GlobalParams")
        .def(nb::init<>())
        .def_rw("ALPHA_UVB", &GlobalParams::ALPHA_UVB)
        .def_rw("EVOLVE_DENSITY_LINEARLY", &GlobalParams::EVOLVE_DENSITY_LINEARLY)
        .def_rw("SMOOTH_EVOLVED_DENSITY_FIELD", &GlobalParams::SMOOTH_EVOLVED_DENSITY_FIELD)
        .def_rw("R_smooth_density", &GlobalParams::R_smooth_density)
        .def_rw("HII_ROUND_ERR", &GlobalParams::HII_ROUND_ERR)
        .def_rw("FIND_BUBBLE_ALGORITHM", &GlobalParams::FIND_BUBBLE_ALGORITHM)
        .def_rw("N_POISSON", &GlobalParams::N_POISSON)
        .def_rw("T_USE_VELOCITIES", &GlobalParams::T_USE_VELOCITIES)
        .def_rw("MAX_DVDR", &GlobalParams::MAX_DVDR)
        .def_rw("DELTA_R_HII_FACTOR", &GlobalParams::DELTA_R_HII_FACTOR)
        .def_rw("DELTA_R_FACTOR", &GlobalParams::DELTA_R_FACTOR)
        .def_rw("HII_FILTER", &GlobalParams::HII_FILTER)
        .def_rw("INITIAL_REDSHIFT", &GlobalParams::INITIAL_REDSHIFT)
        .def_rw("R_OVERLAP_FACTOR", &GlobalParams::R_OVERLAP_FACTOR)
        .def_rw("DELTA_CRIT_MODE", &GlobalParams::DELTA_CRIT_MODE)
        .def_rw("HALO_FILTER", &GlobalParams::HALO_FILTER)
        .def_rw("OPTIMIZE", &GlobalParams::OPTIMIZE)
        .def_rw("OPTIMIZE_MIN_MASS", &GlobalParams::OPTIMIZE_MIN_MASS)
        .def_rw("CRIT_DENS_TRANSITION", &GlobalParams::CRIT_DENS_TRANSITION)
        .def_rw("MIN_DENSITY_LOW_LIMIT", &GlobalParams::MIN_DENSITY_LOW_LIMIT)
        .def_rw("RecombPhotonCons", &GlobalParams::RecombPhotonCons)
        .def_rw("PhotonConsStart", &GlobalParams::PhotonConsStart)
        .def_rw("PhotonConsEnd", &GlobalParams::PhotonConsEnd)
        .def_rw("PhotonConsAsymptoteTo", &GlobalParams::PhotonConsAsymptoteTo)
        .def_rw("PhotonConsEndCalibz", &GlobalParams::PhotonConsEndCalibz)
        .def_rw("PhotonConsSmoothing", &GlobalParams::PhotonConsSmoothing)
        .def_rw("HEAT_FILTER", &GlobalParams::HEAT_FILTER)
        .def_rw("CLUMPING_FACTOR", &GlobalParams::CLUMPING_FACTOR)
        .def_rw("Z_HEAT_MAX", &GlobalParams::Z_HEAT_MAX)
        .def_rw("R_XLy_MAX", &GlobalParams::R_XLy_MAX)
        .def_rw("NUM_FILTER_STEPS_FOR_Ts", &GlobalParams::NUM_FILTER_STEPS_FOR_Ts)
        .def_rw("ZPRIME_STEP_FACTOR", &GlobalParams::ZPRIME_STEP_FACTOR)
        .def_rw("TK_at_Z_HEAT_MAX", &GlobalParams::TK_at_Z_HEAT_MAX)
        .def_rw("XION_at_Z_HEAT_MAX", &GlobalParams::XION_at_Z_HEAT_MAX)
        .def_rw("Pop", &GlobalParams::Pop)
        .def_rw("Pop2_ion", &GlobalParams::Pop2_ion)
        .def_rw("Pop3_ion", &GlobalParams::Pop3_ion)
        .def_rw("NU_X_BAND_MAX", &GlobalParams::NU_X_BAND_MAX)
        .def_rw("NU_X_MAX", &GlobalParams::NU_X_MAX)
        .def_rw("NBINS_LF", &GlobalParams::NBINS_LF)
        .def_rw("P_CUTOFF", &GlobalParams::P_CUTOFF)
        .def_rw("M_WDM", &GlobalParams::M_WDM)
        .def_rw("g_x", &GlobalParams::g_x)
        .def_rw("OMn", &GlobalParams::OMn)
        .def_rw("OMk", &GlobalParams::OMk)
        .def_rw("OMr", &GlobalParams::OMr)
        .def_rw("OMtot", &GlobalParams::OMtot)
        .def_rw("Y_He", &GlobalParams::Y_He)
        .def_rw("wl", &GlobalParams::wl)
        .def_rw("SHETH_b", &GlobalParams::SHETH_b)
        .def_rw("SHETH_c", &GlobalParams::SHETH_c)
        .def_rw("Zreion_HeII", &GlobalParams::Zreion_HeII)
        .def_rw("FILTER", &GlobalParams::FILTER)
        .def_ro("external_table_path", &GlobalParams::external_table_path)
        .def_ro("wisdoms_path", &GlobalParams::wisdoms_path)
        .def_rw("R_BUBBLE_MIN", &GlobalParams::R_BUBBLE_MIN)
        .def_rw("M_MIN_INTEGRAL", &GlobalParams::M_MIN_INTEGRAL)
        .def_rw("M_MAX_INTEGRAL", &GlobalParams::M_MAX_INTEGRAL)
        .def_rw("T_RE", &GlobalParams::T_RE)
        .def_rw("VAVG", &GlobalParams::VAVG)
        .def_rw("USE_ADIABATIC_FLUCTUATIONS", &GlobalParams::USE_ADIABATIC_FLUCTUATIONS)
        .def("set_external_table_path", &set_external_table_path)
        .def("get_external_table_path", &get_external_table_path)
        .def("set_wisdoms_path", &set_wisdoms_path)
        .def("get_wisdoms_path", &get_wisdoms_path);

    m.def(
        "get_global_params", []() -> GlobalParams& { return global_params; },
        nb::rv_policy::reference);

    // Bind output parameters
    nb::class_<InitialConditions>(m, "InitialConditions")
        .def(nb::init<>())
        .def_rw("lowres_density", &InitialConditions::lowres_density)
        .def_rw("lowres_vx", &InitialConditions::lowres_vx)
        .def_rw("lowres_vy", &InitialConditions::lowres_vy)
        .def_rw("lowres_vz", &InitialConditions::lowres_vz)
        .def_rw("lowres_vx_2LPT", &InitialConditions::lowres_vx_2LPT)
        .def_rw("lowres_vy_2LPT", &InitialConditions::lowres_vy_2LPT)
        .def_rw("lowres_vz_2LPT", &InitialConditions::lowres_vz_2LPT)
        .def_rw("hires_density", &InitialConditions::hires_density)
        .def_rw("hires_vx", &InitialConditions::hires_vx)
        .def_rw("hires_vy", &InitialConditions::hires_vy)
        .def_rw("hires_vz", &InitialConditions::hires_vz)
        .def_rw("hires_vx_2LPT", &InitialConditions::hires_vx_2LPT)
        .def_rw("hires_vy_2LPT", &InitialConditions::hires_vy_2LPT)
        .def_rw("hires_vz_2LPT", &InitialConditions::hires_vz_2LPT)
        .def_rw("lowres_vcb", &InitialConditions::lowres_vcb);

    nb::class_<PerturbedField>(m, "PerturbedField")
        .def(nb::init<>())
        .def_rw("density", &PerturbedField::density)
        .def_rw("velocity_x", &PerturbedField::velocity_x)
        .def_rw("velocity_y", &PerturbedField::velocity_y)
        .def_rw("velocity_z", &PerturbedField::velocity_z);

    nb::class_<HaloField>(m, "HaloField")
        .def(nb::init<>())
        .def_rw("n_halos", &HaloField::n_halos)
        .def_rw("buffer_size", &HaloField::buffer_size)
        .def_rw("halo_masses", &HaloField::halo_masses)
        .def_rw("halo_coords", &HaloField::halo_coords)
        .def_rw("star_rng", &HaloField::star_rng)
        .def_rw("sfr_rng", &HaloField::sfr_rng)
        .def_rw("xray_rng", &HaloField::xray_rng);

    nb::class_<PerturbHaloField>(m, "PerturbHaloField")
        .def(nb::init<>())
        .def_rw("n_halos", &PerturbHaloField::n_halos)
        .def_rw("buffer_size", &PerturbHaloField::buffer_size)
        .def_rw("halo_masses", &PerturbHaloField::halo_masses)
        .def_rw("halo_coords", &PerturbHaloField::halo_coords)
        .def_rw("star_rng", &PerturbHaloField::star_rng)
        .def_rw("sfr_rng", &PerturbHaloField::sfr_rng)
        .def_rw("xray_rng", &PerturbHaloField::xray_rng);

    nb::class_<HaloBox>(m, "HaloBox")
        .def(nb::init<>())
        .def_rw("halo_mass", &HaloBox::halo_mass)
        .def_rw("halo_stars", &HaloBox::halo_stars)
        .def_rw("halo_stars_mini", &HaloBox::halo_stars_mini)
        .def_rw("count", &HaloBox::count)
        .def_rw("n_ion", &HaloBox::n_ion)
        .def_rw("halo_sfr", &HaloBox::halo_sfr)
        .def_rw("halo_xray", &HaloBox::halo_xray)
        .def_rw("halo_sfr_mini", &HaloBox::halo_sfr_mini)
        .def_rw("whalo_sfr", &HaloBox::whalo_sfr)
        .def_rw("log10_Mcrit_ACG_ave", &HaloBox::log10_Mcrit_ACG_ave)
        .def_rw("log10_Mcrit_MCG_ave", &HaloBox::log10_Mcrit_MCG_ave);

    nb::class_<XraySourceBox>(m, "XraySourceBox")
        .def(nb::init<>())
        .def_rw("filtered_sfr", &XraySourceBox::filtered_sfr)
        .def_rw("filtered_xray", &XraySourceBox::filtered_xray)
        .def_rw("filtered_sfr_mini", &XraySourceBox::filtered_sfr_mini)
        .def_rw("mean_log10_Mcrit_LW", &XraySourceBox::mean_log10_Mcrit_LW)
        .def_rw("mean_sfr", &XraySourceBox::mean_sfr)
        .def_rw("mean_sfr_mini", &XraySourceBox::mean_sfr_mini);

    nb::class_<TsBox>(m, "TsBox")
        .def(nb::init<>())
        .def_rw("Ts_box", &TsBox::Ts_box)
        .def_rw("x_e_box", &TsBox::x_e_box)
        .def_rw("Tk_box", &TsBox::Tk_box)
        .def_rw("J_21_LW_box", &TsBox::J_21_LW_box);

    nb::class_<IonizedBox>(m, "IonizedBox")
        .def(nb::init<>())
        .def_rw("mean_f_coll", &IonizedBox::mean_f_coll)
        .def_rw("mean_f_coll_MINI", &IonizedBox::mean_f_coll_MINI)
        .def_rw("log10_Mturnover_ave", &IonizedBox::log10_Mturnover_ave)
        .def_rw("log10_Mturnover_MINI_ave", &IonizedBox::log10_Mturnover_MINI_ave)
        .def_rw("xH_box", &IonizedBox::xH_box)
        .def_rw("Gamma12_box", &IonizedBox::Gamma12_box)
        .def_rw("MFP_box", &IonizedBox::MFP_box)
        .def_rw("z_re_box", &IonizedBox::z_re_box)
        .def_rw("dNrec_box", &IonizedBox::dNrec_box)
        .def_rw("temp_kinetic_all_gas", &IonizedBox::temp_kinetic_all_gas)
        .def_rw("Fcoll", &IonizedBox::Fcoll)
        .def_rw("Fcoll_MINI", &IonizedBox::Fcoll_MINI);

    nb::class_<BrightnessTemp>(m, "BrightnessTemp ")
        .def(nb::init<>())
        .def_rw("brightness_temp", &BrightnessTemp::brightness_temp);

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
