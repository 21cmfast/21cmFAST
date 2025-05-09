#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

extern "C" {
#include "21cmFAST.h"
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
        .def_rw("OMn", &CosmoParams::OMn)
        .def_rw("OMk", &CosmoParams::OMk)
        .def_rw("OMr", &CosmoParams::OMr)
        .def_rw("OMtot", &CosmoParams::OMtot)
        .def_rw("Y_He", &CosmoParams::Y_He)
        .def_rw("wl", &CosmoParams::wl)
        .def_rw("POWER_INDEX", &CosmoParams::POWER_INDEX);

    // Bind SimulationOptions
    nb::class_<SimulationOptions>(m, "SimulationOptions")
        .def(nb::init<>())
        .def_rw("HII_DIM", &SimulationOptions::HII_DIM)
        .def_rw("DIM", &SimulationOptions::DIM)
        .def_rw("BOX_LEN", &SimulationOptions::BOX_LEN)
        .def_rw("NON_CUBIC_FACTOR", &SimulationOptions::NON_CUBIC_FACTOR)
        .def_rw("N_THREADS", &SimulationOptions::N_THREADS)
        .def_rw("Z_HEAT_MAX", &SimulationOptions::Z_HEAT_MAX)
        .def_rw("ZPRIME_STEP_FACTOR", &SimulationOptions::ZPRIME_STEP_FACTOR)
        .def_rw("SAMPLER_MIN_MASS", &SimulationOptions::SAMPLER_MIN_MASS)
        .def_rw("SAMPLER_BUFFER_FACTOR", &SimulationOptions::SAMPLER_BUFFER_FACTOR)
        .def_rw("N_COND_INTERP", &SimulationOptions::N_COND_INTERP)
        .def_rw("N_PROB_INTERP", &SimulationOptions::N_PROB_INTERP)
        .def_rw("MIN_LOGPROB", &SimulationOptions::MIN_LOGPROB)
        .def_rw("HALOMASS_CORRECTION", &SimulationOptions::HALOMASS_CORRECTION)
        .def_rw("PARKINSON_G0", &SimulationOptions::PARKINSON_G0)
        .def_rw("PARKINSON_y1", &SimulationOptions::PARKINSON_y1)
        .def_rw("PARKINSON_y2", &SimulationOptions::PARKINSON_y2)
        .def_rw("INITIAL_REDSHIFT", &SimulationOptions::INITIAL_REDSHIFT)
        .def_rw("DELTA_R_FACTOR", &SimulationOptions::DELTA_R_FACTOR)
        .def_rw("DENSITY_SMOOTH_RADIUS", &SimulationOptions::DENSITY_SMOOTH_RADIUS)
        .def_rw("DEXM_OPTIMIZE_MINMASS", &SimulationOptions::DEXM_OPTIMIZE_MINMASS)
        .def_rw("DEXM_R_OVERLAP", &SimulationOptions::DEXM_R_OVERLAP)
        .def_rw("CORR_STAR", &SimulationOptions::CORR_STAR)
        .def_rw("CORR_SFR", &SimulationOptions::CORR_SFR)
        .def_rw("CORR_LX", &SimulationOptions::CORR_LX);

    nb::class_<MatterOptions>(m, "MatterOptions")
        .def(nb::init<>())
        .def_rw("USE_FFTW_WISDOM", &MatterOptions::USE_FFTW_WISDOM)
        .def_rw("HMF", &MatterOptions::HMF)
        .def_rw("USE_RELATIVE_VELOCITIES", &MatterOptions::USE_RELATIVE_VELOCITIES)
        .def_rw("POWER_SPECTRUM", &MatterOptions::POWER_SPECTRUM)
        .def_rw("USE_INTERPOLATION_TABLES", &MatterOptions::USE_INTERPOLATION_TABLES)
        .def_rw("NO_RNG", &MatterOptions::NO_RNG)
        .def_rw("PERTURB_ON_HIGH_RES", &MatterOptions::PERTURB_ON_HIGH_RES)
        .def_rw("PERTURB_ALGORITHM", &MatterOptions::PERTURB_ALGORITHM)
        .def_rw("MINIMIZE_MEMORY", &MatterOptions::MINIMIZE_MEMORY)
        .def_rw("KEEP_3D_VELOCITIES", &MatterOptions::KEEP_3D_VELOCITIES)
        .def_rw("DEXM_OPTIMIZE", &MatterOptions::DEXM_OPTIMIZE)
        .def_rw("FILTER", &MatterOptions::FILTER)
        .def_rw("HALO_FILTER", &MatterOptions::HALO_FILTER)
        .def_rw("SMOOTH_EVOLVED_DENSITY_FIELD", &MatterOptions::SMOOTH_EVOLVED_DENSITY_FIELD)
        .def_rw("USE_HALO_FIELD", &MatterOptions::USE_HALO_FIELD)
        .def_rw("HALO_STOCHASTICITY", &MatterOptions::HALO_STOCHASTICITY)
        .def_rw("FIXED_HALO_GRIDS", &MatterOptions::FIXED_HALO_GRIDS)
        .def_rw("SAMPLE_METHOD", &MatterOptions::SAMPLE_METHOD);

    // Bind AstroParams
    nb::class_<AstroParams>(m, "AstroParams")
        .def(nb::init<>())
        .def_rw("HII_EFF_FACTOR", &AstroParams::HII_EFF_FACTOR)
        .def_rw("F_STAR10", &AstroParams::F_STAR10)
        .def_rw("ALPHA_STAR", &AstroParams::ALPHA_STAR)
        .def_rw("ALPHA_STAR_MINI", &AstroParams::ALPHA_STAR_MINI)
        .def_rw("SIGMA_STAR", &AstroParams::SIGMA_STAR)
        .def_rw("UPPER_STELLAR_TURNOVER_MASS", &AstroParams::UPPER_STELLAR_TURNOVER_MASS)
        .def_rw("UPPER_STELLAR_TURNOVER_INDEX", &AstroParams::UPPER_STELLAR_TURNOVER_INDEX)
        .def_rw("F_STAR7_MINI", &AstroParams::F_STAR7_MINI)
        .def_rw("t_STAR", &AstroParams::t_STAR)
        .def_rw("SIGMA_SFR_INDEX", &AstroParams::SIGMA_SFR_INDEX)
        .def_rw("SIGMA_SFR_LIM", &AstroParams::SIGMA_SFR_LIM)
        .def_rw("L_X", &AstroParams::L_X)
        .def_rw("L_X_MINI", &AstroParams::L_X_MINI)
        .def_rw("SIGMA_LX", &AstroParams::SIGMA_LX)
        .def_rw("F_ESC10", &AstroParams::F_ESC10)
        .def_rw("ALPHA_ESC", &AstroParams::ALPHA_ESC)
        .def_rw("F_ESC7_MINI", &AstroParams::F_ESC7_MINI)
        .def_rw("T_RE", &AstroParams::T_RE)
        .def_rw("M_TURN", &AstroParams::M_TURN)
        .def_rw("R_BUBBLE_MAX", &AstroParams::R_BUBBLE_MAX)
        .def_rw("ION_Tvir_MIN", &AstroParams::ION_Tvir_MIN)
        .def_rw("F_H2_SHIELD", &AstroParams::F_H2_SHIELD)
        .def_rw("NU_X_THRESH", &AstroParams::NU_X_THRESH)
        .def_rw("X_RAY_SPEC_INDEX", &AstroParams::X_RAY_SPEC_INDEX)
        .def_rw("X_RAY_Tvir_MIN", &AstroParams::X_RAY_Tvir_MIN)
        .def_rw("A_LW", &AstroParams::A_LW)
        .def_rw("BETA_LW", &AstroParams::BETA_LW)
        .def_rw("A_VCB", &AstroParams::A_VCB)
        .def_rw("BETA_VCB", &AstroParams::BETA_VCB)
        .def_rw("FIXED_VAVG", &AstroParams::FIXED_VAVG)
        .def_rw("POP2_ION", &AstroParams::POP2_ION)
        .def_rw("POP3_ION", &AstroParams::POP3_ION)
        .def_rw("N_RSD_STEPS", &AstroParams::N_RSD_STEPS)
        .def_rw("PHOTONCONS_CALIBRATION_END", &AstroParams::PHOTONCONS_CALIBRATION_END)
        .def_rw("CLUMPING_FACTOR", &AstroParams::CLUMPING_FACTOR)
        .def_rw("ALPHA_UVB", &AstroParams::ALPHA_UVB)
        .def_rw("R_MAX_TS", &AstroParams::R_MAX_TS)
        .def_rw("N_STEP_TS", &AstroParams::N_STEP_TS)
        .def_rw("DELTA_R_HII_FACTOR", &AstroParams::DELTA_R_HII_FACTOR)
        .def_rw("R_BUBBLE_MIN", &AstroParams::R_BUBBLE_MIN)
        .def_rw("MAX_DVDR", &AstroParams::MAX_DVDR)
        .def_rw("NU_X_MAX", &AstroParams::NU_X_MAX)
        .def_rw("NU_X_BAND_MAX", &AstroParams::NU_X_BAND_MAX);

    // TODO: the getter/setter workaround is clunky, we can go via a C++ std:string
    //   or try something else.
    nb::class_<ConfigSettings>(m, "ConfigSettings")
        .def(nb::init<>())
        .def_rw("HALO_CATALOG_MEM_FACTOR", &ConfigSettings::HALO_CATALOG_MEM_FACTOR)
        .def_ro("external_table_path", &ConfigSettings::external_table_path)
        .def_ro("wisdoms_path", &ConfigSettings::wisdoms_path)
        .def("set_external_table_path", &set_external_table_path)
        .def("get_external_table_path", &get_external_table_path)
        .def("set_wisdoms_path", &set_wisdoms_path)
        .def("get_wisdoms_path", &get_wisdoms_path);

    nb::class_<AstroOptions>(m, "AstroOptions")
        .def(nb::init<>())
        .def_rw("USE_MINI_HALOS", &AstroOptions::USE_MINI_HALOS)
        .def_rw("USE_CMB_HEATING", &AstroOptions::USE_CMB_HEATING)
        .def_rw("USE_LYA_HEATING", &AstroOptions::USE_LYA_HEATING)
        .def_rw("USE_MASS_DEPENDENT_ZETA", &AstroOptions::USE_MASS_DEPENDENT_ZETA)
        .def_rw("SUBCELL_RSD", &AstroOptions::SUBCELL_RSD)
        .def_rw("APPLY_RSDS", &AstroOptions::APPLY_RSDS)
        .def_rw("INHOMO_RECO", &AstroOptions::INHOMO_RECO)
        .def_rw("USE_TS_FLUCT", &AstroOptions::USE_TS_FLUCT)
        .def_rw("M_MIN_in_Mass", &AstroOptions::M_MIN_in_Mass)
        .def_rw("FIX_VCB_AVG", &AstroOptions::FIX_VCB_AVG)
        .def_rw("USE_EXP_FILTER", &AstroOptions::USE_EXP_FILTER)
        .def_rw("CELL_RECOMB", &AstroOptions::CELL_RECOMB)
        .def_rw("PHOTON_CONS_TYPE", &AstroOptions::PHOTON_CONS_TYPE)
        .def_rw("USE_UPPER_STELLAR_TURNOVER", &AstroOptions::USE_UPPER_STELLAR_TURNOVER)
        .def_rw("HALO_SCALING_RELATIONS_MEDIAN", &AstroOptions::HALO_SCALING_RELATIONS_MEDIAN)
        .def_rw("HII_FILTER", &AstroOptions::HII_FILTER)
        .def_rw("HEAT_FILTER", &AstroOptions::HEAT_FILTER)
        .def_rw("IONISE_ENTIRE_SPHERE", &AstroOptions::IONISE_ENTIRE_SPHERE)
        .def_rw("AVG_BELOW_SAMPLER", &AstroOptions::AVG_BELOW_SAMPLER)
        .def_rw("INTEGRATION_METHOD_ATOMIC", &AstroOptions::INTEGRATION_METHOD_ATOMIC)
        .def_rw("INTEGRATION_METHOD_MINI", &AstroOptions::INTEGRATION_METHOD_MINI);

    // Output Struct Bindings
    // Bind InitialConditions
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

    // Bind PerturbedField
    nb::class_<PerturbedField>(m, "PerturbedField")
        .def(nb::init<>())
        .def_rw("density", &PerturbedField::density)
        .def_rw("velocity_x", &PerturbedField::velocity_x)
        .def_rw("velocity_y", &PerturbedField::velocity_y)
        .def_rw("velocity_z", &PerturbedField::velocity_z);

    // Bind HaloField
    nb::class_<HaloField>(m, "HaloField")
        .def(nb::init<>())
        .def_rw("n_halos", &HaloField::n_halos)
        .def_rw("buffer_size", &HaloField::buffer_size)
        .def_rw("halo_masses", &HaloField::halo_masses)
        .def_rw("halo_coords", &HaloField::halo_coords)
        .def_rw("star_rng", &HaloField::star_rng)
        .def_rw("sfr_rng", &HaloField::sfr_rng)
        .def_rw("xray_rng", &HaloField::xray_rng);

    // Bind PerturbHaloField
    nb::class_<PerturbHaloField>(m, "PerturbHaloField")
        .def(nb::init<>())
        .def_rw("n_halos", &PerturbHaloField::n_halos)
        .def_rw("buffer_size", &PerturbHaloField::buffer_size)
        .def_rw("halo_masses", &PerturbHaloField::halo_masses)
        .def_rw("halo_coords", &PerturbHaloField::halo_coords)
        .def_rw("star_rng", &PerturbHaloField::star_rng)
        .def_rw("sfr_rng", &PerturbHaloField::sfr_rng)
        .def_rw("xray_rng", &PerturbHaloField::xray_rng);

    // Bind HaloBox
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

    // Bind XraySourceBox
    nb::class_<XraySourceBox>(m, "XraySourceBox")
        .def(nb::init<>())
        .def_rw("filtered_sfr", &XraySourceBox::filtered_sfr)
        .def_rw("filtered_xray", &XraySourceBox::filtered_xray)
        .def_rw("filtered_sfr_mini", &XraySourceBox::filtered_sfr_mini)
        .def_rw("mean_log10_Mcrit_LW", &XraySourceBox::mean_log10_Mcrit_LW)
        .def_rw("mean_sfr", &XraySourceBox::mean_sfr)
        .def_rw("mean_sfr_mini", &XraySourceBox::mean_sfr_mini);

    // Bind TsBox
    nb::class_<TsBox>(m, "TsBox")
        .def(nb::init<>())
        .def_rw("spin_temperature", &TsBox::spin_temperature)
        .def_rw("xray_ionised_fraction", &TsBox::xray_ionised_fraction)
        .def_rw("kinetic_temp_neutral", &TsBox::kinetic_temp_neutral)
        .def_rw("J_21_LW", &TsBox::J_21_LW);

    // Bind IonizedBox
    nb::class_<IonizedBox>(m, "IonizedBox")
        .def(nb::init<>())
        .def_rw("mean_f_coll", &IonizedBox::mean_f_coll)
        .def_rw("mean_f_coll_MINI", &IonizedBox::mean_f_coll_MINI)
        .def_rw("log10_Mturnover_ave", &IonizedBox::log10_Mturnover_ave)
        .def_rw("log10_Mturnover_MINI_ave", &IonizedBox::log10_Mturnover_MINI_ave)
        .def_rw("neutral_fraction", &IonizedBox::neutral_fraction)
        .def_rw("ionisation_rate_G12", &IonizedBox::ionisation_rate_G12)
        .def_rw("mean_free_path", &IonizedBox::mean_free_path)
        .def_rw("z_reion", &IonizedBox::z_reion)
        .def_rw("cumulative_recombinations", &IonizedBox::cumulative_recombinations)
        .def_rw("kinetic_temperature", &IonizedBox::kinetic_temperature)
        .def_rw("unnormalised_nion", &IonizedBox::unnormalised_nion)
        .def_rw("unnormalised_nion_mini", &IonizedBox::unnormalised_nion_mini);

    // Bind BrightnessTemp
    nb::class_<BrightnessTemp>(m, "BrightnessTemp")
        .def(nb::init<>())
        .def_rw("brightness_temp", &BrightnessTemp::brightness_temp);

    // OutputStruct COMPUTE FUNCTIONS
    m.def("ComputeInitialConditions", &ComputeInitialConditions);
    m.def("ComputePerturbField", &ComputePerturbField);
    m.def("ComputeHaloField", &ComputeHaloField);
    m.def("ComputePerturbHaloField", &ComputePerturbHaloField);
    m.def("ComputeTsBox", &ComputeTsBox);
    m.def("ComputeIonizedBox", &ComputeIonizedBox);
    m.def("ComputeBrightnessTemp", &ComputeBrightnessTemp);
    m.def("ComputeHaloBox", &ComputeHaloBox);
    m.def("UpdateXraySourceBox", &UpdateXraySourceBox);

    // PHOTON CONSERVATION MODEL FUNCTIONS
    m.def("InitialisePhotonCons", &InitialisePhotonCons);
    m.def("PhotonCons_Calibration", &PhotonCons_Calibration);
    m.def("ComputeZstart_PhotonCons", &ComputeZstart_PhotonCons);
    m.def("adjust_redshifts_for_photoncons", &adjust_redshifts_for_photoncons);
    m.def("determine_deltaz_for_photoncons", &determine_deltaz_for_photoncons);
    m.def("ObtainPhotonConsData", &ObtainPhotonConsData);
    m.def("FreePhotonConsMemory", &FreePhotonConsMemory);
    m.def("set_alphacons_params", &set_alphacons_params);

    // Non-OutputStruct data products
    m.def("ComputeLF", &ComputeLF);
    m.def("ComputeTau", &ComputeTau);

    // Initialisation functions needed in the wrapper
    m.def("init_ps", &init_ps);
    m.def("init_heat", &init_heat);
    m.def("CreateFFTWWisdoms", &CreateFFTWWisdoms);
    m.def("Broadcast_struct_global_noastro", &Broadcast_struct_global_noastro);
    m.def("Broadcast_struct_global_all", &Broadcast_struct_global_all);
    m.def("initialiseSigmaMInterpTable", &initialiseSigmaMInterpTable);
    m.def("initialise_GL", &initialise_GL);

    // Integration routines
    m.def("get_sigma", &get_sigma);
    m.def("get_condition_integrals", &get_condition_integrals);
    m.def("get_halo_chmf_interval", &get_halo_chmf_interval);
    m.def("get_halomass_at_probability", &get_halomass_at_probability);
    m.def("get_global_SFRD_z", &get_global_SFRD_z);
    m.def("get_global_Nion_z", &get_global_Nion_z);
    m.def("get_conditional_FgtrM", &get_conditional_FgtrM);
    m.def("get_conditional_SFRD", &get_conditional_SFRD);
    m.def("get_conditional_Nion", &get_conditional_Nion);
    m.def("get_conditional_Xray", &get_conditional_Xray);

    // Error framework testing
    m.def("SomethingThatCatches", &SomethingThatCatches);
    m.def("FunctionThatCatches", &FunctionThatCatches);
    m.def("FunctionThatThrows", &FunctionThatThrows);

    // Test Outputs For Specific Models
    m.def("single_test_sample", &single_test_sample);
    m.def("test_halo_props", &test_halo_props);
    m.def("test_filter", &test_filter);

    // Functions required to access cosmology & mass functions directly
    m.def("dicke", &dicke);
    m.def("sigma_z0", &sigma_z0);
    m.def("dsigmasqdm_z0", &dsigmasqdm_z0);
    m.def("power_in_k", &power_in_k);
    m.def("get_delta_crit", &get_delta_crit);
    m.def("atomic_cooling_threshold", &atomic_cooling_threshold);
    m.def("unconditional_hmf", &unconditional_hmf);
    m.def("conditional_hmf", &conditional_hmf);
    m.def("expected_nhalo", &expected_nhalo);

    m.def(
        "get_config_settings", []() -> ConfigSettings& { return config_settings; },
        nb::rv_policy::reference);
}
