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

    // Bind MatterOptions
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

    // Bind AstroOptions
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

    // Bind ConfigSettings
    nb::class_<ConfigSettings>(m, "ConfigSettings")
        .def(nb::init<>())
        .def_rw("HALO_CATALOG_MEM_FACTOR", &ConfigSettings::HALO_CATALOG_MEM_FACTOR)
        .def("set_external_table_path",
             [](ConfigSettings& self, const std::string& path) {
                 strcpy(self.external_table_path, path.c_str());
             })
        .def("get_external_table_path",
             [](ConfigSettings& self) { return std::string(self.external_table_path); })
        .def("set_wisdoms_path",
             [](ConfigSettings& self, const std::string& path) {
                 strcpy(self.wisdoms_path, path.c_str());
             })
        .def("get_wisdoms_path",
             [](ConfigSettings& self) { return std::string(self.wisdoms_path); });

    // Output Struct Bindings
    // Bind InitialConditions
    nb::class_<InitialConditions>(m, "InitialConditions")
        .def(nb::init<>())
        .def("set_lowres_density",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.lowres_density = array.data();
             })
        .def("set_lowres_vx", [](InitialConditions& self,
                                 nb::ndarray<float> array) { self.lowres_vx = array.data(); })
        .def("set_lowres_vy", [](InitialConditions& self,
                                 nb::ndarray<float> array) { self.lowres_vy = array.data(); })
        .def("set_lowres_vz", [](InitialConditions& self,
                                 nb::ndarray<float> array) { self.lowres_vz = array.data(); })
        .def("set_lowres_vx_2LPT",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.lowres_vx_2LPT = array.data();
             })
        .def("set_lowres_vy_2LPT",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.lowres_vy_2LPT = array.data();
             })
        .def("set_lowres_vz_2LPT",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.lowres_vz_2LPT = array.data();
             })
        .def("set_hires_density",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.hires_density = array.data();
             })
        .def("set_hires_vx", [](InitialConditions& self,
                                nb::ndarray<float> array) { self.hires_vx = array.data(); })
        .def("set_hires_vy", [](InitialConditions& self,
                                nb::ndarray<float> array) { self.hires_vy = array.data(); })
        .def("set_hires_vz", [](InitialConditions& self,
                                nb::ndarray<float> array) { self.hires_vz = array.data(); })
        .def("set_hires_vx_2LPT",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.hires_vx_2LPT = array.data();
             })
        .def("set_hires_vy_2LPT",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.hires_vy_2LPT = array.data();
             })
        .def("set_hires_vz_2LPT",
             [](InitialConditions& self, nb::ndarray<float> array) {
                 self.hires_vz_2LPT = array.data();
             })
        .def("set_lowres_vcb", [](InitialConditions& self, nb::ndarray<float> array) {
            self.lowres_vcb = array.data();
        });

    // Bind PerturbedField
    nb::class_<PerturbedField>(m, "PerturbedField")
        .def(nb::init<>())
        .def("set_density",
             [](PerturbedField& self, nb::ndarray<float> array) { self.density = array.data(); })
        .def("set_velocity_x",
             [](PerturbedField& self, nb::ndarray<float> array) { self.velocity_x = array.data(); })
        .def("set_velocity_y",
             [](PerturbedField& self, nb::ndarray<float> array) { self.velocity_y = array.data(); })
        .def("set_velocity_z", [](PerturbedField& self, nb::ndarray<float> array) {
            self.velocity_z = array.data();
        });

    // Bind HaloField
    nb::class_<HaloField>(m, "HaloField")
        .def(nb::init<>())
        .def_rw("n_halos", &HaloField::n_halos)
        .def_rw("buffer_size", &HaloField::buffer_size)
        .def("set_halo_masses",
             [](HaloField& self, nb::ndarray<float> array) { self.halo_masses = array.data(); })
        .def("set_halo_coords",
             [](HaloField& self, nb::ndarray<int> array) { self.halo_coords = array.data(); })
        .def("set_star_rng",
             [](HaloField& self, nb::ndarray<float> array) { self.star_rng = array.data(); })
        .def("set_sfr_rng",
             [](HaloField& self, nb::ndarray<float> array) { self.sfr_rng = array.data(); })
        .def("set_xray_rng",
             [](HaloField& self, nb::ndarray<float> array) { self.xray_rng = array.data(); });

    // Bind PerturbHaloField
    nb::class_<PerturbHaloField>(m, "PerturbHaloField")
        .def(nb::init<>())
        .def_rw("n_halos", &PerturbHaloField::n_halos)
        .def_rw("buffer_size", &PerturbHaloField::buffer_size)
        .def("set_halo_masses", [](PerturbHaloField& self,
                                   nb::ndarray<float> array) { self.halo_masses = array.data(); })
        .def("set_halo_coords", [](PerturbHaloField& self,
                                   nb::ndarray<int> array) { self.halo_coords = array.data(); })
        .def("set_star_rng",
             [](PerturbHaloField& self, nb::ndarray<float> array) { self.star_rng = array.data(); })
        .def("set_sfr_rng",
             [](PerturbHaloField& self, nb::ndarray<float> array) { self.sfr_rng = array.data(); })
        .def("set_xray_rng", [](PerturbHaloField& self, nb::ndarray<float> array) {
            self.xray_rng = array.data();
        });

    // Bind HaloBox
    nb::class_<HaloBox>(m, "HaloBox")
        .def(nb::init<>())
        .def("set_halo_mass",
             [](HaloBox& self, nb::ndarray<float> array) { self.halo_mass = array.data(); })
        .def("set_halo_stars",
             [](HaloBox& self, nb::ndarray<float> array) { self.halo_stars = array.data(); })
        .def("set_halo_stars_mini",
             [](HaloBox& self, nb::ndarray<float> array) { self.halo_stars_mini = array.data(); })
        .def("set_count", [](HaloBox& self, nb::ndarray<int> array) { self.count = array.data(); })
        .def("set_n_ion",
             [](HaloBox& self, nb::ndarray<float> array) { self.n_ion = array.data(); })
        .def("set_halo_sfr",
             [](HaloBox& self, nb::ndarray<float> array) { self.halo_sfr = array.data(); })
        .def("set_halo_xray",
             [](HaloBox& self, nb::ndarray<float> array) { self.halo_xray = array.data(); })
        .def("set_halo_sfr_mini",
             [](HaloBox& self, nb::ndarray<float> array) { self.halo_sfr_mini = array.data(); })
        .def("set_whalo_sfr",
             [](HaloBox& self, nb::ndarray<float> array) { self.whalo_sfr = array.data(); })
        .def_rw("log10_Mcrit_ACG_ave", &HaloBox::log10_Mcrit_ACG_ave)
        .def_rw("log10_Mcrit_MCG_ave", &HaloBox::log10_Mcrit_MCG_ave);

    // Bind XraySourceBox
    nb::class_<XraySourceBox>(m, "XraySourceBox")
        .def(nb::init<>())
        .def("set_filtered_sfr", [](XraySourceBox& self,
                                    nb::ndarray<float> array) { self.filtered_sfr = array.data(); })
        .def("set_filtered_xray",
             [](XraySourceBox& self, nb::ndarray<float> array) {
                 self.filtered_xray = array.data();
             })
        .def("set_filtered_sfr_mini",
             [](XraySourceBox& self, nb::ndarray<float> array) {
                 self.filtered_sfr_mini = array.data();
             })
        .def("set_mean_log10_Mcrit_LW",
             [](XraySourceBox& self, nb::ndarray<double> array) {
                 self.mean_log10_Mcrit_LW = array.data();
             })
        .def("set_mean_sfr",
             [](XraySourceBox& self, nb::ndarray<double> array) { self.mean_sfr = array.data(); })
        .def("set_mean_sfr_mini", [](XraySourceBox& self, nb::ndarray<double> array) {
            self.mean_sfr_mini = array.data();
        });

    // Bind TsBox
    nb::class_<TsBox>(m, "TsBox")
        .def(nb::init<>())
        .def("set_spin_temperature",
             [](TsBox& self, nb::ndarray<float> array) { self.spin_temperature = array.data(); })
        .def("set_xray_ionised_fraction",
             [](TsBox& self, nb::ndarray<float> array) {
                 self.xray_ionised_fraction = array.data();
             })
        .def(
            "set_kinetic_temp_neutral",
            [](TsBox& self, nb::ndarray<float> array) { self.kinetic_temp_neutral = array.data(); })
        .def("set_J_21_LW",
             [](TsBox& self, nb::ndarray<float> array) { self.J_21_LW = array.data(); });

    // Bind IonizedBox
    nb::class_<IonizedBox>(m, "IonizedBox")
        .def(nb::init<>())
        .def_rw("mean_f_coll", &IonizedBox::mean_f_coll)
        .def_rw("mean_f_coll_MINI", &IonizedBox::mean_f_coll_MINI)
        .def_rw("log10_Mturnover_ave", &IonizedBox::log10_Mturnover_ave)
        .def_rw("log10_Mturnover_MINI_ave", &IonizedBox::log10_Mturnover_MINI_ave)
        .def("set_neutral_fraction",
             [](IonizedBox& self, nb::ndarray<float> array) {
                 self.neutral_fraction = array.data();
             })
        .def("set_ionisation_rate_G12",
             [](IonizedBox& self, nb::ndarray<float> array) {
                 self.ionisation_rate_G12 = array.data();
             })
        .def("set_mean_free_path",
             [](IonizedBox& self, nb::ndarray<float> array) { self.mean_free_path = array.data(); })
        .def("set_z_reion",
             [](IonizedBox& self, nb::ndarray<float> array) { self.z_reion = array.data(); })
        .def("set_cumulative_recombinations",
             [](IonizedBox& self, nb::ndarray<float> array) {
                 self.cumulative_recombinations = array.data();
             })
        .def("set_kinetic_temperature",
             [](IonizedBox& self, nb::ndarray<float> array) {
                 self.kinetic_temperature = array.data();
             })
        .def("set_unnormalised_nion",
             [](IonizedBox& self, nb::ndarray<float> array) {
                 self.unnormalised_nion = array.data();
             })
        .def("set_unnormalised_nion_mini", [](IonizedBox& self, nb::ndarray<float> array) {
            self.unnormalised_nion_mini = array.data();
        });

    // Bind BrightnessTemp
    nb::class_<BrightnessTemp>(m, "BrightnessTemp")
        .def(nb::init<>())
        .def("set_brightness_temp", [](BrightnessTemp& self, nb::ndarray<float> array) {
            self.brightness_temp = array.data();
        });

    // Function Bindings
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
    m.def("get_sigma", [](int size, nb::ndarray<double> masses, nb::ndarray<double> sigma,
                          nb::ndarray<double> dsigmasq) {
        return get_sigma(size, masses.data(), sigma.data(), dsigmasq.data());
    });
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
