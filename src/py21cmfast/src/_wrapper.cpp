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
    m.def("PhotonCons_Calibration",
          [](nb::ndarray<double> z_estimate, nb::ndarray<double> xH_estimate) {
              int n_spline = z_estimate.size();
              if (xH_estimate.size() != n_spline) {
                  throw std::runtime_error("Array sizes do not match the specified NSpline.");
              }
              int status = PhotonCons_Calibration(z_estimate.data(), xH_estimate.data(), n_spline);
              if (status != 0) {
                  throw std::runtime_error("PhotonCons_Calibration failed with status: " +
                                           std::to_string(status));
              }
          });
    m.def("ComputeZstart_PhotonCons", [](nb::ndarray<double> zstart) {
        if (zstart.size() != 1) {
            throw std::runtime_error("zstart array must have size 1.");
        }
        int status = ComputeZstart_PhotonCons(zstart.data());
        if (status != 0) {
            throw std::runtime_error("ComputeZstart_PhotonCons failed with status: " +
                                     std::to_string(status));
        }
    });
    m.def("adjust_redshifts_for_photoncons",
          [](double z_step_factor, nb::ndarray<float> redshift, nb::ndarray<float> stored_redshift,
             nb::ndarray<float> absolute_delta_z) {
              adjust_redshifts_for_photoncons(z_step_factor, redshift.data(),
                                              stored_redshift.data(), absolute_delta_z.data());
          });
    m.def("determine_deltaz_for_photoncons", &determine_deltaz_for_photoncons);
    m.def("ObtainPhotonConsData",
          [](nb::ndarray<double> z_at_Q_data, nb::ndarray<double> Q_data,
             nb::ndarray<int> Ndata_analytic, nb::ndarray<double> z_cal_data,
             nb::ndarray<double> nf_cal_data, nb::ndarray<int> Ndata_calibration,
             nb::ndarray<double> PhotonCons_NFdata, nb::ndarray<double> PhotonCons_deltaz,
             nb::ndarray<int> Ndata_PhotonCons) {
              if (Ndata_analytic.size() != 1 || Ndata_calibration.size() != 1 ||
                  Ndata_PhotonCons.size() != 1) {
                  throw std::runtime_error(
                      "Ndata_analytic, Ndata_calibration, and Ndata_PhotonCons must have size 1.");
              }
              int status = ObtainPhotonConsData(
                  z_at_Q_data.data(), Q_data.data(), Ndata_analytic.data(), z_cal_data.data(),
                  nf_cal_data.data(), Ndata_calibration.data(), PhotonCons_NFdata.data(),
                  PhotonCons_deltaz.data(), Ndata_PhotonCons.data());
              if (status != 0) {
                  throw std::runtime_error("ObtainPhotonConsData failed with status: " +
                                           std::to_string(status));
              }
          });
    m.def("FreePhotonConsMemory", &FreePhotonConsMemory);
    m.def("set_alphacons_params", &set_alphacons_params);

    // Non-OutputStruct data products
    m.def("ComputeLF",
          [](int component, size_t n_bins_mass, nb::ndarray<float> z_LF, nb::ndarray<float> M_TURNs,
             nb::ndarray<double> M_uv_z, nb::ndarray<double> M_h_z, nb::ndarray<double> log10phi) {
              size_t n_redshifts = z_LF.shape(0);
              if (M_h_z.shape(0) != n_redshifts || M_h_z.shape(1) != n_bins_mass ||
                  M_uv_z.shape(0) != n_redshifts || M_uv_z.shape(1) != n_bins_mass ||
                  log10phi.shape(0) != n_redshifts || log10phi.shape(1) != n_bins_mass ||
                  M_TURNs.shape(0) != n_redshifts) {
                  throw std::runtime_error(
                      "Array size mismatch: M_h_z shape: " + std::to_string(M_h_z.shape(0)) + "x" +
                      std::to_string(M_h_z.shape(1)) + ", M_uv_z shape: " +
                      std::to_string(M_uv_z.shape(0)) + "x" + std::to_string(M_uv_z.shape(1)) +
                      ", log10phi shape: " + std::to_string(log10phi.shape(0)) + "x" +
                      std::to_string(log10phi.shape(1)) +
                      ", M_TURNs shape: " + std::to_string(M_TURNs.shape(0)));
              }
              ComputeLF(n_bins_mass, component, n_redshifts, z_LF.data(), M_TURNs.data(),
                        M_h_z.data(), M_uv_z.data(), log10phi.data());
          });
    m.def("ComputeTau",
          [](nb::ndarray<float> redshifts, nb::ndarray<float> global_xHI, float z_re_HeII) {
              size_t n_redshifts = redshifts.shape(0);
              if (global_xHI.shape(0) != n_redshifts) {
                  throw std::runtime_error("XHI array size" + std::to_string(global_xHI.shape(0)) +
                                           "does not match the number of redshifts." +
                                           std::to_string(n_redshifts));
              }
              return ComputeTau(n_redshifts, redshifts.data(), global_xHI.data(), z_re_HeII);
          });

    // Initialisation functions needed in the wrapper
    m.def("init_ps", &init_ps);
    m.def("init_heat", &init_heat);
    m.def("CreateFFTWWisdoms", &CreateFFTWWisdoms);
    m.def("Broadcast_struct_global_noastro", &Broadcast_struct_global_noastro);
    m.def("Broadcast_struct_global_all", &Broadcast_struct_global_all);
    m.def("initialiseSigmaMInterpTable", &initialiseSigmaMInterpTable);
    m.def("initialise_GL", &initialise_GL);

    // Integration routines
    // TODO: it may be a better choice to rewrite integral_wrappers in C++ directly
    m.def("get_sigma", [](nb::ndarray<double> mass_values, nb::ndarray<double> sigma_out,
                          nb::ndarray<double> dsigmasqdm_out) {
        size_t n_masses = mass_values.shape(0);
        if (sigma_out.shape(0) != n_masses || dsigmasqdm_out.shape(0) != n_masses) {
            throw std::runtime_error("Array sizes do not match the number of masses.");
        }
        get_sigma(n_masses, mass_values.data(), sigma_out.data(), dsigmasqdm_out.data());
    });

    m.def("get_condition_integrals",
          [](double redshift, double z_prev, nb::ndarray<double> cond_values,
             nb::ndarray<double> out_n_exp, nb::ndarray<double> out_m_exp) {
              size_t n_conditions = cond_values.shape(0);
              if (out_n_exp.shape(0) != n_conditions || out_m_exp.shape(0) != n_conditions) {
                  throw std::runtime_error("Array sizes do not match the number of conditions.");
              }
              get_condition_integrals(redshift, z_prev, n_conditions, cond_values.data(),
                                      out_n_exp.data(), out_m_exp.data());
          });

    m.def("get_halo_chmf_interval",
          [](double redshift, double z_prev, nb::ndarray<double> cond_values,
             nb::ndarray<double> lnM_lo, nb::ndarray<double> lnM_hi, nb::ndarray<double> out_n) {
              size_t n_conditions = cond_values.shape(0);
              size_t n_masslim = lnM_lo.shape(0);
              if (lnM_hi.shape(0) != n_masslim || out_n.shape(0) != n_conditions ||
                  out_n.shape(1) != n_masslim) {
                  throw std::runtime_error("Array sizes do not match the specified dimensions.");
              }
              get_halo_chmf_interval(redshift, z_prev, n_conditions, cond_values.data(), n_masslim,
                                     lnM_lo.data(), lnM_hi.data(), out_n.data());
          });

    m.def("get_halomass_at_probability",
          [](double redshift, double z_prev, nb::ndarray<double> cond_values,
             nb::ndarray<double> probabilities, nb::ndarray<double> out_mass) {
              size_t n_conditions = cond_values.shape(0) * cond_values.shape(1);
              if (probabilities.shape(0) * probabilities.shape(1) != n_conditions ||
                  out_mass.shape(0) * out_mass.shape(1) != n_conditions) {
                  throw std::runtime_error("Array sizes do not match the number of conditions.");
              }
              get_halomass_at_probability(redshift, z_prev, n_conditions, cond_values.data(),
                                          probabilities.data(), out_mass.data());
          });

    m.def("get_global_SFRD_z",
          [](nb::ndarray<double> redshifts, nb::ndarray<double> log10_turnovers_mcg,
             nb::ndarray<double> out_sfrd, nb::ndarray<double> out_sfrd_mini) {
              size_t n_redshift = redshifts.size();
              if (log10_turnovers_mcg.size() != n_redshift || out_sfrd.size() != n_redshift ||
                  out_sfrd_mini.size() != n_redshift) {
                  throw std::runtime_error("Array sizes do not match the number of redshifts.");
              }
              get_global_SFRD_z(n_redshift, redshifts.data(), log10_turnovers_mcg.data(),
                                out_sfrd.data(), out_sfrd_mini.data());
          });

    m.def("get_global_Nion_z",
          [](nb::ndarray<double> redshifts, nb::ndarray<double> log10_turnovers_mcg,
             nb::ndarray<double> out_nion, nb::ndarray<double> out_nion_mini) {
              size_t n_redshift = redshifts.size();
              if (log10_turnovers_mcg.size() != n_redshift || out_nion.size() != n_redshift ||
                  out_nion_mini.size() != n_redshift) {
                  throw std::runtime_error("Array sizes do not match the number of redshifts.");
              }
              get_global_Nion_z(n_redshift, redshifts.data(), log10_turnovers_mcg.data(),
                                out_nion.data(), out_nion_mini.data());
          });

    m.def("get_conditional_FgtrM",
          [](double redshift, double R, nb::ndarray<double> densities,
             nb::ndarray<double> out_fcoll, nb::ndarray<double> out_dfcoll) {
              size_t n_densities = densities.size();
              if (out_fcoll.size() != n_densities || out_dfcoll.size() != n_densities) {
                  throw std::runtime_error("Array sizes do not match the number of densities.");
              }
              get_conditional_FgtrM(redshift, R, n_densities, densities.data(), out_fcoll.data(),
                                    out_dfcoll.data());
          });

    m.def("get_conditional_SFRD", [](double redshift, double R, nb::ndarray<double> densities,
                                     nb::ndarray<double> log10_mturns, nb::ndarray<double> out_sfrd,
                                     nb::ndarray<double> out_sfrd_mini) {
        size_t n_densities = densities.size();
        if (log10_mturns.size() != n_densities || out_sfrd.size() != n_densities ||
            out_sfrd_mini.size() != n_densities) {
            throw std::runtime_error("Array sizes do not match the number of densities.");
        }
        get_conditional_SFRD(redshift, R, n_densities, densities.data(), log10_mturns.data(),
                             out_sfrd.data(), out_sfrd_mini.data());
    });

    m.def("get_conditional_Nion", [](double redshift, double R, nb::ndarray<double> densities,
                                     nb::ndarray<double> log10_mturns_acg,
                                     nb::ndarray<double> log10_mturns_mcg,
                                     nb::ndarray<double> out_nion,
                                     nb::ndarray<double> out_nion_mini) {
        size_t n_densities = densities.size();
        if (log10_mturns_acg.size() != n_densities || log10_mturns_mcg.size() != n_densities ||
            out_nion.size() != n_densities || out_nion_mini.size() != n_densities) {
            throw std::runtime_error("Array sizes do not match the number of densities.");
        }
        get_conditional_Nion(redshift, R, n_densities, densities.data(), log10_mturns_acg.data(),
                             log10_mturns_mcg.data(), out_nion.data(), out_nion_mini.data());
    });

    m.def("get_conditional_Xray",
          [](double redshift, double R, nb::ndarray<double> densities,
             nb::ndarray<double> log10_mturns, nb::ndarray<double> out_xray) {
              size_t n_densities = densities.size();
              if (log10_mturns.size() != n_densities || out_xray.size() != n_densities) {
                  throw std::runtime_error("Array sizes do not match the number of densities.");
              }
              get_conditional_Xray(redshift, R, n_densities, densities.data(), log10_mturns.data(),
                                   out_xray.data());
          });

    // Error framework testing
    m.def("SomethingThatCatches", &SomethingThatCatches);
    m.def("FunctionThatCatches", [](bool sub_func, bool pass, nb::ndarray<double> answer) {
        return FunctionThatCatches(sub_func, pass, answer.data());
    });
    m.def("FunctionThatThrows", &FunctionThatThrows);

    m.def("single_test_sample",
          [](unsigned long long int seed, nb::ndarray<float> conditions, nb::ndarray<int> cond_crd,
             double z_out, double z_in, nb::ndarray<int> out_n_tot, nb::ndarray<int> out_n_cell,
             nb::ndarray<double> out_n_exp, nb::ndarray<double> out_m_cell,
             nb::ndarray<double> out_m_exp, nb::ndarray<float> out_halo_masses,
             nb::ndarray<int> out_halo_coords) {
              size_t n_condition = conditions.shape(0);
              if (cond_crd.shape(0) != n_condition || cond_crd.shape(1) != 3) {
                  throw std::runtime_error("cond_crd must have shape (n_condition, 3).");
              }
              if (out_n_cell.shape(0) != n_condition || out_n_exp.shape(0) != n_condition ||
                  out_m_cell.shape(0) != n_condition || out_m_exp.shape(0) != n_condition) {
                  throw std::runtime_error("Output arrays must match the number of conditions.");
              }
              int status = single_test_sample(seed, n_condition, conditions.data(), cond_crd.data(),
                                              z_out, z_in, out_n_tot.data(), out_n_cell.data(),
                                              out_n_exp.data(), out_m_cell.data(), out_m_exp.data(),
                                              out_halo_masses.data(), out_halo_coords.data());
              if (status != 0) {
                  throw std::runtime_error("single_test_sample failed with status: " +
                                           std::to_string(status));
              }
          });

    m.def("test_halo_props", [](double redshift, nb::ndarray<float> vcb_grid,
                                nb::ndarray<float> J21_LW_grid, nb::ndarray<float> z_re_grid,
                                nb::ndarray<float> Gamma12_ion_grid, nb::ndarray<float> halo_masses,
                                nb::ndarray<int> halo_coords, nb::ndarray<float> star_rng,
                                nb::ndarray<float> sfr_rng, nb::ndarray<float> xray_rng,
                                nb::ndarray<float> halo_props_out) {
        size_t n_halos = halo_masses.shape(0);
        if (halo_coords.shape(0) != n_halos || halo_coords.shape(1) != 3 ||
            star_rng.shape(0) != n_halos || sfr_rng.shape(0) != n_halos ||
            xray_rng.shape(0) != n_halos || halo_props_out.shape(0) != n_halos ||
            halo_props_out.shape(1) != 12) {
            throw std::runtime_error(
                "Input/output arrays must have the same shape as the number of halos. halo_coords "
                "shape: " +
                std::to_string(halo_coords.shape(0)) + "x" + std::to_string(halo_coords.shape(1)) +
                ", " + "halo_masses shape: " + std::to_string(halo_masses.shape(0)) + ", " +
                "star_rng shape: " + std::to_string(star_rng.shape(0)) + ", " +
                "sfr_rng shape: " + std::to_string(sfr_rng.shape(0)) + ", " +
                "halo_props_out shape: " + std::to_string(halo_props_out.shape(0)) + "x" +
                std::to_string(halo_props_out.shape(1)));
        }
        int status = test_halo_props(redshift, vcb_grid.data(), J21_LW_grid.data(),
                                     z_re_grid.data(), Gamma12_ion_grid.data(), n_halos,
                                     halo_masses.data(), halo_coords.data(), star_rng.data(),
                                     sfr_rng.data(), xray_rng.data(), halo_props_out.data());
        if (status != 0) {
            throw std::runtime_error("test_halo_props failed with status: " +
                                     std::to_string(status));
        }
    });

    m.def("test_filter", [](nb::ndarray<float> input_box, double R, double R_param, int filter_flag,
                            nb::ndarray<double> result) {
        size_t n_elements = input_box.size();
        if (result.size() != n_elements) {
            throw std::runtime_error("result array must have the same size as input_box.");
        }
        int status = test_filter(input_box.data(), R, R_param, filter_flag, result.data());
        if (status != 0) {
            throw std::runtime_error("test_filter failed with status: " + std::to_string(status));
        }
    });

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

    m.attr("photon_cons_allocated") = nb::cast(&photon_cons_allocated);
}
