/*We need to explicitly define the types used by the warpper using ffi.cdef()
    However, that function does not take directives, so we separate the types here
*/
// WARNING: DO NOT #include THIS FILE IN THE C CODE EXCEPT FOR IN OutputStructs.h

typedef struct InitialConditions {
    float *lowres_density, *lowres_vx, *lowres_vy, *lowres_vz, *lowres_vx_2LPT, *lowres_vy_2LPT,
        *lowres_vz_2LPT;
    float *hires_density, *hires_vx, *hires_vy, *hires_vz, *hires_vx_2LPT, *hires_vy_2LPT,
        *hires_vz_2LPT;  // cw addition
    float *lowres_vcb;
} InitialConditions;

typedef struct PerturbedField {
    float *density, *velocity_x, *velocity_y, *velocity_z;
} PerturbedField;

typedef struct HaloCatalog {
    unsigned long long int n_halos;
    unsigned long long int buffer_size;
    float *halo_masses;
    float *halo_coords;

    // Halo properties for stochastic model
    float *star_rng;
    float *sfr_rng;
    float *xray_rng;
} HaloCatalog;

typedef struct PerturbedHaloCatalog {
    unsigned long long int n_halos;
    unsigned long long int buffer_size;
    float *halo_masses;
    float *halo_coords;

    // Halo properties
    float *sfr_acg;
    float *stellar_masses_acg;
    float *n_ion;
    float *xray_luminosity;
    float *fesc_weighted_sfr;

    float *stellar_masses_mcg;
    float *sfr_mcg;
} PerturbedHaloCatalog;

typedef struct EmissivityFields {
    // Things that aren't used in radiation fields but useful outputs
    float *halo_mass_density;
    float *stellar_mass_density_acg;
    float *stellar_mass_density_mcg;
    float *halo_number;

    // For IonisationBox.c and SpinTemperatureBox.c
    float *n_ion;               // weighted by F_ESC*PopN_ion
    float *sfrd_acg;            // for x-rays and Ts stuff
    float *xray_emissivity;     // for x-rays and Ts stuff
    float *sfrd_mcg;            // for x-rays and Ts stuff
    float *fesc_weighted_sfrd;  // SFR weighted by PopN_ion and F_ESC, used for Gamma12

    // Average volume-weighted log10 Turnover masses are kept in order to compare with the expected
    // MF integrals
    double log10_mturn_acg_ave;
    double log10_mturn_mcg_ave;
} EmissivityFields;

typedef struct RadiationFieldsSetup {
    // R-dependent arrays which are set once
    double *R_values, *zpp_avg, *zpp_edges;

    // Arrays for the filtered emissivity fields
    float *filtered_sfrd_acg_for_lya;
    float *filtered_xray_emissivity;
    float *filtered_sfrd_mcg_for_lya;
    float *filtered_sfrd_acg_for_lw;
    float *filtered_sfrd_mcg_for_lw;

    // frequency integral tables
    double *freq_int_heat_tbl, *freq_int_ion_tbl, *freq_int_lya_tbl, *freq_int_heat_tbl_diff;
    double *freq_int_ion_tbl_diff, *freq_int_lya_tbl_diff;

    // helpers for the interpolation
    float *inverse_diff;
    float *inverse_val_box;
    int *m_xHII_low_box;

    // arrays for R-dependent prefactors
    double *lya_flux_continuum_injected_prefactor_acg, *lya_flux_continuum_injected_prefactor_mcg;
    double *lyw_flux_prefactor_acg, *lyw_flux_prefactor_mcg;
    double *lya_flux_continuum_prefactor_acg, *lya_flux_injected_prefactor_acg;
    double *lya_flux_continuum_prefactor_mcg, *lya_flux_injected_prefactor_mcg;

    // array and floats required for the X-ray optical depth calculation
    double *ave_log10_MturnLW;
    double x_e_ave_zp;
    double Q_HI_zp;

    // boolean to indicate whether there's enough light
    int NO_LIGHT;
} RadiationFieldsSetup;

typedef struct RadiationFields {
    // TODO: these arrays are defined as double, but should be float - see
    // https://github.com/21cmfast/21cmFAST/issues/744
    double *xray_heating_rate;
    double *xray_ionization_rate;
    double *xray_lya_flux;
    double *lya_flux_continuum_injected;
    double *lya_flux_continuum;
    double *lya_flux_injected;
    double *lyw_flux;

    double Q_HI;
} RadiationFields;

typedef struct TsBox {
    float *spin_temperature;
    float *xray_ionised_fraction;
    float *kinetic_temp_neutral;
    float *J_21_LW;
    double Q_HI;
} TsBox;

typedef struct IonizedBox {
    double nion_unconditional_acg;
    double nion_unconditional_mcg;
    double log10_mturn_ave_acg;
    double log10_mturn_ave_mcg;
    float *neutral_fraction;
    float *ionisation_rate_G12;
    float *mean_free_path;
    float *z_reion;
    float *cumulative_recombinations;
    float *kinetic_temperature;
    float *nion_conditional_filtered_acg;
    float *nion_conditional_filtered_mcg;
} IonizedBox;

typedef struct BrightnessTemp {
    float *brightness_temp;
    float *tau_21;
} BrightnessTemp;
