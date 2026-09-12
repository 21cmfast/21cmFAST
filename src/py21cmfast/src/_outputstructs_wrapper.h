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
    float *sfr;
    float *stellar_masses;
    float *ion_emissivity;
    float *xray_emissivity;
    float *fesc_sfr;

    float *stellar_mini;
    float *sfr_mini;
} PerturbedHaloCatalog;

typedef struct HaloBox {
    // Things that aren't used in radiation fields but useful outputs
    float *halo_mass;
    float *halo_stars;
    float *halo_stars_mini;
    float *count;

    // For IonisationBox.c and SpinTemperatureBox.c
    float *n_ion;     // weighted by F_ESC*PopN_ion
    float *halo_sfr;  // for x-rays and Ts stuff
    float *halo_xray;
    float *halo_sfr_mini;  // for x-rays and Ts stuff
    float *whalo_sfr;      // SFR weighted by PopN_ion and F_ESC, used for Gamma12

    // Average volume-weighted log10 Turnover masses are kept in order to compare with the expected
    // MF integrals
    double log10_Mcrit_ACG_ave;
    double log10_Mcrit_MCG_ave;
} HaloBox;

typedef struct RadiationFieldsSetup {
    // R-dependent arrays which are set once
    double *R_values, *zpp_avg, *zpp_edges;

    // Arrays for the filtered emissivity fields
    float *filtered_sfr;
    float *filtered_xray;
    float *filtered_sfr_mini;
    float *filtered_sfr_lw;
    float *filtered_sfr_mini_lw;

    // frequency integral tables
    double *freq_int_heat_tbl, *freq_int_ion_tbl, *freq_int_lya_tbl, *freq_int_heat_tbl_diff;
    double *freq_int_ion_tbl_diff, *freq_int_lya_tbl_diff;

    // helpers for the interpolation
    float *inverse_diff;
    float *inverse_val_box;
    int *m_xHII_low_box;

    // arrays for R-dependent prefactors
    double *lya_flux_continuum_injected_prefactor, *lya_flux_continuum_injected_prefactor_MINI;
    double *lyw_flux_prefactor, *lyw_flux_prefactor_MINI;
    double *lya_flux_continuum_prefactor, *lya_flux_injected_prefactor;
    double *lya_flux_continuum_prefactor_MINI, *lya_flux_injected_prefactor_MINI;

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
    double mean_f_coll;
    double mean_f_coll_MINI;
    double log10_Mturnover_ave;
    double log10_Mturnover_MINI_ave;
    float *neutral_fraction;
    float *ionisation_rate_G12;
    float *mean_free_path;
    float *z_reion;
    float *cumulative_recombinations;
    float *kinetic_temperature;
    float *unnormalised_nion;
    float *unnormalised_nion_mini;
} IonizedBox;

typedef struct BrightnessTemp {
    float *brightness_temp;
    float *tau_21;
} BrightnessTemp;
