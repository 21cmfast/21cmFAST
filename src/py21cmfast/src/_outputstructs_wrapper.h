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

typedef struct HaloField {
    long long unsigned int n_halos;
    long long unsigned int buffer_size;
    float *halo_masses;
    float *halo_coords;

    // Halo properties for stochastic model
    float *star_rng;
    float *sfr_rng;
    float *xray_rng;
} HaloField;

typedef struct PerturbHaloField {
    long long unsigned int n_halos;
    long long unsigned int buffer_size;
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
} PerturbHaloField;

typedef struct HaloBox {
    // Things that aren't used in radiation fields but useful outputs
    float *halo_mass;
    float *halo_stars;
    float *halo_stars_mini;
    int *count;

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

typedef struct XraySourceBox {
    float *filtered_sfr;
    float *filtered_xray;
    float *filtered_sfr_mini;

    double *mean_log10_Mcrit_LW;
    double *mean_sfr;
    double *mean_sfr_mini;
} XraySourceBox;

typedef struct TsBox {
    float *spin_temperature;
    float *xray_ionised_fraction;
    float *kinetic_temp_neutral;
    float *J_21_LW;
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
