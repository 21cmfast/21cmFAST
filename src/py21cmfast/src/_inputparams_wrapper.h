/*We need to explicitly define the types used by the warpper using ffi.cdef()
    However, that function does not take directives, so we separate the types here
*/
// WARNING: DO NOT #include THIS FILE IN THE C CODE EXCEPT FOR IN InputParameters.h

typedef struct CosmoParams {
    float SIGMA_8;
    float hlittle;
    float OMm;
    float OMl;
    float OMb;
    float POWER_INDEX;

    float OMn;
    float OMk;
    float OMr;
    float OMtot;
    float Y_He;
    float wl;

} CosmoParams;

typedef struct SimulationOptions {
    // Parameters taken from INIT_PARAMS.H
    int HII_DIM;
    int DIM;
    float BOX_LEN;
    float NON_CUBIC_FACTOR;
    int N_THREADS;
    double Z_HEAT_MAX;
    double ZPRIME_STEP_FACTOR;

    // Halo Sampler Options
    float SAMPLER_MIN_MASS;
    double SAMPLER_BUFFER_FACTOR;
    int N_COND_INTERP;
    int N_PROB_INTERP;
    double MIN_LOGPROB;
    double HALOMASS_CORRECTION;
    double PARKINSON_G0;
    double PARKINSON_y1;
    double PARKINSON_y2;

    float INITIAL_REDSHIFT;
    double DELTA_R_FACTOR;
    double DENSITY_SMOOTH_RADIUS;

    double DEXM_OPTIMIZE_MINMASS;
    double DEXM_R_OVERLAP;

    double CORR_STAR;
    double CORR_SFR;
    double CORR_LX;
} SimulationOptions;

typedef struct MatterOptions {
    bool USE_FFTW_WISDOM;
    int HMF;
    int USE_RELATIVE_VELOCITIES;
    int POWER_SPECTRUM;
    int USE_INTERPOLATION_TABLES;
    bool PERTURB_ON_HIGH_RES;
    int PERTURB_ALGORITHM;
    bool MINIMIZE_MEMORY;
    bool KEEP_3D_VELOCITIES;
    bool DEXM_OPTIMIZE;
    int FILTER;
    int HALO_FILTER;
    bool SMOOTH_EVOLVED_DENSITY_FIELD;

    bool USE_HALO_FIELD;
    bool HALO_STOCHASTICITY;
    bool FIXED_HALO_GRIDS;
    int SAMPLE_METHOD;
} MatterOptions;

typedef struct AstroParams {
    float HII_EFF_FACTOR;

    // SHMR
    float F_STAR10;
    float ALPHA_STAR;
    float ALPHA_STAR_MINI;
    float SIGMA_STAR;
    double UPPER_STELLAR_TURNOVER_MASS;
    double UPPER_STELLAR_TURNOVER_INDEX;
    float F_STAR7_MINI;

    // SFMS
    float t_STAR;
    double SIGMA_SFR_INDEX;
    double SIGMA_SFR_LIM;

    // L_X/SFR
    double L_X;
    double L_X_MINI;
    double SIGMA_LX;

    // Escape Fraction
    float F_ESC10;
    float ALPHA_ESC;
    float BETA_ESC;
    float F_ESC7_MINI;

    float T_RE;

    float M_TURN;
    float R_BUBBLE_MAX;
    float ION_Tvir_MIN;
    double F_H2_SHIELD;
    float NU_X_THRESH;
    float X_RAY_SPEC_INDEX;
    float X_RAY_Tvir_MIN;

    double A_LW;
    double BETA_LW;
    double A_VCB;
    double BETA_VCB;

    double FIXED_VAVG;
    double POP2_ION;
    double POP3_ION;

    int N_RSD_STEPS;
    double PHOTONCONS_CALIBRATION_END;
    double CLUMPING_FACTOR;
    double ALPHA_UVB;

    float R_MAX_TS;
    int N_STEP_TS;
    double DELTA_R_HII_FACTOR;
    float R_BUBBLE_MIN;
    double MAX_DVDR;
    double NU_X_MAX;
    double NU_X_BAND_MAX;
} AstroParams;

typedef struct AstroOptions {
    bool USE_MINI_HALOS;
    bool USE_CMB_HEATING;  // CMB Heating Flag
    bool USE_LYA_HEATING;  // Lya Heating Flag
    bool USE_MASS_DEPENDENT_ZETA;
    bool SUBCELL_RSD;
    bool APPLY_RSDS;
    bool INHOMO_RECO;
    bool USE_TS_FLUCT;
    bool M_MIN_in_Mass;
    bool FIX_VCB_AVG;
    bool USE_EXP_FILTER;
    bool CELL_RECOMB;
    int PHOTON_CONS_TYPE;
    bool USE_UPPER_STELLAR_TURNOVER;
    bool HALO_SCALING_RELATIONS_MEDIAN;
    int HII_FILTER;
    int HEAT_FILTER;
    bool IONISE_ENTIRE_SPHERE;
    bool AVG_BELOW_SAMPLER;
    int INTEGRATION_METHOD_ATOMIC;
    int INTEGRATION_METHOD_MINI;
} AstroOptions;

typedef struct ConfigSettings {
    double HALO_CATALOG_MEM_FACTOR;

    char *external_table_path;
    char *wisdoms_path;
} ConfigSettings;

/* Previously, we had a few structures spread throughout the code e.g simulation_options_ufunc which
   were all globally defined and separately broadcast at different times. Several of these were used
   across different files and some inside #defines (e.g indexing.h), so for now I've combined
   the parameter structures to avoid confusion (we shouldn't have the possibility of two files using
   different parameters).

   In future we should have a parameter structure in each .c file containing ONLY parameters
   relevant to it (look at HaloBox.c), and force the broadcast at each _compute() step (or even
   decorate any library call) However this would require us to be very careful about initialising
   the globals when ANY function from that file is called */
// The structs declared here defined in InputParameters.c
extern SimulationOptions *simulation_options_global;
extern MatterOptions *matter_options_global;
extern CosmoParams *cosmo_params_global;
extern AstroParams *astro_params_global;
extern AstroOptions *astro_options_global;

extern ConfigSettings config_settings;
