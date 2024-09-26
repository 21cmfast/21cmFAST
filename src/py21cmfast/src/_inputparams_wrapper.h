/*We need to explicitly define the types used by the warpper using ffi.cdef()
    However, that function does not take directives, so we separate the types here
*/
//WARNING: DO NOT #include THIS FILE IN THE C CODE EXCEPT FOR IN InputParameters.h


typedef struct CosmoParams{

    float SIGMA_8;
    float hlittle;
    float OMm;
    float OMl;
    float OMb;
    float POWER_INDEX;

} CosmoParams;

typedef struct UserParams{
    // Parameters taken from INIT_PARAMS.H
    int HII_DIM;
    int DIM;
    float BOX_LEN;
    float NON_CUBIC_FACTOR;
    bool USE_FFTW_WISDOM;
    int HMF;
    int USE_RELATIVE_VELOCITIES;
    int POWER_SPECTRUM;
    int N_THREADS;
    bool PERTURB_ON_HIGH_RES;
    bool NO_RNG;
    bool USE_INTERPOLATION_TABLES;
    int INTEGRATION_METHOD_ATOMIC;
    int INTEGRATION_METHOD_MINI;
    bool USE_2LPT;
    bool MINIMIZE_MEMORY;
    bool KEEP_3D_VELOCITIES;

    //Halo Sampler Options
    float SAMPLER_MIN_MASS;
    double SAMPLER_BUFFER_FACTOR;
    float MAXHALO_FACTOR;
    int N_COND_INTERP;
    int N_PROB_INTERP;
    double MIN_LOGPROB;
    int SAMPLE_METHOD;
    bool AVG_BELOW_SAMPLER;
    double HALOMASS_CORRECTION;
    double PARKINSON_G0;
    double PARKINSON_y1;
    double PARKINSON_y2;
} UserParams;

typedef struct AstroParams{
    // Parameters taken from INIT_PARAMS.H
    float HII_EFF_FACTOR;

    //SHMR
    float F_STAR10;
    float ALPHA_STAR;
    float ALPHA_STAR_MINI;
    float SIGMA_STAR;
    float CORR_STAR;
    double UPPER_STELLAR_TURNOVER_MASS;
    double UPPER_STELLAR_TURNOVER_INDEX;
    float F_STAR7_MINI;

    //SFMS
    float t_STAR;
    float CORR_SFR;
    double SIGMA_SFR_INDEX;
    double SIGMA_SFR_LIM;

    //L_X/SFR
    double L_X;
    double L_X_MINI;
    double SIGMA_LX;
    double CORR_LX;

    //Escape Fraction
    float F_ESC10;
    float ALPHA_ESC;
    float F_ESC7_MINI;

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

    int N_RSD_STEPS;
} AstroParams;

typedef struct FlagOptions{
    // Parameters taken from INIT_PARAMS.H
    bool USE_HALO_FIELD;
    bool USE_MINI_HALOS;
    bool USE_CMB_HEATING; //CMB Heating Flag
    bool USE_LYA_HEATING; //Lya Heating Flag
    bool USE_MASS_DEPENDENT_ZETA;
    bool SUBCELL_RSD;
    bool APPLY_RSDS;
    bool INHOMO_RECO;
    bool USE_TS_FLUCT;
    bool M_MIN_in_Mass;
    bool FIX_VCB_AVG;
    bool HALO_STOCHASTICITY;
    bool USE_EXP_FILTER;
    bool FIXED_HALO_GRIDS;
    bool CELL_RECOMB;
    int PHOTON_CONS_TYPE;
    bool USE_UPPER_STELLAR_TURNOVER;
    bool HALO_SCALING_RELATIONS_MEDIAN;
} FlagOptions;

typedef struct GlobalParams{
    float ALPHA_UVB;
    int EVOLVE_DENSITY_LINEARLY;
    int SMOOTH_EVOLVED_DENSITY_FIELD;
    float R_smooth_density;
    float HII_ROUND_ERR;
    int FIND_BUBBLE_ALGORITHM;
    int N_POISSON;
    int T_USE_VELOCITIES;
    float MAX_DVDR;
    float DELTA_R_HII_FACTOR;
    float DELTA_R_FACTOR;
    int HII_FILTER;
    float INITIAL_REDSHIFT;
    float R_OVERLAP_FACTOR;
    int DELTA_CRIT_MODE;
    int HALO_FILTER;
    int OPTIMIZE;
    float OPTIMIZE_MIN_MASS;


    float CRIT_DENS_TRANSITION;
    float MIN_DENSITY_LOW_LIMIT;

    int RecombPhotonCons;
    float PhotonConsStart;
    float PhotonConsEnd;
    float PhotonConsAsymptoteTo;
    float PhotonConsEndCalibz;
    int PhotonConsSmoothing;

    int HEAT_FILTER;
    double CLUMPING_FACTOR;
    float Z_HEAT_MAX;
    float R_XLy_MAX;
    int NUM_FILTER_STEPS_FOR_Ts;
    float ZPRIME_STEP_FACTOR;
    double TK_at_Z_HEAT_MAX;
    double XION_at_Z_HEAT_MAX;
    int Pop;
    float Pop2_ion;
    float Pop3_ion;

    float NU_X_BAND_MAX;
    float NU_X_MAX;

    int NBINS_LF;

    int P_CUTOFF;
    float M_WDM;
    float g_x;
    float OMn;
    float OMk;
    float OMr;
    float OMtot;
    float Y_He;
    float wl;
    float SHETH_b;
    float SHETH_c;
    double Zreion_HeII;
    int FILTER;

    char *external_table_path;
    char *wisdoms_path;
    float R_BUBBLE_MIN;
    float M_MIN_INTEGRAL;
    float M_MAX_INTEGRAL;

    float T_RE;

    float VAVG;

    bool USE_ADIABATIC_FLUCTUATIONS;
}GlobalParams;


/* Previously, we had a few structures spread throughout the code e.g user_params_ufunc which
   were all globally defined and separately broadcast at different times. Several of these were used
   across different files and some inside #defines (e.g indexing.h), so for now I've combined
   the parameter structures to avoid confusion (we shouldn't have the possibility of two files using
   different parameters).

   In future we should have a parameter structure in each .c file containing ONLY parameters relevant to it
   (look at HaloBox.c), and force the broadcast at each _compute() step (or even decorate any library call)
   However this would require us to be very careful about initialising the globals when ANY function from that
   file is called */
extern UserParams *user_params_global;
extern CosmoParams *cosmo_params_global;
extern AstroParams *astro_params_global;
extern FlagOptions *flag_options_global;

extern GlobalParams global_params;
