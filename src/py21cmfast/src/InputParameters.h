#ifndef _PARAMSTRUCTURES_H
#define _PARAMSTRUCTURES_H

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
} FlagOptions;

#endif
