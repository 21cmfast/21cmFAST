/*
    This is the header file for the wrappable version of 21cmFAST, or 21cmMC.
    It contains function signatures, struct definitions and globals to which the Python wrapper code
    requires access.
*/


/*
    --------------------------------------------------------------------------------------------------------------------
    PARAMETER STRUCTURES (these should be trimmed accordingly)
    --------------------------------------------------------------------------------------------------------------------
*/
struct FlagOptions{
    int INCLUDE_ZETA_PL;   // Requires work... see comment in function.
    int READ_FROM_FILE;    // Read parameters from file rather than variable.
    int GenerateNewICs;    // Whether to create a new density field at each sampling (i.e. new initial conditions). Must use if the cosmology is being varied
    int SUBCELL_RSD;       // Whether to include redshift space distortions along the line-of-sight (z-direction only).
    int USE_FCOLL_IONISATION_TABLE; //Whether to use an interpolation for the collapsed fraction for the find_HII_bubbles part of the computation
    int SHORTEN_FCOLL;     // Whether to use an interpolation for the collapsed fraction for the Ts.c computation
    int USE_TS_FLUCT;      // Whether to perform the full evolution of the IGM spin temperature, or just assume the saturated spin temperature limit
    int INHOMO_RECO;       // Whether to include inhomogeneous recombinations into the calculation of the ionisation fraction
    double *redshifts;    // this shouldb't be in the flag options struct!!!
};

struct AstroParams{
    float EFF_FACTOR_PL_INDEX;
    float HII_EFF_FACTOR;
    float R_BUBBLE_MAX;
    float ION_Tvir_MIN;
    float L_X;
    float NU_X_THRESH;
    float NU_X_BAND_MAX;
    float NU_X_MAX;
    float X_RAY_SPEC_INDEX;
    float X_RAY_Tvir_MIN;
    float X_RAY_Tvir_LOWERBOUND;
    float X_RAY_Tvir_UPPERBOUND;
    float F_STAR;
    float t_STAR;
    int N_RSD_STEPS;
    int LOS_direction;
    float Z_HEAT_MAX;
    float ZPRIME_STEP_FACTOR;
};

struct CosmoParams{
    unsigned long long RANDOM_SEED;
    float SIGMA8;
    float hlittle;
    float OMm;
    float OMl;
    float OMb;
    float POWER_INDEX;

};


struct BoxDim{
    int HII_DIM;
    int DIM;
    float BOX_LEN;
};


/*
    --------------------------------------------------------------------------------------------------------------------
    OUTPUT STRUCTURES
    --------------------------------------------------------------------------------------------------------------------
*/

struct InitBoxes{
    float *lowres_density;
    float *lowres_vx;
    float *lowres_vy;
    float *lowres_vz;
    float *lowres_vx_2LPT;
    float *lowres_vy_2LPT;
    float *lowres_vz_2LPT;
    float *hires_density;
};

struct PerturbField{
    float *density;
    float *velocity;
}


/*
    --------------------------------------------------------------------------------------------------------------------
    USER-FACING FUNCTIONS
    --------------------------------------------------------------------------------------------------------------------
*/
int ComputeInitialConditions(struct BoxDim *box_dim, struct CosmoParams *cosmo_params, struct InitBoxes *boxes);
int ComputePerturbField(float redshift, struct InitBoxes *init_boxes, struct PerturbField *field);
int ComputeIonizationField(float *redshifts, struct FlagOptions *flag_options, struct AstroParams *astro_params,
                           struct PerturbField *field, ...); // Not sure how this should be as yet

