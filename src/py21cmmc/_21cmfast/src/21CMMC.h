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

struct CosmoParams{
    
    unsigned long long RANDOM_SEED;
    float SIGMA_8;
    float hlittle;
    float OMm;
    float OMl;
    float OMb;
    float POWER_INDEX;
    
};

struct UserParams{
    
    // Parameters taken from INIT_PARAMS.H
    int HII_DIM;
    int DIM;
    float BOX_LEN;
    
};


struct InitialConditions{
//    float *lowres_density, *hires_density, *lowres_vz, *lowres_vz_2LPT,PSnormalisation;
//    struct UserParams user_params;
//    struct CosmoParams cosmo_params;
    float *lowres_density, *hires_density, *lowres_vx, *lowres_vy, *lowres_vz, *lowres_vx_2LPT, *lowres_vy_2LPT, *lowres_vz_2LPT;
//    struct UserParams user_params;
//    struct CosmoParams cosmo_params;
};

struct PerturbedField{
    float *density, *velocity;
};

struct TsBox{
    int first_box;
    float *Ts_box;
};

struct IonizedBox{
    int first_box;
    float *ionized_box;
};

struct BrightnessTemp{
    float *brightness_temp;
};

void ComputeInitialConditions(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes);
//void ComputePerturbField(float redshift, struct InitialConditions *boxes, struct PerturbedField *p_cubes);
void ComputePerturbField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes, struct PerturbedField *p_cubes);
void ComputeTsBox(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, float perturbed_field_redshift,
                  struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp, struct TsBox this_spin_temp);
void ComputeIonizedBox(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options,
                       struct PerturbedField *perturbed_field(), struct IonizedBox *previous_ionize_box,
                       int do_spin_temp, struct TsBox *spin_temp, struct IonizedBox *box)
void ComputeBrightnessTemp(float redshift, int saturated_limit, struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box)

void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params);
