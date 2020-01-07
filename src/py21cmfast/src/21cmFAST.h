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
    bool USE_FFTW_WISDOM;
    int HMF;
    int USE_RELATIVE_VELOCITIES;
    int POWER_SPECTRUM;

};

struct AstroParams{

    // Parameters taken from INIT_PARAMS.H
    float HII_EFF_FACTOR;

    float F_STAR10;
    float ALPHA_STAR;
    float F_ESC10;
    float ALPHA_ESC;
    float M_TURN;

    float R_BUBBLE_MAX;

    float ION_Tvir_MIN;

    double L_X;
    float NU_X_THRESH;
    float X_RAY_SPEC_INDEX;
    float X_RAY_Tvir_MIN;

    float t_STAR;

    int N_RSD_STEPS;
};

struct FlagOptions{

    // Parameters taken from INIT_PARAMS.H
    bool USE_MASS_DEPENDENT_ZETA;
    bool SUBCELL_RSD;
    bool INHOMO_RECO;
    bool USE_TS_FLUCT;
    bool M_MIN_in_Mass;
    bool PHOTON_CONS;
};


struct InitialConditions{
    float *lowres_density, *lowres_vx, *lowres_vy, *lowres_vz, *lowres_vx_2LPT, *lowres_vy_2LPT, *lowres_vz_2LPT, *hires_density;
    float *lowres_vcb, *hires_vcb;
};

struct PerturbedField{
    float *density, *velocity;
};

struct TsBox{
    int first_box;
    float *Ts_box;
    float *x_e_box;
    float *Tk_box;
};

struct IonizedBox{
    int first_box;
    float *xH_box;
    float *Gamma12_box;
    float *z_re_box;
    float *dNrec_box;
};

struct BrightnessTemp{
    float *brightness_temp;
};

int ComputeInitialConditions(unsigned long long random_seed, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes);

int ComputePerturbField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes, struct PerturbedField *perturbed_field);

int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options, float perturbed_field_redshift,
                  short cleanup,
                  struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp, struct TsBox *this_spin_temp);

int ComputeIonizedBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options,
                       struct PerturbedField *perturbed_field, struct IonizedBox *previous_ionize_box,
                       struct TsBox *spin_temp, struct IonizedBox *box);

int ComputeBrightnessTemp(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                           struct AstroParams *astro_params, struct FlagOptions *flag_options,
                           struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box);


int InitialisePhotonCons(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                         struct AstroParams *astro_params, struct FlagOptions *flag_options);

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
double ComputeZstart_PhotonCons();

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons);

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, double *M_uv_z, double *M_h_z, double *log10phi);

float ComputeTau(struct UserParams *user_params, struct CosmoParams *cosmo_params, int Npoints, float *redshifts, float *global_xHI);

void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_HF(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options);

void free_TsCalcBoxes(struct FlagOptions *flag_options);
bool photon_cons_inited = false;
