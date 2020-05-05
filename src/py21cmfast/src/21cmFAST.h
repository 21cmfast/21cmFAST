/*
    This is the header file for the wrappable version of 21cmFAST, or 21cmMC.
    It contains function signatures, struct definitions and globals to which the Python wrapper code
    requires access.
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
    int N_THREADS;
    bool PERTURB_ON_HIGH_RES;
    bool NO_RNG;
};

struct AstroParams{

    // Parameters taken from INIT_PARAMS.H
    float HII_EFF_FACTOR;

    float F_STAR10;
    float ALPHA_STAR;
    float F_ESC10;
    float ALPHA_ESC;
    float M_TURN;
    float F_STAR7_MINI;
    float F_ESC7_MINI;
    float R_BUBBLE_MAX;
    float ION_Tvir_MIN;
    double F_H2_SHIELD;
    double L_X;
    double L_X_MINI;
    float NU_X_THRESH;
    float X_RAY_SPEC_INDEX;
    float X_RAY_Tvir_MIN;

    float t_STAR;

    int N_RSD_STEPS;
};

struct FlagOptions{

    // Parameters taken from INIT_PARAMS.H
    bool USE_MINI_HALOS;
    bool USE_MASS_DEPENDENT_ZETA;
    bool SUBCELL_RSD;
    bool INHOMO_RECO;
    bool USE_TS_FLUCT;
    bool M_MIN_in_Mass;
    bool PHOTON_CONS;
};


struct InitialConditions{
    float *lowres_density, *lowres_vx, *lowres_vy, *lowres_vz, *lowres_vx_2LPT, *lowres_vy_2LPT, *lowres_vz_2LPT;
    float *hires_density, *hires_vx, *hires_vy, *hires_vz, *hires_vx_2LPT, *hires_vy_2LPT, *hires_vz_2LPT; //cw addition
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
    float *J_21_LW_box;
};

struct IonizedBox{
    int first_box;
    double mean_f_coll;
    double mean_f_coll_MINI;
    double log10_Mturnover_ave;
    double log10_Mturnover_MINI_ave;
    float *xH_box;
    float *Gamma12_box;
    float *z_re_box;
    float *dNrec_box;
    float *temp_kinetic_all_gas;
    float *Fcoll;
    float *Fcoll_MINI;
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
                       struct AstroParams *astro_params, struct FlagOptions *flag_options, struct PerturbedField *perturbed_field,
                       struct PerturbedField *previous_perturbed_field, struct IonizedBox *previous_ionize_box,
                       struct TsBox *spin_temp, struct IonizedBox *box);

int ComputeBrightnessTemp(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                           struct AstroParams *astro_params, struct FlagOptions *flag_options,
                           struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box);

int InitialisePhotonCons(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                         struct AstroParams *astro_params, struct FlagOptions *flag_options);

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
int ComputeZstart_PhotonCons(double *zstart);

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons);

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi);

float ComputeTau(struct UserParams *user_params, struct CosmoParams *cosmo_params, int Npoints, float *redshifts, float *global_xHI);

void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_HF(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options);

void free_TsCalcBoxes(struct FlagOptions *flag_options);
void FreePhotonConsMemory();
bool photon_cons_allocated = false;
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double* result);
void FunctionThatThrows();
