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
    int INTEGRATION_METHOD_HALOS;
    bool USE_2LPT;
    bool MINIMIZE_MEMORY;
    bool KEEP_3D_VELOCITIES;
};

struct AstroParams{

    // Parameters taken from INIT_PARAMS.H
    float HII_EFF_FACTOR;

    float F_STAR10;
    float ALPHA_STAR;
    float ALPHA_STAR_MINI;
    float SIGMA_STAR;
    float SIGMA_SFR;
    float CORR_STAR;
    float CORR_SFR;
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

    double A_LW;
    double BETA_LW;
    double A_VCB;
    double BETA_VCB;

    float t_STAR;

    int N_RSD_STEPS;
};

struct FlagOptions{

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
};


struct InitialConditions{
    float *lowres_density, *lowres_vx, *lowres_vy, *lowres_vz, *lowres_vx_2LPT, *lowres_vy_2LPT, *lowres_vz_2LPT;
    float *hires_density, *hires_vx, *hires_vy, *hires_vz, *hires_vx_2LPT, *hires_vy_2LPT, *hires_vz_2LPT; //cw addition
    float *lowres_vcb;
};

struct PerturbedField{
    float *density, *velocity_x, *velocity_y, *velocity_z;
};

struct HaloField{
    long long unsigned int n_halos;
    long long unsigned int buffer_size;
    float *halo_masses;
    int *halo_coords;

    //Halo properties for stochastic model
    float *star_rng;
    float *sfr_rng;
};

//gridded halo properties
struct HaloBox{
    //Things that aren't used in radiation fields but useful outputs
    float *halo_mass;
    float *halo_stars;
    float *halo_stars_mini;
    int *count;

    //For IonisationBox.c and SpinTemperatureBox.c
    float *n_ion; //weighted by F_ESC*PopN_ion
    float *halo_sfr; //for x-rays and Ts stuff
    float *halo_sfr_mini; //for x-rays and Ts stuff
    float *whalo_sfr;

    double log10_Mcrit_LW_ave;
};

struct PerturbHaloField{
    long long unsigned int n_halos;
    long long unsigned int buffer_size;
    float *halo_masses;
    int *halo_coords;

    //Halo properties for stochastic model
    float *star_rng;
    float *sfr_rng;
};

struct TsBox{
    float *Ts_box;
    float *x_e_box;
    float *Tk_box;
    float *J_21_LW_box;
};

struct XraySourceBox{
    float *filtered_sfr;
    float *filtered_sfr_mini;

    double *mean_log10_Mcrit_LW;
    double *mean_sfr;
    double *mean_sfr_mini;
};

struct IonizedBox{
    double mean_f_coll;
    double mean_f_coll_MINI;
    double log10_Mturnover_ave;
    double log10_Mturnover_MINI_ave;
    float *xH_box;
    float *Gamma12_box;
    float *MFP_box;
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

int ComputeHaloField(float redshift_desc, float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, int random_seed, struct HaloField * halos_desc, struct HaloField *halos);

int ComputePerturbHaloField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, struct HaloField *halos, struct PerturbHaloField *halos_perturbed);

int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options, float perturbed_field_redshift,
                  short cleanup,
                  struct PerturbedField *perturbed_field, struct XraySourceBox * source_box, struct TsBox *previous_spin_temp, struct InitialConditions *ini_boxes,
                  struct TsBox *this_spin_temp);

int ComputeIonizedBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options, struct PerturbedField *perturbed_field,
                       struct PerturbedField *previous_perturbed_field, struct IonizedBox *previous_ionize_box,
                       struct TsBox *spin_temp, struct HaloBox *halos, struct InitialConditions *ini_boxes,
                       struct IonizedBox *box);

int ComputeBrightnessTemp(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                           struct AstroParams *astro_params, struct FlagOptions *flag_options,
                           struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box);

int InitialisePhotonCons(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                         struct AstroParams *astro_params, struct FlagOptions *flag_options);

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
int ComputeZstart_PhotonCons(double *zstart);

//(jdavies): I need this to be accessible in python to pass the right haloboxes to IonizeBox
float adjust_redshifts_for_photoncons(
    struct AstroParams *astro_params, struct FlagOptions *flag_options, float *redshift,
    float *stored_redshift, float *absolute_delta_z
);

void determine_deltaz_for_photoncons();

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons);

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi);

float ComputeTau(struct UserParams *user_params, struct CosmoParams *cosmo_params, int Npoints, float *redshifts, float *global_xHI);

int CreateFFTWWisdoms(struct UserParams *user_params, struct CosmoParams *cosmo_params);

void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_HF(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options);
void Broadcast_struct_global_STOC(struct UserParams *user_params, struct CosmoParams *cosmo_params,struct AstroParams *astro_params, struct FlagOptions *flag_options);
void Broadcast_struct_global_TS(struct UserParams *user_params, struct CosmoParams *cosmo_params,struct AstroParams *astro_params, struct FlagOptions *flag_options);
void Broadcast_struct_global_IT(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options);

// void free_TsCalcBoxes(struct UserParams *user_params, struct FlagOptions *flag_options);
void FreePhotonConsMemory();
bool photon_cons_allocated = false;
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double* result);
void FunctionThatThrows();
int init_heat();
void free(void *ptr);

//need a python visible function to test lower level functions using the package
int single_test_sample(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options,
                        int seed, int n_condition, float *conditions, int *cond_crd, double z_out, double z_in, int *out_n_tot, int *out_n_cell, double *out_n_exp,
                        double *out_m_cell, double *out_m_exp, float *out_halo_masses, int *out_halo_coords);

//This function, designed to be used in the wrapper to estimate Halo catalogue size, takes the parameters and returns average number of halos within the box
double expected_nhalo(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions * flag_options);

//these two functions compute the new classes HaloBox and XraySourceBox, and need to be visible
int ComputeHaloBox(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params
                    , struct FlagOptions * flag_options, struct InitialConditions * ini_boxes, struct PerturbedField * perturbed_field, struct PerturbHaloField *halos
                    , struct TsBox *previous_spin_temp, struct IonizedBox *previous_ionize_box, struct HaloBox *grids);

int UpdateXraySourceBox(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options, struct HaloBox *halobox,
                  double R_inner, double R_outer, int R_ct, struct XraySourceBox *source_box);

//alpha photoncons functions
void set_alphacons_params(double norm, double slope);

//exposing lower-level functions for testing
//Initialisation of tables
double init_ps();
double dicke(double z);
double sigma_z0(double M);
double dsigmasqdm_z0(double M);
double get_delta_crit(int HMF, double sigma, double growthf);
void initialiseSigmaMInterpTable(float M_Min, float M_Max);
double EvaluateSigma(double lnM);
double EvaluatedSigmasqdm(double lnM);

void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI,
                             float mturn_a_const, bool minihalos);
void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10,
                                float Fesc10, float Fstar7_MINI, float Fesc7_MINI, float mturn_a_const, bool minihalos);
void initialise_FgtrM_delta_table(double min_dens, double max_dens, double zpp, double growth_zpp, double smin_zpp, double smax_zpp);
void init_FcollTable(double zmin, double zmax, bool x_ray);
void initialise_Nion_Conditional_spline(float z, float Mcrit_atom, float min_density, float max_density,
                                     float Mmin, float Mmax, float Mcond, float log10Mturn_min, float log10Mturn_max,
                                     float log10Mturn_min_MINI, float log10Mturn_max_MINI, float Alpha_star,
                                     float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10,
                                     float Mlim_Fstar, float Mlim_Fesc, float Fstar7_MINI, float Fesc7_MINI,
                                     float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, int method, int method_mini,
                                     bool minihalos, bool prev);
void initialise_SFRD_Conditional_table(double min_density, double max_density, double growthf,
                                    float Mcrit_atom, double Mmin, double Mmax, double Mcond, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI, int method, int method_mini, bool minihalos);

void initialise_dNdM_tables(double xmin, double xmax, double ymin, double ymax, double growth1, double param, bool from_catalog);

//evaluation of tables
double EvaluateNionTs(double redshift, double Mlim_Fstar, double Mlim_Fesc);
double EvaluateNionTs_MINI(double redshift, double log10_Mturn_LW_ave, double Mlim_Fstar_MINI, double Mlim_Fesc_MINI);
double EvaluateSFRD(double redshift, double Mlim_Fstar);
double EvaluateSFRD_MINI(double redshift, double log10_Mturn_LW_ave, double Mlim_Fstar_MINI);
double EvaluateSFRD_Conditional(double delta, double growthf, double M_min, double M_max, double M_cond, double sigma_max, double Mturn_a, double Mlim_Fstar);
double EvaluateSFRD_Conditional_MINI(double delta, double log10Mturn_m, double growthf, double M_min, double M_max, double M_cond, double sigma_max, double Mturn_a, double Mlim_Fstar);
double EvaluateNion_Conditional(double delta, double log10Mturn, double growthf, double M_min, double M_max, double M_cond, double sigma_max,
                                double Mlim_Fstar, double Mlim_Fesc, bool prev);
double EvaluateNion_Conditional_MINI(double delta, double log10Mturn_m, double growthf, double M_min, double M_max, double M_cond, double sigma_max,
                                    double Mturn_a, double Mlim_Fstar, double Mlim_Fesc, bool prev);
double EvaluateNhalo(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);
double EvaluateMcoll(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);
double EvaluateNhaloInv(double condition, double prob);
double EvaluateFcoll_delta(double delta, double growthf, double sigma_min, double sigma_max);
double EvaluatedFcolldz(double delta, double redshift, double sigma_min, double sigma_max);

//integrals
void initialise_GL(int n, float lnM_Min, float lnM_Max);
double Nhalo_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method);
double Mcoll_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method);
double Nion_ConditionalM(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
                        double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar,
                        double Mlim_Fesc, int method);
double Nion_ConditionalM_MINI(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
                            double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar7,
                            double Fesc7, double Mlim_Fstar, double Mlim_Fesc, int method);
double Nion_General(double z, double lnM_Min, double lnM_Max, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10,
                     double Fesc10, double Mlim_Fstar, double Mlim_Fesc);
double Nion_General_MINI(double z, double lnM_Min, double lnM_Max, double MassTurnover, double MassTurnover_upper, double Alpha_star,
                         double Alpha_esc, double Fstar7_MINI, double Fesc7_MINI, double Mlim_Fstar, double Mlim_Fesc);
double FgtrM_General(double z, double M);
double unconditional_mf(double growthf, double lnM, double z, int HMF);
double conditional_mf(double growthf, double lnM, double delta_cond, double sigma_cond, int HMF);
double atomic_cooling_threshold(float z);
