/* This file contains the repeated function prototypes which are needed by CFFI
    to be included explicitly via ffi.cdef(), These are the only functions which
    are visible to the python wrapper */

/* OutputStruct COMPUTE FUNCTIONS */
int ComputeInitialConditions(unsigned long long random_seed,  UserParams *user_params,  CosmoParams *cosmo_params,  InitialConditions *boxes);

int ComputePerturbField(float redshift,  UserParams *user_params,  CosmoParams *cosmo_params,  InitialConditions *boxes,  PerturbedField *perturbed_field);

int ComputeHaloField(float redshift_desc, float redshift,  UserParams *user_params,  CosmoParams *cosmo_params,
                      AstroParams *astro_params,  FlagOptions *flag_options,
                      InitialConditions *boxes, unsigned long long int random_seed,  HaloField * halos_desc,  HaloField *halos);

int ComputePerturbHaloField(float redshift,  UserParams *user_params,  CosmoParams *cosmo_params,
                      AstroParams *astro_params,  FlagOptions *flag_options,
                      InitialConditions *boxes,  HaloField *halos,  PerturbHaloField *halos_perturbed);

int ComputeTsBox(float redshift, float prev_redshift,  UserParams *user_params,  CosmoParams *cosmo_params,
                   AstroParams *astro_params,  FlagOptions *flag_options, float perturbed_field_redshift,
                  short cleanup,
                   PerturbedField *perturbed_field,  XraySourceBox * source_box,  TsBox *previous_spin_temp,  InitialConditions *ini_boxes,
                   TsBox *this_spin_temp);

int ComputeIonizedBox(float redshift, float prev_redshift,  UserParams *user_params,  CosmoParams *cosmo_params,
                        AstroParams *astro_params,  FlagOptions *flag_options,  PerturbedField *perturbed_field,
                        PerturbedField *previous_perturbed_field,  IonizedBox *previous_ionize_box,
                        TsBox *spin_temp,  HaloBox *halos,  InitialConditions *ini_boxes,
                        IonizedBox *box);

int ComputeBrightnessTemp(float redshift,  UserParams *user_params,  CosmoParams *cosmo_params,
                            AstroParams *astro_params,  FlagOptions *flag_options,
                            TsBox *spin_temp,  IonizedBox *ionized_box,
                            PerturbedField *perturb_field,  BrightnessTemp *box);

int ComputeHaloBox(double redshift,  UserParams *user_params,  CosmoParams *cosmo_params,  AstroParams *astro_params
                    ,  FlagOptions * flag_options,  InitialConditions * ini_boxes,  PerturbedField * perturbed_field,  PerturbHaloField *halos
                    ,  TsBox *previous_spin_temp,  IonizedBox *previous_ionize_box,  HaloBox *grids);

int UpdateXraySourceBox( UserParams *user_params,  CosmoParams *cosmo_params,
                   AstroParams *astro_params,  FlagOptions *flag_options,  HaloBox *halobox,
                  double R_inner, double R_outer, int R_ct,  XraySourceBox *source_box);
/*--------------------------*/

/* PHOTON CONSERVATION MODEL FUNCTIONS */
int InitialisePhotonCons( UserParams *user_params,  CosmoParams *cosmo_params,
                          AstroParams *astro_params,  FlagOptions *flag_options);

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
int ComputeZstart_PhotonCons(double *zstart);

void adjust_redshifts_for_photoncons(UserParams *user_params,
     AstroParams *astro_params,  FlagOptions *flag_options, float *redshift,
    float *stored_redshift, float *absolute_delta_z
);

void determine_deltaz_for_photoncons();

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons);

void FreePhotonConsMemory();
extern bool photon_cons_allocated;

void set_alphacons_params(double norm, double slope);
/* ------------------------------- */

/* Non-OutputStruct data products */
int ComputeLF(int nbins,  UserParams *user_params,  CosmoParams *cosmo_params,  AstroParams *astro_params,
                FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi);

float ComputeTau(UserParams *user_params, CosmoParams *cosmo_params, int NPoints, float *redshifts, float *global_xHI, float z_re_HeII);
/*-----------------------------*/


/* Initialisation functions needed in the wrapper*/
double init_ps();
int init_heat();
int CreateFFTWWisdoms( UserParams *user_params,  CosmoParams *cosmo_params);
void Broadcast_struct_global_noastro( UserParams *user_params,  CosmoParams *cosmo_params);
void Broadcast_struct_global_all( UserParams *user_params,  CosmoParams *cosmo_params,  AstroParams *astro_params,  FlagOptions *flag_options);
/*---------------------------*/

/* Interpolation Table Functions */
//Initialisation
void initialiseSigmaMInterpTable(float M_Min, float M_Max);
void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI);
void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10,
                                float Fesc10, float Fstar7_MINI, float Fesc7_MINI);
void initialise_FgtrM_delta_table(double min_dens, double max_dens, double zpp, double growth_zpp, double smin_zpp, double smax_zpp);
void init_FcollTable(double zmin, double zmax, bool x_ray);
void initialise_Nion_Conditional_spline(double z, double min_density, double max_density,
                                     double Mmin, double Mmax, double Mcond, double log10Mturn_min, double log10Mturn_max,
                                     double log10Mturn_min_MINI, double log10Mturn_max_MINI, float Alpha_star,
                                     float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10,
                                     float Fstar7_MINI, float Fesc7_MINI, bool prev);
void initialise_SFRD_Conditional_table(double z, double min_density, double max_density,
                                    double Mmin, double Mmax, double Mcond, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI);
void initialise_Xray_Conditional_table(double min_density, double max_density, double redshift,
                                    double Mmin, double Mmax, double Mcond, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI, double l_x, double l_x_mini, double t_h, double t_star);

void initialise_dNdM_tables(double xmin, double xmax, double ymin, double ymax, double growth1, double param, bool from_catalog);
void initialise_dNdM_inverse_table(double xmin, double xmax, double lnM_min, double growth1, double param, bool from_catalog);

//Evaluation
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
double EvaluateXray_Conditional(double delta, double log10Mturn_m, double redshift, double growthf, double M_min, double M_max, double M_cond, double sigma_max,
                                     double Mturn_a, double t_h, double Mlim_Fstar, double Mlim_Fstar_MINI);
double EvaluateNhalo(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);
double EvaluateMcoll(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);
double EvaluateNhaloInv(double condition, double prob);
double EvaluateFcoll_delta(double delta, double growthf, double sigma_min, double sigma_max);
double EvaluatedFcolldz(double delta, double redshift, double sigma_min, double sigma_max);
double EvaluateSigma(double lnM);
double EvaluatedSigmasqdm(double lnM);

/*--------------------------------*/

/*HMF Integrals*/
void initialise_GL(double lnM_Min, double lnM_Max);
double Nhalo_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method);
double Mcoll_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method);
double Nion_ConditionalM(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
                        double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar,
                        double Mlim_Fesc, int method);
double Nion_ConditionalM_MINI(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
                            double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar7,
                            double Fesc7, double Mlim_Fstar, double Mlim_Fesc, int method);
double Xray_ConditionalM(double redshift, double growthf, double lnM1, double lnM2, double lnM_cond, double sigma2, double delta2,
                         double mturn_acg, double mturn_mcg,
                        double Alpha_star, double Alpha_star_mini, double Fstar10, double Fstar7, double Mlim_Fstar,
                        double Mlim_Fstar_mini, double l_x, double l_x_mini, double t_h, double t_star, int method);
double Nion_General(double z, double lnM_Min, double lnM_Max, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10,
                     double Fesc10, double Mlim_Fstar, double Mlim_Fesc);
double Nion_General_MINI(double z, double lnM_Min, double lnM_Max, double MassTurnover, double MassTurnover_upper, double Alpha_star,
                         double Alpha_esc, double Fstar7_MINI, double Fesc7_MINI, double Mlim_Fstar, double Mlim_Fesc);
double Xray_General(double z, double lnM_Min, double lnM_Max, double mturn_acg, double mturn_mcg, double Alpha_star,
                     double Alpha_star_mini, double Fstar10, double Fstar7, double l_x, double l_x_mini, double t_h,
                     double t_star, double Mlim_Fstar, double Mlim_Fstar_mini);
double Fcoll_General(double z, double lnM_min, double lnM_max);
double unconditional_mf(double growthf, double lnM, double z, int HMF);
double conditional_mf(double growthf, double lnM, double delta_cond, double sigma_cond, int HMF);
/*----------------------------*/

/* Error framework testing */
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double* result);
void FunctionThatThrows();
/*------------------------*/

/* Test Outputs For Specific Models */
int single_test_sample( UserParams *user_params,  CosmoParams *cosmo_params,  AstroParams *astro_params,  FlagOptions *flag_options,
                        int seed, int n_condition, float *conditions, int *cond_crd, double z_out, double z_in, int *out_n_tot, int *out_n_cell, double *out_n_exp,
                        double *out_m_cell, double *out_m_exp, float *out_halo_masses, int *out_halo_coords);
//test function for getting halo properties from the wrapper, can use a lot of memory for large catalogs
int test_halo_props(double redshift,  UserParams *user_params,  CosmoParams *cosmo_params,  AstroParams *astro_params,
                     FlagOptions * flag_options, float * vcb_grid, float *J21_LW_grid, float *z_re_grid, float *Gamma12_ion_grid,
                     PerturbHaloField *halos, float *halo_props_out);
int test_filter(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double R_param, int filter_flag, double *result);

/* Miscellaneous exposed functions for testing */
double dicke(double z);
double sigma_z0(double M);
double dsigmasqdm_z0(double M);
double get_delta_crit(int HMF, double sigma, double growthf);
double atomic_cooling_threshold(float z);
double expected_nhalo(double redshift,  UserParams *user_params,  CosmoParams *cosmo_params);
/*-----------------------*/
