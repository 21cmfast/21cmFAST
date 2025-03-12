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
void initialiseSigmaMInterpTable(float M_Min, float M_Max);
/*---------------------------*/

/* Intergration routines */
void get_sigma(UserParams *user_params, CosmoParams *cosmo_params, int n_masses, double *mass_values, double *sigma_out, double *dsigmasqdm_out);
void get_condition_integrals(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double z_prev, int n_conditions, double *cond_values,
                        double *out_n_exp, double *out_m_exp)
void get_halomass_at_probability(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double z_prev, int n_conditions, double *cond_values, double *probabilities,
                        double *out_mass);
void get_global_SFRD_z(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        int n_redshift, double *redshifts, double *log10_turnovers_mcg, double *out_sfrd, double *out_sfrd_mini);
void get_global_Nion_z(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        int n_redshift, double *redshifts, double *log10_turnovers_mcg, double *out_nion, double *out_nion_mini);
void get_conditional_FgtrM(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double *out_fcoll, double *out_dfcoll);
void get_conditional_SFRD(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double *log10_mturns,
                        double *out_sfrd, double *out_sfrd_mini);
void get_conditional_Nion(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double log10_mturns_acg, double log10_mturns_mcg,
                        double *out_nion, double *out_nion_mini);
void get_conditional_Xray(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double *log10_mturns,
                        double *out_xray);
/*--------------------------------*/

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
int test_halo_props(double redshift, UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params,
                    FlagOptions * flag_options, float *vcb_grid, float *J21_LW_grid, float *z_re_grid, float *Gamma12_ion_grid, int n_halos,
                    float *halo_masses, int *halo_coords, float *star_rng, float *sfr_rnd, float *xray_rng, float *halo_props_out);
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
