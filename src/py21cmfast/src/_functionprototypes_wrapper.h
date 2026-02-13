/* This file contains the repeated function prototypes which are needed by CFFI
    to be included explicitly via ffi.cdef(), These are the only functions which
    are visible to the python wrapper */

/* OutputStruct COMPUTE FUNCTIONS */
int ComputeInitialConditions(unsigned long long random_seed, InitialConditions *boxes);

int ComputePerturbedField(float redshift, InitialConditions *boxes,
                          PerturbedField *perturbed_field);

int ComputeHaloCatalog(float redshift_desc, float redshift, InitialConditions *boxes,
                       unsigned long long int random_seed, HaloCatalog *halos_desc,
                       HaloCatalog *halos);

int ComputePerturbedHaloCatalog(float redshift, InitialConditions *boxes, TsBox *prev_ts,
                                IonizedBox *prev_ion, HaloCatalog *halos,
                                PerturbedHaloCatalog *halos_perturbed);

int ComputeTsBox(float redshift, float prev_redshift, float perturbed_field_redshift, short cleanup,
                 PerturbedField *perturbed_field, XraySourceBox *source_box,
                 TsBox *previous_spin_temp, InitialConditions *ini_boxes, TsBox *this_spin_temp);

int ComputeIonizedBox(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                      PerturbedField *previous_perturbed_field, IonizedBox *previous_ionize_box,
                      TsBox *spin_temp, HaloBox *halos, InitialConditions *ini_boxes,
                      IonizedBox *box);

int ComputeBrightnessTemp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                          PerturbedField *perturb_field, BrightnessTemp *box);

int ComputeHaloBox(double redshift, InitialConditions *ini_boxes, HaloCatalog *halos,
                   TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, HaloBox *grids);

int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct,
                        XraySourceBox *source_box);
/*--------------------------*/

/* PHOTON CONSERVATION MODEL FUNCTIONS */
int InitialisePhotonCons();

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
int ComputeZstart_PhotonCons(double *zstart);

void adjust_redshifts_for_photoncons(double z_step_factor, float *redshift, float *stored_redshift,
                                     float *absolute_delta_z);

void determine_deltaz_for_photoncons();

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic,
                         double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz,
                         int *Ndata_PhotonCons);

void FreePhotonConsMemory();
extern bool photon_cons_allocated;

void set_alphacons_params(double norm, double slope);
/* ------------------------------- */

/* Non-OutputStruct data products */
int ComputeLF(int nbins, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs,
              double *M_uv_z, double *M_h_z, double *log10phi);

float ComputeTau(int NPoints, float *redshifts, float *global_xHI, float z_re_HeII);
/*-----------------------------*/

/* Initialisation functions needed in the wrapper*/
double init_ps();
int init_heat();
int CreateFFTWWisdoms();
void Broadcast_struct_global_noastro(SimulationOptions *simulation_options,
                                     MatterOptions *matter_options, CosmoParams *cosmo_params);
void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options,
                                 CosmoTables *cosmo_tables,
                                 OptionalQuantities *optional_quantities);
void Free_cosmo_tables_global();
void initialiseSigmaMInterpTable(float M_Min, float M_Max);
void initialise_GL(double lnM_Min, double lnM_Max);
/*---------------------------*/

/* Intergration routines */
void get_sigma(int n_masses, double *mass_values, double *sigma_out, double *dsigmasqdm_out);
void get_condition_integrals(double redshift, double z_prev, int n_conditions, double *cond_values,
                             double *out_n_exp, double *out_m_exp);
void get_halo_chmf_interval(double redshift, double z_prev, int n_conditions, double *cond_values,
                            int n_masslim, double *lnM_lo, double *lnM_hi, double *out_n);
void get_halomass_at_probability(double redshift, double z_prev, int n_conditions,
                                 double *cond_values, double *probabilities, double *out_mass);
void get_global_SFRD_z(int n_redshift, double *redshifts, double *log10_turnovers_mcg,
                       double *out_sfrd, double *out_sfrd_mini);
void get_global_Nion_z(int n_redshift, double *redshifts, double *log10_turnovers_mcg,
                       double *out_nion, double *out_nion_mini);
void get_conditional_FgtrM(double redshift, double R, int n_densities, double *densities,
                           double *out_fcoll, double *out_dfcoll);
void get_conditional_SFRD(double redshift, double R, int n_densities, double *densities,
                          double *log10_mturns, double *out_sfrd, double *out_sfrd_mini);
void get_conditional_Nion(double redshift, double R, int n_densities, double *densities,
                          double *log10_mturns_acg, double *log10_mturns_mcg, double *out_nion,
                          double *out_nion_mini);
void get_conditional_Xray(double redshift, double R, int n_densities, double *densities,
                          double *log10_mturns, double *out_xray);
/*--------------------------------*/

/* Error framework testing */
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double *result);
void FunctionThatThrows();
/*------------------------*/

/* Test Outputs For Specific Models */
int single_test_sample(unsigned long long int seed, int n_condition, float *conditions,
                       float *cond_crd, double z_out, double z_in, int *out_n_tot, int *out_n_cell,
                       double *out_n_exp, double *out_m_cell, double *out_m_exp,
                       float *out_halo_masses, float *out_halo_coords);
// test function for getting halo properties from the wrapper, can use a lot of memory for large
// catalogs
int test_halo_props(double redshift, float *vcb_grid, float *J21_LW_grid, float *z_re_grid,
                    float *Gamma12_ion_grid, int n_halos, float *halo_masses, float *halo_coords,
                    float *star_rng, float *sfr_rnd, float *xray_rng, float *halo_props_out);
int test_filter(float *input_box, double R, double R_param, int filter_flag, double *result);

/* Functions required to access cosmology & mass functions directly */
double dicke(double z);
double sigma_z0(double M);
double dsigmasqdm_z0(double M);
double power_in_k(double k);
double power_in_vcb(double k);
double get_delta_crit(int HMF, double sigma, double growthf);
double atomic_cooling_threshold(float z);
double unconditional_hmf(double growthf, double lnM, double z, int HMF);
double conditional_hmf(double growthf, double lnM, double delta, double sigma, int HMF);
double expected_nhalo(double redshift);
/*-----------------------*/
