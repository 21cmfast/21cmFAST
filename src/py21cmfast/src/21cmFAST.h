/*
    This is the header file for the wrappable version of 21cmFAST, or 21cmMC.
    It contains function signatures, struct definitions and globals to which the Python wrapper code
    requires access. This is not included in any C file but is called by cffi to expose the functions within to the wrapper
*/

#include "BrightnessTemperatureBox.h"
#include "Globals.h"
#include "HaloBox.h"
#include "HaloField.h"
#include "InitialConditions.h"
#include "IonisationBox.h"
#include "InputParamters.h"
#include "PerturbField.h"
#include "PerturbHaloField.h"
#include "SpinTemperatureBox.h"
#include "photoncons.h"

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi);

float ComputeTau(struct UserParams *user_params, struct CosmoParams *cosmo_params, int Npoints, float *redshifts, float *global_xHI);


void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_HF(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options);

int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double* result);
void FunctionThatThrows();
int init_heat();

//exposing lower-level functions for testing

double init_ps();
double dicke(double z);
double sigma_z0(double M);
double dsigmasqdm_z0(double M);
double get_delta_crit(int HMF, double sigma, double growthf);

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
double Fcoll_General(double z, double lnM_min, double lnM_max);
double unconditional_mf(double growthf, double lnM, double z, int HMF);
double conditional_mf(double growthf, double lnM, double delta_cond, double sigma_cond, int HMF);
double atomic_cooling_threshold(float z);

//CFFI needs to be able to access a free function to make the __del__ method for OutputStruct fields
//  This will expose the standard free() function to the wrapper so it can be used
void free(void *ptr);
