#ifndef _HMF_H
#define _HMF_H

#include "InputParameters.h"
#include "scaling_relations.h"
//integrals

#define MAX_DELTAC_FRAC (float)0.99 //max delta/deltac for the mass function integrals
#define DELTA_MIN -1 //minimum delta for Lagrangian mass function integrals
#define M_MIN_INTEGRAL 1e5
#define M_MAX_INTEGRAL 1e16

/* HMF Integrals */
void initialise_GL(double lnM_Min, double lnM_Max);

double Nion_General(double z, double lnM_Min, double lnM_Max, double MassTurnover, struct ScalingConstants *sc);
double Nion_General_MINI(double z, double lnM_Min, double lnM_Max, double MassTurnover, struct ScalingConstants *sc);
double Xray_General(double z, double lnM_Min, double lnM_Max, double mturn_acg, double mturn_mcg, struct ScalingConstants *sc);
double Fcoll_General(double z, double lnM_min, double lnM_max);
double Nhalo_General(double z, double lnM_min, double lnM_max);

double Nhalo_Conditional(double growthf, double lnM1, double lnM2, double lnM_cond, double sigma, double delta, int method);
double Mcoll_Conditional(double growthf, double lnM1, double lnM2, double lnM_cond, double sigma, double delta, int method);
double Nion_ConditionalM_MINI(double growthf, double lnM1, double lnM2, double lnM_cond, double sigma2, double delta2, double MassTurnover,
                                 struct ScalingConstants *sc, int method);
double Nion_ConditionalM(double growthf, double lnM1, double lnM2, double lnM_cond, double sigma2, double delta2, double MassTurnover,
                        struct ScalingConstants * sc, int method);
double Xray_ConditionalM(double redshift, double growthf, double lnM1, double lnM2, double lnM_cond, double sigma2, double delta2,
                         double mturn_acg, double mturn_mcg, struct ScalingConstants * sc, int method);

double unconditional_hmf(double growthf, double lnM, double z, int HMF);
double conditional_hmf(double growthf, double lnM, double delta_cond, double sigma_cond, int HMF);

/* erfc-based HMF integrals (!USE_MASS_DEPENDENT_ZETA and EPS) */
double FgtrM(double z, double M);
double FgtrM_bias_fast(float growthf, float del_bias, float sig_small, float sig_large);
float dfcoll_dz(float z, float sigma_min, float del_bias, float sig_bias);
double splined_erfc(double x);

/* Other values required in other files */
double get_delta_crit(int HMF, double sigma, double growthf);
double st_taylor_factor(double sig, double sig_cond, double growthf, double *zeroth_order);
double atomic_cooling_threshold(float z);
double minimum_source_mass(double redshift, bool xray, AstroParams *astro_params, FlagOptions *flag_options);
double sheth_delc_dexm(double del, double sig);
float Mass_limit_bisection(float Mmin, float Mmax, float PL, float FRAC);
double euler_to_lagrangian_delta(double delta);

#endif
