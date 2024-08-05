#ifndef _HMF_H
#define _HMF_H

#include "InputParameters.h"
//integrals

#define MAX_DELTAC_FRAC (float)0.99 //max delta/deltac for the mass function integrals
#define DELTA_MIN -1 //minimum delta for Lagrangian mass function integrals

//Parameters used for gsl integral on the mass function
struct parameters_gsl_MF_integrals{
    //parameters for all MF integrals
    double redshift;
    double growthf;
    int HMF;

    //Conditional parameters
    double sigma_cond;
    double delta;

    //SFR additions
    double Mturn;
    double f_star_norm;
    double alpha_star;
    double Mlim_star;

    //Nion additions
    double f_esc_norm;
    double alpha_esc;
    double Mlim_esc;

    //Minihalo additions
    double Mturn_upper;
};

void initialise_GL(float lnM_Min, float lnM_Max);
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
double Nhalo_General(double z, double lnM_min, double lnM_max);
double Fcoll_General(double z, double lnM_min, double lnM_max);
double unconditional_mf(double growthf, double lnM, double z, int HMF);
double conditional_mf(double growthf, double lnM, double delta_cond, double sigma_cond, int HMF);

double atomic_cooling_threshold(float z);
double minimum_soruce_mass(double redshift, bool xray, AstroParams *astro_params, FlagOptions *flag_options);

#endif