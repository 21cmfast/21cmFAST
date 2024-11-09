#ifndef _INTERP_TABLES_H
#define _INTERP_TABLES_H

#include "InputParameters.h"
#include "interpolation.h"

//Functions within interp_tables.c need the parameter structures, but we don't want to pass them all down the chain, so we broadcast them
//TODO: in future it would be better to use a context struct. See `HaloBox.c`

#ifdef __cplusplus
extern "C" {
#endif
void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI,
                             float mturn_a_const, bool minihalos);
double EvaluateSFRD(double redshift, double Mlim_Fstar);
double EvaluateSFRD_MINI(double redshift, double log10_Mturn_LW_ave, double Mlim_Fstar_MINI);

void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10,
                                float Fesc10, float Fstar7_MINI, float Fesc7_MINI, float mturn_a_const, bool minihalos);
double EvaluateNionTs(double redshift, double Mlim_Fstar, double Mlim_Fesc);
double EvaluateNionTs_MINI(double redshift, double log10_Mturn_LW_ave, double Mlim_Fstar_MINI, double Mlim_Fesc_MINI);

void initialise_FgtrM_delta_table(double min_dens, double max_dens, double zpp, double growth_zpp, double smin_zpp, double smax_zpp);
double EvaluateFcoll_delta(double delta, double growthf, double sigma_min, double sigma_max);

void init_FcollTable(double zmin, double zmax, bool x_ray);
double EvaluatedFcolldz(double delta, double redshift, double sigma_min, double sigma_max);

void initialise_Nion_Conditional_spline(float z, float Mcrit_atom, float min_density, float max_density,
                                     float Mmin, float Mmax, float Mcond, float log10Mturn_min, float log10Mturn_max,
                                     float log10Mturn_min_MINI, float log10Mturn_max_MINI, float Alpha_star,
                                     float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10,
                                     float Mlim_Fstar, float Mlim_Fesc, float Fstar7_MINI, float Fesc7_MINI,
                                     float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, int method, int method_mini,
                                     bool minihalos, bool prev);
double EvaluateNion_Conditional(double delta, double log10Mturn, double growthf, double M_min, double M_max, double M_cond, double sigma_max,
                                double Mlim_Fstar, double Mlim_Fesc, bool prev);
double EvaluateNion_Conditional_MINI(double delta, double log10Mturn_m, double growthf, double M_min, double M_max, double M_cond, double sigma_max,
                                    double Mturn_a, double Mlim_Fstar, double Mlim_Fesc, bool prev);

void initialise_SFRD_Conditional_table(double min_density, double max_density, double growthf,
                                    float Mcrit_atom, double Mmin, double Mmax, double Mcond, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI, int method, int method_mini, bool minihalos);
double EvaluateSFRD_Conditional(double delta, double growthf, double M_min, double M_max, double M_cond, double sigma_max, double Mturn_a, double Mlim_Fstar);
double EvaluateSFRD_Conditional_MINI(double delta, double log10Mturn_m, double growthf, double M_min, double M_max, double M_cond, double sigma_max, double Mturn_a, double Mlim_Fstar);

void initialise_dNdM_tables(double xmin, double xmax, double ymin, double ymax, double growth1, double param, bool from_catalog);
double EvaluateNhalo(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);
double EvaluateMcoll(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);

void initialise_dNdM_inverse_table(double xmin, double xmax, double lnM_min, double growth1, double param, bool from_catalog);
double EvaluateNhaloInv(double condition, double prob);

void initialise_J_split_table(int Nbin, double umin, double umax, double gamma1);
double EvaluateJ(double u_res,double gamma1);

void initialiseSigmaMInterpTable(float M_Min, float M_Max);
double EvaluateSigma(double lnM);
double EvaluatedSigmasqdm(double lnM);

void InitialiseSigmaInverseTable();
double EvaluateSigmaInverse(double sigma);

void freeSigmaMInterpTable();
void free_conditional_tables();
void free_global_tables();
void free_dNdM_tables();

RGTable1D_f* get_SFRD_conditional_table(void);

#ifdef __cplusplus
}
#endif
#endif
