#ifndef _INTERP_TABLES_H
#define _INTERP_TABLES_H

#include "InputParameters.h"
#include "scaling_relations.h"

// Functions within interp_tables.c need the parameter structures, but we don't want to pass them
// all down the chain, so we broadcast them
// TODO: in future it would be better to use a context struct. See `HaloBox.c`

void initialise_SFRD_spline(int Nbin, float zmin, float zmax, struct ScalingConstants *sc);
double EvaluateSFRD(double redshift, struct ScalingConstants *sc);
double EvaluateSFRD_MINI(double redshift, double log10_Mturn_LW_ave, struct ScalingConstants *sc);

void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, struct ScalingConstants *sc);
double EvaluateNionTs(double redshift, struct ScalingConstants *sc);
double EvaluateNionTs_MINI(double redshift, double log10_Mturn_LW_ave, struct ScalingConstants *sc);

void initialise_FgtrM_delta_table(double min_dens, double max_dens, double zpp, double growth_zpp,
                                  double smin_zpp, double smax_zpp);
double EvaluateFcoll_delta(double delta, double growthf, double sigma_min, double sigma_max);

void init_FcollTable(double zmin, double zmax, bool x_ray);
double EvaluatedFcolldz(double delta, double redshift, double sigma_min, double sigma_max);

void initialise_Nion_Conditional_spline(double z, double min_density, double max_density,
                                        double Mmin, double Mmax, double Mcond,
                                        double log10Mturn_min, double log10Mturn_max,
                                        double log10Mturn_min_MINI, double log10Mturn_max_MINI,
                                        struct ScalingConstants *sc, bool prev);
double EvaluateNion_Conditional(double delta, double log10Mturn, double growthf, double M_min,
                                double M_max, double M_cond, double sigma_max,
                                struct ScalingConstants *sc, bool prev);
double EvaluateNion_Conditional_MINI(double delta, double log10Mturn_m, double growthf,
                                     double M_min, double M_max, double M_cond, double sigma_max,
                                     struct ScalingConstants *sc, bool prev);
void initialise_Xray_Conditional_table(double redshift, double min_density, double max_density,
                                       double Mmin, double Mmax, double Mcond,
                                       struct ScalingConstants *sc);
double EvaluateXray_Conditional(double delta, double log10Mturn_m, double redshift, double growthf,
                                double M_min, double M_max, double M_cond, double sigma_max,
                                struct ScalingConstants *sc);
void initialise_SFRD_Conditional_table(double z, double min_density, double max_density,
                                       double Mmin, double Mmax, double Mcond,
                                       struct ScalingConstants *sc);
double EvaluateSFRD_Conditional(double delta, double growthf, double M_min, double M_max,
                                double M_cond, double sigma_max, struct ScalingConstants *sc);
double EvaluateSFRD_Conditional_MINI(double delta, double log10Mturn_m, double growthf,
                                     double M_min, double M_max, double M_cond, double sigma_max,
                                     struct ScalingConstants *sc);

void initialise_dNdM_tables(double xmin, double xmax, double ymin, double ymax, double growth1,
                            double param, bool from_catalog);
double EvaluateNhalo(double condition, double growthf, double lnMmin, double lnMmax, double M_cond,
                     double sigma, double delta);
double EvaluateMcoll(double condition, double growthf, double lnMmin, double lnMmax, double M_cond,
                     double sigma, double delta);

void initialise_dNdM_inverse_table(double xmin, double xmax, double lnM_min, double growth1,
                                   double param, bool from_catalog);
double EvaluateNhaloInv(double condition, double prob);

void initialise_J_split_table(int Nbin, double umin, double umax, double gamma1);
double EvaluateJ(double u_res, double gamma1);

void initialiseSigmaMInterpTable(float M_Min, float M_Max);
double EvaluateSigma(double lnM);
double EvaluatedSigmasqdm(double lnM);

void InitialiseSigmaInverseTable();
double EvaluateSigmaInverse(double sigma);

void freeSigmaMInterpTable();
void free_conditional_tables();
void free_global_tables();
void free_dNdM_tables();

#endif
