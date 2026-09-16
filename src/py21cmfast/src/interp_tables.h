#ifndef _INTERP_TABLES_H
#define _INTERP_TABLES_H

#include "InputParameters.h"
#include "scaling_relations.h"

// Functions within interp_tables.c need the parameter structures, but we don't want to pass them
// all down the chain, so we broadcast them

void initialize_sfrd_unconditional_tables(int Nbin, float zmin, float zmax, ScalingConstants *sc);
double evaluate_sfrd_unconditional_acg(double redshift, double log10_Mturn_ACG_ave,
                                       ScalingConstants *sc);
double evaluate_sfrd_unconditional_mcg(double redshift, double log10_Mturn_ACG_ave,
                                       double log10_Mturn_MCG_ave, ScalingConstants *sc);

void initialize_nion_unconditional_tables(int Nbin, float zmin, float zmax, ScalingConstants *sc);
double evaluate_nion_unconditional_acg(double redshift, double log10_Mturn_ACG_ave,
                                       ScalingConstants *sc);
double evaluate_nion_unconditional_mcg(double redshift, double log10_Mturn_ACG_ave,
                                       double log10_Mturn_MCG_ave, ScalingConstants *sc);

void initialize_fcoll_unconditional_table(double zmin, double zmax, bool x_ray);
void initialize_fcoll_conditional_tables(double min_dens, double max_dens, double zpp,
                                         double growth_zpp, double smin_zpp, double smax_zpp);
double evaluate_fcoll_conditional_eps(double delta, double growthf, double sigma_min,
                                      double sigma_max);
double evaluate_dfcoll_dz_conditional_eps(double delta, double redshift, double sigma_min,
                                          double sigma_max);

void initialize_nion_conditional_tables(double z, double min_density, double max_density,
                                        double Mmin, double Mmax, double Mcond,
                                        ScalingConstants *sc, bool prev);
double evaluate_nion_conditional_acg(double delta, double log10Mturn_acg, double growthf,
                                     double M_min, double M_max, double M_cond, double sigma_max,
                                     ScalingConstants *sc, bool prev);
double evaluate_nion_conditional_mcg(double delta, double log10Mturn_acg, double log10Mturn_mcg,
                                     double growthf, double M_min, double M_max, double M_cond,
                                     double sigma_max, ScalingConstants *sc, bool prev);
void initialize_xray_emissivity_conditional_tables(double redshift, double min_density,
                                                   double max_density, double Mmin, double Mmax,
                                                   double Mcond, ScalingConstants *sc);
double evaluate_xray_emissivity_conditional_acg(double delta, double log10Mturn_acg,
                                                double redshift, double growthf, double M_min,
                                                double M_max, double M_cond, double sigma_max,
                                                ScalingConstants *sc);
double evaluate_xray_emissivity_conditional_mcg(double delta, double log10Mturn_acg,
                                                double log10Mturn_mcg, double redshift,
                                                double growthf, double M_min, double M_max,
                                                double M_cond, double sigma_max,
                                                ScalingConstants *sc);
void initialize_sfrd_conditional_tables(double z, double min_density, double max_density,
                                        double Mmin, double Mmax, double Mcond,
                                        ScalingConstants *sc);
double evaluate_sfrd_conditional_acg(double delta, double log10Mturn_acg, double growthf,
                                     double M_min, double M_max, double M_cond, double sigma_max,
                                     ScalingConstants *sc);
double evaluate_sfrd_conditional_mcg(double delta, double log10Mturn_acg, double log10Mturn_mcg,
                                     double growthf, double M_min, double M_max, double M_cond,
                                     double sigma_max, ScalingConstants *sc);

void initialize_dndm_tables(double xmin, double xmax, double ymin, double ymax, double growth1,
                            double param, bool from_catalog);
double evaluate_nhalo_conditional(double condition, double growthf, double lnMmin, double lnMmax,
                                  double M_cond, double sigma, double delta);
double evaluate_fcoll_conditional(double condition, double growthf, double lnMmin, double lnMmax,
                                  double M_cond, double sigma, double delta);

void initialize_dndm_inverse_table(double xmin, double xmax, double lnM_min, double growth1,
                                   double param, bool from_catalog);
double evaluate_nhalo_inverse(double condition, double prob);

void initialize_j_split_table(int Nbin, double umin, double umax, double gamma1);
double evaluate_j_split(double u_res, double gamma1);

void initialize_sigma_tables(float M_Min, float M_Max);
double evaluate_sigma(double lnM);
double evaluate_dsigma_square_dm(double lnM);

void initialize_sigma_inverse_table();
double evaluate_sigma_inverse(double sigma);

void free_sigma_tables();
void free_conditional_tables();
void free_unconditional_tables();
void free_dndm_tables();

#endif
