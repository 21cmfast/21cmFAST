#ifndef _STOCHASTICITY_H
#define _STOCHASTICITY_H

#include "InputParameters.h"
#include "OutputStructs.h"

struct HaloSamplingConstants
{
    // calculated per redshift
    int from_catalog; // flag for first box or updating halos
    double corr_sfr;
    double corr_star;
    double corr_xray;

    double z_in;
    double z_out;
    double growth_in;
    double growth_out;
    double M_min;
    double lnM_min;
    double M_max_tables;
    double lnM_max_tb;
    double sigma_min;

    // per-condition/redshift depending on from_catalog or not
    double delta;
    double M_cond;
    double lnM_cond;
    double sigma_cond;

    // calculated per condition
    double cond_val; // This is the table x value (density for grids, log mass for progenitors)
    double expected_N;
    double expected_M;
};

#ifdef __cplusplus
extern "C"
{
#endif
    int stochastic_halofield(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options, unsigned long long int seed, float redshift_desc, float redshift, float *dens_field, float *halo_overlap_box,
                             HaloField *halos_desc, HaloField *halos);

    int single_test_sample(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                           unsigned long long int seed, int n_condition, float *conditions, int *cond_crd, double z_out, double z_in,
                           int *out_n_tot, int *out_n_cell, double *out_n_exp,
                           double *out_m_cell, double *out_m_exp, float *out_halo_masses, int *out_halo_coords);

    // This function, designed to be used in the wrapper to estimate Halo catalogue size, takes the parameters and returns average number of halos within the box
    double expected_nhalo(double redshift, UserParams *user_params, CosmoParams *cosmo_params);

    // used in HaloField.c to assign rng to DexM halos
    int add_properties_cat(unsigned long long int seed, float redshift, HaloField *halos);
#ifdef __cplusplus
}
#endif
#endif //_STOCHASTICITY_H
