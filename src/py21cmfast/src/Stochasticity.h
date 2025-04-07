#ifndef _STOCHASTICITY_H
#define _STOCHASTICITY_H

#include "InputParameters.h"
#include "OutputStructs.h"

// parameters for the halo mass->stars calculations
// Note: ideally I would split this into constants set per snapshot and
//   constants set per condition, however some variables (delta or Mass)
//   can be set with differing frequencies depending on the condition type
struct HaloSamplingConstants {
    // calculated per redshift
    int from_catalog;  // flag for first box or updating halos
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
    double delta_crit;

    // calculated per condition
    // This is the table x value (density for grids, log mass for progenitors)
    double cond_val;
    double expected_N;
    double expected_M;
};

int stochastic_halofield(unsigned long long int seed, float redshift_desc, float redshift,
                         float *dens_field, float *halo_overlap_box, HaloField *halos_desc,
                         HaloField *halos);

int single_test_sample(unsigned long long int seed, int n_condition, float *conditions,
                       int *cond_crd, double z_out, double z_in, int *out_n_tot, int *out_n_cell,
                       double *out_n_exp, double *out_m_cell, double *out_m_exp,
                       float *out_halo_masses, int *out_halo_coords);

// This function, designed to be used in the wrapper to estimate Halo catalogue size, takes the
// parameters and returns average number of halos within the box
double expected_nhalo(double redshift);

// used in HaloField.c to assign rng to DexM halos
int add_properties_cat(unsigned long long int seed, float redshift, HaloField *halos);

void stoc_set_consts_z(struct HaloSamplingConstants *const_struct, double redshift,
                       double redshift_desc);
void stoc_set_consts_cond(struct HaloSamplingConstants *const_struct, double cond_val);

#endif
