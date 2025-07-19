#ifndef _SCALING_H
#define _SCALING_H

#include <stdbool.h>

#include "InputParameters.h"

// Parameters for the scaling relations
//   These are just the values which come from the InputStruct objects and don't change within the
//   snapshot using this reduces the use of the global parameter structs and allows fewer exp/log
//   unit changes
struct ScalingConstants {
    double redshift;
    bool fix_mean;
    bool scaling_median;

    double fstar_10;
    double alpha_star;
    double sigma_star;

    double alpha_upper;
    double pivot_upper;
    double upper_pivot_ratio;

    double fstar_7;
    double alpha_star_mini;

    double t_h;
    double t_star;
    double sigma_sfr_lim;
    double sigma_sfr_idx;

    double l_x;
    double l_x_mini;
    double sigma_xray;

    double fesc_10;
    double alpha_esc;
    double beta_esc;
    double fesc_7;
    double pop2_ion;
    double pop3_ion;

    double vcb_norel;
    double acg_thresh;
    double mturn_a_nofb;
    double mturn_m_nofb;

    double Mlim_Fstar;
    double Mlim_Fesc;
    double Mlim_Fstar_mini;
    double Mlim_Fesc_mini;
};

void set_scaling_constants(double redshift, struct ScalingConstants *consts, bool use_photoncons);

double get_lx_on_sfr(double sfr, double metallicity, double lx_constant);
void get_halo_stellarmass(double halo_mass, double mturn_acg, double mturn_mcg, double star_rng,
                          struct ScalingConstants *consts, double *star_acg, double *star_mcg);
void get_halo_sfr(double stellar_mass, double stellar_mass_mini, double sfr_rng,
                  struct ScalingConstants *consts, double *sfr, double *sfr_mini);
void get_halo_metallicity(double sfr, double stellar, double redshift, double *z_out);
void get_halo_xray(double sfr, double sfr_mini, double metallicity, double xray_rng,
                   struct ScalingConstants *consts, double *xray_out);

double scaling_PL_limit(double M, double norm, double alpha, double pivot, double limit);
double log_scaling_PL_limit(double lnM, double ln_norm, double alpha, double ln_pivot,
                            double ln_limit);
double scaling_double_PL(double M, double alpha_lo, double pivot_ratio, double alpha_hi,
                         double pivot_hi);
struct ScalingConstants evolve_scaling_constants_sfr(struct ScalingConstants *sc);
struct ScalingConstants evolve_scaling_constants_to_redshift(double redshift,
                                                             struct ScalingConstants *sc,
                                                             bool use_photoncons);
void print_sc_consts(struct ScalingConstants *c);

#endif
