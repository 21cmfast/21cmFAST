#ifndef _SCALING_H
#define _SCALING_H

#include <gsl/gsl_matrix.h>
#include <stdbool.h>

#include "InputParameters.h"

// Parameters for the scaling relations
//   These are just the values which come from the InputStruct objects and don't change within the
//   snapshot using this reduces the use of the global parameter structs and allows fewer exp/log
//   unit changes
typedef struct ScalingConstants {
    double redshift;
    bool fix_mean;
    bool scaling_median;

    double fstar_10;
    double alpha_star;

    double alpha_upper;
    double pivot_upper;
    double upper_pivot_ratio;

    double fstar_7;
    double alpha_star_mini;

    double l_x;
    double l_x_mini;

    double sampled_mean_correction[3];
    double integral_mean_correction[3];

    double fesc_10;
    double alpha_esc;
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

    // TODO: remove after making the integrals consistent
    double t_h;
    double t_star;
} ScalingConstants;

void set_scaling_constants(double redshift, ScalingConstants *consts, bool use_photoncons);

double get_lx_on_sfr(double sfr, double metallicity, double lx_constant);
void get_halo_sfh(double snapshot_time, double halo_mass, double mturn_acg, double mturn_mcg,
                  double prog_hm, double prog_sm[2], double rng[3], ScalingConstants *consts,
                  double sfr_out[3], double sfr_out_mini[3]);
void get_halo_metallicity(double sfr, double stellar, double redshift, double *z_out);
void get_halo_xray(double sfr, double sfr_mini, double metallicity, double xray_rng,
                   ScalingConstants *consts, double *xray_out);

double scaling_PL_limit(double M, double norm, double alpha, double pivot, double limit);
double log_scaling_PL_limit(double lnM, double ln_norm, double alpha, double ln_pivot,
                            double ln_limit);
double scaling_double_PL(double M, double alpha_lo, double pivot_ratio, double alpha_hi,
                         double pivot_hi);
ScalingConstants evolve_scaling_constants_sfr(ScalingConstants *sc);
ScalingConstants evolve_scaling_constants_to_redshift(double redshift, ScalingConstants *sc);
ScalingConstants mimic_scatter_in_consts(ScalingConstants *sc);
void print_sc_consts(ScalingConstants *c);

// Forward define GSL types to avoid including GSL headers here
void eval_sfh_moments(double tau, gsl_matrix *out_chol_cov, gsl_matrix *out_mean_correction);
void free_sfh_correlation();

#endif
