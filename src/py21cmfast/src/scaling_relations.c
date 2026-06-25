/*This file contains the halo scaling relations which can be used by
    the integrals in hmf.c or the sampled halos in HaloBox.c*/
#include "scaling_relations.h"

#include <gsl/gsl_sf_gamma.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "cexcept.h"
#include "cosmology.h"
#include "exceptions.h"
#include "hmf.h"
#include "logger.h"
#include "photoncons.h"
#include "thermochem.h"

void print_sc_consts(ScalingConstants *c) {
    LOG_DEBUG("Printing scaling relation constants z = %.3f....", c->redshift);
    LOG_DEBUG("SHMR: f10 %.2e a %.2e f7 %.2e a_mini %.2e sigma %.2e", c->fstar_10, c->alpha_star,
              c->fstar_7, c->alpha_star_mini, c->sigma_star);
    LOG_DEBUG("Upper: a_upper %.2e pivot %.2e", c->alpha_upper, c->pivot_upper);
    LOG_DEBUG("FESC: f10 %.2e a %.2e f7 %.2e", c->fesc_10, c->alpha_esc, c->fesc_7);
    LOG_DEBUG("SSFR: t* %.2e th %.8e sigma %.2e idx %.2e", c->t_star, c->t_h, c->sigma_sfr_lim,
              c->sigma_sfr_idx);
    LOG_DEBUG("Turnovers (nofb) ACG %.2e", c->mturn_a_nofb);
    LOG_DEBUG("Limits (ACG,MCG) F* (%.2e %.2e) Fesc (%.2e %.2e)", c->Mlim_Fstar, c->Mlim_Fstar_mini,
              c->Mlim_Fesc, c->Mlim_Fesc_mini);
    return;
}

void set_scaling_constants(double redshift, ScalingConstants *consts, bool use_photoncons) {
    consts->redshift = redshift;

    // Set on for the fixed grid case since we are missing halos above the cell mass
    consts->fix_mean =
        matter_options_global->HMF == HMF_WATSON || matter_options_global->HMF == HMF_WATSON_Z ||
        matter_options_global->HMF == HMF_REED07 || matter_options_global->HMF == HMF_YUNG24;
    // whether to fix *integrated* (not sampled) galaxy properties to the expected mean
    consts->scaling_median = astro_options_global->HALO_SCALING_RELATIONS_MEDIAN;

    consts->fstar_10 = astro_params_global->F_STAR10;
    consts->alpha_star = astro_params_global->ALPHA_STAR;
    consts->sigma_star = astro_params_global->SIGMA_STAR;

    consts->alpha_upper = astro_params_global->UPPER_STELLAR_TURNOVER_INDEX;
    consts->pivot_upper = astro_params_global->UPPER_STELLAR_TURNOVER_MASS;
    consts->upper_pivot_ratio = pow(consts->pivot_upper / 1e10, consts->alpha_star) +
                                pow(consts->pivot_upper / 1e10, consts->alpha_upper);

    consts->fstar_7 = astro_params_global->F_STAR7_MINI;
    consts->alpha_star_mini = astro_params_global->ALPHA_STAR_MINI;

    consts->t_h = t_hubble(redshift);
    consts->t_star = astro_params_global->t_STAR;
    consts->sigma_sfr_lim = astro_params_global->SIGMA_SFR_LIM;
    consts->sigma_sfr_idx = astro_params_global->SIGMA_SFR_INDEX;
    // setting units to 1e38 erg s -1 so we can store in float
    consts->l_x = astro_params_global->L_X * 1e-38;
    consts->l_x_mini = astro_params_global->L_X_MINI * 1e-38;
    consts->sigma_xray = astro_params_global->SIGMA_LX;

    consts->alpha_esc = astro_params_global->ALPHA_ESC;
    consts->fesc_10 = astro_params_global->F_ESC10;
    consts->fesc_7 = astro_params_global->F_ESC7_MINI;

    if (use_photoncons) {
        if (astro_options_global->PHOTON_CONS_TYPE == PHOTON_CONS_ALPHA)
            consts->alpha_esc = get_fesc_fit(redshift);
        else if (astro_options_global->PHOTON_CONS_TYPE == PHOTON_CONS_F)
            consts->fesc_10 = get_fesc_fit(redshift);
    }

    consts->pop2_ion = astro_params_global->POP2_ION;
    consts->pop3_ion = astro_params_global->POP3_ION;

    consts->mturn_a_nofb =
        fmax(atomic_cooling_threshold(redshift), astro_params_global->M_TURN_STELLAR_FEEDBACK);

    switch (matter_options_global->V_CB_MODEL) {
        case V_CB_MODEL_NO:
            consts->vcb_const = 0.;
            break;
        case V_CB_MODEL_AVG_AUTO:
            consts->vcb_const = cosmo_tables_global->V_CB_AVG;
            break;
        case V_CB_MODEL_AVG_DEBUG:
            consts->vcb_const = astro_params_global->V_CB_AVG_DEBUG;
            break;
        default:  // V_CB_MODEL_FLUCTS
            consts->vcb_const = 0.;
            break;
    }

    consts->Mlim_Fstar =
        Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL, consts->alpha_star, consts->fstar_10);
    consts->Mlim_Fesc =
        Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL, consts->alpha_esc, consts->fesc_10);

    if (astro_options_global->USE_MINI_HALOS) {
        consts->Mlim_Fstar_mini =
            Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL, consts->alpha_star_mini,
                                 consts->fstar_7 * pow(1e3, consts->alpha_star_mini));
        consts->Mlim_Fesc_mini =
            Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL, consts->alpha_esc,
                                 consts->fesc_7 * pow(1e3, consts->alpha_esc));
    }
}

// It's often useful to create a copy of scaling constants without F_ESC
ScalingConstants evolve_scaling_constants_sfr(ScalingConstants *sc) {
    ScalingConstants sc_sfrd = *sc;
    sc_sfrd.fesc_10 = 1.;
    sc_sfrd.fesc_7 = 1.;
    sc_sfrd.alpha_esc = 0.;
    sc_sfrd.Mlim_Fesc = 0.;
    sc_sfrd.Mlim_Fesc_mini = 0.;

    return sc_sfrd;
}

// It's often useful to create a copy of scaling relations at a different z
ScalingConstants evolve_scaling_constants_to_redshift(double redshift, ScalingConstants *sc,
                                                      bool use_photoncons) {
    ScalingConstants sc_z = *sc;
    sc_z.redshift = redshift;
    sc_z.t_h = t_hubble(redshift);

    if (use_photoncons) {
        if (astro_options_global->PHOTON_CONS_TYPE == PHOTON_CONS_ALPHA)
            sc_z.alpha_esc = get_fesc_fit(redshift);
        else if (astro_options_global->PHOTON_CONS_TYPE == PHOTON_CONS_F)
            sc_z.fesc_10 = get_fesc_fit(redshift);

        // if we altered the escape fraction, we need to recalculate the mass limits
        sc_z.Mlim_Fesc =
            Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL, sc_z.alpha_esc, sc_z.fesc_10);

        if (astro_options_global->USE_MINI_HALOS) {
            sc_z.Mlim_Fesc_mini =
                Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL, sc_z.alpha_esc,
                                     sc_z.fesc_7 * pow(1e3, sc_z.alpha_esc));
        }
    }

    sc_z.mturn_a_nofb =
        fmax(atomic_cooling_threshold(redshift), astro_params_global->M_TURN_STELLAR_FEEDBACK);

    return sc_z;
}

ScalingConstants mimic_scatter_in_consts(ScalingConstants *sc) {
    // This function mimics the effect of log-normal scatter in the scaling relations by increasing
    //  the normalisation of the relations appropriately.
    //  These should be used in individual integrals / table initialisations, scoped tightly,
    //  and applied after evolving to the correct redshift / relation.
    ScalingConstants ev_consts = *sc;
    ev_consts.fstar_10 *= exp(0.5 * pow(ev_consts.sigma_star, 2));
    ev_consts.fstar_7 *= exp(0.5 * pow(ev_consts.sigma_star, 2));
    ev_consts.l_x *= exp(0.5 * pow(ev_consts.sigma_xray, 2));
    ev_consts.l_x_mini *= exp(0.5 * pow(ev_consts.sigma_xray, 2));

    // This is a lower-limit on the effect of scatter in SSFR
    //  since the scatter depends on stellar mass. To fully apply the limit we would need
    //  a new HMF integrand. Explicit Monte-Carlo Integration over the property PDFS might also
    //  work.
    // TODO: Something better than this
    ev_consts.t_star /= exp(0.5 * pow(ev_consts.sigma_sfr_lim, 2));

    // By altering the normalisations we need to recalculate the mass limits
    ev_consts.Mlim_Fstar = Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL,
                                                ev_consts.alpha_star, ev_consts.fstar_10);

    if (astro_options_global->USE_MINI_HALOS) {
        ev_consts.Mlim_Fstar_mini =
            Mass_limit_bisection(M_MIN_INTEGRAL, M_MAX_INTEGRAL, ev_consts.alpha_star_mini,
                                 ev_consts.fstar_7 * pow(1e3, ev_consts.alpha_star_mini));
    }

    return ev_consts;
}

/*
General Scaling realtions used in mass function integrals and sampling.

These are usually integrated over log(M) and can avoid a few of pow/exp calls
by keeping them in log. Since they are called within integrals they don't use the
ScalingConstants. Instead pulling from the GSL integration parameter structs
*/

double scaling_single_PL(double M, double alpha, double pivot) { return pow(M / pivot, alpha); }

double log_scaling_single_PL(double lnM, double alpha, double ln_pivot) {
    return alpha * (lnM - ln_pivot);
}

// single power-law with provided limit for scaling == 1, returns f(M)/f(pivot) == norm
// limit is provided ahead of time for efficiency
double scaling_PL_limit(double M, double norm, double alpha, double pivot, double limit) {
    if ((alpha > 0. && M > limit) || (alpha < 0. && M < limit)) return 1 / norm;

    // if alpha is zero, this returns 1 as expected (note strict inequalities above)
    return scaling_single_PL(M, alpha, pivot);
}

// log version for possible acceleration
double log_scaling_PL_limit(double lnM, double ln_norm, double alpha, double ln_pivot,
                            double ln_limit) {
    if ((alpha > 0. && lnM > ln_limit) || (alpha < 0. && lnM < ln_limit)) return -ln_norm;

    // if alpha is zero, this returns log(1) as expected (note strict inequalities above)
    return log_scaling_single_PL(lnM, alpha, ln_pivot);
}

// concave-down double power-law, we pass pivot_ratio == denominator(M=pivot_lo)
//   to save pow() calls. Due to the double power-law we gain little from a log verison
//   returns f(M)/f(M==pivot_lo)
double scaling_double_PL(double M, double alpha_lo, double pivot_ratio, double alpha_hi,
                         double pivot_hi) {
    // if alpha is zero, this returns 1 as expected (note strict inequalities above)
    return pivot_ratio / (pow(M / pivot_hi, -alpha_lo) + pow(M / pivot_hi, -alpha_hi));
}

/*
Scaling relations used in the halo sampler. Quantities calculated for a single halo mass
and are summed onto grids.
*/

// The mean Lx_over_SFR given by Lehmer+2021, by integrating analyitically over their double
// power-law + exponential Luminosity function NOTE: this relation is fit to low-z, and there is a
// PEAK of Lx/SFR around 10% solar due to the critical L term NOTE: this relation currently also has
// no normalisation, and is fixed to the Lehmer values for now
double lx_on_sfr_Lehmer(double metallicity) {
    double l10z = log10(metallicity);

    // LF parameters from Lehmer+2021
    double slope_low = 1.74;
    double slope_high = 1.16 + 1.34 * l10z;
    double xray_norm = 1.29;
    double l10break_L = 38.54 - 38.;  // the -38 written explicitly for clarity on our units
    double l10crit_L = 39.98 - 38. + 0.6 * l10z;
    double L_ratio = pow(10, l10break_L - l10crit_L);

    // The double power-law + exponential integrates to an upper and lower incomplete Gamma function
    // since the slope is < 2 we don't need to set a lower limit, but this can be done by replacing
    // gamma(2-y_low) with gamma_inc(2-y_low,L_lower/L_crit)
    double prefactor_low = pow(10, l10crit_L * (2 - slope_low));
    double prefactor_high =
        pow(10, l10crit_L * (2 - slope_high) + l10break_L * (slope_high - slope_low));
    double gamma_low = gsl_sf_gamma(2 - slope_low) - gsl_sf_gamma_inc(2 - slope_low, L_ratio);
    double gamma_high = gsl_sf_gamma_inc(2 - slope_high, L_ratio);

    double lx_over_sfr = xray_norm * (prefactor_low * gamma_low + prefactor_high * gamma_high);

    return lx_over_sfr;
}

// double power-law in Z with the low-metallicity PL fixed as constant
double lx_on_sfr_doublePL(double metallicity, double lx_constant) {
    double hi_z_index = -0.64;  // power-law index of LX/SFR at high-z
    double lo_z_index = 0.;
    double z_pivot = 0.05;  // Z at which LX/SFR == lx_constant / 2

    return lx_constant * scaling_double_PL(metallicity, lo_z_index, 1., hi_z_index, z_pivot);
}

// first order power law Lx with cross-term (e.g Kaur+22)
// here the constant defines the value at 1 Zsun and 1 Msun yr-1
double lx_on_sfr_PL_Kaur(double sfr, double metallicity, double lx_constant) {
    // Hardcoded for now (except the lx normalisation and the scatter): 3 extra fit parameters in
    // the equation taking values from Kaur+22, constant factors controlled by
    // astro_params_global->L_X
    double sfr_index = 0.03;
    double z_index = -0.64;
    double cross_index = 0.0;
    double l10z = log10(metallicity);

    double lx_over_sfr =
        (cross_index * l10z + sfr_index) * log10(sfr * physconst.s_per_yr) + z_index * l10z;
    return pow(10, lx_over_sfr) * lx_constant;
}

// Schechter function from Kaur+22
// Here the constant defines the value minus 1 at the turnover Z
double lx_on_sfr_Schechter(double metallicity, double lx_constant) {
    // Hardcoded for now (except the lx normalisation and the scatter): 3 extra fit parameters in
    // the equation taking values from Kaur+22, constant factors controlled by
    // astro_params_global->L_X
    double z_turn = 8e-3 / 0.02;  // convert to solar
    double logz_index = 0.3;
    double l10z = log10(metallicity / z_turn);

    double lx_over_sfr = logz_index * l10z - metallicity / z_turn;
    return pow(10, lx_over_sfr) * lx_constant;
}

double get_lx_on_sfr(double sfr, double metallicity, double lx_constant) {
    // Future TODO: experiment more with these models and parameterise properly
    //  return lx_on_sfr_Lehmer(metallicity);
    //  return lx_on_sfr_Schechter(metallicity, lx_constant);
    //  return lx_on_sfr_PL_Kaur(sfr,metallicity, lx_constant);
    // HACK: new/old model switch with upperstellar flag
    if (astro_options_global->USE_UPPER_STELLAR_TURNOVER)
        return lx_on_sfr_doublePL(metallicity, lx_constant);
    return lx_constant;
}

void get_halo_stellarmass(double halo_mass, double mturn_acg, double mturn_mcg, double star_rng,
                          ScalingConstants *consts, double *star_acg, double *star_mcg) {
    // low-mass ACG power-law parameters
    double f_10 = consts->fstar_10;
    double f_a = consts->alpha_star;
    double sigma_star = consts->sigma_star;

    // high-mass ACG power-law parameters
    double fu_a = consts->alpha_upper;
    double fu_p = consts->pivot_upper;

    // MCG parameters
    double f_7 = consts->fstar_7;
    double f_a_mini = consts->alpha_star_mini;

    // intermediates
    double mu_fstar, mu_fstar_mini;
    double f_sample, f_sample_mini;
    double star_mass_sample, star_mass_sample_mini;

    double baryon_ratio = cosmo_params_global->OMb / cosmo_params_global->OMm;
    // adjustment to the mean for lognormal scatter
    double stoc_adjustment_term = consts->scaling_median ? 0 : sigma_star * sigma_star / 2.;

    // Set the mu parameter for the distribution (according to Eq. 10 in
    // https://arxiv.org/pdf/2504.17254) NOTE: Since we have a constraint that the fraction of
    // baryonic mass that is converted into stars must be less than unity,
    //       we have to compute f_star first, and we treat it as the lognormal variable below.
    //       Note also that Eq. 10 contains an additional exponent, it is absorbed in the line below
    //       for f_sample for computational efficiency. Finally, the mu parameter is adjusted with
    //       exp(-sigma^2 /2), in case we want to interpret it as the mean of the f_star
    //       distribution, this exponent is also absorbed in the line for f_sample below, for
    //       computational efficiency
    // We don't want an upturn even with a negative ALPHA_STAR
    if (astro_options_global->USE_UPPER_STELLAR_TURNOVER && (f_a > fu_a)) {
        mu_fstar = f_10 * scaling_double_PL(halo_mass, f_a, consts->upper_pivot_ratio, fu_a, fu_p);
    } else {
        mu_fstar = f_10 * scaling_single_PL(halo_mass, consts->alpha_star, 1e10);
    }
    // 1e10 normalisation of stellar mass
    // NOTE: while the lognormal distribution in the C code is with respect to base-e (since the rng
    // variable distributes normally, and
    //      we use an exponent below), the sigma parameters are converted from the user's specified
    //      input ("sigma_user") in units of base-10 (dex) to base-e via dex2exp_transformer (in
    //      inputs.py). Therefore, the samples below are effectively distributed as lognormal in
    //      base-10, namely the log_10 of the sample distributes as a Gaussian with s.t.d equals to
    //      sigma_user, while the mean (median) of the samples (not their logarithm!) is given by
    //      the above mu_fstar, if HALO_SCALING_RELATIONS_MEDIAN = False (True)
    f_sample =
        mu_fstar * exp(-mturn_acg / halo_mass + star_rng * sigma_star - stoc_adjustment_term);
    if (f_sample > 1.) f_sample = 1.;

    star_mass_sample = f_sample * halo_mass * baryon_ratio;
    *star_acg = star_mass_sample;

    if (!astro_options_global->USE_MINI_HALOS) {
        *star_mcg = 0.;
        return;
    }

    // No MCGs can form if their turnover mass is above the ACG turnover mass,
    // or if the ACG and MCG turnover masses are the same (can happen if the reionization feedback
    // is the dominant effect)
    if (mturn_mcg >= mturn_acg) {
        f_sample_mini = 0.;
    } else {
        // See comments above for how f_sample_mini is distributed
        mu_fstar_mini = f_7 * scaling_single_PL(halo_mass, f_a_mini, 1e7);
        f_sample_mini = mu_fstar_mini * exp(-mturn_mcg / halo_mass - halo_mass / mturn_acg +
                                            star_rng * sigma_star - stoc_adjustment_term);
    }
    if (f_sample_mini > 1.) f_sample_mini = 1.;

    star_mass_sample_mini = f_sample_mini * halo_mass * baryon_ratio;
    *star_mcg = star_mass_sample_mini;
}

void get_halo_sfr(double stellar_mass, double stellar_mass_mini, double sfr_rng,
                  ScalingConstants *consts, double *sfr, double *sfr_mini) {
    double mu_sfr, mu_sfr_mini;
    double sfr_sample, sfr_sample_mini;

    double sigma_sfr_lim = consts->sigma_sfr_lim;
    double sigma_sfr_idx = consts->sigma_sfr_idx;

    // set the scatter based on the total Stellar mass
    // We use the total stellar mass (MCG + ACG) NOTE: it might be better to separate later
    double sigma_sfr = 0.;

    if (sigma_sfr_lim > 0.) {
        // Set the sigma parameter for the distribution (according to Eq. 12 in
        // https://arxiv.org/pdf/2504.17254)
        sigma_sfr =
            sigma_sfr_idx * log10((stellar_mass + stellar_mass_mini) / 1e10) + sigma_sfr_lim;
        if (sigma_sfr < sigma_sfr_lim) sigma_sfr = sigma_sfr_lim;
    }

    // Set the mu parameter for the distribution (according to Eq. 11 in
    // https://arxiv.org/pdf/2504.17254) Note that the mu parameter is adjusted with exp(-sigma^2
    // /2), in case we want to interpret it as the mean of the sfr distribution, this exponent is
    // absorbed in the line for sfr_sample below for computational efficiency
    mu_sfr = stellar_mass / (consts->t_star * consts->t_h);

    // adjustment to the mean for lognormal scatter
    double stoc_adjustment_term = consts->scaling_median ? 0 : sigma_sfr * sigma_sfr / 2.;
    // NOTE: while the lognormal distribution in the C code is with respect to base-e (since the rng
    // variable distributes normally, and
    //      we use an exponent below), the sigma parameters are converted from the user's specified
    //      input ("sigma_user") in units of base-10 (dex) to base-e via dex2exp_transformer (in
    //      inputs.py). Therefore, the samples below are effectively distributed as lognormal in
    //      base-10, namely the log_10 of the sample distributes as a Gaussian with s.t.d equals to
    //      sigma_user, while the mean (median) of the samples (not their logarithm!) is given by
    //      the above mu_sfr, if HALO_SCALING_RELATIONS_MEDIAN = False (True)
    sfr_sample = mu_sfr * exp(sfr_rng * sigma_sfr - stoc_adjustment_term);
    *sfr = sfr_sample;

    if (!astro_options_global->USE_MINI_HALOS) {
        *sfr_mini = 0.;
        return;
    }

    // See comments above for how sfr_sample_mini is distributed
    mu_sfr_mini = stellar_mass_mini / (consts->t_star * consts->t_h);
    sfr_sample_mini = mu_sfr_mini * exp(sfr_rng * sigma_sfr - stoc_adjustment_term);
    *sfr_mini = sfr_sample_mini;
}

void get_halo_metallicity(double sfr, double stellar, double redshift, double *z_out) {
    // This function follows Eq. 14 and 15 in https://arxiv.org/pdf/2504.17254
    // Hardcoded for now: 6 extra fit parameters in the equation
    double M0, z_sample;
    double redshift_scaling = pow(10, -0.056 * redshift + 0.064);
    double stellar_term = 1.;

    // We need to avoid denominator == 0. In this case at sfr == 0 the
    //   metallicity won't matter since it's used solely for lx/sfr
    //   the case where stellar > 0 and sfr == 0 makes no sense in this
    //   scaling relation and only currently occurs due to underflow of sfr
    //   where M* ~ 1e-300, but if we change the SSFR relation to a MAR relation
    //   we will need to change this as well
    if (stellar > 0 && sfr > 0.) {
        M0 = (1.28825e10 * pow(sfr * physconst.s_per_yr, 0.56));
        stellar_term = pow(1 + pow(stellar / M0, -2.1), -0.148);
    }

    z_sample = 1.23 * stellar_term * redshift_scaling;

    *z_out = z_sample;
}

void get_halo_xray(double sfr, double sfr_mini, double metallicity, double xray_rng,
                   ScalingConstants *consts, double *xray_out) {
    double sigma_xray = consts->sigma_xray;
    double mu_x, xray_sample;

    // Set the mu parameter for the distribution (according to Eq. 13 in
    // https://arxiv.org/pdf/2504.17254) Note that the mu parameter is adjusted with exp(-sigma^2
    // /2), in case we want to interpret it as the mean of the xray distribution, this exponent is
    // absorbed in the line for xray_sample below for computational efficiency
    mu_x = get_lx_on_sfr(sfr, metallicity, consts->l_x) * (sfr * physconst.s_per_yr);

    double mu_x_mini = 0.;
    if (astro_options_global->USE_MINI_HALOS) {
        // Since there *are* some SFR-dependent
        // models, this is done separately
        mu_x_mini = get_lx_on_sfr(sfr_mini, metallicity, consts->l_x_mini) *
                    (sfr_mini * physconst.s_per_yr);
    }
    mu_x += mu_x_mini;

    // adjustment to the mean for lognormal scatter
    double stoc_adjustment_term = consts->scaling_median ? 0 : sigma_xray * sigma_xray / 2.;
    // NOTE: while the lognormal distribution in the C code is with respect to base-e (since the rng
    // variable distributes normally, and
    //      we use an exponent below), the sigma parameters are converted from the user's specified
    //      input ("sigma_user") in units of base-10 (dex) to base-e via dex2exp_transformer (in
    //      inputs.py). Therefore, the samples below are effectively distributed as lognormal in
    //      base-10, namely the log_10 of the sample distributes as a Gaussian with s.t.d equals to
    //      sigma_user, while the mean (median) of the samples (not their logarithm!) is given by
    //      the above mu_x, if HALO_SCALING_RELATIONS_MEDIAN = False (True)
    xray_sample = mu_x * exp(xray_rng * consts->sigma_xray - stoc_adjustment_term);
    *xray_out = xray_sample;
}
