/* This file will contain functions which contain methods to directly calculate integrals from the
 * frontend */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "Stochasticity.h"
#include "cosmology.h"
#include "hmf.h"
#include "interp_tables.h"
#include "interpolation.h"
#include "logger.h"
#include "scaling_relations.h"

void get_sigma(int n_masses, double *mass_values, double *sigma_out, double *dsigmasqdm_out) {
    init_ps();

    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_MIN_INTEGRAL, 1e20);

    int i;
    for (i = 0; i < n_masses; i++) {
        sigma_out[i] = EvaluateSigma(log(mass_values[i]));
        dsigmasqdm_out[i] = EvaluatedSigmasqdm(log(mass_values[i]));
    }
}

// we normally don't bounds-check the tables since they're hidden in the backend
//  but these functions are exposed to the user pretty directly so we do it here
bool cond_table_out_of_bounds(struct HaloSamplingConstants *consts) {
    return consts->M_cond < simulation_options_global->SAMPLER_MIN_MASS ||
           consts->M_cond > consts->M_max_tables || consts->delta < -1 ||
           consts->delta > MAX_DELTAC_FRAC * get_delta_crit(matter_options_global->HMF,
                                                            consts->sigma_cond, consts->growth_out);
}

// integrates at fixed (set by parameters) mass range for many conditions
void get_condition_integrals(double redshift, double z_prev, int n_conditions, double *cond_values,
                             double *out_n_exp, double *out_m_exp) {
    struct HaloSamplingConstants hs_const_struct;
    // unneccessarily creates the inverse table (a few seconds) but much cleaner this way
    stoc_set_consts_z(&hs_const_struct, redshift, z_prev);

    int i;
    for (i = 0; i < n_conditions; i++) {
        stoc_set_consts_cond(&hs_const_struct, cond_values[i]);
        if (cond_table_out_of_bounds(&hs_const_struct)) {
            out_n_exp[i] = -1.;
            continue;
        }
        out_n_exp[i] = hs_const_struct.expected_N;
        out_m_exp[i] = hs_const_struct.expected_M;
    }
}

// A more flexible form of the function above, but with many mass ranges for outputting tables of
// CHMF integrals
//   Requires extra arguments for the mass limits
void get_halo_chmf_interval(double redshift, double z_prev, int n_conditions, double *cond_values,
                            int n_masslim, double *lnM_lo, double *lnM_hi, double *out_n) {
    // unneccessarily creates tables if flags are set (a few seconds)
    struct HaloSamplingConstants hs_const_struct;
    stoc_set_consts_z(&hs_const_struct, redshift, z_prev);

    // we're only using the HS constants here to do mass/sigma calculations
    //   re-doing the sigma tables here lets us integrate below SAMPLER_MIN_MASS
    //   if requested by the user.
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_MIN_INTEGRAL, M_MAX_INTEGRAL);

    int i, j;
    double exp_n_total;
    double buf;
    for (i = 0; i < n_conditions; i++) {
        stoc_set_consts_cond(&hs_const_struct, cond_values[i]);
        for (j = 0; j < n_masslim; j++) {
            buf = Nhalo_Conditional(hs_const_struct.growth_out, lnM_lo[j], lnM_hi[j],
                                    hs_const_struct.lnM_cond, hs_const_struct.sigma_cond,
                                    hs_const_struct.delta,
                                    0  // QAG
                                    ) *
                  hs_const_struct.M_cond;
            out_n[i * n_masslim + j] = buf;
        }
    }
}

void get_halomass_at_probability(double redshift, double z_prev, int n_conditions,
                                 double *cond_values, double *probabilities, double *out_mass) {
    struct HaloSamplingConstants hs_const_struct;
    stoc_set_consts_z(&hs_const_struct, redshift, z_prev);

    int i;
    bool out_of_bounds;
    for (i = 0; i < n_conditions; i++) {
        stoc_set_consts_cond(&hs_const_struct, cond_values[i]);
        out_of_bounds = cond_table_out_of_bounds(&hs_const_struct);
        out_of_bounds = out_of_bounds || probabilities[i] < 0 || probabilities[i] > 1;
        if (out_of_bounds)
            out_mass[i] = -1;  // mark invalid
        else
            out_mass[i] = EvaluateNhaloInv(hs_const_struct.cond_val, probabilities[i]) *
                          hs_const_struct.M_cond;
    }
}

void get_global_SFRD_z(int n_redshift, double *redshifts, double *log10_turnovers_mcg,
                       double *out_sfrd, double *out_sfrd_mini) {
    init_ps();

    // a bit hacky, but we need a lower limit for the tables
    double M_min = minimum_source_mass(redshifts[0], true);
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min, 1e20);

    ScalingConstants sc;
    set_scaling_constants(redshifts[0], &sc, false);

    int i;
    double z_min = simulation_options_global->Z_HEAT_MAX;
    double z_max = 0.;
    for (i = 0; i < n_redshift; i++) {
        if (redshifts[i] < z_min) z_min = redshifts[i];
        if (redshifts[i] > z_max) z_max = redshifts[i];
    }

    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        initialise_SFRD_spline(zpp_interp_points_SFR, z_min, z_max + 0.01, &sc);
    }

    for (i = 0; i < n_redshift; i++) {
        out_sfrd[i] = EvaluateSFRD(redshifts[i], &sc);
        if (astro_options_global->USE_MINI_HALOS)
            out_sfrd_mini[i] = EvaluateSFRD_MINI(redshifts[i], log10_turnovers_mcg[i], &sc);
    }
}

void get_global_Nion_z(int n_redshift, double *redshifts, double *log10_turnovers_mcg,
                       double *out_nion, double *out_nion_mini) {
    init_ps();

    double M_min = minimum_source_mass(redshifts[0], true);
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min, 1e20);

    ScalingConstants sc;
    set_scaling_constants(redshifts[0], &sc, false);

    int i;
    double z_min = simulation_options_global->Z_HEAT_MAX;
    double z_max = 0.;
    for (i = 0; i < n_redshift; i++) {
        if (redshifts[i] < z_min) z_min = redshifts[i];
        if (redshifts[i] > z_max) z_max = redshifts[i];
    }

    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        initialise_Nion_Ts_spline(zpp_interp_points_SFR, z_min, z_max + 0.01, &sc);
    }
    for (i = 0; i < n_redshift; i++) {
        out_nion[i] = EvaluateNionTs(redshifts[i], &sc);
        if (astro_options_global->USE_MINI_HALOS)
            out_nion_mini[i] = EvaluateNionTs_MINI(redshifts[i], log10_turnovers_mcg[i], &sc);
    }
}

void get_conditional_FgtrM(double redshift, double R, int n_densities, double *densities,
                           double *out_fcoll, double *out_dfcoll) {
    init_ps();

    double M_min = minimum_source_mass(redshift, true);
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min, 1e20);
    double sigma_min = EvaluateSigma(log(M_min));
    double sigma_cond = EvaluateSigma(log(RtoM(R)));
    double growthf = dicke(redshift);

    LOG_DEBUG("db F R = %.3e M = %.3e s = %.3e", R, RtoM(R), sigma_cond);

    int i;
    double min_dens = 10;
    double max_dens = -10;
    double dens;
    for (i = 0; i < n_densities; i++) {
        dens = densities[i];
        if (dens < min_dens) min_dens = dens;
        if (dens > max_dens) max_dens = dens;
    }
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        initialise_FgtrM_delta_table(min_dens, max_dens + 0.01, redshift, growthf, sigma_min,
                                     sigma_cond);
    }
    LOG_DEBUG("Done tables");

    for (i = 0; i < n_densities; i++) {
        out_fcoll[i] = EvaluateFcoll_delta(densities[i], growthf, sigma_min, sigma_cond);
        out_dfcoll[i] = EvaluatedFcolldz(densities[i], redshift, sigma_min, sigma_cond);
    }
}

void get_conditional_SFRD(double redshift, double R, int n_densities, double *densities,
                          double *log10_mturns, double *out_sfrd, double *out_sfrd_mini) {
    init_ps();

    double M_min = minimum_source_mass(redshift, true);
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min, 1e20);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == 1 ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == 1))
        initialise_GL(log(M_min), log(M_cond));

    ScalingConstants sc;
    set_scaling_constants(redshift, &sc, false);

    int i;
    double min_dens = -1;
    double max_dens = 10;
    double dens;
    for (i = 0; i < n_densities; i++) {
        dens = densities[i];
        if (dens < min_dens) min_dens = dens;
        if (dens > max_dens) max_dens = dens;
    }

    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        initialise_SFRD_Conditional_table(redshift, min_dens, max_dens, M_min, M_cond, M_cond, &sc);
    }
    for (i = 0; i < n_densities; i++)
        out_sfrd[i] =
            EvaluateSFRD_Conditional(densities[i], growthf, M_min, M_cond, M_cond, sigma_cond, &sc);
    if (astro_options_global->USE_MINI_HALOS) {
        for (i = 0; i < n_densities; i++)
            out_sfrd_mini[i] = EvaluateSFRD_Conditional_MINI(
                densities[i], log10_mturns[i], growthf, M_min, M_cond, M_cond, sigma_cond, &sc);
    }
}

void get_conditional_Nion(double redshift, double R, int n_densities, double *densities,
                          double *log10_mturns_acg, double *log10_mturns_mcg, double *out_nion,
                          double *out_nion_mini) {
    init_ps();

    double M_min = minimum_source_mass(redshift, true);
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min, 1e20);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == 1 ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == 1))
        initialise_GL(log(M_min), log(M_cond));

    ScalingConstants sc;
    set_scaling_constants(redshift, &sc, false);

    int i;
    double min_dens = -1;
    double max_dens = 10;
    double min_l10mturn_acg = 10.;
    double min_l10mturn_mcg = 10.;
    double max_l10mturn_acg = 5.;
    double max_l10mturn_mcg = 5.;
    double dens, l10mturn_a, l10mturn_m;
    for (i = 0; i < n_densities; i++) {
        dens = densities[i];
        l10mturn_a = log10_mturns_acg[i];
        l10mturn_m = log10_mturns_mcg[i];
        if (dens < min_dens) min_dens = dens;
        if (dens > max_dens) max_dens = dens;
        if (l10mturn_a < min_l10mturn_acg) min_l10mturn_acg = l10mturn_a;
        if (l10mturn_a > max_l10mturn_acg) max_l10mturn_acg = l10mturn_a;
        if (l10mturn_m < min_l10mturn_mcg) min_l10mturn_mcg = l10mturn_m;
        if (l10mturn_m > max_l10mturn_mcg) max_l10mturn_mcg = l10mturn_m;
    }

    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        initialise_Nion_Conditional_spline(redshift, min_dens, max_dens, M_min, M_cond, M_cond,
                                           min_l10mturn_acg, max_l10mturn_acg, min_l10mturn_mcg,
                                           max_l10mturn_mcg, &sc, false);
    }
    for (i = 0; i < n_densities; i++)
        out_nion[i] = EvaluateNion_Conditional(densities[i], log10_mturns_acg[i], growthf, M_min,
                                               M_cond, M_cond, sigma_cond, &sc, false);
    if (astro_options_global->USE_MINI_HALOS) {
        for (i = 0; i < n_densities; i++)
            out_nion_mini[i] =
                EvaluateNion_Conditional_MINI(densities[i], log10_mturns_mcg[i], growthf, M_min,
                                              M_cond, M_cond, sigma_cond, &sc, false);
    }
}

void get_conditional_Xray(double redshift, double R, int n_densities, double *densities,
                          double *log10_mturns, double *out_xray) {
    init_ps();

    double M_min = minimum_source_mass(redshift, true);
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min, 1e20);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == 1 ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == 1))
        initialise_GL(log(M_min), log(M_cond));

    ScalingConstants sc;
    set_scaling_constants(redshift, &sc, false);

    int i;
    double min_dens = -1;
    double max_dens = 10;
    double dens;
    for (i = 0; i < n_densities; i++) {
        dens = densities[i];
        if (dens < min_dens) min_dens = dens;
        if (dens > max_dens) max_dens = dens;
    }

    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        initialise_Xray_Conditional_table(redshift, min_dens, max_dens, M_min, M_cond, M_cond, &sc);
    }
    for (i = 0; i < n_densities; i++)
        out_xray[i] = EvaluateXray_Conditional(densities[i], log10_mturns[i], redshift, growthf,
                                               M_min, M_cond, M_cond, sigma_cond, &sc);
}
