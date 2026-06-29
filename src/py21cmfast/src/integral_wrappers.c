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

void get_global_SFRD_z(int n_redshift, double *redshifts, double *log10_turnovers_acg,
                       double *log10_turnovers_mcg, double *out_sfrd, double *out_sfrd_mini) {
    ScalingConstants sc;
    set_scaling_constants(redshifts[0], &sc, false);

    int i;
    double z_min = simulation_options_global->Z_HEAT_MAX;
    double z_max = 0.;
    double SFRD_FACTOR, SFRD_FACTOR_MINI;

    for (i = 0; i < n_redshift; i++) {
        if (redshifts[i] < z_min) z_min = redshifts[i];
        if (redshifts[i] > z_max) z_max = redshifts[i];
    }

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        initialise_SFRD_spline(zpp_interp_points_SFR, z_min, z_max + 0.01, &sc);
    }

    for (i = 0; i < n_redshift; i++) {
        SFRD_FACTOR = astro_params_global->F_STAR10 * cosmo_params_global->OMb * RHOcrit /
                      astro_params_global->t_STAR / t_hubble(redshifts[i]);
        out_sfrd[i] = SFRD_FACTOR * EvaluateSFRD(redshifts[i], log10_turnovers_acg[i], &sc);
        if (astro_options_global->USE_MINI_HALOS) {
            SFRD_FACTOR_MINI = astro_params_global->F_STAR7_MINI * cosmo_params_global->OMb *
                               RHOcrit / astro_params_global->t_STAR / t_hubble(redshifts[i]);
            out_sfrd_mini[i] =
                SFRD_FACTOR_MINI * EvaluateSFRD_MINI(redshifts[i], log10_turnovers_acg[i],
                                                     log10_turnovers_mcg[i], &sc);
        }
    }
}

void get_global_Nion_z(int n_redshift, double *redshifts, double *log10_turnovers_acg,
                       double *log10_turnovers_mcg, double *out_nion, double *out_nion_mini) {
    ScalingConstants sc;
    set_scaling_constants(redshifts[0], &sc, false);

    int i;
    double z_min = simulation_options_global->Z_HEAT_MAX;
    double z_max = 0.;
    for (i = 0; i < n_redshift; i++) {
        if (redshifts[i] < z_min) z_min = redshifts[i];
        if (redshifts[i] > z_max) z_max = redshifts[i];
    }

    double ION_EFF_FACTOR, ION_EFF_FACTOR_MINI;
    if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
        ION_EFF_FACTOR = astro_params_global->F_STAR10 * astro_params_global->F_ESC10 *
                         astro_params_global->POP2_ION;
        ION_EFF_FACTOR_MINI = astro_params_global->F_STAR7_MINI * astro_params_global->F_ESC7_MINI *
                              astro_params_global->POP3_ION;
    } else {
        // no mini-halos when SOURCE_MODE is mass independent (constant ionization efficiency)
        ION_EFF_FACTOR = astro_params_global->HII_EFF_FACTOR;
    }

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        initialise_Nion_Ts_spline(zpp_interp_points_SFR, z_min, z_max + 0.01, &sc);
    }
    for (i = 0; i < n_redshift; i++) {
        out_nion[i] = ION_EFF_FACTOR * EvaluateNionTs(redshifts[i], log10_turnovers_acg[i], &sc);
        if (astro_options_global->USE_MINI_HALOS)
            out_nion_mini[i] =
                ION_EFF_FACTOR_MINI * EvaluateNionTs_MINI(redshifts[i], log10_turnovers_acg[i],
                                                          log10_turnovers_mcg[i], &sc);
    }
}

void get_conditional_FgtrM(double redshift, double R, int n_densities, double *densities,
                           double *out_fcoll, double *out_dfcoll) {
    double M_min = minimum_source_mass(redshift, true);
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
    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
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
                          double log10_mturn_acg, double log10_mturn_mcg, double *out_sfrd,
                          double *out_sfrd_mini) {
    double M_min = minimum_source_mass(redshift, true);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == INTEGRATION_METHOD_GAUSS_LEGENDRE ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == INTEGRATION_METHOD_GAUSS_LEGENDRE))
        initialise_GL(log(M_min), log(M_cond));

    ScalingConstants sc;
    set_scaling_constants(redshift, &sc, false);

    int i;
    double min_dens = -1;
    double max_dens = 10;
    double dens;
    double SFRD_FACTOR = astro_params_global->F_STAR10 * cosmo_params_global->OMb * RHOcrit /
                         astro_params_global->t_STAR / t_hubble(redshift);
    double SFRD_FACTOR_MINI = astro_params_global->F_STAR7_MINI * cosmo_params_global->OMb *
                              RHOcrit / astro_params_global->t_STAR / t_hubble(redshift);

    for (i = 0; i < n_densities; i++) {
        dens = densities[i];
        if (dens < min_dens) min_dens = dens;
        if (dens > max_dens) max_dens = dens;
    }

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        initialise_SFRD_Conditional_table(redshift, min_dens, max_dens, M_min, M_cond, M_cond, &sc);
    }
    for (i = 0; i < n_densities; i++) {
        out_sfrd[i] =
            SFRD_FACTOR * EvaluateSFRD_Conditional(densities[i], log10_mturn_acg, growthf, M_min,
                                                   M_cond, M_cond, sigma_cond, &sc);
        if (astro_options_global->USE_MINI_HALOS) {
            out_sfrd_mini[i] =
                SFRD_FACTOR_MINI * EvaluateSFRD_Conditional_MINI(densities[i], log10_mturn_acg,
                                                                 log10_mturn_mcg, growthf, M_min,
                                                                 M_cond, M_cond, sigma_cond, &sc);
        }
    }
}

void get_conditional_Nion(double redshift, double R, int n_densities, double *densities,
                          double log10_mturn_acg, double log10_mturn_mcg, double *out_nion,
                          double *out_nion_mini) {
    double M_min = minimum_source_mass(redshift, true);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == INTEGRATION_METHOD_GAUSS_LEGENDRE ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == INTEGRATION_METHOD_GAUSS_LEGENDRE))
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

    // TODO: these lines do no appear in get_conditional_SFRD, even though both functions are very
    // similar. See https://github.com/21cmfast/21cmFAST/issues/732 for more details
    double eps = 0.01;
    double min_l10mturn_acg = log10_mturn_acg - eps;
    double max_l10mturn_acg = log10_mturn_acg + eps;
    double min_l10mturn_mcg = log10_mturn_mcg - eps;
    double max_l10mturn_mcg = log10_mturn_mcg + eps;

    double ION_EFF_FACTOR, ION_EFF_FACTOR_MINI;
    if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
        ION_EFF_FACTOR = astro_params_global->F_STAR10 * astro_params_global->F_ESC10 *
                         astro_params_global->POP2_ION;
        ION_EFF_FACTOR_MINI = astro_params_global->F_STAR7_MINI * astro_params_global->F_ESC7_MINI *
                              astro_params_global->POP3_ION;
    } else {
        // no mini-halos when SOURCE_MODE is mass independent (constant ionization efficiency)
        ION_EFF_FACTOR = astro_params_global->HII_EFF_FACTOR;
    }

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        initialise_Nion_Conditional_spline(redshift, min_dens, max_dens, M_min, M_cond, M_cond,
                                           min_l10mturn_acg, max_l10mturn_acg, min_l10mturn_mcg,
                                           max_l10mturn_mcg, &sc, false);
    }
    for (i = 0; i < n_densities; i++)
        out_nion[i] =
            ION_EFF_FACTOR * EvaluateNion_Conditional(densities[i], log10_mturn_acg, growthf, M_min,
                                                      M_cond, M_cond, sigma_cond, &sc, false);
    if (astro_options_global->USE_MINI_HALOS) {
        for (i = 0; i < n_densities; i++)
            out_nion_mini[i] =
                ION_EFF_FACTOR_MINI * EvaluateNion_Conditional_MINI(
                                          densities[i], log10_mturn_acg, log10_mturn_mcg, growthf,
                                          M_min, M_cond, M_cond, sigma_cond, &sc, false);
    }
}

void get_conditional_Xray(double redshift, double R, int n_densities, double *densities,
                          double log10_mturn_acg, double log10_mturn_mcg, double *out_xray) {
    double M_min = minimum_source_mass(redshift, true);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == INTEGRATION_METHOD_GAUSS_LEGENDRE ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == INTEGRATION_METHOD_GAUSS_LEGENDRE))
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

    double X_RAY_FACTOR = RHOcrit * cosmo_params_global->OMm * 1e38;

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        initialise_Xray_Conditional_table(redshift, min_dens, max_dens, M_min, M_cond, M_cond, &sc);
    }
    for (i = 0; i < n_densities; i++)
        out_xray[i] = X_RAY_FACTOR * EvaluateXray_Conditional(
                                         densities[i], log10_mturn_acg, log10_mturn_mcg, redshift,
                                         growthf, M_min, M_cond, M_cond, sigma_cond, &sc);
}
