/* This file contains fucntions for calculating the HaloBox output for 21cmfast, containing the
 * gridded source properties, either from integrating the conditional mass functions in a cell or
 * from the halo sampler */
#include "HaloBox.h"

#include <gsl/gsl_sf_gamma.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "exceptions.h"
#include "hmf.h"
#include "indexing.h"
#include "interp_tables.h"
#include "logger.h"
#include "scaling_relations.h"
#include "thermochem.h"

// struct holding each halo property we currently need.
// This is only used for both averages over the box/catalogues
//   as well as an individual halo's properties
struct HaloProperties {
    double halo_mass;
    double stellar_mass;
    double halo_sfr;
    double stellar_mass_mini;
    double sfr_mini;
    double fescweighted_sfr;
    double n_ion;
    double halo_xray;
    double metallicity;
    double m_turn_acg;
    double m_turn_mcg;
    double m_turn_reion;
};

// calculates halo properties from astro parameters plus the correlated rng
// The inputs include all properties with a separate RNG
// The outputs include all sampled halo properties PLUS all properties which cannot be recovered
// when mixing all the halos together
//   i.e escape fraction weighting, minihalo stuff that has separate parameters
// Since there are so many spectral terms in the spin temperature calculation, it will be most
// efficient to split SFR into regular and minihalos
//   BUT not split the ionisedbox fields.
// in order to remain consistent with the minihalo treatment in default (Nion_a * exp(-M/M_a) +
// Nion_m * exp(-M/M_m - M_a/M))
//   we treat the minihalos as a shift in the mean, where each halo will have both components,
//   representing a smooth transition in halo mass from one set of SFR/emmissivity parameters to the
//   other.
void set_halo_properties(double halo_mass, double M_turn_a, double M_turn_m,
                         struct ScalingConstants *consts, double *input_rng,
                         struct HaloProperties *output) {
    double n_ion_sample, wsfr_sample;
    double fesc;
    double fesc_mini = 0.;

    double stellar_mass, stellar_mass_mini;
    get_halo_stellarmass(halo_mass, M_turn_a, M_turn_m, input_rng[0], consts, &stellar_mass,
                         &stellar_mass_mini);

    double sfr, sfr_mini;
    get_halo_sfr(stellar_mass, stellar_mass_mini, input_rng[1], consts, &sfr, &sfr_mini);

    double metallicity = 0;
    double xray_lum = 0;
    if (astro_options_global->USE_TS_FLUCT) {
        get_halo_metallicity(sfr + sfr_mini, stellar_mass + stellar_mass_mini, consts->redshift,
                             &metallicity);
        get_halo_xray(sfr, sfr_mini, metallicity, input_rng[2], consts, &xray_lum);
    }

    // no rng for escape fraction yet
    // add redshift dependence to fesc
    fesc = fmin(consts->fesc_10 * pow(halo_mass / 1e10, consts->alpha_esc) * consts->zesc_power_law,
                1);
    if (astro_options_global->USE_MINI_HALOS)
        fesc_mini = fmin(consts->fesc_7 * pow(halo_mass / 1e7, consts->alpha_esc), 1);

    n_ion_sample =
        stellar_mass * consts->pop2_ion * fesc + stellar_mass_mini * consts->pop3_ion * fesc_mini;
    wsfr_sample = sfr * consts->pop2_ion * fesc + sfr_mini * consts->pop3_ion * fesc_mini;

    output->halo_mass = halo_mass;
    output->stellar_mass = stellar_mass;
    output->stellar_mass_mini = stellar_mass_mini;
    output->halo_sfr = sfr;
    output->sfr_mini = sfr_mini;
    output->fescweighted_sfr = wsfr_sample;
    output->n_ion = n_ion_sample;
    output->metallicity = metallicity;
    output->halo_xray = xray_lum;
}

// Expected global averages for box quantities for mean adjustment
// WARNING: THESE AVERAGE BOXES ARE WRONG, CHECK THEM
int get_box_averages(double M_min, double M_max, double M_turn_a, double M_turn_m,
                     struct ScalingConstants *consts, struct HaloProperties *averages_out) {
    LOG_SUPER_DEBUG("Getting Box averages z=%.2f M [%.2e %.2e] Mt [%.2e %.2e]", consts->redshift,
                    M_min, M_max, M_turn_a, M_turn_m);
    double t_h = consts->t_h;
    double lnMmax = log(M_max);
    double lnMmin = log(M_min);

    double prefactor_mass = RHOcrit * cosmo_params_global->OMm;
    double prefactor_stars = RHOcrit * cosmo_params_global->OMb * consts->fstar_10;
    double prefactor_stars_mini = RHOcrit * cosmo_params_global->OMb * consts->fstar_7;
    double prefactor_sfr = prefactor_stars / consts->t_star / t_h;
    double prefactor_sfr_mini = prefactor_stars_mini / consts->t_star / t_h;
    double prefactor_nion =
        prefactor_stars * consts->fesc_10 * consts->pop2_ion * consts->zesc_power_law;
    double prefactor_nion_mini = prefactor_stars_mini * consts->fesc_7 * consts->pop3_ion;
    double prefactor_wsfr =
        prefactor_sfr * consts->fesc_10 * consts->pop2_ion * consts->zesc_power_law;
    double prefactor_wsfr_mini = prefactor_sfr_mini * consts->fesc_7 * consts->pop3_ion;
    double prefactor_xray = RHOcrit * cosmo_params_global->OMm;

    double mass_intgrl;
    double intgrl_fesc_weighted, intgrl_stars_only;
    double intgrl_fesc_weighted_mini = 0., intgrl_stars_only_mini = 0., integral_xray = 0.;

    // NOTE: we use the atomic method for all halo mass/count here
    mass_intgrl = Fcoll_General(consts->redshift, lnMmin, lnMmax);
    struct ScalingConstants consts_sfrd = evolve_scaling_constants_sfr(consts);

    intgrl_fesc_weighted = Nion_General(consts->redshift, lnMmin, lnMmax, M_turn_a, consts);
    intgrl_stars_only = Nion_General(consts->redshift, lnMmin, lnMmax, M_turn_a, &consts_sfrd);
    if (astro_options_global->USE_MINI_HALOS) {
        intgrl_fesc_weighted_mini =
            Nion_General_MINI(consts->redshift, lnMmin, lnMmax, M_turn_m, consts);

        intgrl_stars_only_mini =
            Nion_General_MINI(consts->redshift, lnMmin, lnMmax, M_turn_m, &consts_sfrd);
    }
    if (astro_options_global->USE_TS_FLUCT) {
        integral_xray = Xray_General(consts->redshift, lnMmin, lnMmax, M_turn_a, M_turn_m, consts);
    }

    averages_out->halo_mass = mass_intgrl * prefactor_mass;
    averages_out->stellar_mass = intgrl_stars_only * prefactor_stars;
    averages_out->halo_sfr = intgrl_stars_only * prefactor_sfr;
    averages_out->stellar_mass_mini = intgrl_stars_only_mini * prefactor_stars_mini;
    averages_out->sfr_mini = intgrl_stars_only_mini * prefactor_sfr_mini;
    averages_out->n_ion =
        intgrl_fesc_weighted * prefactor_nion + intgrl_fesc_weighted_mini * prefactor_nion_mini;
    averages_out->fescweighted_sfr =
        intgrl_fesc_weighted * prefactor_wsfr + intgrl_fesc_weighted_mini * prefactor_wsfr_mini;
    averages_out->halo_xray = prefactor_xray * integral_xray;
    averages_out->m_turn_acg = M_turn_a;
    averages_out->m_turn_mcg = M_turn_m;

    return 0;
}

// This takes a HaloBox struct and fixes it's mean to exactly what we expect from the UMF integrals.
//   Generally should only be done for the fixed portion of the grids, since
//   it will otherwise make the box inconsistent with the input catalogue
void mean_fix_grids(double M_min, double M_max, HaloBox *grids, struct HaloProperties *averages_box,
                    struct ScalingConstants *consts) {
    struct HaloProperties averages_global;
    double M_turn_a_global = averages_box->m_turn_acg;
    double M_turn_m_global = averages_box->m_turn_mcg;
    get_box_averages(M_min, M_max, M_turn_a_global, M_turn_m_global, consts, &averages_global);

    unsigned long long int idx;
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS) private(idx)
    for (idx = 0; idx < HII_TOT_NUM_PIXELS; idx++) {
        grids->halo_mass[idx] *= averages_global.halo_mass / averages_box->halo_mass;
        grids->halo_stars[idx] *= averages_global.stellar_mass / averages_box->stellar_mass;
        grids->halo_sfr[idx] *= averages_global.halo_sfr / averages_box->halo_sfr;
        grids->n_ion[idx] *= averages_global.n_ion / averages_box->n_ion;
        if (astro_options_global->USE_MINI_HALOS) {
            grids->halo_stars_mini[idx] *=
                averages_global.stellar_mass_mini / averages_box->stellar_mass_mini;
            grids->halo_sfr_mini[idx] *= averages_global.sfr_mini / averages_box->sfr_mini;
        }
        if (astro_options_global->USE_TS_FLUCT) {
            grids->halo_xray[idx] *= averages_global.halo_xray / averages_box->halo_xray;
        }
        if (astro_options_global->INHOMO_RECO) {
            grids->whalo_sfr[idx] *=
                averages_global.fescweighted_sfr / averages_box->fescweighted_sfr;
        }
    }
}

// Fixed halo grids, where each property is set as the integral of the CMF on the EULERIAN cell
// scale As per default 21cmfast (strange pretending that the lagrangian density is eulerian and
// then *(1+delta)) This outputs the UN-NORMALISED grids (before mean-adjustment)
int set_fixed_grids(double M_min, double M_max, InitialConditions *ini_boxes,
                    PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                    IonizedBox *previous_ionize_box, struct ScalingConstants *consts,
                    HaloBox *grids, struct HaloProperties *averages, const bool eulerian) {
    double M_cell = RHOcrit * cosmo_params_global->OMm * VOLUME /
                    HII_TOT_NUM_PIXELS;  // mass in cell of mean dens
    double growth_z = dicke(consts->redshift);

    double lnMmin = log(M_min);
    double lnMcell = log(M_cell);
    double lnMmax = log(M_max);

    double sigma_cell = EvaluateSigma(lnMcell);

    double prefactor_mass = RHOcrit * cosmo_params_global->OMm;
    double prefactor_stars = RHOcrit * cosmo_params_global->OMb * consts->fstar_10;
    double prefactor_stars_mini = RHOcrit * cosmo_params_global->OMb * consts->fstar_7;
    double prefactor_sfr = prefactor_stars / consts->t_star / consts->t_h;
    double prefactor_sfr_mini = prefactor_stars_mini / consts->t_star / consts->t_h;
    double prefactor_nion =
        prefactor_stars * consts->fesc_10 * consts->pop2_ion * consts->zesc_power_law;
    double prefactor_nion_mini = prefactor_stars_mini * consts->fesc_7 * consts->pop3_ion;
    double prefactor_wsfr =
        prefactor_sfr * consts->fesc_10 * consts->pop2_ion * consts->zesc_power_law;
    double prefactor_wsfr_mini = prefactor_sfr_mini * consts->fesc_7 * consts->pop3_ion;
    double prefactor_xray = RHOcrit * cosmo_params_global->OMm;

    double hm_sum = 0, nion_sum = 0, wsfr_sum = 0, xray_sum = 0;
    double sm_sum = 0, sm_sum_mini = 0, sfr_sum = 0, sfr_sum_mini = 0;
    double l10_mlim_m_sum = 0., l10_mlim_a_sum = 0., l10_mlim_r_sum = 0.;

    // find grid limits for tables
    double min_density = 0.;
    double max_density = 0.;
    double min_log10_mturn_a = log10(M_MAX_INTEGRAL);
    double min_log10_mturn_m = log10(M_MAX_INTEGRAL);
    double max_log10_mturn_a = log10(astro_params_global->M_TURN);
    double max_log10_mturn_m = log10(astro_params_global->M_TURN);
    float *mturn_a_grid = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
    float *mturn_m_grid = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        unsigned long long int i;
        double dens;
        double J21_val, Gamma12_val, zre_val;
        double M_turn_r = 0.;
        double M_turn_m = consts->mturn_m_nofb;
        double M_turn_a = consts->mturn_a_nofb;
        double curr_vcb = consts->vcb_norel;
#pragma omp for reduction(min : min_density, min_log10_mturn_a, min_log10_mturn_m) \
    reduction(max : max_density, max_log10_mturn_a, max_log10_mturn_m)             \
    reduction(+ : l10_mlim_m_sum, l10_mlim_a_sum, l10_mlim_r_sum)
        for (i = 0; i < HII_TOT_NUM_PIXELS; i++) {
            if (eulerian)
                dens = perturbed_field->density[i];
            else
                dens = ini_boxes->lowres_density[i] * growth_z;
            if (dens > max_density) max_density = dens;
            if (dens < min_density) min_density = dens;

            if (astro_options_global->USE_MINI_HALOS) {
                if (!astro_options_global->FIX_VCB_AVG &&
                    matter_options_global->USE_RELATIVE_VELOCITIES) {
                    curr_vcb = ini_boxes->lowres_vcb[i];
                }
                J21_val = Gamma12_val = zre_val = 0.;
                if (consts->redshift < simulation_options_global->Z_HEAT_MAX) {
                    J21_val = previous_spin_temp->J_21_LW[i];
                    Gamma12_val = previous_ionize_box->ionisation_rate_G12[i];
                    zre_val = previous_ionize_box->z_reion[i];
                }
                M_turn_a = consts->mturn_a_nofb;
                M_turn_m = lyman_werner_threshold(consts->redshift, J21_val, curr_vcb);
                M_turn_r = reionization_feedback(consts->redshift, Gamma12_val, zre_val);
                M_turn_a = fmax(M_turn_a, fmax(M_turn_r, astro_params_global->M_TURN));
                M_turn_m = fmax(M_turn_m, fmax(M_turn_r, astro_params_global->M_TURN));
            }
            mturn_a_grid[i] = log10(M_turn_a);
            mturn_m_grid[i] = log10(M_turn_m);

            if (min_log10_mturn_a > mturn_a_grid[i]) min_log10_mturn_a = mturn_a_grid[i];
            if (min_log10_mturn_m > mturn_m_grid[i]) min_log10_mturn_m = mturn_m_grid[i];
            if (max_log10_mturn_a < mturn_a_grid[i]) max_log10_mturn_a = mturn_a_grid[i];
            if (max_log10_mturn_m < mturn_m_grid[i]) max_log10_mturn_m = mturn_m_grid[i];

            l10_mlim_a_sum += mturn_a_grid[i];
            l10_mlim_m_sum += mturn_m_grid[i];
            l10_mlim_r_sum += log10(M_turn_r);
        }
    }
    // buffers for table ranges
    min_density = min_density * 1.001;  // negative
    max_density = max_density * 1.001;
    min_log10_mturn_a = min_log10_mturn_a * 0.999;
    min_log10_mturn_m = min_log10_mturn_m * 0.999;
    max_log10_mturn_a = max_log10_mturn_a * 1.001;
    max_log10_mturn_m = max_log10_mturn_m * 1.001;

    LOG_DEBUG("Mean halo boxes || M = [%.2e %.2e] | Mcell = %.2e (s=%.2e) | z = %.2e | D = %.2e",
              M_min, M_max, M_cell, sigma_cell, consts->redshift, growth_z);

    // These tables are coarser than needed, an initial loop for Mturn to find limits may help
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        if (astro_options_global->INTEGRATION_METHOD_ATOMIC == 1 ||
            (astro_options_global->USE_MINI_HALOS &&
             astro_options_global->INTEGRATION_METHOD_MINI == 1)) {
            initialise_GL(lnMmin, lnMmax);
        }

        // This table assumes no reionisation feedback
        initialise_SFRD_Conditional_table(consts->redshift, min_density, max_density, M_min, M_max,
                                          M_cell, consts);

        // This table includes reionisation feedback, but takes the atomic turnover anyway for the
        // upper turnover
        initialise_Nion_Conditional_spline(consts->redshift, min_density, max_density, M_min, M_max,
                                           M_cell, min_log10_mturn_a, max_log10_mturn_a,
                                           min_log10_mturn_m, max_log10_mturn_m, consts, false);

        initialise_dNdM_tables(min_density, max_density, lnMmin, lnMmax, growth_z, lnMcell, false);
        if (astro_options_global->USE_TS_FLUCT) {
            initialise_Xray_Conditional_table(consts->redshift, min_density, max_density, M_min,
                                              M_max, M_cell, consts);
        }
    }

#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        unsigned long long int i;
        double dens;
        double l10_mturn_a, l10_mturn_m;
        double mass_intgrl, h_count;
        double intgrl_fesc_weighted, intgrl_stars_only;
        double intgrl_fesc_weighted_mini = 0., intgrl_stars_only_mini = 0., integral_xray = 0;
        double dens_fac;

#pragma omp for reduction(+ : hm_sum, sm_sum, sm_sum_mini, sfr_sum, sfr_sum_mini, xray_sum, \
                              nion_sum, wsfr_sum)
        for (i = 0; i < HII_TOT_NUM_PIXELS; i++) {
            if (eulerian) {
                dens = perturbed_field->density[i];
                dens_fac = (1. + dens);
            } else {
                dens = ini_boxes->lowres_density[i] * growth_z;
                dens_fac = 1.;
            }
            l10_mturn_a = mturn_a_grid[i];
            l10_mturn_m = mturn_m_grid[i];

            h_count = EvaluateNhalo(dens, growth_z, lnMmin, lnMmax, M_cell, sigma_cell, dens);
            mass_intgrl = EvaluateMcoll(dens, growth_z, lnMmin, lnMmax, M_cell, sigma_cell, dens);
            intgrl_fesc_weighted = EvaluateNion_Conditional(
                dens, l10_mturn_a, growth_z, M_min, M_max, M_cell, sigma_cell, consts, false);
            intgrl_stars_only =
                EvaluateSFRD_Conditional(dens, growth_z, M_min, M_max, M_cell, sigma_cell, consts);
            // TODO: SFRD tables still assume no reion feedback, this should be fixed
            //   although it doesn't affect the histories (only used in Ts) it makes outputs wrong
            //   for post-processing
            if (astro_options_global->USE_MINI_HALOS) {
                intgrl_stars_only_mini = EvaluateSFRD_Conditional_MINI(
                    dens, l10_mturn_m, growth_z, M_min, M_max, M_cell, sigma_cell, consts);
                intgrl_fesc_weighted_mini = EvaluateNion_Conditional_MINI(
                    dens, l10_mturn_m, growth_z, M_min, M_max, M_cell, sigma_cell, consts, false);
            }

            if (astro_options_global->USE_TS_FLUCT) {
                integral_xray =
                    EvaluateXray_Conditional(dens, l10_mturn_m, consts->redshift, growth_z, M_min,
                                             M_max, M_cell, sigma_cell, consts);
            }

            grids->count[i] = (int)(h_count * M_cell * dens_fac);  // NOTE: truncated
            grids->halo_mass[i] = mass_intgrl * prefactor_mass * dens_fac;
            grids->halo_sfr[i] = (intgrl_stars_only * prefactor_sfr) * dens_fac;
            grids->n_ion[i] = (intgrl_fesc_weighted * prefactor_nion +
                               intgrl_fesc_weighted_mini * prefactor_nion_mini) *
                              dens_fac;
            grids->halo_stars[i] = intgrl_stars_only * prefactor_stars * dens_fac;

            hm_sum += grids->halo_mass[i];
            nion_sum += grids->n_ion[i];
            sfr_sum += grids->halo_sfr[i];
            sm_sum += grids->halo_stars[i];

            if (astro_options_global->USE_TS_FLUCT) {
                grids->halo_xray[i] = prefactor_xray * integral_xray * dens_fac;
                xray_sum += grids->halo_xray[i];
            }
            if (astro_options_global->INHOMO_RECO) {
                grids->whalo_sfr[i] = (intgrl_fesc_weighted * prefactor_wsfr +
                                       intgrl_fesc_weighted_mini * prefactor_wsfr_mini) *
                                      dens_fac;
                wsfr_sum += grids->whalo_sfr[i];
            }
            if (astro_options_global->USE_MINI_HALOS) {
                grids->halo_stars_mini[i] =
                    intgrl_stars_only_mini * prefactor_stars_mini * dens_fac;
                grids->halo_sfr_mini[i] = intgrl_stars_only_mini * prefactor_sfr_mini * dens_fac;
                sm_sum_mini += grids->halo_stars_mini[i];
                sfr_sum_mini += grids->halo_sfr_mini[i];
            }
        }
    }

    LOG_ULTRA_DEBUG("Cell 0 Totals: HM: %.2e SM: %.2e SF: %.2e NI: %.2e ct : %d",
                    grids->halo_mass[HII_R_INDEX(0, 0, 0)], grids->halo_stars[HII_R_INDEX(0, 0, 0)],
                    grids->halo_sfr[HII_R_INDEX(0, 0, 0)], grids->halo_xray[HII_R_INDEX(0, 0, 0)],
                    grids->n_ion[HII_R_INDEX(0, 0, 0)], grids->count[HII_R_INDEX(0, 0, 0)]);
    if (astro_options_global->INHOMO_RECO) {
        LOG_ULTRA_DEBUG("FESC * SF %.2e", grids->whalo_sfr[HII_R_INDEX(0, 0, 0)]);
    }
    if (astro_options_global->USE_TS_FLUCT) {
        LOG_ULTRA_DEBUG("X-ray %.2e", grids->halo_xray[HII_R_INDEX(0, 0, 0)]);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        LOG_ULTRA_DEBUG("MINI SM %.2e SF %.2e", grids->halo_stars_mini[HII_R_INDEX(0, 0, 0)],
                        grids->halo_sfr_mini[HII_R_INDEX(0, 0, 0)]);
    }
    LOG_ULTRA_DEBUG("Mturn_a %.2e Mturn_m %.2e", mturn_a_grid[HII_R_INDEX(0, 0, 0)],
                    mturn_m_grid[HII_R_INDEX(0, 0, 0)]);

    free(mturn_a_grid);
    free(mturn_m_grid);
    free_conditional_tables();

    averages->halo_mass = hm_sum / HII_TOT_NUM_PIXELS;
    averages->stellar_mass = sm_sum / HII_TOT_NUM_PIXELS;
    averages->stellar_mass_mini = sm_sum_mini / HII_TOT_NUM_PIXELS;
    averages->halo_sfr = sfr_sum / HII_TOT_NUM_PIXELS;
    averages->sfr_mini = sfr_sum_mini / HII_TOT_NUM_PIXELS;
    averages->n_ion = nion_sum / HII_TOT_NUM_PIXELS;
    averages->halo_xray = xray_sum / HII_TOT_NUM_PIXELS;
    averages->fescweighted_sfr = wsfr_sum / HII_TOT_NUM_PIXELS;
    averages->m_turn_acg = pow(10, l10_mlim_a_sum / HII_TOT_NUM_PIXELS);
    averages->m_turn_mcg = pow(10, l10_mlim_m_sum / HII_TOT_NUM_PIXELS);
    averages->m_turn_reion = pow(10, l10_mlim_r_sum / HII_TOT_NUM_PIXELS);

    // mean-fix the grids
    // TODO: put this behind a flag
    if (consts->fix_mean) mean_fix_grids(M_min, M_max, grids, averages, consts);

    // assign the log10 average Mturn for the Ts global tables
    grids->log10_Mcrit_MCG_ave = l10_mlim_m_sum / HII_TOT_NUM_PIXELS;
    grids->log10_Mcrit_ACG_ave = l10_mlim_a_sum / HII_TOT_NUM_PIXELS;

    return 0;
}

void halobox_debug_print_avg(struct HaloProperties *averages_box,
                             struct HaloProperties *averages_subsampler,
                             struct ScalingConstants *consts, double M_min, double M_max) {
    if (LOG_LEVEL < DEBUG_LEVEL) return;
    struct HaloProperties averages_sub_expected, averages_global;
    LOG_DEBUG("HALO BOXES REDSHIFT %.2f [%.2e %.2e]", consts->redshift, M_min, M_max);
    if (matter_options_global->FIXED_HALO_GRIDS) {
        get_box_averages(M_min, M_max, averages_box->m_turn_acg, averages_box->m_turn_mcg, consts,
                         &averages_global);
    } else {
        get_box_averages(simulation_options_global->SAMPLER_MIN_MASS, M_max,
                         averages_box->m_turn_acg, averages_box->m_turn_mcg, consts,
                         &averages_global);
        if (astro_options_global->AVG_BELOW_SAMPLER &&
            M_min < simulation_options_global->SAMPLER_MIN_MASS) {
            get_box_averages(M_min, simulation_options_global->SAMPLER_MIN_MASS,
                             averages_box->m_turn_acg, averages_box->m_turn_mcg, consts,
                             &averages_sub_expected);
        }
    }

    LOG_DEBUG(
        "Exp. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY "
        "%11.3e, NION %11.3e)",
        averages_global.halo_mass, averages_global.stellar_mass, averages_global.stellar_mass_mini,
        averages_global.halo_sfr, averages_global.sfr_mini, averages_global.halo_xray,
        averages_global.n_ion);
    LOG_DEBUG(
        "Box. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY "
        "%11.3e, NION %11.3e)",
        averages_box->halo_mass, averages_box->stellar_mass, averages_box->stellar_mass_mini,
        averages_box->halo_sfr, averages_box->sfr_mini, averages_box->halo_xray,
        averages_box->n_ion);

    if (!matter_options_global->FIXED_HALO_GRIDS && astro_options_global->AVG_BELOW_SAMPLER &&
        M_min < simulation_options_global->SAMPLER_MIN_MASS) {
        LOG_DEBUG("SUB-SAMPLER");
        LOG_DEBUG(
            "Exp. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY "
            "%11.3e, NION %11.3e)",
            averages_sub_expected.halo_mass, averages_sub_expected.stellar_mass,
            averages_sub_expected.stellar_mass_mini, averages_sub_expected.halo_sfr,
            averages_sub_expected.sfr_mini, averages_sub_expected.halo_xray,
            averages_sub_expected.n_ion);
        LOG_DEBUG(
            "Box. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY "
            "%11.3e, NION %11.3e)",
            averages_subsampler->halo_mass, averages_subsampler->stellar_mass,
            averages_subsampler->stellar_mass_mini, averages_subsampler->halo_sfr,
            averages_subsampler->sfr_mini, averages_subsampler->halo_xray,
            averages_subsampler->n_ion);
    }
}

// We need the mean log10 turnover masses for comparison with expected global Nion and SFRD.
// Sometimes we don't calculate these on the grid (if we use halos and no sub-sampler)
// So this function simply returns the volume-weighted average log10 turnover mass
void get_mean_log10_turnovers(InitialConditions *ini_boxes, TsBox *previous_spin_temp,
                              IonizedBox *previous_ionize_box, PerturbedField *perturbed_field,
                              struct ScalingConstants *consts, double turnovers[3]) {
    if (!astro_options_global->USE_MINI_HALOS) {
        turnovers[0] = log10(consts->mturn_a_nofb);  // ACG
        turnovers[1] = log10(consts->mturn_m_nofb);  // MCG
        turnovers[2] = 0.;                           // reion (log10 so effectively 1 solar mass)
        return;
    }
    double l10_mturn_a_avg = 0., l10_mturn_m_avg = 0., l10_mturn_r_avg = 0.;

#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        unsigned long long int i;
        double J21_val, Gamma12_val, zre_val;
        double curr_vcb = consts->vcb_norel;
        double M_turn_m;
        double M_turn_a = consts->mturn_a_nofb;
        double M_turn_r;

#pragma omp for reduction(+ : l10_mturn_m_avg, l10_mturn_a_avg, l10_mturn_r_avg)
        for (i = 0; i < HII_TOT_NUM_PIXELS; i++) {
            if (!astro_options_global->FIX_VCB_AVG &&
                matter_options_global->USE_RELATIVE_VELOCITIES) {
                curr_vcb = ini_boxes->lowres_vcb[i];
            }
            J21_val = Gamma12_val = zre_val = 0.;
            if (consts->redshift < simulation_options_global->Z_HEAT_MAX) {
                J21_val = previous_spin_temp->J_21_LW[i];
                Gamma12_val = previous_ionize_box->ionisation_rate_G12[i];
                zre_val = previous_ionize_box->z_reion[i];
            }
            M_turn_m = lyman_werner_threshold(consts->redshift, J21_val, curr_vcb);
            M_turn_r = reionization_feedback(consts->redshift, Gamma12_val, zre_val);
            M_turn_a = fmax(M_turn_a, fmax(M_turn_r, astro_params_global->M_TURN));
            M_turn_m = fmax(M_turn_m, fmax(M_turn_r, astro_params_global->M_TURN));
            l10_mturn_a_avg += log10(M_turn_a);
            l10_mturn_m_avg += log10(M_turn_m);
            l10_mturn_r_avg += log10(M_turn_r);
        }
        l10_mturn_a_avg /= HII_TOT_NUM_PIXELS;
        l10_mturn_m_avg /= HII_TOT_NUM_PIXELS;
        l10_mturn_r_avg /= HII_TOT_NUM_PIXELS;

        turnovers[0] = l10_mturn_a_avg;
        turnovers[1] = l10_mturn_m_avg;
        turnovers[2] = l10_mturn_r_avg;
    }
}

void sum_halos_onto_grid(InitialConditions *ini_boxes, TsBox *previous_spin_temp,
                         IonizedBox *previous_ionize_box, PerturbHaloField *halos,
                         struct ScalingConstants *consts, HaloBox *grids,
                         struct HaloProperties *averages) {
    double redshift = consts->redshift;
    // averages
    double hm_avg = 0., sm_avg = 0., sfr_avg = 0.;
    double sm_avg_mini = 0., sfr_avg_mini = 0.;
    double M_turn_a_avg = 0., M_turn_m_avg = 0., M_turn_r_avg = 0.;
    double n_ion_avg = 0., wsfr_avg = 0., xray_avg = 0.;
    // counts
    unsigned long long int total_n_halos, n_halos_cut = 0.;

    double cell_volume = VOLUME / HII_TOT_NUM_PIXELS;
    double cell_length = simulation_options_global->BOX_LEN / simulation_options_global->HII_DIM;
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        int x, y, z;
        double halo_pos[3];
        unsigned long long int i_halo, i_cell;
        double hmass, nion, sfr, wsfr, sfr_mini, stars_mini, stars, xray;
        double J21_val, Gamma12_val, zre_val;

        double curr_vcb = consts->vcb_norel;
        double M_turn_m = consts->mturn_m_nofb;
        double M_turn_a = consts->mturn_a_nofb;
        double M_turn_r = 0.;

        double in_props[3];
        struct HaloProperties out_props;

#pragma omp for reduction(+ : hm_avg, sm_avg, sm_avg_mini, sfr_avg, sfr_avg_mini, n_ion_avg, \
                              xray_avg, wsfr_avg, M_turn_a_avg, M_turn_m_avg, M_turn_r_avg,  \
                              n_halos_cut)
        for (i_halo = 0; i_halo < halos->n_halos; i_halo++) {
            hmass = halos->halo_masses[i_halo];
            // It is sometimes useful to make cuts to the halo catalogues before gridding.
            //   We implement this in a simple way, if the user sets a halo's mass to zero we skip
            //   it
            if (hmass == 0.) {
                n_halos_cut++;
                continue;
            }

            for (int i = 0; i < 3; i++) {
                halo_pos[i] = halos->halo_coords[i + 3 * i_halo] / cell_length;
                // This is a special case, where the halo is exactly at the edge of the box
                // This can happen due to floating point errors when multiplied by the cell length
                if (halo_pos[i] == (float)simulation_options_global->HII_DIM) {
                    halo_pos[i] =
                        (float)simulation_options_global->HII_DIM - 0.1;  // will place in last cell
                }
            }

            x = (int)(halo_pos[0]);
            y = (int)(halo_pos[1]);
            z = (int)(halo_pos[2]);
            i_cell = HII_R_INDEX(x, y, z);

            if (i_cell >= HII_TOT_NUM_PIXELS || i_cell < 0 || x < 0 || y < 0 || z < 0 ||
                x >= simulation_options_global->HII_DIM ||
                y >= simulation_options_global->HII_DIM || z >= HII_D_PARA) {
                LOG_ERROR("Halo %llu (%d %d %d) out of bounds (%llu)", i_halo, x, y, z, i_cell);
                LOG_ERROR("Halo Position (%f %f %f) Box Length %.2f",
                          halos->halo_coords[0 + 3 * i_halo], halos->halo_coords[1 + 3 * i_halo],
                          halos->halo_coords[2 + 3 * i_halo], simulation_options_global->BOX_LEN);
                Throw(ValueError);
            }

            // set values before reionisation feedback
            // NOTE: I could easily apply reionization feedback without minihalos but this was not
            // done previously
            if (astro_options_global->USE_MINI_HALOS) {
                if (!astro_options_global->FIX_VCB_AVG &&
                    matter_options_global->USE_RELATIVE_VELOCITIES)
                    curr_vcb = ini_boxes->lowres_vcb[i_cell];

                J21_val = Gamma12_val = zre_val = 0.;
                if (consts->redshift < simulation_options_global->Z_HEAT_MAX) {
                    J21_val = previous_spin_temp->J_21_LW[i_cell];
                    Gamma12_val = previous_ionize_box->ionisation_rate_G12[i_cell];
                    zre_val = previous_ionize_box->z_reion[i_cell];
                }

                M_turn_a = consts->mturn_a_nofb;
                M_turn_m = lyman_werner_threshold(redshift, J21_val, curr_vcb);
                M_turn_r = reionization_feedback(redshift, Gamma12_val, zre_val);
                M_turn_a = fmax(M_turn_a, fmax(M_turn_r, astro_params_global->M_TURN));
                M_turn_m = fmax(M_turn_m, fmax(M_turn_r, astro_params_global->M_TURN));
            }

            // these are the halo property RNG sequences
            in_props[0] = halos->star_rng[i_halo];
            in_props[1] = halos->sfr_rng[i_halo];
            in_props[2] = halos->xray_rng[i_halo];

            set_halo_properties(hmass, M_turn_a, M_turn_m, consts, in_props, &out_props);

            sfr = out_props.halo_sfr;
            sfr_mini = out_props.sfr_mini;
            nion = out_props.n_ion;
            wsfr = out_props.fescweighted_sfr;
            stars = out_props.stellar_mass;
            stars_mini = out_props.stellar_mass_mini;
            xray = out_props.halo_xray;

#if LOG_LEVEL >= ULTRA_DEBUG_LEVEL
            if (x + y + z == 0) {
                // LOG_ULTRA_DEBUG("(%d %d %d) i_cell %llu i_halo %llu",x,y,z,i_cell, i_halo);
                LOG_ULTRA_DEBUG(
                    "Cell 0 Halo: HM: %.2e SM: %.2e (%.2e) SF: %.2e (%.2e) X: %.2e NI: %.2e WS: "
                    "%.2e Z : %.2e ct : %llu",
                    hmass, stars, stars_mini, sfr, sfr_mini, xray, nion, wsfr,
                    out_props.metallicity, i_halo);

                // LOG_ULTRA_DEBUG("Cell 0 Sums: HM: %.2e SM: %.2e (%.2e) SF: %.2e (%.2e) X: %.2e
                // NI: %.2e WS: %.2e ct : %d",
                //         grids->halo_mass[HII_R_INDEX(0,0,0)],
                //         grids->halo_stars[HII_R_INDEX(0,0,0)],grids->halo_stars_mini[HII_R_INDEX(0,0,0)],
                //         grids->halo_sfr[HII_R_INDEX(0,0,0)],grids->halo_sfr_mini[HII_R_INDEX(0,0,0)],
                //         grids->halo_xray[HII_R_INDEX(0,0,0)],grids->n_ion[HII_R_INDEX(0,0,0)],
                //         grids->whalo_sfr[HII_R_INDEX(0,0,0)],grids->count[HII_R_INDEX(0,0,0)]);
                LOG_ULTRA_DEBUG("Mturn_a %.2e Mturn_m %.2e RNG %.3f %.3f %.3f", M_turn_a, M_turn_m,
                                in_props[0], in_props[1], in_props[2]);
            }
#endif

// update the grids
#pragma omp atomic update
            grids->halo_mass[i_cell] += hmass;
#pragma omp atomic update
            grids->halo_stars[i_cell] += stars;
#pragma omp atomic update
            grids->n_ion[i_cell] += nion;
#pragma omp atomic update
            grids->halo_sfr[i_cell] += sfr;
#pragma omp atomic update
            grids->count[i_cell] += 1;

            if (astro_options_global->USE_MINI_HALOS) {
#pragma omp atomic update
                grids->halo_stars_mini[i_cell] += stars_mini;
#pragma omp atomic update
                grids->halo_sfr_mini[i_cell] += sfr_mini;
            }

            if (astro_options_global->INHOMO_RECO) {
#pragma omp atomic update
                grids->whalo_sfr[i_cell] += wsfr;
            }

            if (astro_options_global->USE_TS_FLUCT) {
#pragma omp atomic update
                grids->halo_xray[i_cell] += xray;
            }

            hm_avg += hmass;
            sfr_avg += sfr;
            sfr_avg_mini += sfr_mini;
            sm_avg += stars;
            sm_avg_mini += stars_mini;
            xray_avg += xray;
            n_ion_avg += nion;
            wsfr_avg += wsfr;
            M_turn_a_avg += M_turn_a;
            M_turn_r_avg += M_turn_r;
            M_turn_m_avg += M_turn_m;
        }

#pragma omp for
        for (i_cell = 0; i_cell < HII_TOT_NUM_PIXELS; i_cell++) {
            grids->halo_mass[i_cell] /= cell_volume;
            grids->halo_sfr[i_cell] /= cell_volume;
            grids->halo_stars[i_cell] /= cell_volume;
            grids->n_ion[i_cell] /= cell_volume;
            if (astro_options_global->USE_TS_FLUCT) {
                grids->halo_xray[i_cell] /= cell_volume;
            }
            if (astro_options_global->INHOMO_RECO) {
                grids->whalo_sfr[i_cell] /= cell_volume;
            }
            if (astro_options_global->USE_MINI_HALOS) {
                grids->halo_sfr_mini[i_cell] /= cell_volume;
                grids->halo_stars_mini[i_cell] /= cell_volume;
            }
        }
    }
    total_n_halos = halos->n_halos - n_halos_cut;
    LOG_SUPER_DEBUG("Cell 0 Totals: HM: %.2e SM: %.2e SF: %.2e NI: %.2e ct : %d",
                    grids->halo_mass[HII_R_INDEX(0, 0, 0)], grids->halo_stars[HII_R_INDEX(0, 0, 0)],
                    grids->halo_sfr[HII_R_INDEX(0, 0, 0)], grids->halo_xray[HII_R_INDEX(0, 0, 0)],
                    grids->n_ion[HII_R_INDEX(0, 0, 0)], grids->count[HII_R_INDEX(0, 0, 0)]);
    if (astro_options_global->INHOMO_RECO) {
        LOG_SUPER_DEBUG("FESC * SF %.2e", grids->whalo_sfr[HII_R_INDEX(0, 0, 0)]);
    }
    if (astro_options_global->USE_TS_FLUCT) {
        LOG_SUPER_DEBUG("X-ray %.2e", grids->halo_xray[HII_R_INDEX(0, 0, 0)]);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        LOG_SUPER_DEBUG("MINI SM %.2e SF %.2e", grids->halo_stars_mini[HII_R_INDEX(0, 0, 0)],
                        grids->halo_sfr_mini[HII_R_INDEX(0, 0, 0)]);
    }

    // NOTE: There is an inconsistency here, the sampled grids use a halo-averaged turnover mass
    //   whereas the fixed grids / default 21cmfast uses the volume averaged LOG10(turnover mass).
    //   Neither of these are a perfect representation due to the nonlinear way turnover mass
    //   affects N_ion
    if (total_n_halos > 0) {
        M_turn_r_avg /= total_n_halos;
        M_turn_a_avg /= total_n_halos;
        M_turn_m_avg /= total_n_halos;
    } else {
        // If we have no halos, assume the turnover has no reion feedback & no LW
        M_turn_m_avg = consts->mturn_m_nofb;
        M_turn_a_avg = consts->mturn_a_nofb;
        M_turn_r_avg = 0.;
    }

    hm_avg /= VOLUME;
    sm_avg /= VOLUME;
    sm_avg_mini /= VOLUME;
    sfr_avg /= VOLUME;
    sfr_avg_mini /= VOLUME;
    n_ion_avg /= VOLUME;
    xray_avg /= VOLUME;

    averages->halo_mass = hm_avg;
    averages->stellar_mass = sm_avg;
    averages->halo_sfr = sfr_avg;
    averages->stellar_mass_mini = sm_avg_mini;
    averages->sfr_mini = sfr_avg_mini;
    averages->halo_xray = xray_avg;
    averages->n_ion = n_ion_avg;
    averages->m_turn_acg = M_turn_a_avg;
    averages->m_turn_mcg = M_turn_m_avg;
    averages->m_turn_reion = M_turn_r_avg;
}

// We grid a PERTURBED halofield into the necessary quantities for calculating radiative backgrounds
int ComputeHaloBox(double redshift, InitialConditions *ini_boxes, PerturbedField *perturbed_field,
                   PerturbHaloField *halos, TsBox *previous_spin_temp,
                   IonizedBox *previous_ionize_box, HaloBox *grids) {
    int status;
    Try {
        // get parameters

#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeCosmoParams(cosmo_params_global);
        writeAstroParams(astro_params_global);
        writeAstroOptions(astro_options_global);
#endif

        unsigned long long int idx;
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS) private(idx)
        for (idx = 0; idx < HII_TOT_NUM_PIXELS; idx++) {
            grids->halo_mass[idx] = 0.0;
            grids->n_ion[idx] = 0.0;
            grids->halo_sfr[idx] = 0.0;
            grids->halo_stars[idx] = 0.0;
            grids->count[idx] = 0;
            if (astro_options_global->USE_TS_FLUCT) {
                grids->halo_xray[idx] = 0.0;
            }
            if (astro_options_global->USE_MINI_HALOS) {
                grids->halo_stars_mini[idx] = 0.0;
                grids->halo_sfr_mini[idx] = 0.0;
            }
            if (astro_options_global->INHOMO_RECO) {
                grids->whalo_sfr[idx] = 0.0;
            }
        }

        struct ScalingConstants hbox_consts;

        set_scaling_constants(redshift, &hbox_consts, true);

        LOG_DEBUG("Gridding %llu halos...", halos->n_halos);

        double M_min = minimum_source_mass(redshift, false);
        double M_max_integral;
        double cell_volume = VOLUME / HII_TOT_NUM_PIXELS;

        double turnovers[3];

        struct HaloProperties averages_box, averages_subsampler;

        init_ps();
        if (matter_options_global->USE_INTERPOLATION_TABLES > 0) {
            initialiseSigmaMInterpTable(
                M_min / 2,
                M_MAX_INTEGRAL);  // this needs to be initialised above MMax because of Nion_General
        }
        // do the mean HMF box
        // The default 21cmFAST has a strange behaviour where the nonlinear density is used as
        // linear, the condition mass is at mean density, but the total cell mass is multiplied by
        // delta This part mimics that behaviour Since we need the average turnover masses before we
        // can calculate the global means, we do the CMF integrals first Then we calculate the
        // expected UMF integrals before doing the adjustment
        if (matter_options_global->FIXED_HALO_GRIDS) {
            M_max_integral = M_MAX_INTEGRAL;
            set_fixed_grids(M_min, M_max_integral, ini_boxes, perturbed_field, previous_spin_temp,
                            previous_ionize_box, &hbox_consts, grids, &averages_box, true);
        } else {
            // set below-resolution properties
            if (astro_options_global->AVG_BELOW_SAMPLER) {
                if (matter_options_global->HALO_STOCHASTICITY) {
                    M_max_integral = simulation_options_global->SAMPLER_MIN_MASS;
                } else {
                    M_max_integral = RtoM(L_FACTOR * simulation_options_global->BOX_LEN /
                                          simulation_options_global->DIM);
                }
                if (M_min < M_max_integral) {
                    set_fixed_grids(M_min, M_max_integral, ini_boxes, perturbed_field,
                                    previous_spin_temp, previous_ionize_box, &hbox_consts, grids,
                                    &averages_subsampler, false);
// This is pretty redundant, but since the fixed grids have density units (X Mpc-3) I have to
// re-multiply before adding the halos.
//       I should instead have a flag to output the summed values in cell. (2*N_pixel > N_halo so
//       generally i don't want to do it in the halo loop)
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS) private(idx)
                    for (idx = 0; idx < HII_TOT_NUM_PIXELS; idx++) {
                        grids->halo_mass[idx] *= cell_volume;
                        grids->halo_stars[idx] *= cell_volume;
                        grids->n_ion[idx] *= cell_volume;
                        grids->halo_sfr[idx] *= cell_volume;
                        if (astro_options_global->USE_TS_FLUCT) {
                            grids->halo_xray[idx] *= cell_volume;
                        }
                        if (astro_options_global->INHOMO_RECO) {
                            grids->whalo_sfr[idx] *= cell_volume;
                        }
                        if (astro_options_global->USE_MINI_HALOS) {
                            grids->halo_stars_mini[idx] *= cell_volume;
                            grids->halo_sfr_mini[idx] *= cell_volume;
                        }
                    }
                    LOG_DEBUG("finished subsampler M[%.2e %.2e]", M_min, M_max_integral);
                }
            } else {
                // we still need the average turnovers for global values in spintemp, so get them
                // here
                get_mean_log10_turnovers(ini_boxes, previous_spin_temp, previous_ionize_box,
                                         perturbed_field, &hbox_consts, turnovers);
                grids->log10_Mcrit_ACG_ave = turnovers[0];
                grids->log10_Mcrit_MCG_ave = turnovers[1];
            }
            sum_halos_onto_grid(ini_boxes, previous_spin_temp, previous_ionize_box, halos,
                                &hbox_consts, grids, &averages_box);
        }
        halobox_debug_print_avg(&averages_box, &averages_subsampler, &hbox_consts, M_min,
                                M_MAX_INTEGRAL);

        // NOTE: the density-grid based calculations (!USE_HALO_FIELD)
        //  use the cell-weighted average of the log10(Mturn) (see issue #369)
        LOG_SUPER_DEBUG("log10 Mutrn ACG: log10 cell-weighted %.6e Halo-weighted %.6e",
                        pow(10, grids->log10_Mcrit_ACG_ave), averages_box.m_turn_acg);
        LOG_SUPER_DEBUG("log10 Mutrn MCG: log10 cell-weighted %.6e Halo-weighted %.6e",
                        pow(10, grids->log10_Mcrit_MCG_ave), averages_box.m_turn_mcg);

        if (matter_options_global->USE_INTERPOLATION_TABLES > 0) {
            freeSigmaMInterpTable();
        }
    }
    Catch(status) { return (status); }
    LOG_DEBUG("Done.");
    return 0;
}

// test function for getting halo properties from the wrapper, can use a lot of memory for large
// catalogs
int test_halo_props(double redshift, float *vcb_grid, float *J21_LW_grid, float *z_re_grid,
                    float *Gamma12_ion_grid, int n_halos, float *halo_masses, float *halo_coords,
                    float *star_rng, float *sfr_rng, float *xray_rng, float *halo_props_out) {
    int status;
    Try {
        // get parameters

        struct ScalingConstants hbox_consts;
        set_scaling_constants(redshift, &hbox_consts, true);
        print_sc_consts(&hbox_consts);

        LOG_DEBUG("Getting props for %llu halos at z=%.2f", n_halos, redshift);

        double cell_length =
            simulation_options_global->BOX_LEN / simulation_options_global->HII_DIM;

#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
            int x, y, z;
            unsigned long long int i_halo, i_cell;
            double m;
            double J21_val, Gamma12_val, zre_val;

            double curr_vcb = hbox_consts.vcb_norel;
            double M_turn_m = hbox_consts.mturn_m_nofb;
            double M_turn_a = hbox_consts.mturn_a_nofb;
            double M_turn_r = 0.;

            double in_props[3], halo_pos[3];
            struct HaloProperties out_props;

#pragma omp for
            for (i_halo = 0; i_halo < n_halos; i_halo++) {
                m = halo_masses[i_halo];
                // It is sometimes useful to make cuts to the halo catalogues before gridding.
                //   We implement this in a simple way, if the user sets a halo's mass to zero we
                //   skip it
                if (m == 0.) {
                    continue;
                }

                for (int i = 0; i < 3; i++) {
                    halo_pos[i] = halo_coords[i + 3 * i_halo] / cell_length;
                    // This is a special case, where the halo is exactly at the edge of the box
                    // This can happen due to floating point errors when multiplied by the cell
                    // length
                    if (halo_pos[i] == (float)simulation_options_global->HII_DIM) {
                        halo_pos[i] = (float)simulation_options_global->HII_DIM -
                                      0.1;  // will place in last cell
                    }
                }

                x = (int)(halo_pos[0]);
                y = (int)(halo_pos[1]);
                z = (int)(halo_pos[2]);
                i_cell = HII_R_INDEX(x, y, z);

                // set values before reionisation feedback
                // NOTE: I could easily apply reionization feedback without minihalos but this was
                // not done previously
                if (astro_options_global->USE_MINI_HALOS) {
                    if (!astro_options_global->FIX_VCB_AVG &&
                        matter_options_global->USE_RELATIVE_VELOCITIES)
                        curr_vcb = vcb_grid[i_cell];

                    J21_val = Gamma12_val = zre_val = 0.;
                    if (redshift < simulation_options_global->Z_HEAT_MAX) {
                        J21_val = J21_LW_grid[i_cell];
                        Gamma12_val = Gamma12_ion_grid[i_cell];
                        zre_val = z_re_grid[i_cell];
                    }
                    M_turn_a = hbox_consts.mturn_a_nofb;
                    M_turn_m = lyman_werner_threshold(redshift, J21_val, curr_vcb);
                    M_turn_r = reionization_feedback(redshift, Gamma12_val, zre_val);
                    M_turn_a = fmax(M_turn_a, fmax(M_turn_r, astro_params_global->M_TURN));
                    M_turn_m = fmax(M_turn_m, fmax(M_turn_r, astro_params_global->M_TURN));
                }

                // these are the halo property RNG sequences
                in_props[0] = star_rng[i_halo];
                in_props[1] = sfr_rng[i_halo];
                in_props[2] = xray_rng[i_halo];

                set_halo_properties(m, M_turn_a, M_turn_m, &hbox_consts, in_props, &out_props);

                halo_props_out[12 * i_halo + 0] = out_props.halo_mass;
                halo_props_out[12 * i_halo + 1] = out_props.stellar_mass;
                halo_props_out[12 * i_halo + 2] = out_props.halo_sfr;

                halo_props_out[12 * i_halo + 3] = out_props.halo_xray;
                halo_props_out[12 * i_halo + 4] = out_props.n_ion;
                halo_props_out[12 * i_halo + 5] = out_props.fescweighted_sfr;

                halo_props_out[12 * i_halo + 6] = out_props.stellar_mass_mini;
                halo_props_out[12 * i_halo + 7] = out_props.sfr_mini;

                halo_props_out[12 * i_halo + 8] = M_turn_a;
                halo_props_out[12 * i_halo + 9] = M_turn_m;
                halo_props_out[12 * i_halo + 10] = M_turn_r;
                halo_props_out[12 * i_halo + 11] = out_props.metallicity;

                if (i_halo < 10) {
                    LOG_ULTRA_DEBUG("HM %.2e SM %.2e SF %.2e NI %.2e LX %.2e", out_props.halo_mass,
                                    out_props.stellar_mass, out_props.halo_sfr, out_props.n_ion,
                                    out_props.halo_xray);
                    LOG_ULTRA_DEBUG("MINI: SM %.2e SF %.2e WSF %.2e", out_props.stellar_mass_mini,
                                    out_props.sfr_mini, out_props.fescweighted_sfr);
                    LOG_ULTRA_DEBUG("Mturns ACG %.2e MCG %.2e Reion %.2e", M_turn_a, M_turn_m,
                                    M_turn_r);
                    LOG_ULTRA_DEBUG("RNG: STAR %.2e SFR %.2e XRAY %.2e", in_props[0], in_props[1],
                                    in_props[2]);
                }
            }
        }
    }
    Catch(status) { return (status); }
    LOG_DEBUG("Done.");
    return 0;
}
