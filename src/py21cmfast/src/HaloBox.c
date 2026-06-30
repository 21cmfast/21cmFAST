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
#include "map_mass.h"
#include "scaling_relations.h"
#include "thermochem.h"

// TODO: this should probably be somewhere else
void set_integral_constants(IntegralCondition *consts, double redshift, double M_min, double M_max,
                            double M_cell) {
    consts->redshift = redshift;
    consts->growth_factor = dicke(redshift);
    consts->M_min = M_min;
    consts->M_max = M_max;
    consts->lnM_min = log(M_min);
    consts->lnM_max = log(M_max);
    consts->M_cell = M_cell;
    consts->lnM_cell = log(M_cell);
    if (simulation_options_global->HII_DIM == 1 && simulation_options_global->BOX_LEN > 1e5) {
        // When simulating only the global signal, the box/cell size should be infinite, so the
        // conditional sigma is 0
        consts->sigma_cell = 0.;
    } else {
        // no table since this should be called once
        consts->sigma_cell = sigma_z0(M_cell);
    }
}

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
                         ScalingConstants *consts, double *input_rng, HaloProperties *output) {
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
    fesc = fmin(consts->fesc_10 * pow(halo_mass / 1e10, consts->alpha_esc), 1);
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
int get_uhmf_averages(double M_min, double M_max, double M_turn_a, double M_turn_m,
                      ScalingConstants *consts, HaloProperties *averages_out) {
    LOG_SUPER_DEBUG("Getting Box averages z=%.2f M [%.2e %.2e] Mt [%.2e %.2e]", consts->redshift,
                    M_min, M_max, M_turn_a, M_turn_m);
    double t_h = consts->t_h;
    double lnMmax = log(M_max);
    double lnMmin = log(M_min);

    double prefactor_mass = RHOcrit * cosmo_params_global->OMm;
    double prefactor_stars = RHOcrit * cosmo_params_global->OMb * consts->fstar_10;
    double prefactor_stars_mini = RHOcrit * cosmo_params_global->OMb * consts->fstar_7;
    double prefactor_xray = RHOcrit * cosmo_params_global->OMm;

    double prefactor_sfr = prefactor_stars / consts->t_star / t_h;
    double prefactor_sfr_mini = prefactor_stars_mini / consts->t_star / t_h;
    double prefactor_nion = prefactor_stars * consts->fesc_10 * consts->pop2_ion;
    double prefactor_nion_mini = prefactor_stars_mini * consts->fesc_7 * consts->pop3_ion;
    double prefactor_wsfr = prefactor_sfr * consts->fesc_10 * consts->pop2_ion;
    double prefactor_wsfr_mini = prefactor_sfr_mini * consts->fesc_7 * consts->pop3_ion;

    double mass_intgrl;
    double intgrl_fesc_weighted, intgrl_stars_only;
    double intgrl_fesc_weighted_mini = 0., intgrl_stars_only_mini = 0., integral_xray = 0.;

    // NOTE: we use the atomic method for all halo mass/count here
    mass_intgrl = Fcoll_General(consts->redshift, lnMmin, lnMmax);
    ScalingConstants consts_sfrd = evolve_scaling_constants_sfr(consts);

    intgrl_fesc_weighted = Nion_General(consts->redshift, lnMmin, lnMmax, M_turn_a, consts);
    intgrl_stars_only = Nion_General(consts->redshift, lnMmin, lnMmax, M_turn_a, &consts_sfrd);
    if (astro_options_global->USE_MINI_HALOS) {
        intgrl_fesc_weighted_mini =
            Nion_General_MINI(consts->redshift, lnMmin, lnMmax, M_turn_a, M_turn_m, consts);

        intgrl_stars_only_mini =
            Nion_General_MINI(consts->redshift, lnMmin, lnMmax, M_turn_a, M_turn_m, &consts_sfrd);
    }
    if (astro_options_global->USE_TS_FLUCT) {
        integral_xray = Xray_General(consts->redshift, lnMmin, lnMmax, M_turn_a, M_turn_m, consts);
    }

    averages_out->count = Nhalo_General(consts->redshift, lnMmin, lnMmax) * prefactor_mass *
                          VOLUME / HII_TOT_NUM_PIXELS;
    averages_out->halo_mass = mass_intgrl * prefactor_mass;
    averages_out->stellar_mass = intgrl_stars_only * prefactor_stars;
    averages_out->halo_sfr = intgrl_stars_only * prefactor_sfr;
    averages_out->stellar_mass_mini = intgrl_stars_only_mini * prefactor_stars_mini;
    averages_out->sfr_mini = intgrl_stars_only_mini * prefactor_sfr_mini;
    averages_out->n_ion =
        (intgrl_fesc_weighted * prefactor_nion) + (intgrl_fesc_weighted_mini * prefactor_nion_mini);
    averages_out->fescweighted_sfr =
        (intgrl_fesc_weighted * prefactor_wsfr) + (intgrl_fesc_weighted_mini * prefactor_wsfr_mini);
    averages_out->halo_xray = prefactor_xray * integral_xray;
    averages_out->m_turn_acg = M_turn_a;
    averages_out->m_turn_mcg = M_turn_m;

    return 0;
}
HaloProperties get_halobox_averages(HaloBox *grids) {
    double mean_count = 0.;
    double mean_mass = 0., mean_stars = 0., mean_stars_mini = 0., mean_sfr = 0., mean_sfr_mini = 0.;
    double mean_n_ion = 0., mean_xray = 0., mean_wsfr = 0.;

#pragma omp parallel for reduction(+ : mean_count, mean_mass, mean_stars, mean_stars_mini, \
                                       mean_sfr, mean_sfr_mini, mean_n_ion, mean_xray, mean_wsfr)
    for (index_huge i = 0; i < HII_TOT_NUM_PIXELS; i++) {
        mean_sfr += grids->halo_sfr[i];
        mean_n_ion += grids->n_ion[i];
        if (astro_options_global->USE_TS_FLUCT) {
            mean_xray += grids->halo_xray[i];
        }
        if (astro_options_global->USE_MINI_HALOS) {
            mean_sfr_mini += grids->halo_sfr_mini[i];
        }
        if (uses_recombination(astro_options_global->RECOMB_MODEL))
            mean_wsfr += grids->whalo_sfr[i];

        if (config_settings.EXTRA_HALOBOX_FIELDS) {
            mean_count += grids->count[i];
            mean_mass += grids->halo_mass[i];
            mean_stars += grids->halo_stars[i];
            if (astro_options_global->USE_MINI_HALOS) mean_stars_mini += grids->halo_stars_mini[i];
        }
    }

    HaloProperties averages = {
        .count = mean_count / HII_TOT_NUM_PIXELS,
        .halo_mass = mean_mass / HII_TOT_NUM_PIXELS,
        .stellar_mass = mean_stars / HII_TOT_NUM_PIXELS,
        .stellar_mass_mini = mean_stars_mini / HII_TOT_NUM_PIXELS,
        .halo_sfr = mean_sfr / HII_TOT_NUM_PIXELS,
        .sfr_mini = mean_sfr_mini / HII_TOT_NUM_PIXELS,
        .n_ion = mean_n_ion / HII_TOT_NUM_PIXELS,
        .halo_xray = mean_xray / HII_TOT_NUM_PIXELS,
        .fescweighted_sfr = mean_wsfr / HII_TOT_NUM_PIXELS,
    };
    return averages;
}

// This takes a HaloBox struct and fixes it's mean to exactly what we expect from the UMF integrals.
//   Generally should only be done for the fixed portion of the grids, since
//   it will otherwise make the box inconsistent with the input catalogue
void mean_fix_grids(double M_min, double M_max, HaloBox *grids, ScalingConstants *consts) {
    HaloProperties averages_global;
    // NOTE: requires the mean mcrits to be set on the grids
    double M_turn_a_global = pow(10, grids->log10_Mcrit_ACG_ave);
    double M_turn_m_global = pow(10, grids->log10_Mcrit_MCG_ave);
    get_uhmf_averages(M_min, M_max, M_turn_a_global, M_turn_m_global, consts, &averages_global);
    HaloProperties averages_hbox;
    averages_hbox = get_halobox_averages(grids);

    index_huge idx;
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS) private(idx)
    for (idx = 0; idx < HII_TOT_NUM_PIXELS; idx++) {
        grids->halo_sfr[idx] *= averages_global.halo_sfr / averages_hbox.halo_sfr;
        grids->n_ion[idx] *= averages_global.n_ion / averages_hbox.n_ion;
        if (astro_options_global->USE_MINI_HALOS) {
            grids->halo_sfr_mini[idx] *= averages_global.sfr_mini / averages_hbox.sfr_mini;
        }
        if (astro_options_global->USE_TS_FLUCT) {
            grids->halo_xray[idx] *= averages_global.halo_xray / averages_hbox.halo_xray;
        }
        if (uses_recombination(astro_options_global->RECOMB_MODEL)) {
            grids->whalo_sfr[idx] *=
                averages_global.fescweighted_sfr / averages_hbox.fescweighted_sfr;
        }

        if (config_settings.EXTRA_HALOBOX_FIELDS) {
            grids->count[idx] *= averages_global.count / averages_hbox.count;
            grids->halo_mass[idx] *= averages_global.halo_mass / averages_hbox.halo_mass;
            grids->halo_stars[idx] *= averages_global.stellar_mass / averages_hbox.stellar_mass;
            if (astro_options_global->USE_MINI_HALOS) {
                grids->halo_stars_mini[idx] *=
                    averages_global.stellar_mass_mini / averages_hbox.stellar_mass_mini;
            }
        }
    }
}

// Evaluate Mass function integrals given information from the cell
void get_cell_integrals(double dens, double l10_mturn_a, double l10_mturn_m,
                        ScalingConstants *consts, IntegralCondition *int_consts,
                        HaloProperties *properties) {
    double M_min = int_consts->M_min;
    double M_max = int_consts->M_max;
    double growth_z = int_consts->growth_factor;
    double M_cell = int_consts->M_cell;
    double sigma_cell = int_consts->sigma_cell;

    // set all fields to zero
    memset(properties, 0, sizeof(HaloProperties));

    // using the properties struct:
    // stellar_mass --> no F_esc integral ACG
    // stellar_mass_mini --> no F_esc integral MCG
    // n_ion --> F_esc integral ACG
    // fescweighted_sfr --> F_esc integral MCG
    // halo_xray --> Xray integral
    // halo_mass --> total mass
    properties->n_ion = EvaluateNion_Conditional(dens, l10_mturn_a, growth_z, M_min, M_max, M_cell,
                                                 sigma_cell, consts, false);
    properties->stellar_mass = EvaluateSFRD_Conditional(dens, l10_mturn_a, growth_z, M_min, M_max,
                                                        M_cell, sigma_cell, consts);
    if (astro_options_global->USE_MINI_HALOS) {
        properties->stellar_mass_mini = EvaluateSFRD_Conditional_MINI(
            dens, l10_mturn_a, l10_mturn_m, growth_z, M_min, M_max, M_cell, sigma_cell, consts);
        // re-using field
        properties->fescweighted_sfr =
            EvaluateNion_Conditional_MINI(dens, l10_mturn_a, l10_mturn_m, growth_z, M_min, M_max,
                                          M_cell, sigma_cell, consts, false);
    }

    if (astro_options_global->USE_TS_FLUCT) {
        properties->halo_xray =
            EvaluateXray_Conditional(dens, l10_mturn_a, l10_mturn_m, consts->redshift, growth_z,
                                     M_min, M_max, M_cell, sigma_cell, consts);
    }

    if (config_settings.EXTRA_HALOBOX_FIELDS) {
        properties->count =
            EvaluateNhalo(dens, growth_z, log(M_min), log(M_max), M_cell, sigma_cell, dens) *
            M_cell;
        properties->halo_mass =
            EvaluateMcoll(dens, growth_z, log(M_min), log(M_max), M_cell, sigma_cell, dens);
    }
}

// Fixed halo grids, where each property is set as the integral of the CMF on the LAGRANGIAN cell,
// and then the properties are moved to the EULERIAN grid according to the velocities.
// This outputs the UN-NORMALISED grids (before mean-adjustment)
int set_fixed_grids(double M_min, double M_max, InitialConditions *ini_boxes,
                    float *log10_mturn_a_grid, float *log10_mturn_m_grid, ScalingConstants *consts,
                    HaloBox *grids) {
    double M_cell;
    // If our scaling relations define a median, the scatter will will increase the mean value
    // due to the asymmetry of the lognormal distribution, we mimic this in the
    // sub-sampler component.
    ScalingConstants _ev_consts = *consts;
    ScalingConstants *ev_consts = &_ev_consts;

    if (astro_options_global->HALO_SCALING_RELATIONS_MEDIAN) {
        _ev_consts = mimic_scatter_in_consts(consts);
    }
    double growthf = dicke(ev_consts->redshift);

    // find grid limits for tables
    double min_density = 0.;
    double max_density = 0.;

    float *vel_pointers[3];
    float *vel_pointers_2LPT[3];
    int grid_dim[3];
    size_huge num_pixels;
    float *dens_pointer;
    int out_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};  // always output to lowres grid
    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        grid_dim[0] = simulation_options_global->DIM;
        grid_dim[1] = simulation_options_global->DIM;
        grid_dim[2] = D_PARA;
        vel_pointers[0] = ini_boxes->hires_vx;
        vel_pointers[1] = ini_boxes->hires_vy;
        vel_pointers[2] = ini_boxes->hires_vz;
        vel_pointers_2LPT[0] = ini_boxes->hires_vx_2LPT;
        vel_pointers_2LPT[1] = ini_boxes->hires_vy_2LPT;
        vel_pointers_2LPT[2] = ini_boxes->hires_vz_2LPT;
        dens_pointer = ini_boxes->hires_density;
        num_pixels = TOT_NUM_PIXELS;
        M_cell = RHOcrit * cosmo_params_global->OMm * VOLUME / TOT_NUM_PIXELS;
    } else {
        grid_dim[0] = simulation_options_global->HII_DIM;
        grid_dim[1] = simulation_options_global->HII_DIM;
        grid_dim[2] = HII_D_PARA;
        vel_pointers[0] = ini_boxes->lowres_vx;
        vel_pointers[1] = ini_boxes->lowres_vy;
        vel_pointers[2] = ini_boxes->lowres_vz;
        vel_pointers_2LPT[0] = ini_boxes->lowres_vx_2LPT;
        vel_pointers_2LPT[1] = ini_boxes->lowres_vy_2LPT;
        vel_pointers_2LPT[2] = ini_boxes->lowres_vz_2LPT;
        dens_pointer = ini_boxes->lowres_density;
        num_pixels = HII_TOT_NUM_PIXELS;
        M_cell = RHOcrit * cosmo_params_global->OMm * VOLUME / HII_TOT_NUM_PIXELS;
    }

    IntegralCondition integral_cond;
    set_integral_constants(&integral_cond, ev_consts->redshift, M_min, M_max, M_cell);
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        index_huge i;
        double dens;
#pragma omp for reduction(min : min_density) reduction(max : max_density)
        for (i = 0; i < num_pixels; i++) {
            dens = dens_pointer[i] * growthf;
            if (dens > max_density) max_density = dens;
            if (dens < min_density) min_density = dens;
        }
    }
    // buffers for table ranges
    min_density = min_density * 1.001;  // negative
    max_density = max_density * 1.001;

    LOG_DEBUG("Mean halo boxes || M = [%.2e %.2e] | Mcell = %.2e", M_min, M_max, M_cell);
    // These tables are coarser than needed, an initial loop for Mturn to find limits may help
    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        if (astro_options_global->INTEGRATION_METHOD_ATOMIC == INTEGRATION_METHOD_GAUSS_LEGENDRE ||
            (astro_options_global->USE_MINI_HALOS &&
             astro_options_global->INTEGRATION_METHOD_MINI == INTEGRATION_METHOD_GAUSS_LEGENDRE)) {
            initialise_GL(integral_cond.lnM_min, integral_cond.lnM_max);
        }
        // This table assumes no reionisation feedback
        initialise_SFRD_Conditional_table(ev_consts->redshift, min_density, max_density, M_min,
                                          M_max, M_cell, ev_consts);

        // This table includes reionisation feedback
        initialise_Nion_Conditional_spline(ev_consts->redshift, min_density, max_density, M_min,
                                           M_max, M_cell, ev_consts, false);

        initialise_dNdM_tables(min_density, max_density, integral_cond.lnM_min,
                               integral_cond.lnM_max, integral_cond.growth_factor,
                               integral_cond.lnM_cell, false);
        if (astro_options_global->USE_TS_FLUCT) {
            initialise_Xray_Conditional_table(ev_consts->redshift, min_density, max_density, M_min,
                                              M_max, M_cell, ev_consts);
        }
    }
    move_grid_galprops(ev_consts->redshift, dens_pointer, grid_dim, vel_pointers, vel_pointers_2LPT,
                       grid_dim, grids, out_dim, log10_mturn_a_grid, log10_mturn_m_grid, ev_consts,
                       &integral_cond);

    LOG_ULTRA_DEBUG("Cell 0 Totals: SF: %.2e, NI: %.2e", grids->halo_sfr[0], grids->n_ion[0]);
    if (uses_recombination(astro_options_global->RECOMB_MODEL)) {
        LOG_ULTRA_DEBUG("FESC * SF %.2e", grids->whalo_sfr[0]);
    }
    if (astro_options_global->USE_TS_FLUCT) {
        LOG_ULTRA_DEBUG("X-ray %.2e", grids->halo_xray[0]);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        LOG_ULTRA_DEBUG("MINI SM %.2e SF %.2e", grids->halo_stars_mini[0], grids->halo_sfr_mini[0]);
        LOG_ULTRA_DEBUG("Mturn_a %.2e Mturn_m %.2e", mturn_a_grid[0], mturn_m_grid[0]);
    }
    free_conditional_tables();

    if (ev_consts->fix_mean) mean_fix_grids(M_min, M_max, grids, ev_consts);

    return 0;
}

void halobox_debug_print_avg(HaloBox *halobox, ScalingConstants *consts, double M_min,
                             double M_max) {
    if (LOG_LEVEL < DEBUG_LEVEL) return;
    HaloProperties averages_box;
    averages_box = get_halobox_averages(halobox);
    HaloProperties averages_global;
    LOG_DEBUG("HALO BOXES REDSHIFT %.2f [%.2e %.2e]", consts->redshift, M_min, M_max);
    double mturn_a_avg = pow(10, halobox->log10_Mcrit_ACG_ave);
    double mturn_m_avg = pow(10, halobox->log10_Mcrit_MCG_ave);
    get_uhmf_averages(M_min, M_max, mturn_a_avg, mturn_m_avg, consts, &averages_global);

    LOG_DEBUG(
        "Exp. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY "
        "%11.3e, NION %11.3e)",
        averages_global.halo_mass, averages_global.stellar_mass, averages_global.stellar_mass_mini,
        averages_global.halo_sfr, averages_global.sfr_mini, averages_global.halo_xray,
        averages_global.n_ion);
    LOG_DEBUG(
        "Box. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY "
        "%11.3e, NION %11.3e)",
        averages_box.halo_mass, averages_box.stellar_mass, averages_box.stellar_mass_mini,
        averages_box.halo_sfr, averages_box.sfr_mini, averages_box.halo_xray, averages_box.n_ion);
}

// We need the mean log10 turnover masses for comparison with expected global Nion and SFRD.
// Sometimes we don't calculate these on the grid (if we use halos and no sub-sampler)
// So this function simply returns the volume-weighted average log10 turnover mass
void get_log10_turnovers(InitialConditions *ini_boxes, TsBox *previous_spin_temp,
                         IonizedBox *previous_ionize_box, float *log10_mturn_a_grid,
                         float *log10_mturn_m_grid, ScalingConstants *consts, double averages[2]) {
    double log10_mturn_m_avg = 0., log10_mturn_a_avg = 0.;
    // If we either use mini-halos or at least the reionization feedback model is applied on the ACG
    // turnover mass, we need to compute the local fluctuating turnover mass at every cell. The mean
    // of the log10 of these turnover mass fields is then computed from averaging over the box
    if (astro_options_global->USE_MINI_HALOS ||
        uses_reionization_feedback_in_acgs(astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
            index_huge i;
            double J21_val = 0., Gamma12_val = 0., zre_val = 0.;
            double curr_vcb = consts->vcb_const;
            double M_turn_a = consts->mturn_a_nofb;
            double M_turn_m;
            double M_turn_r;

#pragma omp for reduction(+ : log10_mturn_a_avg, log10_mturn_m_avg)
            for (i = 0; i < HII_TOT_NUM_PIXELS; i++) {
                if (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS &&
                    astro_options_global->USE_MINI_HALOS) {
                    curr_vcb = ini_boxes->lowres_vcb[i];
                }
                if (consts->redshift < simulation_options_global->Z_HEAT_MAX) {
                    if (astro_options_global->USE_MINI_HALOS) {
                        J21_val = previous_spin_temp->J_21_LW[i];
                    }
                    if (uses_reionization_feedback(
                            astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                        Gamma12_val = previous_ionize_box->ionisation_rate_G12[i];
                        zre_val = previous_ionize_box->z_reion[i];
                    }
                }
                // TODO: This code is almost identical to the code in compute_mturns in
                // thermochem.c. The only difference is that the homogeneous (feedback-free) ACG
                // turnover mass is computed once outside the loop. For best modularity, it's worth
                // to consider to use compute_mturns, at the cost of computing the homogeneous ACG
                // turnover mass at each cell. I am not sure how much overhead this would be, if it
                // is negligible then we should definitely use compute_mturns for code clarity
                if (uses_reionization_feedback(astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                    M_turn_r = reionization_feedback(consts->redshift, Gamma12_val, zre_val);
                }
                if (uses_reionization_feedback_in_acgs(
                        astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                    M_turn_a = fmax(M_turn_a, M_turn_r);
                    log10_mturn_a_grid[i] = log10(M_turn_a);
                    log10_mturn_a_avg += log10(M_turn_a);
                }
                if (astro_options_global->USE_MINI_HALOS) {
                    M_turn_m = fmax(molecular_cooling_threshold_with_feedbacks(consts->redshift,
                                                                               J21_val, curr_vcb),
                                    astro_params_global->M_TURN_STELLAR_FEEDBACK);
                    if (uses_reionization_feedback_in_mcgs(
                            astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                        M_turn_m = fmax(M_turn_m, M_turn_r);
                    }
                    log10_mturn_m_grid[i] = log10(M_turn_m);
                    log10_mturn_m_avg += log10(M_turn_m);
                }
            }
        }
    }

    if (uses_reionization_feedback_in_acgs(astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
        // NOTE: This average log10 Mturn will be passed onto the spin temperature calculations
        // where It is used to perform the frequency integrals (over tau, dependent on <XHI>), and
        // possibly for mean fixing. It is the volume-weighted mean of LOG10 Mturn, although we
        // could do another weighting or use Mturn directly None of these are a perfect
        // representation due to the nonlinear way turnover mass affects N_ion
        log10_mturn_a_avg /= HII_TOT_NUM_PIXELS;
        averages[0] = log10_mturn_a_avg;
    } else {
        averages[0] = log10(consts->mturn_a_nofb);
    }

    if (astro_options_global->USE_MINI_HALOS) {
        log10_mturn_m_avg /= HII_TOT_NUM_PIXELS;
        averages[1] = log10_mturn_m_avg;
    } else {
        averages[1] = 0.;  // dummy value for the USE_MINI_HALOS = false branch
    }
}

void sum_halos_onto_grid(double redshift, InitialConditions *ini_boxes, HaloCatalog *halos,
                         float *log10_mturn_a_grid, float *log10_mturn_m_grid,
                         ScalingConstants *consts, HaloBox *grids) {
    float *vel_pointers[3];
    float *vel_pointers_2LPT[3];
    int vel_dim[3];
    int out_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};  // always output to lowres grid
    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        vel_dim[0] = simulation_options_global->DIM;
        vel_dim[1] = simulation_options_global->DIM;
        vel_dim[2] = D_PARA;
        vel_pointers[0] = ini_boxes->hires_vx;
        vel_pointers[1] = ini_boxes->hires_vy;
        vel_pointers[2] = ini_boxes->hires_vz;
        vel_pointers_2LPT[0] = ini_boxes->hires_vx_2LPT;
        vel_pointers_2LPT[1] = ini_boxes->hires_vy_2LPT;
        vel_pointers_2LPT[2] = ini_boxes->hires_vz_2LPT;
    } else {
        vel_dim[0] = simulation_options_global->HII_DIM;
        vel_dim[1] = simulation_options_global->HII_DIM;
        vel_dim[2] = HII_D_PARA;
        vel_pointers[0] = ini_boxes->lowres_vx;
        vel_pointers[1] = ini_boxes->lowres_vy;
        vel_pointers[2] = ini_boxes->lowres_vz;
        vel_pointers_2LPT[0] = ini_boxes->lowres_vx_2LPT;
        vel_pointers_2LPT[1] = ini_boxes->lowres_vy_2LPT;
        vel_pointers_2LPT[2] = ini_boxes->lowres_vz_2LPT;
    }
    move_halo_galprops(redshift, halos, vel_pointers, vel_pointers_2LPT, vel_dim,
                       log10_mturn_a_grid, log10_mturn_m_grid, grids, out_dim, consts);

    LOG_SUPER_DEBUG("Cell 0 Totals: SF: %.2e NI: %.2e", grids->halo_sfr[0], grids->n_ion[0]);
    if (uses_recombination(astro_options_global->RECOMB_MODEL)) {
        LOG_SUPER_DEBUG("FESC * SF %.2e", grids->whalo_sfr[0]);
    }
    if (astro_options_global->USE_TS_FLUCT) {
        LOG_SUPER_DEBUG("X-ray %.2e", grids->halo_xray[0]);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        LOG_SUPER_DEBUG("MINI SM %.2e SF %.2e", grids->halo_stars_mini[0], grids->halo_sfr_mini[0]);
    }
}

// We grid a PERTURBED halofield into the necessary quantities for calculating radiative backgrounds
int ComputeHaloBox(double redshift, InitialConditions *ini_boxes, HaloCatalog *halos,
                   TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, HaloBox *grids) {
    int status;
    Try {
        // get parameters

#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeCosmoParams(cosmo_params_global);
        writeMatterOptions(matter_options_global);
        writeAstroParams(astro_params_global);
        writeAstroOptions(astro_options_global);
#endif

        LOG_DEBUG("Resetting halobox dim %d %llu %llu", simulation_options_global->HII_DIM,
                  HII_D_PARA, HII_TOT_NUM_PIXELS);
        index_huge idx;
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS) private(idx)
        for (idx = 0; idx < HII_TOT_NUM_PIXELS; idx++) {
            grids->n_ion[idx] = 0.0;
            grids->halo_sfr[idx] = 0.0;
            if (astro_options_global->USE_TS_FLUCT) {
                grids->halo_xray[idx] = 0.0;
            }
            if (astro_options_global->USE_MINI_HALOS) {
                grids->halo_sfr_mini[idx] = 0.0;
            }
            if (uses_recombination(astro_options_global->RECOMB_MODEL)) {
                grids->whalo_sfr[idx] = 0.0;
            }
            if (config_settings.EXTRA_HALOBOX_FIELDS) {
                grids->halo_mass[idx] = 0.0;
                grids->halo_stars[idx] = 0.0;
                grids->count[idx] = 0.0;
                if (astro_options_global->USE_MINI_HALOS) {
                    grids->halo_stars_mini[idx] = 0.0;
                }
            }
        }

        ScalingConstants hbox_consts;
        set_scaling_constants(redshift, &hbox_consts, true);

        LOG_DEBUG("Gridding %llu halos...", halos->n_halos);

        double M_min = minimum_source_mass(redshift, false);
        double M_max_integral;

        float *log10_mturn_a_grid = NULL;
        float *log10_mturn_m_grid = NULL;

        if (uses_reionization_feedback_in_acgs(astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
            log10_mturn_a_grid = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        }
        if (astro_options_global->USE_MINI_HALOS) {
            log10_mturn_m_grid = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        }
        double log10_mturn_averages[2];
        get_log10_turnovers(ini_boxes, previous_spin_temp, previous_ionize_box, log10_mturn_a_grid,
                            log10_mturn_m_grid, &hbox_consts, log10_mturn_averages);
        grids->log10_Mcrit_ACG_ave = log10_mturn_averages[0];
        grids->log10_Mcrit_MCG_ave = log10_mturn_averages[1];
        if (source_model_uses_sampled_halos(matter_options_global->SOURCE_MODEL)) {
            sum_halos_onto_grid(redshift, ini_boxes, halos, log10_mturn_a_grid, log10_mturn_m_grid,
                                &hbox_consts, grids);
        }
        // set sub-catalogue properties
        if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_CHMF_SAMPLER) {
            M_max_integral = simulation_options_global->SAMPLER_MIN_MASS;
        } else if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_DEXM_ESF) {
            M_max_integral = RtoM(physconst.l_factor * simulation_options_global->BOX_LEN /
                                  simulation_options_global->DIM);
        } else {
            M_max_integral = M_MAX_INTEGRAL;
        }
        if (M_min < M_max_integral) {
            set_fixed_grids(M_min, M_max_integral, ini_boxes, log10_mturn_a_grid,
                            log10_mturn_m_grid, &hbox_consts, grids);
            LOG_DEBUG("finished integrated component M[%.2e %.2e]", M_min, M_max_integral);
        }
        halobox_debug_print_avg(grids, &hbox_consts, M_min, M_MAX_INTEGRAL);

        if (uses_reionization_feedback_in_acgs(astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
            free(log10_mturn_a_grid);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            free(log10_mturn_m_grid);
        }
        // NOTE: the density-grid based calculations (SOURCE_MODEL='E-INTEGRAL')
        //  use the cell-weighted average of the log10(Mturn) (see issue #369)
        LOG_SUPER_DEBUG("log10 Mutrn ACG: %.6e", pow(10, grids->log10_Mcrit_ACG_ave));
        LOG_SUPER_DEBUG("log10 Mutrn MCG: %.6e", pow(10, grids->log10_Mcrit_MCG_ave));
    }
    Catch(status) { return (status); }
    LOG_DEBUG("Done.");
    return 0;
}

// test function for getting halo properties from the wrapper, can use a lot of memory for large
// catalogs
int test_halo_props(double redshift, float *vcb_grid, float *J21_LW_grid, float *z_re_grid,
                    float *Gamma12_ion_grid, size_huge n_halos, float *halo_masses,
                    float *halo_coords, float *star_rng, float *sfr_rng, float *xray_rng,
                    float *halo_props_out) {
    int status;
    Try {
        // get parameters

        ScalingConstants hbox_consts;
        set_scaling_constants(redshift, &hbox_consts, true);
        print_sc_consts(&hbox_consts);

        LOG_DEBUG("Getting props for %llu halos at z=%.2f", n_halos, redshift);

        double cell_length =
            simulation_options_global->BOX_LEN / simulation_options_global->HII_DIM;

        int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                          HII_D_PARA};

#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
            int x, y, z;
            index_huge i_halo, i_cell;
            double m;
            double J21_val = 0., Gamma12_val = 0., zre_val = 0.;
            double curr_vcb = hbox_consts.vcb_const;
            double M_turn_a = hbox_consts.mturn_a_nofb;
            double M_turn_m = 0.;  // dummy value for the USE_MINI_HALOS = false branch
            double M_turn_r;

            double in_props[3], halo_pos[3];
            HaloProperties out_props;

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
                i_cell = grid_index_general(x, y, z, box_dim);

                // set values before reionisation feedback
                // NOTE: I could easily apply reionization feedback without minihalos but this was
                // not done previously
                if (astro_options_global->USE_MINI_HALOS ||
                    uses_reionization_feedback_in_acgs(
                        astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                    if (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS &&
                        astro_options_global->USE_MINI_HALOS) {
                        curr_vcb = vcb_grid[i_cell];
                    }
                    if (redshift < simulation_options_global->Z_HEAT_MAX) {
                        J21_val = J21_LW_grid[i_cell];
                        Gamma12_val = Gamma12_ion_grid[i_cell];
                        zre_val = z_re_grid[i_cell];
                    }
                    // TODO: This code is almost identical to the code in compute_mturns in
                    // thermochem.c. The only difference is that the homogeneous (feedback-free) ACG
                    // turnover mass is computed once outside the loop. For best modularity, it's
                    // worth to consider to use compute_mturns, at the cost of computing the
                    // homogeneous ACG turnover mass at each cell. I am not sure how much overhead
                    // this would be, if it is negligible then we should definitely use
                    // compute_mturns for code clarity
                    if (uses_reionization_feedback(
                            astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                        M_turn_r = reionization_feedback(redshift, Gamma12_val, zre_val);
                    }
                    if (uses_reionization_feedback_in_acgs(
                            astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                        M_turn_a = fmax(M_turn_a, M_turn_r);
                    }
                    if (astro_options_global->USE_MINI_HALOS) {
                        M_turn_m = fmax(
                            molecular_cooling_threshold_with_feedbacks(redshift, J21_val, curr_vcb),
                            astro_params_global->M_TURN_STELLAR_FEEDBACK);
                        if (uses_reionization_feedback_in_mcgs(
                                astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                            M_turn_m = fmax(M_turn_m, M_turn_r);
                        }
                    }
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

int convert_halo_props(double redshift, InitialConditions *ics, TsBox *prev_ts,
                       IonizedBox *prev_ion, HaloCatalog *halo_catalog,
                       PerturbedHaloCatalog *halo_catalog_out) {
    ScalingConstants hbox_consts;
    set_scaling_constants(redshift, &hbox_consts, true);
    // print_sc_consts(&hbox_consts);
    float *log10_mturn_a_grid = NULL;
    float *log10_mturn_m_grid = NULL;

    if (uses_reionization_feedback_in_acgs(astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
        log10_mturn_a_grid = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
    }
    if (astro_options_global->USE_MINI_HALOS) {
        log10_mturn_m_grid = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
    }
    double mturn_averages[2];
    get_log10_turnovers(ics, prev_ts, prev_ion, log10_mturn_a_grid, log10_mturn_m_grid,
                        &hbox_consts, mturn_averages);

    int lo_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                     HII_D_PARA};  // always output to lowres grid

    double box_to_lores_factor =
        simulation_options_global->HII_DIM / (double)simulation_options_global->DIM;
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        index_huge i_halo;
        double m;

        double M_turn_m = 0.;  // dummy value for the USE_MINI_HALOS = false branch
        double M_turn_a = hbox_consts.mturn_a_nofb;

        double in_props[3];
        double halo_pos[3];
        HaloProperties out_props;

#pragma omp for
        for (i_halo = 0; i_halo < halo_catalog->n_halos; i_halo++) {
            m = halo_catalog->halo_masses[i_halo];
            // It is sometimes useful to make cuts to the halo catalogues before gridding.
            //   We implement this in a simple way, if the user sets a halo's mass to zero we
            //   skip it
            if (m == 0.) {
                continue;
            }

            // the coordinates are already done in PerturbedHaloCatalog
            halo_pos[0] = halo_catalog_out->halo_coords[3 * i_halo + 0] * box_to_lores_factor;
            halo_pos[1] = halo_catalog_out->halo_coords[3 * i_halo + 1] * box_to_lores_factor;
            halo_pos[2] = halo_catalog_out->halo_coords[3 * i_halo + 2] * box_to_lores_factor;

            LOG_ULTRA_DEBUG("getting mturns for halo at (%.2f, %.2f, %.2f)", halo_pos[0],
                            halo_pos[1], halo_pos[2]);

            if (uses_reionization_feedback_in_acgs(
                    astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
                M_turn_a = pow(10, cic_read_float_wrapper(log10_mturn_a_grid, halo_pos, lo_dim));
            }
            if (astro_options_global->USE_MINI_HALOS) {
                M_turn_m = pow(10, cic_read_float_wrapper(log10_mturn_m_grid, halo_pos, lo_dim));
            }

            // these are the halo property RNG sequences
            in_props[0] = halo_catalog->star_rng[i_halo];
            in_props[1] = halo_catalog->sfr_rng[i_halo];
            in_props[2] = halo_catalog->xray_rng[i_halo];

            LOG_ULTRA_DEBUG("Halo %llu mass %.2e Mturn_a %.2e Mturn_m %.2e", i_halo, m, M_turn_a,
                            M_turn_m);
            LOG_ULTRA_DEBUG("RNG: STAR %.2e SFR %.2e XRAY %.2e", in_props[0], in_props[1],
                            in_props[2]);
            set_halo_properties(m, M_turn_a, M_turn_m, &hbox_consts, in_props, &out_props);

            halo_catalog_out->halo_masses[i_halo] = out_props.halo_mass;
            halo_catalog_out->stellar_masses[i_halo] = out_props.stellar_mass;
            halo_catalog_out->sfr[i_halo] = out_props.halo_sfr;
            halo_catalog_out->ion_emissivity[i_halo] = out_props.n_ion;

            if (astro_options_global->USE_MINI_HALOS) {
                halo_catalog_out->stellar_mini[i_halo] = out_props.stellar_mass_mini;
                halo_catalog_out->sfr_mini[i_halo] = out_props.sfr_mini;
            }
            if (uses_recombination(astro_options_global->RECOMB_MODEL)) {
                halo_catalog_out->fesc_sfr[i_halo] = out_props.fescweighted_sfr;
            }
            if (astro_options_global->USE_TS_FLUCT) {
                halo_catalog_out->xray_emissivity[i_halo] = out_props.halo_xray;
            }

            if (i_halo < 10) {
                LOG_ULTRA_DEBUG("HM %.2e SM %.2e SF %.2e NI %.2e LX %.2e", out_props.halo_mass,
                                out_props.stellar_mass, out_props.halo_sfr, out_props.n_ion,
                                out_props.halo_xray);
                LOG_ULTRA_DEBUG("MINI: SM %.2e SF %.2e WSF %.2e", out_props.stellar_mass_mini,
                                out_props.sfr_mini, out_props.fescweighted_sfr);
                LOG_ULTRA_DEBUG("Mturns ACG %.2e MCG %.2e", M_turn_a, M_turn_m);
                LOG_ULTRA_DEBUG("RNG: STAR %.2e SFR %.2e XRAY %.2e", in_props[0], in_props[1],
                                in_props[2]);
            }
        }
    }
    if (uses_reionization_feedback_in_acgs(astro_options_global->REIONIZATION_FEEDBACK_MODEL)) {
        free(log10_mturn_a_grid);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        free(log10_mturn_m_grid);
    }
    return 0;
}
