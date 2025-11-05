// Functions in this file map units of mass from Lagrangian (IC)
//  coordinates to their real (Eulerian) Locations, these can sum
//  masses or galaxy properties from grids or from coordinate catalogues

#include "map_mass.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "HaloBox.h"
#include "InputParameters.h"
#include "cosmology.h"
#include "indexing.h"
#include "logger.h"

#define do_cic_interpolation(arr, ...)                                                           \
    _Generic((arr), float *: do_cic_interpolation_float, double *: do_cic_interpolation_double)( \
        arr, __VA_ARGS__)

static inline void do_cic_interpolation_double(double *resampled_box, double pos[3], int box_dim[3],
                                               double curr_dens) {
    // get the CIC indices and distances
    int ipos[3], iposp1[3];
    double dist[3];
    // NOTE: assumes the cell at idx == 0 is *centred* at (0,0,0)
    for (int axis = 0; axis < 3; axis++) {
        ipos[axis] = (int)floor(pos[axis]);
        iposp1[axis] = ipos[axis] + 1;
        dist[axis] = pos[axis] - ipos[axis];
    }

    wrap_coord(ipos, box_dim);
    wrap_coord(iposp1, box_dim);

    unsigned long long int cic_indices[8] = {
        grid_index_general(ipos[0], ipos[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], ipos[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(ipos[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], iposp1[2], box_dim)};

    double cic_weights[8] = {(1. - dist[0]) * (1. - dist[1]) * (1. - dist[2]),
                             dist[0] * (1. - dist[1]) * (1. - dist[2]),
                             (1. - dist[0]) * dist[1] * (1. - dist[2]),
                             dist[0] * dist[1] * (1. - dist[2]),
                             (1. - dist[0]) * (1. - dist[1]) * dist[2],
                             dist[0] * (1. - dist[1]) * dist[2],
                             (1. - dist[0]) * dist[1] * dist[2],
                             dist[0] * dist[1] * dist[2]};

    for (int i = 0; i < 8; i++) {
#pragma omp atomic update
        resampled_box[cic_indices[i]] += curr_dens * cic_weights[i];
    }
}

// Identical code as above, using a single precision output
static inline void do_cic_interpolation_float(float *resampled_box, double pos[3], int box_dim[3],
                                              double curr_dens) {
    // get the CIC indices and distances
    int ipos[3], iposp1[3];
    double dist[3];
    // NOTE: assumes the cell at idx == 0 is *centred* at (0,0,0)
    for (int axis = 0; axis < 3; axis++) {
        ipos[axis] = (int)floor(pos[axis]);
        iposp1[axis] = ipos[axis] + 1;
        dist[axis] = pos[axis] - ipos[axis];
    }

    wrap_coord(ipos, box_dim);
    wrap_coord(iposp1, box_dim);

    unsigned long long int cic_indices[8] = {
        grid_index_general(ipos[0], ipos[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], ipos[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(ipos[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], iposp1[2], box_dim)};

    double cic_weights[8] = {(1. - dist[0]) * (1. - dist[1]) * (1. - dist[2]),
                             dist[0] * (1. - dist[1]) * (1. - dist[2]),
                             (1. - dist[0]) * dist[1] * (1. - dist[2]),
                             dist[0] * dist[1] * (1. - dist[2]),
                             (1. - dist[0]) * (1. - dist[1]) * dist[2],
                             dist[0] * (1. - dist[1]) * dist[2],
                             (1. - dist[0]) * dist[1] * dist[2],
                             dist[0] * dist[1] * dist[2]};

    for (int i = 0; i < 8; i++) {
#pragma omp atomic update
        resampled_box[cic_indices[i]] += curr_dens * cic_weights[i];
    }
}

static inline double cic_read_float(float *box, double pos[3], int box_dim[3]) {
    // get the CIC indices and distances
    int ipos[3], iposp1[3];
    double dist[3];
    double sum = 0;
    // NOTE: assumes the cell at idx == 0 is *centred* at (0,0,0)
    for (int axis = 0; axis < 3; axis++) {
        ipos[axis] = (int)floor(pos[axis]);
        iposp1[axis] = ipos[axis] + 1;
        dist[axis] = pos[axis] - ipos[axis];
    }

    wrap_coord(ipos, box_dim);
    wrap_coord(iposp1, box_dim);

    unsigned long long int cic_indices[8] = {
        grid_index_general(ipos[0], ipos[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], ipos[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(ipos[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], iposp1[2], box_dim)};

    double cic_weights[8] = {(1. - dist[0]) * (1. - dist[1]) * (1. - dist[2]),
                             dist[0] * (1. - dist[1]) * (1. - dist[2]),
                             (1. - dist[0]) * dist[1] * (1. - dist[2]),
                             dist[0] * dist[1] * (1. - dist[2]),
                             (1. - dist[0]) * (1. - dist[1]) * dist[2],
                             dist[0] * (1. - dist[1]) * dist[2],
                             (1. - dist[0]) * dist[1] * dist[2],
                             dist[0] * dist[1] * dist[2]};

    for (int i = 0; i < 8; i++) {
        sum += cic_weights[i] * box[cic_indices[i]];
    }
    return sum;
}

double cic_read_float_wrapper(float *box, double pos[3], int box_dim[3]) {
    return cic_read_float(box, pos, box_dim);
}

// Function that maps a IC density grid to the perturbed density grid
void move_grid_masses(double redshift, float *dens_pointer, int dens_dim[3], float *vel_pointers[3],
                      float *vel_pointers_2LPT[3], int vel_dim[3], double *resampled_box,
                      int out_dim[3]) {
    // grid dimension constants
    double boxlen = simulation_options_global->BOX_LEN;
    double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
    double box_size[3] = {boxlen, boxlen, boxlen_z};
    double dim_ratio_vel = (double)vel_dim[0] / (double)dens_dim[0];
    double dim_ratio_out = (double)out_dim[0] / (double)dens_dim[0];

    // Setup IC velocity factors
    double growth_factor = dicke(redshift);
    double displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

    double init_growth_factor = dicke(simulation_options_global->INITIAL_REDSHIFT);
    double init_displacement_factor_2LPT =
        -(3.0 / 7.0) * init_growth_factor * init_growth_factor;  // 2LPT eq. D8

    double velocity_displacement_factor[3] = {
        (growth_factor - init_growth_factor) / box_size[0] * dens_dim[0],
        (growth_factor - init_growth_factor) / box_size[1] * dens_dim[1],
        (growth_factor - init_growth_factor) / box_size[2] * dens_dim[2]};
    double velocity_displacement_factor_2LPT[3] = {
        (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[0] * dens_dim[0],
        (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[1] * dens_dim[1],
        (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[2] * dens_dim[2]};
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        int i, j, k, axis;
        double pos[3], curr_dens;
        int ipos[3];
        unsigned long long vel_index, dens_index;
#pragma omp for
        for (i = 0; i < dens_dim[0]; i++) {
            for (j = 0; j < dens_dim[1]; j++) {
                for (k = 0; k < dens_dim[2]; k++) {
                    // Transform position to units of box size
                    pos[0] = i;
                    pos[1] = j;
                    pos[2] = k;
                    resample_index((int[3]){i, j, k}, dim_ratio_vel, ipos);
                    wrap_coord(ipos, vel_dim);
                    vel_index = grid_index_general(ipos[0], ipos[1], ipos[2], vel_dim);
                    for (axis = 0; axis < 3; axis++) {
                        pos[axis] +=
                            vel_pointers[axis][vel_index] * velocity_displacement_factor[axis];
                        // add 2LPT second order corrections
                        if (matter_options_global->PERTURB_ALGORITHM == 2) {
                            pos[axis] -= vel_pointers_2LPT[axis][vel_index] *
                                         velocity_displacement_factor_2LPT[axis];
                        }
                        pos[axis] *= dim_ratio_out;
                    }

                    // CIC interpolation
                    dens_index = grid_index_general(i, j, k, dens_dim);
                    curr_dens = 1.0 + dens_pointer[dens_index] * init_growth_factor;
                    do_cic_interpolation(resampled_box, pos, out_dim, curr_dens);
                }
            }
        }
    }
}

// Function that maps a IC density grid to the perturbed density grid
// TODO: This shares a lot of code with move_grid_masses and (future) move_cat_galprops.
//  I should look into combining elements, however since the differences
//  are on the innermost loops, any generalisation is likely to slow things down.
void move_grid_galprops(double redshift, float *dens_pointer, int dens_dim[3],
                        float *vel_pointers[3], float *vel_pointers_2LPT[3], int vel_dim[3],
                        HaloBox *boxes, int out_dim[3], float *mturn_a_grid, float *mturn_m_grid,
                        ScalingConstants *consts, IntegralCondition *integral_cond) {
    // grid dimension constants
    double boxlen = simulation_options_global->BOX_LEN;
    double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
    double box_size[3] = {boxlen, boxlen, boxlen_z};
    double dim_ratio_vel = (double)vel_dim[0] / (double)dens_dim[0];
    double dim_ratio_out = (double)out_dim[0] / (double)dens_dim[0];
    double vol_ratio_out = (double)(out_dim[0] * out_dim[1] * out_dim[2]) /
                           (double)(dens_dim[0] * dens_dim[1] * dens_dim[2]);

    double prefactor_mass = RHOcrit * cosmo_params_global->OMm * vol_ratio_out;
    double prefactor_stars = RHOcrit * cosmo_params_global->OMb * consts->fstar_10 * vol_ratio_out;
    double prefactor_stars_mini =
        RHOcrit * cosmo_params_global->OMb * consts->fstar_7 * vol_ratio_out;
    double prefactor_xray = RHOcrit * cosmo_params_global->OMm * vol_ratio_out;

    double prefactor_sfr = prefactor_stars / consts->t_star / consts->t_h;
    double prefactor_sfr_mini = prefactor_stars_mini / consts->t_star / consts->t_h;
    double prefactor_nion = prefactor_stars * consts->fesc_10 * consts->pop2_ion;
    double prefactor_nion_mini = prefactor_stars_mini * consts->fesc_7 * consts->pop3_ion;

    // Setup IC velocity factors
    double growth_factor = dicke(redshift);
    double displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

    double init_growth_factor = dicke(simulation_options_global->INITIAL_REDSHIFT);
    double init_displacement_factor_2LPT =
        -(3.0 / 7.0) * init_growth_factor * init_growth_factor;  // 2LPT eq. D8

    double velocity_displacement_factor[3] = {
        (growth_factor - init_growth_factor) / box_size[0] * dens_dim[0],
        (growth_factor - init_growth_factor) / box_size[1] * dens_dim[1],
        (growth_factor - init_growth_factor) / box_size[2] * dens_dim[2]};
    double velocity_displacement_factor_2LPT[3] = {
        (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[0] * dens_dim[0],
        (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[1] * dens_dim[1],
        (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[2] * dens_dim[2]};
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        int i, j, k, axis;
        double pos[3], curr_dens;
        int ipos[3];
        unsigned long long vel_index, dens_index, mturn_index;
        double l10_mturn_a = log10(consts->mturn_a_nofb);
        double l10_mturn_m = log10(consts->mturn_m_nofb);
        HaloProperties properties;
#pragma omp for
        for (i = 0; i < dens_dim[0]; i++) {
            for (j = 0; j < dens_dim[1]; j++) {
                for (k = 0; k < dens_dim[2]; k++) {
                    // Transform position to units of box size
                    pos[0] = i;
                    pos[1] = j;
                    pos[2] = k;
                    resample_index((int[3]){i, j, k}, dim_ratio_vel, ipos);
                    wrap_coord(ipos, vel_dim);
                    vel_index = grid_index_general(ipos[0], ipos[1], ipos[2], vel_dim);
                    for (axis = 0; axis < 3; axis++) {
                        pos[axis] +=
                            vel_pointers[axis][vel_index] * velocity_displacement_factor[axis];
                        // add 2LPT second order corrections
                        if (matter_options_global->PERTURB_ALGORITHM == 2) {
                            pos[axis] -= vel_pointers_2LPT[axis][vel_index] *
                                         velocity_displacement_factor_2LPT[axis];
                        }
                        pos[axis] *= dim_ratio_out;
                    }

                    // CIC interpolation
                    dens_index = grid_index_general(i, j, k, dens_dim);
                    curr_dens = dens_pointer[dens_index] * growth_factor;

                    // mturn grids are at the output resolution (lower res)
                    if (astro_options_global->USE_MINI_HALOS) {
                        resample_index((int[3]){i, j, k}, dim_ratio_out, ipos);
                        mturn_index = grid_index_general(ipos[0], ipos[1], ipos[2], out_dim);
                        l10_mturn_a = mturn_a_grid[dens_index];
                        l10_mturn_m = mturn_m_grid[dens_index];
                    }

                    get_cell_integrals(curr_dens, l10_mturn_a, l10_mturn_m, consts, integral_cond,
                                       &properties);

                    // using the properties struct:
                    // stellar_mass --> no F_esc integral ACG
                    // stellar_mass_mini --> no F_esc integral MCG
                    // n_ion --> F_esc integral ACG
                    // fescweighted_sfr --> F_esc integral MCG
                    // halo_xray --> Xray integral
                    do_cic_interpolation(boxes->halo_sfr, pos, out_dim,
                                         properties.stellar_mass * prefactor_sfr);
                    do_cic_interpolation(boxes->n_ion, pos, out_dim,
                                         properties.n_ion * prefactor_nion +
                                             properties.fescweighted_sfr * prefactor_nion_mini);

                    if (astro_options_global->USE_MINI_HALOS) {
                        do_cic_interpolation(boxes->halo_sfr_mini, pos, out_dim,
                                             properties.stellar_mass_mini * prefactor_sfr_mini);
                    }
                    if (astro_options_global->USE_TS_FLUCT) {
                        do_cic_interpolation(boxes->halo_xray, pos, out_dim,
                                             properties.halo_xray * prefactor_xray);
                    }

                    if (config_settings.EXTRA_HALOBOX_FIELDS) {
                        do_cic_interpolation(boxes->halo_mass, pos, out_dim,
                                             properties.halo_mass * prefactor_mass);
                        do_cic_interpolation(boxes->halo_stars, pos, out_dim,
                                             properties.stellar_mass * prefactor_stars);
                        if (astro_options_global->USE_MINI_HALOS) {
                            do_cic_interpolation(
                                boxes->halo_stars_mini, pos, out_dim,
                                properties.stellar_mass_mini * prefactor_stars_mini);
                        }
                    }
                }
            }
        }
    }
    // Without stochasticity, these grids are the same to a constant
    double prefactor_wsfr = 1 / consts->t_h / consts->t_star;
    if (astro_options_global->INHOMO_RECO) {
        for (int i = 0; i < HII_TOT_NUM_PIXELS; i++) {
            boxes->whalo_sfr[i] = boxes->n_ion[i] * prefactor_wsfr;
        }
    }
}

void move_halo_galprops(double redshift, HaloCatalog *halos, float *vel_pointers[3],
                        float *vel_pointers_2LPT[3], int vel_dim[3], float *mturn_a_grid,
                        float *mturn_m_grid, HaloBox *boxes, int out_dim[3],
                        ScalingConstants *consts) {
    // grid dimension constants
    double boxlen = simulation_options_global->BOX_LEN;
    double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
    double box_size[3] = {boxlen, boxlen, boxlen_z};
    double cell_size_inv_v = vel_dim[0] / simulation_options_global->BOX_LEN;
    double cell_size_inv_o = out_dim[0] / simulation_options_global->BOX_LEN;
    double cell_vol_inv = cell_size_inv_o * cell_size_inv_o * cell_size_inv_o;

    // Setup IC velocity factors
    double growth_factor = dicke(redshift);
    double displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

    double init_growth_factor = dicke(simulation_options_global->INITIAL_REDSHIFT);
    double init_displacement_factor_2LPT =
        -(3.0 / 7.0) * init_growth_factor * init_growth_factor;  // 2LPT eq. D8

    // Since the halo coords are already in Mpc units, we don't need the box factors
    double velocity_displacement_factor = growth_factor - init_growth_factor;
    double velocity_displacement_factor_2LPT =
        displacement_factor_2LPT - init_displacement_factor_2LPT;
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        int i, axis;
        double pos[3];
        int ipos[3];
        unsigned long long vel_index;
        HaloProperties properties;
        double M_turn_a = consts->mturn_a_nofb;
        double M_turn_m = consts->mturn_m_nofb;
        double halo_rng[3];
        double hmass;
#pragma omp for
        for (i = 0; i < halos->n_halos; i++) {
            hmass = halos->halo_masses[i];
            // It is sometimes useful to make cuts to the halo catalogues before gridding.
            //   We implement this in a simple way, if the halo's mass is set to zero we skip it
            if (hmass == 0.) {
                continue;
            }
            // Transform position to units of box size
            pos[0] = halos->halo_coords[3 * i + 0];
            pos[1] = halos->halo_coords[3 * i + 1];
            pos[2] = halos->halo_coords[3 * i + 2];
            pos_to_index(pos, cell_size_inv_v, ipos);
            wrap_coord(ipos, vel_dim);
            vel_index = grid_index_general(ipos[0], ipos[1], ipos[2], vel_dim);
            for (axis = 0; axis < 3; axis++) {
                pos[axis] += vel_pointers[axis][vel_index] * velocity_displacement_factor;
                // add 2LPT second order corrections
                if (matter_options_global->PERTURB_ALGORITHM == 2) {
                    pos[axis] -=
                        vel_pointers_2LPT[axis][vel_index] * velocity_displacement_factor_2LPT;
                }
            }

            // convert to cell size for the cic
            pos[0] = pos[0] * out_dim[0] / box_size[0];
            pos[1] = pos[1] * out_dim[1] / box_size[1];
            pos[2] = pos[2] * out_dim[2] / box_size[2];

            if (astro_options_global->USE_MINI_HALOS) {
                M_turn_a = pow(10, cic_read_float(mturn_a_grid, pos, out_dim));
                M_turn_m = pow(10, cic_read_float(mturn_m_grid, pos, out_dim));
            }
            halo_rng[0] = halos->star_rng[i];
            halo_rng[1] = halos->sfr_rng[i];
            halo_rng[2] = halos->xray_rng[i];

            // CIC interpolation
            set_halo_properties(hmass, M_turn_a, M_turn_m, consts, halo_rng, &properties);
            do_cic_interpolation(boxes->halo_sfr, pos, out_dim, properties.halo_sfr);
            do_cic_interpolation(boxes->n_ion, pos, out_dim, properties.n_ion);
            if (astro_options_global->USE_MINI_HALOS) {
                do_cic_interpolation(boxes->halo_sfr_mini, pos, out_dim, properties.sfr_mini);
            }
            if (astro_options_global->USE_TS_FLUCT) {
                do_cic_interpolation(boxes->halo_xray, pos, out_dim, properties.halo_xray);
            }
            if (astro_options_global->INHOMO_RECO) {
                do_cic_interpolation(boxes->whalo_sfr, pos, out_dim, properties.fescweighted_sfr);
            }
            if (config_settings.EXTRA_HALOBOX_FIELDS) {
                do_cic_interpolation(boxes->halo_mass, pos, out_dim, properties.halo_mass);
                do_cic_interpolation(boxes->halo_stars, pos, out_dim, properties.stellar_mass);
                if (astro_options_global->USE_MINI_HALOS) {
                    do_cic_interpolation(boxes->halo_stars_mini, pos, out_dim,
                                         properties.stellar_mass_mini);
                }
            }

#if LOG_LEVEL >= ULTRA_DEBUG_LEVEL
            if (i < 10) {
                LOG_ULTRA_DEBUG(
                    "First 10 Halos: HM: %.2e SM: %.2e (%.2e) SF: %.2e (%.2e) X: %.2e NI: %.2e WS: "
                    "%.2e Z : %.2e ct : %llu",
                    hmass, properties.stellar_mass, properties.stellar_mass_mini,
                    properties.halo_sfr, properties.sfr_mini, properties.halo_xray,
                    properties.n_ion, properties.fescweighted_sfr, properties.metallicity, i);
                LOG_ULTRA_DEBUG("Mturn_a %.2e Mturn_m %.2e RNG %.3f %.3f %.3f", M_turn_a, M_turn_m,
                                halo_rng[0], halo_rng[1], halo_rng[2]);
            }
#endif
        }
#pragma omp for
        for (unsigned long long int i_cell = 0; i_cell < HII_TOT_NUM_PIXELS; i_cell++) {
            boxes->n_ion[i_cell] *= cell_vol_inv;
            boxes->halo_sfr[i_cell] *= cell_vol_inv;
            if (astro_options_global->USE_TS_FLUCT) {
                boxes->halo_xray[i_cell] *= cell_vol_inv;
            }
            if (astro_options_global->INHOMO_RECO) {
                boxes->whalo_sfr[i_cell] *= cell_vol_inv;
            }
            if (astro_options_global->USE_MINI_HALOS) {
                boxes->halo_sfr_mini[i_cell] *= cell_vol_inv;
            }
            if (config_settings.EXTRA_HALOBOX_FIELDS) {
                boxes->halo_mass[i_cell] *= cell_vol_inv;
                boxes->halo_stars[i_cell] *= cell_vol_inv;
                if (astro_options_global->USE_MINI_HALOS) {
                    boxes->halo_stars_mini[i_cell] *= cell_vol_inv;
                }
            }
        }
    }
}
