/* This file defines specific interpolation table initialisation functions, kept separate from the
   general interpolation table routines In order to allow them to use calculations based on other
   interpolation tables. Most importantly these fucntions require those from ps.c which requires the
   sigma(M) interpolation tables */
#include "interp_tables.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_spline.h>
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
#include "interpolation.h"
#include "logger.h"

// fixed limits and bin numbers for tables
#define NDELTA 400
#define NMTURN 50  // 100
#define LOG10_MTURN_MAX ((double)(10))
#define LOG10_MTURN_MIN ((double)(5. - 9e-8))
#define MAX_ITER_RF 200
#define N_MASS_INTERP 300

// we need to define a density minimum for the tables, since we are in lagrangian density / linear
// growth it's possible to go below -1 so we explicitly set a minimum here which sets table limits
// and puts no halos in cells below that (Lagrangian) density

// Tables for the grids
static RGTable1D SFRD_z_table = {.allocated = false};
static RGTable1D Nion_z_table = {.allocated = false};
static RGTable1D Xray_z_table_1D = {.allocated = false};
static RGTable2D SFRD_z_table_MINI = {.allocated = false};
static RGTable2D Nion_z_table_MINI = {.allocated = false};
static RGTable2D Xray_z_table_2D = {.allocated = false};
// TODO: SFRD tables assume no reionisation feedback, this is self-inconsistent, but probably okay
// given
//  it's used (mostly) in the SpinTemperature, which deals with neutral regions
//  Will overestimate integral component of SFRD lightcones used in observation
static RGTable1D_f SFRD_conditional_table = {.allocated = false};
static RGTable1D_f Nion_conditional_table1D = {.allocated = false};
static RGTable2D_f Nion_conditional_table2D = {.allocated = false};
static RGTable2D_f Nion_conditional_table_MINI = {.allocated = false};
static RGTable2D_f SFRD_conditional_table_MINI = {.allocated = false};
static RGTable2D_f Nion_conditional_table_prev = {.allocated = false};
static RGTable2D_f Nion_conditional_table_MINI_prev = {.allocated = false};
static RGTable2D_f Xray_conditional_table_2D = {.allocated = false};
static RGTable1D_f Xray_conditional_table_1D = {.allocated = false};

// Tables for the catalogues
static RGTable1D Nhalo_table = {.allocated = false};
static RGTable1D Mcoll_table = {.allocated = false};
static RGTable2D Nhalo_inv_table = {.allocated = false};

// Tables for the old parametrization
static RGTable1D fcoll_z_table = {.allocated = false};
static RGTable1D_f fcoll_conditional_table = {
    .allocated = false,
};
static RGTable1D_f dfcoll_conditional_table = {
    .allocated = false,
};

// J table for binary split algorithm
static RGTable1D J_split_table = {.allocated = false};

// Sigma inverse table for partition algorithm
// Since we want to easily construct it from the sigma table, it won't be uniform so use GSL
// TODO: Consider a rootfind on the integrals for accuracy and speed if we want an RGTable
//       It should only need one calculation per run
static gsl_spline *Sigma_inv_table;
static gsl_interp_accel *Sigma_inv_table_acc;
#pragma omp threadprivate(Sigma_inv_table_acc)

// Sigma interpolation tables
static RGTable1D_f Sigma_InterpTable = {
    .allocated = false,
};
static RGTable1D_f dSigmasqdm_InterpTable = {
    .allocated = false,
};

// NOTE: this table is initialised for up to N_redshift x N_Mturn, but only called N_filter times to
// assign ST_over_PS in Spintemp.
//   It may be better to just do the integrals at each R
void initialise_SFRD_spline(int Nbin, float zmin, float zmax, struct ScalingConstants *sc) {
    int i, j;
    double Mmax = M_MAX_INTEGRAL;
    double lnMmax = log(Mmax);

    LOG_SUPER_DEBUG("initing SFRD spline from %.2f to %.2f", zmin, zmax);
    if (!SFRD_z_table.allocated) {
        allocate_RGTable1D(Nbin, &SFRD_z_table);
    }
    if (astro_options_global->USE_MINI_HALOS && !SFRD_z_table_MINI.allocated) {
        allocate_RGTable2D(Nbin, NMTURN, &SFRD_z_table_MINI);
    }

    SFRD_z_table.x_min = zmin;
    SFRD_z_table.x_width = (zmax - zmin) / ((double)Nbin - 1.);

    if (astro_options_global->USE_MINI_HALOS) {
        SFRD_z_table_MINI.x_min = zmin;
        SFRD_z_table_MINI.x_width = (zmax - zmin) / ((double)Nbin - 1.);
        SFRD_z_table_MINI.y_min = LOG10_MTURN_MIN;
        SFRD_z_table_MINI.y_width = (LOG10_MTURN_MAX - LOG10_MTURN_MIN) / ((double)NMTURN - 1.);
    }

#pragma omp parallel private(i, j) num_threads(simulation_options_global -> N_THREADS)
    {
        struct ScalingConstants sc_sfrd;
        sc_sfrd = evolve_scaling_constants_sfr(sc);
        double mturn_mcg;
        double lnMmin;
        double z_val;
#pragma omp for
        for (i = 0; i < Nbin; i++) {
            z_val = SFRD_z_table.x_min +
                    i * SFRD_z_table.x_width;  // both tables will have the same values here
            sc_sfrd = evolve_scaling_constants_to_redshift(z_val, &sc_sfrd, false);
            lnMmin = log(minimum_source_mass(z_val, true));

            if (astro_options_global->USE_MINI_HALOS) {
                for (j = 0; j < NMTURN; j++) {
                    mturn_mcg = pow(10, SFRD_z_table_MINI.y_min + j * SFRD_z_table_MINI.y_width);
                    SFRD_z_table_MINI.z_arr[i][j] =
                        Nion_General_MINI(z_val, lnMmin, lnMmax, mturn_mcg, &sc_sfrd);
                }
            }
            SFRD_z_table.y_arr[i] =
                Nion_General(z_val, lnMmin, lnMmax, sc_sfrd.mturn_a_nofb, &sc_sfrd);
        }
    }

    for (i = 0; i < Nbin; i++) {
        if (isfinite(SFRD_z_table.y_arr[i]) == 0) {
            LOG_ERROR("Detected either an infinite or NaN value in SFRD table");
            Throw(TableGenerationError);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            for (j = 0; j < NMTURN; j++) {
                if (isfinite(SFRD_z_table_MINI.z_arr[i][j]) == 0) {
                    LOG_ERROR("Detected either an infinite or NaN value in SFRD_MINI table");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

// Unlike the SFRD spline, this one is used more due to the nu_tau_one() rootfind
// although still ignores reionisation feedback
void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, struct ScalingConstants *sc) {
    int i, j;
    double Mmax = M_MAX_INTEGRAL;
    double lnMmax = log(Mmax);

    LOG_SUPER_DEBUG("initing Nion spline from %.2f to %.2f", zmin, zmax);

    if (!Nion_z_table.allocated) {
        allocate_RGTable1D(Nbin, &Nion_z_table);
    }
    if (astro_options_global->USE_MINI_HALOS && !Nion_z_table_MINI.allocated) {
        allocate_RGTable2D(Nbin, NMTURN, &Nion_z_table_MINI);
    }
    Nion_z_table.x_min = zmin;
    Nion_z_table.x_width = (zmax - zmin) / ((double)Nbin - 1.);
    if (astro_options_global->USE_MINI_HALOS) {
        Nion_z_table_MINI.x_min = zmin;
        Nion_z_table_MINI.x_width = (zmax - zmin) / ((double)Nbin - 1.);
        Nion_z_table_MINI.y_min = LOG10_MTURN_MIN;
        Nion_z_table_MINI.y_width = (LOG10_MTURN_MAX - LOG10_MTURN_MIN) / ((double)NMTURN - 1.);
    }

#pragma omp parallel private(i, j) num_threads(simulation_options_global -> N_THREADS)
    {
        struct ScalingConstants sc_z;
        double mturn_mcg;
        double z_val;
        double lnMmin;
#pragma omp for
        for (i = 0; i < Nbin; i++) {
            z_val = Nion_z_table.x_min +
                    i * Nion_z_table.x_width;  // both tables will have the same values here
            sc_z = evolve_scaling_constants_to_redshift(z_val, sc, false);
            // Minor note: while this is called in xray, we use it to estimate ionised fraction, do
            // we use ION_Tvir_MIN if applicable?
            lnMmin = log(minimum_source_mass(z_val, true));
            if (astro_options_global->USE_MINI_HALOS) {
                for (j = 0; j < NMTURN; j++) {
                    mturn_mcg = pow(10, Nion_z_table_MINI.y_min + j * Nion_z_table_MINI.y_width);
                    Nion_z_table_MINI.z_arr[i][j] =
                        Nion_General_MINI(z_val, lnMmin, lnMmax, mturn_mcg, &sc_z);
                }
            }
            Nion_z_table.y_arr[i] = Nion_General(z_val, lnMmin, lnMmax, sc_z.mturn_a_nofb, &sc_z);
        }
    }

    for (i = 0; i < Nbin; i++) {
        if (isfinite(Nion_z_table.y_arr[i]) == 0) {
            LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val");
            Throw(TableGenerationError);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            for (j = 0; j < NMTURN; j++) {
                if (isfinite(Nion_z_table_MINI.z_arr[i][j]) == 0) {
                    LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val_MINI");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

void initialise_FgtrM_delta_table(double min_dens, double max_dens, double zpp, double growth_zpp,
                                  double smin_zpp, double smax_zpp) {
    int i;
    double dens;

    LOG_SUPER_DEBUG("Initialising FgtrM table between delta %.3e and %.3e, sigma %.3e and %.3e",
                    min_dens, max_dens, smin_zpp, smax_zpp);

    if (!fcoll_conditional_table.allocated) {
        allocate_RGTable1D_f(dens_Ninterp, &fcoll_conditional_table);
    }
    fcoll_conditional_table.x_min = min_dens;
    fcoll_conditional_table.x_width = (max_dens - min_dens) / (dens_Ninterp - 1.);
    if (!dfcoll_conditional_table.allocated) {
        allocate_RGTable1D_f(dens_Ninterp, &dfcoll_conditional_table);
    }
    dfcoll_conditional_table.x_min = fcoll_conditional_table.x_min;
    dfcoll_conditional_table.x_width = fcoll_conditional_table.x_width;

    // dens_Ninterp is a global define, probably shouldn't be
    for (i = 0; i < dens_Ninterp; i++) {
        dens = fcoll_conditional_table.x_min + i * fcoll_conditional_table.x_width;
        fcoll_conditional_table.y_arr[i] = FgtrM_bias_fast(growth_zpp, dens, smin_zpp, smax_zpp);
        dfcoll_conditional_table.y_arr[i] = dfcoll_dz(zpp, smin_zpp, dens, smax_zpp);
    }
}

void init_FcollTable(double zmin, double zmax, bool x_ray) {
    int i;
    double z_val, M_min, lnMmin, lnMmax;

    fcoll_z_table.x_min = zmin;
    fcoll_z_table.x_width = 0.1;

    int n_z = (int)ceil((zmax - zmin) / fcoll_z_table.x_width) + 1;

    if (!fcoll_z_table.allocated) {
        allocate_RGTable1D(n_z, &fcoll_z_table);
    }

    for (i = 0; i < n_z; i++) {
        z_val = fcoll_z_table.x_min + i * fcoll_z_table.x_width;
        M_min = minimum_source_mass(z_val, x_ray);
        lnMmin = log(M_min);
        lnMmax = log(fmax(M_MAX_INTEGRAL, M_min * 100));

        // if we are press-schechter we can save time by calling the erfc
        if (matter_options_global->HMF == 0)
            fcoll_z_table.y_arr[i] = FgtrM(z_val, M_min);
        else {
            if (astro_options_global->INTEGRATION_METHOD_ATOMIC == 1 ||
                (astro_options_global->USE_MINI_HALOS &&
                 astro_options_global->INTEGRATION_METHOD_MINI == 1))
                initialise_GL(lnMmin, lnMmax);
            fcoll_z_table.y_arr[i] = Fcoll_General(z_val, lnMmin, lnMmax);
        }
    }
}

// NOTE: since reionisation feedback is not included in the Ts calculation, the SFRD spline
//   is Rx1D unlike the Mini table, which is Rx2D
// NOTE: SFRD tables have fixed Mturn range, Nion tables vary
// NOTE: it would be slightly less accurate but maybe faster to tabulate in linear delta, linear
// Fcoll rather than linear-log, check the profiles
void initialise_Nion_Conditional_spline(double z, double min_density, double max_density,
                                        double Mmin, double Mmax, double Mcond,
                                        double log10Mturn_min, double log10Mturn_max,
                                        double log10Mturn_min_MINI, double log10Mturn_max_MINI,
                                        struct ScalingConstants *sc, bool prev) {
    int i, j;
    double overdense_table[NDELTA];
    double mturns[NMTURN], mturns_MINI[NMTURN];
    RGTable2D_f *table_2d, *table_mini;

    LOG_SUPER_DEBUG("Initialising Nion conditional table at mass %.2e from delta %.2e to %.2e",
                    Mcond, min_density, max_density);
    LOG_SUPER_DEBUG("l10Mturns ACG %.2e %.2e MCG %.2e %.2e", log10Mturn_min, log10Mturn_max,
                    log10Mturn_min_MINI, log10Mturn_max_MINI);

    double growthf = dicke(z);
    double lnMmin = log(Mmin);
    double lnMmax = log(Mmax);
    double sigma2 = EvaluateSigma(log(Mcond));

    // If we use minihalos, both tables are 2D (delta,mturn) due to reionisaiton feedback
    // otherwise, the Nion table is 1D, since reionsaiton feedback is only active with minihalos
    if (astro_options_global->USE_MINI_HALOS) {
        if (prev) {
            table_2d = &Nion_conditional_table_prev;
            table_mini = &Nion_conditional_table_MINI_prev;
        } else {
            table_2d = &Nion_conditional_table2D;
            table_mini = &Nion_conditional_table_MINI;
        }
        if (!table_2d->allocated) {
            allocate_RGTable2D_f(NDELTA, NMTURN, table_2d);
        }
        if (!table_mini->allocated) {
            allocate_RGTable2D_f(NDELTA, NMTURN, table_mini);
        }
        table_2d->x_min = min_density;
        table_2d->x_width = (max_density - min_density) / (NDELTA - 1.);
        table_2d->y_min = log10Mturn_min;
        table_2d->y_width = (log10Mturn_max - log10Mturn_min) / (NMTURN - 1.);

        table_mini->x_min = min_density;
        table_mini->x_width = (max_density - min_density) / (NDELTA - 1.);
        table_mini->y_min = log10Mturn_min_MINI;
        table_mini->y_width = (log10Mturn_max_MINI - log10Mturn_min_MINI) / (NMTURN - 1.);
    } else {
        if (!Nion_conditional_table1D.allocated) {
            allocate_RGTable1D_f(NDELTA, &Nion_conditional_table1D);
        }
        Nion_conditional_table1D.x_min = min_density;
        Nion_conditional_table1D.x_width = (max_density - min_density) / (NDELTA - 1.);
    }

    for (i = 0; i < NDELTA; i++) {
        overdense_table[i] =
            min_density + (float)i / ((float)NDELTA - 1.) * (max_density - min_density);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        for (i = 0; i < NMTURN; i++) {
            mturns[i] = pow(10., log10Mturn_min + (float)i / ((float)NMTURN - 1.) *
                                                      (log10Mturn_max - log10Mturn_min));
            mturns_MINI[i] =
                pow(10., log10Mturn_min_MINI + (float)i / ((float)NMTURN - 1.) *
                                                   (log10Mturn_max_MINI - log10Mturn_min_MINI));
        }
    }

#pragma omp parallel private(i, j) num_threads(simulation_options_global -> N_THREADS)
    {
#pragma omp for
        for (i = 0; i < NDELTA; i++) {
            if (!astro_options_global->USE_MINI_HALOS) {
                // pass constant M_turn as minimum
                Nion_conditional_table1D.y_arr[i] = log(Nion_ConditionalM(
                    growthf, lnMmin, lnMmax, log(Mcond), sigma2, overdense_table[i],
                    sc->mturn_a_nofb, sc, astro_options_global->INTEGRATION_METHOD_ATOMIC));
                if (Nion_conditional_table1D.y_arr[i] < -40.)
                    Nion_conditional_table1D.y_arr[i] = -40.;

                continue;
            }
            for (j = 0; j < NMTURN; j++) {
                table_2d->z_arr[i][j] = log(Nion_ConditionalM(
                    growthf, lnMmin, lnMmax, log(Mcond), sigma2, overdense_table[i], mturns[j], sc,
                    astro_options_global->INTEGRATION_METHOD_ATOMIC));

                if (table_2d->z_arr[i][j] < -40.) table_2d->z_arr[i][j] = -40.;

                table_mini->z_arr[i][j] = log(Nion_ConditionalM_MINI(
                    growthf, lnMmin, lnMmax, log(Mcond), sigma2, overdense_table[i], mturns_MINI[j],
                    sc, astro_options_global->INTEGRATION_METHOD_MINI));

                if (table_mini->z_arr[i][j] < -40.) table_mini->z_arr[i][j] = -40.;
            }
        }
    }

    for (i = 0; i < NDELTA; i++) {
        if (!astro_options_global->USE_MINI_HALOS) {
            if (isfinite(Nion_conditional_table1D.y_arr[i]) == 0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline_1D");
                Throw(TableGenerationError);
            }
            continue;
        }
        for (j = 0; j < NMTURN; j++) {
            if (isfinite(table_2d->z_arr[i][j]) == 0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline");
                Throw(TableGenerationError);
            }

            if (isfinite(table_2d->z_arr[i][j]) == 0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline_MINI");
                Throw(TableGenerationError);
            }
        }
    }
}

// since SFRD is not used in Ionisationbox, and reionisation feedback is not included in the Ts
// calculation,
//     The non-minihalo table is always Rx1D and the minihalo table is always Rx2D

// This function initialises one table, for table Rx arrays I will call this function in a loop
void initialise_SFRD_Conditional_table(double z, double min_density, double max_density,
                                       double Mmin, double Mmax, double Mcond,
                                       struct ScalingConstants *sc) {
    float sigma2;
    int i, k;

    LOG_SUPER_DEBUG("Initialising SFRD conditional table at mass %.2e from delta %.2e to %.2e",
                    Mcond, min_density, max_density);

    double lnM_condition = log(Mcond);
    double lnMmin = log(Mmin);
    double lnMmax = log(Mmax);
    sigma2 = EvaluateSigma(
        lnM_condition);  // sigma is always the condition, whereas lnMmax is just the integral limit
    double growthf = dicke(z);

    float MassTurnover[NMTURN];
    for (i = 0; i < NMTURN; i++) {
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i / ((float)NMTURN - 1.) *
                                                         (LOG10_MTURN_MAX - LOG10_MTURN_MIN));
    }

    // NOTE: Here we use the constant Mturn limits instead of variables like in the Nion tables
    if (!SFRD_conditional_table.allocated) {
        allocate_RGTable1D_f(NDELTA, &SFRD_conditional_table);
    }
    SFRD_conditional_table.x_min = min_density;
    SFRD_conditional_table.x_width = (max_density - min_density) / (NDELTA - 1.);

    if (astro_options_global->USE_MINI_HALOS) {
        if (!SFRD_conditional_table_MINI.allocated) {
            allocate_RGTable2D_f(NDELTA, NMTURN, &SFRD_conditional_table_MINI);
        }
        SFRD_conditional_table_MINI.x_min = min_density;
        SFRD_conditional_table_MINI.x_width = (max_density - min_density) / (NDELTA - 1.);
        SFRD_conditional_table_MINI.y_min = LOG10_MTURN_MIN;
        SFRD_conditional_table_MINI.y_width = (LOG10_MTURN_MAX - LOG10_MTURN_MIN) / (NMTURN - 1.);
    }

    struct ScalingConstants sc_sfrd = evolve_scaling_constants_sfr(sc);

#pragma omp parallel private(i, k) num_threads(simulation_options_global -> N_THREADS)
    {
        double curr_dens;
#pragma omp for
        for (i = 0; i < NDELTA; i++) {
            curr_dens = min_density + (float)i / ((float)NDELTA - 1.) * (max_density - min_density);
            SFRD_conditional_table.y_arr[i] = log(Nion_ConditionalM(
                growthf, lnMmin, lnMmax, lnM_condition, sigma2, curr_dens, sc_sfrd.mturn_a_nofb,
                &sc_sfrd, astro_options_global->INTEGRATION_METHOD_ATOMIC));

            if (SFRD_conditional_table.y_arr[i] < -50.) SFRD_conditional_table.y_arr[i] = -50.;

            if (!astro_options_global->USE_MINI_HALOS) continue;

            for (k = 0; k < NMTURN; k++) {
                SFRD_conditional_table_MINI.z_arr[i][k] = log(Nion_ConditionalM_MINI(
                    growthf, lnMmin, lnMmax, lnM_condition, sigma2, curr_dens, MassTurnover[k],
                    &sc_sfrd, astro_options_global->INTEGRATION_METHOD_MINI));

                if (SFRD_conditional_table_MINI.z_arr[i][k] < -50.)
                    SFRD_conditional_table_MINI.z_arr[i][k] = -50.;
            }
        }
    }
    for (i = 0; i < NDELTA; i++) {
        if (isfinite(SFRD_conditional_table.y_arr[i]) == 0) {
            LOG_ERROR("Detected either an infinite or NaN value in ACG SFRD conditional table");
            Throw(TableGenerationError);
        }
        if (!astro_options_global->USE_MINI_HALOS) continue;

        for (k = 0; k < NMTURN; k++) {
            if (isfinite(SFRD_conditional_table_MINI.z_arr[i][k]) == 0) {
                LOG_ERROR("Detected either an infinite or NaN value in MCG SFRD conditional table");
                Throw(TableGenerationError);
            }
        }
    }
}

// This function initialises one table, for table Rx arrays I will call this function in a loop
void initialise_Xray_Conditional_table(double redshift, double min_density, double max_density,
                                       double Mmin, double Mmax, double Mcond,
                                       struct ScalingConstants *sc) {
    int i, k;

    LOG_SUPER_DEBUG("Initialising Xray conditional table at mass %.2e from delta %.2e to %.2e",
                    Mcond, min_density, max_density);

    double lnM_condition = log(Mcond);
    double growthf = dicke(redshift);
    double lnMmin = log(Mmin);
    double lnMmax = log(Mmax);
    // sigma is always the condition, whereas lnMmax is just the integral limit
    double sigma2 = EvaluateSigma(lnM_condition);

    float MassTurnover[NMTURN];
    for (i = 0; i < NMTURN; i++) {
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i / ((float)NMTURN - 1.) *
                                                         (LOG10_MTURN_MAX - LOG10_MTURN_MIN));
    }

    // NOTE: Like the SFRD tables we ignore reionisation feedback
    if (astro_options_global->USE_MINI_HALOS) {
        if (!Xray_conditional_table_2D.allocated) {
            allocate_RGTable2D_f(NDELTA, NMTURN, &Xray_conditional_table_2D);
        }
        Xray_conditional_table_2D.x_min = min_density;
        Xray_conditional_table_2D.x_width = (max_density - min_density) / (NDELTA - 1.);
        Xray_conditional_table_2D.y_min = LOG10_MTURN_MIN;
        Xray_conditional_table_2D.y_width = (LOG10_MTURN_MAX - LOG10_MTURN_MIN) / (NMTURN - 1.);
    } else {
        if (!Xray_conditional_table_1D.allocated) {
            allocate_RGTable1D_f(NDELTA, &Xray_conditional_table_1D);
        }
        Xray_conditional_table_1D.x_min = min_density;
        Xray_conditional_table_1D.x_width = (max_density - min_density) / (NDELTA - 1.);
    }

#pragma omp parallel private(i, k) num_threads(simulation_options_global -> N_THREADS)
    {
        double curr_dens;
#pragma omp for
        for (i = 0; i < NDELTA; i++) {
            curr_dens = min_density + (float)i / ((float)NDELTA - 1.) * (max_density - min_density);
            if (!astro_options_global->USE_MINI_HALOS) {
                Xray_conditional_table_1D.y_arr[i] = log(Xray_ConditionalM(
                    redshift, growthf, lnMmin, lnMmax, lnM_condition, sigma2, curr_dens,
                    sc->mturn_a_nofb, 0., sc, astro_options_global->INTEGRATION_METHOD_ATOMIC));

                if (Xray_conditional_table_1D.y_arr[i] < -50.)
                    Xray_conditional_table_1D.y_arr[i] = -50.;
                continue;
            }

            for (k = 0; k < NMTURN; k++) {
                // Using mini integration method for both
                Xray_conditional_table_2D.z_arr[i][k] =
                    log(Xray_ConditionalM(redshift, growthf, lnMmin, lnMmax, lnM_condition, sigma2,
                                          curr_dens, sc->mturn_a_nofb, MassTurnover[k], sc,
                                          astro_options_global->INTEGRATION_METHOD_MINI));

                if (Xray_conditional_table_2D.z_arr[i][k] < -50.)
                    Xray_conditional_table_2D.z_arr[i][k] = -50.;
            }
        }
    }
    for (i = 0; i < NDELTA; i++) {
        if (!astro_options_global->USE_MINI_HALOS) {
            if (isfinite(Xray_conditional_table_1D.y_arr[i]) == 0) {
                LOG_ERROR("Detected either an infinite or NaN value in 1D Xray conditional table");
                Throw(TableGenerationError);
            }
            continue;
        }
        for (k = 0; k < NMTURN; k++) {
            if (isfinite(Xray_conditional_table_2D.z_arr[i][k]) == 0) {
                LOG_ERROR("Detected either an infinite or NaN value in 2D Xray conditional table");
                Throw(TableGenerationError);
            }
        }
    }
}

void initialise_dNdM_tables(double xmin, double xmax, double ymin, double ymax, double growth_out,
                            double param, bool from_catalog) {
    int nx;
    double lnM_cond = 0.;
    double sigma_cond = 0.;
    LOG_SUPER_DEBUG("Initialising dNdM Tables from [%.2e,%.2e] (Intg. Limits %.2e %.2e)", xmin,
                    xmax, ymin, ymax);
    LOG_SUPER_DEBUG("D_out %.2e P %.2e from_cat %d", growth_out, param, from_catalog);

    if (!from_catalog) {
        lnM_cond = param;
        sigma_cond = EvaluateSigma(lnM_cond);
    }

    nx = simulation_options_global->N_COND_INTERP;

    double xa[nx];
    int i;
    // set up coordinate grids
    for (i = 0; i < nx; i++) xa[i] = xmin + (xmax - xmin) * ((double)i) / ((double)nx - 1);

    // allocate tables
    if (!Nhalo_table.allocated) allocate_RGTable1D(nx, &Nhalo_table);

    Nhalo_table.x_min = xmin;
    Nhalo_table.x_width = (xmax - xmin) / ((double)nx - 1);

    if (!Mcoll_table.allocated) allocate_RGTable1D(nx, &Mcoll_table);

    Mcoll_table.x_min = xmin;
    Mcoll_table.x_width = (xmax - xmin) / ((double)nx - 1);

#pragma omp parallel num_threads(simulation_options_global->N_THREADS) private(i) \
    firstprivate(sigma_cond, lnM_cond)
    {
        double x;
        double delta;

#pragma omp for
        for (i = 0; i < nx; i++) {
            x = xa[i];
            // set the condition
            if (from_catalog) {
                lnM_cond = x;
                sigma_cond = EvaluateSigma(lnM_cond);
                // barrier at descendant mass
                delta = get_delta_crit(matter_options_global->HMF, sigma_cond, param) / param *
                        growth_out;
            } else {
                delta = x;
            }

            if (i == nx - 1) {
                LOG_INFO(
                    "Last bin in NhaloConditional: growth %.6g ymin %.6g ymax %.6g "
                    "lnM_cond %.6g sigma_cond %.6g delta %.6g result %.6g",
                    growth_out, ymin, ymax, lnM_cond, sigma_cond, delta,
                    Nhalo_Conditional(growth_out, ymin, ymax, lnM_cond, sigma_cond, delta, 0));
            }
            Nhalo_table.y_arr[i] =
                Nhalo_Conditional(growth_out, ymin, ymax, lnM_cond, sigma_cond, delta, 0);
            Mcoll_table.y_arr[i] =
                Mcoll_Conditional(growth_out, ymin, ymax, lnM_cond, sigma_cond, delta, 0);
        }
    }
    LOG_SUPER_DEBUG("Done.");
}

struct rf_inv_params {
    double growthf;
    double lnM_cond;
    double M_cond;
    double delta;
    double sigma;

    double rf_norm;
    double rf_target;
};

double dndm_inv_f(double lnM_min, void *params) {
    struct rf_inv_params *p = (struct rf_inv_params *)params;
    double integral =
        Nhalo_Conditional(p->growthf, lnM_min, p->lnM_cond, p->lnM_cond, p->sigma, p->delta, 0);
    // This ensures that we never find the root if the ratio is zero, since that will set to M_cond
    double result =
        integral == 0 ? 2 * simulation_options_global->MIN_LOGPROB : log(integral / p->rf_norm);

    return result - p->rf_target;
}

// This table is N(>M | M_in), the CDF of dNdM_conditional
// NOTE: Assumes you give it ymin as the minimum lower-integral limit, and ymax as the maximum
//  `param` is either the constant log condition mass for the grid case (!from_catalog) OR the
//  descendant growth factor with from_catalog
void initialise_dNdM_inverse_table(double xmin, double xmax, double lnM_min, double growth_out,
                                   double param, bool from_catalog) {
    LOG_SUPER_DEBUG("Initialising dNdM Tables from [%.2e,%.2e] (Intg. Min. %.2e)", xmin, xmax,
                    lnM_min);
    LOG_SUPER_DEBUG("D_out %.2e P %.2e up %d", growth_out, param, from_catalog);

    int nx = simulation_options_global->N_COND_INTERP;
    int np = simulation_options_global->N_PROB_INTERP;
    double xa[nx], pa[np];
    double rf_tol_abs = 1e-4;
    double rf_tol_rel = 0.;

    double lnM_cond = 0.;
    double sigma_cond = 0.;
    double min_lp = simulation_options_global->MIN_LOGPROB;
    if (!from_catalog) {
        lnM_cond = param;
        sigma_cond = EvaluateSigma(lnM_cond);
    }

    int i, k;
    // set up coordinate grids
    for (i = 0; i < nx; i++) xa[i] = xmin + (xmax - xmin) * ((double)i) / ((double)nx - 1);
    // avoiding floating point errors in final bin due to the hard boundary at Deltac
    xa[nx - 1] = xmax;
    for (k = 0; k < np; k++) {
        pa[k] = min_lp * (1 - (double)k / (double)(np - 1));
    }

    if (!Nhalo_inv_table.allocated) allocate_RGTable2D(nx, np, &Nhalo_inv_table);

    Nhalo_inv_table.x_min = xmin;
    Nhalo_inv_table.x_width = xa[1] - xa[0];
    Nhalo_inv_table.y_min = pa[0];
    Nhalo_inv_table.y_width = pa[1] - pa[0];

#pragma omp parallel num_threads(simulation_options_global->N_THREADS) private(i, k) \
    firstprivate(sigma_cond, lnM_cond)
    {
        double x;
        double norm;
        double lnM_lo, lnM_hi, lnM_guess;
        double delta;
        double M_cond;
        double lnM_init;

        // RF stuff
        int status, iter;
        const gsl_root_fsolver_type *T;
        gsl_root_fsolver *solver;
        gsl_function F;
        struct rf_inv_params params_rf;
        params_rf.growthf = growth_out;

        F.function = &dndm_inv_f;
        F.params = &params_rf;

        T = gsl_root_fsolver_brent;
        solver = gsl_root_fsolver_alloc(T);

#pragma omp for
        for (i = 0; i < nx; i++) {
            x = xa[i];
            // set the condition
            if (from_catalog) {
                lnM_cond = x;
                sigma_cond = EvaluateSigma(lnM_cond);
                // Barrier at descendant mass scaled to progenitor redshift
                delta = get_delta_crit(matter_options_global->HMF, sigma_cond, param) / param *
                        growth_out;
            } else {
                delta = x;
                if (delta > MAX_DELTAC_FRAC * get_delta_crit(matter_options_global->HMF, sigma_cond,
                                                             growth_out)) {
                    for (k = 1; k < np; k++) Nhalo_inv_table.z_arr[i][k] = 1.;
                    continue;
                }
            }

            M_cond = exp(lnM_cond);

            params_rf.M_cond = M_cond;
            params_rf.lnM_cond = lnM_cond;
            params_rf.delta = delta;
            params_rf.sigma = sigma_cond;

            // NOTE: The total number density and collapsed fraction must be
            norm = Nhalo_Conditional(growth_out, lnM_min, lnM_cond, lnM_cond, sigma_cond, delta, 0);
            // LOG_ULTRA_DEBUG("cond x: %.2e M_min %.2e M_cond %.2e d %.4f D %.2f n %d ==>
            // %.8e",x,exp(lnM_min),exp(lnM_cond),delta,growth_out,i,norm);
            params_rf.rf_norm = norm;

            // if the condition has no halos set the dndm table directly to avoid integration and
            // divide by zero
            if (norm == 0) {
                for (k = 1; k < np - 1; k++) Nhalo_inv_table.z_arr[i][k] = exp(lnM_min) / M_cond;
                continue;
            }

            Nhalo_inv_table.z_arr[i][np - 1] = exp(lnM_min) / M_cond;
            lnM_init = lnM_min;
            for (k = np - 2; k >= 0; k--) {
                iter = 0;
                params_rf.rf_target = pa[k];
                // LOG_ULTRA_DEBUG("Target %.6e",pa[k]);
                gsl_root_fsolver_set(solver, &F, lnM_init, lnM_cond);
                do {
                    iter++;
                    status = gsl_root_fsolver_iterate(solver);
                    lnM_guess = gsl_root_fsolver_root(solver);
                    lnM_lo = gsl_root_fsolver_x_lower(solver);
                    lnM_hi = gsl_root_fsolver_x_upper(solver);
                    status = gsl_root_test_interval(lnM_lo, lnM_hi, rf_tol_abs, rf_tol_rel);

                    // LOG_ULTRA_DEBUG("Current step %d | [%.6e,%.6e] -
                    // %.6e",iter,lnM_lo,lnM_hi,lnM_guess);

                    if (status == GSL_SUCCESS) {
                        lnM_init = lnM_lo;
                        Nhalo_inv_table.z_arr[i][k] = exp(lnM_guess) / M_cond;
                        // LOG_ULTRA_DEBUG("Found (M %.2e d %.2f p %.3e) %.6e -->
                        // %.6e",M_cond,delta,exp(pa[k]),lnM_guess,exp(lnM_guess)/M_cond);
                    }

                } while ((status == GSL_CONTINUE) && (iter < MAX_ITER_RF));
                if (status != GSL_SUCCESS) {
                    LOG_ERROR("gsl RF error occured! %d", status);
                    CATCH_GSL_ERROR(status);
                }
            }
        }
        gsl_root_fsolver_free(solver);
    }
    LOG_SUPER_DEBUG("Done.");
}

double J_integrand(double u, void *params) {
    double gamma1 = *(double *)params;
    return pow((1. + 1. / u / u), gamma1 * 0.5);
}

double integrate_J(double u_res, double gamma1) {
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 1e-4;  //<- relative tolerance
    int w_size = 1000;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);

    int status;
    F.function = &J_integrand;
    F.params = &gamma1;
    lower_limit = 0.;
    upper_limit = u_res;

    gsl_set_error_handler_off();
    status = gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol, w_size,
                                 GSL_INTEG_GAUSS61, w, &result, &error);

    if (status != 0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("J: gamma1 = %.4e u_res = %.4e", gamma1, u_res);
        CATCH_GSL_ERROR(status);
    }

    gsl_integration_workspace_free(w);

    return result;
}

void initialise_J_split_table(int Nbin, double umin, double umax, double gamma1) {
    int i;
    if (!J_split_table.allocated) allocate_RGTable1D(Nbin, &J_split_table);

    J_split_table.x_min = 1e-3;
    J_split_table.x_width = (umax - umin) / ((double)Nbin - 1);

    for (i = 0; i < Nbin; i++) {
        J_split_table.y_arr[i] =
            integrate_J(J_split_table.x_min + i * J_split_table.x_width, gamma1);
    }
}

void free_dNdM_tables() {
    free_RGTable2D(&Nhalo_inv_table);
    free_RGTable1D(&Nhalo_table);
    free_RGTable1D(&Mcoll_table);
    free_RGTable1D(&J_split_table);
    if (matter_options_global->SAMPLE_METHOD == 2) {
        gsl_spline_free(Sigma_inv_table);
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
            gsl_interp_accel_free(Sigma_inv_table_acc);
        }
    }
}

void free_conditional_tables() {
    free_RGTable1D_f(&fcoll_conditional_table);
    free_RGTable1D_f(&dfcoll_conditional_table);
    free_RGTable1D_f(&SFRD_conditional_table);
    free_RGTable2D_f(&SFRD_conditional_table_MINI);
    free_RGTable1D_f(&Nion_conditional_table1D);
    free_RGTable2D_f(&Nion_conditional_table2D);
    free_RGTable2D_f(&Nion_conditional_table_MINI);
    free_RGTable2D_f(&Nion_conditional_table_prev);
    free_RGTable2D_f(&Nion_conditional_table_MINI_prev);
    free_RGTable1D_f(&Xray_conditional_table_1D);
    free_RGTable2D_f(&Xray_conditional_table_2D);
}

void free_global_tables() {
    free_RGTable1D(&SFRD_z_table);
    free_RGTable2D(&SFRD_z_table_MINI);
    free_RGTable1D(&Nion_z_table);
    free_RGTable2D(&Nion_z_table_MINI);
    free_RGTable1D(&fcoll_z_table);
    free_RGTable1D(&Xray_z_table_1D);
    free_RGTable2D(&Xray_z_table_2D);
}

// JD: moving the interp table evaluations here since some of them are needed in nu_tau_one
// NOTE: with !USE_MASS_DEPENDENT_ZETA both EvaluateNionTs and EvaluateSFRD return Fcoll
double EvaluateNionTs(double redshift, struct ScalingConstants *sc) {
    // differences in turnover are handled by table setup
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        if (astro_options_global->USE_MASS_DEPENDENT_ZETA)
            return EvaluateRGTable1D(redshift, &Nion_z_table);
        return EvaluateRGTable1D(redshift, &fcoll_z_table);
    }

    // Currently assuming this is only called in the X-ray/spintemp calculation, this will only
    // affect !USE_MASS_DEPENDENT_ZETA and !M_MIN_in_mass and only if the minimum virial
    // temperatures ION_Tvir_min and X_RAY_Tvir_min are different
    double lnMmin = log(minimum_source_mass(redshift, true));
    double lnMmax = log(M_MAX_INTEGRAL);

    struct ScalingConstants sc_z = evolve_scaling_constants_to_redshift(redshift, sc, false);

    // minihalos uses a different turnover mass
    if (astro_options_global->USE_MASS_DEPENDENT_ZETA)
        return Nion_General(redshift, lnMmin, lnMmax, sc_z.mturn_a_nofb, &sc_z);

    return Fcoll_General(redshift, lnMmin, lnMmax);
}

double EvaluateNionTs_MINI(double redshift, double log10_Mturn_LW_ave,
                           struct ScalingConstants *sc) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        return EvaluateRGTable2D(redshift, log10_Mturn_LW_ave, &Nion_z_table_MINI);
    }
    double lnMmin = log(minimum_source_mass(redshift, true));
    double lnMmax = log(M_MAX_INTEGRAL);
    struct ScalingConstants sc_z = evolve_scaling_constants_to_redshift(redshift, sc, false);

    return Nion_General_MINI(redshift, lnMmin, lnMmax, pow(10., log10_Mturn_LW_ave), &sc_z);
}

double EvaluateSFRD(double redshift, struct ScalingConstants *sc) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        if (astro_options_global->USE_MASS_DEPENDENT_ZETA)
            return EvaluateRGTable1D(redshift, &SFRD_z_table);
        return EvaluateRGTable1D(redshift, &fcoll_z_table);
    }

    // Currently assuming this is only called in the X-ray/spintemp calculation, this will only
    // affect !USE_MASS_DEPENDENT_ZETA and !M_MIN_in_mass and only if the minimum virial
    // temperatures ION_Tvir_min and X_RAY_Tvir_min are different
    double lnMmin = log(minimum_source_mass(redshift, true));
    double lnMmax = log(M_MAX_INTEGRAL);

    // The SFRD calls the same function as N_ion but sets escape fractions to unity
    // NOTE: since this only occurs on integration, the struct copy shouldn't be a bottleneck
    struct ScalingConstants sc_sfrd = evolve_scaling_constants_sfr(sc);
    sc_sfrd = evolve_scaling_constants_to_redshift(redshift, &sc_sfrd, false);

    if (astro_options_global->USE_MASS_DEPENDENT_ZETA)
        return Nion_General(redshift, lnMmin, lnMmax, sc_sfrd.mturn_a_nofb, &sc_sfrd);
    return Fcoll_General(redshift, lnMmin, lnMmax);
}

double EvaluateSFRD_MINI(double redshift, double log10_Mturn_LW_ave, struct ScalingConstants *sc) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        return EvaluateRGTable2D(redshift, log10_Mturn_LW_ave, &SFRD_z_table_MINI);
    }

    double lnMmin = log(minimum_source_mass(redshift, true));
    double lnMmax = log(M_MAX_INTEGRAL);

    struct ScalingConstants sc_sfrd = evolve_scaling_constants_sfr(sc);
    sc_sfrd = evolve_scaling_constants_to_redshift(redshift, &sc_sfrd, false);

    return Nion_General_MINI(redshift, lnMmin, lnMmax, pow(10., log10_Mturn_LW_ave), &sc_sfrd);
}

double EvaluateSFRD_Conditional(double delta, double growthf, double M_min, double M_max,
                                double M_cond, double sigma_max, struct ScalingConstants *sc) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        return exp(EvaluateRGTable1D_f(delta, &SFRD_conditional_table));
    }

    struct ScalingConstants sc_sfrd = evolve_scaling_constants_sfr(sc);
    // SFRD in Ts assumes no (reion) feedback on ACG
    return Nion_ConditionalM(growthf, log(M_min), log(M_max), log(M_cond), sigma_max, delta,
                             sc_sfrd.mturn_a_nofb, &sc_sfrd,
                             astro_options_global->INTEGRATION_METHOD_ATOMIC);
}

double EvaluateSFRD_Conditional_MINI(double delta, double log10Mturn_m, double growthf,
                                     double M_min, double M_max, double M_cond, double sigma_max,
                                     struct ScalingConstants *sc) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        return exp(EvaluateRGTable2D_f(delta, log10Mturn_m, &SFRD_conditional_table_MINI));
    }

    struct ScalingConstants sc_sfrd = evolve_scaling_constants_sfr(sc);
    return Nion_ConditionalM_MINI(growthf, log(M_min), log(M_max), log(M_cond), sigma_max, delta,
                                  pow(10, log10Mturn_m), &sc_sfrd,
                                  astro_options_global->INTEGRATION_METHOD_MINI);
}

double EvaluateNion_Conditional(double delta, double log10Mturn, double growthf, double M_min,
                                double M_max, double M_cond, double sigma_max,
                                struct ScalingConstants *sc, bool prev) {
    RGTable2D_f *table = prev ? &Nion_conditional_table_prev : &Nion_conditional_table2D;
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        if (astro_options_global->USE_MINI_HALOS)
            return exp(EvaluateRGTable2D_f(delta, log10Mturn, table));
        return exp(EvaluateRGTable1D_f(delta, &Nion_conditional_table1D));
    }

    // NOTE: turning minihalos off turns off feedback in the model. This may be slightly misleading
    //   to ignore a passed parameter but until we make the change in the model we force it here
    double mturn = astro_options_global->USE_MINI_HALOS ? pow(10, log10Mturn) : sc->mturn_a_nofb;
    return Nion_ConditionalM(growthf, log(M_min), log(M_max), log(M_cond), sigma_max, delta, mturn,
                             sc, astro_options_global->INTEGRATION_METHOD_ATOMIC);
}

double EvaluateNion_Conditional_MINI(double delta, double log10Mturn_m, double growthf,
                                     double M_min, double M_max, double M_cond, double sigma_max,
                                     struct ScalingConstants *sc, bool prev) {
    RGTable2D_f *table = prev ? &Nion_conditional_table_MINI_prev : &Nion_conditional_table_MINI;
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        return exp(EvaluateRGTable2D_f(delta, log10Mturn_m, table));
    }

    return Nion_ConditionalM_MINI(growthf, log(M_min), log(M_max), log(M_cond), sigma_max, delta,
                                  pow(10, log10Mturn_m), sc,
                                  astro_options_global->INTEGRATION_METHOD_MINI);
}

double EvaluateXray_Conditional(double delta, double log10Mturn_m, double redshift, double growthf,
                                double M_min, double M_max, double M_cond, double sigma_max,
                                struct ScalingConstants *sc) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        if (astro_options_global->USE_MINI_HALOS)
            return exp(EvaluateRGTable2D_f(delta, log10Mturn_m, &Xray_conditional_table_2D));
        return exp(EvaluateRGTable1D_f(delta, &Xray_conditional_table_1D));
    }

    // TODO: I shouldn't need to pass both redshift and growthf here
    // NOTE: same as SFRD, we assume no feedback on ACGs
    return Xray_ConditionalM(redshift, growthf, log(M_min), log(M_max), log(M_cond), sigma_max,
                             delta, sc->mturn_a_nofb, pow(10, log10Mturn_m), sc,
                             astro_options_global->INTEGRATION_METHOD_MINI);
}

double EvaluateFcoll_delta(double delta, double growthf, double sigma_min, double sigma_max) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        return EvaluateRGTable1D_f(delta, &fcoll_conditional_table);
    }

    return FgtrM_bias_fast(growthf, delta, sigma_min, sigma_max);
}
double EvaluatedFcolldz(double delta, double redshift, double sigma_min, double sigma_max) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        return EvaluateRGTable1D_f(delta, &dfcoll_conditional_table);
    }
    return dfcoll_dz(redshift, sigma_min, delta, sigma_max);
}

double EvaluateNhalo(double condition, double growthf, double lnMmin, double lnMmax, double M_cond,
                     double sigma, double delta) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1 &&
        (1.427450 < condition && condition < 1.427455)) {
        int idx = (int)floor((condition - Nhalo_table.x_min) / Nhalo_table.x_width);
        double table_val = Nhalo_table.x_min + Nhalo_table.x_width * (double)idx;
        double interp_point = (condition - table_val) / Nhalo_table.x_width;
        // a + f(a-b) is one fewer operation but less precise
        double result = Nhalo_table.y_arr[idx] * (1 - interp_point) +
                        Nhalo_table.y_arr[idx + 1] * (interp_point);
        LOG_INFO(
            "Extrap for Nhalo at condition %.6e, result %.6e, idx %d, table_val %.6g interppoint "
            "%.6g lower %.6e upper %.6e upperer %.6e anl %.6e ntable %d",
            condition, result, idx, table_val, interp_point, Nhalo_table.y_arr[idx],
            Nhalo_table.y_arr[idx + 1], Nhalo_table.y_arr[idx + 2],
            Nhalo_Conditional(growthf, lnMmin, lnMmax, log(M_cond), sigma, delta, 0),
            Nhalo_table.n_bin);
    }
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1)
        return EvaluateRGTable1D(condition, &Nhalo_table);
    return Nhalo_Conditional(growthf, lnMmin, lnMmax, log(M_cond), sigma, delta, 0);
}

double EvaluateMcoll(double condition, double growthf, double lnMmin, double lnMmax, double M_cond,
                     double sigma, double delta) {
    if (matter_options_global->USE_INTERPOLATION_TABLES > 1)
        return EvaluateRGTable1D(condition, &Mcoll_table);
    return Mcoll_Conditional(growthf, lnMmin, lnMmax, log(M_cond), sigma, delta, 0);
}

// extrapolation function for log-probability based tables
// NOTE: this is very similar to the EvaluateRGTableX function,
//   it may be worth allowing extrapolation there by simply setting the indices to the min/max
double extrapolate_dNdM_inverse(double condition, double lnp) {
    double x_min = Nhalo_inv_table.x_min;
    double x_width = Nhalo_inv_table.x_width;
    int x_idx = (int)floor((condition - x_min) / x_width);
    double x_table = x_min + x_idx * x_width;
    double interp_point_x = (condition - x_table) / x_width;

    double extrap_point_y =
        (lnp - simulation_options_global->MIN_LOGPROB) / Nhalo_inv_table.y_width;

    // find the log-mass at the edge of the table for this condition
    double xlimit = Nhalo_inv_table.z_arr[x_idx][0] * (interp_point_x) +
                    Nhalo_inv_table.z_arr[x_idx + 1][0] * (1 - interp_point_x);
    double xlimit_m1 = Nhalo_inv_table.z_arr[x_idx][1] * (interp_point_x) +
                       Nhalo_inv_table.z_arr[x_idx + 1][1] * (1 - interp_point_x);

    double result = xlimit + (xlimit_m1 - xlimit) * (extrap_point_y);

    return result;
}

// This one is always a table
double EvaluateNhaloInv(double condition, double prob) {
    double lnp = log(prob);
    if (prob == 0 || lnp < simulation_options_global->MIN_LOGPROB)
        lnp = simulation_options_global->MIN_LOGPROB;
    // if(prob < simulation_options_global->MIN_LOGPROB)
    // return extrapolate_dNdM_inverse(condition,lnp);
    return EvaluateRGTable2D(condition, lnp, &Nhalo_inv_table);
}

double EvaluateJ(double u_res, double gamma1) {
    if (fabs(gamma1) < FRACT_FLOAT_ERR) return u_res;
    // small u approximation
    if (u_res < J_split_table.x_min) return pow(u_res, 1. - gamma1) / (1. - gamma1);
    double u_max = J_split_table.x_min + (J_split_table.n_bin - 1) * J_split_table.x_width;
    // asymptotic expansion
    if (u_res > u_max)
        return J_split_table.y_arr[J_split_table.n_bin - 1] + u_res -
               0.5 * gamma1 * (1. / u_res - 1. / u_max);
    return EvaluateRGTable1D(u_res, &J_split_table);
}

void InitialiseSigmaInverseTable() {
    if (!Sigma_InterpTable.allocated) {
        LOG_ERROR("Must construct the sigma table before the inverse table");
        Throw(TableGenerationError);
    }
    int i;
    int n_bin = Sigma_InterpTable.n_bin;

    double xa[n_bin], ya[n_bin];

    Sigma_inv_table = gsl_spline_alloc(gsl_interp_linear, n_bin);
    Sigma_inv_table_acc = gsl_interp_accel_alloc();

    for (i = 0; i < n_bin; i++) {
        xa[i] = Sigma_InterpTable.y_arr[n_bin - i - 1];
        ya[i] = Sigma_InterpTable.x_min + (n_bin - i - 1) * Sigma_InterpTable.x_width;
    }

    gsl_spline_init(Sigma_inv_table, xa, ya, n_bin);
}

double EvaluateSigmaInverse(double sigma) {
    if (!(matter_options_global->USE_INTERPOLATION_TABLES > 0)) {
        LOG_ERROR("Cannot currently do sigma inverse without USE_INTERPOLATION_TABLES");
        Throw(ValueError);
    }
    return gsl_spline_eval(Sigma_inv_table, sigma, Sigma_inv_table_acc);
}

void initialiseSigmaMInterpTable(float M_min, float M_max) {
    int i;

    if (!Sigma_InterpTable.allocated) allocate_RGTable1D_f(N_MASS_INTERP, &Sigma_InterpTable);
    if (!dSigmasqdm_InterpTable.allocated)
        allocate_RGTable1D_f(N_MASS_INTERP, &dSigmasqdm_InterpTable);

    Sigma_InterpTable.x_min = log(M_min);
    Sigma_InterpTable.x_width = (log(M_max) - log(M_min)) / (N_MASS_INTERP - 1.);
    dSigmasqdm_InterpTable.x_min = log(M_min);
    dSigmasqdm_InterpTable.x_width = (log(M_max) - log(M_min)) / (N_MASS_INTERP - 1.);

#pragma omp parallel private(i) num_threads(simulation_options_global -> N_THREADS)
    {
        float Mass;
#pragma omp for
        for (i = 0; i < N_MASS_INTERP; i++) {
            Mass = exp(Sigma_InterpTable.x_min + i * Sigma_InterpTable.x_width);
            Sigma_InterpTable.y_arr[i] = sigma_z0(Mass);
            dSigmasqdm_InterpTable.y_arr[i] = log10(-dsigmasqdm_z0(Mass));
        }
    }

    for (i = 0; i < N_MASS_INTERP; i++) {
        if (isfinite(Sigma_InterpTable.y_arr[i]) == 0 ||
            isfinite(dSigmasqdm_InterpTable.y_arr[i]) == 0) {
            LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
            Throw(TableGenerationError);
        }
    }
}

void freeSigmaMInterpTable() {
    free_RGTable1D_f(&Sigma_InterpTable);
    free_RGTable1D_f(&dSigmasqdm_InterpTable);
}

double EvaluateSigma(double lnM) {
    // using log units to make the fast option faster and the slow option slower
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0) {
        return EvaluateRGTable1D_f(lnM, &Sigma_InterpTable);
    }
    return sigma_z0(exp(lnM));
}

double EvaluatedSigmasqdm(double lnM) {
    // this may be slow, figure out why the dsigmadm table is in log10
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0) {
        return -pow(10., EvaluateRGTable1D_f(lnM, &dSigmasqdm_InterpTable));
    }
    return dsigmasqdm_z0(exp(lnM));
}
