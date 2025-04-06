
// Re-write of update_halo_pos from the original 21cmFAST

// ComputePerturbHaloField reads in the linear velocity field, and uses
// it to update halo locations with a corresponding displacement field

#include "PerturbHaloField.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "exceptions.h"
#include "indexing.h"
#include "logger.h"

int ComputePerturbHaloField(float redshift, MatterParams *matter_params, MatterFlags *matter_flags,
                            CosmoParams *cosmo_params, InitialConditions *boxes, HaloField *halos,
                            PerturbHaloField *halos_perturbed) {
    int status;

    Try {  // This Try brackets the whole function, so we don't indent.

        LOG_DEBUG("input value:");
        LOG_DEBUG("redshift=%f", redshift);
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        writeMatterParams(matter_params);
        writeCosmoParams(cosmo_params);
        writeAstroParams(astro_flags, astro_params);
        writeAstroFlags(astro_flags);
#endif

        // Makes the parameter structs visible to a variety of functions/macros
        // Do each time to avoid Python garbage collection issues
        Broadcast_struct_global_noastro(matter_params, matter_flags, cosmo_params);

        omp_set_num_threads(matter_params->N_THREADS);

        float growth_factor, displacement_factor_2LPT, xf, yf, zf, growth_factor_over_BOX_LEN,
            displacement_factor_2LPT_over_BOX_LEN;
        unsigned long long int i, j, k, DI, dimension;
        unsigned long long i_halo;

        LOG_DEBUG("Begin Initialisation");

        // Function for deciding the dimensions of loops when we could
        // use either the low or high resolution grids.
        dimension = matter_flags->PERTURB_ON_HIGH_RES ? matter_params->DIM : matter_params->HII_DIM;

        // ***************** END INITIALIZATION ***************** //
        init_ps();
        growth_factor = dicke(redshift);  // normalized to 1 at z=0
        displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

        growth_factor_over_BOX_LEN = growth_factor / matter_params->BOX_LEN;
        displacement_factor_2LPT_over_BOX_LEN = displacement_factor_2LPT / matter_params->BOX_LEN;

        // now add the missing factor of Ddot to velocity field
#pragma omp parallel shared(boxes, dimension, growth_factor_over_BOX_LEN) private(i, j, k) \
    num_threads(matter_params -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < dimension; i++) {
                for (j = 0; j < dimension; j++) {
                    for (k = 0;
                         k < (unsigned long long)(matter_params->NON_CUBIC_FACTOR * dimension);
                         k++) {
                        if (matter_flags->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->hires_vy[R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->hires_vz[R_INDEX(i, j, k)] *=
                                (growth_factor_over_BOX_LEN / matter_params->NON_CUBIC_FACTOR);
                        } else {
                            boxes->lowres_vx[HII_R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->lowres_vy[HII_R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->lowres_vz[HII_R_INDEX(i, j, k)] *=
                                (growth_factor_over_BOX_LEN / matter_params->NON_CUBIC_FACTOR);
                        }
                        // this is now comoving displacement in units of box size
                    }
                }
            }
        }

        // ************************************************************************* //
        //                          BEGIN 2LPT PART                                  //
        // ************************************************************************* //

        // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        if (matter_flags->PERTURB_ALGORITHM == 2) {
            // now add the missing factor in eq. D9
#pragma omp parallel shared(boxes, displacement_factor_2LPT_over_BOX_LEN, dimension) private( \
        i, j, k) num_threads(matter_params -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < dimension; i++) {
                    for (j = 0; j < dimension; j++) {
                        for (k = 0;
                             k < (unsigned long long)(matter_params->NON_CUBIC_FACTOR * dimension);
                             k++) {
                            if (matter_flags->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vy_2LPT[R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vz_2LPT[R_INDEX(i, j, k)] *=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     matter_params->NON_CUBIC_FACTOR);
                            } else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i, j, k)] *=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     matter_params->NON_CUBIC_FACTOR);
                            }
                            // this is now comoving displacement in units of box size
                        }
                    }
                }
            }
        }

        // ************************************************************************* //
        //                            END 2LPT PART                                  //
        // ************************************************************************* //
        halos_perturbed->n_halos = halos->n_halos;

        // ******************   END INITIALIZATION     ******************************** //

#pragma omp parallel shared(boxes, halos, halos_perturbed) private(i_halo, i, j, k, xf, yf, zf) \
    num_threads(matter_params -> N_THREADS)
        {
#pragma omp for
            for (i_halo = 0; i_halo < halos->n_halos; i_halo++) {
                // convert location to fractional value
                xf = halos->halo_coords[i_halo * 3 + 0] / (matter_params->DIM + 0.);
                yf = halos->halo_coords[i_halo * 3 + 1] / (matter_params->DIM + 0.);
                zf = halos->halo_coords[i_halo * 3 + 2] / (D_PARA + 0.);

                // determine halo position (downsampled if required)
                if (matter_flags->PERTURB_ON_HIGH_RES) {
                    i = halos->halo_coords[i_halo * 3 + 0];
                    j = halos->halo_coords[i_halo * 3 + 1];
                    k = halos->halo_coords[i_halo * 3 + 2];
                } else {
                    i = xf * matter_params->HII_DIM;
                    j = yf * matter_params->HII_DIM;
                    k = zf * HII_D_PARA;
                }

                // get new positions using linear velocity displacement from z=INITIAL
                if (matter_flags->PERTURB_ON_HIGH_RES) {
                    xf += boxes->hires_vx[R_INDEX(i, j, k)];
                    yf += boxes->hires_vy[R_INDEX(i, j, k)];
                    zf += boxes->hires_vz[R_INDEX(i, j, k)];
                } else {
                    xf += boxes->lowres_vx[HII_R_INDEX(i, j, k)];
                    yf += boxes->lowres_vy[HII_R_INDEX(i, j, k)];
                    zf += boxes->lowres_vz[HII_R_INDEX(i, j, k)];
                }

                // 2LPT PART
                // add second order corrections
                if (matter_flags->PERTURB_ALGORITHM == 2) {
                    if (matter_flags->PERTURB_ON_HIGH_RES) {
                        xf -= boxes->hires_vx_2LPT[R_INDEX(i, j, k)];
                        yf -= boxes->hires_vy_2LPT[R_INDEX(i, j, k)];
                        zf -= boxes->hires_vz_2LPT[R_INDEX(i, j, k)];
                    } else {
                        xf -= boxes->lowres_vx_2LPT[HII_R_INDEX(i, j, k)];
                        yf -= boxes->lowres_vy_2LPT[HII_R_INDEX(i, j, k)];
                        zf -= boxes->lowres_vz_2LPT[HII_R_INDEX(i, j, k)];
                    }
                }

                // check if we wrapped around, note the casting to ensure < 1.00000
                DI = 10000;
                xf = roundf(xf * DI);
                yf = roundf(yf * DI);
                zf = roundf(zf * DI);
                while (xf >= (float)DI) {
                    xf -= DI;
                }
                while (xf < 0) {
                    xf += DI;
                }
                while (yf >= (float)DI) {
                    yf -= DI;
                }
                while (yf < 0) {
                    yf += DI;
                }
                while (zf >= (float)DI) {
                    zf -= DI;
                }
                while (zf < 0) {
                    zf += DI;
                }
                xf = fabs(xf / (float)DI);  // fabs gets rid of minus sign in -0.00000
                yf = fabs(yf / (float)DI);
                zf = fabs(zf / (float)DI);

                xf *= matter_params->HII_DIM;
                yf *= matter_params->HII_DIM;
                zf *= HII_D_PARA;

                halos_perturbed->halo_coords[i_halo * 3 + 0] = xf;
                halos_perturbed->halo_coords[i_halo * 3 + 1] = yf;
                halos_perturbed->halo_coords[i_halo * 3 + 2] = zf;

                halos_perturbed->halo_masses[i_halo] = halos->halo_masses[i_halo];
                halos_perturbed->star_rng[i_halo] = halos->star_rng[i_halo];
                halos_perturbed->sfr_rng[i_halo] = halos->sfr_rng[i_halo];
                halos_perturbed->xray_rng[i_halo] = halos->xray_rng[i_halo];
            }
        }

        // Divide out multiplicative factor to return to pristine state
#pragma omp parallel shared(boxes, growth_factor_over_BOX_LEN, dimension,               \
                                displacement_factor_2LPT_over_BOX_LEN) private(i, j, k) \
    num_threads(matter_params -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < dimension; i++) {
                for (j = 0; j < dimension; j++) {
                    for (k = 0;
                         k < (unsigned long long)(matter_params->NON_CUBIC_FACTOR * dimension);
                         k++) {
                        if (matter_flags->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->hires_vy[R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->hires_vz[R_INDEX(i, j, k)] /=
                                (growth_factor_over_BOX_LEN / matter_params->NON_CUBIC_FACTOR);

                            if (matter_flags->PERTURB_ALGORITHM == 2) {
                                boxes->hires_vx_2LPT[R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vy_2LPT[R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vz_2LPT[R_INDEX(i, j, k)] /=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     matter_params->NON_CUBIC_FACTOR);
                            }
                        } else {
                            boxes->lowres_vx[HII_R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->lowres_vy[HII_R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->lowres_vz[HII_R_INDEX(i, j, k)] /=
                                (growth_factor_over_BOX_LEN / matter_params->NON_CUBIC_FACTOR);

                            if (matter_flags->PERTURB_ALGORITHM == 2) {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i, j, k)] /=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     matter_params->NON_CUBIC_FACTOR);
                            }
                        }
                        // this is now comoving displacement in units of box size
                    }
                }
            }
        }

        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();
        LOG_DEBUG("Perturbed positions of %llu Halos", halos_perturbed->n_halos);

    }  // End of Try()
    Catch(status) { return (status); }
    return (0);
}

void free_phf(PerturbHaloField *halos) {
    LOG_DEBUG("Freeing PerturbHaloField");
    free(halos->halo_masses);
    free(halos->halo_coords);
    free(halos->star_rng);
    free(halos->sfr_rng);
    free(halos->xray_rng);
    LOG_DEBUG("Done Freeing PerturbHaloField");
    halos->n_halos = 0;
}
