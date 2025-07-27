// Re-write of update_halo_coords from the original 21cmFAST

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

int ComputePerturbHaloField(float redshift, InitialConditions *boxes, HaloField *halos,
                            PerturbHaloField *halos_perturbed) {
    int status;

    Try {  // This Try brackets the whole function, so we don't indent.

        LOG_DEBUG("input value:");
        LOG_DEBUG("redshift=%f", redshift);
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeMatterOptions(matter_options_global);
        writeCosmoParams(cosmo_params_global);
#endif

        // Makes the parameter structs visible to a variety of functions/macros
        // Do each time to avoid Python garbage collection issues

        omp_set_num_threads(simulation_options_global->N_THREADS);

        double growth_factor, displacement_factor_2LPT, xf, yf, zf, growth_factor_over_BOX_LEN,
            displacement_factor_2LPT_over_BOX_LEN;
        double boxlen = simulation_options_global->BOX_LEN;
        double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
        unsigned long long int i, j, k, dimension;
        unsigned long long i_halo;
        bool error_in_parallel = false;

        float box_size[3] = {boxlen, boxlen, boxlen_z};
        unsigned long long int box_dim[3];
        float *vel_pointers[3], *vel_pointers_2LPT[3];
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            box_dim[0] = simulation_options_global->DIM;
            box_dim[1] = simulation_options_global->DIM;
            box_dim[2] = D_PARA;
            vel_pointers[0] = boxes->hires_vx;
            vel_pointers[1] = boxes->hires_vy;
            vel_pointers[2] = boxes->hires_vz;
            vel_pointers_2LPT[0] = boxes->hires_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->hires_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->hires_vz_2LPT;
        } else {
            box_dim[0] = simulation_options_global->HII_DIM;
            box_dim[1] = simulation_options_global->HII_DIM;
            box_dim[2] = HII_D_PARA;
            vel_pointers[0] = boxes->lowres_vx;
            vel_pointers[1] = boxes->lowres_vy;
            vel_pointers[2] = boxes->lowres_vz;
            vel_pointers_2LPT[0] = boxes->lowres_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->lowres_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->lowres_vz_2LPT;
        }

        LOG_DEBUG("Begin Initialisation");

        // Function for deciding the dimensions of loops when we could
        // use either the low or high resolution grids.
        dimension = matter_options_global->PERTURB_ON_HIGH_RES ? simulation_options_global->DIM
                                                               : simulation_options_global->HII_DIM;

        // ***************** END INITIALIZATION ***************** //
        init_ps();
        growth_factor = dicke(redshift);  // normalized to 1 at z=0
        displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

        // TODO: combine/match with PerturbField.c
        //  which uses (D(z) - D(init))/BOXLEN
        growth_factor_over_BOX_LEN = growth_factor / boxlen;
        displacement_factor_2LPT_over_BOX_LEN = displacement_factor_2LPT / boxlen;

        // now add the missing factor of Ddot to velocity field
#pragma omp parallel shared(boxes, dimension, growth_factor_over_BOX_LEN) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < dimension; i++) {
                for (j = 0; j < dimension; j++) {
                    for (k = 0;
                         k < (unsigned long long)(simulation_options_global->NON_CUBIC_FACTOR *
                                                  dimension);
                         k++) {
                        if (matter_options_global->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->hires_vy[R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->hires_vz[R_INDEX(i, j, k)] *=
                                (growth_factor_over_BOX_LEN /
                                 simulation_options_global->NON_CUBIC_FACTOR);
                        } else {
                            boxes->lowres_vx[HII_R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->lowres_vy[HII_R_INDEX(i, j, k)] *= growth_factor_over_BOX_LEN;
                            boxes->lowres_vz[HII_R_INDEX(i, j, k)] *=
                                (growth_factor_over_BOX_LEN /
                                 simulation_options_global->NON_CUBIC_FACTOR);
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
        if (matter_options_global->PERTURB_ALGORITHM == 2) {
            // now add the missing factor in eq. D9
#pragma omp parallel shared(boxes, displacement_factor_2LPT_over_BOX_LEN, dimension) \
    private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < dimension; i++) {
                    for (j = 0; j < dimension; j++) {
                        for (k = 0;
                             k < (unsigned long long)(simulation_options_global->NON_CUBIC_FACTOR *
                                                      dimension);
                             k++) {
                            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vy_2LPT[R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vz_2LPT[R_INDEX(i, j, k)] *=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     simulation_options_global->NON_CUBIC_FACTOR);
                            } else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i, j, k)] *=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i, j, k)] *=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     simulation_options_global->NON_CUBIC_FACTOR);
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
        int n_exact_dim = 0;
#pragma omp parallel shared(boxes, halos, halos_perturbed) private(i_halo) \
    num_threads(simulation_options_global -> N_THREADS) reduction(+ : n_exact_dim)
        {
            float pos[3];
            unsigned long long grid_index;
            int ipos[3];
#pragma omp for
            for (i_halo = 0; i_halo < halos->n_halos; i_halo++) {
                if (error_in_parallel) continue;
                // convert location to fractional value
                for (int i_dim = 0; i_dim < 3; i_dim++) {
                    pos[i_dim] = halos->halo_coords[i_halo * 3 + i_dim] / box_size[i_dim];
                    // Sometimes, halos are exactly on the edge of the box even after wrapping
                    //   from floating point errors.
                    if (pos[i_dim] == 1.0) {
                        ipos[i_dim] = box_dim[i_dim] - 1;
                        n_exact_dim++;
                    } else {
                        ipos[i_dim] = (int)(pos[i_dim] * box_dim[i_dim]);
                    }
                }
                if (ipos[0] >= box_dim[0] || ipos[1] >= box_dim[1] || ipos[2] >= box_dim[2] ||
                    ipos[0] < 0 || ipos[1] < 0 || ipos[2] < 0) {
                    LOG_ERROR(
                        "Halo %llu is out of bounds: (%d, %d, %llu) for box size (%f, %f, %f) "
                        "struct (%f, %f, %f) norm (%f, %f, %f)",
                        i_halo, ipos[0], ipos[1], ipos[2], boxlen, boxlen, boxlen_z,
                        halos->halo_coords[i_halo * 3 + 0], halos->halo_coords[i_halo * 3 + 1],
                        halos->halo_coords[i_halo * 3 + 2], pos[0], pos[1], pos[2]);
                    error_in_parallel = true;
                    continue;  // skip this halo
                }
                grid_index = matter_options_global->PERTURB_ON_HIGH_RES
                                 ? R_INDEX(ipos[0], ipos[1], ipos[2])
                                 : HII_R_INDEX(ipos[0], ipos[1], ipos[2]);

                for (int i_dim = 0; i_dim < 3; i_dim++) {
                    pos[i_dim] += vel_pointers[i_dim][grid_index];
                    if (matter_options_global->PERTURB_ALGORITHM == 2)
                        pos[i_dim] -= vel_pointers_2LPT[i_dim][grid_index];
                    pos[i_dim] *= box_size[i_dim];  // convert to comoving position
                }

                // Mutliplying before the wrapping to ensure that floating point errors
                //  do not cause the halo to be placed outside the box.
                wrap_position(pos, box_size);

                halos_perturbed->halo_coords[i_halo * 3 + 0] = pos[0];
                halos_perturbed->halo_coords[i_halo * 3 + 1] = pos[1];
                halos_perturbed->halo_coords[i_halo * 3 + 2] = pos[2];

                halos_perturbed->halo_masses[i_halo] = halos->halo_masses[i_halo];
                halos_perturbed->star_rng[i_halo] = halos->star_rng[i_halo];
                halos_perturbed->sfr_rng[i_halo] = halos->sfr_rng[i_halo];
                halos_perturbed->xray_rng[i_halo] = halos->xray_rng[i_halo];
            }
        }
        // Divide out multiplicative factor to return to pristine state
        LOG_ULTRA_DEBUG("Number of halos exactly on the box edge = %d of %d", n_exact_dim,
                        halos->n_halos);
        if (error_in_parallel) {
            LOG_ERROR("Error in parallel processing, some halos were out of bounds.");
            Throw(ValueError);
        }
#pragma omp parallel shared(boxes, growth_factor_over_BOX_LEN, dimension,               \
                                displacement_factor_2LPT_over_BOX_LEN) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < dimension; i++) {
                for (j = 0; j < dimension; j++) {
                    for (k = 0;
                         k < (unsigned long long)(simulation_options_global->NON_CUBIC_FACTOR *
                                                  dimension);
                         k++) {
                        if (matter_options_global->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->hires_vy[R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->hires_vz[R_INDEX(i, j, k)] /=
                                (growth_factor_over_BOX_LEN /
                                 simulation_options_global->NON_CUBIC_FACTOR);

                            if (matter_options_global->PERTURB_ALGORITHM == 2) {
                                boxes->hires_vx_2LPT[R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vy_2LPT[R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->hires_vz_2LPT[R_INDEX(i, j, k)] /=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     simulation_options_global->NON_CUBIC_FACTOR);
                            }
                        } else {
                            boxes->lowres_vx[HII_R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->lowres_vy[HII_R_INDEX(i, j, k)] /= growth_factor_over_BOX_LEN;
                            boxes->lowres_vz[HII_R_INDEX(i, j, k)] /=
                                (growth_factor_over_BOX_LEN /
                                 simulation_options_global->NON_CUBIC_FACTOR);

                            if (matter_options_global->PERTURB_ALGORITHM == 2) {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i, j, k)] /=
                                    displacement_factor_2LPT_over_BOX_LEN;
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i, j, k)] /=
                                    (displacement_factor_2LPT_over_BOX_LEN /
                                     simulation_options_global->NON_CUBIC_FACTOR);
                            }
                        }
                        // this is now comoving displacement in units of box size
                    }
                }
            }
        }
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
