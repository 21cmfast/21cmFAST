// Re-write of update_halo_coords from the original 21cmFAST

// ComputePerturbedHaloCatalog reads in the linear velocity field, and uses
// it to update halo locations with a corresponding displacement field

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "HaloBox.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "PerturbedHaloCatalog.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "exceptions.h"
#include "indexing.h"
#include "logger.h"

int ComputePerturbedHaloCatalog(float redshift, InitialConditions *boxes, TsBox *prev_ts,
                                IonizedBox *prev_ion, HaloCatalog *halos,
                                PerturbedHaloCatalog *halos_perturbed) {
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

        float growth_factor, displacement_factor_2LPT, init_growth_factor,
            init_displacement_factor_2LPT;
        unsigned long long int i_halo;

        // Function for deciding the dimensions of loops when we could
        // use either the low or high resolution grids.
        double boxlen = simulation_options_global->BOX_LEN;
        double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
        double box_size[3] = {boxlen, boxlen, boxlen_z};
        int box_dim[3];
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
        double cell_size_inv = box_dim[0] / box_size[0];

        growth_factor = dicke(redshift);
        displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

        init_growth_factor = dicke(simulation_options_global->INITIAL_REDSHIFT);
        init_displacement_factor_2LPT =
            -(3.0 / 7.0) * init_growth_factor * init_growth_factor;  // 2LPT eq. D8

        // ***************   BEGIN INITIALIZATION   ************************** //
        double velocity_displacement_factor = growth_factor - init_growth_factor;
        double velocity_displacement_factor_2LPT =
            displacement_factor_2LPT - init_displacement_factor_2LPT;

        // Function for deciding the dimensions of loops when we could
        // use either the low or high resolution grids.

        init_ps();
        growth_factor = dicke(redshift);  // normalized to 1 at z=0
        halos_perturbed->n_halos = halos->n_halos;

        // ******************   END INITIALIZATION     ******************************** //
        unsigned long long int n_exact_dim = 0;
        bool error_in_parallel = false;
#pragma omp parallel private(i_halo) num_threads(simulation_options_global->N_THREADS) \
    reduction(+ : n_exact_dim)
        {
            double pos[3];
            unsigned long long grid_index;
            int ipos[3];
#pragma omp for
            for (i_halo = 0; i_halo < halos->n_halos; i_halo++) {
                if (error_in_parallel) continue;
                // convert location to fractional value
                pos[0] = halos->halo_coords[i_halo * 3 + 0];
                pos[1] = halos->halo_coords[i_halo * 3 + 1];
                pos[2] = halos->halo_coords[i_halo * 3 + 2];

                pos_to_index(pos, cell_size_inv, ipos);
                wrap_coord(ipos, box_dim);
                grid_index = grid_index_general(ipos[0], ipos[1], ipos[2], box_dim);

                for (int i_dim = 0; i_dim < 3; i_dim++) {
                    pos[i_dim] += vel_pointers[i_dim][grid_index] * velocity_displacement_factor;
                    if (matter_options_global->PERTURB_ALGORITHM == 2)
                        pos[i_dim] -= vel_pointers_2LPT[i_dim][grid_index] *
                                      velocity_displacement_factor_2LPT;
                }

                // Mutliplying before the wrapping to ensure that floating point errors
                //  do not cause the halo to be placed outside the box.
                wrap_position(pos, box_size);

                halos_perturbed->halo_coords[i_halo * 3 + 0] = pos[0];
                halos_perturbed->halo_coords[i_halo * 3 + 1] = pos[1];
                halos_perturbed->halo_coords[i_halo * 3 + 2] = pos[2];
            }
        }

        LOG_DEBUG("starting haloprops");
        convert_halo_props(redshift, boxes, prev_ts, prev_ion, halos, halos_perturbed);
        // Divide out multiplicative factor to return to pristine state
        LOG_SUPER_DEBUG("Number of halos exactly on the box edge = %llu of %llu", n_exact_dim,
                        halos->n_halos);
        if (error_in_parallel) {
            LOG_ERROR("Error in parallel processing, some halos were out of bounds.");
            Throw(ValueError);
        }
        LOG_DEBUG("Perturbed positions of %llu Halos", halos_perturbed->n_halos);

    }  // End of Try()
    Catch(status) { return (status); }
    return (0);
}
