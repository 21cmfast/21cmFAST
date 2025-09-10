#include "dft.h"

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "exceptions.h"
#include "indexing.h"
#include "logger.h"

int dft_c2r_cube(bool use_wisdom, int dim, int dim_los, int n_threads, fftwf_complex* box) {
    char wisdom_filename[500];
    unsigned flag = FFTW_ESTIMATE;
    int status;
    fftwf_plan plan;

    Try {
        if (use_wisdom) {
            // Check to see if the wisdom exists
            sprintf(wisdom_filename, "%s/c2r_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                    dim, dim_los, n_threads);

            if (fftwf_import_wisdom_from_filename(wisdom_filename) != 0) {
                flag = FFTW_WISDOM_ONLY;
            } else {
                LOG_WARNING(
                    "Cannot locate FFTW Wisdom: %s file not found. Reverting to FFTW_ESTIMATE.",
                    wisdom_filename);
            }
        }
        plan = fftwf_plan_dft_c2r_3d(dim, dim, dim_los, (fftwf_complex*)box, (float*)box, flag);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }
    Catch(status) { return (status); }
    return (0);
}

int dft_r2c_cube(bool use_wisdom, int dim, int dim_los, int n_threads, fftwf_complex* box) {
    char wisdom_filename[500];
    unsigned flag = FFTW_ESTIMATE;
    int status;
    fftwf_plan plan;

    Try {
        if (use_wisdom) {
            // Check to see if the wisdom exists
            sprintf(wisdom_filename, "%s/r2c_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                    dim, dim_los, n_threads);

            if (fftwf_import_wisdom_from_filename(wisdom_filename) != 0) {
                flag = FFTW_WISDOM_ONLY;
            } else {
                LOG_WARNING(
                    "Cannot locate FFTW Wisdom: %s file not found. Reverting to FFTW_ESTIMATE.",
                    wisdom_filename);
            }
        }
        plan = fftwf_plan_dft_r2c_3d(dim, dim, dim_los, (float*)box, (fftwf_complex*)box, flag);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }
    Catch(status) { return (status); }
    return (0);
}

int CreateFFTWWisdoms() {
    int status;

    Try {  // This Try wraps the entire function so we don't indent.
        fftwf_plan plan;

        char wisdom_filename[500];

        omp_set_num_threads(simulation_options_global->N_THREADS);
        fftwf_init_threads();
        fftwf_plan_with_nthreads(simulation_options_global->N_THREADS);
        fftwf_cleanup_threads();

        // allocate array for the k-space and real-space boxes
        fftwf_complex* HIRES_box =
            (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        fftwf_complex* LOWRES_box =
            (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        sprintf(wisdom_filename, "%s/r2c_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->DIM, (int)D_PARA, simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_r2c_3d(simulation_options_global->DIM,
                                         simulation_options_global->DIM, D_PARA, (float*)HIRES_box,
                                         (fftwf_complex*)HIRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename, "%s/c2r_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->DIM, (int)D_PARA, simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_c2r_3d(
                simulation_options_global->DIM, simulation_options_global->DIM, D_PARA,
                (fftwf_complex*)HIRES_box, (float*)HIRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename, "%s/r2c_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->HII_DIM, (int)HII_D_PARA,
                simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_r2c_3d(
                simulation_options_global->HII_DIM, simulation_options_global->HII_DIM, HII_D_PARA,
                (float*)LOWRES_box, (fftwf_complex*)LOWRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename, "%s/c2r_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->HII_DIM, (int)HII_D_PARA,
                simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_c2r_3d(
                simulation_options_global->HII_DIM, simulation_options_global->HII_DIM, HII_D_PARA,
                (fftwf_complex*)LOWRES_box, (float*)LOWRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();

        // deallocate
        fftwf_free(HIRES_box);
        fftwf_free(LOWRES_box);

    }  // End of Try{}

    Catch(status) { return (status); }
    return (0);
}
