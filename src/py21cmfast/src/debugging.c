/*
    The following functions are simply for testing the exception framework
*/

#include "debugging.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "exceptions.h"
#include "indexing.h"
#include "logger.h"

// definition of the global exception context
struct exception_context the_exception_context[1];

void FunctionThatThrows() { Throw(PhotonConsError); }

int SomethingThatCatches(bool sub_func) {
    // A simple function that catches a thrown error.
    int status;
    Try {
        if (sub_func)
            FunctionThatThrows();
        else
            Throw(PhotonConsError);
    }
    Catch(status) { return status; }
    return 0;
}

int FunctionThatCatches(bool sub_func, bool pass, double *result) {
    int status;
    if (!pass) {
        Try {
            if (sub_func)
                FunctionThatThrows();
            else
                Throw(PhotonConsError);
        }
        Catch(status) {
            LOG_DEBUG("Caught the problem with status %d.", status);
            return status;
        }
    }
    *result = 5.0;
    return 0;
}
void writeSimulationOptions(SimulationOptions *p) {
    LOG_INFO(
        "\n        SimulationOptions:\n"
        "       HII_DIM = %4d\n"
        "       DIM = %4d\n"
        "       BOX_LEN=%8.3f\n"
        "       NON_CUBIC_FACTOR=%8.3f,\n"
        "       N_THREADS=%2d\n"
        "       Z_HEAT_MAX=%8.3f\n"
        "       ZPRIME_STEP_FACTOR=%8.3f\n"
        "       SAMPLER_MIN_MASS=%10.3e\n"
        "       SAMPLER_BUFFER_FACTOR=%8.3f\n"
        "       N_COND_INTERP=%4d\n"
        "       N_PROB_INTERP=%4d\n"
        "       MIN_LOGPROB=%8.3f\n"
        "       HALOMASS_CORRECTION=%8.3f\n",
        p->HII_DIM, p->DIM, p->BOX_LEN, p->NON_CUBIC_FACTOR, p->N_THREADS, p->Z_HEAT_MAX,
        p->ZPRIME_STEP_FACTOR, p->SAMPLER_MIN_MASS, p->SAMPLER_BUFFER_FACTOR, p->N_COND_INTERP,
        p->N_PROB_INTERP, p->MIN_LOGPROB, p->HALOMASS_CORRECTION);
}

void writeMatterOptions(MatterOptions *p) {
    LOG_INFO(
        "\n        MatterOptions:\n"
        "       HMF=%2d\n"
        "       POWER_SPECTRUM=%2d\n"
        "       USE_RELATIVE_VELOCITIES=%1d\n"
        "       PERTURB_ON_HIGH_RES=%1d\n"
        "       USE_FFTW_WISDOM=%1d\n"
        "       USE_INTERPOLATION_TABLES=%1d\n"
        "       PERTURB_ALGORITHM=%1d\n"
        "       MINIMIZE_MEMORY=%1d,\n"
        "       KEEP_3D_VELOCITIES=%1d\n"
        "       SAMPLE_METHOD=%2d\n"
        "       FILTER=%2d\n"
        "       HALO_FILTER=%2d\n"
        "       SOURCE_MODEL=%2d\n"
        "       SMOOTH_EVOLVED_DENSITY_FIELD=%1d\n"
        "       DEXM_OPTIMIZE=%1d\n",
        p->HMF, p->POWER_SPECTRUM, p->USE_RELATIVE_VELOCITIES, p->PERTURB_ON_HIGH_RES,
        p->USE_FFTW_WISDOM, p->USE_INTERPOLATION_TABLES, p->PERTURB_ALGORITHM, p->MINIMIZE_MEMORY,
        p->KEEP_3D_VELOCITIES, p->SAMPLE_METHOD, p->FILTER, p->HALO_FILTER, p->SOURCE_MODEL,
        p->SMOOTH_EVOLVED_DENSITY_FIELD, p->DEXM_OPTIMIZE);
}

void writeCosmoParams(CosmoParams *p) {
    LOG_INFO(
        "\n        CosmoParams:\n"
        "       hlittle=%8.3f\n"
        "       OMm=%8.3f\n"
        "       OMl=%8.3f\n"
        "       OMb=%8.3f\n"
        "       OMn=%8.3f\n"
        "       OMk=%8.3f\n"
        "       OMr=%8.3f\n"
        "       OMtot=%8.3f\n"
        "       Y_He=%8.3f\n"
        "       wl=%8.3f\n"
        "       POWER_INDEX=%8.3f\n",
        p->hlittle, p->OMm, p->OMl, p->OMb, p->OMn, p->OMk, p->OMr, p->OMtot, p->Y_He, p->wl,
        p->POWER_INDEX);
}

void writeAstroParams(AstroParams *p) {
    LOG_INFO(
        "\n        AstroParams:\n"
        "       M_TURN=%10.3e\n"
        "       R_BUBBLE_MAX=%8.3f\n"
        "       F_STAR10=%8.3f\n"
        "       ALPHA_STAR=%8.3f\n"
        "       F_ESC10=%8.3f\n"
        "       ALPHA_ESC=%8.3f\n"
        "       t_STAR=%8.3f\n"
        "       L_X=%10.3e\n"
        "       NU_X_THRESH=%8.3f\n"
        "       X_RAY_SPEC_INDEX=%8.3f,\n"
        "       UPPER_STELLAR_TURNOVER_MASS=%10.3e\n"
        "       UPPER_STELLAR_TURNOVER_INDEX=%8.3e\n",
        p->M_TURN, p->R_BUBBLE_MAX, p->F_STAR10, p->ALPHA_STAR, p->F_ESC10, p->ALPHA_ESC, p->t_STAR,
        p->L_X, p->NU_X_THRESH, p->X_RAY_SPEC_INDEX, p->UPPER_STELLAR_TURNOVER_MASS,
        p->UPPER_STELLAR_TURNOVER_INDEX);
    LOG_INFO(
        "\n        HaloCatalog AstroParams:\n"
        "       SIGMA_STAR=%8.3f\n"
        "       SIGMA_SFR_LIM=%8.3f\n"
        "       SIGMA_SFR_INDEX=%8.3f\n"
        "       SIGMA_LX=%8.3f\n",
        p->SIGMA_STAR, p->SIGMA_SFR_LIM, p->SIGMA_SFR_INDEX, p->SIGMA_LX);
    LOG_INFO(
        "\n        MiniHalo AstroParams:\n"
        "       ALPHA_STAR_MINI=%8.3f\n"
        "       F_ESC7_MINI=%8.3f\n"
        "       L_X_MINI=%10.3e\n"
        "       F_STAR7_MINI=%8.3f\n"
        "       F_H2_SHIELD=%8.3f\n"
        "       A_LW=%8.3f\n"
        "       BETA_LW=%8.3f\n"
        "       A_VCB=%8.3f\n"
        "       BETA_VCB=%8.3f\n",
        p->ALPHA_STAR_MINI, p->F_ESC7_MINI, p->L_X_MINI, p->F_STAR7_MINI, p->F_H2_SHIELD, p->A_LW,
        p->BETA_LW, p->A_VCB, p->BETA_VCB);
    LOG_INFO(
        "\n     Const-z  AstroParams:\n"
        "       HII_EFF_FACTOR=%10.3e\n"
        "       ION_Tvir_MIN=%10.3e\n"
        "       X_RAY_Tvir_MIN=%10.3e\n",
        p->HII_EFF_FACTOR, p->ION_Tvir_MIN, p->X_RAY_Tvir_MIN);
}

void writeAstroOptions(AstroOptions *p) {
    LOG_INFO(
        "\n        AstroOptions:\n"
        "       USE_MINI_HALOS=%1d\n"
        "       INHOMO_RECO=%1d\n"
        "       USE_TS_FLUCT=%1d\n"
        "       M_MIN_in_Mass=%1d\n"
        "       USE_EXP_FILTER=%1d\n"
        "       USE_CMB_HEATING=%1d\n"
        "       USE_LYA_HEATING=%1d\n"
        "       FIX_VCB_AVG=%1d\n"
        "       CELL_RECOMB=%1d\n"
        "       PHOTON_CONS_TYPE=%2d\n"
        "       USE_UPPER_STELLAR_TURNOVER=%1d\n"
        "       HALO_SCALING_RELATIONS_MEDIAN=%1d\n"
        "       HII_FILTER=%2d\n"
        "       HEAT_FILTER=%2d\n"
        "       IONISE_ENTIRE_SPHERE=%1d\n"
        "       INTEGRATION_METHOD_ATOMIC=%2d\n"
        "       INTEGRATION_METHOD_MINI=%2d\n",
        p->USE_MINI_HALOS, p->INHOMO_RECO, p->USE_TS_FLUCT, p->M_MIN_in_Mass, p->USE_EXP_FILTER,
        p->USE_CMB_HEATING, p->USE_LYA_HEATING, p->FIX_VCB_AVG, p->CELL_RECOMB, p->PHOTON_CONS_TYPE,
        p->USE_UPPER_STELLAR_TURNOVER, p->HALO_SCALING_RELATIONS_MEDIAN, p->HII_FILTER,
        p->HEAT_FILTER, p->IONISE_ENTIRE_SPHERE, p->INTEGRATION_METHOD_ATOMIC,
        p->INTEGRATION_METHOD_MINI);
}

// Wrapper function to select appropriate indexing function based on layout
static index_t get_corner_index(int x, int y, int z, int dim[3], GridLayout layout) {
    if (layout == FFTW_REAL_LAYOUT) {
        return grid_index_fftw_r(x, y, z, dim);
    } else if (layout == FFTW_COMPLEX_LAYOUT) {
        return grid_index_fftw_c(x, y, z, dim);
    } else {  // STANDARD_LAYOUT
        return grid_index_general(x, y, z, dim);
    }
}

// Helper to compute dimensions based on layout type
static void compute_layout_dims(int size_x, int size_y, int size_z, GridLayout layout, int dim[3],
                                int *original_z) {
    dim[0] = size_x;
    dim[1] = size_y;

    if (layout == FFTW_REAL_LAYOUT) {
        *original_z = 2 * (size_z / 2 - 1);
        dim[2] = *original_z;
    } else if (layout == FFTW_COMPLEX_LAYOUT) {
        *original_z = 2 * (size_z - 1);
        dim[2] = size_z;
    } else {  // STANDARD_LAYOUT
        *original_z = size_z;
        dim[2] = size_z;
    }
}

void get_corner_indices(int size_x, int size_y, int size_z, GridLayout layout, index_t indices[8]) {
    int original_z;
    int dim[3];

    compute_layout_dims(size_x, size_y, size_z, layout, dim, &original_z);

    // Compute all 8 corner indices using wrapper function
    indices[0] = get_corner_index(0, 0, 0, dim, layout);
    indices[1] = get_corner_index(size_x - 1, 0, 0, dim, layout);
    indices[2] = get_corner_index(0, size_y - 1, 0, dim, layout);
    indices[3] = get_corner_index(size_x - 1, size_y - 1, 0, dim, layout);
    indices[4] = get_corner_index(0, 0, original_z - 1, dim, layout);
    indices[5] = get_corner_index(size_x - 1, 0, original_z - 1, dim, layout);
    indices[6] = get_corner_index(0, size_y - 1, original_z - 1, dim, layout);
    indices[7] = get_corner_index(size_x - 1, size_y - 1, original_z - 1, dim, layout);
}

void debugSummarizeBox(float *box, int size_x, int size_y, int size_z, GridLayout layout,
                       char *indent) {
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
    float corners[8];
    index_t indices[8];
    index_t idx;

    get_corner_indices(size_x, size_y, size_z, layout, indices);
    for (idx = 0; idx < 8; idx++) {
        corners[idx] = box[indices[idx]];
    }

    LOG_SUPER_DEBUG("%sCorners: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e", indent, corners[0],
                    corners[1], corners[2], corners[3], corners[4], corners[5], corners[6],
                    corners[7]);

    float sum, mean, mn, mx;
    sum = 0;
    mn = box[0];
    mx = box[0];

    // Select indexing function and extract unpadded z based on layout
    int original_z;
    int dim[3];

    compute_layout_dims(size_x, size_y, size_z, layout, dim, &original_z);

    // Uniform loop using wrapper function
    int i, j, k;
    index_t grid_idx;

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            for (k = 0; k < original_z; k++) {
                grid_idx = get_corner_index(i, j, k, dim, layout);
                sum += box[grid_idx];
                mn = fminf(mn, box[grid_idx]);
                mx = fmaxf(mx, box[grid_idx]);
            }
        }
    }
    index_t tot_size = (index_t)size_x * size_y * original_z;
    mean = sum / tot_size;

    LOG_SUPER_DEBUG("%sSum/Mean/Min/Max: %.4e, %.4e, %.4e, %.4e", indent, sum, mean, mn, mx);
#endif
}

void debugSummarizeBoxDouble(double *box, int size_x, int size_y, int size_z, GridLayout layout,
                             char *indent) {
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
    double corners[8];
    index_t indices[8];
    index_t idx;

    get_corner_indices(size_x, size_y, size_z, layout, indices);
    for (idx = 0; idx < 8; idx++) {
        corners[idx] = box[indices[idx]];
    }

    LOG_SUPER_DEBUG("%sCorners: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e", indent, corners[0],
                    corners[1], corners[2], corners[3], corners[4], corners[5], corners[6],
                    corners[7]);

    double sum, mean, mn, mx;
    sum = 0;
    mn = box[0];
    mx = box[0];

    // Select indexing function and extract unpadded z based on layout
    int original_z;
    int dim[3];

    compute_layout_dims(size_x, size_y, size_z, layout, dim, &original_z);

    // Uniform loop using wrapper function
    int i, j, k;
    index_t grid_idx;

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            for (k = 0; k < original_z; k++) {
                grid_idx = get_corner_index(i, j, k, dim, layout);
                sum += box[grid_idx];
                mn = fmin(mn, box[grid_idx]);
                mx = fmax(mx, box[grid_idx]);
            }
        }
    }
    index_t tot_size = (index_t)size_x * size_y * original_z;
    mean = sum / tot_size;

    LOG_SUPER_DEBUG("%sSum/Mean/Min/Max: %.4e, %.4e, %.4e, %.4e", indent, sum, mean, mn, mx);
#endif
}

void debugSummarizeBoxComplex(fftwf_complex *box, int size_x, int size_y, int size_z,
                              GridLayout layout, char *indent) {
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
    float corners_real[8];
    float corners_imag[8];
    index_t indices[8];
    index_t idx;

    get_corner_indices(size_x, size_y, size_z, layout, indices);
    for (idx = 0; idx < 8; idx++) {
        float *elem = (float *)&box[indices[idx]];
        corners_real[idx] = elem[0];
        corners_imag[idx] = elem[1];
    }

    LOG_SUPER_DEBUG("%sCorners (Real Part): %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e", indent,
                    corners_real[0], corners_real[1], corners_real[2], corners_real[3],
                    corners_real[4], corners_real[5], corners_real[6], corners_real[7]);

    LOG_SUPER_DEBUG("%sCorners (Imag Part): %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e", indent,
                    corners_imag[0], corners_imag[1], corners_imag[2], corners_imag[3],
                    corners_imag[4], corners_imag[5], corners_imag[6], corners_imag[7]);

    double complex sum, mean, mn, mx;
    sum = 0 + 0 * I;
    mn = box[0];
    mx = box[0];

    // Select indexing function and extract unpadded z based on layout
    int original_z;
    int dim[3];

    compute_layout_dims(size_x, size_y, size_z, layout, dim, &original_z);

    // Uniform loop using wrapper function
    int i, j, k;
    index_t grid_idx;

    for (i = 0; i < size_x; i++) {
        for (j = 0; j < size_y; j++) {
            for (k = 0; k < original_z; k++) {
                grid_idx = get_corner_index(i, j, k, dim, layout);
                sum += box[grid_idx];
                mn = fminf(mn, box[grid_idx]);
                mx = fmaxf(mx, box[grid_idx]);
            }
        }
    }
    index_t tot_size = (index_t)size_x * size_y * original_z;
    mean = sum / tot_size;

    LOG_SUPER_DEBUG("%sSum/Mean/Min/Max: %.4e+%.4ei, %.4e+%.4ei, %.4e+%.4ei, %.4e+%.4ei", indent,
                    creal(sum), cimag(sum), creal(mean), cimag(mean), creal(mn), cimag(mn),
                    creal(mx), cimag(mx));
#endif
}

void debugSummarizeIC(InitialConditions *x, int HII_DIM, int DIM, float NCF) {
    LOG_SUPER_DEBUG("Summary of InitialConditions:");
    LOG_SUPER_DEBUG("  lowres_density: ");
    debugSummarizeBox(x->lowres_density, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
    LOG_SUPER_DEBUG("  hires_density: ");
    debugSummarizeBox(x->hires_density, DIM, DIM, D_PARA, STANDARD_LAYOUT, "    ");
    LOG_SUPER_DEBUG("  lowres_vx: ");
    debugSummarizeBox(x->lowres_vx, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
    LOG_SUPER_DEBUG("  lowres_vy: ");
    debugSummarizeBox(x->lowres_vy, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
    LOG_SUPER_DEBUG("  lowres_vz: ");
    debugSummarizeBox(x->lowres_vz, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
}

void debugSummarizePerturbedField(PerturbedField *x, int HII_DIM, float NCF) {
    LOG_SUPER_DEBUG("Summary of PerturbedField:");
    LOG_SUPER_DEBUG("  density: ");
    debugSummarizeBox(x->density, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
    LOG_SUPER_DEBUG("  velocity_x: ");
    debugSummarizeBox(x->velocity_x, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
    LOG_SUPER_DEBUG("  velocity_y: ");
    debugSummarizeBox(x->velocity_y, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
    LOG_SUPER_DEBUG("  velocity_z: ");
    debugSummarizeBox(x->velocity_z, HII_DIM, HII_DIM, HII_D_PARA, STANDARD_LAYOUT, "    ");
}
