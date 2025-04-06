
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "cexcept.h"
#include "dft.h"
#include "exceptions.h"
#include "indexing.h"
#include "logger.h"

double real_tophat_filter(double kR) {
    // Second order taylor expansion around kR==0
    if (kR < 1e-4) return 1 - kR * kR / 10;
    return 3.0 * pow(kR, -3) * (sin(kR) - cos(kR) * kR);
}

// TODO: TEST USING kR^2 INSTEAD FOR SPEED
//   ALSO TEST ASSIGNMENT vs MULTIPLICATION
double sharp_k_filter(double kR) {
    // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
    if (kR * 0.413566994 > 1) return 0.;
    return 1;
}

double gaussian_filter(double kR_squared) { return exp(-0.643 * 0.643 * kR_squared / 2.); }

double filter_function(double kR, int filter_type) {
    switch (filter_type) {
        case 0:
            return real_tophat_filter(kR);
        case 1:
            return sharp_k_filter(kR);
        case 2:
            return gaussian_filter(kR * kR);
        default:
            LOG_ERROR("No such filter: %i.", filter_type);
            Throw(ValueError);
    }
}

// NOTE: Only used in dsigmasqdm, so I didn't bother making functions for each filter
double dwdm_filter(double k, double R, int filter_type) {
    double kR = k * R;
    double w, dwdr, drdm;
    if (filter_type == 0) {  // top hat
        if ((kR) < 1.0e-4) {
            w = 1.0;
        }  // w converges to 1 as (kR) -> 0
        else {
            w = 3.0 * (sin(kR) / pow(kR, 3) - cos(kR) / pow(kR, 2));
        }
        if ((kR) < 1.0e-10) {
            dwdr = 0;
        } else {
            dwdr = 9 * cos(kR) * k / pow(kR, 3) + 3 * sin(kR) * (1 - 3 / (kR * kR)) / (kR * R);
        }
        // TODO: figure out what this is/was
        // 3*k*( 3*cos(kR)/pow(kR,3) + sin(kR)*(-3*pow(kR, -4) + 1/(kR*kR)) );}
        //      dwdr = -1e8 * k / (R*1e3);
        drdm = 1.0 / (4.0 * PI * cosmo_params_global->OMm * RHOcrit * R * R);
    } else if (filter_type == 2) {  // gaussian of width 1/R
        w = exp(-kR * kR / 2.0);
        dwdr = -k * kR * w;
        drdm = 1.0 / (pow(2 * PI, 1.5) * cosmo_params_global->OMm * RHOcrit * 3 * R * R);
    } else {
        LOG_ERROR("No such filter for dWdM: %i", matter_flags_global->FILTER);
        Throw(ValueError);
    }
    // now do d(w^2)/dm = 2 w dw/dr dr/dm
    return 2 * w * dwdr * drdm;
}

double exp_mfp_filter(double k, double R, double mfp, double exp_term) {
    double f;

    double kR = k * R;
    double ratio = mfp / R;
    // Second order taylor expansion around kR==0
    // NOTE: the taylor coefficients could be stored and passed in
    //   but there aren't any super expensive operations here
    //   assuming the integer pow calls are optimized by the compiler
    //   test with the profiler
    if (kR < 1e-4) {
        double ts_0 =
            6 * pow(ratio, 3) - exp_term * (6 * pow(ratio, 3) + 6 * pow(ratio, 2) + 3 * ratio);
        return ts_0 +
               (exp_term * (2 * pow(ratio, 2) + 0.5 * ratio) - 2 * ts_0 * pow(ratio, 2)) * kR * kR;
    }

    // Davies & Furlanetto MFP-eps(r) window function
    f = (kR * kR * pow(ratio, 2) + 2 * ratio + 1) * ratio * cos(kR);
    f += (kR * kR * (pow(ratio, 2) - pow(ratio, 3)) + ratio + 1) * sin(kR) / kR;
    f *= exp_term;
    f -= 2 * pow(ratio, 2);
    f *= -3 * ratio / pow(pow(kR * ratio, 2) + 1, 2);
    return f;
}

double spherical_shell_filter(double k, double R_outer, double R_inner) {
    double kR_inner = k * R_inner;
    double kR_outer = k * R_outer;

    // Second order taylor expansion around kR_outer==0
    if (kR_outer < 1e-4)
        return 1. - kR_outer * kR_outer / 10 * (pow(R_inner / R_outer, 5) - 1) /
                        (pow(R_inner / R_outer, 3) - 1);

    return 3.0 / (pow(kR_outer, 3) - pow(kR_inner, 3)) *
           (sin(kR_outer) - cos(kR_outer) * kR_outer - sin(kR_inner) + cos(kR_inner) * kR_inner);
}

void filter_box(fftwf_complex *box, int RES, int filter_type, float R, float R_param) {
    int dimension, midpoint;  // TODO: figure out why defining as ULL breaks this
    switch (RES) {
        case 0:
            dimension = matter_params_global->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = matter_params_global->HII_DIM;
            midpoint = HII_MIDDLE;
            break;
        default:
            LOG_ERROR("Resolution for filter functions must be 0(DIM) or 1(HII_DIM)");
            Throw(ValueError);
            break;
    }

    // setup constants if needed
    double R_const;
    if (filter_type == 3) {
        R_const = exp(-R / R_param);
    }

// loop through k-box
#pragma omp parallel num_threads(matter_params_global->N_THREADS)
    {
        int n_x, n_z, n_y;
        float k_x, k_y, k_z, k_mag_sq, kR;
        unsigned long long grid_index;
#pragma omp for
        for (n_x = 0; n_x < dimension; n_x++) {
            if (n_x > midpoint) {
                k_x = (n_x - dimension) * DELTA_K;
            } else {
                k_x = n_x * DELTA_K;
            }

            for (n_y = 0; n_y < dimension; n_y++) {
                if (n_y > midpoint) {
                    k_y = (n_y - dimension) * DELTA_K;
                } else {
                    k_y = n_y * DELTA_K;
                }

                for (n_z = 0; n_z <= (int)(matter_params_global->NON_CUBIC_FACTOR * midpoint);
                     n_z++) {
                    k_z = n_z * DELTA_K_PARA;
                    k_mag_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                    grid_index = RES == 1 ? HII_C_INDEX(n_x, n_y, n_z) : C_INDEX(n_x, n_y, n_z);

                    // TODO: it would be nice to combine these into the filter_function call, *but*
                    // since
                    //  each can take different arguments more thought is needed
                    if (filter_type == 0) {  // real space top-hat
                        kR = sqrt(k_mag_sq) * R;
                        box[grid_index] *= real_tophat_filter(kR);
                    } else if (filter_type == 1) {  // k-space top hat
                        // NOTE: why was this commented????
                        //  This is actually (kR^2) but since we zero the value and find kR > 1 this
                        //  is more computationally efficient kR = 0.17103765852*( k_x*k_x + k_y*k_y
                        //  + k_z*k_z )*R*R;
                        kR = sqrt(k_mag_sq) * R;
                        box[grid_index] *= sharp_k_filter(kR);
                    } else if (filter_type == 2) {  // gaussian
                        // This is actually (kR^2) but since we zero the value and find kR > 1 this
                        // is more computationally efficient
                        kR = k_mag_sq * R * R;
                        box[grid_index] *= gaussian_filter(kR);
                    }
                    // The next two filters are not given by the HII_FILTER global, but used for
                    // specific grids
                    else if (filter_type ==
                             3) {  // exponentially decaying tophat, param == scale of decay (MFP)
                        // NOTE: This should be optimized, I havne't looked at it in a while
                        box[grid_index] *= exp_mfp_filter(sqrt(k_mag_sq), R, R_param, R_const);
                    } else if (filter_type == 4) {  // spherical shell, R_param == inner radius
                        box[grid_index] *= spherical_shell_filter(sqrt(k_mag_sq), R, R_param);
                    } else {
                        if ((n_x == 0) && (n_y == 0) && (n_z == 0))
                            LOG_WARNING("Filter type %i is undefined. Box is unfiltered.",
                                        filter_type);
                    }
                }
            }
        }  // end looping through k box
    }

    return;
}

// Test function to filter a box without computing a whole output box
int test_filter(MatterParams *matter_params, MatterFlags *matter_flags, CosmoParams *cosmo_params,
                AstroParams *astro_params, AstroFlags *astro_flags, float *input_box, double R,
                double R_param, int filter_flag, double *result) {
    int i, j, k;
    unsigned long long int ii;

    Broadcast_struct_global_all(matter_params, matter_flags, cosmo_params, astro_params,
                                astro_flags);

    // setup the box
    fftwf_complex *box_unfiltered =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box_filtered =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    for (i = 0; i < matter_params->HII_DIM; i++)
        for (j = 0; j < matter_params->HII_DIM; j++)
            for (k = 0; k < HII_D_PARA; k++)
                *((float *)box_unfiltered + HII_R_FFT_INDEX(i, j, k)) =
                    input_box[HII_R_INDEX(i, j, k)];

    dft_r2c_cube(matter_flags->USE_FFTW_WISDOM, matter_params->HII_DIM, HII_D_PARA,
                 matter_params->N_THREADS, box_unfiltered);

    for (ii = 0; ii < HII_KSPACE_NUM_PIXELS; ii++) {
        box_unfiltered[ii] /= (double)HII_TOT_NUM_PIXELS;
    }

    memcpy(box_filtered, box_unfiltered, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    filter_box(box_filtered, 1, filter_flag, R, R_param);

    dft_c2r_cube(matter_flags->USE_FFTW_WISDOM, matter_params->HII_DIM, HII_D_PARA,
                 matter_params->N_THREADS, box_filtered);

    for (i = 0; i < matter_params->HII_DIM; i++)
        for (j = 0; j < matter_params->HII_DIM; j++)
            for (k = 0; k < HII_D_PARA; k++)
                result[HII_R_INDEX(i, j, k)] = *((float *)box_filtered + HII_R_FFT_INDEX(i, j, k));

    fftwf_free(box_unfiltered);
    fftwf_free(box_filtered);

    return 0;
}
