
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_sf_gamma.h>
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
        LOG_ERROR("No such filter for dWdM: %i", matter_options_global->FILTER);
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

double spherical_shell_filter(double k, double R_inner, double R_outer) {
    double kR_inner = k * R_inner;
    double kR_outer = k * R_outer;

    // Second order taylor expansion around kR_outer==0
    if (kR_outer < 1e-4)
        return 1. - kR_outer * kR_outer / 10 * (pow(R_inner / R_outer, 5) - 1) /
                        (pow(R_inner / R_outer, 3) - 1);

    return 3.0 / (pow(kR_outer, 3) - pow(kR_inner, 3)) *
           (sin(kR_outer) - cos(kR_outer) * kR_outer - sin(kR_inner) + cos(kR_inner) * kR_inner);
}

struct multiple_scattering_params {
    double alpha_outer;
    double beta_outer;
    double alpha_inner;
    double beta_inner;
};

void compute_alpha_and_beta_for_multiple_scattering(double R_SL, double r_star,
                                                    struct multiple_scattering_params *consts) {
    double x_em = R_SL / r_star;
    double zeta_em = log10(x_em);
    double mu, eta;
    if (x_em > 30) {
        mu = 1. - 1.0478 * pow(x_em, -0.7266);
    } else if (x_em > 3.) {
        mu = -0.104 * pow(zeta_em, 5) + 0.4867 * pow(zeta_em, 4) - 0.8217 * pow(zeta_em, 3) +
             0.4889 * zeta_em * zeta_em + 0.264 * zeta_em + 0.518;
    } else if (x_em > 0.2) {
        mu = -0.0285 * pow(zeta_em, 5) + 0.087 * pow(zeta_em, 4) - 0.1205 * pow(zeta_em, 3) -
             0.0456 * zeta_em * zeta_em + 0.3787 * zeta_em + 0.5285;
    } else {
        mu = 0.3982 * pow(x_em, 0.1592);
    }
    if (x_em > 20.) {
        eta = 1. - 2.804 * pow(x_em, -1.242);
    } else if (x_em > 3.) {
        eta = 2.17 * pow(zeta_em, 5) - 8.832 * pow(zeta_em, 4) + 13.579 * pow(zeta_em, 3) -
              10.04 * zeta_em * zeta_em + 4.166 * zeta_em - 0.17;
    } else if (x_em > 0.2) {
        eta = 0.352 * pow(zeta_em, 5) - 0.0516 * pow(zeta_em, 4) - 0.293 * pow(zeta_em, 3) +
              0.342 * zeta_em * zeta_em + 0.582 * zeta_em + 0.266;
    } else {
        eta = 0.4453 * pow(x_em, 1.296);
    }
    // mu = alpha/(alpha+beta), eta = alpha/(alpha+beta^2)
    consts->alpha_outer = (1. / eta - 1.) / pow(1. / mu - 1., 2);
    consts->beta_outer = (1. / eta - 1.) / (1. / mu - 1.);
}

void initialize_alphas_and_betas_for_multiple_scattering(
    double R_inner, double R_outer, double r_star, struct multiple_scattering_params *consts) {
    struct multiple_scattering_params consts_outer, consts_inner;
    compute_alpha_and_beta_for_multiple_scattering(R_inner, r_star, &consts_inner);
    compute_alpha_and_beta_for_multiple_scattering(R_outer, r_star, &consts_outer);
    consts->alpha_outer = consts_outer.alpha_outer;
    consts->beta_outer = consts_outer.beta_outer;
    // Note that we use the "outer" alpha and beta here.
    // This is because the above function only fills the values of the outer parameters.
    consts->alpha_inner = consts_inner.alpha_outer;
    consts->beta_inner = consts_inner.beta_outer;
}

double asymptotic_2F3(double kR, double alpha, double beta) {
    // Approximation to the hypergeometric function 2F3(a1,a2;b1,b2,b3;-z) for z >> 1,
    // as given in
    // https://functions.wolfram.com/HypergeometricFunctions/Hypergeometric2F3/06/02/03/01/0003/. In
    // our case, z=-kR^2/4 and the parameters of the hypergeometric function are given below
    double a1 = (2. + alpha) / 2.;
    double a2 = (3. + alpha) / 2.;
    double b1 = 5. / 2.;
    double b2 = (2. + alpha + beta) / 2.;
    double b3 = (3. + alpha + beta) / 2.;

    double gamma_a1 = tgamma(a1);
    double gamma_a2 = tgamma(a2);
    double gamma_b1 = 3. / 4.;  // Actually Gamma(5/2) is 3/4*sqrt(pi), but we absorbed the sqrt(pi)
                                // in the other terms below
    double gamma_b2 = tgamma(b2);
    double gamma_b3 = tgamma(b3);

    double gamma_b2_over_a1, gamma_b3_over_a2, decay_term1, decay_term2, F_asymp;
    // If alpha >> beta, the gamma function becomes very large and overflow can happen if they are
    // naively computed. In this limit, we use the following asymptotic approximation, which is
    // based on Stirling's formula
    if (a1 < 20.) {
        gamma_b2_over_a1 = gamma_b2 / gamma_a1;
        gamma_b3_over_a2 = gamma_b3 / gamma_a2;
    } else {
        double y = beta / 2;  // b2-a1 = b3-a2
        gamma_b2_over_a1 = pow(a1, y) * exp((a1 + y - 0.5) * (y / a1 - y * y / (2. * a1 * a1) +
                                                              y * y * y / (3. * a1 * a1 * a1)) -
                                            y);
        gamma_b3_over_a2 = pow(a2, y) * exp((a2 + y - 0.5) * (y / a2 - y * y / (2. * a2 * a1) +
                                                              y * y * y / (3. * a2 * a2 * a2)) -
                                            y);
    }
    // If alpha is very big the following decaying terms can be completely neglected
    if (alpha < 10.) {
        // There are some Gamma function whose argument could be close to a pole (non-positive
        // integers), but they are at the denominator, so they behave nicely and we actually need
        // the reciprocal gamma function
        double gamma_b1_minus_a1_inv = gsl_sf_gammainv(b1 - a1);  // 1/Gamma((3-alpha)/2)
        double gamma_b1_minus_a2_inv = gsl_sf_gammainv(b1 - a2);  // 1/Gamma((2-alpha)/2)
        double gamma_b2_minus_a2_inv = gsl_sf_gammainv(b2 - a2);  // 1/Gamma((beta-1)/2)
        decay_term1 = PI * gamma_a1 * gamma_b1_minus_a1_inv / tgamma(b2 - a1) / tgamma(b3 - a1) /
                      pow(kR / 2., alpha + 2.);  // gamma(a2-a1) = sqrt(pi)
        decay_term2 = -2. * PI * gamma_a2 * gamma_b1_minus_a2_inv * gamma_b2_minus_a2_inv /
                      tgamma(b3 - a2) / pow(kR / 2., alpha + 3.);  // gamma(a1-a2) = -2*sqrt(pi)
    } else {
        decay_term1 = 0.;
        decay_term2 = 0.;
    }

    F_asymp = (cos(kR - PI * (2. + beta) / 2.) -
               (1. + (alpha - 1.) * beta) / kR * sin(kR - PI * (2. + beta) / 2.)) /
              pow(kR / 2, beta + 2);
    F_asymp += decay_term1 + decay_term2;
    F_asymp *= gamma_b1 * gamma_b2_over_a1 * gamma_b3_over_a2;

    return F_asymp;
}

// Implementation of 2F3((alpha+2)/2, (alpha+3)/2 ; (alpha+beta+2)/2, (alpha+beta+3)/2 ; -kR^2 /4))
double hyper_2F3(double kR, double alpha, double beta) {
    if (astro_options_global->TEST_SL_WITH_MS_FILTER) {
        alpha = 1.e5;
        beta = 1.;
    }

    // For a small argument, we compute the hypergeometric function through power-law expansion
    if (kR < 30.) {
        int n;
        int max_terms = 1000;
        double sum = 0.;
        double term = 1.;
        for (n = 1; n < max_terms; n++) {
            sum += term;
            term *= -1. / (1. + beta / (alpha + 2. * n)) / (1. + beta / (alpha + 1 + 2. * n)) * kR *
                    kR / (2. * n) / (2. * n + 3.);
            if (fabs(term) < fabs(sum) * 1e-4) {
                break;
            }
        }
        return sum;
    }
    // For large arguments, the above sum becomes numerically unstable, and we use instead
    // asymptotic approximation
    else {
        double F_ms, F_sl;
        F_ms = asymptotic_2F3(kR, alpha, beta);
        F_sl = 3.0 / (pow(kR, 3)) * (sin(kR) - cos(kR) * kR);
        /* At large arguments, the hypergeometric function (multiple scattering window function)
           should be below the straight-line window function. However, for large alpha values the
           asymptotic approximation is not adequate at 30 < kR < 100 and the asymptotic formula
           diverges at this range. We could of course increase the threshold for "big" argument, but
           then the above power-law expansion diverges. I therefore use this rule of thumb, which is
           necessary only for large alpha values (where the hypergeometric function approaches the
           straight-line window function) and intermediate kR values. */
        if (fabs(F_ms) < fabs(F_sl)) {
            return F_ms;
        } else {
            return F_sl;
        }
    }
}

double multiple_scattering_filter(double k, double R_inner, double R_outer,
                                  struct multiple_scattering_params *consts) {
    double kR_inner = k * R_inner;
    double kR_outer = k * R_outer;
    double W;

    W = pow(R_outer, 3.) * hyper_2F3(kR_outer, consts->alpha_outer, consts->beta_outer) -
        pow(R_inner, 3.) * hyper_2F3(kR_inner, consts->alpha_inner, consts->beta_inner);
    W /= pow(R_outer, 3.) - pow(R_inner, 3.);
    return W;
}

void filter_box(fftwf_complex *box, int RES, int filter_type, float R, float R_param,
                float r_star) {
    int dimension, midpoint;  // TODO: figure out why defining as ULL breaks this
    struct multiple_scattering_params consts_for_ms;
    switch (RES) {
        case 0:
            dimension = simulation_options_global->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = simulation_options_global->HII_DIM;
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
    if (filter_type == 5) {
        initialize_alphas_and_betas_for_multiple_scattering(R, R_param, r_star, &consts_for_ms);
    }

// loop through k-box
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
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

                for (n_z = 0; n_z <= (int)(simulation_options_global->NON_CUBIC_FACTOR * midpoint);
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
                    } else if (filter_type == 5) {  // spherical ring
                        box[grid_index] *=
                            multiple_scattering_filter(sqrt(k_mag_sq), R, R_param, &consts_for_ms);
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
int test_filter(float *input_box, double R, double R_param, double r_star, int filter_flag,
                double *result) {
    int i, j, k;
    unsigned long long int ii;

    // setup the box
    fftwf_complex *box_unfiltered =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box_filtered =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    for (i = 0; i < simulation_options_global->HII_DIM; i++)
        for (j = 0; j < simulation_options_global->HII_DIM; j++)
            for (k = 0; k < HII_D_PARA; k++)
                *((float *)box_unfiltered + HII_R_FFT_INDEX(i, j, k)) =
                    input_box[HII_R_INDEX(i, j, k)];

    dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                 HII_D_PARA, simulation_options_global->N_THREADS, box_unfiltered);

    for (ii = 0; ii < HII_KSPACE_NUM_PIXELS; ii++) {
        box_unfiltered[ii] /= (double)HII_TOT_NUM_PIXELS;
    }

    memcpy(box_filtered, box_unfiltered, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    filter_box(box_filtered, 1, filter_flag, R, R_param, r_star);

    dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                 HII_D_PARA, simulation_options_global->N_THREADS, box_filtered);

    for (i = 0; i < simulation_options_global->HII_DIM; i++)
        for (j = 0; j < simulation_options_global->HII_DIM; j++)
            for (k = 0; k < HII_D_PARA; k++)
                result[HII_R_INDEX(i, j, k)] = *((float *)box_filtered + HII_R_FFT_INDEX(i, j, k));

    fftwf_free(box_unfiltered);
    fftwf_free(box_filtered);

    return 0;
}
