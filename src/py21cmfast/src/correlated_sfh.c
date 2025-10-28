// This file contains the code to generate and evaluate the correlated
// star formation histories (SFH).

#include "correlated_sfh.h"

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <math.h>

#include "Constants.h"
#include "InputParameters.h"
#include "cosmology.h"
#include "exceptions.h"
#include "interpolation.h"
#include "logger.h"

#define MAX_TAU (double)(2000)               // Myr
#define N_TAU_SFH (int)(10000)               // number of time bins in SFH
#define N_FREQ_SFH (int)(N_TAU_SFH / 2 + 1)  // number of frequency bins in SFH

// We want a singleton struct which holds the SFH correlation functions
// We need the crosses between four timescales: 10 Myr, 100 Myr, snapshot interval, previous
// snapshot interval
typedef struct SFH_Correlation {
    RGTable1D corr_10_10;
    RGTable1D corr_10_100;
    RGTable1D corr_100_100;
    RGTable1D corr_10_curr;
    RGTable1D corr_100_curr;
    RGTable1D corr_curr_curr;
    RGTable1D corr_10_prev;
    RGTable1D corr_100_prev;
    RGTable1D corr_curr_prev;
    RGTable1D corr_prev_prev;
    RGTable1D corr_zero;  // debug table
} SFH_Correlation;

static SFH_Correlation sfh_corr;

// Carvajal-Bohorquez et al. 2025 form
double psd_sfh_powerlaw(double w) {
    // re-using SIGMA_STAR here for the variance normalisation
    //  (FFT of un-smoothed auto-variance at zero lag)
    //  Our normalisation is such that integral of PSD over all frequencies = variance
    double norm = astro_params_global->SIGMA_STAR * astro_params_global->SIGMA_STAR *
                  astro_params_global->SFH_TAU * 2.;

    // TODO: the x2 factor at the end is matching the expected normalisation, but my derivation
    //  gives sqrt(2/pi) instead. Need to double check. the difference of sqrt(2pi) is likely
    //  a fourier convention thing but I need to verify.

    return norm / (1 + pow(w * astro_params_global->SFH_TAU, astro_params_global->SFH_INDEX));
}

double integral_expfunc_pos(double s_min, double s_max, double t0, double A, double B) {
    // Integral of (A + Bs)*exp(-s/t0) from s_min to s_max
    // used to build the integral of (A + Bs)*exp(-|s|/t0) when s is positive
    double exp_min = exp(-s_min / t0);
    double exp_max = exp(-s_max / t0);
    return t0 * (exp_min * (A + B * (s_min + t0)) - exp_max * (A + B * (s_max + t0)));
}

double integral_expfunc_neg(double s_min, double s_max, double t0, double A, double B) {
    // Integral of (A + Bs)*exp(s/t0) from s_min to s_max
    // used to build the integral of (A + Bs)*exp(-|s|/t0) when s is negative
    double exp_min = exp(s_min / t0);
    double exp_max = exp(s_max / t0);
    return -t0 * (exp_min * (A + B * (s_min - t0)) - exp_max * (A + B * (s_max - t0)));
}

double integral_expfunc_mod(double s_min, double s_max, double t0, double A, double B) {
    // Integral of (A + Bs)*exp(-|s|/t0) from s_min to s_max
    // used to build the integral of (A + Bs)*exp(-|s|/t0)
    double result = 0.0;
    if (s_max < 0.0) {
        // entirely negative
        result = integral_expfunc_neg(s_min, s_max, t0, A, B);
    } else if (s_min > 0.0) {
        // entirely positive
        result = integral_expfunc_pos(s_min, s_max, t0, A, B);
    } else {
        // crosses zero
        result =
            integral_expfunc_neg(s_min, 0.0, t0, A, B) + integral_expfunc_pos(0.0, s_max, t0, A, B);
    }
    return result;
}

double smoothed_correlation_func(double tau, double t1, double t2) {
    // Analytic SFH correlation function for two smoothed exponentials
    // with timescales t1 and t2 respectively, at lag tau
    double t0 = astro_params_global->SFH_TAU;
    double sigma = astro_params_global->SIGMA_STAR;  // re-using this parameter for now
    double result = 0.0;

    double first_boundary = fmin(tau, tau + t1 - t2);
    double second_boundary = fmax(tau, tau + t1 - t2);

    // The first and second regions *may* have negative components
    // Region 1
    if (tau - t2 < first_boundary) {
        result += integral_expfunc_mod(tau - t2, first_boundary, t0, t2 - tau, 1.0);
    }

    // Region 2
    if (first_boundary < second_boundary) {
        result += integral_expfunc_mod(first_boundary, second_boundary, t0, fmin(t1, t2), 0.0);
    }

    // Region 3 (second boundary is always positive)
    if (second_boundary < tau + t1) {
        result += integral_expfunc_pos(second_boundary, tau + t1, t0, t1 + tau, -1.0);
    }

    return sigma * sigma * result / (t1 * t2);
}

void fill_covar_analytic(double tau, double tau_prev, gsl_matrix *curr_cov, gsl_matrix *prev_cov,
                         gsl_matrix *cross_cov) {
    /* Fill the covariance matrices required for sampling the SFR correctly */
    double cov_10_10_tau = smoothed_correlation_func(tau, 10.0, 10.0);
    double cov_10_100_tau = smoothed_correlation_func(tau, 10.0, 100.0);
    double cov_10_curr_tau = smoothed_correlation_func(tau, 10.0, tau);
    double cov_10_prev_tau = smoothed_correlation_func(tau, 10.0, tau_prev);
    double cov_100_100_tau = smoothed_correlation_func(tau, 100.0, 100.0);
    double cov_100_curr_tau = smoothed_correlation_func(tau, 100.0, tau);
    double cov_100_prev_tau = smoothed_correlation_func(tau, 100.0, tau_prev);
    double cov_curr_prev_tau = smoothed_correlation_func(tau, tau, tau_prev);

    // Get zero-lag correlations
    double cov_10_10_zero = smoothed_correlation_func(0.0, 10.0, 10.0);
    double cov_10_100_zero = smoothed_correlation_func(0.0, 10.0, 100.0);
    double cov_10_curr_zero = smoothed_correlation_func(0.0, 10.0, tau);
    double cov_10_prev_zero = smoothed_correlation_func(0.0, 10.0, tau_prev);
    double cov_100_100_zero = smoothed_correlation_func(0.0, 100.0, 100.0);
    double cov_100_curr_zero = smoothed_correlation_func(0.0, 100.0, tau);
    double cov_100_prev_zero = smoothed_correlation_func(0.0, 100.0, tau_prev);
    double cov_curr_curr_zero = smoothed_correlation_func(0.0, tau, tau);
    double cov_prev_prev_zero = smoothed_correlation_func(0.0, tau_prev, tau_prev);

    // Previous Snapshot covariance matrix
    gsl_matrix_set(prev_cov, 0, 0, cov_10_10_zero);      // 10_prev vs 10_prev
    gsl_matrix_set(prev_cov, 0, 1, cov_10_100_zero);     // 10_prev vs 100_prev
    gsl_matrix_set(prev_cov, 0, 2, cov_10_prev_zero);    // 10_prev vs snap_prev
    gsl_matrix_set(prev_cov, 1, 0, cov_10_100_zero);     // 100_prev vs 10_prev
    gsl_matrix_set(prev_cov, 1, 1, cov_100_100_zero);    // 100_prev vs 100_prev
    gsl_matrix_set(prev_cov, 1, 2, cov_100_prev_zero);   // 100_prev vs snap_prev
    gsl_matrix_set(prev_cov, 2, 0, cov_10_prev_zero);    // snap_prev vs 10_prev
    gsl_matrix_set(prev_cov, 2, 1, cov_100_prev_zero);   // snap_prev vs 100_prev
    gsl_matrix_set(prev_cov, 2, 2, cov_prev_prev_zero);  // snap_prev vs snap_prev

    // Lower left corner Covariance is Cov(curr,prev) == Cov(prev,curr)^T
    gsl_matrix_set(cross_cov, 0, 0, cov_10_10_tau);      // 10_prev vs 10_curr
    gsl_matrix_set(cross_cov, 0, 1, cov_10_100_tau);     // 10_prev vs 100_curr
    gsl_matrix_set(cross_cov, 0, 2, cov_10_prev_tau);    // 10_prev vs snap_curr
    gsl_matrix_set(cross_cov, 1, 0, cov_10_100_tau);     // 100_prev vs 10_curr
    gsl_matrix_set(cross_cov, 1, 1, cov_100_100_tau);    // 100_prev vs 100_curr
    gsl_matrix_set(cross_cov, 1, 2, cov_100_prev_tau);   // 100_prev vs snap_curr
    gsl_matrix_set(cross_cov, 2, 0, cov_10_curr_tau);    // snap_prev vs 10_curr
    gsl_matrix_set(cross_cov, 2, 1, cov_100_curr_tau);   // snap_prev vs 100_curr
    gsl_matrix_set(cross_cov, 2, 2, cov_curr_prev_tau);  // snap_prev vs snap_curr

    // Current Snapshot covariance matrix
    // NOTE: Since the snapshot lengths are different, the snap variances are different
    gsl_matrix_memcpy(curr_cov, prev_cov);
    gsl_matrix_set(curr_cov, 0, 2, cov_10_curr_zero);    // 10_curr vs 10_curr
    gsl_matrix_set(curr_cov, 1, 2, cov_100_curr_zero);   // 100_curr vs 100_curr
    gsl_matrix_set(curr_cov, 2, 0, cov_10_curr_zero);    // 10_curr vs 10_curr
    gsl_matrix_set(curr_cov, 2, 1, cov_100_curr_zero);   // 100_curr vs 100_curr
    gsl_matrix_set(curr_cov, 2, 2, cov_curr_curr_zero);  // snap_curr vs snap_curr
}

// NOTE: This only differs from a tophat in phase. While we only use filter squared in the
// correlation functions, Since two filters can be shifted by different amounts (due to width)
// we keep the shifting
fftwf_complex shifted_tophat_1d(double wt) {
    // fourier transform of a real space tophat shift in the positive direction by R/2
    // We use this in the SFH model to get SFR between now and R Myr ago
    fftwf_complex result;
    if (wt < 1e-4)
        result = 1.0 + 0.5 * I * wt - wt * wt / 6.0;  // second order taylor expansion around kR==0
    else
        result = -I * (cexp(I * wt) - 1.) / wt;
    return result;
}

// TODO: I only really need a single tau per snapshot, BUT since there is no analyic iFFT for the
// PSD
//  filtered by the shifted tophat, I *think* we need to do the full iFFT anyway
//  to get the correlation function, even at a single tau
void initialise_psd_corrfunc_tables(double tau, double tau_prev) {
    // For easy initialisation, we set arrays of the tables and delays used
    RGTable1D *table_ptrs[11] = {
        &sfh_corr.corr_10_10,     &sfh_corr.corr_10_100,    &sfh_corr.corr_10_curr,
        &sfh_corr.corr_10_prev,   &sfh_corr.corr_100_100,   &sfh_corr.corr_100_curr,
        &sfh_corr.corr_100_prev,  &sfh_corr.corr_curr_curr, &sfh_corr.corr_curr_prev,
        &sfh_corr.corr_prev_prev, &sfh_corr.corr_zero};

    double delays[11][2] = {{10., 10.},           {10., 100.},  {10., tau},
                            {10., tau_prev},      {100., 100.}, {100., tau},
                            {100., tau_prev},     {tau, tau},   {tau, tau_prev},
                            {tau_prev, tau_prev}, {0., 0.}};

    for (int i = 0; i < 11; i++) {
        allocate_RGTable1D(N_TAU_SFH, table_ptrs[i]);
        table_ptrs[i]->x_min = 0.;
        table_ptrs[i]->x_width = MAX_TAU / (N_TAU_SFH - 1);
    }

    double w_arr[N_FREQ_SFH];
    for (int i = 0; i < N_FREQ_SFH; i++) {
        w_arr[i] = 2 * M_PI * i / MAX_TAU;
    }

    fftwf_complex *in;
    float *out;
    in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_FREQ_SFH);
    out = (float *)calloc(N_TAU_SFH, sizeof(float));
    fftwf_plan p = fftwf_plan_dft_c2r_1d(N_TAU_SFH, in, out, FFTW_ESTIMATE);

    // NOTE: it would be more efficient to store all the window functions in arrays prior
    //  i.e calculate them once each (4) instead of once per table (20)
    for (int i = 0; i < 11; i++) {
        double sum = 0.0;
        double t1 = delays[i][0];
        double t2 = delays[i][1];
        RGTable1D *curr_table = table_ptrs[i];
        for (int j = 0; j < N_FREQ_SFH; j++) {
            double w = w_arr[j];
            in[j] = psd_sfh_powerlaw(w) * shifted_tophat_1d(w * t1) *
                    conj(shifted_tophat_1d(w * t2));  // FFTW unnormalised convention

            sum += creal(in[j]);
            if (j > 0) {
                sum += creal(in[j]);  // account for negative frequencies
            }
        }
        fftwf_execute(p);
        sum = 0.0;
        for (int j = 0; j < N_TAU_SFH; j++) {
            curr_table->y_arr[j] = out[j] / MAX_TAU;
            sum += curr_table->y_arr[j];
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    free(out);
}

void fill_covar_from_tables(double tau, gsl_matrix *curr_cov, gsl_matrix *prev_cov,
                            gsl_matrix *cross_cov) {
    /* Fill the covariance matrices required for sampling the SFR correctly */
    // Interpolate correlation functions at lag tau
    double cov_10_10_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_10_10);
    double cov_10_100_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_10_100);
    double cov_10_curr_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_10_curr);
    double cov_10_prev_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_10_prev);
    double cov_100_100_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_100_100);
    double cov_100_curr_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_100_curr);
    double cov_100_prev_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_100_prev);
    double cov_curr_prev_tau = EvaluateRGTable1D(tau, &sfh_corr.corr_curr_prev);

    // Get zero-lag correlations
    double cov_10_10_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_10_10);
    double cov_10_100_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_10_100);
    double cov_10_curr_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_10_curr);
    double cov_10_prev_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_10_prev);
    double cov_100_100_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_100_100);
    double cov_100_curr_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_100_curr);
    double cov_100_prev_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_100_prev);
    double cov_curr_curr_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_curr_curr);
    double cov_prev_prev_zero = EvaluateRGTable1D(0.0, &sfh_corr.corr_prev_prev);

    // Previous Snapshot covariance matrix
    gsl_matrix_set(prev_cov, 0, 0, cov_10_10_zero);      // 10_prev vs 10_prev
    gsl_matrix_set(prev_cov, 0, 1, cov_10_100_zero);     // 10_prev vs 100_prev
    gsl_matrix_set(prev_cov, 0, 2, cov_10_prev_zero);    // 10_prev vs snap_prev
    gsl_matrix_set(prev_cov, 1, 0, cov_10_100_zero);     // 100_prev vs 10_prev
    gsl_matrix_set(prev_cov, 1, 1, cov_100_100_zero);    // 100_prev vs 100_prev
    gsl_matrix_set(prev_cov, 1, 2, cov_100_prev_zero);   // 100_prev vs snap_prev
    gsl_matrix_set(prev_cov, 2, 0, cov_10_prev_zero);    // snap_prev vs 10_prev
    gsl_matrix_set(prev_cov, 2, 1, cov_100_prev_zero);   // snap_prev vs 100_prev
    gsl_matrix_set(prev_cov, 2, 2, cov_prev_prev_zero);  // snap_prev vs snap_prev

    // Lower left corner Covariance is Cov(curr,prev) == Cov(prev,curr)^T
    gsl_matrix_set(cross_cov, 0, 0, cov_10_10_tau);      // 10_prev vs 10_curr
    gsl_matrix_set(cross_cov, 0, 1, cov_10_100_tau);     // 10_prev vs 100_curr
    gsl_matrix_set(cross_cov, 0, 2, cov_10_prev_tau);    // 10_prev vs snap_curr
    gsl_matrix_set(cross_cov, 1, 0, cov_10_100_tau);     // 100_prev vs 10_curr
    gsl_matrix_set(cross_cov, 1, 1, cov_100_100_tau);    // 100_prev vs 100_curr
    gsl_matrix_set(cross_cov, 1, 2, cov_100_prev_tau);   // 100_prev vs snap_curr
    gsl_matrix_set(cross_cov, 2, 0, cov_10_curr_tau);    // snap_prev vs 10_curr
    gsl_matrix_set(cross_cov, 2, 1, cov_100_curr_tau);   // snap_prev vs 100_curr
    gsl_matrix_set(cross_cov, 2, 2, cov_curr_prev_tau);  // snap_prev vs snap_curr

    // Current Snapshot covariance matrix
    // NOTE: Since the snapshot lengths are different, the snap variances are different
    gsl_matrix_memcpy(curr_cov, prev_cov);
    gsl_matrix_set(curr_cov, 0, 2, cov_10_curr_zero);    // 10_curr vs 10_curr
    gsl_matrix_set(curr_cov, 1, 2, cov_100_curr_zero);   // 100_curr vs 100_curr
    gsl_matrix_set(curr_cov, 2, 0, cov_10_curr_zero);    // 10_curr vs 10_curr
    gsl_matrix_set(curr_cov, 2, 1, cov_100_curr_zero);   // 100_curr vs 100_curr
    gsl_matrix_set(curr_cov, 2, 2, cov_curr_curr_zero);  // snap_curr vs snap_curr
}

void eval_sfh_moments(gsl_matrix *prev_cov, gsl_matrix *curr_cov, gsl_matrix *cross_cov,
                      gsl_matrix *out_chol_cov, gsl_matrix *out_mean_correction) {
    /* Evaluate the SFH covariance matrix at a given time step tau

    Outputs are the two matrices required for sampling the SFR correctly

    out_chol_cov = Cholesky factor of Cov(curr|prev), multiplies standard normal vector
    to get correlated & conditioned SFRs. NOTE: The upper triangle is garbage

    out_mean_correction = multiplies the condition vector, to be added to the correlated samples.
    */
    // NOTE: Currently Cov(curr,prev) == Cov(prev,curr) == Cov(prev,curr)^T == Cov(curr,prev)^T
    gsl_matrix *matrix_buf = gsl_matrix_alloc(3, 3);

    // Cholesky factorization of Cov(prev) = L L^T to do implicit inversion
    gsl_linalg_cholesky_decomp1(prev_cov);  // holds L

    // Compute the conditional covariance matrix Cov(curr|prev) = Cov(curr) - Cov(curr,prev)
    // Cov(prev)^-1 Cov(prev,curr)
    gsl_matrix_memcpy(matrix_buf, cross_cov);
    // L^-1  Cov(curr,prev)
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1.0, prev_cov, matrix_buf);

    // compute Cov(curr) - BUF^T*BUF == Cov(curr) - Cov(curr|prev) Cov(prev)^-1 Cov(prev,curr)
    // NOTE, curr_cov is symmetric here
    gsl_blas_dsyrk(CblasLower, CblasTrans, -1.0, matrix_buf, 1.0, curr_cov);
    // The lower triangle of curr_cov now holds Cov(curr|prev)

    // Perform Cholesky decomposition (only uses lower triangle)
    gsl_linalg_cholesky_decomp1(curr_cov);
    gsl_matrix_memcpy(out_chol_cov, curr_cov);

    // Now Compute the mean correction term, since BUF was preserved from the rank-k above
    //  Compute Cov(prev,curr) L^-T^-1 L^-1 = Cov(curr,prev) Cov(prev)^-1
    gsl_matrix_transpose(matrix_buf);
    gsl_blas_dtrsm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, 1.0, prev_cov, matrix_buf);

    // Since, for zero mean, E[X|Y] = Cov(X,Y) Cov(Y)^-1 Y
    gsl_matrix_memcpy(out_mean_correction, matrix_buf);  // store for output

    gsl_matrix_free(prev_cov);
    gsl_matrix_free(curr_cov);
    gsl_matrix_free(cross_cov);
    gsl_matrix_free(matrix_buf);
}

void sample_correlated_sfh(gsl_rng *rng, double prev_values[3], gsl_matrix *L_cov,
                           gsl_matrix *mean_corr, double out_values[3]) {
    /* Sample correlated SFH values given previous values

    Inputs:
    prev_values: array of previous SFR values [SFR_10Myr, SFR_100Myr, SFR_snapshot_prev]
    L_cov: Cholesky factor of Cov(curr|prev), from eval_sfh_moments, L L^T = Cov(curr|prev)
    mean_corr: Mean correction matrix from eval_sfh_moments

    Outputs:
    out_values: array of sampled current SFR values [SFR_10Myr, SFR_100Myr, SFR_snapshot_curr]
    */

    // Generate standard normal random variables
    gsl_vector *cov_term = gsl_vector_alloc(3);
    for (int i = 0; i < 3; i++) {
        gsl_vector_set(cov_term, i, gsl_ran_ugaussian(rng));
    }

    // Multiply by Cholesky factor to get correlated samples
    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, L_cov, cov_term);

    // Create a vector for the conditioned samples
    gsl_vector *cond_term = gsl_vector_alloc(3);
    for (int i = 0; i < 3; i++) {
        gsl_vector_set(cond_term, i, prev_values[i]);
    }
    // Add the mean correction term
    gsl_blas_dgemv(CblasNoTrans, 1.0, mean_corr, cond_term, 1.0, cov_term);

    // Copy to output
    for (int i = 0; i < 3; i++) {
        out_values[i] = gsl_vector_get(cov_term, i);
    }
    gsl_vector_free(cov_term);
    gsl_vector_free(cond_term);
}

void free_sfh_correlation() {
    free_RGTable1D(&sfh_corr.corr_10_10);
    free_RGTable1D(&sfh_corr.corr_10_100);
    free_RGTable1D(&sfh_corr.corr_100_100);
    free_RGTable1D(&sfh_corr.corr_10_curr);
    free_RGTable1D(&sfh_corr.corr_100_curr);
    free_RGTable1D(&sfh_corr.corr_curr_curr);
    free_RGTable1D(&sfh_corr.corr_10_prev);
    free_RGTable1D(&sfh_corr.corr_100_prev);
    free_RGTable1D(&sfh_corr.corr_curr_prev);
    free_RGTable1D(&sfh_corr.corr_prev_prev);
    free_RGTable1D(&sfh_corr.corr_zero);
}

void print_gsl_matrix(gsl_matrix *mat, const char *label) {
    int nrows = mat->size1;
    int ncols = mat->size2;
    fprintf(stdout, "%s\n", label);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fprintf(stdout, "%9.4f ", gsl_matrix_get(mat, i, j));
        }
        fprintf(stdout, "\n");
    }
}

int test_sfh_corr(double z0, double z1, double z2) {
    fprintf(stdout, "Testing SFH matrices at z=%f, %f, %f\n", z0, z1, z2);
    double tau = time_between_z(z0, z1) / (physconst.s_per_yr * 1e6);       // Myr
    double tau_prev = time_between_z(z1, z2) / (physconst.s_per_yr * 1e6);  // Myr
    initialise_psd_corrfunc_tables(tau, tau_prev);

    RGTable1D *table_ptrs[11] = {
        &sfh_corr.corr_10_10,     &sfh_corr.corr_10_100,    &sfh_corr.corr_10_curr,
        &sfh_corr.corr_10_prev,   &sfh_corr.corr_100_100,   &sfh_corr.corr_100_curr,
        &sfh_corr.corr_100_prev,  &sfh_corr.corr_curr_curr, &sfh_corr.corr_curr_prev,
        &sfh_corr.corr_prev_prev, &sfh_corr.corr_zero};

    double delays[11][2] = {{10., 10.},           {10., 100.},  {10., tau},
                            {10., tau_prev},      {100., 100.}, {100., tau},
                            {100., tau_prev},     {tau, tau},   {tau, tau_prev},
                            {tau_prev, tau_prev}, {0., 0.}};

    char names[11][20] = {"10Myr x 10Myr",   "10Myr x 100Myr", "10Myr x Curr",  "10Myr x Prev",
                          "100Myr x 100Myr", "100Myr x Curr",  "100Myr x Prev", "Curr x Curr",
                          "Curr x Prev",     "Prev x Prev",    "Zero Lag"};

    fprintf(stdout, "====== Correlation functions ======\n");
    for (int i = 0; i < 10; i++) {
        fprintf(stdout, "%s (t1=%f, t2=%f):\n", names[i], delays[i][0], delays[i][1]);
        for (double lag = 0.0; lag <= 500.0; lag += 50.0) {
            double val_tables = EvaluateRGTable1D(lag, table_ptrs[i]);
            double val_analytic = smoothed_correlation_func(lag, delays[i][0], delays[i][1]);
            fprintf(stdout, "  Lag %7.2f Tables: %9.4f Analytic: %9.4f Ratio: %9.4f\n", lag,
                    val_tables, val_analytic, val_tables / val_analytic);
        }
    }
    fprintf(stdout, "===== zero smoothing =====\n");
    for (double lag = 0.0; lag <= 500.0; lag += 50.0) {
        double val_tables = EvaluateRGTable1D(lag, &sfh_corr.corr_zero);
        double val_analytic = exp(-fabs(lag) / astro_params_global->SFH_TAU) *
                              astro_params_global->SIGMA_STAR * astro_params_global->SIGMA_STAR;
        fprintf(stdout, "  Lag %7.2f Tables: %9.4f Analytic: %9.4f Ratio: %9.4f\n", lag, val_tables,
                val_analytic, val_tables / val_analytic);
    }

    fprintf(stdout, "====== PSD Covariance Matrices ======\n");
    gsl_matrix *curr_cov = gsl_matrix_alloc(3, 3);
    gsl_matrix *prev_cov = gsl_matrix_alloc(3, 3);
    gsl_matrix *cross_cov = gsl_matrix_alloc(3, 3);
    fill_covar_from_tables(tau, curr_cov, prev_cov, cross_cov);
    print_gsl_matrix(prev_cov, "Previous Covariance Matrix:");
    print_gsl_matrix(cross_cov, "Cross Covariance Matrix:");
    print_gsl_matrix(curr_cov, "Current Covariance Matrix:");

    fprintf(stdout, "====== Analytic Covariance Matrices ======\n");
    gsl_matrix *curr_cov_2 = gsl_matrix_alloc(3, 3);
    gsl_matrix *prev_cov_2 = gsl_matrix_alloc(3, 3);
    gsl_matrix *cross_cov_2 = gsl_matrix_alloc(3, 3);
    fill_covar_analytic(tau, tau_prev, curr_cov_2, prev_cov_2, cross_cov_2);
    print_gsl_matrix(prev_cov_2, "Previous Covariance Matrix:");
    print_gsl_matrix(cross_cov_2, "Cross Covariance Matrix:");
    print_gsl_matrix(curr_cov_2, "Current Covariance Matrix:");

    gsl_matrix_div_elements(prev_cov_2, prev_cov);
    gsl_matrix_div_elements(cross_cov_2, cross_cov);
    gsl_matrix_div_elements(curr_cov_2, curr_cov);
    print_gsl_matrix(prev_cov_2, "Previous Covariance Matrix Ratio:");
    print_gsl_matrix(cross_cov_2, "Cross Covariance Matrix Ratio:");
    print_gsl_matrix(curr_cov_2, "Current Covariance Matrix Ratio:");

    gsl_matrix *L_cov = gsl_matrix_alloc(3, 3);
    gsl_matrix *mean_corr = gsl_matrix_alloc(3, 3);
    eval_sfh_moments(prev_cov, curr_cov, cross_cov, L_cov, mean_corr);
    print_gsl_matrix(mean_corr, "Mean Correction Matrix:");
    print_gsl_matrix(L_cov, "Cholesky Factor of Conditioned Covariance Matrix:");

    // get the conditioned covariance matrix
    gsl_blas_dtrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, 1.0, L_cov, L_cov);
    print_gsl_matrix(L_cov, "Conditioned Covariance Matrix:");

    free_sfh_correlation();
    return 0;
}
