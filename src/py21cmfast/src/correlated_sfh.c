// This file contains the code to generate and evaluate the correlated
// star formation histories (SFH).

#include "correlated_sfh.h"

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <math.h>

#include "Constants.h"
#include "InputParameters.h"
#include "interpolation.h"

#define MAX_TAU (double)(100)                // Myr
#define N_TAU_SFH (double)(1000)             // number of time bins in SFH
#define N_FREQ_SFH (int)(N_TAU_SFH / 2 + 1)  // number of frequency bins in SFH

// We want a singleton struct which holds the SFH correlation functions
// We need the crosses between four timescales: 10 Myr, 100 Myr, snapshot interval, previous
// snapshot interval
typedef struct SFH_Correlation {
    RGTable1D *corr_10_10;
    RGTable1D *corr_10_100;
    RGTable1D *corr_100_100;
    RGTable1D *corr_10_curr;
    RGTable1D *corr_100_curr;
    RGTable1D *corr_curr_curr;
    RGTable1D *corr_10_prev;
    RGTable1D *corr_100_prev;
    RGTable1D *corr_curr_prev;
    RGTable1D *corr_prev_prev;
} SFH_Correlation;

static SFH_Correlation sfh_corr;

// Make a union type for easy initialisation
typedef union sfh_c_u {
    SFH_Correlation sfh_c_s;
    RGTable1D *sfh_c_a[10];
} sfh_c_u;

// Carvajal-Bohorquez et al. 2025 form
// Normalised to 1 for later multiplication
double psd_sfh_powerlaw(double w) {
    return 1.0 / (1 + pow(w * astro_params_global->tau_SFH, astro_params_global->b_SFH));
}

// NOTE: Since two filters can be shifted by different amounts we keep it general here
fftwf_complex shifted_tophat_1d(double wt) {
    // fourier transform of a real space tophat shift in the positive direction by R/2
    // We use this in the SFH model to get SFR between now and R Myr ago
    fftwf_complex result;
    if (wt < 1e-4)
        return 1.0 - 0.5 * I * wt - wt * wt / 6.0;  // second order taylor expansion around kR==0
    return I / wt * (exp(-I * wt) - 1.);
}

void initialise_sfh_correlation(double z, double z_prev, double z_prev_2) {
    sfh_c_u sfh_corr_u;
    RGTable1D *table_ptr;
    sfh_corr_u.sfh_c_s = sfh_corr;

    for (int i = 0; i < 9; i++) {
        table_ptr = sfh_corr_u.sfh_c_a[i];
        allocate_RGTable1D(N_FREQ_SFH, table_ptr);
        table_ptr->x_min = 0.;
        table_ptr->x_width = MAX_TAU / (N_TAU_SFH - 1);
        table_ptr->n_bin = N_TAU_SFH;
    }

    double w_arr[N_FREQ_SFH];
    for (int i = 0; i < N_FREQ_SFH; i++) {
        w_arr[i] = 2 * M_PI * i / MAX_TAU;
    }

    double t_snap = time_between_z(z, z_prev) / (physconst.s_per_yr * 1e6);              // Myr
    double t_snap_prev = time_between_z(z_prev, z_prev_2) / (physconst.s_per_yr * 1e6);  // Myr

    // determine the SFH correlation functions
    fftwf_complex W_10[N_FREQ_SFH], W_100[N_FREQ_SFH], W_curr[N_FREQ_SFH], W_prev[N_FREQ_SFH];
    fftwf_complex psd_unfiltered[N_FREQ_SFH], psd_filtered[N_FREQ_SFH];

    fftwf_complex *in;
    float *out;
    in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_FREQ_SFH);
    out = (float *)calloc(N_FREQ_SFH, sizeof(float));
    fftwf_plan p = fftwf_plan_dft_c2r_1d(N_TAU_SFH, in, out, FFTW_ESTIMATE);

    // get the filters
    for (int i = 0; i < N_FREQ_SFH; i++) {
        psd_unfiltered[i] = psd_sfh_powerlaw(w_arr[i]);
        W_10[i] = shifted_tophat_1d(w_arr[i] * 10.);
        W_100[i] = shifted_tophat_1d(w_arr[i] * 100.);
        W_curr[i] = shifted_tophat_1d(w_arr[i] * t_snap);
        W_prev[i] = shifted_tophat_1d(w_arr[i] * t_snap_prev);
    }

    // TODO: check that there N_TAU_SFH normalization on the iFFT
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_10[i] * conj(W_10[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_10_10->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // 10Myr X 100Myr
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_100[i] * conj(W_10[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_10_100->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // 100Myr X 100Myr
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_100[i] * conj(W_100[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_100_100->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // 10Myr X Current snapshot length
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_10[i] * conj(W_curr[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_10_curr->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // 100Myr X Current snapshot length
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_100[i] * conj(W_curr[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_100_curr->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // Current snapshot length X Current snapshot length
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_curr[i] * conj(W_curr[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_curr_curr->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // 10Myr X Previous snapshot length
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_10[i] * conj(W_prev[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_10_prev->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // 100Myr X Previous snapshot length
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_100[i] * conj(W_prev[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_100_prev->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // Current snapshot length X Previous snapshot length
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_curr[i] * conj(W_prev[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_curr_prev->y_arr[i] = out[i] / N_TAU_SFH;
    }

    // Previous snapshot length X Previous snapshot length
    for (int i = 0; i < N_FREQ_SFH; i++) {
        in[i] = psd_unfiltered[i] * W_prev[i] * conj(W_prev[i]);
    }
    fftwf_execute(p);
    for (int i = 0; i < N_TAU_SFH; i++) {
        sfh_corr.corr_prev_prev->y_arr[i] = out[i] / N_TAU_SFH;
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
}

void eval_sfh_moments(double tau, gsl_matrix *out_chol_cov, gsl_matrix *out_mean_correction) {
    /* Evaluate the SFH covariance matrix at a given time step tau

    Outputs are the two matrices required for sampling the SFR correctly

    out_chol_cov = Cholesky factor of Cov(curr|prev), multiplies standard normal vector
    to get correlated & conditioned SFRs. NOTE: The upper triangle is garbage

    out_mean_correction = multiplies the condition vector, to be added to the correlated samples.
    */

    // Interpolate correlation functions at lag tau
    double cov_10_10_tau = EvaluateRGTable1D(tau, sfh_corr.corr_10_10);
    double cov_10_100_tau = EvaluateRGTable1D(tau, sfh_corr.corr_10_100);
    double cov_10_curr_tau = EvaluateRGTable1D(tau, sfh_corr.corr_10_curr);
    double cov_10_prev_tau = EvaluateRGTable1D(tau, sfh_corr.corr_10_prev);
    double cov_100_100_tau = EvaluateRGTable1D(tau, sfh_corr.corr_100_100);
    double cov_100_curr_tau = EvaluateRGTable1D(tau, sfh_corr.corr_100_curr);
    double cov_100_prev_tau = EvaluateRGTable1D(tau, sfh_corr.corr_100_prev);
    double cov_curr_prev_tau = EvaluateRGTable1D(tau, sfh_corr.corr_curr_prev);

    // Get zero-lag correlations
    double cov_10_10_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_10_10);
    double cov_10_100_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_10_100);
    double cov_10_curr_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_10_curr);
    double cov_10_prev_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_10_prev);
    double cov_100_100_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_100_100);
    double cov_100_curr_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_100_curr);
    double cov_100_prev_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_100_prev);
    double cov_curr_curr_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_curr_curr);
    double cov_prev_prev_zero = EvaluateRGTable1D(0.0, sfh_corr.corr_prev_prev);

    // Previous Snapshot covariance matrix
    gsl_matrix *prev_cov = gsl_matrix_alloc(3, 3);
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
    gsl_matrix *cross_cov = gsl_matrix_alloc(3, 3);
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
    gsl_matrix *curr_cov = gsl_matrix_alloc(3, 3);
    // NOTE: Since the snapshot lengths are different, the snap variances are different
    gsl_matrix_memcpy(curr_cov, prev_cov);
    gsl_matrix_set(curr_cov, 0, 2, cov_10_curr_zero);    // 10_curr vs 10_curr
    gsl_matrix_set(curr_cov, 1, 2, cov_100_curr_zero);   // 100_curr vs 100_curr
    gsl_matrix_set(curr_cov, 2, 0, cov_10_curr_zero);    // 10_curr vs 10_curr
    gsl_matrix_set(curr_cov, 2, 1, cov_100_curr_zero);   // 100_curr vs 100_curr
    gsl_matrix_set(curr_cov, 2, 2, cov_curr_curr_zero);  // snap_curr vs snap_curr

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

void free_sfh_correlation() {
    free_RGTable1D(sfh_corr.corr_10_10);
    free_RGTable1D(sfh_corr.corr_10_100);
    free_RGTable1D(sfh_corr.corr_100_100);
    free_RGTable1D(sfh_corr.corr_10_curr);
    free_RGTable1D(sfh_corr.corr_100_curr);
    free_RGTable1D(sfh_corr.corr_curr_curr);
    free_RGTable1D(sfh_corr.corr_10_prev);
    free_RGTable1D(sfh_corr.corr_100_prev);
    free_RGTable1D(sfh_corr.corr_curr_prev);
    free_RGTable1D(sfh_corr.corr_prev_prev);
}
