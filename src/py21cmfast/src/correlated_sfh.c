// This file contains the code to generate and evaluate the correlated
// star formation histories (SFH).

#include "correlated_sfh.h"

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <math.h>

#include "Constants.h"
#include "InputParameters.h"
#include "cexcept.h"
#include "cosmology.h"
#include "exceptions.h"
#include "interpolation.h"
#include "logger.h"

static const int N_TAU_SFH = 10000;  // number of time bins in SFH

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

typedef struct SFH_matrices {
    gsl_matrix *curr_cov;
    gsl_matrix *prev_cov;
    gsl_matrix *pxc_cov;
    gsl_matrix *L_cov;
    gsl_matrix *mean_correction;
} SFH_matrices;

static SFH_matrices sfh_mats;

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

void force_matrix_symmetric(gsl_matrix *mat) {
    // Make a matrix symmetric by copying lower triangle to upper triangle
    int nrows = mat->size1;
    int ncols = mat->size2;
    if (nrows != ncols) {
        LOG_ERROR("Matrix is not square, cannot symmetrise!");
        Throw(ValueError);
    }
    for (int i = 0; i < nrows; i++) {
        for (int j = i + 1; j < ncols; j++) {
            gsl_matrix_set(mat, i, j, gsl_matrix_get(mat, j, i));
        }
    }
}

void force_matrix_lotri(gsl_matrix *mat) {
    int nrows = mat->size1;
    int ncols = mat->size2;
    if (nrows != ncols) {
        LOG_ERROR("Matrix is not square, cannot make lower triangular!");
        Throw(ValueError);
    }
    for (int i = 0; i < nrows; i++) {
        for (int j = i + 1; j < ncols; j++) {
            gsl_matrix_set(mat, i, j, 0.0);
        }
    }
}

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
    if (s_max <= 0.0) {
        // entirely negative
        result = integral_expfunc_neg(s_min, s_max, t0, A, B);
    } else if (s_min >= 0.0) {
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

    // Region 1
    if (tau - t2 < first_boundary) {
        result += integral_expfunc_mod(tau - t2, first_boundary, t0, t2 - tau, 1.0);
    }

    // Region 2
    if (first_boundary < second_boundary) {
        result += integral_expfunc_mod(first_boundary, second_boundary, t0, fmin(t1, t2), 0.0);
    }

    // Region 3
    if (second_boundary < tau + t1) {
        result += integral_expfunc_mod(second_boundary, tau + t1, t0, t1 + tau, -1.0);
    }

    return sigma * sigma * result / (t1 * t2);
}

void fill_covar_analytic(double tau, double tau_prev, gsl_matrix *curr_cov, gsl_matrix *prev_cov,
                         gsl_matrix *cross_cov) {
    /* Fill the covariance matrices required for sampling the SFR correctly */
    /* A few notes on the covariance:
    Due to the asymmetry of the filter, and the different snapshot lengths,

    rho(tau, t1, t2) = rho(-tau, t2, t1) So...
    Cov(X_prev, Y_curr) == Cov(Y_curr, X_prev) != Cov(X_curr, Y_prev) == Cov(Y_prev, X_curr)
    Also note that positive tau means current snapshot is later than previous snapshot
    and negative means the current snapshot is earlier. These functions should produce the
    correct output for either case.

    The entire covariance matrix is symmetric and positive semi-definite,
    The auto-correlation sub-matrices (top-left/bottom-right) are both symmetric as well
    but the cross-covariance (top-right/bottom-left) is not.
    */

    // Previous Snapshot covariance matrix
    gsl_matrix_set(prev_cov, 0, 0, smoothed_correlation_func(0, 10., 10.));
    gsl_matrix_set(prev_cov, 1, 0, smoothed_correlation_func(0, 100., 10.));
    gsl_matrix_set(prev_cov, 1, 1, smoothed_correlation_func(0, 100., 100.));
    gsl_matrix_set(prev_cov, 2, 0, smoothed_correlation_func(0, tau_prev, 10.));
    gsl_matrix_set(prev_cov, 2, 1, smoothed_correlation_func(0, tau_prev, 100.));
    gsl_matrix_set(prev_cov, 2, 2, smoothed_correlation_func(0, tau_prev, tau_prev));
    force_matrix_symmetric(prev_cov);

    // FOR DEBUG: The forcing of symmetry should be equivalent to:
    //  gsl_matrix_set(prev_cov, 0, 1, smoothed_correlation_func(0,10.,100.));
    //  gsl_matrix_set(prev_cov, 0, 2, smoothed_correlation_func(0,10.,tau_prev));
    //  gsl_matrix_set(prev_cov, 1, 2, smoothed_correlation_func(0,100.,tau_prev));

    // Upper right corner Covariance is Cov(prev,curr) == Cov(curr,prev)^T
    gsl_matrix_set(cross_cov, 0, 0, smoothed_correlation_func(tau, 10., 10.));
    gsl_matrix_set(cross_cov, 0, 1, smoothed_correlation_func(tau, 10., 100.));
    gsl_matrix_set(cross_cov, 0, 2, smoothed_correlation_func(tau, 10., tau));
    gsl_matrix_set(cross_cov, 1, 0, smoothed_correlation_func(tau, 100., 10.));  // NB: != (0,1)
    gsl_matrix_set(cross_cov, 1, 1, smoothed_correlation_func(tau, 100., 100.));
    gsl_matrix_set(cross_cov, 1, 2, smoothed_correlation_func(tau, 100., tau));
    gsl_matrix_set(cross_cov, 2, 0, smoothed_correlation_func(tau, tau_prev, 10.));
    gsl_matrix_set(cross_cov, 2, 1, smoothed_correlation_func(tau, tau_prev, 100.));
    gsl_matrix_set(cross_cov, 2, 2, smoothed_correlation_func(tau, tau_prev, tau));

    // Current Snapshot covariance matrix
    // NOTE: Since the snapshot lengths are different, the snap variances are different
    gsl_matrix_memcpy(curr_cov, prev_cov);
    gsl_matrix_set(curr_cov, 2, 0, smoothed_correlation_func(0, tau, 10.));
    gsl_matrix_set(curr_cov, 2, 1, smoothed_correlation_func(0, tau, 100.));
    gsl_matrix_set(curr_cov, 2, 2, smoothed_correlation_func(0, tau, tau));
    force_matrix_symmetric(curr_cov);

    // FOR DEBUG: The forcing of symmetry should be equivalent to:
    //  gsl_matrix_set(prev_cov, 0, 2, smoothed_correlation_func(0,10.,tau));
    //  gsl_matrix_set(prev_cov, 1, 2, smoothed_correlation_func(0,100.,tau));
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

void initialise_psd_corrfunc_tables(double tau, double tau_prev) {
    // For easy initialisation, we set arrays of the tables and delays used
    const int N_FREQ_SFH = N_TAU_SFH / 2 + 1;  // number of frequency bins in SFH
    const double MAX_TAU = 2000;               // Myr

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
        table_ptrs[i]->x_min = -MAX_TAU;
        table_ptrs[i]->x_width = 2 * MAX_TAU / N_TAU_SFH;
    }

    double w_arr[N_FREQ_SFH];
    for (int i = 0; i < N_FREQ_SFH; i++) {
        w_arr[i] = M_PI * i / MAX_TAU;
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
            double shift_term = cexp(-I * w * MAX_TAU);  // shift to centre around zero lag
            in[j] = psd_sfh_powerlaw(w) * shifted_tophat_1d(w * t1) *
                    conj(shifted_tophat_1d(w * t2)) * shift_term;  // FFTW unnormalised convention
        }
        fftwf_execute(p);
        for (int j = 0; j < N_TAU_SFH; j++) {
            curr_table->y_arr[j] = out[j] / (2 * MAX_TAU);
        }
    }

    fftwf_destroy_plan(p);
    fftwf_free(in);
    free(out);
}

void fill_covar_from_tables(double tau, gsl_matrix *curr_cov, gsl_matrix *prev_cov,
                            gsl_matrix *cross_cov) {
    /* Fill the covariance matrices required for sampling the SFR correctly */
    // Previous Snapshot covariance matrix
    gsl_matrix_set(prev_cov, 0, 0, EvaluateRGTable1D(0, &sfh_corr.corr_10_10));
    gsl_matrix_set(prev_cov, 1, 0, EvaluateRGTable1D(0, &sfh_corr.corr_10_100));  //-0
    gsl_matrix_set(prev_cov, 1, 1, EvaluateRGTable1D(0, &sfh_corr.corr_100_100));
    gsl_matrix_set(prev_cov, 2, 0, EvaluateRGTable1D(0, &sfh_corr.corr_10_prev));   //-0
    gsl_matrix_set(prev_cov, 2, 1, EvaluateRGTable1D(0, &sfh_corr.corr_100_prev));  //-0
    gsl_matrix_set(prev_cov, 2, 2, EvaluateRGTable1D(0, &sfh_corr.corr_prev_prev));
    force_matrix_symmetric(prev_cov);

    // Upper right corner Covariance is Cov(prev,curr) == Cov(prev,curr)^T
    // The -tau is due to the asymmetry of the filter, rho(tau,t1,t2) == rho(-tau,t2,t1)
    // We use -tau instead of defining tables for both directions
    gsl_matrix_set(cross_cov, 0, 0, EvaluateRGTable1D(tau, &sfh_corr.corr_10_10));
    gsl_matrix_set(cross_cov, 0, 1, EvaluateRGTable1D(tau, &sfh_corr.corr_10_100));
    gsl_matrix_set(cross_cov, 0, 2, EvaluateRGTable1D(tau, &sfh_corr.corr_10_curr));
    gsl_matrix_set(cross_cov, 1, 0, EvaluateRGTable1D(-tau, &sfh_corr.corr_10_100));  // 100p x 10c
    gsl_matrix_set(cross_cov, 1, 1, EvaluateRGTable1D(tau, &sfh_corr.corr_100_100));
    gsl_matrix_set(cross_cov, 1, 2, EvaluateRGTable1D(tau, &sfh_corr.corr_100_curr));
    gsl_matrix_set(cross_cov, 2, 0, EvaluateRGTable1D(-tau, &sfh_corr.corr_10_prev));   // tp x 10c
    gsl_matrix_set(cross_cov, 2, 1, EvaluateRGTable1D(-tau, &sfh_corr.corr_100_prev));  // tp x 100c
    gsl_matrix_set(cross_cov, 2, 2, EvaluateRGTable1D(-tau, &sfh_corr.corr_curr_prev));  // tp x tc

    // Current Snapshot covariance matrix
    // NOTE: Since the snapshot lengths are different, the snap variances are different
    gsl_matrix_memcpy(curr_cov, prev_cov);
    gsl_matrix_set(curr_cov, 2, 0, EvaluateRGTable1D(0, &sfh_corr.corr_10_curr));
    gsl_matrix_set(curr_cov, 2, 1, EvaluateRGTable1D(0, &sfh_corr.corr_100_curr));
    gsl_matrix_set(curr_cov, 2, 2, EvaluateRGTable1D(0, &sfh_corr.corr_curr_curr));
    force_matrix_symmetric(curr_cov);
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
    gsl_matrix *partial_buf = gsl_matrix_alloc(3, 3);
    gsl_matrix *covar_buf = gsl_matrix_alloc(3, 3);
    gsl_matrix *L_buf = gsl_matrix_alloc(3, 3);

    gsl_matrix_memcpy(L_buf, prev_cov);         // preserve for Cholesky
    gsl_matrix_memcpy(partial_buf, cross_cov);  // will hold L^-1 Cov(prev,curr)
    gsl_matrix_memcpy(covar_buf, curr_cov);     // will hold the full conditional covariance

    // Cholesky factorization of Cov(prev) = L L^T to do implicit inversion
    gsl_linalg_cholesky_decomp1(L_buf);  // holds L

    // Compute the conditional covariance matrix Cov(curr|prev) = Cov(curr) - Cov(curr,prev)
    // Cov(prev)^-1 Cov(prev,curr)
    // First get L^-1  Cov(prev,curr)
    gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1.0, L_buf, partial_buf);

    // compute Cov(curr) - BUF^T*BUF == Cov(curr) - Cov(curr|prev) L^T^-1 L^-1 Cov(prev,curr)
    // == Cov(curr) - Cov(curr,prev) Cov(prev)^-1 Cov(prev,curr)
    gsl_blas_dsyrk(CblasLower, CblasTrans, -1.0, partial_buf, 1.0, covar_buf);
    // The lower triangle of curr_cov now holds Cov(curr|prev)

    // Perform Cholesky decomposition (only uses lower triangle)
    gsl_linalg_cholesky_decomp1(covar_buf);

    // Not necessary, but it makes it clearer that only the lower triangle is valid
    force_matrix_lotri(covar_buf);  // zero upper triangle

    gsl_matrix_memcpy(out_chol_cov, covar_buf);

    // Now Compute the mean correction term, since BUF was preserved from the rank-k above
    //  Compute Cov(prev,curr) L^-T^-1 L^-1 = Cov(curr,prev) Cov(prev)^-1
    gsl_matrix_transpose(partial_buf);
    gsl_blas_dtrsm(CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, 1.0, L_buf, partial_buf);

    // Since, for zero mean, E[X|Y] = Cov(X,Y) Cov(Y)^-1 Y
    gsl_matrix_memcpy(out_mean_correction, partial_buf);  // store for output

    gsl_matrix_free(covar_buf);
    gsl_matrix_free(L_buf);
    gsl_matrix_free(partial_buf);
}

void eval_sfh_unconditioned(gsl_matrix *curr_cov, gsl_matrix *out_chol_cov,
                            gsl_matrix *out_mean_correction) {
    /* Evaluate the SFH covariance matrix for unconditioned sampling

    Outputs are the Cholesky factor of the current covariance matrix
    which multiplies standard normal vector to get correlated SFRs.

    out_chol_cov = Cholesky factor of Cov(curr), NOTE: The upper triangle is garbage
    */
    gsl_matrix *covar_buf = gsl_matrix_alloc(3, 3);
    gsl_matrix_memcpy(covar_buf, curr_cov);

    // Perform Cholesky decomposition of Cov(curr) = L L^T to do implicit inversion
    gsl_linalg_cholesky_decomp1(covar_buf);

    // Not necessary, but it makes it clearer that only the lower triangle is valid
    force_matrix_lotri(covar_buf);  // zero upper triangle

    gsl_matrix_memcpy(out_chol_cov, covar_buf);

    // set mean correction to zero matrix
    gsl_matrix_set_zero(out_mean_correction);

    gsl_matrix_free(covar_buf);
}

void initialise_sfh_structs(double z0, double z1, double z2, bool conditioned) {
    sfh_mats.curr_cov = gsl_matrix_alloc(3, 3);
    sfh_mats.prev_cov = gsl_matrix_alloc(3, 3);
    sfh_mats.pxc_cov = gsl_matrix_alloc(3, 3);
    sfh_mats.L_cov = gsl_matrix_alloc(3, 3);
    sfh_mats.mean_correction = gsl_matrix_alloc(3, 3);

    if (z0 < 0. || conditioned && !(z2 <= z1 <= z0)) {
        LOG_ERROR("You provided invalid redshifts for SFH initialisation!");
        LOG_ERROR("Provided redshifts: z0 = %f, z1 = %f, z2 = %f", z0, z1, z2);
        LOG_ERROR("All redshifts must satisfy z2 < z1 < z0 if contitioned");
        Throw(ValueError);
    }

    double tau = astro_params_global->SFH_TAU * 100;  // long timescale for uncorrelated sampling
    double tau_prev = astro_params_global->SFH_TAU * 100;
    if (conditioned) {
        tau = time_between_z(z0, z1) / (physconst.s_per_yr * 1e6);  // Myr
        if (z2 >= 0.) {
            tau_prev = time_between_z(z1, z2) / (physconst.s_per_yr * 1e6);  // Myr
        }
    }
    LOG_DEBUG("Initialising SFH structs with tau = %f Myr and tau_prev = %f Myr", tau, tau_prev);
    LOG_DEBUG(" from redshifts z0 = %f, z1 = %f, z2 = %f", z0, z1, z2);

    // initialise_psd_corrfunc_tables(tau, tau_prev);
    // fill_covar_from_tables(tau, sfh_mats.curr_cov, sfh_mats.prev_cov, sfh_mats.pxc_cov);
    fill_covar_analytic(tau, tau_prev, sfh_mats.curr_cov, sfh_mats.prev_cov, sfh_mats.pxc_cov);
    if (conditioned) {
        eval_sfh_moments(sfh_mats.prev_cov, sfh_mats.curr_cov, sfh_mats.pxc_cov, sfh_mats.L_cov,
                         sfh_mats.mean_correction);
    } else {
        eval_sfh_unconditioned(sfh_mats.curr_cov, sfh_mats.L_cov, sfh_mats.mean_correction);
    }
}

void cleanup_sfh_structs() {
    gsl_matrix_free(sfh_mats.curr_cov);
    gsl_matrix_free(sfh_mats.prev_cov);
    gsl_matrix_free(sfh_mats.pxc_cov);
    gsl_matrix_free(sfh_mats.L_cov);
    gsl_matrix_free(sfh_mats.mean_correction);
    // free_sfh_correlation();
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

void sample_correlated_sfh(gsl_rng *rng, double prev_values[3], double out_values[3]) {
    /* Sample correlated SFH values given previous values

    Inputs:
    prev_values: array of previous SFR values [SFR_10Myr, SFR_100Myr, SFR_snapshot_prev]
    L_cov: Cholesky factor of Cov(curr|prev), from eval_sfh_moments, L L^T = Cov(curr|prev)
    mean_corr: Mean correction matrix from eval_sfh_moments

    Method:
    Calculates the produce out = L_cov * N(0,1) + mean_corr * prev_values
    Assumes zero mean of all variables.

    Outputs:
    out_values: array of sampled current SFR values [SFR_10Myr, SFR_100Myr, SFR_snapshot_curr]
    */

    // Generate standard normal random variables
    gsl_vector *cov_term = gsl_vector_alloc(3);
    for (int i = 0; i < 3; i++) {
        gsl_vector_set(cov_term, i, gsl_ran_ugaussian(rng));
    }

    // Multiply by Cholesky factor to get correlated samples
    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, sfh_mats.L_cov, cov_term);

    // Create a vector for the conditioned samples
    gsl_vector *cond_term = gsl_vector_alloc(3);
    for (int i = 0; i < 3; i++) {
        gsl_vector_set(cond_term, i, prev_values[i]);
    }
    // Add the mean correction term
    gsl_blas_dgemv(CblasNoTrans, 1.0, sfh_mats.mean_correction, cond_term, 1.0, cov_term);

    // Copy to output
    for (int i = 0; i < 3; i++) {
        out_values[i] = gsl_vector_get(cov_term, i);
    }
    gsl_vector_free(cov_term);
    gsl_vector_free(cond_term);
}

void get_current_vars(double out[3]) {
    /* Get the variances of the current SFR variables */
    for (int i = 0; i < 3; i++) {
        out[i] = gsl_matrix_get(sfh_mats.curr_cov, i, i);
    }
}

/* TESTING FUNCTIONS */

void print_corrfunc(RGTable1D *ptr, const char *name, int skip_lines) {
    fprintf(stdout, "Correlation Function Table: %s", name);
    for (int i = 0; i < N_TAU_SFH / skip_lines; i += skip_lines) {
        double lag = ptr->x_min + i * ptr->x_width;
        double val = ptr->y_arr[i];
        fprintf(stdout, "Lag: %.2f  Value: %.6f \n", lag, val);
    }
}

int test_sfh_corr(double z0, double z1, double z2) {
    double rel_tol = 1e-2;
    double tau = time_between_z(z0, z1) / (physconst.s_per_yr * 1e6);       // Myr
    double tau_prev = time_between_z(z1, z2) / (physconst.s_per_yr * 1e6);  // Myr
    fprintf(stdout, "Testing SFH Correlation Functions at tau = %f Myr, tau_prev = %f Myr\n", tau,
            tau_prev);
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

    for (int i = 0; i < 10; i++) {
        for (double lag = 0.0; lag <= 500.0; lag += 50.0) {
            double val_tables = EvaluateRGTable1D(lag, table_ptrs[i]);
            double val_analytic = smoothed_correlation_func(lag, delays[i][0], delays[i][1]);
            if (fabs(val_tables / val_analytic - 1.0) > rel_tol) {
                LOG_ERROR("Test Failed for %s at lag %f: Tables %f Analytic %f Ratio %f", names[i],
                          lag, val_tables, val_analytic, val_tables / val_analytic);
                print_corrfunc(table_ptrs[i], names[i], 10);
                Throw(TableGenerationError);
            }
        }
    }

    for (double lag = 0.0; lag <= 500.0; lag += 50.0) {
        double val_tables = EvaluateRGTable1D(lag, &sfh_corr.corr_zero);
        double val_analytic = exp(-fabs(lag) / astro_params_global->SFH_TAU) *
                              astro_params_global->SIGMA_STAR * astro_params_global->SIGMA_STAR;
        if (fabs(val_tables / val_analytic - 1.0) > rel_tol) {
            LOG_ERROR("Test Failed for %s at lag %f: Tables %f Analytic %f Ratio %f", names[10],
                      lag, val_tables, val_analytic, val_tables / val_analytic);
            print_corrfunc(table_ptrs[10], names[10], 50);
            Throw(TableGenerationError);
        }
    }

    fprintf(stdout, "sigma = %f, sq = %f\n", astro_params_global->SIGMA_STAR,
            astro_params_global->SIGMA_STAR * astro_params_global->SIGMA_STAR);

    gsl_matrix *curr_cov = gsl_matrix_alloc(3, 3);
    gsl_matrix *prev_cov = gsl_matrix_alloc(3, 3);
    gsl_matrix *cross_cov = gsl_matrix_alloc(3, 3);
    fill_covar_from_tables(tau, curr_cov, prev_cov, cross_cov);
    gsl_matrix *curr_cov_2 = gsl_matrix_alloc(3, 3);
    gsl_matrix *prev_cov_2 = gsl_matrix_alloc(3, 3);
    gsl_matrix *cross_cov_2 = gsl_matrix_alloc(3, 3);
    fill_covar_analytic(tau, tau_prev, curr_cov_2, prev_cov_2, cross_cov_2);

    print_gsl_matrix(prev_cov, "Previous Covariance Matrix:");
    print_gsl_matrix(curr_cov, "Current Covariance Matrix:");
    print_gsl_matrix(cross_cov, "Cross Covariance Matrix:");

    gsl_matrix *corrcoev = gsl_matrix_alloc(6, 6);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            gsl_matrix_set(corrcoev, i, j, gsl_matrix_get(prev_cov, i, j));
            gsl_matrix_set(corrcoev, i, j + 3, gsl_matrix_get(cross_cov, i, j));
            gsl_matrix_set(corrcoev, i + 3, j, gsl_matrix_get(cross_cov, j, i));
            gsl_matrix_set(corrcoev, i + 3, j + 3, gsl_matrix_get(curr_cov, i, j));
        }
    }
    print_gsl_matrix(corrcoev, "Covariance Matrix:");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            gsl_matrix_set(
                corrcoev, i, j,
                sqrt(gsl_matrix_get(corrcoev, i, j) /
                     sqrt(gsl_matrix_get(corrcoev, i, i) * gsl_matrix_get(corrcoev, j, j))));
        }
    }
    print_gsl_matrix(corrcoev, "Correlation Coefficient Matrix:");

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double val_1 = gsl_matrix_get(prev_cov, i, j);
            double val_2 = gsl_matrix_get(prev_cov_2, i, j);
            if (fabs(val_1 / val_2 - 1.0) > rel_tol) {
                LOG_ERROR(
                    "Test Failed for Previous Covariance Matrix at (%d,%d): Tables %f Analytic %f "
                    "Ratio %f",
                    i, j, val_1, val_2, val_1 / val_2);
                print_gsl_matrix(prev_cov, "Previous Covariance Matrix from Tables:");
                print_gsl_matrix(prev_cov_2, "Previous Covariance Matrix from Analytic:");
                Throw(TableGenerationError);
            }
            val_1 = gsl_matrix_get(cross_cov, i, j);
            val_2 = gsl_matrix_get(cross_cov_2, i, j);
            if (fabs(val_1 / val_2 - 1.0) > rel_tol) {
                LOG_ERROR(
                    "Test Failed for Cross Covariance Matrix at (%d,%d): Tables %f Analytic %f "
                    "Ratio %f",
                    i, j, val_1, val_2, val_1 / val_2);
                print_gsl_matrix(cross_cov, "Cross Covariance Matrix from Tables:");
                print_gsl_matrix(cross_cov_2, "Cross Covariance Matrix from Analytic:");
                Throw(TableGenerationError);
            }
            val_1 = gsl_matrix_get(curr_cov, i, j);
            val_2 = gsl_matrix_get(curr_cov_2, i, j);
            if (fabs(val_1 / val_2 - 1.0) > rel_tol) {
                LOG_ERROR(
                    "Test Failed for Current Covariance Matrix at (%d,%d): Tables %f Analytic %f "
                    "Ratio %f",
                    i, j, val_1, val_2, val_1 / val_2);
                print_gsl_matrix(curr_cov, "Current Covariance Matrix from Tables:");
                print_gsl_matrix(curr_cov_2, "Current Covariance Matrix from Analytic:");
                Throw(TableGenerationError);
            }
        }
    }

    gsl_matrix *L_cov = gsl_matrix_alloc(3, 3);
    gsl_matrix *mean_corr = gsl_matrix_alloc(3, 3);
    eval_sfh_moments(prev_cov_2, curr_cov_2, cross_cov_2, L_cov, mean_corr);
    print_gsl_matrix(mean_corr, "Mean Correction Matrix:");
    print_gsl_matrix(L_cov, "Cholesky Factor of Conditioned Covariance Matrix:");

    // get the conditioned covariance matrix
    gsl_matrix_set(L_cov, 0, 1, 0.0);
    gsl_matrix_set(L_cov, 0, 2, 0.0);
    gsl_matrix_set(L_cov, 1, 2, 0.0);
    gsl_matrix_memcpy(curr_cov, L_cov);  // using curr_cov as buffer
    gsl_matrix_set(curr_cov, 0, 1, 0.0);
    gsl_matrix_set(curr_cov, 0, 2, 0.0);
    gsl_matrix_set(curr_cov, 1, 2, 0.0);
    gsl_blas_dtrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, 1.0, L_cov, curr_cov);
    print_gsl_matrix(curr_cov, "Conditioned Covariance Matrix:");

    free_sfh_correlation();
    return 0;
}
