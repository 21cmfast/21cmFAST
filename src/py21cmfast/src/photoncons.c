// The volume filling factor at a given redshift, Q(z), or find redshift at a given Q, z(Q).
//
// The evolution of Q can be written as
// dQ/dt = n_{ion}/dt - Q/t_{rec},
// where n_{ion} is the number of ionizing photons per baryon. The averaged recombination time is given by
// t_{rec} ~ 0.93 Gyr * (C_{HII}/3)^-1 * (T_0/2e4 K)^0.7 * ((1+z)/7)^-3.
// We assume the clumping factor of C_{HII}=3 and the IGM temperature of T_0 = 2e4 K, following
// Section 2.1 of Kuhlen & Faucher-Gigue`re (2012) MNRAS, 423, 862 and references therein.
// 1) initialise interpolation table
// -> initialise_Q_value_spline(NoRec, M_TURN, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10)
// NoRec = 0: Compute dQ/dt with the recombination time.
// NoRec = 1: Ignore recombination.
// 2) find Q value at a given z -> Q_at_z(z, &(Q))
// or find z at a given Q -> z_at_Q(Q, &(z)).
// 3) free memory allocation -> free_Q_value()

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"
#include "Constants.h"
#include "InputParameters.h"
#include "cosmology.h"
#include "hmf.h"
#include "interp_tables.h"
#include "debugging.h"

#include "photoncons.h"

bool photon_cons_allocated = false;
//These globals hold values relevant for the photon conservation (z-shift) model
static float calibrated_NF_min;
static double *deltaz, *deltaz_smoothed, *NeutralFractions, *z_Q, *Q_value, *nf_vals, *z_vals;
static int N_NFsamples,N_extrapolated, N_analytic, N_calibrated, N_deltaz;
static double FinalNF_Estimate, FirstNF_Estimate;

static gsl_interp_accel *Q_at_z_spline_acc;
static gsl_spline *Q_at_z_spline;
static gsl_interp_accel *z_at_Q_spline_acc;
static gsl_spline *z_at_Q_spline;
static double Zmin, Zmax, Qmin, Qmax;
void Q_at_z(double z, double *splined_value);
void z_at_Q(double Q, double *splined_value);

static gsl_interp_accel *deltaz_spline_for_photoncons_acc;
static gsl_spline *deltaz_spline_for_photoncons;

static gsl_interp_accel *NFHistory_spline_acc;
static gsl_spline *NFHistory_spline;
static gsl_interp_accel *z_NFHistory_spline_acc;
static gsl_spline *z_NFHistory_spline;
void initialise_NFHistory_spline(double *redshifts, double *NF_estimate, int NSpline);
void z_at_NFHist(double xHI_Hist, double *splined_value);
void NFHist_at_z(double z, double *splined_value);

//   Set up interpolation table for the volume filling factor, Q, at a given redshift z and redshift at a given Q.
int InitialisePhotonCons(UserParams *user_params, CosmoParams *cosmo_params,
                         AstroParams *astro_params, FlagOptions *flag_options)
{

    /*
        This is an API-level function for initialising the photon conservation.
    */

    int status;
    Try{  // this try wraps the whole function.
    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    init_ps();
    //     To solve differentail equation, uses Euler's method.
    //     NOTE:
    //     (1) With the fiducial parameter set,
    //	    when the Q value is < 0.9, the difference is less than 5% compared with accurate calculation.
    //	    When Q ~ 0.98, the difference is ~25%. To increase accuracy one can reduce the step size 'da', but it will increase computing time.
    //     (2) With the fiducial parameter set,
    //     the difference for the redshift where the reionization end (Q = 1) is ~0.2 % compared with accurate calculation.
    float ION_EFF_FACTOR,M_MIN,M_MIN_z0,M_MIN_z1,Mlim_Fstar, Mlim_Fesc;
    double lnMmin, lnMmax;
    double a_start = 0.03, a_end = 1./(1. + global_params.PhotonConsEndCalibz); // Scale factors of 0.03 and 0.17 correspond to redshifts of ~32 and ~5.0, respectively.
    double C_HII = 3., T_0 = 2e4;
    double reduce_ratio = 1.003;
    double Q0,Q1,Nion0,Nion1,Trec,da,a,z0,z1,zi,dadt,ans,delta_a,zi_prev,Q1_prev;
    double *z_arr,*Q_arr;
    int Nmax = 2000; // This is the number of step, enough with 'da = 2e-3'. If 'da' is reduced, this number should be checked.
    int cnt, nbin, i, istart;
    int fail_condition, not_mono_increasing, num_fails;
    int gsl_status;

    z_arr = calloc(Nmax,sizeof(double));
    Q_arr = calloc(Nmax,sizeof(double));

    //set the minimum source mass
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;

        M_MIN = astro_params->M_TURN/50.;
        Mlim_Fstar = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR, astro_params->F_STAR10);
        Mlim_Fesc = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC, astro_params->F_ESC10);
        if(user_params->INTEGRATION_METHOD_ATOMIC == 2 || user_params->INTEGRATION_METHOD_MINI == 2){
          initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
        }
        else{
          initialiseSigmaMInterpTable(M_MIN,1e20);
        }
        lnMmin = log(M_MIN);
        lnMmax = log(global_params.M_MAX_INTEGRAL);
    }
    else {
        ION_EFF_FACTOR = astro_params->HII_EFF_FACTOR;
    }

    fail_condition = 1;
    num_fails = 0;

    // We are going to come up with the analytic curve for the photon non conservation correction
    // This can be somewhat numerically unstable and as such we increase the sampling until it works
    // If it fails to produce a monotonically increasing curve (for Q as a function of z) after 10 attempts we crash out
    while(fail_condition!=0) {

        a = a_start;
        if(num_fails < 3) {
            da = 3e-3 - ((double)num_fails)*(1e-3);
       	}
        else {
            da = 1e-3 - ((double)num_fails - 2.)*(1e-4);
       	}
        delta_a = 1e-7;

        zi_prev = Q1_prev = 0.;
        not_mono_increasing = 0;

        if(num_fails>0) {
            for(i=0;i<Nmax;i++) {
                z_arr[i] = 0.;
                Q_arr[i] = 0.;
            }
        }

        cnt = 0;
        Q0 = 0.;

        while (a < a_end) {

            zi = 1./a - 1.;
            z0 = 1./(a+delta_a) - 1.;
            z1 = 1./(a-delta_a) - 1.;

            // Ionizing emissivity (num of photons per baryon)
            //We Force QAG due to the changing limits and messy implementation which I will fix later (hopefully move the whole thing to python)
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                Nion0 = ION_EFF_FACTOR*Nion_General(z0, lnMmin, lnMmax, astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc);
                Nion1 = ION_EFF_FACTOR*Nion_General(z1, lnMmin, lnMmax, astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc);
            }
            else {
                //set the minimum source mass
                if (astro_params->ION_Tvir_MIN < 9.99999e3) { // neutral IGM
                    M_MIN_z0 = (float)TtoM(z0, astro_params->ION_Tvir_MIN, 1.22);
                    M_MIN_z1 = (float)TtoM(z1, astro_params->ION_Tvir_MIN, 1.22);
                }
                else { // ionized IGM
                    M_MIN_z0 = (float)TtoM(z0, astro_params->ION_Tvir_MIN, 0.6);
                    M_MIN_z1 = (float)TtoM(z1, astro_params->ION_Tvir_MIN, 0.6);
                }

                if(M_MIN_z0 < M_MIN_z1) {
                  if(user_params->INTEGRATION_METHOD_ATOMIC == 2){
                    initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN_z0),1e20);
                  }
                  else{
                    initialiseSigmaMInterpTable(M_MIN_z0,1e20);
                  }
                }
                else {
                  if(user_params->INTEGRATION_METHOD_ATOMIC == 2){
                    initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN_z1),1e20);
                  }
                  else{
                    initialiseSigmaMInterpTable(M_MIN_z1,1e20);
                  }
                }

                Nion0 = ION_EFF_FACTOR*Fcoll_General(z0,log(M_MIN_z0),lnMmax);
                Nion1 = ION_EFF_FACTOR*Fcoll_General(z1,log(M_MIN_z1),lnMmax);
                freeSigmaMInterpTable();
            }

            // With scale factor a, the above equation is written as dQ/da = n_{ion}/da - Q/t_{rec}*(dt/da)
            if (!global_params.RecombPhotonCons) {
                Q1 = Q0 + ((Nion0-Nion1)/2/delta_a)*da; // No Recombination
            }
            else {
                dadt = Ho*sqrt(cosmo_params->OMm/a + global_params.OMr/a/a + cosmo_params->OMl*a*a); // da/dt = Ho*a*sqrt(OMm/a^3 + OMr/a^4 + OMl)
                Trec = 0.93 * 1e9 * SperYR * pow(C_HII/3.,-1) * pow(T_0/2e4,0.7) * pow((1.+zi)/7.,-3);
                Q1 = Q0 + ((Nion0-Nion1)/2./delta_a - Q0/Trec/dadt)*da;
            }

            // Curve is no longer monotonically increasing, we are going to have to exit and start again
            if(Q1 < Q1_prev) {
                not_mono_increasing = 1;
                break;
            }

            zi_prev = zi;
            Q1_prev = Q1;

            z_arr[cnt] = zi;
            Q_arr[cnt] = Q1;

            cnt = cnt + 1;
            //if (Q1 >= 1.0) break; // if fully ionized, stop here. NOTE(jdavies) I turned this off to find photon ratios, it shouldn't affect much
            // As the Q value increases, the bin size decreases gradually because more accurate calculation is required.
            if (da < 7e-5) da = 7e-5; // set minimum bin size.
            else da = pow(da,reduce_ratio);
            Q0 = Q1;
            a = a + da;
        }

        // A check to see if we ended up with a monotonically increasing function
        if(not_mono_increasing==0) {
            fail_condition = 0;
        }
        else {
            num_fails += 1;
            if(num_fails>10) {
                LOG_ERROR("Failed too many times.");
//                Throw ParameterError;
                Throw(PhotonConsError);
            }
        }

    }
    cnt = cnt - 1;
    istart = 0;
    for (i=1;i<cnt;i++){
        if (Q_arr[i-1] == 0. && Q_arr[i] != 0.) istart = i-1;
    }
    nbin = cnt - istart;

    N_analytic = nbin;

    // initialise interploation Q as a function of z
    z_Q = calloc(nbin,sizeof(double));
    Q_value = calloc(nbin,sizeof(double));

    Q_at_z_spline_acc = gsl_interp_accel_alloc ();
    Q_at_z_spline = gsl_spline_alloc (gsl_interp_cspline, nbin);

    for (i=0; i<nbin; i++){
        z_Q[i] = z_arr[cnt-i];
        Q_value[i] = Q_arr[cnt-i];
    }

    gsl_set_error_handler_off();
    gsl_status = gsl_spline_init(Q_at_z_spline, z_Q, Q_value, nbin);
    CATCH_GSL_ERROR(gsl_status);

    Zmin = z_Q[0];
    Zmax = z_Q[nbin-1];
    Qmin = Q_value[nbin-1];
    Qmax = Q_value[0];

    // initialise interpolation z as a function of Q
    double *Q_z = calloc(nbin,sizeof(double));
    double *z_value = calloc(nbin,sizeof(double));

    z_at_Q_spline_acc = gsl_interp_accel_alloc ();
    z_at_Q_spline = gsl_spline_alloc (gsl_interp_linear, nbin);
    for (i=0; i<nbin; i++){
        Q_z[i] = Q_value[nbin-1-i];
        z_value[i] = z_Q[nbin-1-i];
    }

    gsl_status = gsl_spline_init(z_at_Q_spline, Q_z, z_value, nbin);
    CATCH_GSL_ERROR(gsl_status);

    free(z_arr);
    free(Q_arr);

    free(Q_z);
    free(z_value);

    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
      freeSigmaMInterpTable();
    }

    LOG_DEBUG("Initialised PhotonCons.");
    } // End of try
    Catch(status){
        return status;
    }

    return(0);
}

// Function to construct the spline for the calibration curve of the photon non-conservation
int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline){
    int status;
    Try{
        if(xH_estimate[NSpline-1] > 0.0 && xH_estimate[NSpline-2] > 0.0 && xH_estimate[NSpline-3] > 0.0 && xH_estimate[0] <= global_params.PhotonConsStart) {
            initialise_NFHistory_spline(z_estimate,xH_estimate,NSpline);
        }
    }
    Catch(status){
        return status;
    }
    return(0);
}

// Function callable from Python to know at which redshift to start sampling the calibration curve (to minimise function calls)
int ComputeZstart_PhotonCons(double *zstart) {
    int status;
    double temp;

    Try{
        if((1.-global_params.PhotonConsStart) > Qmax) {
            // It is possible that reionisation never even starts
            // Just need to arbitrarily set a high redshift to perform the algorithm
            temp = 20.;
        }
        else {
            z_at_Q(1. - global_params.PhotonConsStart,&(temp));
        // Multiply the result by 10 per-cent to fix instances when this isn't high enough
            temp *= 1.1;
        }
    }
    Catch(status){
        return(status); // Use the status to determine if something went wrong.
    }

    *zstart = temp;
    return(0);
}


void determine_deltaz_for_photoncons() {

    int i, j, increasing_val, counter, smoothing_int;
    double temp;
    float z_cal, z_analytic, NF_sample, returned_value, NF_sample_min, gradient_analytic, z_analytic_at_endpoint, const_offset, z_analytic_2, smoothing_width;
    float bin_width, delta_NF, val1, val2, extrapolated_value;

    LOG_DEBUG("Determining deltaz for photon cons.");

    // Number of points for determine the delta z correction of the photon non-conservation
    N_NFsamples = 100;
    // Determine the change in neutral fraction to calculate the gradient for the linear extrapolation of the photon non-conservation correction
    delta_NF = 0.025;
    // A width (in neutral fraction data points) in which point we average over to try and avoid sharp features in the correction (removes some kinks)
    // Effectively acts as filtering step
    smoothing_width = 35.;


    // The photon non-conservation correction has a threshold (in terms of neutral fraction; global_params.PhotonConsEnd) for which we switch
    // from using the exact correction between the calibrated (21cmFAST all flag options off) to analytic expression to some extrapolation.
    // This threshold is required due to the behaviour of 21cmFAST at very low neutral fractions, which cause extreme behaviour with recombinations on

    // A lot of the steps and choices are not completely rubust, just chosed to smooth/average the data to have smoother resultant reionisation histories

    // Determine the number of extrapolated points required, if required at all.
    if(!global_params.PhotonConsSmoothing){
        NF_sample_min = global_params.PhotonConsAsymptoteTo;
        N_extrapolated = 0;
    }
    else if(calibrated_NF_min < global_params.PhotonConsEnd) {
        // We require extrapolation, set minimum point to the threshold, and extrapolate beyond.
        NF_sample_min = global_params.PhotonConsEnd;

        // Determine the number of extrapolation points (to better smooth the correction) between the threshod (global_params.PhotonConsEnd) and a
        // point close to zero neutral fraction (set by global_params.PhotonConsAsymptoteTo)
        // Choice is to get the delta neutral fraction between extrapolated points to be similar to the cadence in the exact correction
        if(calibrated_NF_min > global_params.PhotonConsAsymptoteTo) {
            N_extrapolated = ((float)N_NFsamples - 1.)*(NF_sample_min - calibrated_NF_min)/( global_params.PhotonConsStart - NF_sample_min );
        }
        else {
            N_extrapolated = ((float)N_NFsamples - 1.)*(NF_sample_min - global_params.PhotonConsAsymptoteTo)/( global_params.PhotonConsStart - NF_sample_min );
        }
        N_extrapolated = (int)floor( N_extrapolated ) - 1; // Minus one as the zero point is added below
    }
    else {
        // No extrapolation required, neutral fraction never reaches zero
        NF_sample_min = calibrated_NF_min;

        N_extrapolated = 0;
    }

    // Determine the bin width for the sampling of the neutral fraction for the correction
    bin_width = ( global_params.PhotonConsStart - NF_sample_min )/((float)N_NFsamples - 1.);

    // allocate memory for arrays required to determine the photon non-conservation correction
    deltaz = calloc(N_NFsamples + N_extrapolated + 1,sizeof(double));
    deltaz_smoothed = calloc(N_NFsamples + N_extrapolated + 1,sizeof(double));
    NeutralFractions = calloc(N_NFsamples + N_extrapolated + 1,sizeof(double));

    // Go through and fill the data points (neutral fraction and corresponding delta z between the calibrated and analytic curves).
    for(i=0;i<N_NFsamples;i++) {

        NF_sample = NF_sample_min + bin_width*(float)i;

        // Determine redshift given a neutral fraction for the calibration curve
        z_at_NFHist(NF_sample,&(temp));

        z_cal = temp;

        // Determine redshift given a neutral fraction for the analytic curve
        z_at_Q(1. - NF_sample,&(temp));

        z_analytic = temp;

        deltaz[i+1+N_extrapolated] = fabs( z_cal - z_analytic );
        NeutralFractions[i+1+N_extrapolated] = NF_sample;
    }

    // Determining the end-point (lowest neutral fraction) for the photon non-conservation correction
    if(!global_params.PhotonConsSmoothing){
        NeutralFractions[0] = 0.999*NF_sample_min;
        if(deltaz[1] < deltaz[2])
            deltaz[0] = 0.999*deltaz[1];
        else
            deltaz[0] = 1.001*deltaz[1];

        N_deltaz = N_NFsamples + N_extrapolated + 1;

        // Now, we can construct the spline of the photon non-conservation correction (delta z as a function of neutral fraction)
        deltaz_spline_for_photoncons_acc = gsl_interp_accel_alloc ();
        deltaz_spline_for_photoncons = gsl_spline_alloc (gsl_interp_linear, N_NFsamples + N_extrapolated + 1);

        gsl_set_error_handler_off();
        int gsl_status;
        gsl_status = gsl_spline_init(deltaz_spline_for_photoncons, NeutralFractions, deltaz, N_NFsamples + N_extrapolated + 1);
        if(gsl_status){
            for(i=0;i<N_NFsamples+1;i++){
                LOG_ERROR("NF %.3f dz %.3f",NeutralFractions[i],deltaz[i]);
            }
        }
        CATCH_GSL_ERROR(gsl_status);
        return;
    }

    //SMOOTHING STUFF HERE
    if(calibrated_NF_min >= global_params.PhotonConsEnd) {

        increasing_val = 0;
        counter = 0;

        // Check if all the values of delta z are increasing
        for(i=0;i<(N_NFsamples-1);i++) {
            if(deltaz[i+1+N_extrapolated] >= deltaz[i+N_extrapolated]) {
                counter += 1;
            }
        }
        // If all the values of delta z are increasing, then some of the smoothing of the correction done below cannot be performed
        if(counter==(N_NFsamples-1)) {
            increasing_val = 1;
        }

        // Since we never have reionisation, need to set an appropriate end-point for the correction
        // Take some fraction of the previous point to determine the end-point
        NeutralFractions[0] = 0.999*NF_sample_min;
        if(increasing_val) {
            // Values of delta z are always increasing with decreasing neutral fraction thus make the last point slightly larger
            //NOTE (jdavies): since Neutralfractions[] is increasing, isnt this backwards???
            deltaz[0] = 1.001*deltaz[1];
        }
        else {
            // Values of delta z are always decreasing with decreasing neutral fraction thus make the last point slightly smaller
            deltaz[0] = 0.999*deltaz[1];
        }
    }
    else {

        // Ok, we are going to be extrapolating the photon non-conservation (delta z) beyond the threshold
        // Construct a linear curve for the analytic function to extrapolate to the new endpoint
        // The choice for doing so is to ensure the corrected reionisation history is mostly smooth, and doesn't
        // artificially result in kinks due to switching between how the delta z should be calculated

        z_at_Q(1. - (NeutralFractions[1+N_extrapolated] + delta_NF),&(temp));
        z_analytic = temp;

        z_at_Q(1. - NeutralFractions[1+N_extrapolated],&(temp));
        z_analytic_2 = temp;

        // determine the linear curve
        // Multiplitcation by 1.1 is arbitrary but effectively smooths out most kinks observed in the resultant corrected reionisation histories
        gradient_analytic = 1.1*( delta_NF )/( z_analytic - z_analytic_2 );
        const_offset = ( NeutralFractions[1+N_extrapolated] + delta_NF ) - gradient_analytic * z_analytic;

        // determine the extrapolation end point
        if(calibrated_NF_min > global_params.PhotonConsAsymptoteTo) {
            extrapolated_value = calibrated_NF_min;
        }
        else {
            extrapolated_value = global_params.PhotonConsAsymptoteTo;
        }

        // calculate the delta z for the extrapolated end point
        z_at_NFHist(extrapolated_value,&(temp));
        z_cal = temp;

        z_analytic_at_endpoint = ( extrapolated_value - const_offset )/gradient_analytic ;

        deltaz[0] = fabs( z_cal - z_analytic_at_endpoint );
        NeutralFractions[0] = extrapolated_value;

        // If performing extrapolation, add in all the extrapolated points between the end-point and the threshold to end the correction (global_params.PhotonConsEnd)
        for(i=0;i<N_extrapolated;i++) {
            if(calibrated_NF_min > global_params.PhotonConsAsymptoteTo) {
                NeutralFractions[i+1] = calibrated_NF_min + (NF_sample_min - calibrated_NF_min)*(float)(i+1)/((float)N_extrapolated + 1.);
            }
            else {
                NeutralFractions[i+1] = global_params.PhotonConsAsymptoteTo + (NF_sample_min - global_params.PhotonConsAsymptoteTo)*(float)(i+1)/((float)N_extrapolated + 1.);
            }

            deltaz[i+1] = deltaz[0] + ( deltaz[1+N_extrapolated] - deltaz[0] )*(float)(i+1)/((float)N_extrapolated + 1.);
        }
    }

    // We have added the extrapolated values, now check if they are all increasing or not (again, to determine whether or not to try and smooth the corrected curve
    increasing_val = 0;
    counter = 0;

    for(i=0;i<(N_NFsamples-1);i++) {
        if(deltaz[i+1+N_extrapolated] >= deltaz[i+N_extrapolated]) {
            counter += 1;
        }
    }
    if(counter==(N_NFsamples-1)) {
        increasing_val = 1;
    }

    // For some models, the resultant delta z for extremely high neutral fractions ( > 0.95) seem to oscillate or sometimes drop in value.
    // This goes through and checks if this occurs, and tries to smooth this out
    // This doesn't occur very often, but can cause an artificial drop in the reionisation history (neutral fraction value) connecting the
    // values before/after the photon non-conservation correction starts.
    for(i=0;i<(N_NFsamples+N_extrapolated);i++) {

        val1 = deltaz[i];
        val2 = deltaz[i+1];

        counter = 0;

        // Check if we have a neutral fraction above 0.95, that the values are decreasing (val2 < val1), that we haven't sampled too many points (counter)
        // and that the NF_sample_min is less than around 0.8. That is, if a reasonable fraction of the reionisation history is sampled.
        while( NeutralFractions[i+1] > 0.95 && val2 < val1 && NF_sample_min < 0.8 && counter < 100) {

            NF_sample = global_params.PhotonConsStart - 0.001*(counter+1);

            // Determine redshift given a neutral fraction for the calibration curve
            z_at_NFHist(NF_sample,&(temp));
            z_cal = temp;

            // Determine redshift given a neutral fraction for the analytic curve
            z_at_Q(1. - NF_sample,&(temp));
            z_analytic = temp;

            // Determine the delta z
            val2 = fabs( z_cal - z_analytic );
            deltaz[i+1] = val2;
            counter += 1;

            // If after 100 samplings we couldn't get the value to increase (like it should), just modify it from the previous point.
            if(counter==100) {
                deltaz[i+1] = deltaz[i] * 1.01;
            }

        }
    }

    // Store the data in its intermediate state before averaging
    for(i=0;i<(N_NFsamples+N_extrapolated+1);i++) {
        deltaz_smoothed[i] = deltaz[i];
    }

    // If we are not increasing for all values, we can smooth out some features in delta z when connecting the extrapolated delta z values
    // compared to those from the exact correction (i.e. when we cross the threshold).
    if(!increasing_val) {

        for(i=0;i<(N_NFsamples+N_extrapolated);i++) {

            val1 = deltaz[0]; //NOTE (jdavies): should this be deltaz[i]????
            val2 = deltaz[i+1];

            counter = 0;
            // Try and find a point which can be used to smooth out any dip in delta z as a function of neutral fraction.
            // It can be flat, then drop, then increase. This smooths over this drop (removes a kink in the resultant reionisation history).
            // Choice of 75 is somewhat arbitrary
            while(val2 < val1 && (counter < 75 || (1+(i+1)+counter) > (N_NFsamples+N_extrapolated))) {
                counter += 1;
                val2 = deltaz[i+1+counter];

                deltaz_smoothed[i+1] = ( val1 + deltaz[1+(i+1)+counter] )/2.;
            }
            if(counter==75 || (1+(i+1)+counter) > (N_NFsamples+N_extrapolated)) {
                deltaz_smoothed[i+1] = deltaz[i+1];
            }
        }
    }

    // Here we effectively filter over the delta z as a function of neutral fraction to try and minimise any possible kinks etc. in the functional curve.
    for(i=0;i<(N_NFsamples+N_extrapolated+1);i++) {

        // We are at the end-points, cannot smooth
        if(i==0 || i==(N_NFsamples+N_extrapolated)) {
            deltaz[i] = deltaz_smoothed[i];
        }
        else {

            deltaz[i] = 0.;

            // We are symmetrically smoothing, making sure we have the same number of data points either side of the point we are filtering over
            // This determins the filter width when close to the edge of the data ranges
            if( (i - (int)floor(smoothing_width/2.) ) < 0) {
                smoothing_int = 2*( i ) + (int)((int)smoothing_width%2);
            }
            else if( (i - (int)floor(smoothing_width/2.) + ((int)smoothing_width - 1) ) > (N_NFsamples + N_extrapolated) ) {
                smoothing_int = ((int)smoothing_width - 1) - 2*((i - (int)floor(smoothing_width/2.) + ((int)smoothing_width - 1) ) - (N_NFsamples + N_extrapolated)  ) + (int)((int)smoothing_width%2);
            }
            else {
                smoothing_int = (int)smoothing_width;
            }

            // Average (filter) over the delta z values to smooth the result
            counter = 0;
            for(j=0;j<(int)smoothing_width;j++) {
                if(((i - (int)floor((float)smoothing_int/2.) + j)>=0) && ((i - (int)floor((float)smoothing_int/2.) + j) <= (N_NFsamples + N_extrapolated + 1)) && counter < smoothing_int ) {

                    deltaz[i] += deltaz_smoothed[i - (int)floor((float)smoothing_int/2.) + j];
                    counter += 1;

                }
            }
            deltaz[i] /= (float)counter;
        }

    }

    N_deltaz = N_NFsamples + N_extrapolated + 1;

    // Now, we can construct the spline of the photon non-conservation correction (delta z as a function of neutral fraction)
    deltaz_spline_for_photoncons_acc = gsl_interp_accel_alloc ();
    deltaz_spline_for_photoncons = gsl_spline_alloc (gsl_interp_linear, N_NFsamples + N_extrapolated + 1);

    gsl_set_error_handler_off();
    int gsl_status;
    gsl_status = gsl_spline_init(deltaz_spline_for_photoncons, NeutralFractions, deltaz, N_NFsamples + N_extrapolated + 1);
    CATCH_GSL_ERROR(gsl_status);

}


void adjust_redshifts_for_photoncons(
    AstroParams *astro_params, FlagOptions *flag_options, float *redshift,
    float *stored_redshift, float *absolute_delta_z
) {

    int i, new_counter;
    double temp;
    float required_NF, adjusted_redshift, future_z, gradient_extrapolation, const_extrapolation, temp_redshift, check_required_NF;

    LOG_DEBUG("Adjusting redshifts for photon cons.");

    if(*redshift < global_params.PhotonConsEndCalibz) {
        LOG_ERROR(
            "You have passed a redshift (z = %f) that is lower than the enpoint of the photon non-conservation correction "\
            "(global_params.PhotonConsEndCalibz = %f). If this behaviour is desired then set global_params.PhotonConsEndCalibz "\
            "to a value lower than z = %f.",*redshift,global_params.PhotonConsEndCalibz,*redshift
                  );
//        Throw(ParameterError);
        Throw(PhotonConsError);
    }

    // Determine the neutral fraction (filling factor) of the analytic calibration expression given the current sampled redshift
    Q_at_z(*redshift, &(temp));
    required_NF = 1.0 - (float)temp;

    // Find which redshift we need to sample in order for the calibration reionisation history to match the analytic expression
    if(required_NF > global_params.PhotonConsStart) {
        // We haven't started ionising yet, so keep redshifts the same
        adjusted_redshift = *redshift;

        *absolute_delta_z = 0.;
    }
    else if(required_NF<=global_params.PhotonConsEnd) {
        // We have gone beyond the threshold for the end of the photon non-conservation correction
        // Deemed to be roughly where the calibration curve starts to approach the analytic expression

        if(FirstNF_Estimate <= 0. && required_NF <= 0.0) {
            // Reionisation has already happened well before the calibration
            adjusted_redshift = *redshift;
        }
        else {
            // Initialise the photon non-conservation correction curve
            // It is possible that for certain parameter choices that we can get here without initialisation happening.
            // Thus check and initialise if not already done so
            if(!photon_cons_allocated) {
                determine_deltaz_for_photoncons();
                photon_cons_allocated = true;
            }

            // We have crossed the NF threshold for the photon conservation correction so now set to the delta z at the threshold
            if(required_NF < global_params.PhotonConsAsymptoteTo) {

                // This counts the number of times we have exceeded the extrapolated point and attempts to modify the delta z
                // to try and make the function a little smoother
                *absolute_delta_z = gsl_spline_eval(deltaz_spline_for_photoncons, global_params.PhotonConsAsymptoteTo, deltaz_spline_for_photoncons_acc);

                new_counter = 0;
                temp_redshift = *redshift;
                check_required_NF = required_NF;

                // Ok, find when in the past we exceeded the asymptote threshold value using the global_params.ZPRIME_STEP_FACTOR
                // In doing it this way, co-eval boxes will be the same as lightcone boxes with regard to redshift sampling
                while( check_required_NF < global_params.PhotonConsAsymptoteTo ) {

                    temp_redshift = ((1. + temp_redshift)*global_params.ZPRIME_STEP_FACTOR - 1.);

                    Q_at_z(temp_redshift, &(temp));
                    check_required_NF = 1.0 - (float)temp;

                    new_counter += 1;
                }

                // Now adjust the final delta_z by some amount to smooth if over successive steps
                if(deltaz[1] > deltaz[0]) {
                    *absolute_delta_z = pow( 0.96 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
                }
                else {
                    *absolute_delta_z = pow( 1.04 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
                }

                // Check if we go into the future (z < 0) and avoid it
                adjusted_redshift = (*redshift) - (*absolute_delta_z);
                if(adjusted_redshift < 0.0) {
                    adjusted_redshift = 0.0;
                }

            }
            else {
                *absolute_delta_z = gsl_spline_eval(deltaz_spline_for_photoncons, required_NF, deltaz_spline_for_photoncons_acc);
                adjusted_redshift = (*redshift) - (*absolute_delta_z);
            }
        }
    }
    else {
        // Initialise the photon non-conservation correction curve
        if(!photon_cons_allocated) {
            determine_deltaz_for_photoncons();
            photon_cons_allocated = true;
        }

        // We have exceeded even the end-point of the extrapolation
        // Just smooth ever subsequent point
        // Note that this is deliberately tailored to light-cone quantites, but will still work with co-eval cubes
        // Though might produce some very minor discrepancies when comparing outputs.
        if(required_NF < NeutralFractions[0]) {

            new_counter = 0;
            temp_redshift = *redshift;
            check_required_NF = required_NF;

            // Ok, find when in the past we exceeded the asymptote threshold value using the global_params.ZPRIME_STEP_FACTOR
            // In doing it this way, co-eval boxes will be the same as lightcone boxes with regard to redshift sampling
            while( check_required_NF < NeutralFractions[0] ) {

                temp_redshift = ((1. + temp_redshift)*global_params.ZPRIME_STEP_FACTOR - 1.);

                Q_at_z(temp_redshift, &(temp));
                check_required_NF = 1.0 - (float)temp;

                new_counter += 1;
            }
            if(new_counter > 5) {
                LOG_WARNING(
                    "The photon non-conservation correction has employed an extrapolation for\n"\
                    "more than 5 consecutive snapshots. This can be unstable, thus please check "\
                    "resultant history. Parameters are:\n"
                );
                #if LOG_LEVEL >= LOG_WARNING
                    writeAstroParams(flag_options, astro_params);
                #endif
            }

            // Now adjust the final delta_z by some amount to smooth if over successive steps
            if(deltaz[1] > deltaz[0]) {
                *absolute_delta_z = pow( 0.998 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
            }
            else {
                *absolute_delta_z = pow( 1.002 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
            }

            // Check if we go into the future (z < 0) and avoid it
            adjusted_redshift = (*redshift) - (*absolute_delta_z);
            if(adjusted_redshift < 0.0) {
                adjusted_redshift = 0.0;
            }
        }
        else {
            // Find the corresponding redshift for the calibration curve given the required neutral fraction (filling factor) from the analytic expression
            *absolute_delta_z = gsl_spline_eval(deltaz_spline_for_photoncons, (double)required_NF, deltaz_spline_for_photoncons_acc);
            adjusted_redshift = (*redshift) - (*absolute_delta_z);
        }
    }

    // keep the original sampled redshift
    *stored_redshift = *redshift;

    // This redshift snapshot now uses the modified redshift following the photon non-conservation correction
    *redshift = adjusted_redshift;
}

void Q_at_z(double z, double *splined_value){
    float returned_value;

    if (z >= Zmax) {
        *splined_value = 0.;
    }
    else if (z <= Zmin) {
        *splined_value = 1.;
    }
    else {
        returned_value = gsl_spline_eval(Q_at_z_spline, z, Q_at_z_spline_acc);
        *splined_value = returned_value;
    }
}

void z_at_Q(double Q, double *splined_value){
    float returned_value;

    if (Q < Qmin) {
        LOG_ERROR("The minimum value of Q is %.4e",Qmin);
//        Throw(ParameterError);
        Throw(PhotonConsError);
    }
    else if (Q > Qmax) {
        LOG_ERROR("The maximum value of Q is %.4e. Reionization ends at ~%.4f.",Qmax,Zmin);
        LOG_ERROR("This error can occur if global_params.PhotonConsEndCalibz is close to "\
                  "the final sampled redshift. One can consider a lower value for "\
                  "global_params.PhotonConsEndCalibz to mitigate this");
//        Throw(ParameterError);
        Throw(PhotonConsError);
    }
    else {
        returned_value = gsl_spline_eval(z_at_Q_spline, Q, z_at_Q_spline_acc);
        *splined_value = returned_value;
    }
}

void free_Q_value() {
    gsl_spline_free (Q_at_z_spline);
    gsl_interp_accel_free (Q_at_z_spline_acc);
    gsl_spline_free (z_at_Q_spline);
    gsl_interp_accel_free (z_at_Q_spline_acc);
}

void initialise_NFHistory_spline(double *redshifts, double *NF_estimate, int NSpline){

    int i, counter, start_index, found_start_index;

    // This takes in the data for the calibration curve for the photon non-conservation correction

    counter = 0;
    start_index = 0;
    found_start_index = 0;

    FinalNF_Estimate = NF_estimate[0];
    FirstNF_Estimate = NF_estimate[NSpline-1];

    // Determine the point in the data where its no longer zero (basically to avoid too many zeros in the spline)
    for(i=0;i<NSpline-1;i++) {
        if(NF_estimate[i+1] > NF_estimate[i]) {
            if(found_start_index == 0) {
                start_index = i;
                found_start_index = 1;
            }
        }
        counter += 1;
    }
    counter = counter - start_index;

    N_calibrated = (counter+1);

    // Store the data points for determining the photon non-conservation correction
    nf_vals = calloc((counter+1),sizeof(double));
    z_vals = calloc((counter+1),sizeof(double));

    calibrated_NF_min = 1.;

    // Store the data, and determine the end point of the input data for estimating the extrapolated results
    for(i=0;i<(counter+1);i++) {
        nf_vals[i] = NF_estimate[start_index+i];
        z_vals[i] = redshifts[start_index+i];
        // At the extreme high redshift end, there can be numerical issues with the solution of the analytic expression
        if(i>0) {
            while(nf_vals[i] <= nf_vals[i-1]) {
                nf_vals[i] += 0.000001;
            }
        }

        if(nf_vals[i] < calibrated_NF_min) {
            calibrated_NF_min = nf_vals[i];
        }
    }

    NFHistory_spline_acc = gsl_interp_accel_alloc ();
//    NFHistory_spline = gsl_spline_alloc (gsl_interp_cspline, (counter+1));
    NFHistory_spline = gsl_spline_alloc (gsl_interp_linear, (counter+1));

    gsl_set_error_handler_off();
    int gsl_status;
    gsl_status = gsl_spline_init(NFHistory_spline, nf_vals, z_vals, (counter+1));
    CATCH_GSL_ERROR(gsl_status);

    z_NFHistory_spline_acc = gsl_interp_accel_alloc ();
//    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_cspline, (counter+1));
    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_linear, (counter+1));

    gsl_status = gsl_spline_init(z_NFHistory_spline, z_vals, nf_vals, (counter+1));
    CATCH_GSL_ERROR(gsl_status);
}


void z_at_NFHist(double xHI_Hist, double *splined_value){
    float returned_value;

    returned_value = gsl_spline_eval(NFHistory_spline, xHI_Hist, NFHistory_spline_acc);
    *splined_value = returned_value;
}

void NFHist_at_z(double z, double *splined_value){
    float returned_value;

    returned_value = gsl_spline_eval(z_NFHistory_spline, z, NFHistory_spline_acc);
    *splined_value = returned_value;
}

int ObtainPhotonConsData(
    double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data,
    double *nf_cal_data, int *Ndata_calibration,
    double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons) {

    int i;

    *Ndata_analytic = N_analytic;
    *Ndata_calibration = N_calibrated;
    *Ndata_PhotonCons = N_deltaz;

    for(i=0;i<N_analytic;i++) {
        z_at_Q_data[i] = z_Q[i];
        Q_data[i] = Q_value[i];
    }

    for(i=0;i<N_calibrated;i++) {
        z_cal_data[i] = z_vals[i];
        nf_cal_data[i] = nf_vals[i];
    }

    for(i=0;i<N_deltaz;i++) {
        PhotonCons_NFdata[i] = NeutralFractions[i];
        PhotonCons_deltaz[i] = deltaz[i];
    }

    return(0);
}

void FreePhotonConsMemory() {
    LOG_DEBUG("Freeing some photon cons memory.");
    free(deltaz);
    free(deltaz_smoothed);
    free(NeutralFractions);
    free(z_Q);
    free(Q_value);
    free(nf_vals);
    free(z_vals);

    free_Q_value();

    gsl_spline_free (NFHistory_spline);
    gsl_interp_accel_free (NFHistory_spline_acc);
    gsl_spline_free (z_NFHistory_spline);
    gsl_interp_accel_free (z_NFHistory_spline_acc);
    gsl_spline_free (deltaz_spline_for_photoncons);
    gsl_interp_accel_free (deltaz_spline_for_photoncons_acc);
    LOG_DEBUG("Done Freeing photon cons memory.");

    photon_cons_allocated = false;
}

//quick functions to set/get alpha photoncons parameters
//TODO: pass directly from python into a struct
double fesc_photoncons_yint;
double fesc_photoncons_slope;
void set_alphacons_params(double yint, double slope){
    fesc_photoncons_yint = yint;
    fesc_photoncons_slope = slope;
}

//NOTE: The way I've set this up is a little annoying in that this function needs to match its counterpart in Python
double get_fesc_fit(double redshift){
    double Q, fesc_fit;
    Q_at_z(redshift,&Q);
    if(Q > 1.) Q = 1.;
    fesc_fit = fesc_photoncons_yint + Q*fesc_photoncons_slope;
    LOG_DEBUG("Alpha photon cons fit activated z = %.2e, fit yint,slope = %.2e, %.2e, alpha = %.2e", redshift,
                fesc_photoncons_yint,fesc_photoncons_slope,fesc_fit);
    return fesc_fit;
}
