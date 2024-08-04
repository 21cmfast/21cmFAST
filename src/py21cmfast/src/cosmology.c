/* This file contatins functions regarding the matter power-sepctrum and cosmology */
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <gsl_integration.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "cosmology.h"

//Gauss-Legendre integration constants
#define NGL_INT 100 // 100

#define MAX_DELTAC_FRAC (float)0.99 //max delta/deltac for the mass function integrals
#define JENKINS_a (0.73) //Jenkins+01, SMT has 0.707
#define JENKINS_b (0.34) //Jenkins+01 fit from Barkana+01, SMT has 0.5
#define JENKINS_c (0.81) //Jenkins+01 from from Barkana+01, SMT has 0.6

//These globals hold values initialised in init_ps() and used throughout the rest of the file
static double sigma_norm, theta_cmb, omhh, z_equality, y_d, sound_horizon, alpha_nu, f_nu, f_baryon, beta_c, d2fact, R_CUTOFF, DEL_CURR, SIG_CURR;

/*
  this function reads the z=0 matter (CDM+baryons)  and relative velocity transfer functions from CLASS (from a file)
  flag_int = 0 to initialize interpolator, flag_int = -1 to free memory, flag_int = else to interpolate.
  flag_dv = 0 to output density, flag_dv = 1 to output velocity.
  similar to built-in function "double T_RECFAST(float z, int flag)"
*/

double TF_CLASS(double k, int flag_int, int flag_dv)
{
    static double kclass[CLASS_LENGTH], Tmclass[CLASS_LENGTH], Tvclass_vcb[CLASS_LENGTH];
    static gsl_interp_accel *acc_density, *acc_vcb;
    static gsl_spline *spline_density, *spline_vcb;
    float trash, currk, currTm, currTv;
    double ans;
    int i;
    int gsl_status;
    FILE *F;

    char filename[500];
    sprintf(filename,"%s/%s",global_params.external_table_path,CLASS_FILENAME);


    if (flag_int == 0) {  // Initialize vectors and read file
        if (!(F = fopen(filename, "r"))) {
            LOG_ERROR("Unable to open file: %s for reading.", filename);
            Throw(IOError);
        }

        int nscans;
        for (i = 0; i < CLASS_LENGTH; i++) {
            nscans = fscanf(F, "%e %e %e ", &currk, &currTm, &currTv);
            if (nscans != 3) {
                LOG_ERROR("Reading CLASS Transfer Function failed.");
                Throw(IOError);
            }
            kclass[i] = currk;
            Tmclass[i] = currTm;
            Tvclass_vcb[i] = currTv;
            if (i > 0 && kclass[i] <= kclass[i - 1]) {
                LOG_WARNING("Tk table not ordered");
                LOG_WARNING("k=%.1le kprev=%.1le", kclass[i], kclass[i - 1]);
            }
        }
        fclose(F);


        LOG_SUPER_DEBUG("Read CLASS Transfer file");

        gsl_set_error_handler_off();
        // Set up spline table for densities
        acc_density   = gsl_interp_accel_alloc ();
        spline_density  = gsl_spline_alloc (gsl_interp_cspline, CLASS_LENGTH);
        gsl_status = gsl_spline_init(spline_density, kclass, Tmclass, CLASS_LENGTH);
        GSL_ERROR(gsl_status);

        LOG_SUPER_DEBUG("Generated CLASS Density Spline.");

        //Set up spline table for velocities
        acc_vcb   = gsl_interp_accel_alloc ();
        spline_vcb  = gsl_spline_alloc (gsl_interp_cspline, CLASS_LENGTH);
        gsl_status = gsl_spline_init(spline_vcb, kclass, Tvclass_vcb, CLASS_LENGTH);
        GSL_ERROR(gsl_status);

        LOG_SUPER_DEBUG("Generated CLASS velocity Spline.");
        return 0;
    }
    else if (flag_int == -1) {
        gsl_spline_free (spline_density);
        gsl_interp_accel_free(acc_density);
        gsl_spline_free (spline_vcb);
        gsl_interp_accel_free(acc_vcb);
        return 0;
    }


    if (k > kclass[CLASS_LENGTH-1]) { // k>kmax
        LOG_WARNING("Called TF_CLASS with k=%f, larger than kmax! Returning value at kmax.", k);
        if(flag_dv == 0){ // output is density
            return (Tmclass[CLASS_LENGTH]/kclass[CLASS_LENGTH-1]/kclass[CLASS_LENGTH-1]);
        }
        else if(flag_dv == 1){ // output is rel velocity
            return (Tvclass_vcb[CLASS_LENGTH]/kclass[CLASS_LENGTH-1]/kclass[CLASS_LENGTH-1]);
        }    //we just set it to the last value, since sometimes it wants large k for R<<cell_size, which does not matter much.
    }
    else { // Do spline
        if(flag_dv == 0){ // output is density
            ans = gsl_spline_eval (spline_density, k, acc_density);
        }
        else if(flag_dv == 1){ // output is relative velocity
            ans = gsl_spline_eval (spline_vcb, k, acc_vcb);
        }
        else{
            ans=0.0; //neither densities not velocities?
        }
    }

    return ans/k/k;
    //we have to divide by k^2 to agree with the old-fashioned convention.

}

// FUNCTION sigma_z0(M)
// Returns the standard deviation of the normalized, density excess (delta(x)) field,
// smoothed on the comoving scale of M (see filter definitions for M<->R conversion).
// The sigma is evaluated at z=0, with the time evolution contained in the dicke(z) factor,
// i.e. sigma(M,z) = sigma_z0(m) * dicke(z)

// normalized so that sigma_z0(M->8/h Mpc) = SIGMA8 in ../Parameter_files/COSMOLOGY.H

// NOTE: volume is normalized to = 1, so this is equvalent to the mass standard deviation

// M is in solar masses

// References: Padmanabhan, pg. 210, eq. 5.107
double dsigma_dk(double k, void *params){
    double p, w, T, gamma, q, aa, bb, cc, kR;

    // get the power spectrum.. choice of 5:
    if (user_params_global->POWER_SPECTRUM == 0){ // Eisenstein & Hu
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
    }
    else if (user_params_global->POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
        q = k / (cosmo_params_global->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
    }
    else if (user_params_global->POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_global->hlittle*gamma);
        bb = 3.0/(cosmo_params_global->hlittle*gamma);
        cc = 1.7/(cosmo_params_global->hlittle*gamma);
        p = pow(k, cosmo_params_global->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (user_params_global->POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
        aa = 8.0 / (cosmo_params_global->hlittle*gamma);
        bb = 4.7 / pow(cosmo_params_global->hlittle*gamma, 2);
        p = pow(k, cosmo_params_global->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (user_params_global->POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
        aa = 1.7/(cosmo_params_global->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_global->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_global->hlittle*gamma, 2);
        p = pow(k, cosmo_params_global->POWER_INDEX) * 19400.0 / pow(1 + aa*k + bb*pow(k, 1.5) + cc*k*k, 2);
    }
    else if (user_params_global->POWER_SPECTRUM == 5){ // output of CLASS
        T = TF_CLASS(k, 1, 0); //read from z=0 output of CLASS. Note, flag_int = 1 here always, since now we have to have initialized the interpolator for CLASS
  	    p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
        if(user_params_global->USE_RELATIVE_VELOCITIES) { //jbm:Add average relvel suppression
          p *= 1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM)); //for v=vrms
        }
    }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_global->POWER_SPECTRUM);
        Throw(ValueError);
    }
    double Radius;

    Radius = *(double *)params;

    kR = k*Radius;

    if ( (global_params.FILTER == 0) || (sigma_norm < 0) ){ // top hat
        if ( (kR) < 1.0e-4 ){ w = 1.0;} // w converges to 1 as (kR) -> 0
        else { w = 3.0 * (sin(kR)/pow(kR, 3) - cos(kR)/pow(kR, 2));}
    }
    else if (global_params.FILTER == 1){ // gaussian of width 1/R
        w = pow(E, -kR*kR/2.0);
    }
    else {
        LOG_ERROR("No such filter: %i. Output is bogus.", global_params.FILTER);
        Throw(ValueError);
    }
    return k*k*p*w*w;
}
double sigma_z0(double M){

    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double kstart, kend;

    double Radius;
//    R = MtoR(M);

    Radius = MtoR(M);
    // now lets do the integral for sigma and scale it with sigma_norm

    if(user_params_global->POWER_SPECTRUM == 5){
      kstart = fmax(1.0e-99/Radius, KBOT_CLASS);
      kend = fmin(350.0/Radius, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at KBOT_CLASS,~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-99/Radius;
      kend = 350.0/Radius;
    }

    lower_limit = kstart;//log(kstart);
    upper_limit = kend;//log(kend);

    F.function = &dsigma_dk;
    F.params = &Radius;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: M=%e",M);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return sigma_norm * sqrt(result);
}

//NOTE: There's certainly a better way to do this
double sigma_inverse(double sigma, double lnM_init){
        double rf_tol_abs = 1e-4;
        double rf_tol_rel = 0.;
        int status, iter;
        const gsl_root_fsolver_type *T;
        gsl_root_fsolver *solver;
        gsl_function F;
        params_rf.growthf = growth_out;

        F.function = &sigma_z0; //Maybe EvaluateSigma for speed?
        F.params = NULL;

        double lnM_low,lnM_high,lnM_guess;

        T = gsl_root_fsolver_brent;
        solver = gsl_root_fsolver_alloc(T);
        gsl_root_fsolver_set(solver, &F, lnM_init, lnM_cond);
        do{
            iter++;
            status = gsl_root_fsolver_iterate(solver);
            lnM_guess = gsl_root_fsolver_root(solver);
            lnM_lo = gsl_root_fsolver_x_lower(solver);
            lnM_hi = gsl_root_fsolver_x_upper(solver);
            status = gsl_root_test_interval(lnM_lo, lnM_hi, rf_tol_abs, rf_tol_rel);

            if (status == GSL_SUCCESS){
                lnM_init = lnM_lo;
                break;
            }

        }while((status == GSL_CONTINUE) && (iter < MAX_ITER_RF));
        if(status!=GSL_SUCCESS) {
            LOG_ERROR("gsl RF error occured! %d",status);
            GSL_ERROR(status);
        }
        gsl_root_fsolver_free(solver);
        return lnM_guess;
}


// FUNCTION TFmdm is the power spectrum transfer function from Eisenstein & Hu ApJ, 1999, 511, 5
double TFmdm(double k){
    double q, gamma_eff, q_eff, TF_m, q_nu;

    q = k*pow(theta_cmb,2)/omhh;
    gamma_eff=sqrt(alpha_nu) + (1.0-sqrt(alpha_nu))/(1.0+pow(0.43*k*sound_horizon, 4));
    q_eff = q/gamma_eff;
    TF_m= log(E+1.84*beta_c*sqrt(alpha_nu)*q_eff);
    TF_m /= TF_m + pow(q_eff,2) * (14.4 + 325.0/(1.0+60.5*pow(q_eff,1.11)));
    q_nu = 3.92*q/sqrt(f_nu/N_nu);
    TF_m *= 1.0 + (1.2*pow(f_nu,0.64)*pow(N_nu,0.3+0.6*f_nu)) /
    (pow(q_nu,-1.6)+pow(q_nu,0.8));

    return TF_m;
}


void TFset_parameters(){
    double z_drag, R_drag, R_equality, p_c, p_cb, f_c, f_cb, f_nub, k_equality;

    LOG_DEBUG("Setting Transfer Function parameters.");

    z_equality = 25000*omhh*pow(theta_cmb, -4) - 1.0;
    k_equality = 0.0746*omhh/(theta_cmb*theta_cmb);

    z_drag = 0.313*pow(omhh,-0.419) * (1 + 0.607*pow(omhh, 0.674));
    z_drag = 1 + z_drag*pow(cosmo_params_global->OMb*cosmo_params_global->hlittle*cosmo_params_global->hlittle, 0.238*pow(omhh, 0.223));
    z_drag *= 1291 * pow(omhh, 0.251) / (1 + 0.659*pow(omhh, 0.828));

    y_d = (1 + z_equality) / (1.0 + z_drag);

    R_drag = 31.5 * cosmo_params_global->OMb*cosmo_params_global->hlittle*cosmo_params_global->hlittle * pow(theta_cmb, -4) * 1000 / (1.0 + z_drag);
    R_equality = 31.5 * cosmo_params_global->OMb*cosmo_params_global->hlittle*cosmo_params_global->hlittle * pow(theta_cmb, -4) * 1000 / (1.0 + z_equality);

    sound_horizon = 2.0/3.0/k_equality * sqrt(6.0/R_equality) *
    log( (sqrt(1+R_drag) + sqrt(R_drag+R_equality)) / (1.0 + sqrt(R_equality)) );

    p_c = -(5 - sqrt(1 + 24*(1 - f_nu-f_baryon)))/4.0;
    p_cb = -(5 - sqrt(1 + 24*(1 - f_nu)))/4.0;
    f_c = 1 - f_nu - f_baryon;
    f_cb = 1 - f_nu;
    f_nub = f_nu+f_baryon;

    alpha_nu = (f_c/f_cb) * (2*(p_c+p_cb)+5)/(4*p_cb+5.0);
    alpha_nu *= 1 - 0.553*f_nub+0.126*pow(f_nub,3);
    alpha_nu /= 1-0.193*sqrt(f_nu)+0.169*f_nu;
    alpha_nu *= pow(1+y_d, p_c-p_cb);
    alpha_nu *= 1+ (p_cb-p_c)/2.0 * (1.0+1.0/(4.0*p_c+3.0)/(4.0*p_cb+7.0))/(1.0+y_d);
    beta_c = 1.0/(1.0-0.949*f_nub);
}


// Returns the value of the linear power spectrum DENSITY (i.e. <|delta_k|^2>/V)
// at a given k mode linearly extrapolated to z=0
double power_in_k(double k){
    double p, T, gamma, q, aa, bb, cc;

    // get the power spectrum.. choice of 5:
    if (user_params_global->POWER_SPECTRUM == 0){ // Eisenstein & Hu
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
        //p = pow(k, POWER_INDEX - 0.05*log(k/0.05)) * T * T; //running, alpha=0.05
    }
    else if (user_params_global->POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
        q = k / (cosmo_params_global->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
    }
    else if (user_params_global->POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_global->hlittle*gamma);
        bb = 3.0/(cosmo_params_global->hlittle*gamma);
        cc = 1.7/(cosmo_params_global->hlittle*gamma);
        p = pow(k, cosmo_params_global->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (user_params_global->POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb)/(cosmo_params_global->OMm));
        aa = 8.0 / (cosmo_params_global->hlittle*gamma);
        bb = 4.7 / pow(cosmo_params_global->hlittle*gamma, 2);
        p = pow(k, cosmo_params_global->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (user_params_global->POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
        aa = 1.7/(cosmo_params_global->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_global->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_global->hlittle*gamma, 2);
        p = pow(k, cosmo_params_global->POWER_INDEX) * 19400.0 / pow(1 + aa*k + bb*pow(k, 1.5) + cc*k*k, 2);
    }
    else if (user_params_global->POWER_SPECTRUM == 5){ // output of CLASS
        T = TF_CLASS(k, 1, 0); //read from z=0 output of CLASS. Note, flag_int = 1 here always, since now we have to have initialized the interpolator for CLASS
  	    p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
        if(user_params_global->USE_RELATIVE_VELOCITIES) { //jbm:Add average relvel suppression
          p *= 1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM)); //for v=vrms
        }
    }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_global->POWER_SPECTRUM);
        Throw(ValueError);
    }


    return p*TWOPI*PI*sigma_norm*sigma_norm;
}


/*
  Returns the value of the linear power spectrum of the DM-b relative velocity
  at kinematic decoupling (which we set at zkin=1010)
*/
double power_in_vcb(double k){

    double p, T, gamma, q, aa, bb, cc;

    //only works if using CLASS
    if (user_params_global->POWER_SPECTRUM == 5){ // CLASS
        T = TF_CLASS(k, 1, 1); //read from CLASS file. flag_int=1 since we have initialized before, flag_vcb=1 for velocity
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
    }
    else{
        LOG_ERROR("Cannot get P_cb unless using CLASS: %i\n Set USE_RELATIVE_VELOCITIES 0 or use CLASS.\n", user_params_global->POWER_SPECTRUM);
        Throw(ValueError);
    }

    return p*TWOPI*PI*sigma_norm*sigma_norm;
}


double init_ps(){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double kstart, kend;

    //we start the interpolator if using CLASS:
    if (user_params_global->POWER_SPECTRUM == 5){
        LOG_DEBUG("Setting CLASS Transfer Function inits.");
        TF_CLASS(1.0, 0, 0);
    }

    // Set cuttoff scale for WDM (eq. 4 in Barkana et al. 2001) in comoving Mpc
    R_CUTOFF = 0.201*pow((cosmo_params_global->OMm-cosmo_params_global->OMb)*cosmo_params_global->hlittle*cosmo_params_global->hlittle/0.15, 0.15)*pow(global_params.g_x/1.5, -0.29)*pow(global_params.M_WDM, -1.15);

    omhh = cosmo_params_global->OMm*cosmo_params_global->hlittle*cosmo_params_global->hlittle;
    theta_cmb = T_cmb / 2.7;

    // Translate Parameters into forms GLOBALVARIABLES form
    f_nu = global_params.OMn/cosmo_params_global->OMm;
    f_baryon = cosmo_params_global->OMb/cosmo_params_global->OMm;
    if (f_nu < TINY) f_nu = 1e-10;
    if (f_baryon < TINY) f_baryon = 1e-10;

    TFset_parameters();

    sigma_norm = -1;

    double Radius_8;
    Radius_8 = 8.0/cosmo_params_global->hlittle;

    if(user_params_global->POWER_SPECTRUM == 5){
      kstart = fmax(1.0e-99/Radius_8, KBOT_CLASS);
      kend = fmin(350.0/Radius_8, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at KBOT_CLASS,~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-99/Radius_8;
      kend = 350.0/Radius_8;
    }

    lower_limit = kstart;
    upper_limit = kend;

    LOG_DEBUG("Initializing Power Spectrum with lower_limit=%e, upper_limit=%e, rel_tol=%e, radius_8=%g", lower_limit,upper_limit, rel_tol, Radius_8);

    F.function = &dsigma_dk;
    F.params = &Radius_8;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    LOG_DEBUG("Initialized Power Spectrum.");

    sigma_norm = cosmo_params_global->SIGMA_8/sqrt(result); //takes care of volume factor
    return R_CUTOFF;
}




//function to free arrays related to the power spectrum
void free_ps(){

	//we free the PS interpolator if using CLASS:
	if (user_params_global->POWER_SPECTRUM == 5){
		TF_CLASS(1.0, -1, 0);
	}

  return;
}



/*
 FUNCTION dsigmasqdm_z0(M)
 returns  d/dm (sigma^2) (see function sigma), in units of Msun^-1
 */
double dsigmasq_dm(double k, void *params){
    double p, w, T, gamma, q, aa, bb, cc, dwdr, drdm, kR;

    // get the power spectrum.. choice of 5:
    if (user_params_global->POWER_SPECTRUM == 0){ // Eisenstein & Hu ApJ, 1999, 511, 5
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
        //p = pow(k, POWER_INDEX - 0.05*log(k/0.05)) * T * T; //running, alpha=0.05
    }
    else if (user_params_global->POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb)/(cosmo_params_global->OMm));
        q = k / (cosmo_params_global->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
    }
    else if (user_params_global->POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_global->hlittle*gamma);
        bb = 3.0/(cosmo_params_global->hlittle*gamma);
        cc = 1.7/(cosmo_params_global->hlittle*gamma);
        p = pow(k, cosmo_params_global->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (user_params_global->POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb)/(cosmo_params_global->OMm));
        aa = 8.0 / (cosmo_params_global->hlittle*gamma);
        bb = 4.7 / (cosmo_params_global->hlittle*gamma);
        p = pow(k, cosmo_params_global->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (user_params_global->POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * pow(E, -(cosmo_params_global->OMb) - (cosmo_params_global->OMb)/(cosmo_params_global->OMm));
        aa = 1.7/(cosmo_params_global->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_global->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_global->hlittle*gamma, 2);
        p = pow(k, cosmo_params_global->POWER_INDEX) * 19400.0 / pow(1 + aa*k + pow(bb*k, 1.5) + cc*k*k, 2);
    }
    else if (user_params_global->POWER_SPECTRUM == 5){ // JBM: CLASS
      T = TF_CLASS(k, 1, 0); //read from z=0 output of CLASS
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
        if(user_params_global->USE_RELATIVE_VELOCITIES) { //jbm:Add average relvel suppression
          p *= 1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM)); //for v=vrms
        }
      }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_global->POWER_SPECTRUM);
        Throw(ValueError);
    }

    double Radius;
    Radius = *(double *)params;

    // now get the value of the window function
    kR = k * Radius;
    if (global_params.FILTER == 0){ // top hat
        if ( (kR) < 1.0e-4 ){ w = 1.0; }// w converges to 1 as (kR) -> 0
        else { w = 3.0 * (sin(kR)/pow(kR, 3) - cos(kR)/pow(kR, 2));}

        // now do d(w^2)/dm = 2 w dw/dr dr/dm
        if ( (kR) < 1.0e-10 ){  dwdr = 0;}
        else{ dwdr = 9*cos(kR)*k/pow(kR,3) + 3*sin(kR)*(1 - 3/(kR*kR))/(kR*Radius);}
        //3*k*( 3*cos(kR)/pow(kR,3) + sin(kR)*(-3*pow(kR, -4) + 1/(kR*kR)) );}
        //     dwdr = -1e8 * k / (R*1e3);
        drdm = 1.0 / (4.0*PI * cosmo_params_global->OMm*RHOcrit * Radius*Radius);
    }
    else if (global_params.FILTER == 1){ // gaussian of width 1/R
        w = pow(E, -kR*kR/2.0);
        dwdr = - k*kR * w;
        drdm = 1.0 / (pow(2*PI, 1.5) * cosmo_params_global->OMm*RHOcrit * 3*Radius*Radius);
    }
    else {
        LOG_ERROR("No such filter: %i. Output is bogus.", global_params.FILTER);
        Throw(ValueError);
    }

//    return k*k*p*2*w*dwdr*drdm * d2fact;
    return k*k*p*2*w*dwdr*drdm;
}
double dsigmasqdm_z0(double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    double kstart, kend;

    double Radius;
//    R = MtoR(M);
    Radius = MtoR(M);

    // now lets do the integral for sigma and scale it with sigma_norm
    if(user_params_global->POWER_SPECTRUM == 5){
      kstart = fmax(1.0e-99/Radius, KBOT_CLASS);
      kend = fmin(350.0/Radius, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at KBOT_CLASS,~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-99/Radius;
      kend = 350.0/Radius;
    }

    lower_limit = kstart;//log(kstart);
    upper_limit = kend;//log(kend);


    if (user_params_global->POWER_SPECTRUM == 5){ // for CLASS we do not need to renormalize the sigma integral.
      d2fact=1.0;
    }
    else {
      d2fact = M*10000/sigma_z0(M);
    }


    F.function = &dsigmasq_dm;
    F.params = &Radius;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: M=%e",M);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

//    return sigma_norm * sigma_norm * result /d2fact;
    return sigma_norm * sigma_norm * result;
}

////MASS FUNCTIONS BELOW//////

/* sheth correction to delta crit */
double sheth_delc_dexm(double del, double sig){
    return sqrt(SHETH_a)*del*(1. + global_params.SHETH_b*pow(sig*sig/(SHETH_a*del*del), global_params.SHETH_c));
}

/*DexM uses a fit to this barrier to acheive MF similar to ST, Here I use the fixed version for the sampler*/
//NOTE: if I made this a table it would save a pow call per condition in the sampler
double sheth_delc_fixed(double del, double sig){
    return sqrt(JENKINS_a)*del*(1. + JENKINS_b*pow(sig*sig/(JENKINS_a*del*del), JENKINS_c));
}

//Get the relevant excursion set barrier density given the user-specified HMF
double get_delta_crit(int HMF, double sigma, double growthf){
    if(HMF==4)
        return DELTAC_DELOS;
    if(HMF==1)
        return sheth_delc_fixed(Deltac/growthf,sigma)*growthf;

    return Deltac;
}

/*
Unconditional Mass function from Delos 2023 (https://arxiv.org/pdf/2311.17986.pdf)
Matches well with N-bodies (M200), has a corresponding Conditional Mass Function (below) and
an excursion set method. Hence can be consistently used throughout the Halo Finder, Halo Sampler
And radiation. The mass functions are based off a constant barrier delta = 1.5 and a top-hat window function
*/
double dNdlnM_Delos(double growthf, double lnM){
    double dfdnu,dsigmadm,sigma,sigma_inv,dfdM;
    double nu;
    //hardcoded for now
    const double coeff_nu = 0.519;
    const double index_nu = 0.582;
    const double exp_factor = -0.469;

    sigma = EvaluateSigma(lnM);
    sigma_inv = 1/sigma;
    dsigmadm = EvaluatedSigmasqdm(lnM) * (0.5*sigma_inv); //d(s^2)/dm z0 to dsdm

    nu = DELTAC_DELOS*sigma_inv/growthf;

    dfdnu = coeff_nu*pow(nu,index_nu)*exp(exp_factor*nu*nu);
    dfdM = dfdnu * fabs(dsigmadm) * sigma_inv;

    //NOTE: dfdM == constants*dNdlnM
    return dfdM*cosmo_params_global->OMm*RHOcrit;
}

double dNdlnM_conditional_Delos(double growthf, double lnM, double delta_cond, double sigma_cond){
    double result,dfdnu,dsigmadm,sigma,sigdiff_inv,dfdM,sigma_inv;
    double nu;
    //hardcoded for now
    const double coeff_nu = 0.519;
    const double index_nu = 0.582;
    const double exp_factor = -0.469;

    sigma = EvaluateSigma(lnM);
    if(sigma < sigma_cond) return 0.;
    dsigmadm = EvaluatedSigmasqdm(lnM) * 0.5; //d(s^2)/dm to s*dsdm
    sigdiff_inv = sigma == sigma_cond ? 1e6 : 1/(sigma*sigma - sigma_cond*sigma_cond);

    nu = (DELTAC_DELOS - delta_cond)*sqrt(sigdiff_inv)/growthf;

    dfdnu = coeff_nu*pow(nu,index_nu)*exp(exp_factor*nu*nu);
    dfdM = dfdnu * fabs(dsigmadm) * sigdiff_inv;

    //NOTE: like the other CMFs this is dNdlogM and leaves out
    //   the (cosmo_params_global->OMm)*RHOcrit
    //NOTE: dfdM == constants*dNdlnM
    return dfdM;
}

//Sheth Tormen 2002 fit for the CMF, while the moving barrier does not allow for a simple rescaling, it has been found
//That a taylor expansion of the barrier shape around the point of interest well approximates the simulations
double st_taylor_factor(double sig, double sig_cond, double growthf, double *zeroth_order){
    int i;
    double a = JENKINS_a;
    double alpha = JENKINS_c; //fixed instead of global_params.SHETH_c bc of DexM corrections
    double beta = JENKINS_b; //fixed instead of global_params.SHETH_b

    double del = Deltac/growthf;

    double sigsq = sig*sig;
    double sigsq_inv = 1./sigsq;
    double sigcsq = sig_cond*sig_cond;
    double sigdiff = sig == sig_cond ? 1e-6 : sigsq - sigcsq;

    // This array cumulatively builds the taylor series terms
    // sigdiff^n / n! * df/dsigma (polynomial w alpha)
    double t_array[6];
    t_array[0] = 1.;
    for(i=1;i<6;i++)
        t_array[i] = t_array[i-1] * (-sigdiff) / i * (alpha-i+1) * sigsq_inv;

    //Sum small to large
    double result = 0.;
    for(i=5;i>=0;i--){
        result += t_array[i];
    }

    double prefactor_1 = sqrt(a)*del;
    double prefactor_2 = beta*pow(sigsq_inv*(a*del*del),-alpha);

    result = prefactor_1*(1 + prefactor_2*result);
    *zeroth_order = prefactor_1*(1+prefactor_2); //0th order term gives the barrier for efficiency
    return result;
}

//CMF Corresponding to the Sheth Mo Tormen HMF (Sheth+ 2002)
double dNdM_conditional_ST(double growthf, double lnM, double delta_cond, double sigma_cond){
    double sigma1, dsigmasqdm, Barrier, factor, sigdiff_inv, result;
    double delta_0 = delta_cond / growthf;
    sigma1 = EvaluateSigma(lnM);
    dsigmasqdm = EvaluatedSigmasqdm(lnM);
    if(sigma1 < sigma_cond) return 0.;

    factor = st_taylor_factor(sigma1,sigma_cond,growthf,&Barrier) - delta_0;

    sigdiff_inv = sigma1 == sigma_cond ? 1e6 : 1/(sigma1*sigma1 - sigma_cond*sigma_cond);

    result = -dsigmasqdm*factor*pow(sigdiff_inv,1.5)*exp(-(Barrier - delta_0)*(Barrier - delta_0)*0.5*(sigdiff_inv))/sqrt(2.*PI);
    return result;
}

/*
 FUNCTION dNdM_st(z, M)
 Computes the Press_schechter mass function with Sheth-Torman correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses). Moving barrier B(z,sigma) and sharp-k window functino

 Uses interpolated sigma and dsigmadm to be computed faster. Necessary for mass-dependent ionising efficiencies.

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Sheth, Mo, Torman 2001
 */
double dNdlnM_st(double growthf, double lnM){
    double sigma, dsigmadm, nuhat;

    float MassBinLow;
    int MassBin;

    sigma = EvaluateSigma(lnM);
    dsigmadm = EvaluatedSigmasqdm(lnM);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    nuhat = sqrt(SHETH_a) * Deltac / sigma;

    return (-(cosmo_params_global->OMm)*RHOcrit) * (dsigmadm/sigma) * sqrt(2./PI)*SHETH_A * (1+ pow(nuhat, -2*SHETH_p)) * nuhat * pow(E, -nuhat*nuhat/2.0);
}

//Conditional Extended Press-Schechter Mass function, with constant barrier delta=1.682 and sharp-k window function
double dNdM_conditional_EPS(double growthf, double lnM, double delta_cond, double sigma_cond){
    double sigma1, dsigmasqdm, sigdiff_inv, del;

    sigma1 = EvaluateSigma(lnM);
    dsigmasqdm = EvaluatedSigmasqdm(lnM);

    //limit setting
    if(sigma1 < sigma_cond) return 0.;
    sigdiff_inv = sigma1 == sigma_cond ? 1e6 : 1/(sigma1*sigma1 - sigma_cond*sigma_cond);
    del = (Deltac - delta_cond)/growthf;

    return -del*dsigmasqdm*pow(sigdiff_inv, 1.5)*exp(-del*del*0.5*sigdiff_inv)/sqrt(2.*PI);
}

/*
 FUNCTION dNdM(growthf, M)
 Computes the Press_schechter mass function at
 redshift z (using the growth factor), and dark matter halo mass M (in solar masses).

 Uses interpolated sigma and dsigmadm to be computed faster. Necessary for mass-dependent ionising efficiencies.

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Padmanabhan, pg. 214
 */
double dNdlnM_PS(double growthf, double lnM){
    double sigma, dsigmadm;

    sigma = EvaluateSigma(lnM);
    dsigmadm = EvaluatedSigmasqdm(lnM);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));
    return (-(cosmo_params_global->OMm)*RHOcrit) * sqrt(2/PI) * (Deltac/(sigma*sigma)) * dsigmadm * exp(-(Deltac*Deltac)/(2*sigma*sigma));
}

//The below mass functions do not have a CMF given
/*
 FUNCTION dNdM_WatsonFOF(z, M)
 Computes the Press_schechter mass function with Warren et al. 2011 correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).

 The Universial FOF function (Eq. 12) of Watson et al. 2013

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Watson et al. 2013
 */
double dNdlnM_WatsonFOF(double growthf, double lnM){

    double sigma, dsigmadm, f_sigma;

    sigma = EvaluateSigma(lnM);
    dsigmadm = EvaluatedSigmasqdm(lnM);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    f_sigma = Watson_A * ( pow( Watson_beta/sigma, Watson_alpha) + 1. ) * exp( - Watson_gamma/(sigma*sigma) );

    return (-(cosmo_params_global->OMm)*RHOcrit) * (dsigmadm/sigma) * f_sigma;
}

/*
 FUNCTION dNdM_WatsonFOF_z(z, M)
 Computes the Press_schechter mass function with Warren et al. 2011 correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).

 The Universial FOF function, with redshift evolution (Eq. 12 - 15) of Watson et al. 2013.

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Watson et al. 2013
 */
double dNdlnM_WatsonFOF_z(double z, double growthf, double lnM){
    double sigma, dsigmadm, A_z, alpha_z, beta_z, Omega_m_z, f_sigma;

    sigma = EvaluateSigma(lnM);
    dsigmadm = EvaluatedSigmasqdm(lnM);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    Omega_m_z = (cosmo_params_global->OMm)*pow(1.+z,3.) / ( (cosmo_params_global->OMl) + (cosmo_params_global->OMm)*pow(1.+z,3.) + (global_params.OMr)*pow(1.+z,4.) );

    A_z = Omega_m_z * ( Watson_A_z_1 * pow(1. + z, Watson_A_z_2 ) + Watson_A_z_3 );
    alpha_z = Omega_m_z * ( Watson_alpha_z_1 * pow(1.+z, Watson_alpha_z_2 ) + Watson_alpha_z_3 );
    beta_z = Omega_m_z * ( Watson_beta_z_1 * pow(1.+z, Watson_beta_z_2 ) + Watson_beta_z_3 );

    f_sigma = A_z * ( pow(beta_z/sigma, alpha_z) + 1. ) * exp( - Watson_gamma_z/(sigma*sigma) );

    return (-(cosmo_params_global->OMm)*RHOcrit) * (dsigmadm/sigma) * f_sigma;
}


/* returns the "effective Jeans mass" in Msun
 corresponding to the gas analog of WDM ; eq. 10 in Barkana+ 2001 */
double M_J_WDM(){
    double z_eq, fudge=60;
    if (!(global_params.P_CUTOFF))
        return 0;
    z_eq = 3600*(cosmo_params_global->OMm-cosmo_params_global->OMb)*cosmo_params_global->hlittle*cosmo_params_global->hlittle/0.15;
    return fudge*3.06e8 * (1.5/global_params.g_x) * sqrt((cosmo_params_global->OMm-cosmo_params_global->OMb)*cosmo_params_global->hlittle*cosmo_params_global->hlittle/0.15) * pow(global_params.M_WDM, -4) * pow(z_eq/3000.0, 1.5);
}

/* redshift derivative of the growth function at z */
double ddicke_dz(double z){
    float dz = 1e-10;
    double omegaM_z, ddickdz, dick_0, x, x_0, domegaMdz;

    return (dicke(z+dz)-dicke(z))/dz;
}


/* R in Mpc, M in Msun */
double MtoR(double M){

    // set R according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (global_params.FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return pow(3*M/(4*PI*cosmo_params_global->OMm*RHOcrit), 1.0/3.0);
    else if (global_params.FILTER == 1) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow( M/(pow(2*PI, 1.5) * cosmo_params_global->OMm * RHOcrit), 1.0/3.0 );
    else // filter not defined
        LOG_ERROR("No such filter = %i. Results are bogus.", global_params.FILTER);
    Throw(alueError);
}

/* R in Mpc, M in Msun */
double RtoM(double R){
    // set M according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (global_params.FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return (4.0/3.0)*PI*pow(R,3)*(cosmo_params_global->OMm*RHOcrit);
    else if (global_params.FILTER == 1) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow(2*PI, 1.5) * cosmo_params_global->OMm*RHOcrit * pow(R, 3);
    else // filter not defined
        LOG_ERROR("No such filter = %i. Results are bogus.", global_params.FILTER);
    Throw ValueError;
}

/*
 T in K, M in Msun, mu is mean molecular weight
 from Barkana & Loeb 2001

 SUPRESS = 0 for no radiation field supression;
 SUPRESS = 1 for supression (step function at z=z_ss, at v=v_zz)
 */
double TtoM(double z, double T, double mu){
    return 7030.97 / (cosmo_params_global->hlittle) * sqrt( omega_mz(z) / (cosmo_params_global->OMm*Deltac_nonlinear(z))) *
    pow( T/(mu * (1+z)), 1.5 );
    /*  if (!SUPRESS || (z >= z_re) ) // pre-reionization or don't worry about supression
     return 7030.97 / hlittle * sqrt( omega_mz(z) / (OMm*Deltac_nonlinear(z)) ) *
     pow( T/(mu * (1+z)), 1.5 );

     if (z >= z_ss) // self-shielding dominates, use T = 1e4 K
     return 7030.97 / hlittle * sqrt( omega_mz(z) / (OMm*Deltac_nonlinear(z)) ) *
     pow( 1.0e4 /(mu * (1+z)), 1.5 );

     // optically thin
     return 7030.97 / hlittle * sqrt( omega_mz(z) / (OMm*Deltac_nonlinear(z)) ) *
     pow( VcirtoT(v_ss, mu) /(mu * (1+z)), 1.5 );
     */
}

/* Physical (non-linear) overdensity at virialization (relative to critical density)
 i.e. answer is rho / rho_crit
 In Einstein de sitter model = 178
 (fitting formula from Bryan & Norman 1998) */
double Deltac_nonlinear(float z){
    double d;
    d = omega_mz(z) - 1.0;
    return 18*PI*PI + 82*d - 39*d*d;
}

/* Omega matter at redshift z */
double omega_mz(float z){
    return cosmo_params_global->OMm*pow(1+z,3) / (cosmo_params_global->OMm*pow(1+z,3) + cosmo_params_global->OMl + global_params.OMr*pow(1+z,4) + global_params.OMk*pow(1+z, 2));
}


/*
 FUNCTION dicke(z)
 Computes the dicke growth function at redshift z, i.e. the z dependance part of sigma

 References: Peebles, "Large-Scale...", pg.53 (eq. 11.16). Includes omega<=1
 Nonzero Lambda case from Liddle et al, astro-ph/9512102, eqs. 6-8.
 and quintessence case from Wang et al, astro-ph/9804015

 Normalized to dicke(z=0)=1
 */
double dicke(double z){
    double omegaM_z, dick_z, dick_0, x, x_0;
    double tiny = 1e-4;

    if (fabs(cosmo_params_global->OMm-1.0) < tiny){ //OMm = 1 (Einstein de-Sitter)
        return 1.0/(1.0+z);
    }
    else if ( (cosmo_params_global->OMl > (-tiny)) && (fabs(cosmo_params_global->OMl+cosmo_params_global->OMm+global_params.OMr-1.0) < 0.01) && (fabs(global_params.wl+1.0) < tiny) ){
        //this is a flat, cosmological CONSTANT universe, with only lambda, matter and radiation
        //it is taken from liddle et al.
        omegaM_z = cosmo_params_global->OMm*pow(1+z,3) / ( cosmo_params_global->OMl + cosmo_params_global->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4) );
        dick_z = 2.5*omegaM_z / ( 1.0/70.0 + omegaM_z*(209-omegaM_z)/140.0 + pow(omegaM_z, 4.0/7.0) );
        dick_0 = 2.5*cosmo_params_global->OMm / ( 1.0/70.0 + cosmo_params_global->OMm*(209-cosmo_params_global->OMm)/140.0 + pow(cosmo_params_global->OMm, 4.0/7.0) );
        return dick_z / (dick_0 * (1.0+z));
    }
    else if ( (global_params.OMtot < (1+tiny)) && (fabs(cosmo_params_global->OMl) < tiny) ){ //open, zero lambda case (peebles, pg. 53)
        x_0 = 1.0/(cosmo_params_global->OMm+0.0) - 1.0;
        dick_0 = 1 + 3.0/x_0 + 3*log(sqrt(1+x_0)-sqrt(x_0))*sqrt(1+x_0)/pow(x_0,1.5);
        x = fabs(1.0/(cosmo_params_global->OMm+0.0) - 1.0) / (1+z);
        dick_z = 1 + 3.0/x + 3*log(sqrt(1+x)-sqrt(x))*sqrt(1+x)/pow(x,1.5);
        return dick_z/dick_0;
    }
    else if ( (cosmo_params_global->OMl > (-tiny)) && (fabs(global_params.OMtot-1.0) < tiny) && (fabs(global_params.wl+1) > tiny) ){
        LOG_WARNING("IN WANG.");
        Throw ValueError;
    }

    LOG_ERROR("No growth function!");
    Throw ValueError;
}

/* function DTDZ returns the value of dt/dz at the redshift parameter z. */
double dtdz(float z){
    double x, dxdz, const1, denom, numer;
    x = sqrt( cosmo_params_global->OMl/cosmo_params_global->OMm ) * pow(1+z, -3.0/2.0);
    dxdz = sqrt( cosmo_params_global->OMl/cosmo_params_global->OMm ) * pow(1+z, -5.0/2.0) * (-3.0/2.0);
    const1 = 2 * sqrt( 1 + cosmo_params_global->OMm/cosmo_params_global->OMl ) / (3.0 * Ho) ;

    numer = dxdz * (1 + x*pow( pow(x,2) + 1, -0.5));
    denom = x + sqrt(pow(x,2) + 1);
    return (const1 * numer / denom);
}

/* Time derivative of the growth function at z */
double ddickedt(double z){
    float dz = 1e-10;
    double omegaM_z, ddickdz, dick_0, x, x_0, domegaMdz;
    double tiny = 1e-4;

    return (dicke(z+dz)-dicke(z))/dz/dtdz(z); // lazy non-analytic form getting

    if (fabs(cosmo_params_global->OMm-1.0) < tiny){ //OMm = 1 (Einstein de-Sitter)
        return -pow(1+z,-2)/dtdz(z);
    }
    else if ( (cosmo_params_global->OMl > (-tiny)) && (fabs(cosmo_params_global->OMl+cosmo_params_global->OMm+global_params.OMr-1.0) < 0.01) && (fabs(global_params.wl+1.0) < tiny) ){
        //this is a flat, cosmological CONSTANT universe, with only lambda, matter and radiation
        //it is taken from liddle et al.
        omegaM_z = cosmo_params_global->OMm*pow(1+z,3) / ( cosmo_params_global->OMl + cosmo_params_global->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4) );
        domegaMdz = omegaM_z*3/(1+z) - cosmo_params_global->OMm*pow(1+z,3)*pow(cosmo_params_global->OMl + cosmo_params_global->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4), -2) * (3*cosmo_params_global->OMm*(1+z)*(1+z) + 4*global_params.OMr*pow(1+z,3));
        dick_0 = cosmo_params_global->OMm / ( 1.0/70.0 + cosmo_params_global->OMm*(209-cosmo_params_global->OMm)/140.0 + pow(cosmo_params_global->OMm, 4.0/7.0) );

        ddickdz = (domegaMdz/(1+z)) * (1.0/70.0*pow(omegaM_z,-2) + 1.0/140.0 + 3.0/7.0*pow(omegaM_z, -10.0/3.0)) * pow(1.0/70.0/omegaM_z + (209.0-omegaM_z)/140.0 + pow(omegaM_z, -3.0/7.0) , -2);
        ddickdz -= pow(1+z,-2)/(1.0/70.0/omegaM_z + (209.0-omegaM_z)/140.0 + pow(omegaM_z, -3.0/7.0));

        return ddickdz / dick_0 / dtdz(z);
    }

    LOG_ERROR("No growth function!");
    Throw ValueError;
}

/* returns the hubble "constant" (in 1/sec) at z */
double hubble(float z){
    return Ho*sqrt(cosmo_params_global->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4) + cosmo_params_global->OMl);
}


/* returns hubble time (in sec), t_h = 1/H */
double t_hubble(float z){
    return 1.0/hubble(z);
}

/* comoving distance (in cm) per unit redshift */
double drdz(float z){
    return (1.0+z)*C*dtdz(z);
}