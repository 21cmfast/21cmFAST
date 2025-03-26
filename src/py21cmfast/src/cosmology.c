/* This file contatins functions regarding the matter power-sepctrum and cosmology */
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>

#include "Constants.h"
#include "cexcept.h"
#include "exceptions.h"
#include "filtering.h"
#include "logger.h"
#include "InputParameters.h"

#include "cosmology.h"

struct CosmoConstants {
    //calculated constants in TFset_parameters() for the EH transfer function
    double sound_horizon;
    double alpha_nu;
    double beta_c;

    //convenience constants set in init_ps()
    double omhh;
    double f_nu;
    double f_baryon;
    double theta_cmb;

    //Normalisation of the power-spectrum integral for sigma_8
    double sigma_norm;

};

static struct CosmoConstants cosmo_consts;

//NOTE: All Transfer functions take k in Mpc^-1, and convert to hMpc^-1 internally
// FUNCTION TFmdm is the power spectrum transfer function from Eisenstein & Hu ApJ, 1999, 511, 5
double transfer_function_EH(double k){
    double q, gamma_eff, q_eff, TF_m, q_nu;
    double sound_horizon = cosmo_consts.sound_horizon;
    double alpha_nu = cosmo_consts.alpha_nu;
    double f_nu = cosmo_consts.f_nu;

    q = k*pow(cosmo_consts.theta_cmb,2)/cosmo_consts.omhh;
    gamma_eff=sqrt(alpha_nu) + (1.0-sqrt(alpha_nu))/(1.0+pow(0.43*k*sound_horizon, 4));
    q_eff = q/gamma_eff;
    TF_m= log(E+1.84*cosmo_consts.beta_c*sqrt(alpha_nu)*q_eff);
    TF_m /= TF_m + pow(q_eff,2) * (14.4 + 325.0/(1.0+60.5*pow(q_eff,1.11)));
    q_nu = 3.92*q/sqrt(f_nu/N_nu);
    TF_m *= 1.0 + (1.2*pow(f_nu,0.64)*pow(N_nu,0.3+0.6*f_nu)) / \
        (pow(q_nu,-1.6)+pow(q_nu,0.8));

    //WDM Model previously only included with EH
    //if (P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);

    return TF_m;
}

//Bardeen et al 1986 ApJ, 304, 15
// with baryon correction from Sugiyama 1995 ApJS 100, 281
double transfer_function_BBKS(double k){
    double gamma,q;
    gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * \
        exp( -(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
    q = k / (cosmo_params_global->hlittle*gamma);
    return (log(1.0+2.34*q)/(2.34*q)) * pow( 1.0+3.89*q + pow(16.1*q, 2) + \
        pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
}

//Efstathiou et al 1992 MNRAS 258, 1
//NOTE: different from Bond & Efstathiou 1984
// no baryon correction
double transfer_function_Efstathiou(double k){
    double gamma,aa,bb,cc,nu;
    gamma = cosmo_params_global->OMm*cosmo_params_global->hlittle*cosmo_params_global->hlittle;
    aa = 6.4/gamma;
    bb = 3.0/gamma;
    cc = 1.7/gamma;
    nu = 1.13;
    return pow(1 + pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), nu), -1./nu );
}

//Peebles 1980 p.626
// with baryon correction from Sugiyama 1995 ApJS 100, 281
double transfer_function_Peebles(double k){
    double gamma,aa,bb;
    gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * \
        exp(-(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
    aa = 8.0 / (cosmo_params_global->hlittle*gamma);
    bb = 4.7 / pow(cosmo_params_global->hlittle*gamma, 2); //no ^2 in dsigmasq, but that makes no sense
    return 1 + aa*k + bb*k*k;
}

//Actually from Davies, Efstathiou, Frenk & White 1985 ApJ 292, 371
// with baryon correction from Sugiyama 1995 ApJS 100, 281
double transfer_function_White(double k){
    double gamma,aa,bb,cc;
    gamma = cosmo_params_global->OMm * cosmo_params_global->hlittle * cosmo_params_global->hlittle \
        * exp(-(cosmo_params_global->OMb) - (cosmo_params_global->OMb/cosmo_params_global->OMm));
    aa = 1.7/(gamma);
    bb = 9.0/pow(gamma, 1.5);
    cc = 1.0/pow(gamma, 2);
    return 139.284 / (1 + aa*k + bb*pow(k, 1.5) + cc*k*k);
}

/*
  this function reads the z=0 matter (CDM+baryons)  and relative velocity transfer functions from CLASS (from a file)
  flag_int = 0 to initialize interpolator, flag_int = -1 to free memory, flag_int = else to interpolate.
  flag_dv = 0 to output density, flag_dv = 1 to output velocity.
  similar to built-in function "double T_RECFAST(float z, int flag)"
*/
double transfer_function_CLASS(double k, int flag_int, int flag_dv)
{
    static double kclass[CLASS_LENGTH], Tmclass[CLASS_LENGTH], Tvclass_vcb[CLASS_LENGTH];
    static gsl_interp_accel *acc_density, *acc_vcb;
    static gsl_spline *spline_density, *spline_vcb;
    float currk, currTm, currTv;
    double ans;
    int i;
    int gsl_status;
    FILE *F;

    static bool warning_printed;
    static double eh_ratio_at_kmax;

    char filename[500];
    sprintf(filename,"%s/%s",config_settings.external_table_path,CLASS_FILENAME);

    if (flag_int == 0) {  // Initialize vectors and read file
        if (!(F = fopen(filename, "r"))) {
            LOG_ERROR("Unable to open file: %s for reading.", filename);
            Throw(IOError);
        }
        warning_printed = false;

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
        CATCH_GSL_ERROR(gsl_status);

        LOG_SUPER_DEBUG("Generated CLASS Density Spline.");

        //Set up spline table for velocities
        acc_vcb   = gsl_interp_accel_alloc ();
        spline_vcb  = gsl_spline_alloc (gsl_interp_cspline, CLASS_LENGTH);
        gsl_status = gsl_spline_init(spline_vcb, kclass, Tvclass_vcb, CLASS_LENGTH);
        CATCH_GSL_ERROR(gsl_status);

        LOG_SUPER_DEBUG("Generated CLASS velocity Spline.");

        eh_ratio_at_kmax = Tmclass[CLASS_LENGTH-1]/kclass[CLASS_LENGTH-1]/kclass[CLASS_LENGTH-1] / transfer_function_EH(kclass[CLASS_LENGTH-1]);
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
        if(!warning_printed){
            LOG_WARNING("Called transfer_function_CLASS with k=%f, larger than kmax! performing linear extrapolation with Eisenstein & Hu", k);
            warning_printed = true;
        }
        if(flag_dv == 0){ // output is density
            return eh_ratio_at_kmax * transfer_function_EH(k);
        }
        else if(flag_dv == 1){ // output is rel velocity, do a log-log linear extrapolation
            return exp(log(Tvclass_vcb[CLASS_LENGTH-1]) + (log(Tvclass_vcb[CLASS_LENGTH-1]) - log(Tvclass_vcb[CLASS_LENGTH-2])) \
                / (log(kclass[CLASS_LENGTH-1]) - log(kclass[CLASS_LENGTH-2])) * (log(k) - log(kclass[CLASS_LENGTH-1]))) \
                / k / k;
        }    //we just set it to the last value, since sometimes it wants large k for R<<cell_size, which does not matter much.
        else{
            LOG_ERROR("Invalid flag_dv %d passed to transfer_function_CLASS",flag_dv);
            Throw(ValueError);
        }
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

double transfer_function(double k){
    switch(user_params_global->POWER_SPECTRUM){
        case 0:
            return transfer_function_EH(k);
        case 1:
            return transfer_function_BBKS(k);
        case 2:
            return transfer_function_Efstathiou(k);
        case 3:
            return transfer_function_Peebles(k);
        case 4:
            return transfer_function_White(k);
        case 5:
            return transfer_function_CLASS(k, 1, 0);
        default:
            LOG_ERROR("No such power spectrum defined: %i", user_params_global->POWER_SPECTRUM);
            Throw(ValueError);
    }
}


double power_in_k_integrand(double k){
    double T, p;
    T = transfer_function(k);
    p = T * T * pow(k, cosmo_params_global->POWER_INDEX);

    //TODO: figure this out, it was always a comment?
    //if(user_params_global->POWER_SPECTRUM == 0)
    //  p = pow(k, POWER_INDEX - 0.05*log(k/0.05)) * T * T; //running, alpha=0.05

    //NOTE: USE_RELATIVE_VELOCITIES is only allowed if using CLASS
    if(user_params_global->POWER_SPECTRUM == 5 && \
        user_params_global->USE_RELATIVE_VELOCITIES){
        //jbm:Add average relvel suppression
        p *= 1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/ \
            (2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM));
    }

    return p;
}

//we need a version with the prefactors for output
double power_in_k(double k){
    return TWOPI*PI*cosmo_consts.sigma_norm*cosmo_consts.sigma_norm*power_in_k_integrand(k);
}

/*
  Returns the value of the linear power spectrum of the DM-b relative velocity
  at kinematic decoupling (which we set at zkin=1010)
*/
double power_in_vcb(double k){
    double p, T;

    //only works if using CLASS
    if (user_params_global->POWER_SPECTRUM == 5){ // CLASS
        T = transfer_function_CLASS(k, 1, 1); //read from CLASS file. flag_int=1 since we have initialized before, flag_vcb=1 for velocity
        p = pow(k, cosmo_params_global->POWER_INDEX) * T * T;
    }
    else{
        LOG_ERROR("Cannot get P_cb unless using CLASS: %i\n Set USE_RELATIVE_VELOCITIES 0 or use CLASS.\n", user_params_global->POWER_SPECTRUM);
        Throw(ValueError);
    }

    return p*TWOPI*PI*cosmo_consts.sigma_norm*cosmo_consts.sigma_norm;
}

// FUNCTION sigma_z0(M)
// Returns the standard deviation of the normalized, density excess (delta(x)) field,
// smoothed on the comoving scale of M (see filter definitions for M<->R conversion).
// The sigma is evaluated at z=0, with the time evolution contained in the dicke(z) factor,
// i.e. sigma(M,z) = sigma_z0(m) * dicke(z)
// normalized so that sigma_z0(M->8/h Mpc) = SIGMA_8 in the UserParams

// NOTE: volume is normalized to = 1, so this is equvalent to the mass standard deviation

// M is in solar masses
struct SigmaIntegralParams {
    double radius;
    int filter_type;
};

// References: Padmanabhan, pg. 210, eq. 5.107
double dsigma_dk(double k, void *params){
    double p, w, kR;

    struct SigmaIntegralParams *pars = (struct SigmaIntegralParams *)params;
    double Radius = pars->radius;
    int filter = pars->filter_type;
    p = power_in_k_integrand(k);

    kR = k*Radius;
    w = filter_function(kR,filter);
    return k*k*p*w*w;
}
double sigma_z0(double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

    double Radius = MtoR(M);

    lower_limit = 1.0e-99/Radius; //kstart
    upper_limit = 350.0/Radius;// kend;

    struct SigmaIntegralParams sigma_params = {
        .radius=Radius,
        .filter_type=user_params_global->FILTER
    };
    F.function = &dsigma_dk;
    F.params = &sigma_params;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: M=%e",M);
        CATCH_GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return cosmo_consts.sigma_norm * sqrt(result);
}


/*
 FUNCTION dsigmasqdm_z0(M)
 returns  d/dm (sigma^2) (see function sigma), in units of Msun^-1
 */
double dsigmasq_dm(double k, void *params){
    struct SigmaIntegralParams *pars = (struct SigmaIntegralParams *)params;
    double Radius = pars->radius;
    int filter = pars->filter_type;
    double p = power_in_k_integrand(k);

    // now get the value of the window function
    double dw2dm = dwdm_filter(k, Radius, filter);

    return k*k*p*dw2dm;
}
double dsigmasqdm_z0(double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    double Radius = MtoR(M);

    lower_limit = 1.0e-99/Radius; //kstart
    upper_limit = 350.0/Radius; //kend

    struct SigmaIntegralParams sigma_params = {
        .radius=Radius,
        .filter_type=user_params_global->FILTER
    };
    F.function = &dsigmasq_dm;
    F.params = &sigma_params;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: M=%e",M);
        CATCH_GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return cosmo_consts.sigma_norm * cosmo_consts.sigma_norm * result;
}

//Sets constants used in the EH transfer function
void TFset_parameters(){
    //intermediates for setting the transfer function parameters
    double z_drag, z_equality, R_drag, R_equality, p_c, p_cb, f_c, f_cb, f_nub, k_equality, y_d;

    double f_nu = cosmo_consts.f_nu;
    double f_b = cosmo_consts.f_baryon;
    double omhh = cosmo_consts.omhh;
    double obhh = cosmo_params_global->OMb*cosmo_params_global->hlittle*cosmo_params_global->hlittle;
    double theta_cmb = cosmo_consts.theta_cmb;
    double alpha_nu;

    LOG_DEBUG("Setting Transfer Function parameters.");

    z_equality = 25000*omhh*pow(theta_cmb, -4) - 1.0;
    k_equality = 0.0746*omhh/(theta_cmb*theta_cmb);

    z_drag = 0.313*pow(omhh,-0.419) * (1 + 0.607*pow(omhh, 0.674));
    z_drag = 1 + z_drag*pow(obhh, 0.238*pow(omhh, 0.223));
    z_drag *= 1291 * pow(omhh, 0.251) / (1 + 0.659*pow(omhh, 0.828));

    y_d = (1 + z_equality) / (1.0 + z_drag);

    R_drag = 31.5 * obhh * pow(theta_cmb, -4) * 1000 / (1.0 + z_drag);
    R_equality = 31.5 * obhh * pow(theta_cmb, -4) * 1000 / (1.0 + z_equality);

    cosmo_consts.sound_horizon = 2.0/3.0/k_equality * sqrt(6.0/R_equality) *
    log( (sqrt(1+R_drag) + sqrt(R_drag+R_equality)) / (1.0 + sqrt(R_equality)) );

    p_c = -(5 - sqrt(1 + 24*(1 - f_nu-f_b)))/4.0;
    p_cb = -(5 - sqrt(1 + 24*(1 - f_nu)))/4.0;
    f_c = 1 - f_nu - f_b;
    f_cb = 1 - f_nu;
    f_nub = f_nu+f_b;

    alpha_nu = (f_c/f_cb) * (2*(p_c+p_cb)+5)/(4*p_cb+5.0);
    alpha_nu *= 1 - 0.553*f_nub+0.126*pow(f_nub,3);
    alpha_nu /= 1-0.193*sqrt(f_nu)+0.169*f_nu;
    alpha_nu *= pow(1+y_d, p_c-p_cb);
    alpha_nu *= 1+ (p_cb-p_c)/2.0 * (1.0+1.0/(4.0*p_c+3.0)/(4.0*p_cb+7.0))/(1.0+y_d);
    cosmo_consts.alpha_nu = alpha_nu;
    cosmo_consts.beta_c = 1.0/(1.0-0.949*f_nub);
}

void init_ps(){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

    // Set cuttoff scale for WDM (eq. 4 in Barkana et al. 2001) in comoving Mpc
    // R_CUTOFF = 0.201*pow((cosmo_params_global->OMm-cosmo_params_global->OMb)*cosmo_params_global->hlittle*cosmo_params_global->hlittle/0.15, 0.15)*pow(.g_x/1.5, -0.29)*pow(.M_WDM, -1.15);

    cosmo_consts.omhh = cosmo_params_global->OMm*cosmo_params_global->hlittle*cosmo_params_global->hlittle;
    cosmo_consts.theta_cmb = T_cmb / 2.7;

    cosmo_consts.f_nu = fmax(cosmo_params_global->OMn/cosmo_params_global->OMm,1e-10);
    cosmo_consts.f_baryon = fmax(cosmo_params_global->OMb/cosmo_params_global->OMm,1e-10);

    TFset_parameters();

    cosmo_consts.sigma_norm = -1;

    //we start the interpolator if using CLASS:
    if (user_params_global->POWER_SPECTRUM == 5){
        LOG_DEBUG("Setting CLASS Transfer Function inits.");
        transfer_function_CLASS(1.0, 0, 0);
    }

    double Radius_8;
    Radius_8 = 8.0/cosmo_params_global->hlittle;

    lower_limit = 1.0e-99/Radius_8; //kstart
    upper_limit = 350.0/Radius_8; //kend

    struct SigmaIntegralParams sigma_params = {
        .radius=Radius_8,
        .filter_type=user_params_global->FILTER
    };
    F.function = &dsigma_dk;
    F.params = &sigma_params;

    LOG_DEBUG("Initializing Power Spectrum with lower_limit=%e, upper_limit=%e, rel_tol=%e, radius_8=%g", lower_limit,upper_limit, rel_tol, Radius_8);
    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        CATCH_GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    LOG_DEBUG("Initialized Power Spectrum.");

    cosmo_consts.sigma_norm = cosmo_params_global->SIGMA_8/sqrt(result); //takes care of volume factor
    return;
}


//function to free arrays related to the power spectrum
void free_ps(){

	//we free the PS interpolator if using CLASS:
	if (user_params_global->POWER_SPECTRUM == 5){
		transfer_function_CLASS(1.0, -1, 0);
	}

  return;
}

/* returns the "effective Jeans mass" in Msun
 corresponding to the gas analog of WDM ; eq. 10 in Barkana+ 2001 */
double M_J_WDM(){
    //these were global params but never really used
    double g_x = 1.5; //degrees of freedom (1.5 for fermions)
    double M_WDM = 2.0; // Mass of particle in keV

    double z_eq, fudge=60;
    z_eq = 3600*(cosmo_params_global->OMm-cosmo_params_global->OMb)*cosmo_params_global->hlittle*cosmo_params_global->hlittle/0.15;
    return fudge*3.06e8 * (1.5/g_x) * sqrt((cosmo_params_global->OMm-cosmo_params_global->OMb)*cosmo_params_global->hlittle*cosmo_params_global->hlittle/0.15) * pow(M_WDM, -4) * pow(z_eq/3000.0, 1.5);
}

/* redshift derivative of the growth function at z */
double ddicke_dz(double z){
    float dz = 1e-10;
    //NOTE: this isnt called a lot, I'm not sure why we don't use a similar method to ddicke_dt
    return (dicke(z+dz)-dicke(z))/dz;
}


/* R in Mpc, M in Msun */
double MtoR(double M){

    // set R according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (user_params_global->FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return pow(3*M/(4*PI*cosmo_params_global->OMm*RHOcrit), 1.0/3.0);
    else if (user_params_global->FILTER == 2) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow( M/(pow(2*PI, 1.5) * cosmo_params_global->OMm * RHOcrit), 1.0/3.0 );
    else // filter not defined
        LOG_ERROR("No such filter = %i. Results are bogus.", user_params_global->FILTER);
    Throw(ValueError);
}

/* R in Mpc, M in Msun */
double RtoM(double R){
    // set M according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (user_params_global->FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return (4.0/3.0)*PI*pow(R,3)*(cosmo_params_global->OMm*RHOcrit);
    else if (user_params_global->FILTER == 2) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow(2*PI, 1.5) * cosmo_params_global->OMm*RHOcrit * pow(R, 3);
    else // filter not defined
        LOG_ERROR("No such filter = %i. Results are bogus.", user_params_global->FILTER);
    Throw(ValueError);
}

/* Omega matter at redshift z */
double omega_mz(float z){
    return cosmo_params_global->OMm*pow(1+z,3) / (cosmo_params_global->OMm*pow(1+z,3) + cosmo_params_global->OMl + cosmo_params_global->OMr*pow(1+z,4) + cosmo_params_global->OMk*pow(1+z, 2));
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
    else if ( (cosmo_params_global->OMl > (-tiny)) && (fabs(cosmo_params_global->OMl+cosmo_params_global->OMm+cosmo_params_global->OMr-1.0) < 0.01) && (fabs(cosmo_params_global->wl+1.0) < tiny) ){
        //this is a flat, cosmological CONSTANT universe, with only lambda, matter and radiation
        //it is taken from liddle et al.
        omegaM_z = cosmo_params_global->OMm*pow(1+z,3) / ( cosmo_params_global->OMl + cosmo_params_global->OMm*pow(1+z,3) + cosmo_params_global->OMr*pow(1+z,4) );
        dick_z = 2.5*omegaM_z / ( 1.0/70.0 + omegaM_z*(209-omegaM_z)/140.0 + pow(omegaM_z, 4.0/7.0) );
        dick_0 = 2.5*cosmo_params_global->OMm / ( 1.0/70.0 + cosmo_params_global->OMm*(209-cosmo_params_global->OMm)/140.0 + pow(cosmo_params_global->OMm, 4.0/7.0) );
        return dick_z / (dick_0 * (1.0+z));
    }
    else if ( (cosmo_params_global->OMtot < (1+tiny)) && (fabs(cosmo_params_global->OMl) < tiny) ){ //open, zero lambda case (peebles, pg. 53)
        x_0 = 1.0/(cosmo_params_global->OMm+0.0) - 1.0;
        dick_0 = 1 + 3.0/x_0 + 3*log(sqrt(1+x_0)-sqrt(x_0))*sqrt(1+x_0)/pow(x_0,1.5);
        x = fabs(1.0/(cosmo_params_global->OMm+0.0) - 1.0) / (1+z);
        dick_z = 1 + 3.0/x + 3*log(sqrt(1+x)-sqrt(x))*sqrt(1+x)/pow(x,1.5);
        return dick_z/dick_0;
    }
    else if ( (cosmo_params_global->OMl > (-tiny)) && (fabs(cosmo_params_global->OMtot-1.0) < tiny) && (fabs(cosmo_params_global->wl+1) > tiny) ){
        LOG_WARNING("IN WANG.");
        Throw(ValueError);
    }

    LOG_ERROR("No growth function!");
    Throw(ValueError);
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
    double omegaM_z, ddickdz, dick_0, domegaMdz;
    double tiny = 1e-4;

    //TODO: called once per snapshot in PerturbedField, find out why we don't use analytic
    return (dicke(z+dz)-dicke(z))/dz/dtdz(z); // lazy non-analytic form getting

    if (fabs(cosmo_params_global->OMm-1.0) < tiny){ //OMm = 1 (Einstein de-Sitter)
        return -pow(1+z,-2)/dtdz(z);
    }
    else if ( (cosmo_params_global->OMl > (-tiny)) && (fabs(cosmo_params_global->OMl+cosmo_params_global->OMm+cosmo_params_global->OMr-1.0) < 0.01) && (fabs(cosmo_params_global->wl+1.0) < tiny) ){
        //this is a flat, cosmological CONSTANT universe, with only lambda, matter and radiation
        //it is taken from liddle et al.
        omegaM_z = cosmo_params_global->OMm*pow(1+z,3) / ( cosmo_params_global->OMl + cosmo_params_global->OMm*pow(1+z,3) + cosmo_params_global->OMr*pow(1+z,4) );
        domegaMdz = omegaM_z*3/(1+z) - cosmo_params_global->OMm*pow(1+z,3)*pow(cosmo_params_global->OMl + cosmo_params_global->OMm*pow(1+z,3) + cosmo_params_global->OMr*pow(1+z,4), -2) * (3*cosmo_params_global->OMm*(1+z)*(1+z) + 4*cosmo_params_global->OMr*pow(1+z,3));
        dick_0 = cosmo_params_global->OMm / ( 1.0/70.0 + cosmo_params_global->OMm*(209-cosmo_params_global->OMm)/140.0 + pow(cosmo_params_global->OMm, 4.0/7.0) );

        ddickdz = (domegaMdz/(1+z)) * (1.0/70.0*pow(omegaM_z,-2) + 1.0/140.0 + 3.0/7.0*pow(omegaM_z, -10.0/3.0)) * pow(1.0/70.0/omegaM_z + (209.0-omegaM_z)/140.0 + pow(omegaM_z, -3.0/7.0) , -2);
        ddickdz -= pow(1+z,-2)/(1.0/70.0/omegaM_z + (209.0-omegaM_z)/140.0 + pow(omegaM_z, -3.0/7.0));

        return ddickdz / dick_0 / dtdz(z);
    }

    LOG_ERROR("No growth function!");
    Throw(ValueError);
}

/* returns the hubble "constant" (in 1/sec) at z */
double hubble(float z){
    return Ho*sqrt(cosmo_params_global->OMm*pow(1+z,3) + cosmo_params_global->OMr*pow(1+z,4) + cosmo_params_global->OMl);
}


/* returns hubble time (in sec), t_h = 1/H */
double t_hubble(float z){
    return 1.0/hubble(z);
}

/* comoving distance (in cm) per unit redshift */
double drdz(float z){
    return (1.0+z)*C*dtdz(z);
}
