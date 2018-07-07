//#include "21CMMC.h"

struct CosmoParams *cosmo_params_ps;

double sigma_norm, R, theta_cmb, omhh, z_equality, y_d, sound_horizon, alpha_nu, f_nu, f_baryon, beta_c, d2fact, R_CUTOFF, DEL_CURR, SIG_CURR;

/*****     FUNCTION PROTOTYPES     *****/
double init_ps(); /* initialize global variables, MUST CALL THIS FIRST!!! returns R_CUTOFF */
double sigma_z0(double M); //calculates sigma at z=0 (no dicke)
double power_in_k(double k); /* Returns the value of the linear power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */
double TFmdm(double k); //Eisenstien & Hu power spectrum transfer function
void TFset_parameters();


void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params){
 
    cosmo_params_ps = cosmo_params;
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
    if (global_params.POWER_SPECTRUM == 0){ // Eisenstein & Hu
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else if (global_params.POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        q = k / (cosmo_params_ps->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else if (global_params.POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_ps->hlittle*gamma);
        bb = 3.0/(cosmo_params_ps->hlittle*gamma);
        cc = 1.7/(cosmo_params_ps->hlittle*gamma);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (global_params.POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        aa = 8.0 / (cosmo_params_ps->hlittle*gamma);
        bb = 4.7 / pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (global_params.POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        aa = 1.7/(cosmo_params_ps->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_ps->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * 19400.0 / pow(1 + aa*k + bb*pow(k, 1.5) + cc*k*k, 2);
    }
    else{
        fprintf(stderr, "No such power spectrum defined: %i\nOutput is bogus.\n", global_params.POWER_SPECTRUM);
        p = 0;
    }
    
    
    // now get the value of the window function
    // NOTE: only use top hat for SIGMA8 normalization
    kR = k*R;
    if ( (global_params.FILTER == 0) || (sigma_norm < 0) ){ // top hat
        if ( (kR) < 1.0e-4 ){ w = 1.0;} // w converges to 1 as (kR) -> 0
        else { w = 3.0 * (sin(kR)/pow(kR, 3) - cos(kR)/pow(kR, 2));}
    }
    else if (global_params.FILTER == 1){ // gaussian of width 1/R
        w = pow(E, -kR*kR/2.0);
    }
    else {
        fprintf(stderr, "No such filter: %i\nOutput is bogus.\n", global_params.FILTER);
        w=0;
    }
    
    return k*k*p*w*w;
}
double sigma_z0(double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    double kstart, kend;
    
    R = MtoR(M);
    
    // now lets do the integral for sigma and scale it with sigma_norm
    kstart = 1.0e-99/R;
    kend = 350.0/R;
    lower_limit = kstart;//log(kstart);
    upper_limit = kend;//log(kend);
    
    F.function = &dsigma_dk;
    //  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);
    //    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS41, w, &result, &error);
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS15, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    return sigma_norm * sqrt(result);
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
    
    //   printf("%f  %e  %f  %f  %f  %f\n",omhh,f_nu,f_baryon,N_nu,y_d,alpha_nu);
    // printf("%f  %f  %f  %f\n", beta_c,sound_horizon,theta_cmb,z_equality);
    //printf("%f  %e  %f  %f  %f\n\n",q, k, gamma_eff, q_nu, TF_m);
    
    
    return TF_m;
}


void TFset_parameters(){
    double z_drag, R_drag, R_equality, p_c, p_cb, f_c, f_cb, f_nub, k_equality;
    
    z_equality = 25000*omhh*pow(theta_cmb, -4) - 1.0;
    k_equality = 0.0746*omhh/(theta_cmb*theta_cmb);
    
    z_drag = 0.313*pow(omhh,-0.419) * (1 + 0.607*pow(omhh, 0.674));
    z_drag = 1 + z_drag*pow(cosmo_params_ps->OMb*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle, 0.238*pow(omhh, 0.223));
    z_drag *= 1291 * pow(omhh, 0.251) / (1 + 0.659*pow(omhh, 0.828));
    
    y_d = (1 + z_equality) / (1.0 + z_drag);
    
    R_drag = 31.5 * cosmo_params_ps->OMb*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle * pow(theta_cmb, -4) * 1000 / (1.0 + z_drag);
    R_equality = 31.5 * cosmo_params_ps->OMb*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle * pow(theta_cmb, -4) * 1000 / (1.0 + z_equality);
    
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
    if (global_params.POWER_SPECTRUM == 0){ // Eisenstein & Hu
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
        //p = pow(k, POWER_INDEX - 0.05*log(k/0.05)) * T * T; //running, alpha=0.05
    }
    else if (global_params.POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        q = k / (cosmo_params_ps->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else if (global_params.POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_ps->hlittle*gamma);
        bb = 3.0/(cosmo_params_ps->hlittle*gamma);
        cc = 1.7/(cosmo_params_ps->hlittle*gamma);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (global_params.POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb)/(cosmo_params_ps->OMm));
        aa = 8.0 / (cosmo_params_ps->hlittle*gamma);
        bb = 4.7 / pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (global_params.POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        aa = 1.7/(cosmo_params_ps->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_ps->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * 19400.0 / pow(1 + aa*k + bb*pow(k, 1.5) + cc*k*k, 2);
    }
    else{
        fprintf(stderr, "No such power spectrum defined: %i\nOutput is bogus.\n", global_params.POWER_SPECTRUM);
        p = 0;
    }
    
    
    return p*TWOPI*PI*sigma_norm*sigma_norm;
}


double init_ps(){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    double kstart, kend;
    int i;
    double x;
    
    // Set cuttoff scale for WDM (eq. 4 in Barkana et al. 2001) in comoving Mpc
    R_CUTOFF = 0.201*pow((cosmo_params_ps->OMm-cosmo_params_ps->OMb)*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle/0.15, 0.15)*pow(global_params.g_x/1.5, -0.29)*pow(global_params.M_WDM, -1.15);
    
    //  fprintf(stderr, "For M_DM = %.2e keV, R_CUTOFF is: %.2e comoving Mpc\n", M_WDM, R_CUTOFF);
    //    if (!P_CUTOFF)
    //    fprintf(stderr, "But you have selected CDM, so this is ignored\n");
    
    omhh = cosmo_params_ps->OMm*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle;
    theta_cmb = T_cmb / 2.7;
    
    // Translate Parameters into forms GLOBALVARIABLES form
    f_nu = global_params.OMn/cosmo_params_ps->OMm;
    f_baryon = cosmo_params_ps->OMb/cosmo_params_ps->OMm;
    if (f_nu < TINY) f_nu = 1e-10;
    if (f_baryon < TINY) f_baryon = 1e-10;
    
    TFset_parameters();
    
    sigma_norm = -1;
    
    R = 8.0/cosmo_params_ps->hlittle;
    kstart = 1.0e-99/R;
    kend = 350.0/R;
    lower_limit = kstart;//log(kstart);
    upper_limit = kend;//log(kend);
    
    F.function = &dsigma_dk;
    
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    sigma_norm = cosmo_params_ps->SIGMA_8/sqrt(result); //takes care of volume factor
    
    /* initialize the lookup table for erfc */
    /*
     for (i=0; i<=ERFC_NPTS; i++){
     erfc_params[i] = i*ERFC_PARAM_DELTA;
     log_erfc_table[i] = log(erfcc(erfc_params[i]));
     }
     // Set up spline table
     erfc_acc   = gsl_interp_accel_alloc ();
     erfc_spline  = gsl_spline_alloc (gsl_interp_cspline, ERFC_NPTS);
     gsl_spline_init(erfc_spline, erfc_params, log_erfc_table, ERFC_NPTS);
     */
    
    return R_CUTOFF;
}