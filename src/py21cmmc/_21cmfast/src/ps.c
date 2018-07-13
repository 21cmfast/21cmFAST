//#include "21CMMC.h"

/*** Some usefull math macros ***/
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

static double mnarg1,mnarg2;
#define FMAX(a,b) (mnarg1=(a),mnarg2=(b),(mnarg1) > (mnarg2) ?\
(mnarg1) : (mnarg2))

static double mnarg1,mnarg2;
#define FMIN(a,b) (mnarg1=(a),mnarg2=(b),(mnarg1) < (mnarg2) ?\
(mnarg1) : (mnarg2))

#define ERFC_NPTS (int) 75
#define ERFC_PARAM_DELTA (float) 0.1
static double log_erfc_table[ERFC_NPTS], erfc_params[ERFC_NPTS];
static gsl_interp_accel *erfc_acc;
static gsl_spline *erfc_spline;

#define NGaussLegendre 40  //defines the number of points in the Gauss-Legendre quadrature integration

#define SPLINE_NPTS (int) 250
#define NGLhigh 100
#define NGLlow 100

#define Nhigh 200
#define Nlow 100
#define NMass 200

#define NR_END 1
#define FREE_ARG char*

#define MM 7
#define NSTACK 50

#define EPS2 3.0e-11

static double Fcoll_spline_params[SPLINE_NPTS], log_Fcoll_spline_table[SPLINE_NPTS];
static gsl_interp_accel *Fcoll_spline_acc;
static gsl_spline *Fcoll_spline;

struct parameters_gsl_int_{
    double z_obs;
    double Mval;
    double M_Feed;
    double alpha_pl;
    double del_traj_1;
    double del_traj_2;
};

struct parameters_gsl_ST_int_{
    double z_obs;
    double M_Feed;
    double alpha_pl;
};

struct CosmoParams *cosmo_params_ps;

double sigma_norm, R, theta_cmb, omhh, z_equality, y_d, sound_horizon, alpha_nu, f_nu, f_baryon, beta_c, d2fact, R_CUTOFF, DEL_CURR, SIG_CURR;



double FgtrlnM_general(double lnM, void *params);
double FgtrM_general(float z, float M1, float M_Max, float M2, float MFeedback, float alpha, float delta1, float delta2);

float FgtrConditionalM_second(float z, float M1, float M2, float MFeedback, float alpha, float delta1, float delta2);
float dNdM_conditional_second(float z, float M1, float M2, float delta1, float delta2);

float FgtrConditionallnM(float M1, struct parameters_gsl_int_ parameters_gsl_int);
float GaussLegengreQuad_Fcoll(int n, float z, float M2, float MFeedback, float alpha, float delta1, float delta2);

float *Overdense_spline_gsl,*Overdense_spline_GL_high,*Fcoll_spline_gsl,*Fcoll_spline_GL_high,*xi_low,*xi_high,*wi_high,*wi_low;
float *second_derivs_low_GL,*second_derivs_high_GL,*Overdense_spline_GL_low,*Fcoll_spline_GL_low;

float *Mass_Spline, *Sigma_Spline, *dSigmadm_Spline, *second_derivs_sigma, *second_derivs_dsigma;

void initialiseSplinedSigmaM(float M_Min, float M_Max);
void initialiseGL_Fcoll(int n_low, int n_high, float M_Min, float M_Max);
void initialiseGL_FcollDblPl(int n_low, int n_high, float M_Min, float M_feedback, float M_Max);
void initialiseFcoll_spline(float z, float Mmin, float Mmax, float Mval, float MFeedback, float alphapl);

double dFdlnM_st_PL (double lnM, void *params);
double FgtrM_st_PL(double z, double Mmin, double MFeedback, double alpha_pl);

double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias);


unsigned long *lvector(long nl, long nh);
void free_lvector(unsigned long *v, long nl, long nh);

float *vector(long nl, long nh);
void free_vector(float *v, long nl, long nh);

void spline(float x[], float y[], int n, float yp1, float ypn, float y2[]);
void splint(float xa[], float ya[], float y2a[], int n, float x, float *y);

void gauleg(float x1, float x2, float x[], float w[], int n);

/*****     FUNCTION PROTOTYPES     *****/
double init_ps(); /* initialize global variables, MUST CALL THIS FIRST!!! returns R_CUTOFF */
double sigma_z0(double M); //calculates sigma at z=0 (no dicke)
double power_in_k(double k); /* Returns the value of the linear power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */
double TFmdm(double k); //Eisenstien & Hu power spectrum transfer function
void TFset_parameters();

double FgtrM(double z, double M);
double FgtrM_st(double z, double M);

float erfcc(float x);
double splined_erfc(double x);

double M_J_WDM();




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
//        printf("T = %e\n",T);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
//        printf("p = %e k = %e INDEX = %e T = %e\n",p,k,cosmo_params_ps->POWER_INDEX,T);
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
//    printf("k = %e p = %e w = %e\n",k,p,w);
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
//    printf("sigma_norm = %e\n",sigma_norm);
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


/*
 FUNCTION dsigmasqdm_z0(M)
 returns  d/dm (sigma^2) (see function sigma), in units of Msun^-1
 */
double dsigmasq_dm(double k, void *params){
    double p, w, T, gamma, q, aa, bb, cc, dwdr, drdm, kR;
    
    // get the power spectrum.. choice of 5:
    if (global_params.POWER_SPECTRUM == 0){ // Eisenstein & Hu ApJ, 1999, 511, 5
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
        //p = pow(k, POWER_INDEX - 0.05*log(k/0.05)) * T * T; //running, alpha=0.05
    }
    else if (global_params.POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb)/(cosmo_params_ps->OMm));
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
        bb = 4.7 / (cosmo_params_ps->hlittle*gamma);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (global_params.POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb)/(cosmo_params_ps->OMm));
        aa = 1.7/(cosmo_params_ps->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_ps->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * 19400.0 / pow(1 + aa*k + pow(bb*k, 1.5) + cc*k*k, 2);
    }
    else{
        fprintf(stderr, "No such power spectrum defined: %i\nOutput is bogus.\n", global_params.POWER_SPECTRUM);
        p = 0;
    }
    
    
    // now get the value of the window function
    kR = k * R;
    if (global_params.FILTER == 0){ // top hat
        if ( (kR) < 1.0e-4 ){ w = 1.0; }// w converges to 1 as (kR) -> 0
        else { w = 3.0 * (sin(kR)/pow(kR, 3) - cos(kR)/pow(kR, 2));}
        
        // now do d(w^2)/dm = 2 w dw/dr dr/dm
        if ( (kR) < 1.0e-10 ){  dwdr = 0;}
        else{ dwdr = 9*cos(kR)*k/pow(kR,3) + 3*sin(kR)*(1 - 3/(kR*kR))/(kR*R);}
        //3*k*( 3*cos(kR)/pow(kR,3) + sin(kR)*(-3*pow(kR, -4) + 1/(kR*kR)) );}
        //     dwdr = -1e8 * k / (R*1e3);
        drdm = 1.0 / (4.0*PI * cosmo_params_ps->OMm*RHOcrit * R*R);
    }
    else if (global_params.FILTER == 1){ // gaussian of width 1/R
        w = pow(E, -kR*kR/2.0);
        dwdr = - k*kR * w;
        drdm = 1.0 / (pow(2*PI, 1.5) * cosmo_params_ps->OMm*RHOcrit * 3*R*R);
    }
    else {
        fprintf(stderr, "No such filter: %i\nOutput is bogus.\n", global_params.FILTER);
        w=0;
    }
    
    //  printf("%e\t%e\t%e\t%e\t%e\t%e\t%e\n", k, R, p, w, dwdr, drdm, dsigmadk[1]);
    return k*k*p*2*w*dwdr*drdm * d2fact;
}
double dsigmasqdm_z0(double M){
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
    d2fact = M*10000/sigma_z0(M);
    
    F.function = &dsigmasq_dm;
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);
    //  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS15, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    return sigma_norm * sigma_norm * result /d2fact;
}



/*
 FUNCTION dNdM(z, M)
 Computes the Press_schechter mass function with Sheth-Torman correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).
 
 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1
 
 Reference: Sheth, Mo, Torman 2001
 */
double dNdM_st(double z, double M){
    double sigma, dsigmadm, nuhat, dicke_growth;
    
    dicke_growth = dicke(z);
    sigma = sigma_z0(M) * dicke_growth;
    dsigmadm = dsigmasqdm_z0(M) * dicke_growth*dicke_growth/(2.0*sigma);
    nuhat = sqrt(SHETH_a) * Deltac / sigma;
//    printf("z = %e M = %e gf = %e sigma = %e sigma_z0 = %e dsigmasqdm_z0 = %e dsigmadm = %e nuhat = %e\n",z,M,dicke_growth,sigma_z0(M),sigma,dsigmasqdm_z0(M),dsigmadm,nuhat);
//    printf("1 = %e 2 = %e 3 = %e 4 = %e 5 = %e\n",(-(cosmo_params_ps->OMm)*RHOcrit/M),(dsigmadm/sigma),sqrt(2/PI)*SHETH_A,(1+ pow(nuhat, -2*SHETH_p)),pow(E, -nuhat*nuhat/2.0));
    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * sqrt(2/PI)*SHETH_A * (1+ pow(nuhat, -2*SHETH_p)) * nuhat * pow(E, -nuhat*nuhat/2.0);
}



/*
 FUNCTION dNdM(z, M)
 Computes the Press_schechter mass function at
 redshift z, and dark matter halo mass M (in solar masses).
 
 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1
 
 Reference: Padmanabhan, pg. 214
 */
double dNdM(double z, double M){
    double sigma, dsigmadm, dicke_growth;
    
    dicke_growth = dicke(z);
    sigma = sigma_z0(M) * dicke_growth;
    dsigmadm = dsigmasqdm_z0(M) * (dicke_growth*dicke_growth/(2*sigma));
    
    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * sqrt(2/PI) * (Deltac/(sigma*sigma)) * dsigmadm * pow(E, -(Deltac*Deltac)/(2*sigma*sigma));
}




/*
 FUNCTION FgtrM_st(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 Uses Sheth-Torman correction
 */
double dFdlnM_st (double lnM, void *params){
    double z = *(double *)params;
    double M = exp(lnM);
//        printf("z = %e M = %e dNdM_st(z, M) = %e\n",z,M,dNdM_st(z, M));
    return dNdM_st(z, M) * M * M;
}
double FgtrM_st(double z, double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = 0.001; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    
    F.function = &dFdlnM_st;
    F.params = &z;
    lower_limit = log(M);
    upper_limit = log(FMAX(1e16, M*100));
    
    //  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);
    //  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS51, w, &result, &error);
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS41, w, &result, &error);
    //  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS31, w, &result, &error);
    //  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS21, w, &result, &error);
    //  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS15, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    return result / (cosmo_params_ps->OMm*RHOcrit);
}


/*
 FUNCTION FgtrM(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 */
double FgtrM(double z, double M){
    double del, sig;
    
    del = Deltac/dicke(z); //regular spherical collapse delta
    sig = sigma_z0(M);
    
    return splined_erfc(del / (sqrt(2)*sig));
}



/*
 FUNCTION FgtrM_st_PL(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 Uses Sheth-Torman correction
 */
double dFdlnM_st_PL (double lnM, void *params){
    
    struct parameters_gsl_ST_int_ vals = *(struct parameters_gsl_ST_int_ *)params;
    
    double M = exp(lnM);
    float z = vals.z_obs;
    float MFeedback = vals.M_Feed;
    float alpha = vals.alpha_pl;
    
    return dNdM_st(z, M) * M * M * pow((M/MFeedback),alpha);
}
double FgtrM_st_PL(double z, double Mmin, double MFeedback, double alpha_pl){
    
    double result_lower, result_upper, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = 0.001; //<- relative tolerance
    gsl_integration_workspace * w_lower
    = gsl_integration_workspace_alloc (1000);
    gsl_integration_workspace * w_upper
    = gsl_integration_workspace_alloc (1000);
    
    struct parameters_gsl_ST_int_  parameters_gsl_ST_lower = {
        .z_obs = z,
        .M_Feed = MFeedback,
        .alpha_pl = alpha_pl,
    };
    
    F.function = &dFdlnM_st_PL;
    F.params = &parameters_gsl_ST_lower;
    lower_limit = log(Mmin);
    upper_limit = log(1e16);
    
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w_lower, &result_lower, &error);
    gsl_integration_workspace_free (w_lower);
    
    return (result_lower) / (cosmo_params_ps->OMm*RHOcrit);
}

/* returns the "effective Jeans mass" in Msun
 corresponding to the gas analog of WDM ; eq. 10 in Barkana+ 2001 */
double M_J_WDM(){
    double z_eq, fudge=60;
    if (!(global_params.P_CUTOFF))
        return 0;
    z_eq = 3600*(cosmo_params_ps->OMm-cosmo_params_ps->OMb)*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle/0.15;
    return fudge*3.06e8 * (1.5/global_params.g_x) * sqrt((cosmo_params_ps->OMm-cosmo_params_ps->OMb)*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle/0.15) * pow(global_params.M_WDM, -4) * pow(z_eq/3000.0, 1.5);
}

float erfcc(float x)
{
    double t,q,ans;
    
    q=fabs(x);
    t=1.0/(1.0+0.5*q);
    ans=t*exp(-q*q-1.2655122+t*(1.0000237+t*(0.374092+t*(0.0967842+
                                                         t*(-0.1862881+t*(0.2788681+t*(-1.13520398+t*(1.4885159+
                                                                                                      t*(-0.82215223+t*0.17087277)))))))));
    return x >= 0.0 ? ans : 2.0-ans;
}

double splined_erfc(double x){
    if (x < 0){
        //    fprintf(stderr, "WARNING: Negative value %e passed to splined_erfc. Returning 1\n", x);
        return 1;
    }
    return erfcc(x); // the interpolation below doesn't seem to be stable in Ts.c
    if (x > ERFC_PARAM_DELTA*(ERFC_NPTS-1))
        return erfcc(x);
    else
        return exp(gsl_spline_eval(erfc_spline, x, erfc_acc));
}


float FgtrConditionalM_second(float z, float M1, float M2, float MFeedback, float alpha, float delta1, float delta2) {
    
    return exp(M1)*pow(exp(M1)/MFeedback,alpha)*dNdM_conditional_second(z,M1,M2,delta1,delta2)/sqrt(2.*PI);
}

float dNdM_conditional_second(float z, float M1, float M2, float delta1, float delta2){
    
    float sigma1, sigma2, dsigmadm, dicke_growth,dsigma_val;
    
    M1 = exp(M1);
    M2 = exp(M2);
    
    dicke_growth = dicke(z);
    
    splint(Mass_Spline-1,Sigma_Spline-1,second_derivs_sigma-1,(int)NMass,M1,&(sigma1));
    splint(Mass_Spline-1,Sigma_Spline-1,second_derivs_sigma-1,(int)NMass,M2,&(sigma2));
    
    sigma1 = sigma1*sigma1;
    sigma2 = sigma2*sigma2;
    
    splint(Mass_Spline-1,dSigmadm_Spline-1,second_derivs_dsigma-1,(int)NMass,M1,&(dsigma_val));
    
    dsigmadm = -pow(10.,dsigma_val)/(2.0*sigma1); // This is actually sigma1^{2} as calculated above, however, it should just be sigma1. It cancels with the same factor below. Why I have decided to write it like that I don't know!
    
    if((sigma1 > sigma2)) {
        
        return -(( delta1 - delta2 )/dicke_growth)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*dicke_growth*dicke_growth*( sigma1 - sigma2 ) ) ) )/(pow( sigma1 - sigma2, 1.5));
    }
    else if(sigma1==sigma2) {
        
        return -(( delta1 - delta2 )/dicke_growth)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*dicke_growth*dicke_growth*( 1.e-6 ) ) ) )/(pow( 1.e-6, 1.5));
        
    }
    else {
        return 0.;
    }
}

void gauleg(float x1, float x2, float x[], float w[], int n)
//Given the lower and upper limits of integration x1 and x2, and given n, this routine returns arrays x[1..n] and w[1..n] of length n,
//containing the abscissas and weights of the Gauss- Legendre n-point quadrature formula.
{
    
    int m,j,i;
    double z1,z,xm,xl,pp,p3,p2,p1;
    
    m=(n+1)/2;
    xm=0.5*(x2+x1);
    xl=0.5*(x2-x1);
    for (i=1;i<=m;i++) {
        //High precision is a good idea for this routine.
        //The roots are symmetric in the interval, so we only have to find half of them.
        //Loop over the desired roots.
        
        z=cos(3.141592654*(i-0.25)/(n+0.5));
        
        //Starting with the above approximation to the ith root, we enter the main loop of refinement by Newton’s method.
        do {
            p1=1.0;
            p2=0.0;
            for (j=1;j<=n;j++) {
                //Loop up the recurrence relation to get the Legendre polynomial evaluated at z.
                p3=p2;
                p2=p1;
                p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
            }
            //p1 is now the desired Legendre polynomial. We next compute pp, its derivative, by a standard relation involving also p2,
            //the polynomial of one lower order.
            pp=n*(z*p1-p2)/(z*z-1.0);
            z1=z;
            z=z1-p1/pp;
        } while (fabs(z-z1) > EPS2);
        x[i]=xm-xl*z;
        x[n+1-i]=xm+xl*z;
        w[i]=2.0*xl/((1.0-z*z)*pp*pp);
        w[n+1-i]=w[i];
    }
}

double FgtrlnM_general(double lnM, void *params) {
    
    struct parameters_gsl_int_ vals = *(struct parameters_gsl_int_ *)params;
    
    float z = vals.z_obs;
    float M2 = vals.Mval;
    float MFeedback = vals.M_Feed;
    float alpha = vals.alpha_pl;
    float delta1 = vals.del_traj_1;
    float delta2 = vals.del_traj_2;
    
    return FgtrConditionalM_second(z,lnM,M2,MFeedback,alpha,delta1,delta2);
}
double FgtrM_general(float z, float M1, float M_Max, float M2, float MFeedback, float alpha, float delta1, float delta2) {
    
    double result, error, lower_limit, upper_limit;
    
    double rel_tol = 0.01;
    
    int size;
    size = 1000;
    
    //    printf("delta1 = %e Deltac = %e\n",delta1,Deltac);
    
    if((float)delta1==(float)Deltac) {
        
        gsl_function Fx;
        
        gsl_integration_workspace * w = gsl_integration_workspace_alloc (size);
        
        Fx.function = &FgtrlnM_general;
        
        struct parameters_gsl_int_  parameters_gsl_int = {
            .z_obs = z,
            .Mval = M2,
            .M_Feed = MFeedback,
            .alpha_pl = alpha,
            .del_traj_1 = delta1,
            .del_traj_2 = delta2
        };
        
        Fx.params = &parameters_gsl_int;
        
        lower_limit = M1;
        upper_limit = M_Max;
        
        //        gsl_integration_qag (&Fx, lower_limit, upper_limit, 0, rel_tol, size, GSL_INTEG_GAUSS15, w, &result, &error);
        gsl_integration_qag (&Fx, lower_limit, upper_limit, 0, rel_tol, size, GSL_INTEG_GAUSS61, w, &result, &error);
        gsl_integration_workspace_free (w);
        
        if(delta2 > delta1) {
            return 1.;
        }
        else {
            return result;
        }
    }
}

float FgtrConditionallnM(float M1, struct parameters_gsl_int_ parameters_gsl_int) {
    
    float z = parameters_gsl_int.z_obs;
    float M2 = parameters_gsl_int.Mval;
    float MFeedback = parameters_gsl_int.M_Feed;
    float alpha = parameters_gsl_int.alpha_pl;
    float delta1 = parameters_gsl_int.del_traj_1;
    float delta2 = parameters_gsl_int.del_traj_2;
    
    return exp(M1)*pow(exp(M1)/MFeedback,alpha)*dNdM_conditional_second(z,M1,M2,delta1,delta2)/sqrt(2.*PI);
}

float GaussLegengreQuad_Fcoll(int n, float z, float M2, float MFeedback, float alpha, float delta1, float delta2)
{
    //Performs the Gauss-Legendre quadrature.
    int i;
    
    float integrand,x;
    integrand = 0.0;
    
    struct parameters_gsl_int_  parameters_gsl_int = {
        .z_obs = z,
        .Mval = M2,
        .M_Feed = MFeedback,
        .alpha_pl = alpha,
        .del_traj_1 = delta1,
        .del_traj_2 = delta2
    };
    
    if(delta2>delta1) {
        return 1.;
    }
    else {
        for(i=1;i<(n+1);i++) {
            x = xi_low[i];
            integrand += wi_low[i]*FgtrConditionallnM(x,parameters_gsl_int);
        }
        return integrand;
    }
}

void initialiseSplinedSigmaM(float M_Min, float M_Max)
{
    int i;
    float Mass;
    
    Mass_Spline = calloc(NMass,sizeof(float));
    Sigma_Spline = calloc(NMass,sizeof(float));
    dSigmadm_Spline = calloc(NMass,sizeof(float));
    second_derivs_sigma = calloc(NMass,sizeof(float));
    second_derivs_dsigma = calloc(NMass,sizeof(float));
    
    for(i=0;i<NMass;i++) {
        Mass_Spline[i] = pow(10., log10(M_Min) + (float)i/(NMass-1)*( log10(M_Max) - log10(M_Min) ) );
        Sigma_Spline[i] = sigma_z0(Mass_Spline[i]);
        dSigmadm_Spline[i] = log10(-dsigmasqdm_z0(Mass_Spline[i]));
    }
    spline(Mass_Spline-1,Sigma_Spline-1,NMass,0,0,second_derivs_sigma-1);
    spline(Mass_Spline-1,dSigmadm_Spline-1,NMass,0,0,second_derivs_dsigma-1);
}

void initialiseGL_Fcoll(int n_low, int n_high, float M_Min, float M_Max)
{
    //calculates the weightings and the positions for Gauss-Legendre quadrature.
    
    gauleg(log(M_Min),log(M_Max),xi_low,wi_low,n_low);
    
    gauleg(log(M_Min),log(M_Max),xi_high,wi_high,n_high);
    
}

void initialiseFcoll_spline(float z, float Mmin, float Mmax, float Mval, float MFeedback, float alphapl)
{
    double overdense_val,overdense_small_low,overdense_small_high,overdense_large_low,overdense_large_high;
    int i;
    
    overdense_large_high = Deltac;
    overdense_large_low = 1.5;
    overdense_small_high = 1.5;
    overdense_small_low = -1. + 9.e-8;
    
    Fcoll_spline_acc   = gsl_interp_accel_alloc ();
    Fcoll_spline  = gsl_spline_alloc (gsl_interp_cspline, SPLINE_NPTS);
    
    for (i=0;i<SPLINE_NPTS;i++){
        overdense_val = log10(1.+overdense_small_low) + (float)i/(SPLINE_NPTS-1.)*(log10(1.+overdense_small_high) - log10(1.+overdense_small_low));
        
        log_Fcoll_spline_table[i] = log10(GaussLegengreQuad_Fcoll(NGLlow,z,log(Mval),MFeedback,alphapl,Deltac,pow(10.,overdense_val)-1.));
        Fcoll_spline_params[i] = overdense_val;
        
        if(log_Fcoll_spline_table[i]<-40.) {
            log_Fcoll_spline_table[i] = -40.;
        }
    }
    gsl_spline_init(Fcoll_spline, Fcoll_spline_params, log_Fcoll_spline_table, SPLINE_NPTS);
    
    for(i=0;i<Nhigh;i++) {
        Overdense_spline_GL_high[i] = overdense_large_low + (float)i/((float)Nhigh-1.)*(overdense_large_high - overdense_large_low);
        Fcoll_spline_GL_high[i] = FgtrM_general(z,log(Mmin),log(Mmax),log(Mval),MFeedback,alphapl,Deltac,Overdense_spline_GL_high[i]);
        
        if(Fcoll_spline_GL_high[i]<0.) {
            Fcoll_spline_GL_high[i]=pow(10.,-40.0);
        }
    }
    spline(Overdense_spline_GL_high-1,Fcoll_spline_GL_high-1,Nhigh,0,0,second_derivs_high_GL-1);
}

void FcollSpline(float Overdensity, float *splined_value)
{
    int i;
    float returned_value;
    
    if(Overdensity<1.5) {
        if(Overdensity<-1.) {
            returned_value = 0;
        }
        else {
            returned_value = gsl_spline_eval(Fcoll_spline, log10(Overdensity+1.), Fcoll_spline_acc);
            returned_value = pow(10.,returned_value);
        }
    }
    else {
        if(Overdensity<Deltac) {
            splint(Overdense_spline_GL_high-1,Fcoll_spline_GL_high-1,second_derivs_high_GL-1,(int)Nhigh,Overdensity,&(returned_value));
        }
        else {
            returned_value = 1.;
        }
    }
    *splined_value = returned_value;
}

void nrerror(char error_text[])
{
    fprintf(stderr,"Numerical Recipes run-time error...\n");
    fprintf(stderr,"%s\n",error_text);
    fprintf(stderr,"...now exiting to system...\n");
    exit(1);
}

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
    float *v;
    v = (float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
    if(!v) nrerror("allocation failure in vector()");
    return v - nl + NR_END;
}

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

void spline(float x[], float y[], int n, float yp1, float ypn, float y2[])
/*Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e., yi = f(xi), with
 x1 <x2 < :: : < xN, and given values yp1 and ypn for the first derivative of the interpolating
 function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
 the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
 ypn are equal to 1e30 or larger, the routine is signaled to set the corresponding boundary
 condition for a natural spline, with zero second derivative on that boundary.*/
{
    int i,k;
    float p,qn,sig,un,*u;
    int na,nb,check;
    u=vector(1,n-1);
    if (yp1 > 0.99e30)                     // The lower boundary condition is set either to be "natural"
        y2[1]=u[1]=0.0;
    else {                                 // or else to have a specified first derivative.
        y2[1] = -0.5;
        u[1]=(3.0/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1])-yp1);
    }
    for (i=2;i<=n-1;i++) {                              //This is the decomposition loop of the tridiagonal algorithm.
        sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);                //y2 and u are used for temporary
        na = 1;
        nb = 1;
        check = 0;
        while(((float)(x[i+na*1]-x[i-nb*1])==(float)0.0)) {
            check = check + 1;
            if(check%2==0) {
                na = na + 1;
            }
            else {
                nb = nb + 1;
            }
            sig=(x[i]-x[i-1])/(x[i+na*1]-x[i-nb*1]);
        }
        p=sig*y2[i-1]+2.0;                                //storage of the decomposed
        y2[i]=(sig-1.0)/p;                                //  factors.
        u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
        u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
        
        if(((float)(x[i+1]-x[i])==(float)0.0) || ((float)(x[i]-x[i-1])==(float)0.0)) {
            na = 0;
            nb = 0;
            check = 0;
            while((float)(x[i+na*1]-x[i-nb])==(float)(0.0) || ((float)(x[i+na]-x[i-nb*1])==(float)0.0)) {
                check = check + 1;
                if(check%2==0) {
                    na = na + 1;
                }
                else {
                    nb = nb + 1;
                }
            }
            u[i]=(y[i+1]-y[i])/(x[i+na*1]-x[i-nb]) - (y[i]-y[i-1])/(x[i+na]-x[i-nb*1]);
            
            u[i]=(6.0*u[i]/(x[i+na*1]-x[i-nb*1])-sig*u[i-1])/p;
            
        }
    }
    if (ypn > 0.99e30)                        //The upper boundary condition is set either to be "natural"
        qn=un=0.0;
    else {                                    //or else to have a specified first derivative.
        qn=0.5;
        un=(3.0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]));
    }
    y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.0);
    
    for (k=n-1;k>=1;k--) {                      //This is the backsubstitution loop of the tridiagonal
        y2[k]=y2[k]*y2[k+1]+u[k];               //algorithm.
    }
    free_vector(u,1,n-1);
}


void splint(float xa[], float ya[], float y2a[], int n, float x, float *y)
/*Given the arrays xa[1..n] and ya[1..n], which tabulate a function (with the xai's in order),
 and given the array y2a[1..n], which is the output from spline above, and given a value of
 x, this routine returns a cubic-spline interpolated value y.*/
{
    void nrerror(char error_text[]);
    int klo,khi,k;
    float h,b,a;
    klo=1;                                                  // We will find the right place in the table by means of
    khi=n;                                                  //bisection. This is optimal if sequential calls to this
    while (khi-klo > 1) {                                   //routine are at random values of x. If sequential calls
        k=(khi+klo) >> 1;                                     //are in order, and closely spaced, one would do better
        if (xa[k] > x) khi=k;                                 //to store previous values of klo and khi and test if
        else klo=k;                                           //they remain appropriate on the next call.
    }                                                           // klo and khi now bracket the input value of x.
    h=xa[khi]-xa[klo];
    if (h == 0.0) nrerror("Bad xa input to routine splint");    //The xa's must be distinct.
    a=(xa[khi]-x)/h;
    b=(x-xa[klo])/h;                                            //Cubic spline polynomial is now evaluated.
    *y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
    unsigned long *v;
    v = (unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
    if(!v) nrerror("allocation failure in lvector()");
    return v - nl + NR_END;
}

void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

/* Uses sigma parameters instead of Mass for scale */
double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias){
    double del, sig;
    
    if (!(sig_bias < sigsmallR)){ // biased region is smaller that halo!
        //    fprintf(stderr, "local_FgtrM_bias: Biased region is smaller than halo!\nResult is bogus.\n");
        //    return 0;
        return 0.000001;
    }
    
    del = Deltac/dicke(z) - del_bias;
    sig = sqrt(sigsmallR*sigsmallR - sig_bias*sig_bias);
    
    return splined_erfc(del / (sqrt(2)*sig));
}

/* redshift derivative of the growth function at z */
double ddicke_dz(double z){
    float dz = 1e-10;
    double omegaM_z, ddickdz, dick_0, x, x_0, domegaMdz;
    
    return (dicke(z+dz)-dicke(z))/dz;
}