
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

#define NMass 200

#define NSFR_high 200
#define NSFR_low 250
#define NGL_SFR 100

//#define zpp_interp_points_SFR (int) (300)

#define NR_END 1
#define FREE_ARG char*

#define MM 7
#define NSTACK 50

#define EPS2 3.0e-11

#define Luv_over_SFR (double)(1./1.15/1e-28)

//     Luv/SFR = 1 / 1.15 x 10^-28 [M_solar yr^-1/erg s^-1 Hz^-1]
//     G. Sun and S. R. Furlanetto (2016) MNRAS, 417, 33

#define delta_lnMhalo (double)(5e-6)
#define Mhalo_min (double)(1e6)
#define Mhalo_max (double)(1e16)

bool initialised_ComputeLF = false;

gsl_interp_accel *LF_spline_acc;
gsl_spline *LF_spline;

struct CosmoParams *cosmo_params_ps;
struct UserParams *user_params_ps;
struct FlagOptions *flag_options_ps;

double sigma_norm, R, theta_cmb, omhh, z_equality, y_d, sound_horizon, alpha_nu, f_nu, f_baryon, beta_c, d2fact, R_CUTOFF, DEL_CURR, SIG_CURR;

float MinMass, mass_bin_width, inv_mass_bin_width;

double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias);

float *Mass_InterpTable, *Sigma_InterpTable, *dSigmadm_InterpTable;

float *log10_overdense_spline_SFR, *log10_Nion_spline, *Overdense_spline_SFR, *Nion_spline;

float *xi_SFR,*wi_SFR, *xi_SFR_Xray, *wi_SFR_Xray;

float *Overdense_high_table, *overdense_low_table, *log10_overdense_low_table;
float **log10_SFRD_z_low_table, **SFRD_z_high_table;

double *lnMhalo_param, *Muv_param, *Mhalo_param;
double *log10phi, *M_uv_z, *M_h_z;

double *z_val, *z_X_val, *Nion_z_val, *SFRD_val;

int initialiseSigmaMInterpTable(float M_Min, float M_Max);
void freeSigmaMInterpTable();
void initialiseGL_Nion(int n, float M_TURN, float M_Max);
void initialiseGL_Nion_Xray(int n, float M_TURN, float M_Max);

float Mass_limit (float logM, float PL, float FRAC);
void bisection(float *x, float xlow, float xup, int *iter);
float Mass_limit_bisection(float Mmin, float Mmax, float PL, float FRAC);

float dNdM_conditional(float growthf, float M1, float M2, float delta1, float delta2, float sigma2);
double dNion_ConditionallnM(double lnM, void *params);
double Nion_ConditionalM(double growthf, double M1, double M2, double sigma2, double delta1, double delta2, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc);

float GaussLegendreQuad_Nion(int Type, int n, float growthf, float M2, float sigma2, float delta1, float delta2, float MassTurnover, float Alpha_star, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc);

struct parameters_gsl_FgtrM_int_{
    double z_obs;
    double gf_obs;
};

struct parameters_gsl_SFR_General_int_{
    double z_obs;
    double gf_obs;
    double Mdrop;
    double pl_star;
    double pl_esc;
    double frac_star;
    double frac_esc;
    double LimitMass_Fstar;
    double LimitMass_Fesc;
};

struct parameters_gsl_SFR_con_int_{
    double gf_obs;
    double Mval;
    double sigma2;
    double delta1;
    double delta2;
    double Mdrop;
    double pl_star;
    double pl_esc;
    double frac_star;
    double frac_esc;
    double LimitMass_Fstar;
    double LimitMass_Fesc;
};

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
double FgtrM_Watson(double growthf, double M);
double FgtrM_Watson_z(double z, double growthf, double M);
double FgtrM_General(double z, double M);

float erfcc(float x);
double splined_erfc(double x);

double M_J_WDM();

void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params){
 
    cosmo_params_ps = cosmo_params;
    user_params_ps = user_params;
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
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", global_params.POWER_SPECTRUM);
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
        LOG_ERROR("No such filter: %i. Output is bogus.", global_params.FILTER);
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
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", global_params.POWER_SPECTRUM);
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
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", global_params.POWER_SPECTRUM);
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
        LOG_ERROR("No such filter: %i. Output is bogus.", global_params.FILTER);
        w=0;
    }
    
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

    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * sqrt(2./PI)*SHETH_A * (1+ pow(nuhat, -2*SHETH_p)) * nuhat * pow(E, -nuhat*nuhat/2.0);
}

/*
 FUNCTION dNdM_st_interp(z, M)
 Computes the Press_schechter mass function with Sheth-Torman correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).
 
 Uses interpolated sigma and dsigmadm to be computed faster. Necessary for mass-dependent ionising efficiencies.
 
 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1
 
 Reference: Sheth, Mo, Torman 2001
 */
double dNdM_st_interp(double growthf, double M){
    
    double sigma, dsigmadm, nuhat;
    
    float MassBinLow;
    int MassBin;
    
    MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
    MassBinLow = MinMass + mass_bin_width*(float)MassBin;

    sigma = Sigma_InterpTable[MassBin] + ( log(M) - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    sigma = sigma * growthf;
    
    dsigmadm = dSigmadm_InterpTable[MassBin] + ( log(M) - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
    dsigmadm = -pow(10.,dsigmadm);
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    nuhat = sqrt(SHETH_a) * Deltac / sigma;
    
    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * sqrt(2./PI)*SHETH_A * (1+ pow(nuhat, -2*SHETH_p)) * nuhat * pow(E, -nuhat*nuhat/2.0);
}

/*
 FUNCTION dNdM_WatsonFOF(z, M)
 Computes the Press_schechter mass function with Warren et al. 2011 correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).
 
 The Universial FOF function (Eq. 12) of Watson et al. 2013
 
 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1
 
 Reference: Watson et al. 2013
 */
double dNdM_WatsonFOF(double growthf, double M){

    double sigma, dsigmadm, f_sigma;
    
    float MassBinLow;
    int MassBin;
    
    MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
    MassBinLow = MinMass + mass_bin_width*(float)MassBin;
    
    sigma = Sigma_InterpTable[MassBin] + ( log(M) - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    sigma = sigma * growthf;
    
    dsigmadm = dSigmadm_InterpTable[MassBin] + ( log(M) - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
    dsigmadm = -pow(10.,dsigmadm);
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));
    
    f_sigma = Watson_A * ( pow( Watson_beta/sigma, Watson_alpha) + 1. ) * exp( - Watson_gamma/(sigma*sigma) );
    
    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * f_sigma;
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
double dNdM_WatsonFOF_z(double z, double growthf, double M){
    
    double sigma, dsigmadm, A_z, alpha_z, beta_z, Omega_m_z, f_sigma;
    float MassBinLow;
    int MassBin;
    
    MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
    MassBinLow = MinMass + mass_bin_width*(float)MassBin;

    sigma = Sigma_InterpTable[MassBin] + ( log(M) - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    sigma = sigma * growthf;
    
    dsigmadm = dSigmadm_InterpTable[MassBin] + ( log(M) - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
    dsigmadm = -pow(10.,dsigmadm);
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));
    
    Omega_m_z = (cosmo_params_ps->OMm)*pow(1.+z,3.) / ( (cosmo_params_ps->OMl) + (cosmo_params_ps->OMm)*pow(1.+z,3.) + (global_params.OMr)*pow(1.+z,4.) );
    
    A_z = Omega_m_z * ( Watson_A_z_1 * pow(1. + z, Watson_A_z_2 ) + Watson_A_z_3 );
    alpha_z = Omega_m_z * ( Watson_alpha_z_1 * pow(1.+z, Watson_alpha_z_2 ) + Watson_alpha_z_3 );
    beta_z = Omega_m_z * ( Watson_beta_z_1 * pow(1.+z, Watson_beta_z_2 ) + Watson_beta_z_3 );
    
    f_sigma = A_z * ( pow(beta_z/sigma, alpha_z) + 1. ) * exp( - Watson_gamma_z/(sigma*sigma) );
    
    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * f_sigma;
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
    dsigmadm = dsigmasqdm_z0(M) * (dicke_growth*dicke_growth/(2.*sigma));
    
    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * sqrt(2/PI) * (Deltac/(sigma*sigma)) * dsigmadm * pow(E, -(Deltac*Deltac)/(2*sigma*sigma));
}

/*
 FUNCTION dNdM_interp(z, M)
 Computes the Press_schechter mass function at
 redshift z, and dark matter halo mass M (in solar masses).
 
 Uses interpolated sigma and dsigmadm to be computed faster. Necessary for mass-dependent ionising efficiencies.
 
 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1
 
 Reference: Padmanabhan, pg. 214
 */
double dNdM_interp(double growthf, double M){
    double sigma, dsigmadm;
    float MassBinLow;
    int MassBin;
    
    MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
    MassBinLow = MinMass + mass_bin_width*(float)MassBin;
    
    sigma = Sigma_InterpTable[MassBin] + ( log(M) - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    sigma = sigma * growthf;
    
    dsigmadm = dSigmadm_InterpTable[MassBin] + ( log(M) - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
    dsigmadm = -pow(10.,dsigmadm);
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));
    
    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * sqrt(2/PI) * (Deltac/(sigma*sigma)) * dsigmadm * pow(E, -(Deltac*Deltac)/(2*sigma*sigma));
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
 FUNCTION FgtrM_Watson(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 Uses Watson et al (2013) correction
 */
double dFdlnM_Watson_z (double lnM, void *params){
    struct parameters_gsl_FgtrM_int_ vals = *(struct parameters_gsl_FgtrM_int_ *)params;
    
    double M = exp(lnM);
    double z = vals.z_obs;
    double growthf = vals.gf_obs;

    return dNdM_WatsonFOF_z(z, growthf, M) * M * M;
}
double FgtrM_Watson_z(double z, double growthf, double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = 0.001; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    
    F.function = &dFdlnM_Watson_z;
    struct parameters_gsl_FgtrM_int_ parameters_gsl_FgtrM = {
        .z_obs = z,
        .gf_obs = growthf,
    };

    F.params = &parameters_gsl_FgtrM;
    lower_limit = log(M);
    upper_limit = log(FMAX(1e16, M*100));
    
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    return result / (cosmo_params_ps->OMm*RHOcrit);
}


/*
 FUNCTION FgtrM_Watson(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 Uses Watson et al (2013) correction
 */
double dFdlnM_Watson (double lnM, void *params){
    double growthf = *(double *)params;
    double M = exp(lnM);
    return dNdM_WatsonFOF(growthf, M) * M * M;
}
double FgtrM_Watson(double growthf, double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = 0.001; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    
    F.function = &dFdlnM_Watson;
    F.params = &growthf;
    lower_limit = log(M);
    upper_limit = log(FMAX(1e16, M*100));
    
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    return result / (cosmo_params_ps->OMm*RHOcrit);
}

double dFdlnM_General(double lnM, void *params){
    struct parameters_gsl_FgtrM_int_ vals = *(struct parameters_gsl_FgtrM_int_ *)params;
    
    double M = exp(lnM);
    double z = vals.z_obs;
    double growthf = vals.gf_obs;
    
    double MassFunction;
 
    if(user_params_ps->HMF==0) {
        MassFunction = dNdM(z, M);
    }
    if(user_params_ps->HMF==1) {
        MassFunction = dNdM_st_interp(growthf, M);
    }
    if(user_params_ps->HMF==2) {
        MassFunction = dNdM_WatsonFOF(growthf, M);
    }
    if(user_params_ps->HMF==3) {
        MassFunction = dNdM_WatsonFOF_z(z, growthf, M);
    }
    return MassFunction * M * M;
}

/*
 FUNCTION FgtrM_General(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 */
double FgtrM_General(double z, double M){

    double del, sig, growthf;
    
    growthf = dicke(z);

    struct parameters_gsl_FgtrM_int_ parameters_gsl_FgtrM = {
        .z_obs = z,
        .gf_obs = growthf,
    };
    
    if(user_params_ps->HMF<4 && user_params_ps->HMF>-1) {
    
        double result, error, lower_limit, upper_limit;
        gsl_function F;
        double rel_tol  = 0.001; //<- relative tolerance
        gsl_integration_workspace * w
        = gsl_integration_workspace_alloc (1000);
    
        F.function = &dFdlnM_General;
        F.params = &parameters_gsl_FgtrM;
    
        lower_limit = log(M);
        upper_limit = log(FMAX(1e16, M*100));

LOG_ULTRA_DEBUG("integration range: %f to %f", lower_limit, upper_limit);

        gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &result, &error);
    
        gsl_integration_workspace_free (w);
    
        return result / (cosmo_params_ps->OMm*RHOcrit);
    }
    else {
        LOG_ERROR("Incorrect HMF selected: %i (should be between 0 and 3).", user_params_ps->HMF);
        exit(-1);
    }
}

double dNion_General(double lnM, void *params){
    struct parameters_gsl_SFR_General_int_ vals = *(struct parameters_gsl_SFR_General_int_ *)params;
    
    double M = exp(lnM);
    double z = vals.z_obs;
    double growthf = vals.gf_obs;
    double MassTurnover = vals.Mdrop;
    double Alpha_star = vals.pl_star;
    double Alpha_esc = vals.pl_esc;
    double Fstar10 = vals.frac_star;
    double Fesc10 = vals.frac_esc;
    double Mlim_Fstar = vals.LimitMass_Fstar;
    double Mlim_Fesc = vals.LimitMass_Fesc;
    
    double Fstar, Fesc, MassFunction;
    
    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar10;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1/Fstar10;
    else
        Fstar = pow(M/1e10,Alpha_star);
    
    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc10;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc10;
    else
        Fesc = pow(M/1e10,Alpha_esc);

    if(user_params_ps->HMF==0) {
        MassFunction = dNdM(z, M);
    }
    if(user_params_ps->HMF==1) {
        MassFunction = dNdM_st_interp(growthf,M);
    }
    if(user_params_ps->HMF==2) {
        MassFunction = dNdM_WatsonFOF(growthf, M);
    }
    if(user_params_ps->HMF==3) {
        MassFunction = dNdM_WatsonFOF_z(z, growthf, M);
    }
    
    return MassFunction * M * M * exp(-MassTurnover/M) * Fstar * Fesc;
}

double Nion_General(double z, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc){
    
    double growthf;
    
    growthf = dicke(z);
    
    double M_Min = MassTurnover/50.;
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.001; //<- relative tolerance
    
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    
    struct parameters_gsl_SFR_General_int_ parameters_gsl_SFR = {
        .z_obs = z,
        .gf_obs = growthf,
        .Mdrop = MassTurnover,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar10,
        .frac_esc = Fesc10,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc,
    };
    
    if(user_params_ps->HMF<4 && user_params_ps->HMF>-1) {
    
        F.function = &dNion_General;
        F.params = &parameters_gsl_SFR;
    
        lower_limit = log(M_Min);
        upper_limit = log(FMAX(1e16, M_Min*100));
    
        gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);
        gsl_integration_workspace_free (w);
    
        return result / ((cosmo_params_ps->OMm)*RHOcrit);
    }
    else {
        LOG_ERROR("Incorrect HMF selected: %i (should be between 0 and 3).", user_params_ps->HMF);
        exit(-1);
    }
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

int initialiseSigmaMInterpTable(float M_Min, float M_Max)
{
    int i;
    float Mass;
    
    Mass_InterpTable = calloc(NMass,sizeof(float));
    Sigma_InterpTable = calloc(NMass,sizeof(float));
    dSigmadm_InterpTable = calloc(NMass,sizeof(float));
    
    for(i=0;i<NMass;i++) {
        Mass_InterpTable[i] = log(M_Min) + (float)i/(NMass-1)*( log(M_Max) - log(M_Min) );
        Sigma_InterpTable[i] = sigma_z0(exp(Mass_InterpTable[i]));
        dSigmadm_InterpTable[i] = log10(-dsigmasqdm_z0(exp(Mass_InterpTable[i])));
        
        if(isfinite(Mass_InterpTable[i]) == 0 || isfinite(Sigma_InterpTable[i]) == 0 || isfinite(dSigmadm_InterpTable[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
            return(-1);
        }
    }
    
    MinMass = log(M_Min);
    mass_bin_width = 1./(NMass-1)*( log(M_Max) - log(M_Min) );
    inv_mass_bin_width = 1./mass_bin_width;
    
    return(0);
}

void freeSigmaMInterpTable()
{
    free(Mass_InterpTable);
    free(Sigma_InterpTable);
    free(dSigmadm_InterpTable);
}


void nrerror(char error_text[])
{
    LOG_ERROR("Numerical Recipes run-time error...");
    LOG_ERROR("%s",error_text);
    LOG_ERROR("...now exiting to system...");
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





/* compute a mass limit where the stellar baryon fraction and the escape fraction exceed unity */
float Mass_limit (float logM, float PL, float FRAC) {
    return FRAC*pow(pow(10.,logM)/1e10,PL);
}
void bisection(float *x, float xlow, float xup, int *iter){
    *x=(xlow + xup)/2.;
    ++(*iter);
}

float Mass_limit_bisection(float Mmin, float Mmax, float PL, float FRAC){
    int i, iter, max_iter=200;
    float rel_tol=0.001;
    float logMlow, logMupper, x, x1;
    iter = 0;
    logMlow = log10(Mmin);
    logMupper = log10(Mmax);
    
    if (PL < 0.) {
        if (Mass_limit(logMlow,PL,FRAC) <= 1.) {
            return Mmin;
        }
    }
    else if (PL > 0.) {
        if (Mass_limit(logMupper,PL,FRAC) <= 1.) {
            return Mmax;
        }
    }
    else
        return 0;
    bisection(&x, logMlow, logMupper, &iter);
    do {
        if((Mass_limit(logMlow,PL,FRAC)-1.)*(Mass_limit(x,PL,FRAC)-1.) < 0.)
            logMupper = x;
        else
            logMlow = x;
        bisection(&x1, logMlow, logMupper, &iter);
        if(fabs(x1-x) < rel_tol) {
            return pow(10.,x1);
        }
        x = x1;
    }
    while(iter < max_iter);
    LOG_ERROR("Failed to find a mass limit to regulate stellar fraction/escape fraction is between 0 and 1.");
    LOG_ERROR(" The solution does not converge or iterations are not sufficient.");
    return -1;
}

void initialise_ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options) {

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    
    lnMhalo_param = calloc(nbins,sizeof(double));
    Muv_param = calloc(nbins,sizeof(double));
    Mhalo_param = calloc(nbins,sizeof(double));
    
    LF_spline_acc = gsl_interp_accel_alloc();
    LF_spline = gsl_spline_alloc(gsl_interp_cspline, nbins);
    
    init_ps();
    
    if(initialiseSigmaMInterpTable(0.999*Mhalo_min,1.001*Mhalo_max)!=0) {
        LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable from initialise_ComputeLF");
        return(2);
    }
    
    initialised_ComputeLF = true;
    
}

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, double *M_uv_z, double *M_h_z, double *log10phi) {

    // This NEEDS to be done every time, because the actual object passed in as
    // user_params, cosmo_params etc. can change on each call, freeing up the memory.
    initialise_ComputeLF(nbins, user_params,cosmo_params,astro_params,flag_options);

    int i,i_z;
    double  dlnMhalo, lnMhalo_i, SFRparam, Muv_1, Muv_2, dMuvdMhalo;
    double Mhalo_i, lnMhalo_min, lnMhalo_max, lnMhalo_lo, lnMhalo_hi, dlnM, growthf;
    float Mlim_Fstar,Fstar;
    
    Mlim_Fstar = Mass_limit_bisection((float)Mhalo_min*0.999, (float)Mhalo_max*1.001, astro_params->ALPHA_STAR, astro_params->F_STAR10);
    
    lnMhalo_min = log(Mhalo_min*0.999);
    lnMhalo_max = log(Mhalo_max*1.001);
    dlnMhalo = (lnMhalo_max - lnMhalo_min)/(double)(nbins - 1);
    
    for (i_z=0; i_z<NUM_OF_REDSHIFT_FOR_LF; i_z++) {        
        
        growthf = dicke(z_LF[i_z]);
            
        for (i=0; i<nbins; i++) {
            // generate interpolation arrays
            lnMhalo_param[i] = lnMhalo_min + dlnMhalo*(double)i;
            Mhalo_i = exp(lnMhalo_param[i]);
            
            Fstar = astro_params->F_STAR10*pow(Mhalo_i/1e10,astro_params->ALPHA_STAR);
            if (Fstar > 1.) Fstar = 1;
            
            // parametrization of SFR
            SFRparam = Mhalo_i * cosmo_params->OMb/cosmo_params->OMm * (double)Fstar * (double)(hubble(z_LF[i_z])*SperYR/astro_params->t_STAR); // units of M_solar/year
            
            Muv_param[i] = 51.63 - 2.5*log10(SFRparam*Luv_over_SFR); // UV magnitude
            // except if Muv value is nan or inf, but avoid error put the value as 10.
            if ( isinf(Muv_param[i]) || isnan(Muv_param[i]) ) Muv_param[i] = 10.;
            
            M_uv_z[i + i_z*nbins] = Muv_param[i];
        }
        
        gsl_spline_init(LF_spline, lnMhalo_param, Muv_param, nbins);
        
        lnMhalo_lo = log(Mhalo_min);
        lnMhalo_hi = log(Mhalo_max);
        dlnM = (lnMhalo_hi - lnMhalo_lo)/(double)(nbins - 1);
        
        for (i=0; i<nbins; i++) {
            // calculate luminosity function
            lnMhalo_i = lnMhalo_lo + dlnM*(double)i;
            Mhalo_param[i] = exp(lnMhalo_i);
            
            M_h_z[i + i_z*nbins] = Mhalo_param[i];
            
            Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
            Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);
            
            dMuvdMhalo = (Muv_2 - Muv_1) / (2.*delta_lnMhalo * exp(lnMhalo_i));
            
            if(user_params_ps->HMF==0) {
                log10phi[i + i_z*nbins] = log10( dNdM(z_LF[i_z], exp(lnMhalo_i)) * exp(-(astro_params->M_TURN/Mhalo_param[i])) / fabs(dMuvdMhalo) );
            }
            else if(user_params_ps->HMF==1) {
                log10phi[i + i_z*nbins] = log10( dNdM_st_interp(growthf, exp(lnMhalo_i)) * exp(-(astro_params->M_TURN/Mhalo_param[i])) / fabs(dMuvdMhalo) );
            }
            else if(user_params_ps->HMF==2) {
                log10phi[i + i_z*nbins] = log10( dNdM_WatsonFOF(growthf, exp(lnMhalo_i)) * exp(-(astro_params->M_TURN/Mhalo_param[i])) / fabs(dMuvdMhalo) );
            }
            else if(user_params_ps->HMF==3) {
                log10phi[i + i_z*nbins] = log10( dNdM_WatsonFOF_z(z_LF[i_z], growthf, exp(lnMhalo_i)) * exp(-(astro_params->M_TURN/Mhalo_param[i])) / fabs(dMuvdMhalo) );
            }else{
                LOG_ERROR("HMF should be between 0-3... returning error.");
                return(2);
            }
            if (isinf(log10phi[i + i_z*nbins]) || isnan(log10phi[i + i_z*nbins]) || log10phi[i + i_z*nbins] < -30.) log10phi[i + i_z*nbins] = -30.;
        }
    }


    return(0);
    
}

void initialiseGL_Nion_Xray(int n, float M_TURN, float M_Max){
    float M_Min = M_TURN/50.;
    //calculates the weightings and the positions for Gauss-Legendre quadrature.
    gauleg(log(M_Min),log(M_Max),xi_SFR_Xray,wi_SFR_Xray,n);
}

float dNdM_conditional(float growthf, float M1, float M2, float delta1, float delta2, float sigma2){
    
    float sigma1, dsigmadm,dsigma_val;
    float MassBinLow;
    int MassBin;
    
    MassBin = (int)floor( (M1 - MinMass )*inv_mass_bin_width );
    
    MassBinLow = MinMass + mass_bin_width*(float)MassBin;
    
    sigma1 = Sigma_InterpTable[MassBin] + ( M1 - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    
    dsigma_val = dSigmadm_InterpTable[MassBin] + ( M1 - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
    
    M1 = exp(M1);
    M2 = exp(M2);
    
    sigma1 = sigma1*sigma1;
    sigma2 = sigma2*sigma2;
    
    dsigmadm = -pow(10.,dsigma_val)/(2.0*sigma1); // This is actually sigma1^{2} as calculated above, however, it should just be sigma1. It cancels with the same factor below. Why I have decided to write it like that I don't know!
    
    if((sigma1 > sigma2)) {
        
        return -(( delta1 - delta2 )/growthf)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*growthf*growthf*( sigma1 - sigma2 ) ) ) )/(pow( sigma1 - sigma2, 1.5));
    }
    else if(sigma1==sigma2) {
        
        return -(( delta1 - delta2 )/growthf)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*growthf*growthf*( 1.e-6 ) ) ) )/(pow( 1.e-6, 1.5));
        
    }
    else {
        return 0.;
    }
}

void initialiseGL_Nion(int n, float M_TURN, float M_Max){
    float M_Min = M_TURN/50.;
    //calculates the weightings and the positions for Gauss-Legendre quadrature.
    gauleg(log(M_Min),log(M_Max),xi_SFR,wi_SFR,n);
    
}


double dNion_ConditionallnM(double lnM, void *params) {
    struct parameters_gsl_SFR_con_int_ vals = *(struct parameters_gsl_SFR_con_int_ *)params;
    double M = exp(lnM); // linear scale
    double growthf = vals.gf_obs;
    double M2 = vals.Mval; // natural log scale
    double sigma2 = vals.sigma2;
    double del1 = vals.delta1;
    double del2 = vals.delta2;
    double MassTurnover = vals.Mdrop;
    double Alpha_star = vals.pl_star;
    double Alpha_esc = vals.pl_esc;
    double Fstar10 = vals.frac_star;
    double Fesc10 = vals.frac_esc;
    double Mlim_Fstar = vals.LimitMass_Fstar;
    double Mlim_Fesc = vals.LimitMass_Fesc;
    
    double Fstar,Fesc;
    
    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar10;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1./Fstar10;
    else
        Fstar = pow(M/1e10,Alpha_star);
    
    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc10;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc10;
    else
        Fesc = pow(M/1e10,Alpha_esc);
    
    return M*exp(-MassTurnover/M)*Fstar*Fesc*dNdM_conditional(growthf,log(M),M2,del1,del2,sigma2)/sqrt(2.*PI);
}

double Nion_ConditionalM(double growthf, double M1, double M2, double sigma2, double delta1, double delta2, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc) {
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.01; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    
    struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con = {
        .gf_obs = growthf,
        .Mval = M2,
        .sigma2 = sigma2,
        .delta1 = delta1,
        .delta2 = delta2,
        .Mdrop = MassTurnover,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar10,
        .frac_esc = Fesc10,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc
    };
    
    F.function = &dNion_ConditionallnM;
    F.params = &parameters_gsl_SFR_con;
    lower_limit = M1;
    upper_limit = M2;
    
    gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    if(delta2 > delta1) {
        result = 1.;
        return result;
    }
    else {
        return result;
    }
    
}

float Nion_ConditionallnM_GL(float lnM, struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con){
    float M = exp(lnM);
    float growthf = parameters_gsl_SFR_con.gf_obs;
    float M2 = parameters_gsl_SFR_con.Mval;
    float sigma2 = parameters_gsl_SFR_con.sigma2;
    float del1 = parameters_gsl_SFR_con.delta1;
    float del2 = parameters_gsl_SFR_con.delta2;
    float MassTurnover = parameters_gsl_SFR_con.Mdrop;
    float Alpha_star = parameters_gsl_SFR_con.pl_star;
    float Alpha_esc = parameters_gsl_SFR_con.pl_esc;
    float Fstar10 = parameters_gsl_SFR_con.frac_star;
    float Fesc10 = parameters_gsl_SFR_con.frac_esc;
    float Mlim_Fstar = parameters_gsl_SFR_con.LimitMass_Fstar;
    float Mlim_Fesc = parameters_gsl_SFR_con.LimitMass_Fesc;
    
    float Fstar,Fesc;
    
    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar10;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1./Fstar10;
    else
        Fstar = pow(M/1e10,Alpha_star);
    
    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc10;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc10;
    else
        Fesc = pow(M/1e10,Alpha_esc);
    
    return M*exp(-MassTurnover/M)*Fstar*Fesc*dNdM_conditional(growthf,log(M),M2,del1,del2,sigma2)/sqrt(2.*PI);
}

float GaussLegendreQuad_Nion(int Type, int n, float growthf, float M2, float sigma2, float delta1, float delta2, float MassTurnover, float Alpha_star, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc) {
    //Performs the Gauss-Legendre quadrature.
    int i;
    
    float integrand, x;
    integrand = 0.;
    
    struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con = {
        .gf_obs = growthf,
        .Mval = M2,
        .sigma2 = sigma2,
        .delta1 = delta1,
        .delta2 = delta2,
        .Mdrop = MassTurnover,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar10,
        .frac_esc = Fesc10,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc
    };
    
    if(delta2 > delta1){
        return 1.;
    }
    else{
        for(i=1; i<(n+1); i++){
            if(Type==1) {
                x = xi_SFR_Xray[i];
                integrand += wi_SFR_Xray[i]*Nion_ConditionallnM_GL(x,parameters_gsl_SFR_con);
            }
            if(Type==0) {
                x = xi_SFR[i];
                integrand += wi_SFR[i]*Nion_ConditionallnM_GL(x,parameters_gsl_SFR_con);
            }
        }
        return integrand;
    }
}

int initialise_Nion_General_spline(float z, float min_density, float max_density, float Mmax, float MassTurnover, float Alpha_star, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc){
    
    log10_overdense_spline_SFR = calloc(NSFR_low,sizeof(double));
    log10_Nion_spline = calloc(NSFR_low,sizeof(float));
    
    Overdense_spline_SFR = calloc(NSFR_high,sizeof(float));
    Nion_spline = calloc(NSFR_high,sizeof(float));
    
    float Mmin = MassTurnover/50.;
    double overdense_val, growthf, sigma2;
    double overdense_large_high = Deltac, overdense_large_low = global_params.CRIT_DENS_TRANSITION*0.999;
    double overdense_small_high, overdense_small_low;
    int i;
    
    float ln_10;
    
    if(max_density > global_params.CRIT_DENS_TRANSITION*1.001) {
        overdense_small_high = global_params.CRIT_DENS_TRANSITION*1.001;
    }
    else {
        overdense_small_high = max_density;
    }
    overdense_small_low = min_density;
    
    ln_10 = log(10);
    
    float MassBinLow;
    int MassBin;
    
    growthf = dicke(z);
    
    Mmin = log(Mmin);
    Mmax = log(Mmax);
    
    MassBin = (int)floor( ( Mmax - MinMass )*inv_mass_bin_width );
    
    MassBinLow = MinMass + mass_bin_width*(float)MassBin;
    
    sigma2 = Sigma_InterpTable[MassBin] + ( Mmax - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    
    for (i=0; i<NSFR_low; i++){
        overdense_val = log10(1. + overdense_small_low) + (double)i/((double)NSFR_low-1.)*(log10(1.+overdense_small_high)-log10(1.+overdense_small_low));
        
        log10_overdense_spline_SFR[i] = overdense_val;
        log10_Nion_spline[i] = log10(GaussLegendreQuad_Nion(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,pow(10.,overdense_val)-1.,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc));

        if(isfinite(log10_Nion_spline[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in log10_Nion_spline");
            return(-1);
        }
        
        if(log10_Nion_spline[i] < -40.){
            log10_Nion_spline[i] = -40.;
        }
        
        log10_Nion_spline[i] *= ln_10;

    }
    
    for(i=0;i<NSFR_high;i++) {
        Overdense_spline_SFR[i] = overdense_large_low + (float)i/((float)NSFR_high-1.)*(overdense_large_high - overdense_large_low);
        Nion_spline[i] = Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,Deltac,Overdense_spline_SFR[i],MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc);
        
        if(isfinite(Nion_spline[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in log10_Nion_spline");
            return(-1);
        }

        
        if(Nion_spline[i]<0.) {
            Nion_spline[i]=pow(10.,-40.0);
        }

    }
    
    return(0);
}

int initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, float MassTurn, float Alpha_star, float Alpha_esc, float Fstar10, float Fesc10){
    int i;
    float Mmin = MassTurn/50., Mmax = 1e16;
    float Mlim_Fstar, Mlim_Fesc;
    
    z_val = calloc(Nbin,sizeof(double));
    Nion_z_val = calloc(Nbin,sizeof(double));
    
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fesc = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc10);
    
    for (i=0; i<Nbin; i++){
        z_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin);
        Nion_z_val[i] = Nion_General(z_val[i], MassTurn, Alpha_star, Alpha_esc, Fstar10, Fesc10, Mlim_Fstar, Mlim_Fesc);
        
        if(isfinite(Nion_z_val[i])==0) {
            i = Nbin;
            LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val");
            return(-1);
        }
    }
    
    return(0);
}


int initialise_SFRD_spline(int Nbin, float zmin, float zmax, float MassTurn, float Alpha_star, float Fstar10){
    int i;
    float Mmin = MassTurn/50., Mmax = 1e16;
    float Mlim_Fstar;
    
    z_X_val = calloc(Nbin,sizeof(double));
    SFRD_val = calloc(Nbin,sizeof(double));
    
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    
    for (i=0; i<Nbin; i++){
        z_X_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin);
        SFRD_val[i] = Nion_General(z_X_val[i], MassTurn, Alpha_star, 0., Fstar10, 1.,Mlim_Fstar,0.);
        
        if(isfinite(SFRD_val[i])==0) {
            i = Nbin;
            LOG_ERROR("Detected either an infinite or NaN value in SFRD_val");
            return(-1);
        }
    }
    
    return(0);
}

int initialise_SFRD_Conditional_table(int Nfilter, float min_density[], float max_density[], float growthf[], float R[], float MassTurnover, float Alpha_star, float Fstar10){

    double overdense_val;
    double overdense_large_high = Deltac, overdense_large_low = global_params.CRIT_DENS_TRANSITION;
    double overdense_small_high, overdense_small_low;
    
    overdense_low_table = calloc(NSFR_low,sizeof(double));
    Overdense_high_table = calloc(NSFR_high,sizeof(double));

//    int larger;
//    
//    if(NSFR_low >= NSFR_high) {
//        larger = NSFR_low;
//    }
//    else {
//        larger = NSFR_high;
//    }
    
    float Mmin,Mmax,Mlim_Fstar,sigma2;
    int i,j,k,i_tot;
    
    float ln_10;
    
    ln_10 = log(10);
    
    Mmin = MassTurnover/50;
    Mmax = RtoM(R[Nfilter-1]);
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    
    Mmin = log(Mmin);
    
    for (i=0; i<NSFR_high;i++) {
        Overdense_high_table[i] = overdense_large_low + (float)i/((float)NSFR_high-1.)*(overdense_large_high - overdense_large_low);
    }
    
    float MassBinLow;
    int MassBin;
    
    for (j=0; j < Nfilter; j++) {
        
        Mmax = RtoM(R[j]);
        
        initialiseGL_Nion_Xray(NGL_SFR, MassTurnover, Mmax);
        
        Mmax = log(Mmax);
        MassBin = (int)floor( ( Mmax - MinMass )*inv_mass_bin_width );
        
        MassBinLow = MinMass + mass_bin_width*(float)MassBin;
        
        sigma2 = Sigma_InterpTable[MassBin] + ( Mmax - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
        
        if(min_density[j]*growthf[j] < -1.) {
            overdense_small_low = -1. + global_params.MIN_DENSITY_LOW_LIMIT;
        }
        else {
            overdense_small_low = min_density[j]*growthf[j];
        }
        overdense_small_high = max_density[j]*growthf[j];
        if(overdense_small_high > global_params.CRIT_DENS_TRANSITION) {
            overdense_small_high = global_params.CRIT_DENS_TRANSITION;
        }
            
        for (i=0; i<NSFR_low; i++) {
            overdense_val = log10(1. + overdense_small_low) + (float)i/((float)NSFR_low-1.)*(log10(1.+overdense_small_high)-log10(1.+overdense_small_low));
            overdense_low_table[i] = pow(10.,overdense_val);
        }
        
        for (i=0; i<NSFR_low; i++){
            
            log10_SFRD_z_low_table[j][i] = log10(GaussLegendreQuad_Nion(1,NGL_SFR,growthf[j],Mmax,sigma2,Deltac,overdense_low_table[i]-1.,MassTurnover,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0.));
            
            if(isfinite(log10_SFRD_z_low_table[j][i])==0) {
//                j = Nfilter;
//                i = larger;
                LOG_ERROR("Detected either an infinite or NaN value in log10_SFRD_z_low_table");
                return(-1);
            }
            
            log10_SFRD_z_low_table[j][i] += 10.0;
            log10_SFRD_z_low_table[j][i] *= ln_10;

            
        }
            
        for(i=0;i<NSFR_high;i++) {
            
            SFRD_z_high_table[j][i] = Nion_ConditionalM(growthf[j],Mmin,Mmax,sigma2,Deltac,Overdense_high_table[i],MassTurnover,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0.);
            
            if(isfinite(log10_SFRD_z_low_table[j][i])==0) {
//                j = Nfilter;
//                i = larger;
                LOG_ERROR("Detected either an infinite or NaN value in SFRD_z_high_table");
                return(-1);
            }
            
            SFRD_z_high_table[j][i] *= pow(10., 10.0);
        }
    }
    
    return(0);
    
}
