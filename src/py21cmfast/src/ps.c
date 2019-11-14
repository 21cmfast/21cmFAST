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

#define NMass 300

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

float calibrated_NF_min;
int initialise_photoncons = 1;

double *deltaz, *deltaz_smoothed, *NeutralFractions, *z_Q, *Q_value, *nf_vals, *z_vals;
int N_NFsamples,N_extrapolated, N_analytic, N_calibrated, N_deltaz;


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

//int nbin;
//double *z_Q, *Q_value, *Q_z, *z_value;

double FinalNF_Estimate, FirstNF_Estimate;

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
void free_ps(); /* deallocates the gsl structures from init_ps */
double sigma_z0(double M); //calculates sigma at z=0 (no dicke)
double power_in_k(double k); /* Returns the value of the linear power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */
double TFmdm(double k); //Eisenstein & Hu power spectrum transfer function
void TFset_parameters();

double TF_CLASS(double k, int flag_int, int flag_dv); //transfer function of matter (flag_dv=0) and relative velocities (flag_dv=1) fluctuations from CLASS
double power_in_vcb(double k); /* Returns the value of the DM-b relative velocity power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */


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
  FILE *F;



  char filename[500];
  sprintf(filename,"%s/%s",global_params.external_table_path,CLASS_FILENAME);


  if (flag_int == 0) {// Initialize vectors and read file
    if ( !(F=fopen(filename, "r")) ){
      LOG_ERROR("TF_CLASS: Unable to open file: %s for reading\nAborting\n", filename);
      return -1;
    }

//    for (i=(CLASS_LENGTH-1);i>=0;i--) {
    for (i=0;i<CLASS_LENGTH;i++) {
      fscanf(F, "%e %e %e ", &currk, &currTm, &currTv);
      kclass[i] = currk;
      Tmclass[i] = currTm;//     printf("k=%.1le Tm=%.1le \n", currk,currTm);
      Tvclass_vcb[i] = currTv;//     printf("k=%.1le Tv=%.1le \n", currk,currTv);
      if(kclass[i]<=kclass[i-1] && i>0){
      	printf("WARNING, Tk table not ordered \n");
      	printf("k=%.1le kprev=%.1le \n\n",kclass[i],kclass[i-1]);
      }
    }
    fclose(F);

    // Set up spline table for densities
    acc_density   = gsl_interp_accel_alloc ();
    spline_density  = gsl_spline_alloc (gsl_interp_cspline, CLASS_LENGTH);
    gsl_spline_init(spline_density, kclass, Tmclass, CLASS_LENGTH);


    //Set up spline table for velocities
    acc_vcb   = gsl_interp_accel_alloc ();
    spline_vcb  = gsl_spline_alloc (gsl_interp_cspline, CLASS_LENGTH);
    gsl_spline_init(spline_vcb, kclass, Tvclass_vcb, CLASS_LENGTH);



    return 0;
  }

  if (flag_int == -1) {
    gsl_spline_free (spline_density);
    gsl_interp_accel_free(acc_density);
    gsl_spline_free (spline_vcb);
    gsl_interp_accel_free(acc_vcb);
    return 0;
  }



  if (k > kclass[CLASS_LENGTH-1]) { // k>kmax
    LOG_ERROR("Called TF_CLASS with k=%f, larger than kmax!\n", k);
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
    if (user_params_ps->POWER_SPECTRUM == 0){ // Eisenstein & Hu
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else if (user_params_ps->POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        q = k / (cosmo_params_ps->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else if (user_params_ps->POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_ps->hlittle*gamma);
        bb = 3.0/(cosmo_params_ps->hlittle*gamma);
        cc = 1.7/(cosmo_params_ps->hlittle*gamma);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (user_params_ps->POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        aa = 8.0 / (cosmo_params_ps->hlittle*gamma);
        bb = 4.7 / pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (user_params_ps->POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        aa = 1.7/(cosmo_params_ps->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_ps->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * 19400.0 / pow(1 + aa*k + bb*pow(k, 1.5) + cc*k*k, 2);
    }
    else if (user_params_ps->POWER_SPECTRUM == 5){ // output of CLASS
        T = TF_CLASS(k, 1, 0); //read from z=0 output of CLASS. Note, flag_int = 1 here always, since now we have to have initialized the interpolator for CLASS
  	    p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_ps->POWER_SPECTRUM);
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

    if(user_params_ps->POWER_SPECTRUM == 5){
      kstart = FMAX(1.0e-99/R, KBOT_CLASS);
      kend = FMIN(350.0/R, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at KBOT_CLASS,~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-99/R;
      kend = 350.0/R;
    }




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
    if (user_params_ps->POWER_SPECTRUM == 0){ // Eisenstein & Hu
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
        //p = pow(k, POWER_INDEX - 0.05*log(k/0.05)) * T * T; //running, alpha=0.05
    }
    else if (user_params_ps->POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        q = k / (cosmo_params_ps->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else if (user_params_ps->POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_ps->hlittle*gamma);
        bb = 3.0/(cosmo_params_ps->hlittle*gamma);
        cc = 1.7/(cosmo_params_ps->hlittle*gamma);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (user_params_ps->POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb)/(cosmo_params_ps->OMm));
        aa = 8.0 / (cosmo_params_ps->hlittle*gamma);
        bb = 4.7 / pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (user_params_ps->POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
        aa = 1.7/(cosmo_params_ps->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_ps->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * 19400.0 / pow(1 + aa*k + bb*pow(k, 1.5) + cc*k*k, 2);
    }
    else if (user_params_ps->POWER_SPECTRUM == 5){ // output of CLASS
        T = TF_CLASS(k, 1, 0); //read from z=0 output of CLASS. Note, flag_int = 1 here always, since now we have to have initialized the interpolator for CLASS
  	    p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_ps->POWER_SPECTRUM);
        p = 0;
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
  if (user_params_ps->POWER_SPECTRUM == 5){ // CLASS
    T = TF_CLASS(k, 1, 1); //read from CLASS file. flag_int=1 since we have initialized before, flag_vcb=1 for velocity
	  p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
  }
  else{
    LOG_ERROR("Cannot get P_cb unless using CLASS: %i\n Set USE_RELATIVE_VELOCITIES 0 or use CLASS.\n", user_params_ps->POWER_SPECTRUM);
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

    //we start the interpolator if using CLASS:
  	if (user_params_ps->POWER_SPECTRUM == 5){
  		TF_CLASS(1.0, 0, 0);
  	}


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

    if(user_params_ps->POWER_SPECTRUM == 5){
      kstart = FMAX(1.0e-99/R, KBOT_CLASS);
      kend = FMIN(350.0/R, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at KBOT_CLASS,~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-99/R;
      kend = 350.0/R;
    }

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




//function to free arrays related to the power spectrum
void free_ps(){

	//we free the PS interpolator if using CLASS:
	if (user_params_ps->POWER_SPECTRUM == 5){
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
    if (user_params_ps->POWER_SPECTRUM == 0){ // Eisenstein & Hu ApJ, 1999, 511, 5
        T = TFmdm(k);
        // check if we should cuttoff power spectrum according to Bode et al. 2000 transfer function
        if (global_params.P_CUTOFF) T *= pow(1 + pow(BODE_e*k*R_CUTOFF, 2*BODE_v), -BODE_n/BODE_v);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
        //p = pow(k, POWER_INDEX - 0.05*log(k/0.05)) * T * T; //running, alpha=0.05
    }
    else if (user_params_ps->POWER_SPECTRUM == 1){ // BBKS
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb)/(cosmo_params_ps->OMm));
        q = k / (cosmo_params_ps->hlittle*gamma);
        T = (log(1.0+2.34*q)/(2.34*q)) *
        pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else if (user_params_ps->POWER_SPECTRUM == 2){ // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
        gamma = 0.25;
        aa = 6.4/(cosmo_params_ps->hlittle*gamma);
        bb = 3.0/(cosmo_params_ps->hlittle*gamma);
        cc = 1.7/(cosmo_params_ps->hlittle*gamma);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow( 1+pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 2.0/1.13 );
    }
    else if (user_params_ps->POWER_SPECTRUM == 3){ // Peebles, pg. 626
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb)/(cosmo_params_ps->OMm));
        aa = 8.0 / (cosmo_params_ps->hlittle*gamma);
        bb = 4.7 / (cosmo_params_ps->hlittle*gamma);
        p = pow(k, cosmo_params_ps->POWER_INDEX) / pow(1 + aa*k + bb*k*k, 2);
    }
    else if (user_params_ps->POWER_SPECTRUM == 4){ // White, SDM and Frenk, CS, 1991, 379, 52
        gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb)/(cosmo_params_ps->OMm));
        aa = 1.7/(cosmo_params_ps->hlittle*gamma);
        bb = 9.0/pow(cosmo_params_ps->hlittle*gamma, 1.5);
        cc = 1.0/pow(cosmo_params_ps->hlittle*gamma, 2);
        p = pow(k, cosmo_params_ps->POWER_INDEX) * 19400.0 / pow(1 + aa*k + pow(bb*k, 1.5) + cc*k*k, 2);
    }
    else if (user_params_ps->POWER_SPECTRUM == 5){ // JBM: CLASS
      T = TF_CLASS(k, 1, 0); //read from z=0 output of CLASS
	    p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
	  }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_ps->POWER_SPECTRUM);
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
    if(user_params_ps->POWER_SPECTRUM == 5){
      kstart = FMAX(1.0e-99/R, KBOT_CLASS);
      kend = FMIN(350.0/R, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at KBOT_CLASS,~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-99/R;
      kend = 350.0/R;
    }

    lower_limit = kstart;//log(kstart);
    upper_limit = kend;//log(kend);


    if (user_params_ps->POWER_SPECTRUM == 5){ // for CLASS we do not need to renormalize the sigma integral.
      d2fact=1.0;
    }
    else {
      d2fact = M*10000/sigma_z0(M);
    }


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
    int status;

    if(user_params_ps->HMF<4 && user_params_ps->HMF>-1) {

        F.function = &dNion_General;
        F.params = &parameters_gsl_SFR;

        lower_limit = log(M_Min);
        upper_limit = log(FMAX(1e16, M_Min*100));

        gsl_set_error_handler_off();

        status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);
        if(status!=0) {
            printf("(function argument): %e %e %e %e %e\n",lower_limit,upper_limit,rel_tol,result,error);
            printf("data: %e %e %e %e %e %e %e %e %e\n",z,growthf,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc);
            exit(-1);
        }
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

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        printf("(function argument): %e %e %e %e %e\n",lower_limit,upper_limit,rel_tol,result,error);
        printf("data: %e %e %e %e %e %e %e %e %e %e %e %e %e\n",growthf,M1,M2,sigma2,delta1,delta2,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc);
        exit(-1);
    }

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
        log10_Nion_spline[i] = GaussLegendreQuad_Nion(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,pow(10.,overdense_val)-1.,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc);
        if(fabs(log10_Nion_spline[i]) < 1e-38) {
            log10_Nion_spline[i] = 1e-38;
        }
        log10_Nion_spline[i] = log10(log10_Nion_spline[i]);

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

            log10_SFRD_z_low_table[j][i] = GaussLegendreQuad_Nion(1,NGL_SFR,growthf[j],Mmax,sigma2,Deltac,overdense_low_table[i]-1.,MassTurnover,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0.);
            if(fabs(log10_SFRD_z_low_table[j][i]) < 1e-38) {
                log10_SFRD_z_low_table[j][i] = 1e-38;
            }
            log10_SFRD_z_low_table[j][i] = log10(log10_SFRD_z_low_table[j][i]);

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
            if(isfinite(SFRD_z_high_table[j][i])==0) {
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

//   Set up interpolation table for the volume filling factor, Q, at a given redshift z and redshift at a given Q.
int InitialisePhotonCons(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                         struct AstroParams *astro_params, struct FlagOptions *flag_options)
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    //     To solve differentail equation, uses Euler's method.
    //     NOTE:
    //     (1) With the fiducial parameter set,
    //	    when the Q value is < 0.9, the difference is less than 5% compared with accurate calculation.
    //	    When Q ~ 0.98, the difference is ~25%. To increase accuracy one can reduce the step size 'da', but it will increase computing time.
    //     (2) With the fiducial parameter set,
    //     the difference for the redshift where the reionization end (Q = 1) is ~0.2 % compared with accurate calculation.
    float ION_EFF_FACTOR,M_MIN,M_MIN_z0,M_MIN_z1,Mlim_Fstar, Mlim_Fesc;
    double a_start = 0.03, a_end = 1./(1. + 5.0); // Scale factors of 0.03 and 0.17 correspond to redshifts of ~32 and ~5.0, respectively.
    double C_HII = 3., T_0 = 2e4;
    double reduce_ratio = 1.003;
    double Q0,Q1,Nion0,Nion1,Trec,da,a,z0,z1,zi,dadt,ans,delta_a,zi_prev,Q1_prev;
    double *z_arr,*Q_arr;
    int Nmax = 2000; // This is the number of step, enough with 'da = 2e-3'. If 'da' is reduced, this number should be checked.
    int cnt, nbin, i, istart;
    int fail_condition, not_mono_increasing, num_fails;

    z_arr = calloc(Nmax,sizeof(double));
    Q_arr = calloc(Nmax,sizeof(double));

    //set the minimum source mass
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;

        M_MIN = astro_params->M_TURN/50.;
        Mlim_Fstar = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_STAR, astro_params->F_STAR10);
        Mlim_Fesc = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_ESC, astro_params->F_ESC10);

        initialiseSigmaMInterpTable(M_MIN,1e20);
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
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                Nion0 = ION_EFF_FACTOR*Nion_General(z0, astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc);
                Nion1 = ION_EFF_FACTOR*Nion_General(z1, astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc);
            }
            else {

                //set the minimum source mass
                if (astro_params->ION_Tvir_MIN < 9.99999e3) { // neutral IGM
                    M_MIN_z0 = TtoM(z0, astro_params->ION_Tvir_MIN, 1.22);
                    M_MIN_z1 = TtoM(z1, astro_params->ION_Tvir_MIN, 1.22);
                }
                else { // ionized IGM
                    M_MIN_z0 = TtoM(z0, astro_params->ION_Tvir_MIN, 0.6);
                    M_MIN_z1 = TtoM(z1, astro_params->ION_Tvir_MIN, 0.6);
                }

                if(M_MIN_z0 < M_MIN_z1) {
                    initialiseSigmaMInterpTable(M_MIN_z0,1e20);
                }
                else {
                    initialiseSigmaMInterpTable(M_MIN_z1,1e20);
                }

                Nion0 = ION_EFF_FACTOR*FgtrM_General(z0,M_MIN_z0);
                Nion1 = ION_EFF_FACTOR*FgtrM_General(z1,M_MIN_z1);
            }

            // With scale factor a, the above equation is written as dQ/da = n_{ion}/da - Q/t_{rec}*(dt/da)
            if (!global_params.RecombPhotonCons) {
                Q1 = Q0 + ((Nion0-Nion1)/2/delta_a)*da; // No Recombination
            }
            else {
                dadt = Ho*sqrt(cosmo_params_ps->OMm/a + global_params.OMr/a/a + cosmo_params_ps->OMl*a*a); // da/dt = Ho*a*sqrt(OMm/a^3 + OMr/a^4 + OMl)
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
            if (Q1 >= 1.0) break; // if fully ionized, stop here.
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
                printf("Failed too many times. Exit out!\n");
                exit(-1);
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
//    Q_at_z_spline = gsl_spline_alloc (gsl_interp_linear, nbin);

    for (i=0; i<nbin; i++){
        z_Q[i] = z_arr[cnt-i];
        Q_value[i] = Q_arr[cnt-i];
    }

    gsl_spline_init(Q_at_z_spline, z_Q, Q_value, nbin);
    Zmin = z_Q[0];
    Zmax = z_Q[nbin-1];
    Qmin = Q_value[nbin-1];
    Qmax = Q_value[0];

    // initialise interploation z as a function of Q
    double *Q_z = calloc(nbin,sizeof(double));
    double *z_value = calloc(nbin,sizeof(double));

    z_at_Q_spline_acc = gsl_interp_accel_alloc ();
//    z_at_Q_spline = gsl_spline_alloc (gsl_interp_cspline, nbin);
    z_at_Q_spline = gsl_spline_alloc (gsl_interp_linear, nbin);
    for (i=0; i<nbin; i++){
        Q_z[i] = Q_value[nbin-1-i];
        z_value[i] = z_Q[nbin-1-i];
    }

    gsl_spline_init(z_at_Q_spline, Q_z, z_value, nbin);

    free(z_arr);
    free(Q_arr);

    photon_cons_inited = true;
    return(0);
}

// Function to construct the spline for the calibration curve of the photon non-conservation
int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline)
{

    if(xH_estimate[NSpline-1] > 0.0 && xH_estimate[NSpline-2] > 0.0 && xH_estimate[NSpline-3] > 0.0 && xH_estimate[0] <= global_params.PhotonConsStart) {
        initialise_NFHistory_spline(z_estimate,xH_estimate,NSpline);
    }

    return(0);
}

// Function callable from Python to know at which redshift to start sampling the calibration curve (to minimise function calls)
double ComputeZstart_PhotonCons() {

    double temp;

    if((1.-global_params.PhotonConsStart) > Qmax) {
        // It is possible that reionisation never even starts
        // Just need to arbitrarily set a high redshift to perform the algorithm
        temp = 20.;
    }
    else {
        z_at_Q(1. - global_params.PhotonConsStart,&(temp));
    }

    return(temp);
}


void determine_deltaz_for_photoncons() {

    int i, j, increasing_val, counter, smoothing_int;
    double temp;
    float z_cal, z_analytic, NF_sample, returned_value, NF_sample_min, gradient_analytic, z_analytic_at_endpoint, const_offset, z_analytic_2, smoothing_width;
    float bin_width, delta_NF, val1, val2, extrapolated_value;

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
    if(calibrated_NF_min < global_params.PhotonConsEnd) {
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

            val1 = deltaz[0];
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

    gsl_spline_init(deltaz_spline_for_photoncons, NeutralFractions, deltaz, N_NFsamples + N_extrapolated + 1);

}


float adjust_redshifts_for_photoncons(float *redshift, float *stored_redshift, float *absolute_delta_z) {

    int i, new_counter;
    double temp;
    float required_NF, adjusted_redshift, future_z, gradient_extrapolation, const_extrapolation, temp_redshift, check_required_NF;

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
        if(initialise_photoncons) {
            determine_deltaz_for_photoncons();
            initialise_photoncons = 0;
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
        fprintf(stderr,"The minimum value of Q is %.4e\n Aborting...\n",Qmin);
    }
    else if (Q > Qmax) {
        fprintf(stderr,"The maximum value of Q is %.4e\n Reionization ends at ~%.4f\n Aborting...\n",Qmax,Zmin);
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

    gsl_spline_init(NFHistory_spline, nf_vals, z_vals, (counter+1));

    z_NFHistory_spline_acc = gsl_interp_accel_alloc ();
//    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_cspline, (counter+1));
    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_linear, (counter+1));

    gsl_spline_init(z_NFHistory_spline, z_vals, nf_vals, (counter+1));
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

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
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
