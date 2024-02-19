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

#define NGL_INT 100 // 100

#define NR_END 1
#define FREE_ARG char*

#define MM 7
#define NSTACK 50

#define EPS2 3.0e-11

//     Luv/SFR = 1 / 1.15 x 10^-28 [M_solar yr^-1/erg s^-1 Hz^-1]
//     G. Sun and S. R. Furlanetto (2016) MNRAS, 417, 33
#define Luv_over_SFR (double)(1./1.15/1e-28)

#define delta_lnMhalo (double)(5e-6)
#define Mhalo_min (double)(1e6)
#define Mhalo_max (double)(1e16)

//These globals hold values relevant for the photon conservation (z-shift) model
float calibrated_NF_min;
double *deltaz, *deltaz_smoothed, *NeutralFractions, *z_Q, *Q_value, *nf_vals, *z_vals;
int N_NFsamples,N_extrapolated, N_analytic, N_calibrated, N_deltaz;
double FinalNF_Estimate, FirstNF_Estimate;

bool initialised_ComputeLF = false;

gsl_interp_accel *LF_spline_acc;
gsl_spline *LF_spline;

gsl_interp_accel *deriv_spline_acc;
gsl_spline *deriv_spline;

struct CosmoParams *cosmo_params_ps;
struct UserParams *user_params_ps;
struct FlagOptions *flag_options_ps;

//These globals hold values initialised in init_ps() and used throughout the rest of the file
double sigma_norm, theta_cmb, omhh, z_equality, y_d, sound_horizon, alpha_nu, f_nu, f_baryon, beta_c, d2fact, R_CUTOFF, DEL_CURR, SIG_CURR;

double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias);

//Sigma interpolation tables are defined here instead of interp_tables.c since they are used in their construction
struct RGTable1D_f Sigma_InterpTable = {.allocated = false,};
struct RGTable1D_f dSigmasqdm_InterpTable = {.allocated = false,};

//These arrays hold the points and weights for the Gauss-Legendre integration routine
//(JD) Since these were always malloc'd one at a time with fixed length ~100, I've changed them to fixed-length arrays
float xi_GL[NGL_INT+1], wi_GL[NGL_INT+1], GL_limit[2];

//These globals are used for the LF calculation
double *lnMhalo_param, *Muv_param, *Mhalo_param;
double *log10phi, *M_uv_z, *M_h_z;
double *lnMhalo_param_MINI, *Muv_param_MINI, *Mhalo_param_MINI;
double *log10phi_MINI; *M_uv_z_MINI, *M_h_z_MINI;
double *deriv, *lnM_temp, *deriv_temp;

void initialiseSigmaMInterpTable(float M_Min, float M_Max);
void freeSigmaMInterpTable();

double EvaluateSigma(double lnM, int calc_ds, double *dsigmadm);

float Mass_limit (float logM, float PL, float FRAC);
void bisection(float *x, float xlow, float xup, int *iter);
float Mass_limit_bisection(float Mmin, float Mmax, float PL, float FRAC);

double sheth_delc(double del, double sig);
double dNdM_conditional_EPS(double growthf, double lnM, double delta_cond, double sigma_cond);

double Nion_ConditionalM(double growthf, double M1, double M2, double sigma2, double delta2, double MassTurnover,
                        double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar,
                        double Mlim_Fesc, int method);

double Nion_ConditionalM_MINI(double growthf, double M1, double M2, double sigma2, double delta2, double MassTurnover,
                            double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar10,
                            double Fesc10, double Mlim_Fstar, double Mlim_Fesc, int method);

//JBM: Exact integral for power-law indices non zero (for zero it's erfc)
double Fcollapprox (double numin, double beta);

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


//Parameters used for gsl integral on the mass function
struct parameters_gsl_MF_integrals{
    //parameters for all MF integrals
    double redshift;
    double growthf;
    int HMF;

    //Conditional parameters
    double sigma_cond;
    double delta;

    //SFR additions
    double Mturn;
    double f_star_norm;
    double alpha_star;
    double Mlim_star;

    //Nion additions
    double f_esc_norm;
    double alpha_esc;
    double Mlim_esc;

    //Minihalo additions
    double Mturn_upper;
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
double FgtrM_wsigma(double z, double sig);
double FgtrM_st(double z, double M);
double FgtrM_Watson(double growthf, double M);
double FgtrM_Watson_z(double z, double growthf, double M);
double FgtrM_General(double z, double M, int method);

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
        if(user_params_ps->USE_RELATIVE_VELOCITIES) { //jbm:Add average relvel suppression
          p *= 1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM)); //for v=vrms
        }
    }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_ps->POWER_SPECTRUM);
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

    if(user_params_ps->POWER_SPECTRUM == 5){
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
        if(user_params_ps->USE_RELATIVE_VELOCITIES) { //jbm:Add average relvel suppression
          p *= 1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM)); //for v=vrms
        }
    }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_ps->POWER_SPECTRUM);
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
    if (user_params_ps->POWER_SPECTRUM == 5){ // CLASS
        T = TF_CLASS(k, 1, 1); //read from CLASS file. flag_int=1 since we have initialized before, flag_vcb=1 for velocity
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else{
        LOG_ERROR("Cannot get P_cb unless using CLASS: %i\n Set USE_RELATIVE_VELOCITIES 0 or use CLASS.\n", user_params_ps->POWER_SPECTRUM);
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
    if (user_params_ps->POWER_SPECTRUM == 5){
        LOG_DEBUG("Setting CLASS Transfer Function inits.");
        TF_CLASS(1.0, 0, 0);
    }

    // Set cuttoff scale for WDM (eq. 4 in Barkana et al. 2001) in comoving Mpc
    R_CUTOFF = 0.201*pow((cosmo_params_ps->OMm-cosmo_params_ps->OMb)*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle/0.15, 0.15)*pow(global_params.g_x/1.5, -0.29)*pow(global_params.M_WDM, -1.15);

    omhh = cosmo_params_ps->OMm*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle;
    theta_cmb = T_cmb / 2.7;

    // Translate Parameters into forms GLOBALVARIABLES form
    f_nu = global_params.OMn/cosmo_params_ps->OMm;
    f_baryon = cosmo_params_ps->OMb/cosmo_params_ps->OMm;
    if (f_nu < TINY) f_nu = 1e-10;
    if (f_baryon < TINY) f_baryon = 1e-10;

    TFset_parameters();

    sigma_norm = -1;

    double Radius_8;
    Radius_8 = 8.0/cosmo_params_ps->hlittle;

    if(user_params_ps->POWER_SPECTRUM == 5){
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

    sigma_norm = cosmo_params_ps->SIGMA_8/sqrt(result); //takes care of volume factor
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
        if(user_params_ps->USE_RELATIVE_VELOCITIES) { //jbm:Add average relvel suppression
          p *= 1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM)); //for v=vrms
        }
      }
    else{
        LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_ps->POWER_SPECTRUM);
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
        drdm = 1.0 / (4.0*PI * cosmo_params_ps->OMm*RHOcrit * Radius*Radius);
    }
    else if (global_params.FILTER == 1){ // gaussian of width 1/R
        w = pow(E, -kR*kR/2.0);
        dwdr = - k*kR * w;
        drdm = 1.0 / (pow(2*PI, 1.5) * cosmo_params_ps->OMm*RHOcrit * 3*Radius*Radius);
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
    if(user_params_ps->POWER_SPECTRUM == 5){
      kstart = fmax(1.0e-99/Radius, KBOT_CLASS);
      kend = fmin(350.0/Radius, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at KBOT_CLASS,~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-99/Radius;
      kend = 350.0/Radius;
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
double sheth_delc(double del, double sig){
    return sqrt(SHETH_a)*del*(1. + global_params.SHETH_b*pow(sig*sig/(SHETH_a*del*del), global_params.SHETH_c));
}

/*DexM uses a fit to this barrier to acheive MF similar to ST, Here I use the fixed version for the sampler*/
double sheth_delc_fixed(double del, double sig){
    double a = 0.707;
    double sheth_b = 0.485;
    double sheth_c = 0.615;
    return sqrt(a)*del*(1. + sheth_b*pow(sig*sig/(a*del*del), sheth_c));
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

    sigma = EvaluateSigma(lnM,1,&dsigmadm);
    sigma_inv = 1/(sigma);
    dsigmadm = dsigmadm * 0.5 * sigma_inv;

    nu = DELTAC_DELOS*sigma_inv/growthf;

    dfdnu = coeff_nu*pow(nu,index_nu)*exp(exp_factor*nu*nu);
    dfdM = dfdnu * fabs(dsigmadm) * sigma_inv;

    //NOTE: unlike the other UMFs this is dNdlogM
    //NOTE: dfdM == constants*dNdlnM
    return dfdM*cosmo_params_ps->OMm*RHOcrit;
}

double dNdlnM_conditional_Delos(double growthf, double lnM, double delta_cond, double sigma_cond){
    double result,dfdnu,dsigmadm,sigma,sigdiff_inv,dfdM,sigma_inv;
    double nu;
    //hardcoded for now
    const double coeff_nu = 0.519;
    const double index_nu = 0.582;
    const double exp_factor = -0.469;

    sigma = EvaluateSigma(lnM,1,&dsigmadm);
    if(sigma < sigma_cond) return 0.;
    sigdiff_inv = sigma == sigma_cond ? 1e6 : 1/(sigma*sigma - sigma_cond*sigma_cond);

    sigma_inv = 1/sigma;
    dsigmadm = dsigmadm * (0.5*sigma_inv); //d(s^2)/dm z0 to dsdm

    nu = (DELTAC_DELOS - delta_cond)*sqrt(sigdiff_inv)/growthf;

    dfdnu = coeff_nu*pow(nu,index_nu)*exp(exp_factor*nu*nu);
    dfdM = dfdnu * fabs(dsigmadm) * sigma_inv * sigma * sigma * sigdiff_inv;

    //NOTE: like the other CMFs this is dNdlogM and leaves out
    //   the (cosmo_params_ps->OMm)*RHOcrit
    //NOTE: dfdM == constants*dNdlnM
    // LOG_ULTRA_DEBUG("M = %.3e Barrier = %.3f || dndlnM= %.6e",exp(lnM),DELTAC_DELOS,dfdM);
    return dfdM;
}

//Sheth Tormen 2002 fit for the CMF, while the moving barrier does not allow for a simple rescaling, it has been found
//That a taylor expansion of the barrier shape around the point of interest well approximates the simulations
//TODO: Count the growth factors needed in each term, also move to ps.c
double st_taylor_factor(double sig, double sig_cond, double delta_cond, double growthf){
    int i;
    // double a = SHETH_a;
    double a = 0.707;
    double alpha = 0.615; //fixed instead of global_params.SHETH_c bc of DexM corrections
    double beta = 0.485; //fixed instead of global_params.SHETH_b

    double del = Deltac/growthf;

    double sigsq = sig*sig;
    double sigcsq = sig_cond*sig_cond;
    //See note below
    double sigdiff = sig == sig_cond ? 1e-6 : sigsq - sigcsq;
    double dn_const = sqrt(a)*del*beta*pow(a*del*del,-alpha);

    //define arrays of factors to save time and math calls
    int n_fac[6] = {1,1,2,6,24,120};
    double a_fac[6];
    double s_fac[6];
    a_fac[0] = 1;
    s_fac[0] = 1;
    for(i=1;i<=5;i++){
        a_fac[i] = a_fac[i-1] * (alpha-i+1);
        s_fac[i] = s_fac[i-1] * (-sigdiff);
    }

    double result = 0.;
    //Taylor expansion of the x^a part around (sigsq - sigcondsq) (summing small to large)
    for(i=5;i>=1;i--){
        result += s_fac[i]/n_fac[i] * pow(sigsq,alpha-i)*a_fac[i];
        // LOG_ULTRA_DEBUG("%d term %.2e",i,result);
    }
    result *= dn_const;
    //add the constant terms from the 0th derivative of the barrier (condition delta independent of halo sigma)
    // result += sqrt(a)*delta_crit - delta_cond;

    return result;
}

//CMF Corresponding to the Sheth Mo Tormen HMF, here we assume that we are passing the correct delta2,
//      which is the condition delta, the barrier delta1 is set by the mass, so it is passed usually as Deltac
//NOTE: Currently broken and I don't know why
double dNdM_conditional_ST(double growthf, double lnM, double delta_cond, double sigma_cond){
    double sigma1, dsigmasqdm, Barrier, factor, sigdiff_inv, result;
    double delta_0 = delta_cond / growthf;
    sigma1 = EvaluateSigma(lnM,1,&dsigmasqdm); //WARNING: THE SIGMA TABLE IS STILL SINGLE PRECISION
    if(sigma1 < sigma_cond) return 0.;

    Barrier = sheth_delc_fixed(Deltac/growthf,sigma1);
    factor = st_taylor_factor(sigma1,sigma_cond,delta_0,growthf) + (Barrier - delta_0);

    sigdiff_inv = sigma1 == sigma_cond ? 1e6 : 1/(sigma1*sigma1 - sigma_cond*sigma_cond);

    result = -dsigmasqdm*factor*pow(sigdiff_inv,1.5)*exp(-(Barrier - delta_0)*(Barrier - delta_0)*0.5*(sigdiff_inv))/sqrt(2.*PI);
    // LOG_ULTRA_DEBUG("M = %.3e T = %.3e Barrier = %.3f || dndlnM= %.6e",exp(lnM),factor,Barrier,result);
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
double dNdM_st(double growthf, double M){
    double sigma, dsigmadm, nuhat;

    float MassBinLow;
    int MassBin;

    sigma = EvaluateSigma(log(M),1,&dsigmadm);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    nuhat = sqrt(SHETH_a) * Deltac / sigma;

    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * sqrt(2./PI)*SHETH_A * (1+ pow(nuhat, -2*SHETH_p)) * nuhat * pow(E, -nuhat*nuhat/2.0);
}

//Conditional Extended Press-Schechter Mass function, with constant barrier delta=1.682 and sharp-k window function
double dNdM_conditional_EPS(double growthf, double lnM, double delta_cond, double sigma_cond){
    double sigma1, dsigmasqdm, sigdiff_inv, del;
    sigma1 = EvaluateSigma(lnM,1,&dsigmasqdm); //WARNING: THE SIGMA TABLE IS STILL SINGLE PRECISION

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
double dNdM(double growthf, double M){
    double sigma, dsigmadm;
    float MassBinLow;
    int MassBin;

    sigma = EvaluateSigma(log(M),1,&dsigmadm);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * sqrt(2/PI) * (Deltac/(sigma*sigma)) * dsigmadm * pow(E, -(Deltac*Deltac)/(2*sigma*sigma));
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
double dNdM_WatsonFOF(double growthf, double M){

    double sigma, dsigmadm, f_sigma;

    float MassBinLow;
    int MassBin;

    sigma = EvaluateSigma(log(M),1,&dsigmadm);

    sigma = sigma * growthf;
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

    sigma = EvaluateSigma(log(M),1,&dsigmadm);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    Omega_m_z = (cosmo_params_ps->OMm)*pow(1.+z,3.) / ( (cosmo_params_ps->OMl) + (cosmo_params_ps->OMm)*pow(1.+z,3.) + (global_params.OMr)*pow(1.+z,4.) );

    A_z = Omega_m_z * ( Watson_A_z_1 * pow(1. + z, Watson_A_z_2 ) + Watson_A_z_3 );
    alpha_z = Omega_m_z * ( Watson_alpha_z_1 * pow(1.+z, Watson_alpha_z_2 ) + Watson_alpha_z_3 );
    beta_z = Omega_m_z * ( Watson_beta_z_1 * pow(1.+z, Watson_beta_z_2 ) + Watson_beta_z_3 );

    f_sigma = A_z * ( pow(beta_z/sigma, alpha_z) + 1. ) * exp( - Watson_gamma_z/(sigma*sigma) );

    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * f_sigma;
}

///////MASS FUNCTION INTEGRANDS BELOW//////

//gets the fraction (in units of 1/normalisation at 1e10)
double get_frac_limit(double M, double norm, double alpha, double limit, bool mini){
    double pivot = mini ? 1e7 : 1e10;
    if ((alpha > 0. && M > limit) || (alpha < 0. && M < limit))
        return 1/norm;

    return pow(M/1e10,alpha);
}

double cmf_function(double growthf, double lnM, double delta, double sigma, int HMF){
    //dNdlnM = dfcoll/dM * M / M * constants
    if(HMF==0) {
        return dNdM_conditional_EPS(growthf,lnM,delta,sigma);
    }
    if(HMF==1) {
        return dNdM_conditional_ST(growthf,lnM,delta,sigma);
    }
    if(HMF==4) {
        return dNdlnM_conditional_Delos(growthf,lnM,delta,sigma);
    }
    //NOTE: Normalisation scaling is currently applied outside the integral, per condition
    //This will be the rescaled EPS CMF,
    //TODO: put rescaling options here (normalised EPS, rescaled EPS, local/global scalings of UMFs from Tramonte+17)
    //Filter CMF type by CMF_MODE parameter (==1 for set CMF, ==2 for resnormalised, ==3 for rescaled EPS, ==4 for Tramonte local, ==5 for Tramonte global etc)
    return dNdM_conditional_EPS(growthf,lnM,delta,sigma);

}

double c_mf_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double growthf = params.growthf;
    double delta = params.delta; //the condition delta
    double sigma2 = params.sigma_cond;
    int HMF = params.HMF;

    return cmf_function(growthf,lnM,delta,sigma2,HMF);
}

double c_fcoll_integrand(double lnM, void *param_struct){
    return exp(lnM) * c_mf_integrand(lnM,param_struct);
}

double c_nion_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double M = exp(lnM);

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,false);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,false);

    return M * Fstar * Fesc * exp(-M_turn_lower/M) * c_mf_integrand(lnM,param_struct);
}

//The reason this is separated from the above is the second exponent
double c_nion_integrand_mini(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double M_turn_upper = params.Mturn_upper;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double M = exp(lnM);

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,true);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,true);

    return M * Fstar * Fesc * exp(-M_turn_lower/M) * exp(-M/M_turn_upper) * c_mf_integrand(lnM,param_struct);
}

double umf_function(double growthf, double lnM, double z, int HMF){
    double M_exp = exp(lnM);

    //most of the UMFs are defined with M, but we integrate over lnM
    //TODO: rewrite the old MFs as dNdlnM
    //NOTE: HMF > 4 or < 0 gets caught earlier, so unless some strange change is made this is fine
    if(HMF==0) {
        return dNdM(growthf, M_exp) * M_exp;
    }
    if(HMF==1) {
        return dNdM_st(growthf, M_exp) * M_exp;
    }
    if(HMF==2) {
        return dNdM_WatsonFOF(growthf, M_exp) * M_exp;
    }
    if(HMF==3) {
        return dNdM_WatsonFOF_z(z, growthf, M_exp) * M_exp;
    }
    if(HMF==4) {
        return dNdlnM_Delos(growthf, lnM); //NOTE: dNdlogM
    }
    else{
        LOG_ERROR("Invalid HMF %d",HMF);
        Throw(ValueError);
    }
}

double u_mf_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double mf, m_factor;
    double growthf = params.growthf;
    double z = params.redshift;
    int HMF = params.HMF;

    return umf_function(growthf,lnM,z,HMF);
}

double u_fcoll_integrand(double lnM, void *param_struct){
    return exp(lnM) * u_mf_integrand(lnM,param_struct);
}

double u_nion_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn = params.Mturn;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double M = exp(lnM);

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,false);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,false);

    return M * Fstar * Fesc * exp(-M_turn/M) * u_mf_integrand(lnM,param_struct);
}

//The reason this is separated from the above is the second exponent
double u_nion_integrand_mini(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double M_turn_upper = params.Mturn_upper;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double M = exp(lnM);

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,false);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,false);

    return M * Fstar * Fesc * exp(-M_turn_lower/M) * exp(-M/M_turn_upper) * u_mf_integrand(lnM,param_struct);
}

///// INTEGRATION ROUTINES BELOW /////

//TODO: make type enum for clarity (but cffi doesn't seem to like enum in 21cmFAST.h)
//NOTE: SFR is obtained from nion with alpha_esc==0 and f_esc==1
//Currently the scheme is to use negative numbers for conditionals, and (1,2,3,4) for (number,mass,n_ion,n_ion_mini)
double (*get_integrand_function(int type))(double,void*){
    if(type==1)
        return &u_mf_integrand; //Condtional mass function integral
    if(type==2)
        return &u_fcoll_integrand; //Collapsed fraction integral
    if(type==3)
        return &u_nion_integrand;
    if(type==4)
        return &u_nion_integrand_mini;
    if(type==-1)
        return &c_mf_integrand; //Unconditional mass function integral
    if(type==-2)
        return &c_fcoll_integrand; //Unconditional mass function integral
    if(type==-3)
        return &c_nion_integrand;
    if(type==-4)
        return &c_nion_integrand_mini;

    LOG_ERROR("Invalid type %d for MF integral");
    Throw(ValueError);
}

//Integral of a CMF or UMF
//In future all MF integrals will go through here, simply selecting the integrand function from a switch
double IntegratedNdM_QAG(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    // double rel_tol = FRACT_FLOAT_ERR*128; //<- relative tolerance
    double rel_tol = 1e-4; //<- relative tolerance
    int w_size = 1000;
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (w_size);

    int status;
    F.function = get_integrand_function(type);
    F.params = &params;
    lower_limit = lnM_lo;
    upper_limit = lnM_hi;

    gsl_set_error_handler_off();
    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         w_size, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%.3e upper_limit=%.3e (%.3e) rel_tol=%.3e result=%.3e error=%.3e",lower_limit,upper_limit,exp(upper_limit),rel_tol,result,error);
        LOG_ERROR("data: growthf=%.3e delta=%.3e sigma2=%.3e HMF=%d type=%d ",params.growthf,params.delta,params.sigma_cond,params.HMF,type);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return result;
}

//calculates the weightings and the positions for any Gauss-Legendre quadrature.
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

//Specific initialistion for the global arrays
void initialise_GL(int n, float lnM_Min, float lnM_Max){
    gauleg(lnM_Min,lnM_Max,xi_GL,wi_GL,n);
    GL_limit[0] = lnM_Min;
    GL_limit[1] = lnM_Max;
}

//actually perform the GL integration
//NOTE: that the lnM limits are not used
double IntegratedNdM_GL(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type){
    int i;
    double integral = 0;
    if((float)lnM_lo != (float)GL_limit[0] || (float)lnM_hi != (float)GL_limit[1]){
        LOG_ERROR("Integral limits [%.8e %.8e] do not match Gauss Legendre limits [%.8e %.8e]!",lnM_lo,lnM_hi,GL_limit[0],GL_limit[1]);
        Throw(TableGenerationError);
    }

    for(i=1; i<(NGL_INT+1); i++){
        integral += wi_GL[i]*(get_integrand_function(type))(xi_GL[i],&params);
    }

    return integral;
}

#include <gsl/gsl_sf_gamma.h>
//JBM: Integral of a power-law times exponential for EPS: \int dnu nu^beta * exp(-nu/2)/sqrt(nu) from numin to infty.
double Fcollapprox (double numin, double beta){
//nu is deltacrit^2/sigma^2, corrected by delta(R) and sigma(R)
  double gg = gsl_sf_gamma_inc(0.5+beta,0.5*numin);
  return gg*pow(2,0.5+beta)*pow(2.0*PI,-0.5);
}

//This routine assumes sharp cutoffs for each turnover rather than exponential, assumes a triple power-law form for sigma(M)
//  and takes advantage of the fact that Gamma_inc(x,min) = integral_min^inf (t^(x-1)exp(-t)) dt which is satisfied for the HMF when the
//  above approximations are made
//Originally written by JBM within the GL integration before it was separated here and generalised to the other integrals
double MFIntegral_Approx(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type){
    //variables used in the calculation
    double lnM_higher, lnM_lower;
    double sigma_higher,sigma_lower;
    double nu_higher,nu_lower;
    double beta,sigdiff_inv;
    double fcoll1=0.,fcoll2=0.,fcoll3=0.;

    //parameters unpacked from the struct
    double delta,sigma_c;
    double index_base;

    if(params.HMF != 0){
        LOG_ERROR("Approximate Fcoll is currently only implemented for EPS");
        LOG_ERROR("Ensure parameter input specifically to this function has HMF==0");
        Throw(TableGenerationError);
    }
    double growthf = params.growthf;
    if(type < 0){
        //we are a conditional mf
        delta = params.delta;
        sigma_c = params.sigma_cond;
    }
    else{
        //unconditional
        delta = 0.;
        sigma_c = 0.;
    }

    double lnM_lo_limit = lnM_lo;
    double lnM_hi_limit = lnM_hi;
    //these might be nan but aren't used in those cases
    //TODO (Speed): by passing in log(M_turnover) i can avoid these 2 log calls
    double lnMturn_l = log(params.Mturn);
    double lnMturn_u = log(params.Mturn_upper);
    //TODO (speed): LOG(MPIVOTn) can be pre-defined
    double lnMp1 = log(MPIVOT1);
    double lnMp2 = log(MPIVOT2);

    //The below limit setting is done simply so that variables which do not conern particular integrals
    //      can be left undefined, rather than explicitly set to some value (0 or 1e20)
    //Mass and number integrals set the lower cutoff to the integral limit
    if(fabs(type) >= 2 && lnMturn_l > lnM_lo_limit)
        lnM_lo_limit = lnMturn_l;
    //non-minihalo integrals set the upper cutoff to the integral limit
    if(fabs(type) == 3 && lnMturn_u < lnM_hi_limit)
        lnM_hi_limit = lnMturn_u;

    //n_ion or MINI
    if(fabs(type) >= 3)
        index_base = params.alpha_star + params.alpha_esc;
    //fcoll
    else if(fabs(type)==2)
        index_base = 0.;
    //nhalo
    else
        index_base = -1.;

    double delta_arg = pow((Deltac - delta)/growthf, 2);

    LOG_ULTRA_DEBUG("Starting Fcoll approx [%.3e %.3e] cond %.3e",lnM_lo,lnM_hi,sigma_c);
    //If we need the first (high-mass) segment:
    if(lnM_hi_limit > lnMp1){
        lnM_lower = fmax(lnMp1,lnM_lo_limit);
        lnM_higher = lnM_hi_limit;

        sigma_lower = EvaluateSigma(lnM_lower,0,NULL);
        sigma_higher = EvaluateSigma(lnM_higher,0,NULL);

        //The upper limit is usually the condition mass, this prevents the nan
        if(sigma_higher == sigma_c)
            sigdiff_inv = 1e6;
        else
            sigdiff_inv = 1/(sigma_higher*sigma_higher - sigma_c*sigma_c);

        nu_lower = delta_arg/(sigma_lower*sigma_lower - sigma_c*sigma_c);
        nu_higher = delta_arg*sigdiff_inv;

        beta = (index_base) * AINDEX1 * (0.5);

        fcoll1 = (Fcollapprox(nu_lower,beta) - Fcollapprox(nu_higher,beta))*pow(nu_lower,-beta);

        LOG_ULTRA_DEBUG("Fcoll approx high mass [%.2e %.2e] sigma [%.2e %.2e] nu [%.2e %.2e] %.6e",
                    lnM_lower,lnM_higher,sigma_lower,sigma_higher,nu_lower,nu_higher,fcoll1);
    }

    //If we need the second (mid-mass) segment
    if(lnM_hi_limit > lnMp2 || lnM_lo_limit < lnMp1){
        lnM_lower = fmax(lnMp2,lnM_lo_limit);
        lnM_higher = fmin(lnMp1,lnM_hi_limit);
        sigma_lower = EvaluateSigma(lnM_lower,0,NULL);
        sigma_higher = EvaluateSigma(lnM_higher,0,NULL);

        if(sigma_higher == sigma_c)
            sigdiff_inv = 1e6;
        else
            sigdiff_inv = 1/(sigma_higher*sigma_higher - sigma_c*sigma_c);

        nu_lower = delta_arg/(sigma_lower*sigma_lower - sigma_c*sigma_c);
        nu_higher = delta_arg*sigdiff_inv;
        beta = (index_base) * AINDEX2 * (0.5);
        fcoll2 = (Fcollapprox(nu_lower,beta) - Fcollapprox(nu_higher,beta))*pow(nu_lower,-beta);
        LOG_ULTRA_DEBUG("Fcoll approx mid  mass [%.2e %.2e] sigma [%.2e %.2e] nu [%.2e %.2e] %.6e",
                    lnM_lower,lnM_higher,sigma_lower,sigma_higher,nu_lower,nu_higher,fcoll2);
    }

    //If we need the third (low-mass) segment
    if(lnM_lo_limit < lnMp2){
        lnM_lower = lnM_lo_limit;
        lnM_higher = fmin(lnMp2,lnM_hi_limit);
        sigma_lower = EvaluateSigma(lnM_lower,0,NULL);
        sigma_higher = EvaluateSigma(lnM_higher,0,NULL);

        if(sigma_higher == sigma_c)
            sigdiff_inv = 1e6;
        else
            sigdiff_inv = 1/(sigma_higher*sigma_higher - sigma_c*sigma_c);

        nu_lower = delta_arg/(sigma_lower*sigma_lower - sigma_c*sigma_c);
        nu_higher = delta_arg*sigdiff_inv;
        beta = (index_base) * AINDEX3 * (0.5);
        fcoll3 = (Fcollapprox(nu_lower,beta) - Fcollapprox(nu_higher,beta))*pow(nu_lower,-beta);
        LOG_ULTRA_DEBUG("Fcoll approx low  mass [%.2e %.2e] sigma [%.2e %.2e] nu [%.2e %.2e] %.6e",
                    lnM_lower,lnM_higher,sigma_lower,sigma_higher,nu_lower,nu_higher,fcoll3);
    }

    return fcoll1 + fcoll2 + fcoll3;
}

double IntegratedNdM(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type, int method){
    if(method==0)
        return IntegratedNdM_QAG(lnM_lo, lnM_hi, params, type);
    if(method==1)
        return IntegratedNdM_GL(lnM_lo, lnM_hi, params, type);
    if(method==2)
        return MFIntegral_Approx(lnM_lo, lnM_hi, params, type);
}

//Some wrappers over the integration functions for specific integrals//

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
 FUNCTION FgtrM_wsigma(z, sigma_z0(M))
 Computes the fraction of mass contained in haloes with mass > M at redshift z.
 Requires sigma_z0(M) rather than M to make certain heating integrals faster
 */
double FgtrM_wsigma(double z, double sig){
    double del;

    del = Deltac/dicke(z); //regular spherical collapse delta

    return splined_erfc(del / (sqrt(2)*sig));
}

double FgtrM_General(double z, double M, int method){
    double lower_limit, upper_limit, growthf;

    growthf = dicke(z);
    lower_limit = log(M);
    upper_limit = log(fmax(global_params.M_MAX_INTEGRAL, M*100));
    struct parameters_gsl_MF_integrals integral_params = {
                .redshift = z,
                .growthf = growthf,
                .HMF = user_params_ps->HMF,
    };
    return IntegratedNdM(lower_limit, upper_limit, integral_params, 2, method) / (cosmo_params_ps->OMm*RHOcrit);
}

double Nion_General(double z, double M_Min, double M_Max, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10,
                     double Fesc10, double Mlim_Fstar, double Mlim_Fesc, int method){
    struct parameters_gsl_MF_integrals params = {
        .redshift = z,
        .growthf = dicke(z),
        .Mturn = MassTurnover,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar10,
        .f_esc_norm = Fesc10,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        .HMF = user_params_ps->HMF,
    };
    // LOG_DEBUG("Nion_General: M[%.2e %.2e] z=%.2f Turn %.2e fs (%.2e %.2e) fe (%.2e %.2e) lims (%.2e %.2e)",
    //             M_Min,M_Max,z,MassTurnover,Fstar10,Alpha_star,Fesc10,Alpha_esc,Mlim_Fstar,Mlim_Fesc);
    return IntegratedNdM(log(M_Min),log(M_Max),params,3,method) / ((cosmo_params_ps->OMm)*RHOcrit);
}

double Nion_General_MINI(double z, double M_Min, double M_Max, double MassTurnover, double MassTurnover_upper, double Alpha_star,
                         double Alpha_esc, double Fstar7_MINI, double Fesc7_MINI, double Mlim_Fstar, double Mlim_Fesc, int method){
    struct parameters_gsl_MF_integrals params = {
        .redshift = z,
        .growthf = dicke(z),
        .Mturn = MassTurnover,
        .Mturn_upper = MassTurnover_upper,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar7_MINI,
        .f_esc_norm = Fesc7_MINI,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        .HMF = user_params_ps->HMF,
    };
    return IntegratedNdM(log(M_Min),log(M_Max),params,4,method) / ((cosmo_params_ps->OMm)*RHOcrit);
}

double Nhalo_Conditional(double growthf, double lnM1, double lnM2, double sigma, double delta, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .HMF = user_params_ps->HMF,
        .sigma_cond = sigma,
        .delta = delta,
    };
    return IntegratedNdM(lnM1,lnM2,params,-1, method);
}

double Mcoll_Conditional(double growthf, double lnM1, double lnM2, double sigma, double delta, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .HMF = user_params_ps->HMF,
        .sigma_cond = sigma,
        .delta = delta,
    };
    return IntegratedNdM(lnM1,lnM2,params,-2, method);
}

double Nion_ConditionalM_MINI(double growthf, double lnM1, double lnM2, double sigma2, double delta2, double MassTurnover,
                            double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar7,
                            double Fesc7, double Mlim_Fstar, double Mlim_Fesc, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .Mturn = MassTurnover,
        .Mturn_upper = MassTurnover_upper,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar7,
        .f_esc_norm = Fesc7,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        // .HMF = user_params_ps->HMF,
        .HMF = 0, //FORCE EPS UNTIL THE OTHERS WORK
        .sigma_cond = sigma2,
        .delta = delta2,
    };
    return IntegratedNdM(lnM1,lnM2,params,-4,method);
}

double Nion_ConditionalM(double growthf, double lnM1, double lnM2, double sigma2, double delta2, double MassTurnover,
                        double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar,
                        double Mlim_Fesc, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .Mturn = MassTurnover,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar10,
        .f_esc_norm = Fesc10,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        // .HMF = user_params_ps->HMF,
        .HMF = 0, //FORCE EPS UNTIL THE OTHERS WORK
        .sigma_cond = sigma2,
        .delta = delta2,
    };
    return IntegratedNdM(lnM1,lnM2,params,-3, method);
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
        return 1.0;
    }

    // TODO: This could be wrapped in a Try/Catch to try the fast way and if it doesn't
    // work, use the slow way.
    return erfcc(x); // the interpolation below doesn't seem to be stable in Ts.c
    if (x > ERFC_PARAM_DELTA*(ERFC_NPTS-1))
        return erfcc(x);
    else
        return exp(gsl_spline_eval(erfc_spline, x, erfc_acc));
}


void initialiseSigmaMInterpTable(float M_min, float M_max){
    int i;

    if(!Sigma_InterpTable.allocated)
        allocate_RGTable1D_f(NMass,&Sigma_InterpTable);
    if(!dSigmasqdm_InterpTable.allocated)
        allocate_RGTable1D_f(NMass,&dSigmasqdm_InterpTable);

    Sigma_InterpTable.x_min = log(M_min);
    Sigma_InterpTable.x_width = (log(M_max) - log(M_min))/(NMass-1.);
    dSigmasqdm_InterpTable.x_min = log(M_min);
    dSigmasqdm_InterpTable.x_width = (log(M_max) - log(M_min))/(NMass-1.);

#pragma omp parallel private(i) num_threads(user_params_ps->N_THREADS)
    {
        float Mass;
#pragma omp for
        for(i=0;i<NMass;i++) {
            Mass = exp(Sigma_InterpTable.x_min + i*Sigma_InterpTable.x_width);
            // LOG_DEBUG("i %d lnM %.2e M %.2e",i, Sigma_InterpTable.x_min + i*Sigma_InterpTable.x_width, Mass);
            Sigma_InterpTable.y_arr[i] = sigma_z0(Mass);
            dSigmasqdm_InterpTable.y_arr[i] = log10(-dsigmasqdm_z0(Mass)); //TODO: look into if log/linear is better
            // LOG_DEBUG("S %.2e dS %.2e", Sigma_InterpTable.y_arr[i], dSigmasqdm_InterpTable.y_arr[i]);
        }
    }

    for(i=0;i<NMass;i++) {
        if(isfinite(Sigma_InterpTable.y_arr[i]) == 0 || isfinite(dSigmasqdm_InterpTable.y_arr[i]) == 0){
            LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
            Throw(TableGenerationError);
        }
    }
}

void freeSigmaMInterpTable(){
    free_RGTable1D_f(&Sigma_InterpTable);
    free_RGTable1D_f(&dSigmasqdm_InterpTable);
}


void nrerror(char error_text[])
{
    LOG_ERROR("Numerical Recipes run-time error...");
    LOG_ERROR("%s",error_text);
    Throw(MemoryAllocError);
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


/* dnbiasdM */
double dnbiasdM(double M, float z, double M_o, float del_o){
    double sigsq, del, sig_one, sig_o;

    if ((M_o-M) < TINY){
        LOG_ERROR("In function dnbiasdM: M must be less than M_o!\nAborting...\n");
        Throw(ValueError);
    }
    del = Deltac/dicke(z) - del_o;
    if (del < 0){
        LOG_ERROR(" In function dnbiasdM: del_o must be less than del_1 = del_crit/dicke(z)!\nAborting...\n");
        Throw(ValueError);
    }

    sig_o = sigma_z0(M_o);
    sig_one = sigma_z0(M);
    sigsq = sig_one*sig_one - sig_o*sig_o;
    return -(RHOcrit*cosmo_params_ps->OMm)/M /sqrt(2*PI) *del*pow(sigsq,-1.5)*pow(E, -0.5*del*del/sigsq)*dsigmasqdm_z0(M);
}

/*
 calculates the fraction of mass contained in haloes with mass > M at redshift z, in regions with a linear overdensity of del_bias, and standard deviation sig_bias
 */

//I wrote a version of FgtrM which takes the growth func instead of z for a bit of speed
double FgtrM_bias_fast(float growthf, float del_bias, float sig_small, float sig_large){
    double del, sig;
    if (sig_large > sig_small){ // biased region is smaller that halo!
        LOG_ERROR("Trying to compute FgtrM in region where M_min > M_max");
        Throw(ValueError);
    }
    //sometimes they are the same to float precision, where the M_condition ~ M_Min
    if(sig_large == sig_small){
        return 0.;
    }
    // del = Deltac/growthf - del_bias; //NOTE HERE DELTA EXTRAPOLATED TO z=0
    sig = sqrt(sig_small*sig_small - sig_large*sig_large);
    del = (Deltac - del_bias)/growthf;

    //if the density is above critical on this scale, it is collapsed
    //NOTE: should we allow del < 0???
    if(del < FRACT_FLOAT_ERR){
        return 1.;
    }
    return splined_erfc(del / (sqrt(2)*sig));
}

/* Uses sigma parameters instead of Mass for scale */
double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias){
    return FgtrM_bias_fast(dicke(z),del_bias,sigsmallR,sig_bias);
}

double FgtrM_bias(double z, double M, double del_bias, double sig_bias){
    return sigmaparam_FgtrM_bias(z,EvaluateSigma(log(M),0,NULL),del_bias,sig_bias);
}

//  Redshift derivative of the conditional collapsed fraction
//  TODO: change to use linear growth ddicke_dz, dsigmadz since we have these tabulated?
float dfcoll_dz(float z, float sigma_min, float del_bias, float sig_bias)
{
    double dz,z1,z2;
    double fc1,fc2,ans;

    dz = 0.001;
    z1 = z + dz;
    z2 = z - dz;
    fc1 = sigmaparam_FgtrM_bias(z1, sigma_min, del_bias, sig_bias);
    fc2 = sigmaparam_FgtrM_bias(z2, sigma_min, del_bias, sig_bias);
    ans = (fc1 - fc2)/(2.0*dz);
    return ans;
}

/* redshift derivative of the growth function at z */
double ddicke_dz(double z){
    float dz = 1e-10;
    double omegaM_z, ddickdz, dick_0, x, x_0, domegaMdz;

    return (dicke(z+dz)-dicke(z))/dz;
}

/* compute a mass limit where the stellar baryon fraction and the escape fraction exceed unity */
//NOTE (JD): Why aren't we using 1e10 * pow(FRAC,-1/PL)? what am I missing here that makes the rootfind necessary
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

    // Got to max_iter without finding a solution.
    LOG_ERROR("Failed to find a mass limit to regulate stellar fraction/escape fraction is between 0 and 1.");
    LOG_ERROR(" The solution does not converge or iterations are not sufficient.");
//    Throw(ParameterError);
    Throw(MassDepZetaError);

    return(0.0);
}

int initialise_ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options) {

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    lnMhalo_param = calloc(nbins,sizeof(double));
    Muv_param = calloc(nbins,sizeof(double));
    Mhalo_param = calloc(nbins,sizeof(double));

    LF_spline_acc = gsl_interp_accel_alloc();
    LF_spline = gsl_spline_alloc(gsl_interp_cspline, nbins);

    init_ps();

    int status;
    Try initialiseSigmaMInterpTable(0.999*Mhalo_min,1.001*Mhalo_max);
    Catch(status) {
        LOG_ERROR("\t...called from initialise_ComputeLF");
        return(status);
    }

    initialised_ComputeLF = true;
    return(0);
}

void cleanup_ComputeLF(){
    free(lnMhalo_param);
    free(Muv_param);
    free(Mhalo_param);
    gsl_spline_free (LF_spline);
    gsl_interp_accel_free(LF_spline_acc);
    freeSigmaMInterpTable();
	initialised_ComputeLF = 0;
}

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi) {
    /*
        This is an API-level function and thus returns an int status.
    */
    int status;
    Try{ // This try block covers the whole function.
    // This NEEDS to be done every time, because the actual object passed in as
    // user_params, cosmo_params etc. can change on each call, freeing up the memory.
    initialise_ComputeLF(nbins, user_params,cosmo_params,astro_params,flag_options);

    int i,i_z;
    int i_unity, i_smth, mf, nbins_smth=7;
    double  dlnMhalo, lnMhalo_i, SFRparam, Muv_1, Muv_2, dMuvdMhalo;
    double Mhalo_i, lnMhalo_min, lnMhalo_max, lnMhalo_lo, lnMhalo_hi, dlnM, growthf;
    double f_duty_upper, Mcrit_atom;
    float Fstar, Fstar_temp;
    double dndm;
    int gsl_status;

    gsl_set_error_handler_off();
    if (astro_params->ALPHA_STAR < -0.5)
        LOG_WARNING(
            "ALPHA_STAR is %f, which is unphysical value given the observational LFs.\n"\
            "Also, when ALPHA_STAR < -.5, LFs may show a kink. It is recommended to set ALPHA_STAR > -0.5.",
            astro_params->ALPHA_STAR
        );

    mf = user_params_ps->HMF;

    lnMhalo_min = log(Mhalo_min*0.999);
    lnMhalo_max = log(Mhalo_max*1.001);
    dlnMhalo = (lnMhalo_max - lnMhalo_min)/(double)(nbins - 1);

    for (i_z=0; i_z<NUM_OF_REDSHIFT_FOR_LF; i_z++) {

        growthf = dicke(z_LF[i_z]);
        Mcrit_atom = atomic_cooling_threshold(z_LF[i_z]);

        i_unity = -1;
        for (i=0; i<nbins; i++) {
            // generate interpolation arrays
            lnMhalo_param[i] = lnMhalo_min + dlnMhalo*(double)i;
            Mhalo_i = exp(lnMhalo_param[i]);

            if (component == 1)
                Fstar = astro_params->F_STAR10*pow(Mhalo_i/1e10,astro_params->ALPHA_STAR);
            else
                Fstar = astro_params->F_STAR7_MINI*pow(Mhalo_i/1e7,astro_params->ALPHA_STAR_MINI);
            if (Fstar > 1.) Fstar = 1;

            if (i_unity < 0) { // Find the array number at which Fstar crosses unity.
                if (astro_params->ALPHA_STAR > 0.) {
                    if ( (1.- Fstar) < FRACT_FLOAT_ERR ) i_unity = i;
                }
                else if (astro_params->ALPHA_STAR < 0. && i < nbins-1) {
                    if (component == 1)
                        Fstar_temp = astro_params->F_STAR10*pow( exp(lnMhalo_min + dlnMhalo*(double)(i+1))/1e10,astro_params->ALPHA_STAR);
                    else
                        Fstar_temp = astro_params->F_STAR7_MINI*pow( exp(lnMhalo_min + dlnMhalo*(double)(i+1))/1e7,astro_params->ALPHA_STAR_MINI);
                    if (Fstar_temp < 1. && (1.- Fstar) < FRACT_FLOAT_ERR) i_unity = i;
                }
            }

            // parametrization of SFR
            SFRparam = Mhalo_i * cosmo_params->OMb/cosmo_params->OMm * (double)Fstar * (double)(hubble(z_LF[i_z])*SperYR/astro_params->t_STAR); // units of M_solar/year

            Muv_param[i] = 51.63 - 2.5*log10(SFRparam*Luv_over_SFR); // UV magnitude
            // except if Muv value is nan or inf, but avoid error put the value as 10.
            if ( isinf(Muv_param[i]) || isnan(Muv_param[i]) ) Muv_param[i] = 10.;

            M_uv_z[i + i_z*nbins] = Muv_param[i];
        }

        gsl_status = gsl_spline_init(LF_spline, lnMhalo_param, Muv_param, nbins);
        GSL_ERROR(gsl_status);

        lnMhalo_lo = log(Mhalo_min);
        lnMhalo_hi = log(Mhalo_max);
        dlnM = (lnMhalo_hi - lnMhalo_lo)/(double)(nbins - 1);

        // There is a kink on LFs at which Fstar crosses unity. This kink is a numerical artefact caused by the derivate of dMuvdMhalo.
        // Most of the cases the kink doesn't appear in magnitude ranges we are interested (e.g. -22 < Muv < -10). However, for some extreme
        // parameters, it appears. To avoid this kink, we use the interpolation of the derivate in the range where the kink appears.
        // 'i_unity' is the array number at which the kink appears. 'i_unity-3' and 'i_unity+12' are related to the range of interpolation,
        // which is an arbitrary choice.
        // NOTE: This method does NOT work in cases with ALPHA_STAR < -0.5. But, this parameter range is unphysical given that the
        //       observational LFs favour positive ALPHA_STAR in this model.
        // i_smth = 0: calculates LFs without interpolation.
        // i_smth = 1: calculates LFs using interpolation where Fstar crosses unity.
        if (i_unity-3 < 0) i_smth = 0;
        else if (i_unity+12 > nbins-1) i_smth = 0;
        else i_smth = 1;
        if (i_smth == 0) {
            for (i=0; i<nbins; i++) {
                // calculate luminosity function
                lnMhalo_i = lnMhalo_lo + dlnM*(double)i;
                Mhalo_param[i] = exp(lnMhalo_i);

                M_h_z[i + i_z*nbins] = Mhalo_param[i];

                Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
                Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);

                dMuvdMhalo = (Muv_2 - Muv_1) / (2.*delta_lnMhalo * exp(lnMhalo_i));

                if (component == 1)
                    f_duty_upper = 1.;
                else
                    f_duty_upper = exp(-(Mhalo_param[i]/Mcrit_atom));
                if(mf==0) {
                    log10phi[i + i_z*nbins] = log10( dNdM(growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else if(mf==1) {
                    log10phi[i + i_z*nbins] = log10( dNdM_st(growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else if(mf==2) {
                    log10phi[i + i_z*nbins] = log10( dNdM_WatsonFOF(growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else if(mf==3) {
                    log10phi[i + i_z*nbins] = log10( dNdM_WatsonFOF_z(z_LF[i_z], growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else{
                    LOG_ERROR("HMF should be between 0-3, got %d", mf);
                    Throw(ValueError);
                }
                if (isinf(log10phi[i + i_z*nbins]) || isnan(log10phi[i + i_z*nbins]) || log10phi[i + i_z*nbins] < -30.)
                    log10phi[i + i_z*nbins] = -30.;
            }
        }
        else {
            lnM_temp = calloc(nbins_smth,sizeof(double));
            deriv_temp = calloc(nbins_smth,sizeof(double));
            deriv = calloc(nbins,sizeof(double));

            for (i=0; i<nbins; i++) {
                // calculate luminosity function
                lnMhalo_i = lnMhalo_lo + dlnM*(double)i;
                Mhalo_param[i] = exp(lnMhalo_i);

                M_h_z[i + i_z*nbins] = Mhalo_param[i];

                Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
                Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);

                dMuvdMhalo = (Muv_2 - Muv_1) / (2.*delta_lnMhalo * exp(lnMhalo_i));
                deriv[i] = fabs(dMuvdMhalo);
            }

            deriv_spline_acc = gsl_interp_accel_alloc();
            deriv_spline = gsl_spline_alloc(gsl_interp_cspline, nbins_smth);

            // generate interpolation arrays to smooth discontinuity of the derivative causing a kink
            // Note that the number of array elements and the range of interpolation are made by arbitrary choices.
            lnM_temp[0] = lnMhalo_param[i_unity - 3];
            lnM_temp[1] = lnMhalo_param[i_unity - 2];
            lnM_temp[2] = lnMhalo_param[i_unity + 8];
            lnM_temp[3] = lnMhalo_param[i_unity + 9];
            lnM_temp[4] = lnMhalo_param[i_unity + 10];
            lnM_temp[5] = lnMhalo_param[i_unity + 11];
            lnM_temp[6] = lnMhalo_param[i_unity + 12];

            deriv_temp[0] = deriv[i_unity - 3];
            deriv_temp[1] = deriv[i_unity - 2];
            deriv_temp[2] = deriv[i_unity + 8];
            deriv_temp[3] = deriv[i_unity + 9];
            deriv_temp[4] = deriv[i_unity + 10];
            deriv_temp[5] = deriv[i_unity + 11];
            deriv_temp[6] = deriv[i_unity + 12];

            gsl_status = gsl_spline_init(deriv_spline, lnM_temp, deriv_temp, nbins_smth);
            GSL_ERROR(gsl_status);

            for (i=0;i<9;i++){
                deriv[i_unity + i - 1] = gsl_spline_eval(deriv_spline, lnMhalo_param[i_unity + i - 1], deriv_spline_acc);
            }
            for (i=0; i<nbins; i++) {
                if (component == 1)
                    f_duty_upper = 1.;
                else
                    f_duty_upper = exp(-(Mhalo_param[i]/Mcrit_atom));

                if(mf==0)
                    dndm = dNdM(growthf, Mhalo_param[i]);
                else if(mf==1)
                    dndm = dNdM_st(growthf, Mhalo_param[i]);
                else if(mf==2)
                    dndm = dNdM_WatsonFOF(growthf, Mhalo_param[i]);
                else if(mf==3)
                    dndm = dNdM_WatsonFOF_z(z_LF[i_z], growthf, Mhalo_param[i]);
                else{
                    LOG_ERROR("HMF should be between 0-3, got %d", mf);
                    Throw(ValueError);
                }
                log10phi[i + i_z*nbins] = log10(dndm * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / deriv[i]);
                if (isinf(log10phi[i + i_z*nbins]) || isnan(log10phi[i + i_z*nbins]) || log10phi[i + i_z*nbins] < -30.)
                    log10phi[i + i_z*nbins] = -30.;
            }
        }
    }

	cleanup_ComputeLF();
    } // End try
    Catch(status){
        return status;
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

    /*
        This is an API-level function for initialising the photon conservation.
    */

    int status;
    Try{  // this try wraps the whole function.
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();
    //     To solve differentail equation, uses Euler's method.
    //     NOTE:
    //     (1) With the fiducial parameter set,
    //	    when the Q value is < 0.9, the difference is less than 5% compared with accurate calculation.
    //	    When Q ~ 0.98, the difference is ~25%. To increase accuracy one can reduce the step size 'da', but it will increase computing time.
    //     (2) With the fiducial parameter set,
    //     the difference for the redshift where the reionization end (Q = 1) is ~0.2 % compared with accurate calculation.
    float ION_EFF_FACTOR,M_MIN,M_MIN_z0,M_MIN_z1,Mlim_Fstar, Mlim_Fesc;
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
                Nion0 = ION_EFF_FACTOR*Nion_General(z0, astro_params->M_TURN/50., global_params.M_MAX_INTEGRAL, astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc, 0);
                Nion1 = ION_EFF_FACTOR*Nion_General(z1, astro_params->M_TURN/50., global_params.M_MAX_INTEGRAL, astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc, 0);
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

                Nion0 = ION_EFF_FACTOR*FgtrM_General(z0,M_MIN_z0,0);
                Nion1 = ION_EFF_FACTOR*FgtrM_General(z1,M_MIN_z1,0);
                freeSigmaMInterpTable();
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
    GSL_ERROR(gsl_status);

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
    GSL_ERROR(gsl_status);

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
        GSL_ERROR(gsl_status);
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
    GSL_ERROR(gsl_status);

}


float adjust_redshifts_for_photoncons(
    struct AstroParams *astro_params, struct FlagOptions *flag_options, float *redshift,
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
    GSL_ERROR(gsl_status);

    z_NFHistory_spline_acc = gsl_interp_accel_alloc ();
//    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_cspline, (counter+1));
    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_linear, (counter+1));

    gsl_status = gsl_spline_init(z_NFHistory_spline, z_vals, nf_vals, (counter+1));
    GSL_ERROR(gsl_status);
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
double alpha_photoncons_yint;
double alpha_photoncons_slope;
void set_alphacons_params(double yint, double slope){
    alpha_photoncons_yint = yint;
    alpha_photoncons_slope = slope;
    return;
}
//NOTE: The way I've set this up is a little annoying in that this function needs to match its counterpart in Python
double get_alpha_fit(double redshift){
    double Q, alpha_fit;
    Q_at_z(redshift,&Q);
    if(Q > 1.) Q = 1.;
    alpha_fit = alpha_photoncons_yint + Q*alpha_photoncons_slope;
    LOG_DEBUG("Alpha photon cons fit activated z = %.2e, fit yint,slope = %.2e, %.2e, alpha = %.2e", redshift,
                alpha_photoncons_yint,alpha_photoncons_slope,alpha_fit);
    return alpha_fit;
}

//Modularisation for the evaluation of sigma
double EvaluateSigma(double lnM, int calc_ds, double *dsigmadm){
    //using log units to make the fast option faster and the slow option slower
    double sigma;
    double dsigma_val;

    //all this stuff is defined in ps.c and initialised with InitialiseSigmaInterpTable
    //NOTE: The interpolation tables are `float` in ps.c
    if(user_params_ps->USE_INTERPOLATION_TABLES) {
        sigma = EvaluateRGTable1D_f(lnM, &Sigma_InterpTable);

        if(calc_ds){
            dsigma_val = EvaluateRGTable1D_f(lnM, &dSigmasqdm_InterpTable);
            //NOTE: This is d(sigma squared)/dm
            *dsigmadm = -pow(10.,dsigma_val); //this may be slow, figure out why the dsigmadm table is in log10
        }
    }
    else {
        sigma = sigma_z0(exp(lnM));
        if(calc_ds) *dsigmadm = dsigmasqdm_z0(exp(lnM));
    }

    return sigma;
}

//set the minimum source mass for the integrals, If we have an exponential cutoff we go below the chosen mass by a factor of 50
//NOTE: previously, with USE_MINI_HALOS, the sigma table was initialised with M_MIN_INTEGRAL/50, but then all integrals perofmed
//      from M_MIN_INTEGRAL
//TODO: *sometimes* we need to initialise tables slightly lower than the integral, this is solved ad-hoc in each file that calls this function
//      I should make a proper fix
double minimum_source_mass(double redshift, struct AstroParams *astro_params, struct FlagOptions *flag_options){
    double Mmin,min_factor,ion_factor;
    if(flag_options->USE_MASS_DEPENDENT_ZETA && !flag_options->USE_MINI_HALOS)
        min_factor = 50.; // small lower bound to cover far below the turnover
    else
        min_factor = 1.; //sharp cutoff

    // automatically false if !USE_MASS_DEPENDENT_ZETA
    if(flag_options->USE_MINI_HALOS){
        Mmin = global_params.M_MIN_INTEGRAL;
    }
    // automatically true if USE_MASS_DEPENDENT_ZETA
    else if(flag_options->M_MIN_in_Mass) {
        Mmin = astro_params->M_TURN;
    }
    else {
        ion_factor = astro_params->ION_Tvir_MIN < 9.99999e3 ? 1.22 : 0.6;
        Mmin = TtoM(redshift, astro_params->ION_Tvir_MIN, ion_factor);
    }

    //I doubt this will be used much but previously
    //  it was ONLY in the !USE_MASS_DEP_ZETA case,
    //  and the fuction looks odd (fudge), should be tested
    //  doesn't make much sense with the turnover, unless it is incorporated
    //  with the feedbacks properly (LW,atomic,reion)
    if(global_params.P_CUTOFF){
        Mmin = fmax(Mmin,M_J_WDM());
    }

    Mmin /= min_factor;

    return Mmin;
}
