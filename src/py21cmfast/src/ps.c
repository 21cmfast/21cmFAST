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

#define MAX_DELTAC_FRAC (float)0.99 //max delta/deltac for the mass function integrals
#define JENKINS_a (0.73) //Jenkins+01, SMT has 0.707
#define JENKINS_b (0.34) //Jenkins+01 fit from Barkana+01, SMT has 0.5
#define JENKINS_c (0.81) //Jenkins+01 from from Barkana+01, SMT has 0.6

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
float xi_GL[NGL_INT+1], wi_GL[NGL_INT+1];
float GL_limit[2] = {0};

//These globals are used for the LF calculation
double *lnMhalo_param, *Muv_param, *Mhalo_param;
double *log10phi, *M_uv_z, *M_h_z;
double *lnMhalo_param_MINI, *Muv_param_MINI, *Mhalo_param_MINI;
double *log10phi_MINI; *M_uv_z_MINI, *M_h_z_MINI;
double *deriv, *lnM_temp, *deriv_temp;

void initialiseSigmaMInterpTable(float M_Min, float M_Max);
void freeSigmaMInterpTable();

double EvaluateSigma(double lnM);
double EvaluatedSigmasqdm(double lnM);

//JBM: Exact integral for power-law indices non zero (for zero it's erfc)
double Fcollapprox (double numin, double beta);

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
    return dfdM*cosmo_params_ps->OMm*RHOcrit;
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
    //   the (cosmo_params_ps->OMm)*RHOcrit
    //NOTE: dfdM == constants*dNdlnM
    // LOG_ULTRA_DEBUG("M = %.3e Barrier = %.3f || dndlnM= %.6e",exp(lnM),DELTAC_DELOS,dfdM);
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

//CMF Corresponding to the Sheth Mo Tormen HMF, here we assume that we are passing the correct delta2,
//      which is the condition delta, the barrier delta1 is set by the mass, so it is passed usually as Deltac
//NOTE: Currently broken and I don't know why
double dNdM_conditional_ST(double growthf, double lnM, double delta_cond, double sigma_cond){
    double sigma1, dsigmasqdm, Barrier, factor, sigdiff_inv, result;
    double delta_0 = delta_cond / growthf;
    sigma1 = EvaluateSigma(lnM);
    dsigmasqdm = EvaluatedSigmasqdm(lnM);
    if(sigma1 < sigma_cond) return 0.;

    factor = st_taylor_factor(sigma1,sigma_cond,growthf,&Barrier) - delta_0;

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
double dNdlnM_st(double growthf, double lnM){
    double sigma, dsigmadm, nuhat;

    float MassBinLow;
    int MassBin;

    sigma = EvaluateSigma(lnM);
    dsigmadm = EvaluatedSigmasqdm(lnM);

    sigma = sigma * growthf;
    dsigmadm = dsigmadm * (growthf*growthf/(2.*sigma));

    nuhat = sqrt(SHETH_a) * Deltac / sigma;

    return (-(cosmo_params_ps->OMm)*RHOcrit) * (dsigmadm/sigma) * sqrt(2./PI)*SHETH_A * (1+ pow(nuhat, -2*SHETH_p)) * nuhat * pow(E, -nuhat*nuhat/2.0);
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
    return (-(cosmo_params_ps->OMm)*RHOcrit) * sqrt(2/PI) * (Deltac/(sigma*sigma)) * dsigmadm * exp(-(Deltac*Deltac)/(2*sigma*sigma));
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

    return (-(cosmo_params_ps->OMm)*RHOcrit) * (dsigmadm/sigma) * f_sigma;
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

    Omega_m_z = (cosmo_params_ps->OMm)*pow(1.+z,3.) / ( (cosmo_params_ps->OMl) + (cosmo_params_ps->OMm)*pow(1.+z,3.) + (global_params.OMr)*pow(1.+z,4.) );

    A_z = Omega_m_z * ( Watson_A_z_1 * pow(1. + z, Watson_A_z_2 ) + Watson_A_z_3 );
    alpha_z = Omega_m_z * ( Watson_alpha_z_1 * pow(1.+z, Watson_alpha_z_2 ) + Watson_alpha_z_3 );
    beta_z = Omega_m_z * ( Watson_beta_z_1 * pow(1.+z, Watson_beta_z_2 ) + Watson_beta_z_3 );

    f_sigma = A_z * ( pow(beta_z/sigma, alpha_z) + 1. ) * exp( - Watson_gamma_z/(sigma*sigma) );

    return (-(cosmo_params_ps->OMm)*RHOcrit) * (dsigmadm/sigma) * f_sigma;
}

///////MASS FUNCTION INTEGRANDS BELOW//////

//gets the fraction (in units of 1/normalisation at 1e10)
double get_frac_limit(double M, double norm, double alpha, double limit, bool mini){
    double pivot = mini ? 1e7 : 1e10;
    if ((alpha > 0. && M > limit) || (alpha < 0. && M < limit))
        return 1/norm;

    //if alpha is zero, this returns 1 as expected (note strict inequalities above)
    return pow(M/pivot,alpha);
}

double nion_fraction(double M, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,false);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,false);

    return Fstar * Fesc * exp(-M_turn_lower/M);
}

double nion_fraction_mini(double M, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double M_turn_upper = params.Mturn_upper;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,true);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,true);

    return Fstar * Fesc * exp(-M_turn_lower/M - M/M_turn_upper);
}

double conditional_mf(double growthf, double lnM, double delta, double sigma, int HMF){
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
    return dNdM_conditional_EPS(growthf,lnM,delta,sigma);

}

double c_mf_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double growthf = params.growthf;
    double delta = params.delta; //the condition delta
    double sigma2 = params.sigma_cond;
    int HMF = params.HMF;

    return conditional_mf(growthf,lnM,delta,sigma2,HMF);
}

double c_fcoll_integrand(double lnM, void *param_struct){
    return exp(lnM) * c_mf_integrand(lnM,param_struct);
}

double c_nion_integrand(double lnM, void *param_struct){
    return nion_fraction(exp(lnM),param_struct) * exp(lnM) * c_mf_integrand(lnM,param_struct);
}

//The reason this is separated from the above is the second exponent
double c_nion_integrand_mini(double lnM, void *param_struct){

    return nion_fraction_mini(exp(lnM),param_struct) * exp(lnM) * c_mf_integrand(lnM,param_struct);
}

double unconditional_mf(double growthf, double lnM, double z, int HMF){
    //most of the UMFs are defined with M, but we integrate over lnM
    //NOTE: HMF > 4 or < 0 gets caught earlier, so unless some strange change is made this is fine
    if(HMF==0) {
        return dNdlnM_PS(growthf, lnM);
    }
    if(HMF==1) {
        return dNdlnM_st(growthf, lnM);
    }
    if(HMF==2) {
        return dNdlnM_WatsonFOF(growthf, lnM);
    }
    if(HMF==3) {
        return dNdlnM_WatsonFOF_z(z, growthf, lnM);
    }
    if(HMF==4) {
        return dNdlnM_Delos(growthf, lnM);
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

    return unconditional_mf(growthf,lnM,z,HMF);
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

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,true);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,true);

    return M * Fstar * Fesc * exp(-M_turn_lower/M - M/M_turn_upper) * u_mf_integrand(lnM,param_struct);
}

///// INTEGRATION ROUTINES BELOW /////

//TODO: make type enum for clarity (but cffi doesn't seem to like enum in 21cmFAST.h)
//NOTE: SFR is obtained from nion with alpha_esc==0 and f_esc==1
//Currently the scheme is to use negative numbers for conditionals, and (1,2,3,4) for (number,mass,n_ion,n_ion_mini)
double (*get_integrand_function(int type))(double,void*){
    if(type==1)
        return &u_mf_integrand; //Unondtional mass function integral
    if(type==2)
        return &u_fcoll_integrand; //Unconditional collapsed fraction integral
    if(type==3)
        return &u_nion_integrand; //Unconditional N_ion integral (two power-laws and one exponential)
    if(type==4)
        return &u_nion_integrand_mini; //Unconditional N_ion minihalo integral (two power-laws and two exponentials)
    if(type==-1)
        return &c_mf_integrand; //Conditional mass function integral
    if(type==-2)
        return &c_fcoll_integrand; //Conditional collapsed fraction integral
    if(type==-3)
        return &c_nion_integrand; //Conditional N_ion integral (two power-laws and one exponential)
    if(type==-4)
        return &c_nion_integrand_mini; //Conditional N_ion minihalo integral (two power-laws and two exponentials)

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
        LOG_ERROR("data: z=%.3e growthf=%.3e  HMF=%d type=%d ",params.redshift,params.growthf,params.HMF,type);
        LOG_ERROR("sigma=%.3e delta=%.3e",params.sigma_cond,params.delta);
        LOG_ERROR("Mturn_lo=%.3e f*=%.3e a*=%.3e Mlim*=%.3e",params.Mturn,params.f_star_norm,params.alpha_star,params.Mlim_star);
        LOG_ERROR("f_escn=%.3e a_esc=%.3e Mlim_esc=%.3e",params.f_esc_norm,params.alpha_esc,params.Mlim_esc);
        LOG_ERROR("Mturn_hi %.3e",params.Mturn_upper);
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
    //don't redo if you don't have to
    if(lnM_Min == GL_limit[0] && lnM_Max == GL_limit[1])
        return;

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
        LOG_ERROR("Integral limits [%.8e %.8e] do not match Gauss Legendre limits [%.8e %.8e]!",exp(lnM_lo),exp(lnM_hi),GL_limit[0],GL_limit[1]);
        Throw(TableGenerationError);
    }

    for(i=1; i<(NGL_INT+1); i++){
        integral += wi_GL[i]*(get_integrand_function(type))(xi_GL[i],&params);
    }

    return integral;
}

#include <gsl/gsl_sf_gamma.h>
//JBM: Integral of a power-law times exponential for EPS: \int dnu nu^beta * exp(-nu/2)/sqrt(nu) from numin to infty.
double Fcollapprox(double numin, double beta){
//nu is deltacrit^2/sigma^2, corrected by delta(R) and sigma(R)
  double gg = gsl_sf_gamma_inc(0.5+beta,0.5*numin);
  return gg*pow(2,0.5+beta)*pow(2.0*PI,-0.5);
}

//This takes into account the last approximation in Munoz+22, where erfc (beta=0) is used
//NOTE: even though nu_condition is defined in the unconditional (no sigma_cond), here it
//  represents where nu_tilde == nu_condition (effectively a final pivot point)
//NOTE: This assumes numin < nucondition, otherise it fails
double Fcollapprox_condition(double numin, double nucondition, double beta){
    return (Fcollapprox(numin,beta) - Fcollapprox(nucondition,beta)) + Fcollapprox(nucondition,0.)*pow(nucondition,beta);
}

//This routine assumes sharp cutoffs for each turnover rather than exponential, assumes a triple power-law form for sigma(M)
//  and takes advantage of the fact that Gamma_inc(x,min) = integral_min^inf (t^(x-1)exp(-t)) dt which is satisfied for the HMF when the
//  above approximations are made
//Originally written by JBM within the GL integration before it was separated here and generalised to the other integrals
double MFIntegral_Approx(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type){
    //variables used in the calculation
    double lnM_higher, lnM_lower;

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
    //(Speed): by passing in log(M_turnover) i can avoid these 2 log calls
    double lnMturn_l = log(params.Mturn);
    double lnMturn_u = log(params.Mturn_upper);
    //(Speed): LOG(MPIVOTn) can be pre-defined via macro
    double lnMp1 = log(MPIVOT1);
    double lnMp2 = log(MPIVOT2);

    //The below limit setting is done simply so that variables which do not conern particular integrals
    //      can be left undefined, rather than explicitly set to some value (0 or 1e20)
    //Mass and number integrals set the lower cutoff to the integral limit
    if(fabs(type) >= 3 && lnMturn_l > lnM_lo_limit)
        lnM_lo_limit = lnMturn_l;
    //non-minihalo integrals set the upper cutoff to the integral limit
    if(fabs(type) == 4 && lnMturn_u < lnM_hi_limit)
        lnM_hi_limit = lnMturn_u;

    //it is possible for the lower turnover (LW crit or reion feedback)
    //   to be higher than the upper limit (atomic limit) or the condition
    if(lnM_lo_limit >= lnM_hi_limit || EvaluateSigma(lnM_lo_limit) <= sigma_c){
        return 0.;
    }

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
    double beta1 = index_base * AINDEX1 * 0.5; //exponent for Fcollapprox for nu>nupivot1 (large M)
    double beta2 = index_base * AINDEX2 * 0.5; //exponent for Fcollapprox for nupivot2<nu<nupivot1 (small M)
    double beta3 = index_base * AINDEX3 * 0.5; //exponent for Fcollapprox for nu<nupivot2 (smallest M)

    // There are 5 nu(M) points of interest: the two power-law pivot points, the lower and upper integral limits
    // and the condition.
    //NOTE: Since sigma(M) is approximated as a power law, not (sigma(M)^2 - sigma_cond^2), this is not a simple gamma function.
    //  note especially which nu subtracts the condition sigma and not, see Appendix B of Munoz+22 (2110.13919)
    double sigma_pivot1 = EvaluateSigma(lnMp1);
    double sigma_pivot2 = EvaluateSigma(lnMp2);
    double sigma_lo_limit = EvaluateSigma(lnM_lo_limit);
    double sigma_hi_limit = EvaluateSigma(lnM_hi_limit);

    //These nu use the CMF delta (subtracted the condition delta), but not the condition sigma
    double nu_pivot1_umf = delta_arg / (sigma_pivot1*sigma_pivot1);
    double nu_pivot2_umf = delta_arg / (sigma_pivot2*sigma_pivot2);
    double nu_condition = delta_arg / (sigma_c*sigma_c);

    double nu_pivot1 = delta_arg / (sigma_pivot1*sigma_pivot1 - sigma_c*sigma_c);
    double nu_pivot2 = delta_arg / (sigma_pivot2*sigma_pivot2 - sigma_c*sigma_c);

    //These nu subtract the condition sigma as in the CMF
    double nu_lo_limit = delta_arg / (sigma_lo_limit*sigma_lo_limit - sigma_c*sigma_c);
    double nu_hi_limit = delta_arg / (sigma_hi_limit*sigma_hi_limit - sigma_c*sigma_c);

    double fcoll = 0.;

    //NOTES: For speed the minihalos ignore the condition mass limit (assuming nu_hi_limit(tilde) < nu_condition (no tilde))
    //    and never get into the high mass power law (nu_hi_limit < nu_pivot1 (both tilde))
    //ACGs ignore the upper mass limit (no upper turnover), both assume the condition is above the highest pivot
    if(fabs(type) == 4){
      // re-written for further speedups
      if (nu_hi_limit <= nu_pivot2){ //if both are below pivot2 don't bother adding and subtracting the high contribution
        fcoll += (Fcollapprox(nu_lo_limit,beta3))*pow(nu_pivot2_umf,-beta3);
        fcoll -= (Fcollapprox(nu_hi_limit,beta3))*pow(nu_pivot2_umf,-beta3);
      }
      else {
        fcoll -= (Fcollapprox(nu_hi_limit,beta2))*pow(nu_pivot1_umf,-beta2);
        if (nu_lo_limit > nu_pivot2){
            fcoll += (Fcollapprox(nu_lo_limit,beta2))*pow(nu_pivot1_umf,-beta2);
        }
        else {
            fcoll += (Fcollapprox(nu_pivot2,beta2))*pow(nu_pivot1_umf,-beta2);
            fcoll += (Fcollapprox(nu_lo_limit,beta3)-Fcollapprox(nu_pivot2,beta3) )*pow(nu_pivot2_umf,-beta3);
        }
      }
    }
    else{
        if(nu_lo_limit >= nu_condition){ //fully in the flat part of sigma(nu), M^alpha is nu-independent.
            // This is just an erfc, remembering that the conditional nu can be higher than the unconditional nu of the condition
            return Fcollapprox(nu_lo_limit,0.);
        }

        if(nu_lo_limit >= nu_pivot1){
            //We use the condition version wherever the nu range may intersect nu_condition (i.e beta1)
            fcoll += Fcollapprox_condition(nu_lo_limit,nu_condition,beta1)*pow(nu_pivot1_umf,-beta1);
        }
        else{
            fcoll += Fcollapprox_condition(nu_pivot1,nu_condition,beta1)*pow(nu_pivot1_umf,-beta1);
            if (nu_lo_limit > nu_pivot2){
                fcoll += (Fcollapprox(nu_lo_limit,beta2)-Fcollapprox(nu_pivot1,beta2))*pow(nu_pivot1_umf,-beta2);
            }
            else {
                fcoll += (Fcollapprox(nu_pivot2,beta2)-Fcollapprox(nu_pivot1,beta2) )*pow(nu_pivot1_umf,-beta2);
                fcoll += (Fcollapprox(nu_lo_limit,beta3)-Fcollapprox(nu_pivot2,beta3) )*pow(nu_pivot2_umf,-beta3);
            }
        }
    }

    if (fcoll<=0.0){
        LOG_DEBUG("Negative fcoll? fc=%.1le\n",fcoll);
        fcoll=1e-40;
    }
    return fcoll;
}

double IntegratedNdM(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type, int method){
    if(method==0 || (method==1 && params.delta > global_params.CRIT_DENS_TRANSITION))
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

double FgtrM_General(double z, double M){
    double lower_limit, upper_limit, growthf;

    growthf = dicke(z);
    lower_limit = log(M);
    upper_limit = log(fmax(global_params.M_MAX_INTEGRAL, M*100));
    struct parameters_gsl_MF_integrals integral_params = {
                .redshift = z,
                .growthf = growthf,
                .HMF = user_params_ps->HMF,
    };
    return IntegratedNdM(lower_limit, upper_limit, integral_params, 2, 0) / (cosmo_params_ps->OMm*RHOcrit);
}

double Nion_General(double z, double lnM_Min, double lnM_Max, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10,
                     double Fesc10, double Mlim_Fstar, double Mlim_Fesc){
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
    return IntegratedNdM(lnM_Min,lnM_Max,params,3,0) / ((cosmo_params_ps->OMm)*RHOcrit);
}

double Nion_General_MINI(double z, double lnM_Min, double lnM_Max, double MassTurnover, double MassTurnover_upper, double Alpha_star,
                         double Alpha_esc, double Fstar7_MINI, double Fesc7_MINI, double Mlim_Fstar, double Mlim_Fesc){
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
    return IntegratedNdM(lnM_Min,lnM_Max,params,4,0) / ((cosmo_params_ps->OMm)*RHOcrit);
}

double Nhalo_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .HMF = user_params_ps->HMF,
        .sigma_cond = sigma,
        .delta = delta,
    };

    if(delta <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 1 halo AT THE CONDITION MASS if delta is exceeded
    if(delta > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2)) //this limit is not ideal, but covers floating point errors when we set lnM2 == log(M_cond)
            return 1./M_cond;
        else
            return 0.;
    }

    return IntegratedNdM(lnM1,lnM2,params,-1, method);
}

double Mcoll_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .HMF = user_params_ps->HMF,
        .sigma_cond = sigma,
        .delta = delta,
    };

    if(delta <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 100% of mass AT THE CONDITION MASS if delta is exceeded
    if(delta > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2)) //this limit is not ideal, but covers floating point errors when we set lnM2 == log(M_cond)
            return 1.;
        else
            return 0.;
    }
    return IntegratedNdM(lnM1,lnM2,params,-2, method);
}

double Nion_ConditionalM_MINI(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
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

    if(delta2 <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 1 halo at the condition mass if delta is exceeded
    //NOTE: this will almost always be zero, due to the upper turover,
    // however this replaces an integral so it won't be slow
    if(delta2 > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma2,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2)) //this limit is not ideal, but covers floating point errors when we set lnM2 == log(M_cond)
            return nion_fraction_mini(M_cond,&params); //NOTE: condition mass is used as if it were Lagrangian (no 1+delta)
        else
            return 0.;
    }

    // LOG_ULTRA_DEBUG("params: D=%.2e Mtl=%.2e Mtu=%.2e as=%.2e ae=%.2e fs=%.2e fe=%.2e Ms=%.2e Me=%.2e hmf=%d sig=%.2e del=%.2e",
    //     growthf,MassTurnover,MassTurnover_upper,Alpha_star,Alpha_esc,Fstar7,Fesc7,Mlim_Fstar,Mlim_Fesc,0,sigma2,delta2);
    return IntegratedNdM(lnM1,lnM2,params,-4,method);
}

double Nion_ConditionalM(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
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

    if(delta2 <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 1 halo at the condition mass if delta is exceeded
    if(delta2 > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma2,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2))
            return nion_fraction(M_cond,&params); //NOTE: condition mass is used as if it were Lagrangian (no 1+delta)
        else
            return 0.;
    }

    // LOG_ULTRA_DEBUG("params: D=%.2e Mtl=%.2e as=%.2e ae=%.2e fs=%.2e fe=%.2e Ms=%.2e Me=%.2e sig=%.2e del=%.2e",
    //     growthf,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,sigma2,delta2);

    // LOG_ULTRA_DEBUG("--> %.8e",IntegratedNdM(lnM1,lnM2,params,-3, method));

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
            Sigma_InterpTable.y_arr[i] = sigma_z0(Mass);
            dSigmasqdm_InterpTable.y_arr[i] = log10(-dsigmasqdm_z0(Mass));
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
    //NOTE: should we allow del < 0??? We would need to change dfcolldz to prevent zero dfcoll
    // if(del < FRACT_FLOAT_ERR){
    //     return 1.;
    // }
    return splined_erfc(del / (sqrt(2)*sig));
}

/* Uses sigma parameters instead of Mass for scale */
double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias){
    return FgtrM_bias_fast(dicke(z),del_bias,sigsmallR,sig_bias);
}

double FgtrM_bias(double z, double M, double del_bias, double sig_bias){
    return sigmaparam_FgtrM_bias(z,EvaluateSigma(log(M)),del_bias,sig_bias);
}

//  Redshift derivative of the conditional collapsed fraction
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

                log10phi[i + i_z*nbins] = log10( unconditional_mf(growthf,lnMhalo_i,z_LF[i_z],mf) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );

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

                dndm = unconditional_mf(growthf, log(Mhalo_param[i]),z_LF[i_z], mf);
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

double EvaluateSigma(double lnM){
    //using log units to make the fast option faster and the slow option slower
    if(user_params_ps->USE_INTERPOLATION_TABLES) {
        return EvaluateRGTable1D_f(lnM, &Sigma_InterpTable);
    }
    return sigma_z0(exp(lnM));
}

double EvaluatedSigmasqdm(double lnM){
    //this may be slow, figure out why the dsigmadm table is in log10
    if(user_params_ps->USE_INTERPOLATION_TABLES){
        return -pow(10., EvaluateRGTable1D_f(lnM, &dSigmasqdm_InterpTable));
    }
    return dsigmasqdm_z0(exp(lnM));
}

//set the minimum source mass for the integrals, If we have an exponential cutoff we go below the chosen mass by a factor of 50
//NOTE: previously, with USE_MINI_HALOS, the sigma table was initialised with M_MIN_INTEGRAL/50, but then all integrals perofmed
//      from M_MIN_INTEGRAL
double minimum_source_mass(double redshift, bool xray, struct AstroParams *astro_params, struct FlagOptions *flag_options){
    double Mmin,min_factor,mu_factor,t_vir_min;
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
         //NOTE: previously this divided Mturn by 50 in spin temperature, but not in the ionised box
         //     which I think is a bug with M_MIN_in_Mass, since there is a sharp cutoff
        Mmin = astro_params->M_TURN;
    }
    else {
        //if the virial temp minimum is set below ionisation we need to set mu accordingly
        t_vir_min = xray ? astro_params->X_RAY_Tvir_MIN : astro_params->ION_Tvir_MIN;
        mu_factor = t_vir_min < 9.99999e3 ? 1.22 : 0.6;
        Mmin = TtoM(redshift, t_vir_min, mu_factor);
    }

    //This is mostly unused and needs to be tested
    if(global_params.P_CUTOFF){
        Mmin = fmax(Mmin,M_J_WDM());
    }

    Mmin /= min_factor;

    return Mmin;
}
