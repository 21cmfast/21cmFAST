
#define A_NPTS (int) (60) /*Warning: the calculation of the MHR model parameters is valid only from redshift 2 to A_NPTS+2*/
static double A_table[A_NPTS], A_params[A_NPTS];
static gsl_interp_accel *A_acc;
static gsl_spline *A_spline;

#define C_NPTS (int) (12)
static double C_table[C_NPTS], C_params[C_NPTS];
static gsl_interp_accel *C_acc;
static gsl_spline *C_spline;

#define beta_NPTS (int) (5)
static double beta_table[beta_NPTS], beta_params[beta_NPTS];
static gsl_interp_accel *beta_acc;
static gsl_spline *beta_spline;

#define RR_Z_NPTS (int) (300) // number of points in redshift axis;  we will only interpolate over gamma, and just index sample in redshift
#define RR_DEL_Z (float) (0.2)
#define RR_lnGamma_NPTS (int) (200) // number of samples of gamma for the interpolation tables
#define RR_lnGamma_min (double) (-10) // min ln gamma12 used
#define RR_DEL_lnGamma (float) (0.1)
static double RR_table[RR_Z_NPTS][RR_lnGamma_NPTS], lnGamma_values[RR_lnGamma_NPTS];
static gsl_interp_accel *RR_acc[RR_Z_NPTS];
static gsl_spline *RR_spline[RR_Z_NPTS];


/***  FUNCTION PROTOTYPES ***/
double splined_recombination_rate(double z_eff, double gamma12_bg); // assumes T=1e4 and case B

double recombination_rate(double z_eff, double gamma12_bg, double T4, int usecaseB);
void init_MHR(); /*initializes the lookup table for the PDF density integral in MHR00 model at redshift z*/
void free_MHR(); /* deallocates the gsl structures from init_MHR */
double Gamma_SS(double Gamma_bg, double Delta, double T_4, double z);//ionization rate w. self shielding
double MHR_rr (double del, void *params);
double A_MHR(double z); /*returns the A parameter in MHR00model*/
double C_MHR(double z); /*returns the C parameter in MHR00model*/
double beta_MHR(double z); /*returns the beta parameter in MHR00model*/
double splined_A_MHR(double z); /*returns the splined A parameter in MHR00model*/
double splined_C_MHR(double z); /*returns the splined C parameter in MHR00model*/
double splined_beta_MHR(double z);/*returns the splined beta parameter in MHR00*/
void free_A_MHR(); /* deallocates the gsl structures from init_A */
void free_C_MHR(); /* deallocates the gsl structures from init_C */
void free_beta_MHR(); /* deallocates the gsl structures from init_beta */
void init_A_MHR(); /*initializes the lookup table for the A paremeter in MHR00 model*/
void init_C_MHR(); /*initializes the lookup table for the C paremeter in MHR00 model*/
void init_beta_MHR(); /*initializes the lookup table for the beta paremeter in MHR00 model*/


double splined_recombination_rate(double z_eff, double gamma12_bg){
  int z_ct = (int) (z_eff / RR_DEL_Z + 0.5); // round to nearest int
  double lnGamma = log(gamma12_bg);
    
  // check out of bounds
  if ( z_ct < 0 ){ // out of array bounds
//    fprintf(stderr, "WARNING: splined_recombination_rate: effective redshift %g is outside of array bouds\n", z_eff);
    z_ct = 0;
  }
  else if (z_ct  >= RR_Z_NPTS){
//    fprintf(stderr, "WARNING: splined_recombination_rate: effective redshift %g is outside of array bouds\n", z_eff);
    z_ct = RR_Z_NPTS-1;
  }

  if (lnGamma < RR_lnGamma_min){
    return 0;
  }
  else if (lnGamma >= (RR_lnGamma_min + RR_DEL_lnGamma * RR_lnGamma_NPTS) ){
//    fprintf(stderr, "WARNING: splined_recombination_rate: Gamma12 of %g is outside of interpolation array\n", gamma12_bg);
    lnGamma =  RR_lnGamma_min + RR_DEL_lnGamma * RR_lnGamma_NPTS - FRACT_FLOAT_ERR;
  }

  return gsl_spline_eval(RR_spline[z_ct], lnGamma, RR_acc[z_ct]);
}

void init_MHR(){
  int z_ct, gamma_ct;
  float z, gamma;

  // first initialize the MHR parameter look up tables
  init_C_MHR(); /*initializes the lookup table for the C paremeter in MHR00 model*/
  init_beta_MHR(); /*initializes the lookup table for the beta paremeter in MHR00 model*/
  init_A_MHR(); /*initializes the lookup table for the A paremeter in MHR00 model*/

  // now the recombination rate look up tables
  for (z_ct=0; z_ct < RR_Z_NPTS; z_ct++){

    z = z_ct * RR_DEL_Z; // redshift corresponding to index z_ct of the array

    // Intialize the Gamma values
    for (gamma_ct=0; gamma_ct < RR_lnGamma_NPTS; gamma_ct++){
      lnGamma_values[gamma_ct] = RR_lnGamma_min  + gamma_ct*RR_DEL_lnGamma;  // ln of Gamma12    
      gamma = exp(lnGamma_values[gamma_ct]);
      RR_table[z_ct][gamma_ct] = recombination_rate(z, gamma, 1, 1); // CHANGE THIS TO INCLUDE TEMPERATURE
    }

    // set up the spline in gamma
    RR_acc[z_ct] = gsl_interp_accel_alloc();
    RR_spline[z_ct] = gsl_spline_alloc (gsl_interp_cspline, RR_lnGamma_NPTS);
    gsl_spline_init(RR_spline[z_ct], lnGamma_values, RR_table[z_ct], RR_lnGamma_NPTS);

  } // go to next redshift

  return;
}

void free_MHR(){
  int z_ct;

  free_A_MHR(); 
  free_C_MHR(); 
  free_beta_MHR();

  // now the recombination rate look up tables
  for (z_ct=0; z_ct < RR_Z_NPTS; z_ct++){
    gsl_spline_free (RR_spline[z_ct]);
    gsl_interp_accel_free(RR_acc[z_ct]);
  }

  return;
}

//calculates the attenuated photoionization rate due to self-shielding (in units of 1e-12 s^-1)
// input parameters are the background ionization rate, overdensity, temperature (in 10^4k), redshift, respectively
//  Uses the fitting formula from Rahmati et al, assuming a UVB power law index of alpha=5
double Gamma_SS(double Gamma_bg, double del, double T_4, double z){
  double D_ss = 26.7*pow(T_4, 0.17) * pow( (1+z)/10.0, -3) * pow(Gamma_bg, 2.0/3.0);
  return Gamma_bg * (0.98 * pow( (1.0+pow(del/D_ss, 1.64)), -2.28) + 0.02*pow( 1.0+del/D_ss, -0.84));
}


typedef struct {double z, gamma12_bg, T4, A, C_0, beta, avenH; int usecaseB;} RR_par;

double MHR_rr (double lnD, void *params){
  double del=exp(lnD);
  double alpha;
  RR_par p = *(RR_par *) params;
  double z = p.z;
  double gamma = Gamma_SS(p.gamma12_bg, del, p.T4, z);
  double n_H = p.avenH*del;
  double x_e = 1.0 - neutral_fraction(n_H, p.T4, gamma, p.usecaseB);
  double PDelta;

  PDelta = p.A * exp( - 0.5*pow((pow(del,-2.0/3.0)- p.C_0 ) / ((2.0*7.61/(3.0*(1.0+z)))), 2)) * pow(del, p.beta);
    
  if (p.usecaseB)
    alpha = alpha_B(p.T4*1e4);
  else
    alpha = alpha_A(p.T4*1e4);

  //  fprintf(stderr, "%g\t%g\t%g\t%g\t%g\n", n_H, PDelta, alpha, x_e, D);
  
  return n_H * PDelta * alpha * x_e * x_e * del * del;//note extra D since we are integrating over lnD
}


// returns the recombination rate per baryon (1/s), integrated over the MHR density PDF,
// given an ionizing background of gamma12_bg
// temeperature T4 (in 1e4 K), and usecaseB rate coefficient
// Assumes self-shielding according to Rahmati+ 2013
double recombination_rate(double z, double gamma12_bg, double T4, int usecaseB){
  double result, error, lower_limit, upper_limit, A, C_0, beta, avenH;
  gsl_function F;
  double rel_tol  = 0.01; //<- relative tolerance
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
  RR_par p = {z, gamma12_bg, T4, A_MHR(z), C_MHR(z), beta_MHR(z), No*pow( 1+z, 3), usecaseB};

  F.function = &MHR_rr;
  F.params=&p;
  lower_limit = log(0.01);
  upper_limit = log(200);
			   
  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
		       1000, GSL_INTEG_GAUSS61, w, &result, &error); 
  gsl_integration_workspace_free (w);

  return result;
}

double aux_function(double del, void *params){
  double result;
  double z = *(double *) params;
  
  result = exp(-(pow(del,-2.0/3.0)-C_MHR(z))*(pow(del,-2.0/3.0)-C_MHR(z))/(2.0*(2.0*7.61/(3.0*(1.0+z)))*(2.0*7.61/(3.0*(1.0+z)))))*pow(del, beta_MHR(z));

  return result;
}

double A_aux_integral(double z){
  double result, error, lower_limit, upper_limit;
  gsl_function F;
  double rel_tol  = 0.001; //<- relative tolerance
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

  F.function = &aux_function;
  F.params = &z;
  lower_limit = 1e-25;
  upper_limit = 1e25;
			   
  gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
		       1000, GSL_INTEG_GAUSS61, w, &result, &error); 
  gsl_integration_workspace_free (w);

  return result;
}

double A_MHR(double z){
  double result;
  if(z>=2.0+(float)A_NPTS)
    result = splined_A_MHR(2.0+(float)A_NPTS);
  else
    if(z<=2.0)
      result = splined_A_MHR(2.0);
    else
      result = splined_A_MHR(z);
  return result;
}

void init_A_MHR(){
/* initialize the lookup table for the parameter A in the MHR00 model */
   int i;
   
   for (i=0; i<A_NPTS; i++){
     A_params[i] = 2.0+(float)i;
     A_table[i] = 1.0/A_aux_integral(2.0+(float)i);
   }
 
  // Set up spline table
  A_acc   = gsl_interp_accel_alloc();
  A_spline  = gsl_spline_alloc (gsl_interp_cspline, A_NPTS);
  gsl_spline_init(A_spline, A_params, A_table, A_NPTS);

  return;
 }

 
double splined_A_MHR(double x){
  return gsl_spline_eval(A_spline, x, A_acc);
}

void free_A_MHR(){

  gsl_spline_free (A_spline);
  gsl_interp_accel_free(A_acc);
  
  return;
}



double C_MHR(double z){
  double result;
  if(z>=13.0)
    result = 1.0;
  else
    if(z<=2.0)
      result = 0.558;
    else
      result = splined_C_MHR(z);
  return result;
}

void init_C_MHR(){
/* initialize the lookup table for the parameter C in the MHR00 model */
   int i;
   
  for (i=0; i<C_NPTS; i++)
    C_params[i] = (float)i+2.0;

  C_table[0] = 0.558;
  C_table[1] = 0.599;
  C_table[2] = 0.611;
  C_table[3] = 0.769;
  C_table[4] = 0.868;
  C_table[5] = 0.930;
  C_table[6] = 0.964;
  C_table[7] = 0.983;
  C_table[8] = 0.993;
  C_table[9] = 0.998;
  C_table[10] = 0.999;
  C_table[11] = 1.00;
   
  // Set up spline table
  C_acc   = gsl_interp_accel_alloc ();
  C_spline  = gsl_spline_alloc (gsl_interp_cspline, C_NPTS);
  gsl_spline_init(C_spline, C_params, C_table, C_NPTS);

  return;
 }

 
double splined_C_MHR(double x){
  return gsl_spline_eval(C_spline, x, C_acc);
}

void free_C_MHR(){

  gsl_spline_free (C_spline);
  gsl_interp_accel_free(C_acc);
  
  return;
}



double beta_MHR(double z){
  double result;
  if(z>=6.0)
    result = -2.50;
  else
    if(z<=2.0)
      result = -2.23;
    else
      result = splined_beta_MHR(z);
  return result;
}

void init_beta_MHR(){
/* initialize the lookup table for the parameter C in the MHR00 model */
   int i;
   
  for (i=0; i<beta_NPTS; i++)
    beta_params[i] = (float)i+2.0;

  beta_table[0] = -2.23;
  beta_table[1] = -2.35;
  beta_table[2] = -2.48;
  beta_table[3] = -2.49;
  beta_table[4] = -2.50;
   
  // Set up spline table
  beta_acc   = gsl_interp_accel_alloc ();
  beta_spline  = gsl_spline_alloc (gsl_interp_cspline, beta_NPTS);
  gsl_spline_init(beta_spline, beta_params, beta_table, beta_NPTS);

  return;
 }


double splined_beta_MHR(double x){
  return gsl_spline_eval(beta_spline, x, beta_acc);
}

void free_beta_MHR(){

  gsl_spline_free(beta_spline);
  gsl_interp_accel_free(beta_acc);
  
  return;
}
