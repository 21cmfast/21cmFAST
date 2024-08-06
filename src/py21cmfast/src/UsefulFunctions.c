// -------------------------------------------------------------------------------------
// Taken from COSMOLOGY.H
// -------------------------------------------------------------------------------------
#define Ho  (double) (cosmo_params_ufunc->hlittle*3.2407e-18) // s^-1 at z=0
#define RHOcrit (double) ( (3.0*Ho*Ho / (8.0*PI*G)) * (CMperMPC*CMperMPC*CMperMPC)/Msun) // Msun Mpc^-3 ---- at z=0
#define RHOcrit_cgs (double) (3.0*Ho*Ho / (8.0*PI*G)) // g pcm^-3 ---- at z=0
#define No  (double) (RHOcrit_cgs*cosmo_params_ufunc->OMb*(1-global_params.Y_He)/m_p)  //  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7
#define He_No (double) (RHOcrit_cgs*cosmo_params_ufunc->OMb*global_params.Y_He/(4.0*m_p)) //  current helium number density estimate
#define N_b0 (double) (No+He_No) // present-day baryon num density, H + He
#define f_H (double) (No/(No+He_No))  // hydrogen number fraction
#define f_He (double) (He_No/(No+He_No))  // helium number fraction

struct CosmoParams *cosmo_params_ufunc;
struct UserParams *user_params_ufunc;

void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params){
    cosmo_params_ufunc = cosmo_params;
    user_params_ufunc = user_params;
}

float ComputeFullyIoinizedTemperature(float z_re, float z, float delta){
    // z_re: the redshift of reionization
    // z:    the current redshift
    // delta:the density contrast
    float result, delta_re;
    // just be fully ionized
    if (fabs(z - z_re) < 1e-4)
        result = 1;
    else{
        // linearly extrapolate to get density at reionization
        delta_re = delta * (1. + z ) / (1. + z_re);
        if (delta_re<=-1) delta_re=-1. + global_params.MIN_DENSITY_LOW_LIMIT;
        // evolving ionized box eq. 6 of McQuinn 2015, ignored the dependency of density at ionization
        if (delta<=-1) delta=-1. + global_params.MIN_DENSITY_LOW_LIMIT;
        result  = pow((1. + delta) / (1. + delta_re), 1.1333);
        result *= pow((1. + z) / (1. + z_re), 3.4);
        result *= expf(pow((1. + z)/7.1, 2.5) - pow((1. + z_re)/7.1, 2.5));
    }
    result *= pow(global_params.T_RE, 1.7);
    // 1e4 before helium reionization; double it after
    result += pow(1e4 * ((1. + z)/4.), 1.7) * ( 1 + delta);
    result  = pow(result, 0.5882);
    //LOG_DEBUG("z_re=%.4f, z=%.4f, delta=%e, Tk=%.f", z_re, z, delta, result);
    return result;
}

float ComputePartiallyIoinizedTemperature(float T_HI, float res_xH){
    if (res_xH<=0.) return global_params.T_RE;
    if (res_xH>=1) return T_HI;

    return T_HI * res_xH + global_params.T_RE * (1. - res_xH);
}

void filter_box(fftwf_complex *box, int RES, int filter_type, float R){
    int n_x, n_z, n_y, dimension,midpoint;
    float k_x, k_y, k_z, k_mag, kR;

    switch(RES) {
        case 0:
            dimension = user_params_ufunc->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = user_params_ufunc->HII_DIM;
            midpoint = HII_MIDDLE;
            break;
    }

    // loop through k-box

#pragma omp parallel shared(box) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,kR) num_threads(user_params_ufunc->N_THREADS)
    {
#pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
//        for (n_x=dimension; n_x--;){
            if (n_x>midpoint) {k_x =(n_x-dimension) * DELTA_K;}
            else {k_x = n_x * DELTA_K;}

            for (n_y=0; n_y<dimension; n_y++){
//            for (n_y=dimension; n_y--;){
                if (n_y>midpoint) {k_y =(n_y-dimension) * DELTA_K;}
                else {k_y = n_y * DELTA_K;}

//                for (n_z=(midpoint+1); n_z--;){
                for (n_z=0; n_z<=(unsigned long long)(user_params_ufunc->NON_CUBIC_FACTOR*midpoint); n_z++){
                    k_z = n_z * DELTA_K_PARA;

                    if (filter_type == 0){ // real space top-hat

                        k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                        kR = k_mag*R; // real space top-hat

                        if (kR > 1e-4){
                            if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR); }
                            if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR); }
                        }
                    }
                    else if (filter_type == 1){ // k-space top hat

                        // This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
                        // as we don't need to evaluate the slower sqrt function
//                        kR = 0.17103765852*( k_x*k_x + k_y*k_y + k_z*k_z )*R*R;

                        k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                        kR = k_mag*R; // real space top-hat

                        kR *= 0.413566994; // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
                        if (kR > 1){
                            if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] = 0; }
                            if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] = 0; }
                        }
                    }
                    else if (filter_type == 2){ // gaussian
                        // This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
                        // as we don't need to evaluate the slower sqrt function
                        kR = 0.643*0.643*( k_x*k_x + k_y*k_y + k_z*k_z )*R*R;
//                        kR *= 0.643; // equates integrated volume to the real space top-hat
                        if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR/2.0); }
                        if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR/2.0); }
                    }
                    else{
                        if ( (n_x==0) && (n_y==0) && (n_z==0) )
                            LOG_WARNING("Filter type %i is undefined. Box is unfiltered.", filter_type);
                    }
                }
            }
        } // end looping through k box
    }

    return;
}

double MtoR(double M);
double RtoM(double R);
double TtoM(double z, double T, double mu);
double dicke(double z);
double dtdz(float z);
double ddickedt(double z);
double omega_mz(float z);
double Deltac_nonlinear(float z);
double drdz(float z); /* comoving distance, (1+z)*C*dtdz(in cm) per unit z */
double alpha_A(double T);
/* returns the case B hydrogen recombination coefficient (Spitzer 1978) in cm^3 s^-1*/
double alpha_B(double T);

double HeI_ion_crosssec(double nu);
double HeII_ion_crosssec(double nu);
double HI_ion_crosssec(double nu);



/* R in Mpc, M in Msun */
double MtoR(double M){

    // set R according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (global_params.FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return pow(3*M/(4*PI*cosmo_params_ufunc->OMm*RHOcrit), 1.0/3.0);
    else if (global_params.FILTER == 1) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow( M/(pow(2*PI, 1.5) * cosmo_params_ufunc->OMm * RHOcrit), 1.0/3.0 );
    else // filter not defined
        LOG_ERROR("No such filter = %i. Results are bogus.", global_params.FILTER);
    Throw ValueError;
}

/* R in Mpc, M in Msun */
double RtoM(double R){
    // set M according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (global_params.FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return (4.0/3.0)*PI*pow(R,3)*(cosmo_params_ufunc->OMm*RHOcrit);
    else if (global_params.FILTER == 1) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow(2*PI, 1.5) * cosmo_params_ufunc->OMm*RHOcrit * pow(R, 3);
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
    return 7030.97 / (cosmo_params_ufunc->hlittle) * sqrt( omega_mz(z) / (cosmo_params_ufunc->OMm*Deltac_nonlinear(z))) *
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
    return cosmo_params_ufunc->OMm*pow(1+z,3) / (cosmo_params_ufunc->OMm*pow(1+z,3) + cosmo_params_ufunc->OMl + global_params.OMr*pow(1+z,4) + global_params.OMk*pow(1+z, 2));
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

    if (fabs(cosmo_params_ufunc->OMm-1.0) < tiny){ //OMm = 1 (Einstein de-Sitter)
        return 1.0/(1.0+z);
    }
    else if ( (cosmo_params_ufunc->OMl > (-tiny)) && (fabs(cosmo_params_ufunc->OMl+cosmo_params_ufunc->OMm+global_params.OMr-1.0) < 0.01) && (fabs(global_params.wl+1.0) < tiny) ){
        //this is a flat, cosmological CONSTANT universe, with only lambda, matter and radiation
        //it is taken from liddle et al.
        omegaM_z = cosmo_params_ufunc->OMm*pow(1+z,3) / ( cosmo_params_ufunc->OMl + cosmo_params_ufunc->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4) );
        dick_z = 2.5*omegaM_z / ( 1.0/70.0 + omegaM_z*(209-omegaM_z)/140.0 + pow(omegaM_z, 4.0/7.0) );
        dick_0 = 2.5*cosmo_params_ufunc->OMm / ( 1.0/70.0 + cosmo_params_ufunc->OMm*(209-cosmo_params_ufunc->OMm)/140.0 + pow(cosmo_params_ufunc->OMm, 4.0/7.0) );
        return dick_z / (dick_0 * (1.0+z));
    }
    else if ( (global_params.OMtot < (1+tiny)) && (fabs(cosmo_params_ufunc->OMl) < tiny) ){ //open, zero lambda case (peebles, pg. 53)
        x_0 = 1.0/(cosmo_params_ufunc->OMm+0.0) - 1.0;
        dick_0 = 1 + 3.0/x_0 + 3*log(sqrt(1+x_0)-sqrt(x_0))*sqrt(1+x_0)/pow(x_0,1.5);
        x = fabs(1.0/(cosmo_params_ufunc->OMm+0.0) - 1.0) / (1+z);
        dick_z = 1 + 3.0/x + 3*log(sqrt(1+x)-sqrt(x))*sqrt(1+x)/pow(x,1.5);
        return dick_z/dick_0;
    }
    else if ( (cosmo_params_ufunc->OMl > (-tiny)) && (fabs(global_params.OMtot-1.0) < tiny) && (fabs(global_params.wl+1) > tiny) ){
        LOG_WARNING("IN WANG.");
        Throw ValueError;
    }

    LOG_ERROR("No growth function!");
    Throw ValueError;
}

/* function DTDZ returns the value of dt/dz at the redshift parameter z. */
double dtdz(float z){
    double x, dxdz, const1, denom, numer;
    x = sqrt( cosmo_params_ufunc->OMl/cosmo_params_ufunc->OMm ) * pow(1+z, -3.0/2.0);
    dxdz = sqrt( cosmo_params_ufunc->OMl/cosmo_params_ufunc->OMm ) * pow(1+z, -5.0/2.0) * (-3.0/2.0);
    const1 = 2 * sqrt( 1 + cosmo_params_ufunc->OMm/cosmo_params_ufunc->OMl ) / (3.0 * Ho) ;

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

    if (fabs(cosmo_params_ufunc->OMm-1.0) < tiny){ //OMm = 1 (Einstein de-Sitter)
        return -pow(1+z,-2)/dtdz(z);
    }
    else if ( (cosmo_params_ufunc->OMl > (-tiny)) && (fabs(cosmo_params_ufunc->OMl+cosmo_params_ufunc->OMm+global_params.OMr-1.0) < 0.01) && (fabs(global_params.wl+1.0) < tiny) ){
        //this is a flat, cosmological CONSTANT universe, with only lambda, matter and radiation
        //it is taken from liddle et al.
        omegaM_z = cosmo_params_ufunc->OMm*pow(1+z,3) / ( cosmo_params_ufunc->OMl + cosmo_params_ufunc->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4) );
        domegaMdz = omegaM_z*3/(1+z) - cosmo_params_ufunc->OMm*pow(1+z,3)*pow(cosmo_params_ufunc->OMl + cosmo_params_ufunc->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4), -2) * (3*cosmo_params_ufunc->OMm*(1+z)*(1+z) + 4*global_params.OMr*pow(1+z,3));
        dick_0 = cosmo_params_ufunc->OMm / ( 1.0/70.0 + cosmo_params_ufunc->OMm*(209-cosmo_params_ufunc->OMm)/140.0 + pow(cosmo_params_ufunc->OMm, 4.0/7.0) );

        ddickdz = (domegaMdz/(1+z)) * (1.0/70.0*pow(omegaM_z,-2) + 1.0/140.0 + 3.0/7.0*pow(omegaM_z, -10.0/3.0)) * pow(1.0/70.0/omegaM_z + (209.0-omegaM_z)/140.0 + pow(omegaM_z, -3.0/7.0) , -2);
        ddickdz -= pow(1+z,-2)/(1.0/70.0/omegaM_z + (209.0-omegaM_z)/140.0 + pow(omegaM_z, -3.0/7.0));

        return ddickdz / dick_0 / dtdz(z);
    }

    LOG_ERROR("No growth function!");
    Throw ValueError;
}

/* returns the hubble "constant" (in 1/sec) at z */
double hubble(float z){
    return Ho*sqrt(cosmo_params_ufunc->OMm*pow(1+z,3) + global_params.OMr*pow(1+z,4) + cosmo_params_ufunc->OMl);
}


/* returns hubble time (in sec), t_h = 1/H */
double t_hubble(float z){
    return 1.0/hubble(z);
}

/* comoving distance (in cm) per unit redshift */
double drdz(float z){
    return (1.0+z)*C*dtdz(z);
}

/* returns the case A hydrogen recombination coefficient (Abel et al. 1997) in cm^3 s^-1*/
double alpha_A(double T){
    double logT, ans;
    logT = log(T/(double)1.1604505e4);
    ans = pow(E, -28.6130338 - 0.72411256*logT - 2.02604473e-2*pow(logT, 2)
              - 2.38086188e-3*pow(logT, 3) - 3.21260521e-4*pow(logT, 4)
              - 1.42150291e-5*pow(logT, 5) + 4.98910892e-6*pow(logT, 6)
              + 5.75561414e-7*pow(logT, 7) - 1.85676704e-8*pow(logT, 8)
              - 3.07113524e-9 * pow(logT, 9));
    return ans;
}

/* returns the case B hydrogen recombination coefficient (Spitzer 1978) in cm^3 s^-1*/
double alpha_B(double T){
    return alphaB_10k * pow (T/1.0e4, -0.75);
}


/*
 Function NEUTRAL_FRACTION returns the hydrogen neutral fraction, chi, given:
 hydrogen density (pcm^-3)
 gas temperature (10^4 K)
 ionization rate (1e-12 s^-1)
 */
double neutral_fraction(double density, double T4, double gamma, int usecaseB){
    double chi, b, alpha, corr_He = 1.0/(4.0/global_params.Y_He - 3);

    if (usecaseB)
        alpha = alpha_B(T4*1e4);
    else
        alpha = alpha_A(T4*1e4);

    gamma *= 1e-12;

    // approximation chi << 1
    chi = (1+corr_He)*density * alpha / gamma;
    if (chi < TINY){ return 0;}
    if (chi < 1e-5)
        return chi;

    //  this code, while mathematically accurate, is numerically buggy for very small x_HI, so i will use valid approximation x_HI <<1 above when x_HI < 1e-5, and this otherwise... the two converge seemlessly
    //get solutions of quadratic of chi (neutral fraction)
    b = -2 - gamma / (density*(1+corr_He)*alpha);
    chi = ( -b - sqrt(b*b - 4) ) / 2.0; //correct root
    return chi;
}

/* function HeI_ion_crosssec returns the HI ionization cross section at parameter frequency
 (taken from Verner et al (1996) */
double HeI_ion_crosssec(double nu){
    double x,y,Fy;

    if (nu < HeI_NUIONIZATION)
        return 0;

    x = nu/NU_over_EV/13.61 - 0.4434;
    y = sqrt(x*x + pow(2.136, 2));
    return  9.492e-16*((x-1)*(x-1) + 2.039*2.039) *
    pow(y, (0.5 * 3.188 - 5.5))
    * pow(1.0 + sqrt(y/1.469), -3.188);
}


/* function HeII_ion_crosssec returns the HeII ionization cross section at parameter frequency
 (taken from Osterbrock, pg. 14) */
double HeII_ion_crosssec(double nu){
    double epsilon, Z = 2;

    if (nu < HeII_NUIONIZATION)
        return 0;

    if (nu == HeII_NUIONIZATION)
        nu+=TINY;

    epsilon = sqrt( nu/HeII_NUIONIZATION - 1);
    return (6.3e-18)/Z/Z * pow(HeII_NUIONIZATION/nu, 4)
    * pow(E, 4-(4*atan(epsilon)/epsilon)) / (1-pow(E, -2*PI/epsilon));
}


/* function HI_ion_crosssec returns the HI ionization cross section at parameter frequency
 (taken from Osterbrock, pg. 14) */
double HI_ion_crosssec(double nu){
    double epsilon, Z = 1;

    if (nu < NUIONIZATION)
        return 0;

    if (nu == NUIONIZATION)
        nu+=TINY;

    epsilon = sqrt( nu/NUIONIZATION - 1);
    return (6.3e-18)/Z/Z * pow(NUIONIZATION/nu, 4)
    * pow(E, 4-(4*atan(epsilon)/epsilon)) / (1-pow(E, -2*PI/epsilon));
}



/* Return the thomspon scattering optical depth from zstart to zend through fully ionized IGM.
 The hydrogen reionization history is given by the zarry and xHarry parameters, in increasing
 redshift order of length len.*/
typedef struct{
    float *z, *xH;
    int len;
} tau_e_params;
double dtau_e_dz(double z, void *params){
    float xH, xi;
    int i=1;
    tau_e_params p = *(tau_e_params *)params;

    if ((p.len == 0) || !(p.z)) {
        return (1+z)*(1+z)*drdz(z);
    }
    else{
        // find where we are in the redshift array
        if (p.z[0]>z) // ionization fraction is 1 prior to start of array
            return (1+z)*(1+z)*drdz(z);
        while ( (i < p.len) && (p.z[i] < z) ) {i++;}
        if (i == p.len)
            return 0;

        // linearly interpolate in redshift
        xH = p.xH[i-1] + (p.xH[i] - p.xH[i-1])/(p.z[i] - p.z[i-1]) * (z - p.z[i-1]);
        xi = 1.0-xH;
        if (xi<0){
            LOG_WARNING("in taue: funny business xi=%e, changing to 0.", xi);
            xi=0;
        }
        if (xi>1){
            LOG_WARNING("in taue: funny business xi=%e, changing to 1", xi);
            xi=1;
        }

        return xi*(1+z)*(1+z)*drdz(z);
    }
}
double tau_e(float zstart, float zend, float *zarry, float *xHarry, int len){
    double prehelium, posthelium, error;
    gsl_function F;
    double rel_tol  = 1e-3; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    tau_e_params p;

    if (zstart >= zend){
        LOG_ERROR("in tau_e: First parameter must be smaller than the second.\n");
        Throw ValueError;
    }

    F.function = &dtau_e_dz;
    p.z = zarry;
    p.xH = xHarry;
    p.len = len;
    F.params = &p;
    if ((len > 0) && zarry)
        zend = zarry[len-1] - FRACT_FLOAT_ERR;

    int status;

    gsl_set_error_handler_off();

    if (zend > global_params.Zreion_HeII){// && (zstart < Zreion_HeII)){
        if (zstart < global_params.Zreion_HeII){
            status = gsl_integration_qag (&F, global_params.Zreion_HeII, zstart, 0, rel_tol,
                                 1000, GSL_INTEG_GAUSS61, w, &prehelium, &error);

            if(status!=0) {
                LOG_ERROR("gsl integration error occured!");
                LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",global_params.Zreion_HeII,zstart,rel_tol,prehelium,error);
                LOG_ERROR("data: zstart=%e zend=%e",zstart,zend);
                GSL_ERROR(status);
            }

            status = gsl_integration_qag (&F, zend, global_params.Zreion_HeII, 0, rel_tol,
                                 1000, GSL_INTEG_GAUSS61, w, &posthelium, &error);

            if(status!=0) {
                LOG_ERROR("gsl integration error occured!");
                LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",zend,global_params.Zreion_HeII,rel_tol,posthelium,error);
                LOG_ERROR("data: zstart=%e zend=%e",zstart,zend);
                GSL_ERROR(status);
            }
        }
        else{
            prehelium = 0;
            status = gsl_integration_qag (&F, zend, zstart, 0, rel_tol,
                                 1000, GSL_INTEG_GAUSS61, w, &posthelium, &error);

            if(status!=0) {
                LOG_ERROR("gsl integration error occured!");
                LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",zend,zstart,rel_tol,posthelium,error);
                GSL_ERROR(status);
            }
        }
    }
    else{
        posthelium = 0;
        status = gsl_integration_qag (&F, zend, zstart, 0, rel_tol,
                             1000, GSL_INTEG_GAUSS61, w, &prehelium, &error);

        if(status!=0) {
            LOG_ERROR("gsl integration error occured!");
            LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",zend,zstart,rel_tol,prehelium,error);
            GSL_ERROR(status);
        }
    }
    gsl_integration_workspace_free (w);

    return SIGMAT * ( (N_b0+He_No)*prehelium + N_b0*posthelium );
}

float ComputeTau(struct UserParams *user_params, struct CosmoParams *cosmo_params, int NPoints, float *redshifts, float *global_xHI) {

    int i;
    float tau;

    Broadcast_struct_global_UF(user_params,cosmo_params);

    tau = tau_e(0, redshifts[NPoints-1], redshifts, global_xHI, NPoints);

    return tau;
}


void writeUserParams(struct UserParams *p){
    LOG_INFO("UserParams: [HII_DIM=%d, DIM=%d, BOX_LEN=%f, NON_CUBIC_FACTOR=%f, HMF=%d, POWER_SPECTRUM=%d, USE_RELATIVE_VELOCITIES=%d, N_THREADS=%d, PERTURB_ON_HIGH_RES=%d, NO_RNG=%d, USE_FFTW_WISDOM=%d, USE_INTERPOLATION_TABLES=%d, FAST_FCOLL_TABLES=%d]",
             p->HII_DIM, p->DIM, p->BOX_LEN, p->NON_CUBIC_FACTOR, p->HMF, p->POWER_SPECTRUM, p->USE_RELATIVE_VELOCITIES, p->N_THREADS, p->PERTURB_ON_HIGH_RES, p->NO_RNG, p->USE_FFTW_WISDOM, p->USE_INTERPOLATION_TABLES, p->FAST_FCOLL_TABLES);
}

void writeCosmoParams(struct CosmoParams *p){
    LOG_INFO("CosmoParams: [SIGMA_8=%f, hlittle=%f, OMm=%f, OMl=%f, OMb=%f, POWER_INDEX=%f]",
             p->SIGMA_8, p->hlittle, p->OMm, p->OMl, p->OMb, p->POWER_INDEX);
}

void writeAstroParams(struct FlagOptions *fo, struct AstroParams *p){

    if(fo->USE_MASS_DEPENDENT_ZETA) {
        LOG_INFO("AstroParams: [HII_EFF_FACTOR=%f, ALPHA_STAR=%f, ALPHA_STAR_MINI=%f, F_ESC10=%f (F_ESC7_MINI=%f), ALPHA_ESC=%f, M_TURN=%f, R_BUBBLE_MAX=%f, L_X=%e (L_X_MINI=%e), NU_X_THRESH=%f, X_RAY_SPEC_INDEX=%f, F_STAR10=%f (F_STAR7_MINI=%f), t_STAR=%f, N_RSD_STEPS=%f]",
             p->HII_EFF_FACTOR, p->ALPHA_STAR, p->ALPHA_STAR_MINI, p->F_ESC10,p->F_ESC7_MINI, p->ALPHA_ESC, p->M_TURN,
             p->R_BUBBLE_MAX, p->L_X, p->L_X_MINI, p->NU_X_THRESH, p->X_RAY_SPEC_INDEX, p->F_STAR10, p->F_STAR7_MINI, p->t_STAR, p->N_RSD_STEPS);
    }
    else {
        LOG_INFO("AstroParams: [HII_EFF_FACTOR=%f, ION_Tvir_MIN=%f, X_RAY_Tvir_MIN=%f, R_BUBBLE_MAX=%f, L_X=%e, NU_X_THRESH=%f, X_RAY_SPEC_INDEX=%f, F_STAR10=%f, t_STAR=%f, N_RSD_STEPS=%f]",
             p->HII_EFF_FACTOR, p->ION_Tvir_MIN, p->X_RAY_Tvir_MIN,
             p->R_BUBBLE_MAX, p->L_X, p->NU_X_THRESH, p->X_RAY_SPEC_INDEX, p->F_STAR10, p->t_STAR, p->N_RSD_STEPS);
    }
}

void writeFlagOptions(struct FlagOptions *p){
    LOG_INFO("FlagOptions: [USE_HALO_FIELD=%d, USE_MINI_HALOS=%d, USE_MASS_DEPENDENT_ZETA=%d, SUBCELL_RSD=%d, INHOMO_RECO=%d, USE_TS_FLUCT=%d, M_MIN_in_Mass=%d, PHOTON_CONS=%d]",
           p->USE_HALO_FIELD, p->USE_MINI_HALOS, p->USE_MASS_DEPENDENT_ZETA, p->SUBCELL_RSD, p->INHOMO_RECO, p->USE_TS_FLUCT, p->M_MIN_in_Mass, p->PHOTON_CONS);
}


char *print_output_header(int print_pid, const char *name){
    char * pid = malloc(12*sizeof(char));

    if(print_pid){
        sprintf(pid, "<%d>\t", getpid());
    }else{
        sprintf(pid, "");
    }

    printf("%s%s:\n", pid, name);
    return (pid);
}


void print_corners_real(float *x, int size, float ncf){
    int s = size-1;
    int s_ncf = size*ncf-1;
    int i,j,k;
    for(i=0;i<size;i=i+s){
        for(j=0;j<size;j=j+s){
            for(k=0;k<(int)(size*ncf);k=k+s_ncf){
                printf("%f, ", x[k + (int)(size*ncf)*(j + size*i)]);
            }
        }
    }
    printf("\n");
}

void debugSummarizeBox(float *box, int size, float ncf, char *indent){
    if(LOG_LEVEL >= SUPER_DEBUG_LEVEL){

        float corners[8];

        int i,j,k, counter;
        int s = size-1;
        int s_ncf = size*ncf-1;

        counter = 0;
        for(i=0;i<size;i=i+s){
            for(j=0;j<size;j=j+s){
                for(k=0;k<(int)(size*ncf);k=k+s_ncf){
                    corners[counter] =  box[k + (int)(size*ncf)*(j + size*i)];
                    counter++;
                }
            }
        }

        LOG_SUPER_DEBUG("%sCorners: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e",
            indent,
            corners[0], corners[1], corners[2], corners[3],
            corners[4], corners[5], corners[6], corners[7]
        );

        float sum, mean, mn, mx;
        sum=0;
        mn=box[0];
        mx=box[0];

        for (i=0; i<size*size*((int)(size*ncf)); i++){
            sum+=box[i];
            mn=fminf(mn, box[i]);
            mx = fmaxf(mx, box[i]);
        }
        mean=sum/(size*size*((int)(size*ncf)));

        LOG_SUPER_DEBUG("%sSum/Mean/Min/Max: %.4e, %.4e, %.4e, %.4e", indent, sum, mean, mn, mx);
    }
}

void debugSummarizeBoxComplex(fftwf_complex *box, int size, float ncf, char *indent){
    if(LOG_LEVEL >= SUPER_DEBUG_LEVEL){

        float corners_real[8];
        float corners_imag[8];

        int i,j,k, counter;
        int s = size-1;
        int s_ncf = size*ncf-1;

        counter = 0;
        for(i=0;i<size;i=i+s){
            for(j=0;j<size;j=j+s){
                for(k=0;k<(int)(size*ncf);k=k+s_ncf){
                    corners_real[counter] =  creal(box[k + (int)(size*ncf)*(j + size*i)]);
                    corners_imag[counter] =  cimag(box[k + (int)(size*ncf)*(j + size*i)]);

                    counter++;
                }
            }
        }

        LOG_SUPER_DEBUG("%sCorners (Real Part): %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e",
            indent,
            corners_real[0], corners_real[1], corners_real[2], corners_real[3],
            corners_real[4], corners_real[5], corners_real[6], corners_real[7]
        );

        LOG_SUPER_DEBUG("%sCorners (Imag Part): %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e",
            indent,
            corners_imag[0], corners_imag[1], corners_imag[2], corners_imag[3],
            corners_imag[4], corners_imag[5], corners_imag[6], corners_imag[7]
        );


        complex sum, mean, mn, mx;
        sum=0+I;
        mn=box[0];
        mx=box[0];

        for (i=0; i<size*size*((int)(size*ncf)); i++){
            sum+=box[i];
            mn=fminf(mn, box[i]);
            mx = fmaxf(mx, box[i]);
        }
        mean=sum/(size*size*((int)(size*ncf)));

        LOG_SUPER_DEBUG(
            "%sSum/Mean/Min/Max: %.4e+%.4ei, %.4e+%.4ei, %.4e+%.4ei, %.4e+%.4ei",
            indent, creal(sum), cimag(sum), creal(mean), cimag(mean), creal(mn), cimag(mn),
            creal(mx), cimag(mx));
    }
}

void debugSummarizeBoxDouble(double *box, int size, float ncf, char *indent){
    if(LOG_LEVEL >= SUPER_DEBUG_LEVEL){

        double corners[8];

        int i,j,k, counter;
        int s = size-1;
        int s_ncf = size*ncf-1;

        counter = 0;
        for(i=0;i<size;i=i+s){
            for(j=0;j<size;j=j+s){
                for(k=0;k<(int)(size*ncf);k=k+s_ncf){
                    corners[counter] =  box[k + (int)(size*ncf)*(j + size*i)];
                    counter++;
                }
            }
        }

        LOG_SUPER_DEBUG("%sCorners: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e",
            indent,
            corners[0], corners[1], corners[2], corners[3],
            corners[4], corners[5], corners[6], corners[7]
        );

        double sum, mean, mn, mx;
        sum=0;
        mn=box[0];
        mx=box[0];

        for (i=0; i<size*size*((int)(size*ncf)); i++){
            sum+=box[i];
            mn=fmin(mn, box[i]);
            mx = fmax(mx, box[i]);
        }
        mean=sum/(size*size*((int)(size*ncf)));

        LOG_SUPER_DEBUG("%sSum/Mean/Min/Max: %.4e, %.4e, %.4e, %.4e", indent, sum, mean, mn, mx);
    }
}

void debugSummarizeIC(struct InitialConditions *x, int HII_DIM, int DIM, float NCF){
    LOG_SUPER_DEBUG("Summary of InitialConditions:");
    LOG_SUPER_DEBUG("  lowres_density: ");
    debugSummarizeBox(x->lowres_density, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  hires_density: ");
    debugSummarizeBox(x->hires_density, DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  lowres_vx: ");
    debugSummarizeBox(x->lowres_vx, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  lowres_vy: ");
    debugSummarizeBox(x->lowres_vy, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  lowres_vz: ");
    debugSummarizeBox(x->lowres_vz, HII_DIM, NCF, "    ");
}

void debugSummarizePerturbField(struct PerturbedField *x, int HII_DIM, float NCF){
    LOG_SUPER_DEBUG("Summary of PerturbedField:");
    LOG_SUPER_DEBUG("  density: ");
    debugSummarizeBox(x->density, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  velocity_x: ");
    debugSummarizeBox(x->velocity_x, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  velocity_y: ");
    debugSummarizeBox(x->velocity_y, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  velocity_z: ");
    debugSummarizeBox(x->velocity_z, HII_DIM, NCF, "    ");

}
void inspectInitialConditions(struct InitialConditions *x, int print_pid, int print_corners, int print_first,
                              int HII_DIM, float NCF){
    int i;
    char *pid = print_output_header(print_pid, "InitialConditions");

    if(print_first){
        printf("%s\tFirstRow: ",pid);

        printf("%s\t\tlowres_density: ");
        for(i=0;i<10;i++){
            printf("%f, ", x->lowres_density[i]);
        }
        printf("\n");

        printf("%s\t\tlowres_vx     : ");
        for(i=0;i<10;i++){
            printf("%f, ", x->lowres_vx[i]);
        }
        printf("\n");

        printf("%s\t\tlowres_vx_2LPT: ");
        for(i=0;i<10;i++){
            printf("%f, ", x->lowres_vx_2LPT[i]);
        }
        printf("\n");
    }

    if(print_corners){
        printf("%s\tCorners: ",pid);

        printf("%s\t\tlowres_density: ",pid);
        print_corners_real(x->lowres_density, HII_DIM, NCF);

        printf("%s\t\tlowres_vx     : ", pid);
        print_corners_real(x->lowres_vx, HII_DIM, NCF);

        printf("%s\t\tlowres_vx_2LPT: ", pid);
        print_corners_real(x->lowres_vx_2LPT, HII_DIM, NCF);
    }
}


void inspectPerturbedField(struct PerturbedField *x, int print_pid, int print_corners, int print_first,
                           int HII_DIM, float NCF){
    int i;
    char *pid = print_output_header(print_pid, "PerturbedField");

    if(print_first){
        printf("%s\tFirstRow: \n",pid);

        printf("%s\t\tdensity: ", pid);
        for(i=0;i<10;i++){
            printf("%f, ", x->density[i]);
        }
        printf("\n");

        printf("%s\t\tvelocity: ", pid);
        for(i=0;i<10;i++){
            printf("%f, ", x->velocity_x[i]);
        }
        printf("\n");

        printf("%s\t\tvelocity: ", pid);
        for(i=0;i<10;i++){
            printf("%f, ", x->velocity_y[i]);
        }
        printf("\n");

        printf("%s\t\tvelocity: ", pid);
        for(i=0;i<10;i++){
            printf("%f, ", x->velocity_z[i]);
        }
        printf("\n");

    }

    if(print_corners){
        printf("%s\tCorners: \n",pid);

        printf("%s\t\tdensity: ",pid);
        print_corners_real(x->density, HII_DIM, NCF);

        printf("%s\t\tvelocity: ", pid);
        print_corners_real(x->velocity_x, HII_DIM, NCF);

        printf("%s\t\tvelocity: ", pid);
        print_corners_real(x->velocity_y, HII_DIM, NCF);

        printf("%s\t\tvelocity: ", pid);
        print_corners_real(x->velocity_z, HII_DIM, NCF);
    }

}


void inspectTsBox(struct TsBox *x, int print_pid, int print_corners, int print_first, int HII_DIM, float NCF){
    int i;
    char *pid = print_output_header(print_pid, "TsBox");

    if(print_first){
        printf("%s\tFirstRow: ",pid);

        printf("%s\t\tTs_box : ");
        for(i=0;i<10;i++){
            printf("%f, ", x->Ts_box[i]);
        }
        printf("\n");

        printf("%s\t\tx_e_box: ");
        for(i=0;i<10;i++){
            printf("%f, ", x->x_e_box[i]);
        }
        printf("\n");

        printf("%s\t\tTk_box : ");
        for(i=0;i<10;i++){
            printf("%f, ", x->Tk_box[i]);
        }
        printf("\n");
    }

    if(print_corners){
        printf("%s\tCorners: ",pid);

        printf("%s\t\tTs_box : ",pid);
        print_corners_real(x->Ts_box, HII_DIM, NCF);

        printf("%s\t\tx_e_box: ", pid);
        print_corners_real(x->x_e_box, HII_DIM, NCF);

        printf("%s\t\tTk_box : ", pid);
        print_corners_real(x->Tk_box, HII_DIM, NCF);
    }
}

void inspectIonizedBox(struct IonizedBox *x, int print_pid, int print_corners, int print_first, int HII_DIM, float NCF){
    int i;
    char *pid = print_output_header(print_pid, "IonizedBox");

    if(print_first){
        printf("%s\tFirstRow: ",pid);

        printf("%s\t\txH_box     : ");
        for(i=0;i<10;i++){
            printf("%f, ", x->xH_box[i]);
        }
        printf("\n");

        printf("%s\t\tGamma12_box: ");
        for(i=0;i<10;i++){
            printf("%f, ", x->Gamma12_box[i]);
        }
        printf("\n");

        printf("%s\t\tz_re_box  : ");
        for(i=0;i<10;i++){
            printf("%f, ", x->z_re_box[i]);
        }
        printf("\n");

        printf("%s\t\tdNrec_box : ");
        for(i=0;i<10;i++){
            printf("%f, ", x->dNrec_box[i]);
        }
        printf("\n");
    }

    if(print_corners){
        printf("%s\tCorners: ",pid);

        printf("%s\t\txH_box     : ",pid);
        print_corners_real(x->xH_box, HII_DIM, NCF);

        printf("%s\t\tGamma12_box: ", pid);
        print_corners_real(x->Gamma12_box, HII_DIM, NCF);

        printf("%s\t\tz_re_box   : ", pid);
        print_corners_real(x->z_re_box, HII_DIM, NCF);

        printf("%s\t\tdNrec_box  : ", pid);
        print_corners_real(x->dNrec_box, HII_DIM, NCF);
    }
}

void inspectBrightnessTemp(struct BrightnessTemp *x, int print_pid, int print_corners, int print_first, int HII_DIM, float NCF){
    int i;

    char *pid = print_output_header(print_pid, "BrightnessTemp");

    if(print_first){
        printf("%s\tFirstRow: ",pid);

        printf("%s\t\tbrightness_temp: ");
        for(i=0;i<10;i++){
            printf("%f, ", x->brightness_temp[i]);
        }
        printf("\n");
    }

    if(print_corners){
        printf("%s\tCorners: ",pid);

        printf("%s\t\tbrightness_temp: ",pid);
        print_corners_real(x->brightness_temp, HII_DIM, NCF);
    }
}
double atomic_cooling_threshold(float z){
    return TtoM(z, 1e4, 0.59);
}

double molecular_cooling_threshold(float z){
    return TtoM(z, 600, 1.22);
}

double lyman_werner_threshold(float z, float J_21_LW, float vcb, struct AstroParams *astro_params){
    // correction follows Schauer+20, fit jointly to LW feedback and relative velocities. They find weaker effect of LW feedback than before (Stacy+11, Greif+11, etc.) due to HII self shielding.
    double mcrit_noLW = 3.314e7 * pow( 1.+z, -1.5);// this follows Visbal+15, which is taken as the optimal fit from Fialkov+12 which was calibrated with the simulations of Stacy+11 and Greif+11;

    double f_LW = 1.0 + astro_params->A_LW * pow(J_21_LW, astro_params->BETA_LW);

    double f_vcb = pow(1.0 + astro_params->A_VCB * vcb/SIGMAVCB, astro_params->BETA_VCB);

    // double mcrit_LW = mcrit_noLW * (1.0 + 10. * sqrt(J_21_LW)); //Eq. (12) in Schauer+20
    // return pow(10.0, log10(mcrit_LW) + 0.416 * vcb/SIGMAVCB ); //vcb and sigmacb in km/s, from Eq. (9)

    return (mcrit_noLW * f_LW * f_vcb);

}

double reionization_feedback(float z, float Gamma_halo_HII, float z_IN){
    if (z_IN<=1e-19)
        return 1e-40;
    return REION_SM13_M0 * pow(HALO_BIAS * Gamma_halo_HII, REION_SM13_A) * pow((1.+z)/10, REION_SM13_B) *
        pow(1 - pow((1.+z)/(1.+z_IN), REION_SM13_C), REION_SM13_D);
}


/*
    The following functions are simply for testing the exception framework
*/
void FunctionThatThrows(){
    Throw(PhotonConsError);
}

int SomethingThatCatches(bool sub_func){
    // A simple function that catches a thrown error.
    int status;
    Try{
        if(sub_func) FunctionThatThrows();
        else Throw(PhotonConsError);
    }
    Catch(status){
        return status;
    }
    return 0;
}

int FunctionThatCatches(bool sub_func, bool pass, double *result){
    int status;
    if(!pass){
        Try{
            if(sub_func) FunctionThatThrows();
            else Throw(PhotonConsError);
        }
        Catch(status){
            LOG_DEBUG("Caught the problem with status %d.", status);
            return status;
        }
    }
    *result = 5.0;
    return 0;
}

float clip(float x, float min, float max){
    if(x<min) return min;
    if(x>max) return max;
    return x;
}
