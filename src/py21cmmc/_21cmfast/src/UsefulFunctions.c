// ----------------------------------------------------------------------------------------- //

// Taken from INIT_PARAMS.H

// ----------------------------------------------------------------------------------------- //



#define MIDDLE (user_params_ufunc->DIM/2)
#define D (unsigned long long)user_params_ufunc->DIM // the unsigned long long dimension
#define MID ((unsigned long long)MIDDLE)
#define VOLUME (user_params_ufunc->BOX_LEN*user_params_ufunc->BOX_LEN*user_params_ufunc->BOX_LEN) // in Mpc^3
#define DELTA_K (TWOPI/user_params_ufunc->BOX_LEN)
#define TOT_NUM_PIXELS ((unsigned long long)(D*D*D)) // no padding
#define TOT_FFT_NUM_PIXELS ((unsigned long long)(D*D*2llu*(MID+1llu)))
#define KSPACE_NUM_PIXELS ((unsigned long long)(D*D*(MID+1llu)))

// Define some useful macros

// for 3D complex array
#define C_INDEX(x,y,z)((unsigned long long)((z)+(MID+1llu)*((y)+D*(x))))

// for 3D real array with the FFT padding
#define R_FFT_INDEX(x,y,z)((unsigned long long)((z)+2llu*(MID+1llu)*((y)+D*(x))))

// for 3D real array with no padding
#define R_INDEX(x,y,z)((unsigned long long)((z)+D*((y)+D*(x))))


// ----------------------------------------------------------------------------------------- //

// Taken from ANAL_PARAMS.H

// ----------------------------------------------------------------------------------------- //



#define HII_D (unsigned long long) (user_params_ufunc->HII_DIM)
#define HII_MIDDLE (user_params_ufunc->HII_DIM/2)
#define HII_MID ((unsigned long long)HII_MIDDLE)

#define HII_TOT_NUM_PIXELS (unsigned long long)(HII_D*HII_D*HII_D)
#define HII_TOT_FFT_NUM_PIXELS ((unsigned long long)(HII_D*HII_D*2llu*(HII_MID+1llu)))
#define HII_KSPACE_NUM_PIXELS ((unsigned long long)(HII_D*HII_D*(HII_MID+1llu)))

// INDEXING MACROS //
// for 3D complex array
#define HII_C_INDEX(x,y,z)((unsigned long long)((z)+(HII_MID+1llu)*((y)+HII_D*(x))))
// for 3D real array with the FFT padding
#define HII_R_FFT_INDEX(x,y,z)((unsigned long long)((z)+2llu*(HII_MID+1llu)*((y)+HII_D*(x))))
// for 3D real array with no padding
#define HII_R_INDEX(x,y,z)((unsigned long long)((z)+HII_D*((y)+HII_D*(x))))



// ----------------------------------------------------------------------------------------- //

// Taken from COSMOLOGY.H

// ----------------------------------------------------------------------------------------- //



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



void filter_box(fftwf_complex *box, int RES, int filter_type, float R){
    int n_x, n_z, n_y, dimension,midpoint;
    float k_x, k_y, k_z, k_mag, kR;
    
//    printf("before = %e\n",box[C_INDEX(50, 50, 50)]);
    
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
    for (n_x=dimension; n_x--;){
        if (n_x>midpoint) {k_x =(n_x-dimension) * DELTA_K;}
        else {k_x = n_x * DELTA_K;}
        
        for (n_y=dimension; n_y--;){
            if (n_y>midpoint) {k_y =(n_y-dimension) * DELTA_K;}
            else {k_y = n_y * DELTA_K;}
            
            for (n_z=(midpoint+1); n_z--;){
                k_z = n_z * DELTA_K;
                
                k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                
                kR = k_mag*R; // real space top-hat
                
//                if(n_x == 50 && n_y == 50 && n_z == 50) {
//                    printf("n_x = %d n_y = %d n_z = %d k_x = %e k_y = %e k_z = %e k_mag = %e kR = %e\n",n_x,n_y,n_z,k_x,k_y,k_z,k_mag,kR);
//                }
                
                if (filter_type == 0){ // real space top-hat
                    if (kR > 1e-4){
//                        box[HII_C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR);
//                        printf("n_x = %d n_y = %d n_z = %d HIRES_box = %e\n",n_x,n_y,n_z,box[C_INDEX(n_x, n_y, n_z)]);
                        if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR); }
                        if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR); }
//                        printf("k_x = %e k_y = %e k_z = %e k_mag = %e kR = %e arg = %e HIRES_box = %e\n",k_x,k_y,k_z,k_mag,kR,3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR),box[C_INDEX(n_x, n_y, n_z)]);
                    }
                }
                else if (filter_type == 1){ // k-space top hat
                    kR *= 0.413566994; // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
                    if (kR > 1){
                        if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] = 0; }
                        if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] = 0; }
                    }
                }
                else if (filter_type == 2){ // gaussian
                    kR *= 0.643; // equates integrated volume to the real space top-hat
                    if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR*kR/2.0); }
                    if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR*kR/2.0); }
                }
                else{
                    if ( (n_x==0) && (n_y==0) && (n_z==0) )
                        fprintf(stderr, "HII_filter.c: Warning, filter type %i is undefined\nBox is unfiltered\n", filter_type);
                }
            }
        }
    } // end looping through k box
    
//    printf("after = %e\n",box[C_INDEX(50, 50, 50)]);
    
    return;
}

double MtoR(double M);
double RtoM(double R);
float TtoM(float z, float T, float mu);
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
        fprintf(stderr, "No such filter = %i.\nResults are bogus.\n", global_params.FILTER);
    return -1;
}

/* R in Mpc, M in Msun */
double RtoM(double R){
    // set M according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (global_params.FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return (4.0/3.0)*PI*pow(R,3)*(cosmo_params_ufunc->OMm*RHOcrit);
    else if (global_params.FILTER == 1) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow(2*PI, 1.5) * cosmo_params_ufunc->OMm*RHOcrit * pow(R, 3);
    else // filter not defined
        fprintf(stderr, "No such filter = %i.\nResults are bogus.\n", global_params.FILTER);
    return -1;
}

/*
 T in K, M in Msun, mu is mean molecular weight
 from Barkana & Loeb 2001
 
 SUPRESS = 0 for no radiation field supression;
 SUPRESS = 1 for supression (step function at z=z_ss, at v=v_zz)
 */
float TtoM(float z, float T, float mu){
    return 7030.97 / (cosmo_params_ufunc->hlittle) * sqrt( omega_mz(z) / (cosmo_params_ufunc->OMm*Deltac_nonlinear(z)) ) *
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
        fprintf(stderr, "IN WANG\n");
        return -1;
    }
    
    fprintf(stderr, "No growth function!!! Output will be fucked up.");
    return -1;
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
    
    fprintf(stderr, "No growth function!!! Output will be fucked up.");
    return -1;
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



void writeUserParams(struct UserParams *p, int print_pid){
    if(print_pid){
        printf("UserParams (pid=%d):\n", getpid());
    }else{
        printf("UserParams:\n", getpid());
    }

    printf("\tHII_DIM        : %d\n",p->HII_DIM);
    printf("\tDIM            : %d\n",p->DIM);
    printf("\tBOX_LEN        : %f\n",p->BOX_LEN);
    printf("\tUSE_FFTW_WISDOM: %d\n",p->USE_FFTW_WISDOM);
}

void writeCosmoParams(struct CosmoParams *p, int print_pid){
    if(print_pid){
        printf("CosmoParams (pid=%d):\n", getpid());
    }else{
        printf("CosmoParams\n", getpid());
    }

    printf("\tRANDOM_SEED: %d\n",p->RANDOM_SEED);
    printf("\tSIGMA_8    : %f\n",p->SIGMA_8);
    printf("\thlittle    : %f\n",p->hlittle);
    printf("\tOMm        : %f\n",p->OMm);
    printf("\tOMl        : %f\n",p->OMl);
    printf("\tOMb        : %f\n",p->OMb);
    printf("\tPOWER_INDEX: %f\n",p->POWER_INDEX);
}

void writeAstroParams(struct FlagOptions *fo, struct AstroParams *p, int print_pid){
    if(print_pid){
        printf("AstroParams (pid=%d):\n", getpid());
    }else{
        printf("AstroParams:\n", getpid());
    }
    
    if(fo->USE_MASS_DEPENDENT_ZETA) {
        printf("\tHII_EFF_FACTOR     : %f\n",p->HII_EFF_FACTOR);
        printf("\tALPHA_STAR         : %f\n",p->ALPHA_STAR);
        printf("\tF_ESC10            : %f\n",p->F_ESC10);
        printf("\tALPHA_ESC          : %f\n",p->ALPHA_ESC);
        printf("\tM_TURN             : %f\n",p->M_TURN);
        
    }
    else {
        printf("\tHII_EFF_FACTOR     : %f\n",p->HII_EFF_FACTOR);
        printf("\tION_Tvir_MIN       : %f\n",p->ION_Tvir_MIN);
        printf("\tX_RAY_Tvir_MIN     : %f\n",p->X_RAY_Tvir_MIN);
    }
    printf("\tR_BUBBLE_MAX       : %f\n",p->R_BUBBLE_MAX);
    printf("\tL_X                : %f\n",p->L_X);
    printf("\tNU_X_THRESH        : %f\n",p->NU_X_THRESH);
    printf("\tX_RAY_SPEC_INDEX   : %f\n",p->X_RAY_SPEC_INDEX);
    printf("\tF_STAR10           : %f\n",p->F_STAR10);
    printf("\tt_STAR             : %f\n",p->t_STAR);
    printf("\tN_RSD_STEPS        : %d\n",p->N_RSD_STEPS);
    
}

void writeFlagOptions(struct FlagOptions *p, int print_pid){
    if(print_pid){
        printf("FlagOptions (pid=%d):\n", getpid());
    }else{
        printf("FlagOptions:\n", getpid());
    }

    printf("\tUSE_MASS_DEPENDENT_ZETA: %d\n",p->USE_MASS_DEPENDENT_ZETA);
    printf("\tSUBCELL_RSD            : %d\n",p->SUBCELL_RSD);
    printf("\tINHOMO_RECO            : %d\n",p->INHOMO_RECO);
    printf("\tUSE_TS_FLUCT           : %d\n",p->USE_TS_FLUCT);
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

void inspectInitialConditions(struct InitialConditions *x, int print_pid, int print_corners, int print_first,
                              int HII_DIM){
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
        print_corners_real(x->lowres_density, HII_DIM);

        printf("%s\t\tlowres_vx     : ", pid);
        print_corners_real(x->lowres_vx, HII_DIM);

        printf("%s\t\tlowres_vx_2LPT: ", pid);
        print_corners_real(x->lowres_vx_2LPT, HII_DIM);
    }
}


void inspectPerturbedField(struct PerturbedField *x, int print_pid, int print_corners, int print_first,
                           int HII_DIM){
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
            printf("%f, ", x->velocity[i]);
        }
        printf("\n");

    }

    if(print_corners){
        printf("%s\tCorners: \n",pid);

        printf("%s\t\tdensity: ",pid);
        print_corners_real(x->density, HII_DIM);

        printf("%s\t\tvelocity: ", pid);
        print_corners_real(x->velocity, HII_DIM);
    }

}


void inspectTsBox(struct TsBox *x, int print_pid, int print_corners, int print_first, int HII_DIM){
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
        print_corners_real(x->Ts_box, HII_DIM);

        printf("%s\t\tx_e_box: ", pid);
        print_corners_real(x->x_e_box, HII_DIM);

        printf("%s\t\tTk_box : ", pid);
        print_corners_real(x->Tk_box, HII_DIM);
    }
}

void inspectIonizedBox(struct IonizedBox *x, int print_pid, int print_corners, int print_first, int HII_DIM){
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
        print_corners_real(x->xH_box, HII_DIM);

        printf("%s\t\tGamma12_box: ", pid);
        print_corners_real(x->Gamma12_box, HII_DIM);

        printf("%s\t\tz_re_box   : ", pid);
        print_corners_real(x->z_re_box, HII_DIM);

        printf("%s\t\tdNrec_box  : ", pid);
        print_corners_real(x->dNrec_box, HII_DIM);
    }
}

void inspectBrightnessTemp(struct BrightnessTemp *x, int print_pid, int print_corners, int print_first, int HII_DIM){
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
        print_corners_real(x->brightness_temp, HII_DIM);
    }
}


void print_corners_real(float *x, int size){
    int s = size-1;
    int i,j,k;
    for(i=0;i<size;i=i+s){
        for(j=0;j<size;j=j+s){
            for(k=0;k<size;k=k+s){
                printf("%f, ", x[k + size*(j + size*i)]);
            }
        }
    }
    printf("\n");
}