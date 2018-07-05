/*
    This is a header file contains fundamental constants that should not need to 
    be changed by the user.

 */



/****** New in v1.1. ******/
/*** WDM parameters ***/
#define P_CUTOFF (int) 0 // supress the power spectrum? 0= CDM; 1=WDM
#define M_WDM (float) 2 // mass of WDM particle in keV.  this is ignored if P_CUTOFF is set to zero
#define g_x (float) 1.5 /* degrees of freedom of WDM particles; 1.5 for fermions */

#define OMn  (0.0)
#define OMk  (0.0)
#define OMr  (8.6e-5)
#define OMtot (1.0)
#define Y_He (0.245)
#define wl   (-1.0) /* dark energy equation of state parameter (wl = -1 for vacuum ) */

/*
 Note that the best fit b and c ST params for these 3D realisations have a redshift,
 and a DELTA_R_FACTOR (see ANAL_PARAMS.H) dependence, as (will be) shown in Mesinger+.
 For converged mass functions at z~5-10, set DELTA_R_FACTOR=1.1 and SHETH_b~0.15 SHETH_c~0.05 (work in progress)
 
 For most purposes, a larger step size is quite sufficient and provides an excelent match
 to N-body and smoother mass functions, though the b and c parameters should be changed
 to make up for some "stepping-over" massive collapsed halos (see Mesinger, Perna, Haiman (2005)
 and Mesinger et al., in preparation)
 
 For example, at z~7-10, one can set DELTA_R_FACTOR=1.3 and SHETH_b=0.15 SHETH_c=0.25, to increase the speed of the halo finder.
 */
#define SHETH_b (0.15) /*  1-D realisation best fit from Barkana et al. 2001: SHETH_b 0.34 */
#define SHETH_c (0.05) /* 1-D realisation best fit from Barkana et al. 2001: SHETH_c 0.81 */

#define Zreion_HeII (double) 3 /* redshift of helium reionization, currently only used for tau_e */

/*******  END USER CHANGABLE DEFINITIONS ****************/
/******* THINGS BELOW SHOULD NOT REQUIRE CHANGING *******/
/********************************************************/
/* STRUCTURE */
#define Deltac (1.68) /* at z=0, density excess at virialization */
#define FILTER (0) /* smoothing: 0=tophat, 1=gaussian */
#define POWER_SPECTRUM (0) /* EH=0 BBKS=1  EFSTATHIOU=2  PEEBLES=3  WHITE=4 */
#define N_nu (1.0) /* # of heavy neutrinos (for EH trasfer function) */
#define BODE_e (0.361) /* Epsilon parameter in Bode et al. 2000 trans. funct.*/
#define BODE_n (5.0) /* Eda parameter in Bode et al. 2000 trans. funct.*/
#define BODE_v (1.2) /* Nu parameter in Bode et al. 2000 trans. funct.*/
/* SHETH params gotten from barkana et al. 2001, pg. 487 */
#define SHETH_a (0.73) /* Sheth and Tormen a parameter (from Jenkins et al. 2001) */
#define SHETH_p (0.175) /* Sheth and Tormen p parameter (from Jenkins et al. 2001) */
#define SHETH_A (0.353) /* Sheth and Tormen A parameter (from Jenkins et al. 2001) */
/********************************************************/

/* CONSTANTS */
#define LN10 (double) (2.30258509299)
#define SIGMAT (double) (6.6524e-25)  /* Thomson scattering cross section in cm^-2 */
#define SIGMA_HI (double) (6.3e-18)  /* HI ionization  cross section at 13.6 eV in cm^-2 */
#define E (double) (2.71828182846)
#define PI (double) (3.14159265358979323846264338327)
#define TWOPI (double) (2.0*PI)
#define FOURPI (double) (4.0*PI)
#define G (double) 6.67259e-8 /* cm^3 g^-1 s^-2*/
#define hplank (double) 6.62606896e-27 /* erg s */
#define TINY (double) 1e-30
#define FRACT_FLOAT_ERR (double) 1e-7 /* fractional floating point error */
#define f_alpha (float) 0.4162 /* oscillator strength of Lya */
#define Ly_alpha_HZ  (double ) 2.46606727e15  /* frequency of Lyalpha */
#define C  (double) 29979245800.0  /*  speed of light  (cm/s)  */
#define alphaA_10k (double) 4.18e-13 /* taken from osterbrock for T=10000 */
#define alphaB_10k (double) 2.59e-13 /* taken from osterbrock for T=10000 */
#define alphaB_20k (double) 2.52e-13 /* taken from osterbrock for T=20000 */
#define Ly_alpha_ANG (double) 1215.67
#define Ly_beta_ANG (double) 1025.18
#define Ly_gamma_ANG (double) 972.02
#define NV_ANG (double) 1240.81 /* NV line center */
#define CMperMPC (double) 3.086e24 /* cm/Mpc */
#define SperYR (double) 31556925.9747 /* s/yr */
#define Msun (double) 1.989e33 /* g */
#define Rsun (double) 6.9598e10 /* cm */
#define Lsun (double) 3.90e33 /* erg/s */
#define T_cmb (double) 2.728 /* K */
#define k_B (double) 1.380658e-16 /* erg / K */
#define m_p (double) 1.6726231e-24 /* proton mass (g) */
#define m_e (double) 9.10938188e-28 /* electron mass (g) */
#define e_charge (double) 4.8033e-10 /* elemetary charge (esu=g^1/2 cm^3/2 s^-1*/
#define SQDEG_ALLSKY (double) ((360.0*360.0)/PI) /* Square degrees in all sky */
#define G_AB_Jy (double) 3631.0 /* AB mag constant in Jy */
#define NU_over_EV (double) (1.60217646e-12 / hplank)
#define NUIONIZATION (double) (13.60*NU_over_EV)  /* ionization frequency of H */
#define HeII_NUIONIZATION (double) (NUIONIZATION*4) /* ionization frequency of HeII */
#define HeI_NUIONIZATION (double) (24.59*NU_over_EV) /* ionization frequency of HeI */
#define Ho  (double) (cosmo_params.hlittle*3.2407e-18) /* s^-1 at z=0 */
#define RHOcrit (double) ( (3.0*Ho*Ho / (8.0*PI*G)) * (CMperMPC*CMperMPC*CMperMPC)/Msun) /* Msun Mpc^-3 */ /* at z=0 */
#define RHOcrit_cgs (double) (3.0*Ho*Ho / (8.0*PI*G)) /* g pcm^-3 */ /* at z=0 */
#define No  (double) (RHOcrit_cgs*OMb*(1-Y_He)/m_p)  /*  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7*/
#define He_No (double) (RHOcrit_cgs*OMb*Y_He/(4.0*m_p)) /*  current helium number density estimate */
#define N_b0 (double) (No+He_No) /* present-day baryon num density, H + He */
#define f_H (double) (No/(No+He_No))  /* hydrogen number fraction */
#define f_He (double) (He_No/(No+He_No))  /* helium number fraction */
#define T21 (double) 0.0628 /* temperature corresponding to the 21cm photon */
#define A10_HYPERFINE (double) 2.85e-15 /* spontaneous emission coefficient in s^-1 */
