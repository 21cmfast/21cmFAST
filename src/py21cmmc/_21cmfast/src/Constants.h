/*
    This is a header file contains fundamental constants that should not need to 
    be changed by the user.

    Do a text search to find parameters from a specific .H file from 21cmFAST 
    (i.e. INIT_PARAMS.H, COSMOLOGY.H, ANAL_PARAMS.H and HEAT_PARAMS)
 
    NOTE: Not all 21cmFAST variables will be found below. Only those useful for 21CMMC
 
 */

// ----------------------------------------------------------------------------------------- //

// Taken from INIT_PARAMS.H

// ----------------------------------------------------------------------------------------- //



#define MIDDLE (user_params->DIM/2)
#define D (unsigned long long)user_params->DIM // the unsigned long long dimension
#define MID ((unsigned long long)MIDDLE)
#define VOLUME (user_params->BOX_LEN*user_params->BOX_LEN*user_params->BOX_LEN) // in Mpc^3
#define DELTA_K (TWOPI/user_params->BOX_LEN)
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



#define R_BUBBLE_MIN (float) (L_FACTOR*1)
// Minimum radius of an HII region in cMpc.  One can set this to 0, but should be careful with
// shot noise if the find_HII_bubble algorithm is run on a fine, non-linear density grid.

#define L_FACTOR (float) (0.620350491) // factor relating cube length to filter radius = (4PI/3)^(-1/3)

#define HII_D (unsigned long long) (user_params->HII_DIM)
#define HII_MIDDLE (user_params->HII_DIM/2)
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

// STRUCTURE //
#define Deltac (1.68) // at z=0, density excess at virialization
#define N_nu (1.0) // # of heavy neutrinos (for EH trasfer function)
#define BODE_e (0.361) // Epsilon parameter in Bode et al. 2000 trans. funct.
#define BODE_n (5.0) // Eda parameter in Bode et al. 2000 trans. funct.
#define BODE_v (1.2) // Nu parameter in Bode et al. 2000 trans. funct.
// SHETH params gotten from barkana et al. 2001, pg. 487
#define SHETH_a (0.73) // Sheth and Tormen a parameter (from Jenkins et al. 2001)
#define SHETH_p (0.175) // Sheth and Tormen p parameter (from Jenkins et al. 2001)
#define SHETH_A (0.353) // Sheth and Tormen A parameter (from Jenkins et al. 2001)

// CONSTANTS //
#define LN10 (double) (2.30258509299)
#define SIGMAT (double) (6.6524e-25)  // Thomson scattering cross section in cm^-2
#define SIGMA_HI (double) (6.3e-18)  // HI ionization  cross section at 13.6 eV in cm^-2
#define E (double) (2.71828182846)
#define PI (double) (3.14159265358979323846264338327)
#define TWOPI (double) (2.0*PI)
#define FOURPI (double) (4.0*PI)
#define G (double) 6.67259e-8 // cm^3 g^-1 s^-2
#define hplank (double) 6.62606896e-27 /. erg s
#define TINY (double) 1e-30
#define FRACT_FLOAT_ERR (double) 1e-7 // fractional floating point error
#define f_alpha (float) 0.4162 / oscillator strength of Lya
#define Ly_alpha_HZ  (double ) 2.46606727e15  // frequency of Lyalpha
#define C  (double) 29979245800.0  //  speed of light  (cm/s)
#define alphaA_10k (double) 4.18e-13 // taken from osterbrock for T=10000
#define alphaB_10k (double) 2.59e-13 // taken from osterbrock for T=10000
#define alphaB_20k (double) 2.52e-13 // taken from osterbrock for T=20000
#define Ly_alpha_ANG (double) 1215.67
#define Ly_beta_ANG (double) 1025.18
#define Ly_gamma_ANG (double) 972.02
#define NV_ANG (double) 1240.81 // NV line center
#define CMperMPC (double) 3.086e24 // cm/Mpc
#define SperYR (double) 31556925.9747 // s/yr
#define Msun (double) 1.989e33 // g
#define Rsun (double) 6.9598e10 // cm
#define Lsun (double) 3.90e33 // erg/s
#define T_cmb (double) 2.728 // K
#define k_B (double) 1.380658e-16 // erg / K
#define m_p (double) 1.6726231e-24 // proton mass (g)
#define m_e (double) 9.10938188e-28 // electron mass (g)
#define e_charge (double) 4.8033e-10 // elemetary charge (esu=g^1/2 cm^3/2 s^-1
#define SQDEG_ALLSKY (double) ((360.0*360.0)/PI) // Square degrees in all sky
#define G_AB_Jy (double) 3631.0 // AB mag constant in Jy
#define NU_over_EV (double) (1.60217646e-12 / hplank)
#define NUIONIZATION (double) (13.60*NU_over_EV)  // ionization frequency of H
#define HeII_NUIONIZATION (double) (NUIONIZATION*4) // ionization frequency of HeII
#define HeI_NUIONIZATION (double) (24.59*NU_over_EV) // ionization frequency of HeI
#define Ho  (double) (cosmo_params->hlittle*3.2407e-18) // s^-1 at z=0
#define RHOcrit (double) ( (3.0*Ho*Ho / (8.0*PI*G)) * (CMperMPC*CMperMPC*CMperMPC)/Msun) // Msun Mpc^-3 ---- at z=0
#define RHOcrit_cgs (double) (3.0*Ho*Ho / (8.0*PI*G)) // g pcm^-3 ---- at z=0
#define No  (double) (RHOcrit_cgs*OMb*(1-Y_He)/m_p)  //  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7
#define He_No (double) (RHOcrit_cgs*OMb*Y_He/(4.0*m_p)) //  current helium number density estimate
#define N_b0 (double) (No+He_No) // present-day baryon num density, H + He
#define f_H (double) (No/(No+He_No))  // hydrogen number fraction
#define f_He (double) (He_No/(No+He_No))  // helium number fraction
#define T21 (double) 0.0628 // temperature corresponding to the 21cm photon
#define A10_HYPERFINE (double) 2.85e-15 // spontaneous emission coefficient in s^-1
