/*
    This is a header file contains fundamental constants that should not need to
    be changed by the user.

    Do a text search to find parameters from a specific .H file from 21cmFAST
    (i.e. INIT_PARAMS.H, COSMOLOGY.H, ANAL_PARAMS.H and HEAT_PARAMS)

    NOTE: Not all 21cmFAST variables will be found below. Only those useful for 21CMMC

 */
#ifndef _CONSTANTS_21CM_H
#define _CONSTANTS_21CM_H

#include "InputParameters.h"

// ----------------------------------------------------------------------------------------- //

// Taken from ANAL_PARAMS.H

// ----------------------------------------------------------------------------------------- //

// factor relating cube length to filter radius = (4PI/3)^(-1/3)
#define L_FACTOR (float)(0.620350491)

// STRUCTURE //
#define Deltac (1.68)  // at z=0, density excess at virialization
#define DELTAC_DELOS (1.5)

// CONSTANTS //
#define SIGMAT (double)(6.6524e-25)    // Thomson scattering cross section in cm^-2
#define SIGMA_HI (double)(6.3e-18)     // HI ionization  cross section at 13.6 eV in cm^-2
#define G (double)6.67259e-8           // cm^3 g^-1 s^-2
#define hplank (double)6.62606896e-27  // erg s
#define TINY (double)1e-30
#define FRACT_FLOAT_ERR (double)1e-7       // fractional floating point error
#define f_alpha (float)0.4162              // oscillator strength of Lya
#define Ly_alpha_HZ (double)2.46606727e15  // frequency of Lyalpha
#define C (double)29979245800.0            //  speed of light  (cm/s)
#define C_KMS (double)C / 1e5              /* speed of light in km/s  */
#define alphaA_10k (double)4.18e-13        // taken from osterbrock for T=10000
#define alphaB_10k (double)2.59e-13        // taken from osterbrock for T=10000
#define alphaB_20k (double)2.52e-13        // taken from osterbrock for T=20000
#define Ly_alpha_ANG (double)1215.67
#define Ly_beta_ANG (double)1025.18
#define Ly_gamma_ANG (double)972.02
#define NV_ANG (double)1240.81                         // NV line center
#define CMperMPC (double)3.086e24                      // cm/Mpc
#define SperYR (double)31556925.9747                   // s/yr
#define Msun (double)1.989e33                          // g
#define Rsun (double)6.9598e10                         // cm
#define Lsun (double)3.90e33                           // erg/s
#define T_cmb (double)2.7255                           // K
#define k_B (double)1.380658e-16                       // erg / K
#define m_p (double)1.6726231e-24                      // proton mass (g)
#define m_e (double)9.10938188e-28                     // electron mass (g)
#define e_charge (double)4.80320467e-10                // elemetary charge (esu=g^1/2 cm^3/2 s^-1
#define SQDEG_ALLSKY (double)((360.0 * 360.0) / M_PI)  // Square degrees in all sky
#define G_AB_Jy (double)3631.0                         // AB mag constant in Jy
#define NU_over_EV (double)(1.60217646e-12 / hplank)
#define NU_LW_THRESH (double)(11.18 * NU_over_EV)
#define NUIONIZATION (double)(13.60 * NU_over_EV)      // ionization frequency of H
#define HeII_NUIONIZATION (double)(NUIONIZATION * 4)   // ionization frequency of HeII
#define HeI_NUIONIZATION (double)(24.59 * NU_over_EV)  // ionization frequency of HeI
#define T21 (double)0.0682              // temperature corresponding to the 21cm photon
#define A10_HYPERFINE (double)2.85e-15  // spontaneous emission coefficient in s^-1

#define Lambda_21 (double)21.106114054160  // Wavelength of 21cm Radiation in cm
#define A21_Lya (double)6.24e8  // Spontaneous emission coefficient for Lyman-Alpha line in s^-1

#define vac_perm (double)8.8541878128e-12  // vacuum permittivity in farads m^-1

// ----------------------------------------------------------------------------------------- //

// Taken from heating_helper_progs.c

// ----------------------------------------------------------------------------------------- //

#define NSPEC_MAX (int)23

/* Number of interpolation points for the interpolation table for z'' */
#define zpp_interp_points_SFR (int)(400)

// min mass at which the sigma table is computed if FAST_FCOLL_TABLES is turned
// on. Has to be below MPIVOT2
#define MMIN_FAST (double)(1e5)

// -------------------------------------------------------------------------------------
// Taken from COSMOLOGY.H
// -------------------------------------------------------------------------------------
#define Ho (double)(cosmo_params_global->hlittle * 3.2407e-18)  // s^-1 at z=0
// Msun Mpc^-3 ---- at z=0
#define RHOcrit \
    (double)((3.0 * Ho * Ho / (8.0 * M_PI * G)) * (CMperMPC * CMperMPC * CMperMPC) / Msun)
#define RHOcrit_cgs (double)(3.0 * Ho * Ho / (8.0 * M_PI * G))  // g pcm^-3 ---- at z=0
//  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7
#define No (double)(RHOcrit_cgs * cosmo_params_global->OMb * (1 - cosmo_params_global->Y_He) / m_p)
//  current helium number density estimate
#define He_No \
    (double)(RHOcrit_cgs * cosmo_params_global->OMb * cosmo_params_global->Y_He / (4.0 * m_p))
#define N_b0 (double)(No + He_No)            // present-day baryon num density, H + He
#define f_H (double)(No / (No + He_No))      // hydrogen number fraction
#define f_He (double)(He_No / (No + He_No))  // helium number fraction

#endif
