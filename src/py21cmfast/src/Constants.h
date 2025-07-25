/*
    This is a header file contains fundamental constants that should not need to
    be changed by the user.

    Do a text search to find parameters from a specific .H file from 21cmFAST
    (i.e. INIT_PARAMS.H, COSMOLOGY.H, ANAL_PARAMS.H and HEAT_PARAMS)

    NOTE: Not all 21cmFAST variables will be found below. Only those useful for 21CMMC

 */
#ifndef _CONSTANTS_H
#define _CONSTANTS_H

#include "InputParameters.h"

// ----------------------------------------------------------------------------------------- //

// Taken from ANAL_PARAMS.H

// ----------------------------------------------------------------------------------------- //

// factor relating cube length to filter radius = (4PI/3)^(-1/3)
#define L_FACTOR (float)(0.620350491)

/*
 Filenames of the appropriate output from RECFAST to be used as boundary conditions in Ts.c
 as well as other tables used to compute the spin temperature
 */
#define RECFAST_FILENAME (const char *)"recfast_LCDM.dat"
#define STELLAR_SPECTRA_FILENAME (const char *)"stellar_spectra.dat"
#define KAPPA_EH_FILENAME (const char *)"kappa_eH_table.dat"
#define KAPPA_PH_FILENAME (const char *)"kappa_pH_table.dat"
// Interpolation table for Lya heating efficiencies
#define LYA_HEATING_FILENAME (const char *)"Lyman_alpha_heating_table.dat"

// ----------------------------------------------------------------------------------------- //

// Taken from HEAT_PARAMS.H

// ----------------------------------------------------------------------------------------- //

/* Maximum allowed value for the kinetic temperature. Useful to set to avoid some spurious behaviour
 when the code is run with redshift poor resolution and very high X-ray heating efficiency */
#define MAX_TK (float)5e4

// ----------------------------------------------------------------------------------------- //

// Taken from COSMOLOGY.H

// ----------------------------------------------------------------------------------------- //

// STRUCTURE //
#define Deltac (1.68)  // at z=0, density excess at virialization
#define DELTAC_DELOS (1.5)
#define N_nu (1.0)      // # of heavy neutrinos (for EH trasfer function)
#define BODE_e (0.361)  // Epsilon parameter in Bode et al. 2000 trans. funct.
#define BODE_n (5.0)    // Eda parameter in Bode et al. 2000 trans. funct.
#define BODE_v (1.2)    // Nu parameter in Bode et al. 2000 trans. funct.

#define SIGMAVCB (29.0)  // rms value of the DM-b relative velocity [im km/s]

// Universal FOF HMF (Watson et al. 2013)
#define Watson_A (0.282)      // Watson FOF HMF, A parameter (Watson et al. 2013)
#define Watson_alpha (2.163)  // Watson FOF HMF, alpha parameter (Watson et al. 2013)
#define Watson_beta (1.406)   // Watson FOF HMF, beta parameter (Watson et al. 2013)
#define Watson_gamma (1.210)  // Watson FOF HMF, gamma parameter (Watson et al. 2013)

// Universal FOF HMF with redshift evolution (Watson et al. 2013)
#define Watson_A_z_1 (0.990)  // Watson FOF HMF, normalisation of A_z parameter (Watson et al. 2013)
#define Watson_A_z_2 (-3.216)  // Watson FOF HMF, power law of A_z parameter (Watson et al. 2013)
#define Watson_A_z_3 (0.074)   // Watson FOF HMF, offset of A_z parameter (Watson et al. 2013)
// Watson FOF HMF, normalisation of alpha_z parameter (Watson et al. 2013)
#define Watson_alpha_z_1 (5.907)
// Watson FOF HMF, power law of alpha_z parameter (Watson et al. 2013)
#define Watson_alpha_z_2 (-3.058)
#define Watson_alpha_z_3 (2.349)  // Watson FOF HMF, offset of beta_z parameter (Watson et al. 2013)
// Watson FOF HMF, normalisation of beta_z parameter (Watson et al. 2013)
#define Watson_beta_z_1 (3.136)
// Watson FOF HMF, power law of beta_z parameter (Watson et al. 2013)
#define Watson_beta_z_2 (-3.599)
#define Watson_beta_z_3 (2.344)  // Watson FOF HMF, offset of beta_z parameter (Watson et al. 2013)
#define Watson_gamma_z (1.318)   // Watson FOF HMF, gamma parameter (Watson et al. 2013)

// CONSTANTS //
#define LN10 (double)(2.30258509299)
#define SIGMAT (double)(6.6524e-25)  // Thomson scattering cross section in cm^-2
#define SIGMA_HI (double)(6.3e-18)   // HI ionization  cross section at 13.6 eV in cm^-2
#define E (double)(2.71828182846)
#define PI (double)(3.14159265358979323846264338327)
#define TWOPI (double)(2.0 * PI)
#define FOURPI (double)(4.0 * PI)
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
#define NV_ANG (double)1240.81                       // NV line center
#define CMperMPC (double)3.086e24                    // cm/Mpc
#define SperYR (double)31556925.9747                 // s/yr
#define Msun (double)1.989e33                        // g
#define Rsun (double)6.9598e10                       // cm
#define Lsun (double)3.90e33                         // erg/s
#define T_cmb (double)2.7255                         // K
#define k_B (double)1.380658e-16                     // erg / K
#define m_p (double)1.6726231e-24                    // proton mass (g)
#define m_e (double)9.10938188e-28                   // electron mass (g)
#define e_charge (double)4.80320467e-10              // elemetary charge (esu=g^1/2 cm^3/2 s^-1
#define SQDEG_ALLSKY (double)((360.0 * 360.0) / PI)  // Square degrees in all sky
#define G_AB_Jy (double)3631.0                       // AB mag constant in Jy
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
#define RECFAST_NPTS (int)501
#define KAPPA_10_NPTS (int)27
#define KAPPA_10_elec_NPTS (int)20
#define KAPPA_10_pH_NPTS (int)17

#define KAPPA_10_NPTS_Spline (int)30
#define KAPPA_10_elec_NPTS_Spline (int)30
#define KAPPA_10_pH_NPTS_Spline (int)30

/* Number of interpolation points for the interpolation table for z'' */
#define zpp_interp_points_SFR (int)(400)

/* Number of interpolation points for the interpolation table for the value of the density field */
#define dens_Ninterp (int)(400)

// Variables needed to read the Lyman-Alpha heating table
#define Tk_max (double)3.0   // log10 (Tk_max)
#define Tk_min (double)-1.0  // log10 (Tk_min)
#define Ts_max (double)3.0   // log10 (Ts_max)
#define Ts_min (double)-1.0  // log10 (Ts_min)
#define nT (int)101
#define taugp_max (double)7.0  // log10 (taugp_max)
#define taugp_min (double)1.0  // log10 (taugp_min)
#define ngp (int)51

// ----------------------------------------------------------------------------------------- //

// Taken from elec_interp.c

// ----------------------------------------------------------------------------------------- //

// ----------------------------------------------------------------------------------------- //

// From ps.c (temporary way to including transfer function from CLASS

// ----------------------------------------------------------------------------------------- //

#define CLASS_FILENAME (const char *)"Transfers_z0.dat"
#define CLASS_LENGTH 150  // length of the CLASS transfer function
// max and min k in  CLASS transfer function, temporary until interfaced properly
#define KBOT_CLASS (float)(1e-5)
#define KTOP_CLASS (float)(1e3)

// parameters for the M(sigma) power-law relation for FAST_FCOLL_TABLES
#define MPIVOT1 (double)(1.5e9)  // pivot masses
#define MPIVOT2 (double)(5.3e5)
#define AINDEX1 (double)(9.0)   // power-law index of nu(M) between MPIVOT1 and infinite
#define AINDEX2 (double)(13.6)  // power-law index of nu(M) between MPIVOT2 and MPIVOT1
#define AINDEX3 (double)(21.0)  // power-law index of nu(M) between 0 and MPIVOT2
// min mass at which the sigma table is computed if FAST_FCOLL_TABLES is turned
// on. Has to be below MPIVOT2
#define MMIN_FAST (double)(1e5)

// parameters for DM-baryon relative velocity effect on the power spectrum
#define KP_VCB_PM (300.0)  // Mpc-1
#define A_VCB_PM (0.24)
#define SIGMAK_VCB_PM (0.9)
// this is for vcb=vrms at z=20. It scales roughly as sqrt(v) and (1+z)^(-1/6.)

// ----------------------------------------------------------------------------------------- //

// For reionization_feedback, reference Sobacchi & Mesinger 2013

// ----------------------------------------------------------------------------------------- //

#define HALO_BIAS (double)2.0
#define REION_SM13_M0 (double)3e9
#define REION_SM13_A (double)0.17
#define REION_SM13_B (double)-2.1
#define REION_SM13_C (double)2.0
#define REION_SM13_D (double)2.5

#endif

// -------------------------------------------------------------------------------------
// Taken from COSMOLOGY.H
// -------------------------------------------------------------------------------------
#define Ho (double)(cosmo_params_global->hlittle * 3.2407e-18)  // s^-1 at z=0
// Msun Mpc^-3 ---- at z=0
#define RHOcrit (double)((3.0 * Ho * Ho / (8.0 * PI * G)) * (CMperMPC * CMperMPC * CMperMPC) / Msun)
#define RHOcrit_cgs (double)(3.0 * Ho * Ho / (8.0 * PI * G))  // g pcm^-3 ---- at z=0
//  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7
#define No (double)(RHOcrit_cgs * cosmo_params_global->OMb * (1 - cosmo_params_global->Y_He) / m_p)
//  current helium number density estimate
#define He_No \
    (double)(RHOcrit_cgs * cosmo_params_global->OMb * cosmo_params_global->Y_He / (4.0 * m_p))
#define N_b0 (double)(No + He_No)            // present-day baryon num density, H + He
#define f_H (double)(No / (No + He_No))      // hydrogen number fraction
#define f_He (double)(He_No / (No + He_No))  // helium number fraction
