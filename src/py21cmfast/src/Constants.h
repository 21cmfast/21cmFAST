/*
    This is a header file contains fundamental constants that should not need to
    be changed by the user.

    Do a text search to find parameters from a specific .H file from 21cmFAST
    (i.e. INIT_PARAMS.H, COSMOLOGY.H, ANAL_PARAMS.H and HEAT_PARAMS)

    NOTE: Not all 21cmFAST variables will be found below. Only those useful for 21CMMC

 */
#ifndef _CONSTANTS_21CM_H
#define _CONSTANTS_21CM_H

#include <math.h>

#include "InputParameters.h"

typedef struct PhysicalConstants {
    // Fundamental Constants
    const double c_cms;     // speed of light in cm/s
    const double c_kms;     // speed of light in km/s
    const double h_p;       // Planck's constant in erg s
    const double k_B;       // Boltzmann's constant in erg/K
    const double m_p;       // proton mass in g
    const double m_e;       // electron mass in g
    const double G;         // gravitational constant in cgs
    const double e_charge;  // electron charge in esu (g^1/2 cm^3/2 s^-1)
    const double vac_perm;  // permittivity of free space in F/m (C^2 s^2 / kg m^3)

    // Units
    const double Msun;        // solar mass in g
    const double s_per_yr;    // seconds per year
    const double cm_per_Mpc;  // cm per Mpc
    const double eV_to_Hz;    // convert eV to Hz

    // Photon frequencies and temperatures
    const double nu_ion_HI;        // ionization frequency of HI in Hz
    const double nu_ion_HeI;       // ionization frequency of HeI in Hz
    const double nu_ion_HeII;      // ionization frequency of HeII in Hz
    const double nu_LW_thresh;     // Lyman-Werner threshold frequency in Hz
    const double nu_Ly_alpha;      // frequency of Lyman-alpha in Hz
    const double T_cmb;            // CMB temperature at z=0 in K
    const double T_21;             // Temperature corresponding to 21cm photon in K
    const double lambda_21;        // Wavelength of 21cm Radiation in cm
    const double lambda_Ly_alpha;  // Wavelength of Lyman-Alpha in Angstroms
    const double lambda_Ly_beta;   // Wavelength of Lyman-Beta in Angstroms
    const double lambda_Ly_gamma;  // Wavelength of Lyman-Gamma in Angstroms

    // Cross-sections and rate coefficients
    const double sigma_T;      // Thomson scattering cross section in cm^2
    const double sigma_HI;     // HI ionization cross section at 13.6 eV in cm^2
    const double A10;          // spontaneous emission coefficient of 21cm in s^-1
    const double A_Ly_alpha;   // Spontaneous emission coefficient for Lyman-Alpha line in s^-1
    const double f_alpha;      // oscillator strength of Lya
    const double alpha_A_10k;  // case A hydrogen recombination coefficient at 10,000 K in cm^3 s^-1
    const double alpha_B_10k;  // case B hydrogen recombination coefficient at 10,000 K in cm^3 s^-1
    const double alpha_B_20k;  // case B hydrogen recombination coefficient at 20,000 K in cm^3 s^-1
} PhysicalConstants;

extern const PhysicalConstants physconst;

// Below are a few leftover macros, where they were used across multiple files or
// depended on input parameters.

// BEWARE: Since these macros are defined in a header, they *can* be applied to
// any code included by 21cmFAST, e.g. if file A includes Constants.h, then includes
// file B which includes fftw.h, the macros do find/replaces on fftw.h during compilation.

// We should work toward *only* having definitions in .c files to avoid this.

// factor relating cube length to filter radius = (4PI/3)^(-1/3)
#define L_FACTOR (float)(0.620350491)

// STRUCTURE //
#define Deltac (1.68)  // at z=0, density excess at virialization
#define DELTAC_DELOS (1.5)

// Small numbers for comparison and avoiding division by zero
#define TINY (double)1e-30
#define FRACT_FLOAT_ERR (double)1e-7  // fractional floating point error

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
#define RHOcrit                                                                     \
    (double)((3.0 * Ho * Ho / (8.0 * M_PI * physconst.G)) *                         \
             (physconst.cm_per_Mpc * physconst.cm_per_Mpc * physconst.cm_per_Mpc) / \
             physconst.Msun)
#define RHOcrit_cgs (double)(3.0 * Ho * Ho / (8.0 * M_PI * physconst.G))  // g pcm^-3 ---- at z=0
//  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7
#define No                                                                              \
    (double)(RHOcrit_cgs * cosmo_params_global->OMb * (1 - cosmo_params_global->Y_He) / \
             physconst.m_p)
//  current helium number density estimate
#define He_No                                                                     \
    (double)(RHOcrit_cgs * cosmo_params_global->OMb * cosmo_params_global->Y_He / \
             (4.0 * physconst.m_p))
#define N_b0 (double)(No + He_No)  // present-day baryon num density, H + He

#define H_FRAC                                  \
    (double)((1. - cosmo_params_global->Y_He) / \
             (1. - 3. * cosmo_params_global->Y_He / 4.))  // hydrogen number fraction
#define HE_FRAC                                 \
    (double)((cosmo_params_global->Y_He / 4.) / \
             (1. - 3. * cosmo_params_global->Y_He / 4.))  // helium number fraction

#endif
