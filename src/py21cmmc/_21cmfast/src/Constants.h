/*
    This is a header file contains fundamental constants that should not need to 
    be changed by the user.

    Do a text search to find parameters from a specific .H file from 21cmFAST 
    (i.e. INIT_PARAMS.H, COSMOLOGY.H, ANAL_PARAMS.H and HEAT_PARAMS)
 */

// ----------------------------------------------------------------------------------------- //

// Constants taken from INIT_PARAMS.H

// ----------------------------------------------------------------------------------------- //



#define MIDDLE (user_params.DIM/2)
#define D (unsigned long long)user_params.DIM // the unsigned long long dimension
#define MID ((unsigned long long)MIDDLE)
#define VOLUME (user_params.BOX_LEN*user_params.BOX_LEN*user_params.BOX_LEN) // in Mpc^3
#define DELTA_K (TWOPI/user_params.BOX_LEN)
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

// Constants taken from ANAL_PARAMS.H

// ----------------------------------------------------------------------------------------- //



/******** BEGIN USER CHANGABLE DEFINITIONS   **********/

#define ALPHA_UVB (float) (5)
// Power law index of the UVB during the EoR.  This is only used if INHOMO_RECO is on,
// in order to compute the local mean free path inside the cosmic HII regions

#define EVOLVE_DENSITY_LINEARLY (int) (0)
// EVOLVE_DENSITY_LINEARLY = 1, evolve the density field with linear theory.
// If choosing this option, make sure that your cell size is
// in the linear regime at the redshift of interest
 
// EVOLVE_DENSITY_LINEARLY = 0, evolve the density field with 1st order perturbation theory.
// If choosing this option, make sure that you resolve small
// enough scales, roughly we find BOX_LEN/DIM should be < 1Mpc

#define SMOOTH_EVOLVED_DENSITY_FIELD (int) (0)
#define R_smooth_density (float) (0.2)
// If set to 1, the ZA density field is additionally smoothed (asside from the implicit
// boxcar smoothing performed when re-binning the ICs from DIM to HII_DIM) with a Gaussian
// filter of width R_smooth_density*BOX_LEN/HII_DIM.  The implicit boxcar smoothing in
// perturb_field.c bins the density field on scale DIM/HII_DIM, similar to what Lagrangian
// codes do when constructing Eulerian grids. In other words, the density field, \delta,
// is quantized into (DIM/HII_DIM)^3 values. If your usage requires smooth density fields,
// it is recommended to set SMOOTH_EVOLVED_FIELD to 1.  This also decreases the shot noise
// present in all grid based codes, though it overcompensates by an effective loss in resolution.
// New in v1.1

#define SECOND_ORDER_LPT_CORRECTIONS (int) (1)
// Use second-order Lagrangian perturbation theory (2LPT).
// Set this to 1 if the density field or the halo positions are extrapolated to low redshifts.
// The current implementation is very naive and add a factor ~6 to the memory requirements.
// Reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D

#define HII_ROUND_ERR (float) (1e-3)
// Allows one to set a flag allowing find_HII_bubbles to skip constructing the
// ionization field if it is estimated that the mean neutral fraction, <xH>, is
// within HII_ROUND_ERR of 1. In other words, if <xH> > 1-HII_ROUND_ERR,
// then find_HII_bubbles just prints a homogeneous xHI field
// of  1's.
 // This is a new option in v1.1. Previous versions had a hardcoded value of 1e-15.

#define FIND_BUBBLE_ALGORITHM (int) (2)
// Choice of:
// 1 - Mesinger & Furlanetto 2007 method of overlaping spheres:
// paint an ionized sphere with radius R, centered on pixel
// where R is filter radius
// This method, while somewhat more accurate, is slower than (2) especially
// in mostly ionized unverses, so only use for lower resolution boxes (HII_DIM<~400)
// 2 - Center pixel only method (Zahn et al. 2007). this is faster.

#define R_BUBBLE_MIN (float) (L_FACTOR*1)
// Minimum radius of an HII region in cMpc.  One can set this to 0, but should be careful with
// shot noise if the find_HII_bubble algorithm is run on a fine, non-linear density grid.

#define N_POISSON (int) (-1)
// If not using the halo field to generate HII regions, we provide the option of
// including Poisson scatter in the number of sources obtained through the conditional
// collapse fraction (which only gives the *mean* collapse fraction on a particular
// scale.  If the predicted mean collapse fraction is < N_POISSON * M_MIN,
// then Poisson scatter is added to mimic discrete halos on the subgrid scale
// (see Zahn+ 2010).
 
// NOTE: If you are interested in snapshots of the same realization at several redshifts,
// it is recommended to turn off this feature, as halos can stocastically
// "pop in and out of" existance from one redshift to the next...

#define T_USE_VELOCITIES (int) (1
// Parameter choice of whether to use velocity corrections in 21-cm fields
// 1=use velocities in delta_T; 0=do not use velocities
// NOTE: The approximation used to include peculiar velocity effects works
// only in the linear regime, so be careful using this (see Mesinger+2010)

#define MAX_DVDR (float) (0.2)
// Maximum velocity gradient along the line of sight in units of the hubble parameter at z.
// This is only used in computing the 21cm fields.
// Note, setting this too high can add spurious 21cm power in the early stages, due to the
// 1-e^-tau ~ tau approximation (see my 21cm intro paper and mao+2011).  However, this is still
// a good approximation at the <~10% level.  Future builds will include the mao+ model for
// redshift space distortions.

#define VELOCITY_COMPONENT (int) (3)
// Component of the velocity to be used in 21-cm temperature maps
// 1 = x
// 2 = y
// 3 = z

#define DIMENSIONAL_T_POWER_SPEC (int) (1)
// 0 = plot 21cm temperature power spectrum in non-dimensional units
// 1 = plot 21cm...  in mK^2

#define DELTA_R_FACTOR (float) (1.1) 
// factor by which to scroll through filter radius for halos

#define DELTA_R_HII_FACTOR (float) (1.1) 
// factor by which to scroll through filter radius for bubbles

#define R_OVERLAP_FACTOR (float) (1.0)
// Factor of the halo's radius, R, so that the effective radius
// is R_eff = R_OVERLAP_FACTOR * R.  Halos whose centers are less than
// R_eff away from another halo are not allowed.
// R_OVERLAP_FACTOR = 1 is fully disjoint
// R_OVERLAP_FACTOR = 0 means that centers are allowed to luy on the edges of
// neighboring halos

#define DELTA_CRIT_MODE (int) (1)
// 0 = delta_crit is constant 1.68
// 1 = delta_crit is the sheth tormen ellipsoidal collapse
// correction to delta_crit (see ps.c)

#define HII_FILTER (int) (1)
// Filter for the Halo or density field used to generate ionization field
// 0 = use real space top hat filter
// 1 = use k-space top hat filter
// 2 = use gaussian filter

#define INITIAL_REDSHIFT (float) 300 // used to perturb field

/************  END USER CHANGABLE DEFINITIONS  **************/

#define L_FACTOR (float) (0.620350491) // factor relating cube length to filter radius = (4PI/3)^(-1/3)

#define HII_D (unsigned long long) (user_params.HII_DIM)
#define HII_MIDDLE (user_params.HII_DIM/2)
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

// Constants taken from HEAT_PARAMS.H

// ----------------------------------------------------------------------------------------- //



#define HEAT_FILTER (int) 0
// Filter used for smoothing the linear density field to obtain the collapsed fraction in Ts.c
// 0 = use real space top hat filter
// 1 = use sharp k-space filter
// 2 = use gaussian filter

#define CLUMPING_FACTOR (double) 2
// sub grid scale.  note that if you want to run-down from a very high redshift (>50),
// you should set this to one..

#define Z_HEAT_MAX (float) 35
// Maximum redshift used in the Tk and x_e evolution equations.
// Temperature and x_e are assumed to be homogenous at higher redshifts.

#define R_XLy_MAX (float) 500
// Maximum radius of influence for computing Xray and Lya pumping in cMpc. This
// should be larger than the mean free path of the relevant photons.  If increasing,
// you might want to adjust the z'' loop in Ts.c to skip integrating from early times,
// to increase speed (i.e. the code integrates as far back as ZPP_MAX, where R_XLy_MAX
// is the comoving separation between ZPP_MAX and Z_HEAT_MAX).

#define NUM_FILTER_STEPS_FOR_Ts (int) 40
// Number of spherical anulli used to compute df_coll/dz' in the simulation box.
// The spherical annulii are evenly spaced in log R, ranging from the cell size to the box size.
// Ts.c will create this many boxes of size HII_DIM, so be wary of memory usage if values are high.

#define ZPRIME_STEP_FACTOR (float) 1.02
// Redshift step-size used in the z' integral.  Logarithmic dz.

#define TK_at_Z_HEAT_MAX (double) -1
#define XION_at_Z_HEAT_MAX (double) -1
// If the following are >0, then the user chooses to overwrite the default boundary conditions
// at Z_HEAT_MAX obtained from RECFAST, and use his/her own.

#define Ts_verbose (int) 1
// Set this to 1, if you want the spin temperature field printed at eary step of the evolution.
// This would be useful if investigating redshift evolution, so Ts.c needs to be called only once
// at the low redshift, and then one has all of the higher redshift Ts boxes upon completion.

#define Pop (int) 2
// Stellar Population responsible for early heating
// Pop == 2 Pop2 stars heat the early universe
// Pop == 3 Pop3 stars heat the early universe

#define Pop2_ion (float) 4361
#define Pop3_ion (float) 44021
// Number of ionizing photons per baryon of the two stellar species

#define DEBUG_ON (int) 0
// Flag to turn off or on verbose status messages in Ts.  The GSL libraries are very finicky,
// and this is useful when to help issolate why Ts crashed, if it did...



// ----------------------------------------------------------------------------------------- //

// Constants taken from COSMOLOGY.H

// ----------------------------------------------------------------------------------------- //



//      New in v1.1.        //
//      WDM parameters      //
#define P_CUTOFF (int) 0 // supress the power spectrum? 0= CDM; 1=WDM
#define M_WDM (float) 2 // mass of WDM particle in keV.  this is ignored if P_CUTOFF is set to zero
#define g_x (float) 1.5 // degrees of freedom of WDM particles; 1.5 for fermions

#define OMn  (0.0)
#define OMk  (0.0)
#define OMr  (8.6e-5)
#define OMtot (1.0)
#define Y_He (0.245)
#define wl   (-1.0) // dark energy equation of state parameter (wl = -1 for vacuum )

// Note that the best fit b and c ST params for these 3D realisations have a redshift,
// and a DELTA_R_FACTOR (see ANAL_PARAMS.H) dependence, as (will be) shown in Mesinger+.
// For converged mass functions at z~5-10, set DELTA_R_FACTOR=1.1 and SHETH_b~0.15
// SHETH_c~0.05 (work in progress)
 
// For most purposes, a larger step size is quite sufficient and provides an excelent match
// to N-body and smoother mass functions, though the b and c parameters should be changed
// to make up for some "stepping-over" massive collapsed halos (see Mesinger, Perna, Haiman (2005)
// and Mesinger et al., in preparation)
 
// For example, at z~7-10, one can set DELTA_R_FACTOR=1.3 and SHETH_b=0.15 SHETH_c=0.25,
// to increase the speed of the halo finder.
#define SHETH_b (0.15) //  1-D realisation best fit from Barkana et al. 2001: SHETH_b 0.34
#define SHETH_c (0.05) // 1-D realisation best fit from Barkana et al. 2001: SHETH_c 0.81

#define Zreion_HeII (double) 3 // redshift of helium reionization, currently only used for tau_e

//              END USER CHANGABLE DEFINITIONS                  //
//          THINGS BELOW SHOULD NOT REQUIRE CHANGING            //
// **********************************************************  //
// STRUCTURE //
#define Deltac (1.68) // at z=0, density excess at virialization
#define FILTER (0) // smoothing: 0=tophat, 1=gaussian
#define POWER_SPECTRUM (0) // EH=0 BBKS=1  EFSTATHIOU=2  PEEBLES=3  WHITE=4
#define N_nu (1.0) // # of heavy neutrinos (for EH trasfer function)
#define BODE_e (0.361) // Epsilon parameter in Bode et al. 2000 trans. funct.
#define BODE_n (5.0) // Eda parameter in Bode et al. 2000 trans. funct.
#define BODE_v (1.2) // Nu parameter in Bode et al. 2000 trans. funct.
// SHETH params gotten from barkana et al. 2001, pg. 487
#define SHETH_a (0.73) // Sheth and Tormen a parameter (from Jenkins et al. 2001)
#define SHETH_p (0.175) // Sheth and Tormen p parameter (from Jenkins et al. 2001)
#define SHETH_A (0.353) // Sheth and Tormen A parameter (from Jenkins et al. 2001)
// **********************************************************  //

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
#define Ho  (double) (cosmo_params.hlittle*3.2407e-18) // s^-1 at z=0
#define RHOcrit (double) ( (3.0*Ho*Ho / (8.0*PI*G)) * (CMperMPC*CMperMPC*CMperMPC)/Msun) // Msun Mpc^-3 ---- at z=0
#define RHOcrit_cgs (double) (3.0*Ho*Ho / (8.0*PI*G)) // g pcm^-3 ---- at z=0
#define No  (double) (RHOcrit_cgs*OMb*(1-Y_He)/m_p)  //  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7
#define He_No (double) (RHOcrit_cgs*OMb*Y_He/(4.0*m_p)) //  current helium number density estimate
#define N_b0 (double) (No+He_No) // present-day baryon num density, H + He
#define f_H (double) (No/(No+He_No))  // hydrogen number fraction
#define f_He (double) (He_No/(No+He_No))  // helium number fraction
#define T21 (double) 0.0628 // temperature corresponding to the 21cm photon
#define A10_HYPERFINE (double) 2.85e-15 // spontaneous emission coefficient in s^-1
