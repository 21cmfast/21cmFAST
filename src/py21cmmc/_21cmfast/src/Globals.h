/*
    This is a header file containing some global variables that the user might want to change
    on the rare occasion.

    Do a text search to find parameters from a specific .H file from 21cmFAST 
    (i.e. INIT_PARAMS.H, COSMOLOGY.H, ANAL_PARAMS.H and HEAT_PARAMS)
 
    NOTE: Not all 21cmFAST variables will be found below. Only those useful for 21CMMC
 
 */

struct GlobalParams{
    float ALPHA_UVB;
    int EVOLVE_DENSITY_LINEARLY;
    int SMOOTH_EVOLVED_DENSITY_FIELD;
    float R_smooth_density;
    int SECOND_ORDER_LPT_CORRECTIONS;
    float HII_ROUND_ERR;
    int FIND_BUBBLE_ALGORITHM;
    int N_POISSON;
    int T_USE_VELOCITIES;
    float MAX_DVDR;
//    int DIMENSIONAL_T_POWER_SPEC;
//    float DELTA_R_FACTOR;
    float DELTA_R_HII_FACTOR;
//    float R_OVERLAP_FACTOR;
//    int DELTA_CRIT_MODE;
    int HII_FILTER;
    float INITIAL_REDSHIFT;
    
    float CRIT_DENS_TRANSITION;
    float MIN_DENSITY_LOW_LIMIT;
    
    int HEAT_FILTER;
    double CLUMPING_FACTOR;
    float Z_HEAT_MAX;
    float R_XLy_MAX;
    int NUM_FILTER_STEPS_FOR_Ts;
    float ZPRIME_STEP_FACTOR;
    double TK_at_Z_HEAT_MAX;
    double XION_at_Z_HEAT_MAX;
//    int Ts_verbose;
    int Pop;
    float Pop2_ion;
    float Pop3_ion;
//    int DEBUG_ON;

    float NU_X_BAND_MAX;
    float NU_X_MAX;
    
    int NBINS_LF;
    
    int P_CUTOFF;
    float M_WDM;
    float g_x;
    float OMn;
    float OMk;
    float OMr;
    float OMtot;
    float Y_He;
    float wl;
    float SHETH_b;
    float SHETH_c;
    double Zreion_HeII;
    int FILTER;
    int POWER_SPECTRUM;

    char *external_table_path;
};

struct GlobalParams global_params = {
    
    .ALPHA_UVB = 5.0,
    .EVOLVE_DENSITY_LINEARLY = 0,
    .SMOOTH_EVOLVED_DENSITY_FIELD = 0,
    .R_smooth_density = 0.2,
    .SECOND_ORDER_LPT_CORRECTIONS = 1,
    .HII_ROUND_ERR = 1e-3,
    .FIND_BUBBLE_ALGORITHM = 2,
    .N_POISSON = 5,
    .T_USE_VELOCITIES = 1,
    .MAX_DVDR = 0.2,
//    .DIMENSIONAL_T_POWER_SPEC = 1,
//    .DELTA_R_FACTOR = 1.1,
    .DELTA_R_HII_FACTOR = 1.1,
//    .R_OVERLAP_FACTOR = 1.0,
//    .DELTA_CRIT_MODE = 1,
    .HII_FILTER = 1,
    .INITIAL_REDSHIFT = 300.,
    
    .CRIT_DENS_TRANSITION = 1.5,
    .MIN_DENSITY_LOW_LIMIT = 9e-8,
    
    .HEAT_FILTER = 0,
    .CLUMPING_FACTOR = 2.,
    .Z_HEAT_MAX = 35.0,
    .R_XLy_MAX = 500.,
    .NUM_FILTER_STEPS_FOR_Ts = 40,
    .ZPRIME_STEP_FACTOR = 1.02,
    .TK_at_Z_HEAT_MAX = -1,
    .XION_at_Z_HEAT_MAX = -1,
//    .Ts_verbose = 1,
    .Pop = 2,
    .Pop2_ion = 5000,
    .Pop3_ion = 44021,
//    .DEBUG_ON = 0,
    
    .NU_X_BAND_MAX = 2000.0,
    .NU_X_MAX = 10000.0,
    
    .NBINS_LF = 100,
    
    .P_CUTOFF = 0,
    .M_WDM = 2,
    .g_x = 1.5,
    .OMn = 0.0,
    .OMk = 0.0,
    .OMr = 8.6e-5,
    .OMtot = 1.0,
    .Y_He = 0.245,
    .wl = -1.0,
    .SHETH_b = 0.15,
    .SHETH_c = 0.05,
    .Zreion_HeII = 3.0,
    .FILTER = 0,
    .POWER_SPECTRUM = 0,
};

/*

// ----------------------------------------------------------------------------------------- //

// Taken from ANAL_PARAMS.H

// ----------------------------------------------------------------------------------------- //



static float ALPHA_UVB = 5.;
// Power law index of the UVB during the EoR.  This is only used if INHOMO_RECO is on,
// in order to compute the local mean free path inside the cosmic HII regions

static int EVOLVE_DENSITY_LINEARLY = 0;
// EVOLVE_DENSITY_LINEARLY = 1, evolve the density field with linear theory.
// If choosing this option, make sure that your cell size is
// in the linear regime at the redshift of interest
 
// EVOLVE_DENSITY_LINEARLY = 0, evolve the density field with 1st order perturbation theory.
// If choosing this option, make sure that you resolve small
// enough scales, roughly we find BOX_LEN/DIM should be < 1Mpc

static int SMOOTH_EVOLVED_DENSITY_FIELD = 0;
static float R_smooth_density = 0.2;
// If set to 1, the ZA density field is additionally smoothed (asside from the implicit
// boxcar smoothing performed when re-binning the ICs from DIM to HII_DIM) with a Gaussian
// filter of width R_smooth_density*BOX_LEN/HII_DIM.  The implicit boxcar smoothing in
// perturb_field.c bins the density field on scale DIM/HII_DIM, similar to what Lagrangian
// codes do when constructing Eulerian grids. In other words, the density field, \delta,
// is quantized into (DIM/HII_DIM)^3 values. If your usage requires smooth density fields,
// it is recommended to set SMOOTH_EVOLVED_FIELD to 1.  This also decreases the shot noise
// present in all grid based codes, though it overcompensates by an effective loss in resolution.
// New in v1.1

static int SECOND_ORDER_LPT_CORRECTIONS = 1;
// Use second-order Lagrangian perturbation theory (2LPT).
// Set this to 1 if the density field or the halo positions are extrapolated to low redshifts.
// The current implementation is very naive and add a factor ~6 to the memory requirements.
// Reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D

static float HII_ROUND_ERR = 1e-3;
// Allows one to set a flag allowing find_HII_bubbles to skip constructing the
// ionization field if it is estimated that the mean neutral fraction, <xH>, is
// within HII_ROUND_ERR of 1. In other words, if <xH> > 1-HII_ROUND_ERR,
// then find_HII_bubbles just prints a homogeneous xHI field
// of  1's.
 // This is a new option in v1.1. Previous versions had a hardcoded value of 1e-15.

static int FIND_BUBBLE_ALGORITHM = 2;
// Choice of:
// 1 - Mesinger & Furlanetto 2007 method of overlaping spheres:
// paint an ionized sphere with radius R, centered on pixel
// where R is filter radius
// This method, while somewhat more accurate, is slower than (2) especially
// in mostly ionized unverses, so only use for lower resolution boxes (HII_DIM<~400)
// 2 - Center pixel only method (Zahn et al. 2007). this is faster.

static int N_POISSON = -1;
// If not using the halo field to generate HII regions, we provide the option of
// including Poisson scatter in the number of sources obtained through the conditional
// collapse fraction (which only gives the *mean* collapse fraction on a particular
// scale.  If the predicted mean collapse fraction is < N_POISSON * M_MIN,
// then Poisson scatter is added to mimic discrete halos on the subgrid scale
// (see Zahn+ 2010).
 
// NOTE: If you are interested in snapshots of the same realization at several redshifts,
// it is recommended to turn off this feature, as halos can stocastically
// "pop in and out of" existance from one redshift to the next...

static int T_USE_VELOCITIES = 1;
// Parameter choice of whether to use velocity corrections in 21-cm fields
// 1=use velocities in delta_T; 0=do not use velocities
// NOTE: The approximation used to include peculiar velocity effects works
// only in the linear regime, so be careful using this (see Mesinger+2010)

static float MAX_DVDR = 0.2;
// Maximum velocity gradient along the line of sight in units of the hubble parameter at z.
// This is only used in computing the 21cm fields.
// Note, setting this too high can add spurious 21cm power in the early stages, due to the
// 1-e^-tau ~ tau approximation (see my 21cm intro paper and mao+2011).  However, this is still
// a good approximation at the <~10% level.  Future builds will include the mao+ model for
// redshift space distortions.

static int VELOCITY_COMPONENT = 3;
// Component of the velocity to be used in 21-cm temperature maps
// 1 = x
// 2 = y
// 3 = z

static int DIMENSIONAL_T_POWER_SPEC = 1;
// 0 = plot 21cm temperature power spectrum in non-dimensional units
// 1 = plot 21cm...  in mK^2

static float DELTA_R_FACTOR = 1.1;
// factor by which to scroll through filter radius for halos

static float DELTA_R_HII_FACTOR = 1.1;
// factor by which to scroll through filter radius for bubbles

static float R_OVERLAP_FACTOR = 1.0;
// Factor of the halo's radius, R, so that the effective radius
// is R_eff = R_OVERLAP_FACTOR * R.  Halos whose centers are less than
// R_eff away from another halo are not allowed.
// R_OVERLAP_FACTOR = 1 is fully disjoint
// R_OVERLAP_FACTOR = 0 means that centers are allowed to luy on the edges of
// neighboring halos

static int DELTA_CRIT_MODE = 1;
// 0 = delta_crit is constant 1.68
// 1 = delta_crit is the sheth tormen ellipsoidal collapse
// correction to delta_crit (see ps.c)

static int HII_FILTER = 1;
// Filter for the Halo or density field used to generate ionization field
// 0 = use real space top hat filter
// 1 = use k-space top hat filter
// 2 = use gaussian filter

static float INITIAL_REDSHIFT = 300.; // used to perturb field



// ----------------------------------------------------------------------------------------- //

// Taken from HEAT_PARAMS.H

// ----------------------------------------------------------------------------------------- //



static int HEAT_FILTER = 0;
// Filter used for smoothing the linear density field to obtain the collapsed fraction in Ts.c
// 0 = use real space top hat filter
// 1 = use sharp k-space filter
// 2 = use gaussian filter

static double CLUMPING_FACTOR = 2;
// sub grid scale.  note that if you want to run-down from a very high redshift (>50),
// you should set this to one..

static float Z_HEAT_MAX = 35;
// Maximum redshift used in the Tk and x_e evolution equations.
// Temperature and x_e are assumed to be homogenous at higher redshifts.

static float R_XLy_MAX = 500;
// Maximum radius of influence for computing Xray and Lya pumping in cMpc. This
// should be larger than the mean free path of the relevant photons.  If increasing,
// you might want to adjust the z'' loop in Ts.c to skip integrating from early times,
// to increase speed (i.e. the code integrates as far back as ZPP_MAX, where R_XLy_MAX
// is the comoving separation between ZPP_MAX and Z_HEAT_MAX).

static int NUM_FILTER_STEPS_FOR_Ts = 40;
// Number of spherical anulli used to compute df_coll/dz' in the simulation box.
// The spherical annulii are evenly spaced in log R, ranging from the cell size to the box size.
// Ts.c will create this many boxes of size HII_DIM, so be wary of memory usage if values are high.

static float ZPRIME_STEP_FACTOR = 1.02;
// Redshift step-size used in the z' integral.  Logarithmic dz.

static double TK_at_Z_HEAT_MAX = -1;
static double XION_at_Z_HEAT_MAX = -1;
// If the following are >0, then the user chooses to overwrite the default boundary conditions
// at Z_HEAT_MAX obtained from RECFAST, and use his/her own.

static int Ts_verbose = 1;
// Set this to 1, if you want the spin temperature field printed at eary step of the evolution.
// This would be useful if investigating redshift evolution, so Ts.c needs to be called only once
// at the low redshift, and then one has all of the higher redshift Ts boxes upon completion.

static int Pop = 2;
// Stellar Population responsible for early heating
// Pop == 2 Pop2 stars heat the early universe
// Pop == 3 Pop3 stars heat the early universe

static float Pop2_ion = 4361;
static float Pop3_ion = 44021;
// Number of ionizing photons per baryon of the two stellar species

static int DEBUG_ON = 0;
// Flag to turn off or on verbose status messages in Ts.  The GSL libraries are very finicky,
// and this is useful when to help issolate why Ts crashed, if it did...



// ----------------------------------------------------------------------------------------- //

// Taken from COSMOLOGY.H

// ----------------------------------------------------------------------------------------- //



//      New in v1.1.        //
//      WDM parameters      //
static int P_CUTOFF = 0; // supress the power spectrum? 0= CDM; 1=WDM
static float M_WDM = 2; // mass of WDM particle in keV.  this is ignored if P_CUTOFF is set to zero
static float g_x = 1.5; // degrees of freedom of WDM particles; 1.5 for fermions

static float OMn = 0.0;
static float OMk = 0.0;
static float OMr = 8.6e-5;
static float OMtot = 1.0;
static float Y_He = 0.245;
static float wl = -1.0; // dark energy equation of state parameter (wl = -1 for vacuum )

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
static float SHETH_b = 0.15; //  1-D realisation best fit from Barkana et al. 2001: SHETH_b 0.34
static float SHETH_c = 0.05; // 1-D realisation best fit from Barkana et al. 2001: SHETH_c 0.81

static double Zreion_HeII = 3; // redshift of helium reionization, currently only used for tau_e

static int FILTER = 0; // smoothing: 0=tophat, 1=gaussian
static int POWER_SPECTRUM = 0; // EH=0 BBKS=1  EFSTATHIOU=2  PEEBLES=3  WHITE=4

*/