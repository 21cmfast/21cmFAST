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
    float HII_ROUND_ERR;
    int FIND_BUBBLE_ALGORITHM;
    int N_POISSON;
    int T_USE_VELOCITIES;
    float MAX_DVDR;
    float DELTA_R_HII_FACTOR;
    float DELTA_R_FACTOR;
    int HII_FILTER;
    float INITIAL_REDSHIFT;
    float R_OVERLAP_FACTOR;
    int DELTA_CRIT_MODE;
    int HALO_FILTER;
    int OPTIMIZE;
    float OPTIMIZE_MIN_MASS;


    float CRIT_DENS_TRANSITION;
    float MIN_DENSITY_LOW_LIMIT;

    int RecombPhotonCons;
    float PhotonConsStart;
    float PhotonConsEnd;
    float PhotonConsAsymptoteTo;
    float PhotonConsEndCalibz;

    int HEAT_FILTER;
    double CLUMPING_FACTOR;
    float Z_HEAT_MAX;
    float R_XLy_MAX;
    int NUM_FILTER_STEPS_FOR_Ts;
    float ZPRIME_STEP_FACTOR;
    double TK_at_Z_HEAT_MAX;
    double XION_at_Z_HEAT_MAX;
    int Pop;
    float Pop2_ion;
    float Pop3_ion;

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

    char *external_table_path;
    char *wisdoms_path;
    float R_BUBBLE_MIN;
    float M_MIN_INTEGRAL;
    float M_MAX_INTEGRAL;

    float T_RE;

    float VAVG;

    bool USE_FAST_ATOMIC; //whether to apply the fast fcoll tables for atomic cooling haloes, usually turned off as it's not a big computational cost and those can deviate ~5-10% at z<10.
    bool USE_ADIABATIC_FLUCTUATIONS;
};

extern struct GlobalParams global_params = {

    .ALPHA_UVB = 5.0,
    .EVOLVE_DENSITY_LINEARLY = 0,
    .SMOOTH_EVOLVED_DENSITY_FIELD = 0,
    .R_smooth_density = 0.2,
    .HII_ROUND_ERR = 1e-5,
    .FIND_BUBBLE_ALGORITHM = 2,
    .N_POISSON = 5,
    .T_USE_VELOCITIES = 1,
    .MAX_DVDR = 0.2,
    .DELTA_R_HII_FACTOR = 1.1,
    .DELTA_R_FACTOR = 1.1,
    .HII_FILTER = 1,
    .INITIAL_REDSHIFT = 300.,
    .R_OVERLAP_FACTOR = 1.,
    .DELTA_CRIT_MODE = 1,
    .HALO_FILTER = 0,
    .OPTIMIZE = 0,
    .OPTIMIZE_MIN_MASS = 1e11,


    .CRIT_DENS_TRANSITION = 1.5,
    .MIN_DENSITY_LOW_LIMIT = 9e-8,

    .RecombPhotonCons = 0,
    .PhotonConsStart = 0.995,
    .PhotonConsEnd = 0.3,
    .PhotonConsAsymptoteTo = 0.01,
    .PhotonConsEndCalibz = 3.5,

    .HEAT_FILTER = 0,
    .CLUMPING_FACTOR = 2.,
    .Z_HEAT_MAX = 35.0,
    .R_XLy_MAX = 500.,
    .NUM_FILTER_STEPS_FOR_Ts = 40,
    .ZPRIME_STEP_FACTOR = 1.02,
    .TK_at_Z_HEAT_MAX = -1,
    .XION_at_Z_HEAT_MAX = -1,
    .Pop = 2,
    .Pop2_ion = 5000,
    .Pop3_ion = 44021,

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
    .R_BUBBLE_MIN = 0.620350491,
    .M_MIN_INTEGRAL = 1e5,
    .M_MAX_INTEGRAL = 1e16,

    .T_RE = 2e4,

    .VAVG=25.86,

    .USE_FAST_ATOMIC = 0,
    .USE_ADIABATIC_FLUCTUATIONS = 1,
};
