#include "Constants.h"

#include <math.h>

// Values differing from the original 21cmFAST were taken from Astropy constants v7.1.0
const PhysicalConstants physconst = {
    .c_cms = 2.99792458e10,
    .c_kms = 2.99792458e5,
    .h_p = 6.62607015e-27,              // was 6.62606896e-27,
    .k_B = 1.380649e-16,                // was 1.380658e-16,
    .m_p = 1.67262192369e-24,           // was 1.6726231e-24,
    .m_e = 9.1093837015e-28,            // was 9.10938188e-28,
    .G = 6.6743e-8,                     // was 6.67259e-8,
    .e_charge = 4.803204712570263e-10,  // was 4.80320467e-10,
    .vac_perm = 8.8541878128e-12,

    .Msun = 1.989e33,
    .s_per_yr = 31556925.9747,
    .cm_per_Mpc = 3.08567758e24,  // was 3.086e24,
    .eV_to_Hz = 2.417989e14,  // was (1.60217646e-12 / hplank) (can't do operations in this context)

    .nu_ion_HI = 3.288465e15,       // was (13.60 * NU_over_EV),
    .nu_ion_HeI = 5.945836e15,      // was (24.59 * NU_over_EV)
    .nu_ion_HeII = 1.3153862e16,    // was (NUIONIZATION * 4),
    .nu_LW_thresh = 2.70331197e15,  // was (11.18 * NU_over_EV),
    .nu_Ly_alpha = 2.46606727e15,
    .T_cmb = 2.7255,
    .T_21 = 0.0682,
    .lambda_21 = 21.106114054160,
    .lambda_Ly_alpha = 1215.67,
    .lambda_Ly_beta = 1025.18,
    .lambda_Ly_gamma = 972.02,

    .sigma_T = 6.6524587321e-25,  // was 6.6524e-25,
    .sigma_HI = 6.3e-18,
    .A10 = 2.85e-15,
    .A_Ly_alpha = 6.24e8,
    .f_alpha = 0.4162,
    .alpha_A_10k = 4.18e-13,  // taken from osterbrock
    .alpha_B_10k = 2.59e-13,  // taken from osterbrock
    .alpha_B_20k = 2.52e-13,  // taken from osterbrock

    .l_factor = 0.620350491,
    .delta_c_sph = 1.686,  // was 1.68
    .delta_c_delos = 1.5};
