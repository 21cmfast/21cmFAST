#ifndef _THERMOCHEM_H
#define _THERMOCHEM_H

#include "InputParameters.h"

float ComputeTau(UserParams *user_params, CosmoParams *cosmo_params, int Npoints, float *redshifts, float *global_xHI);
double molecular_cooling_threshold(float z);
double atomic_cooling_threshold(float z);
double lyman_werner_threshold(float z, float J_21_LW, float vcb, AstroParams *astro_params);
double reionization_feedback(float z, float Gamma_halo_HII, float z_IN);
float ComputeFullyIoinizedTemperature(float z_re, float z, float delta);
float ComputePartiallyIoinizedTemperature(float T_HI, float res_xH);

#endif
