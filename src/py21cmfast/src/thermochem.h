#ifndef _THERMOCHEM_H
#define _THERMOCHEM_H

#include "InputParameters.h"

float ComputeTau(UserParams *user_params, CosmoParams *cosmo_params, int Npoints, float *redshifts, float *global_xHI);

#endif