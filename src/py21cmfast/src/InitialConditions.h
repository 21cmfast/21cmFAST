
#ifndef _INITCONDITIONS_H
#define _INITCONDITIONS_H

#include "InputParameters.h"

typedef struct InitialConditions{
    float *lowres_density, *lowres_vx, *lowres_vy, *lowres_vz, *lowres_vx_2LPT, *lowres_vy_2LPT, *lowres_vz_2LPT;
    float *hires_density, *hires_vx, *hires_vy, *hires_vz, *hires_vx_2LPT, *hires_vy_2LPT, *hires_vz_2LPT; //cw addition
    float *lowres_vcb;
} InitialConditions;

int ComputeInitialConditions(
    unsigned long long random_seed, UserParams *user_params,
    CosmoParams *cosmo_params, InitialConditions *boxes
);

#endif
