#ifndef _PTHALOS_H
#define _PTHALOS_H

#include "InputParameters.h"
#include "InitialConditions.h"
#include "HaloField.h"

typedef struct PerturbHaloField{
    long long unsigned int n_halos;
    long long unsigned int buffer_size;
    float *halo_masses;
    int *halo_coords;

    //Halo properties for stochastic model
    float *star_rng;
    float *sfr_rng;
    float *xray_rng;
} PerturbHaloField;

int ComputePerturbHaloField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                            struct AstroParams *astro_params, struct FlagOptions *flag_options,
                            struct InitialConditions *boxes, struct HaloField *halos,
                            struct PerturbHaloField *halos_perturbed);

#endif
