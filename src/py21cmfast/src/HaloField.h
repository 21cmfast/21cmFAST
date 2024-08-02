/* Function Prototypes and definitions for the Halo Catalogs */
#ifndef _HALOFIELD_H
#define _HALOFIELD_H

#include "InputParameters.h"
#include "InitialConditions.h"

typedef struct HaloField{
    long long unsigned int n_halos;
    long long unsigned int buffer_size;
    float *halo_masses;
    int *halo_coords;

    //Halo properties for stochastic model
    float *star_rng;
    float *sfr_rng;
    float *xray_rng;
} HaloField;

int ComputeHaloField(float redshift_desc, float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, unsigned long long int random_seed,
                     struct HaloField * halos_desc, struct HaloField *halos);

#endif
