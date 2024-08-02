#ifndef _PERTURBFIELD_H
#define _PERTURBFIELD_H

#include "InputParameters.h"
#include "InitialConditions.h"

typedef struct PerturbedField{
    float *density, *velocity_x, *velocity_y, *velocity_z;
} PerturbedField;

int ComputePerturbField(
    float redshift, UserParams *user_params, CosmoParams *cosmo_params,
    InitialConditions *boxes, PerturbedField *perturbed_field
);

#endif
