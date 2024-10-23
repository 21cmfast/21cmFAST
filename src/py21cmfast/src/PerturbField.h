#ifndef _PERTURBFIELD_H
#define _PERTURBFIELD_H

#include <cuda_runtime.h>

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif
int ComputePerturbField(
    float redshift, UserParams *user_params, CosmoParams *cosmo_params,
    InitialConditions *boxes, PerturbedField *perturbed_field
);
int ComputePerturbField_cpu(
    float redshift, UserParams *user_params, CosmoParams *cosmo_params,
    InitialConditions *boxes, PerturbedField *perturbed_field
);
int ComputePerturbField_gpu(
    float redshift, UserParams *user_params, CosmoParams *cosmo_params,
    InitialConditions *boxes, PerturbedField *perturbed_field
);

#ifdef __cplusplus
}
#endif
#endif
