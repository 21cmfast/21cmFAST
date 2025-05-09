#ifndef _PERTURBFIELD_H
#define _PERTURBFIELD_H

// #include <cuda_runtime.h>

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif
int ComputePerturbField(float redshift, InitialConditions *boxes, PerturbedField *perturbed_field);
double *MapMass_cpu(InitialConditions *boxes, double *resampled_box, int dimension,
                    float f_pixel_factor, float init_growth_factor);
double *MapMass_gpu(InitialConditions *boxes, double *resampled_box, int dimension,
                    float f_pixel_factor, float init_growth_factor);

#ifdef __cplusplus
}
#endif
#endif
