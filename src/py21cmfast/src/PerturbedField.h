#ifndef _PERTURBFIELD_H
#define _PERTURBFIELD_H

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif

int ComputePerturbedField(float redshift, InitialConditions *boxes,
                          PerturbedField *perturbed_field);

#ifdef __cplusplus
}
#endif
#endif
