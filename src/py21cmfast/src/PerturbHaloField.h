#ifndef _PTHALOS_H
#define _PTHALOS_H

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif
int ComputePerturbHaloField(float redshift, InitialConditions *boxes, HaloField *halos,
                            PerturbHaloField *halos_perturbed);

#ifdef __cplusplus
}
#endif
#endif
