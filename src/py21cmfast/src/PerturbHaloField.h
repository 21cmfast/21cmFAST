#ifndef _PTHALOS_H
#define _PTHALOS_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputePerturbHaloField(float redshift, InitialConditions *boxes, HaloField *halos,
                            PerturbHaloField *halos_perturbed);

#endif
