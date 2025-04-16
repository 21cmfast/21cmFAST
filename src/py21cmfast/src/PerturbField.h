#ifndef _PERTURBFIELD_H
#define _PERTURBFIELD_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputePerturbField(float redshift, InitialConditions *boxes, PerturbedField *perturbed_field);

#endif
