#ifndef _SPINTEMP_H
#define _SPINTEMP_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeTsBox(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                 RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                 InitialConditions *ini_boxes, TsBox *this_spin_temp);

#endif
