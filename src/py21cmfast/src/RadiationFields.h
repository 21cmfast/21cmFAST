#ifndef _RADIATIONFIELDS_H
#define _RADIATIONFIELDS_H

#include "OutputStructs.h"

int UpdateRadiationFields(float redshift, EmissivityFields *emissivity_fields, int R_ct,
                          double R_star, PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                          RadiationFieldsSetup *rad_setup, RadiationFields *radiation_fields);

#endif
