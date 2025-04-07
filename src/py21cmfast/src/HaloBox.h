#ifndef _HALOBOX_H
#define _HALOBOX_H

#include "HaloField.h"
#include "InitialConditions.h"
#include "InputParameters.h"
#include "IonisationBox.h"
#include "OutputStructs.h"
#include "PerturbHaloField.h"
#include "SpinTemperatureBox.h"

// Compute the HaloBox Object
int ComputeHaloBox(double redshift, InitialConditions *ini_boxes, PerturbedField *perturbed_field,
                   PerturbHaloField *halos, TsBox *previous_spin_temp,
                   IonizedBox *previous_ionize_box, HaloBox *grids);

#endif
