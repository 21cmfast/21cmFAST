#ifndef _HALOBOX_H
#define _HALOBOX_H

#include "InputParameters.h"
#include "IonisationBox.h"
#include "SpinTemperatureBox.h"
#include "InitialConditions.h"
#include "HaloField.h"
#include "PerturbHaloField.h"

#include "InputParameters.h"
#include "OutputStructs.h"

//Compute the HaloBox Object
int ComputeHaloBox(double redshift, UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params,
                    FlagOptions * flag_options, InitialConditions *ini_boxes, PerturbedField * perturbed_field, PerturbHaloField *halos,
                    TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, HaloBox *grids);

#endif
