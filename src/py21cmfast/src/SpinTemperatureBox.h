#ifndef _SPINTEMP_H
#define _SPINTEMP_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeTsBox(short cleanup, PerturbedField *perturbed_field, XraySourceBox *source_box,
                 TsBox *previous_spin_temp, InitialConditions *ini_boxes, TsBox *this_spin_temp);

int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct,
                        XraySourceBox *source_box);

#endif
