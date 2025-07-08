#ifndef _SPINTEMP_H
#define _SPINTEMP_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeTsBox(float redshift, float prev_redshift, float perturbed_field_redshift, short cleanup,
                 PerturbedField *perturbed_field, XraySourceBox *source_box,
                 TsBox *previous_spin_temp, InitialConditions *ini_boxes, TsBox *this_spin_temp);

int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct, double r_star,
                        XraySourceBox *source_box);

#endif
