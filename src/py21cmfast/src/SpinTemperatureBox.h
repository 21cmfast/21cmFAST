#ifndef _SPINTEMP_H
#define _SPINTEMP_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeTsBox(float redshift, float prev_redshift, UserParams *user_params, CosmoParams *cosmo_params,
                  AstroParams *astro_params, FlagOptions *flag_options,
                  float perturbed_field_redshift, short cleanup,
                  PerturbedField *perturbed_field, XraySourceBox *source_box, TsBox *previous_spin_temp,
                  InitialConditions *ini_boxes, TsBox *this_spin_temp);

int UpdateXraySourceBox(UserParams *user_params, CosmoParams *cosmo_params,
                  AstroParams *astro_params, FlagOptions *flag_options, HaloBox *halobox,
                  double R_inner, double R_outer, int R_ct, XraySourceBox *source_box);

#endif
