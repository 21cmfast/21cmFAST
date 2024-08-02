#ifndef _SPINTEMP_H
#define _SPINTEMP_H

#include "InputParamters.h"
#include "InitialConditions.h"
#include "PerturbField.h"
#include "HaloBox.h"

typedef struct TsBox{
    float *Ts_box;
    float *x_e_box;
    float *Tk_box;
    float *J_21_LW_box;
}TsBox;

typedef struct XraySourceBox{
    float *filtered_sfr;
    float *filtered_xray;
    float *filtered_sfr_mini;

    double *mean_log10_Mcrit_LW;
    double *mean_sfr;
    double *mean_sfr_mini;
}XraySourceBox;

void Broadcast_struct_global_TS(UserParams *user_params, CosmoParams *cosmo_params,AstroParams *astro_params, FlagOptions *flag_options);

int ComputeTsBox(float redshift, float prev_redshift, UserParams *user_params, CosmoParams *cosmo_params,
                  AstroParams *astro_params, FlagOptions *flag_options,
                  float perturbed_field_redshift, short cleanup,
                  PerturbedField *perturbed_field, XraySourceBox *source_box, TsBox *previous_spin_temp,
                  InitialConditions *ini_boxes, TsBox *this_spin_temp);

int UpdateXraySourceBox(UserParams *user_params, CosmoParams *cosmo_params,
                  AstroParams *astro_params, FlagOptions *flag_options, HaloBox *halobox,
                  double R_inner, double R_outer, int R_ct, XraySourceBox *source_box);

#endif
