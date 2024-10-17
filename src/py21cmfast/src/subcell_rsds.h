#ifndef _SUBCELL_RSD_H
#define _SUBCELL_RSD_H

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif
double apply_subcell_rsds(
    UserParams *user_params,
    CosmoParams *cosmo_params,
    FlagOptions *flag_options,
    AstroParams *astro_params,
    IonizedBox *ionized_box,
    BrightnessTemp *box,
    float redshift,
    TsBox *spin_temp,
    float T_rad,
    float *v,
    float H
);

#ifdef __cplusplus
}
#endif
#endif
