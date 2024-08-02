#ifndef _IONBOX_H
#define _IONBOX_H

#include "InputParameters.h"
#include "InitialConditions.h"
#include "PerturbField.h"
#include "PerturbHaloField.h"
#include "SpintemperatureBox.h"
#include "HaloBox.h"

struct IonizedBox{
    double mean_f_coll;
    double mean_f_coll_MINI;
    double log10_Mturnover_ave;
    double log10_Mturnover_MINI_ave;
    float *xH_box;
    float *Gamma12_box;
    float *MFP_box;
    float *z_re_box;
    float *dNrec_box;
    float *temp_kinetic_all_gas;
    float *Fcoll;
    float *Fcoll_MINI;
};

int ComputeIonizedBox(float redshift, float prev_redshift, UserParams *user_params, CosmoParams *cosmo_params,
                        AstroParams *astro_params, FlagOptions *flag_options,
                        PerturbedField *perturbed_field, PerturbedField *previous_perturbed_field,
                        IonizedBox *previous_ionize_box, TsBox *spin_temp,
                        HaloBox *halos, InitialConditions *ini_boxes,
                        IonizedBox *box);

#endif
