#ifndef _HALOBOX_H
#define _HALOBOX_H

#include "InputParamters.h"
#include "IonisationBox.h"
#include "SpinTemperatureBox.h"
#include "InitialConditions.h"
#include "HaloField.h"
#include "PerturbHaloField.h"

typedef struct HaloBox{
    //Things that aren't used in radiation fields but useful outputs
    float *halo_mass;
    float *halo_stars;
    float *halo_stars_mini;
    int *count;

    //For IonisationBox.c and SpinTemperatureBox.c
    float *n_ion; //weighted by F_ESC*PopN_ion
    float *halo_sfr; //for x-rays and Ts stuff
    float *halo_xray;
    float *halo_sfr_mini; //for x-rays and Ts stuff
    float *whalo_sfr; //SFR weighted by PopN_ion and F_ESC, used for Gamma12

    //Average volume-weighted log10 Turnover masses are kept in order to compare with the expected MF integrals
    double log10_Mcrit_ACG_ave;
    double log10_Mcrit_MCG_ave;
};

//Compute the HaloBox Object
int ComputeHaloBox(double redshift, UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params,
                    FlagOptions * flag_options, InitialConditions *ini_boxes, PerturbedField * perturbed_field, PerturbHaloField *halos,
                    TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, HaloBox *grids);

//Test function which stops early and returns galaxy properties for each halo
int test_halo_props(double redshift, UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params,
                    FlagOptions * flag_options, float * vcb_grid, float *J21_LW_grid, float *z_re_grid, float *Gamma12_ion_grid,
                    PerturbHaloField *halos, float *halo_props_out);

#endif
