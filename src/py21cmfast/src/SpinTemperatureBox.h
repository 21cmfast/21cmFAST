#ifndef _SPINTEMP_H
#define _SPINTEMP_H

#include <cuda_runtime.h>

#include "InputParameters.h"
#include "OutputStructs.h"
#include "interpolation.h"

#ifdef __cplusplus
extern "C" {
#endif
int ComputeTsBox(float redshift, float prev_redshift, UserParams *user_params, CosmoParams *cosmo_params,
                  AstroParams *astro_params, FlagOptions *flag_options,
                  float perturbed_field_redshift, short cleanup,
                  PerturbedField *perturbed_field, XraySourceBox *source_box, TsBox *previous_spin_temp,
                  InitialConditions *ini_boxes, TsBox *this_spin_temp);

int UpdateXraySourceBox(UserParams *user_params, CosmoParams *cosmo_params,
                  AstroParams *astro_params, FlagOptions *flag_options, HaloBox *halobox,
                  double R_inner, double R_outer, int R_ct, XraySourceBox *source_box);

// void calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
//                   float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini);

// double calculate_sfrd_from_grid_gpu(RGTable1D_f *SFRD_conditional_table, float *dens_R_grid, double *zpp_growth,
//                   int R_ct, float *sfrd_grid, unsigned int num_pixels);

void calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
                  float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini,
                  float *d_y_arr, float *d_dens_R_grid, float *d_sfrd_grid, double *d_ave_sfrd_buf);

double calculate_sfrd_from_grid_gpu(RGTable1D_f *SFRD_conditional_table, float *dens_R_grid, double *zpp_growth,
                  int R_ct, float *sfrd_grid, unsigned int num_pixels,
                  float *d_y_arr, float *d_dens_R_grid, float *d_sfrd_grid, double *d_ave_sfrd_buf);

#ifdef __cplusplus
}
#endif
#endif
