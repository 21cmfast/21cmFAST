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

void calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
                  float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini, unsigned int threadsPerBlock,
                  float *d_y_arr, float *d_dens_R_grid, float *d_sfrd_grid, double *d_ave_sfrd_buf);

// simple
void init_sfrd_gpu_data_simple(
    float *dens_R_grid, // input data
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    unsigned int nbins, // nbins for sfrd_grid->y
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_fcoll_tmp
);

double calculate_sfrd_gpu_simple(
    RGTable1D_f *SFRD_conditional_table, // input data
    float *dens_R_grid, // input data
    double *zpp_growth, // input data
    int R_ct, // filter step/loop iteration/spherical annuli (out of 40 by default)
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    float *d_y_arr,
    float *d_dens_R_grid,
    float *d_sfrd_grid,
    double *d_fcoll_tmp
);

void free_sfrd_gpu_data_simple(
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_fcoll_tmp
);

// complex
unsigned int init_sfrd_gpu_data(float *dens_R_grid, float *sfrd_grid, unsigned long long num_pixels, unsigned int nbins,
                  float **d_y_arr, float **d_dens_R_grid, float **d_sfrd_grid, double **d_ave_sfrd_buf);

double calculate_sfrd_from_grid_gpu(RGTable1D_f *SFRD_conditional_table, float *dens_R_grid,
                  double *zpp_growth, int R_ct, float *sfrd_grid, unsigned long long num_pixels, unsigned int threadsPerBlock,
                  float *d_y_arr, float *d_dens_R_grid, float *d_sfrd_grid, double *d_ave_sfrd_buf);

void free_sfrd_gpu_data(float **d_y_arr, float **d_dens_R_grid, float **d_sfrd_grid, double **d_ave_sfrd_buf);

// warp shuffle
void init_sfrd_gpu_data_ws(
    float *dens_R_grid, // input data
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    unsigned int nbins, // nbins for sfrd_grid->y
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_fcoll_tmp
);

double calculate_sfrd_gpu_ws(
    RGTable1D_f *SFRD_conditional_table, // input data
    float *dens_R_grid, // input data
    double *zpp_growth, // input data
    int R_ct, // filter step/loop iteration/spherical annuli (out of 40 by default)
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    float *d_y_arr,
    float *d_dens_R_grid,
    float *d_sfrd_grid,
    double *d_fcoll_tmp
);

void free_sfrd_gpu_data_ws(
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_fcoll_tmp
);

#ifdef __cplusplus
}
#endif
#endif
