#ifndef _SPINTEMP_H
#define _SPINTEMP_H

// #include <cuda_runtime.h>

#include "InputParameters.h"
#include "OutputStructs.h"
#include "interpolation.h"
#include "scaling_relations.h"

#ifdef __cplusplus
extern "C" {
#endif
// typedef struct sfrd_gpu_data {
//     float *d_y_arr;
//     float *d_dens_R_grid;
//     float *d_sfrd_grid;
//     double *d_ave_sfrd_buf;
// } sfrd_gpu_data;

int ComputeTsBox(float redshift, float prev_redshift, float perturbed_field_redshift, short cleanup,
                 PerturbedField *perturbed_field, XraySourceBox *source_box,
                 TsBox *previous_spin_temp, InitialConditions *ini_boxes, TsBox *this_spin_temp);

int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct,
                        XraySourceBox *source_box);

// pointers
// --------------------------------------------------------------------------------------------------------
void calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
                              float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini,
                              unsigned int threadsPerBlock, float *d_y_arr, float *d_dens_R_grid,
                              float *d_sfrd_grid, double *d_ave_sfrd_buf,
                              struct ScalingConstants *sc);

unsigned int init_sfrd_gpu_data(float *dens_R_grid, float *sfrd_grid, unsigned long long num_pixels,
                                unsigned int nbins, float **d_y_arr, float **d_dens_R_grid,
                                float **d_sfrd_grid, double **d_ave_sfrd_buf);

double calculate_sfrd_from_grid_gpu(RGTable1D_f *SFRD_conditional_table, float *dens_R_grid,
                                    double *zpp_growth, int R_ct, float *sfrd_grid,
                                    unsigned long long num_pixels, unsigned int threadsPerBlock,
                                    float *d_y_arr, float *d_dens_R_grid, float *d_sfrd_grid,
                                    double *d_ave_sfrd_buf, struct ScalingConstants *sc);

void free_sfrd_gpu_data(float **d_y_arr, float **d_dens_R_grid, float **d_sfrd_grid,
                        double **d_ave_sfrd_buf);

// wrap pointers in struct
// ------------------------------------------------------------------------------------------ void
// calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
//                   float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini,
//                   unsigned int threadsPerBlock, const sfrd_gpu_data *d_data);

// unsigned int init_sfrd_gpu_data(float *dens_R_grid, float *sfrd_grid, unsigned long long
// num_pixels,
//                   unsigned int nbins, sfrd_gpu_data *d_data);

// double calculate_sfrd_from_grid_gpu(RGTable1D_f *SFRD_conditional_table, float *dens_R_grid,
//                   double *zpp_growth, int R_ct, float *sfrd_grid, unsigned long long num_pixels,
//                   unsigned int threadsPerBlock, const sfrd_gpu_data *d_data);

// void free_sfrd_gpu_data(sfrd_gpu_data *d_data);

#ifdef __cplusplus
}
#endif
#endif
