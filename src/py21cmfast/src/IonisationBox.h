#ifndef _IONBOX_H
#define _IONBOX_H

#include <fftw3.h>
#include <cuda_runtime.h>

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif
int ComputeIonizedBox(float redshift, float prev_redshift, UserParams *user_params, CosmoParams *cosmo_params,
                        AstroParams *astro_params, FlagOptions *flag_options,
                        PerturbedField *perturbed_field, PerturbedField *previous_perturbed_field,
                        IonizedBox *previous_ionize_box, TsBox *spin_temp,
                        HaloBox *halos, InitialConditions *ini_boxes,
                        IonizedBox *box);
void init_ionbox_gpu_data(
    fftwf_complex **d_deltax_filtered, // copies of pointers to pointers
    fftwf_complex **d_N_rec_filtered,
    fftwf_complex **d_xe_filtered,
    float **d_y_arr,
    float **d_Fcoll,
    unsigned int nbins, // nbins for Nion_conditional_table1D->y
    unsigned long long hii_tot_num_pixels, // HII_TOT_NUM_PIXELS
    unsigned long long hii_kspace_num_pixels, // HII_KSPACE_NUM_PIXELS
    unsigned int *threadsPerBlock,
    unsigned int *numBlocks
);
void calculate_fcoll_grid_gpu(
    IonizedBox *box, // for box->Fcoll
    fftwf_complex *h_deltax_filtered, // members of fg_struct
    fftwf_complex *h_N_rec_filtered,
    fftwf_complex *h_xe_filtered,
    bool filter_recombinations, // member of consts
    double *f_coll_grid_mean, // member of rspec
    fftwf_complex *d_deltax_filtered, // device pointers
    fftwf_complex *d_N_rec_filtered,
    fftwf_complex *d_xe_filtered,
    float *d_Fcoll,
    float *d_y_arr,
    unsigned long long hii_tot_num_pixels, // HII_TOT_NUM_PIXELS
    unsigned long long hii_kspace_num_pixels, // HII_KSPACE_NUM_PIXELS
    unsigned int *threadsPerBlock,
    unsigned int *numBlocks
);
void free_ionbox_gpu_data(
    fftwf_complex **d_deltax_filtered, // copies of pointers to pointers
    fftwf_complex **d_N_rec_filtered,
    fftwf_complex **d_xe_filtered,
    float **d_y_arr,
    float **d_Fcoll
);

#ifdef __cplusplus
}
#endif
#endif
