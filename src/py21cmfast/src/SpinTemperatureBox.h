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

int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct, double R_star,
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

// Phase 11.6a: Spectral integration GPU kernel
// Opaque struct — defined in SpinTemperatureBox.cu, used as void* from C
typedef struct SpectralIntegDeviceData SpectralIntegDeviceData;

SpectralIntegDeviceData *init_spectral_integration_gpu(
    unsigned long long num_pixels,
    int n_step_ts,
    double **freq_int_heat_tbl,
    double **freq_int_ion_tbl,
    double **freq_int_lya_tbl,
    double **freq_int_heat_tbl_diff,
    double **freq_int_ion_tbl_diff,
    double **freq_int_lya_tbl_diff,
    int *m_xHII_low_box,
    float *inverse_val_box,
    bool use_mini_halos,
    bool use_x_ray_heating,
    bool use_lya_heating
);

void launch_spectral_integration_kernel(
    SpectralIntegDeviceData *dev,
    int R_ct,
    float *del_fcoll_Rct,
    float *del_fcoll_Rct_MINI,
    double z_edge_factor,
    double xray_R_factor,
    double avg_fix_term,
    double avg_fix_term_MINI,
    double F_STAR10,
    double L_X,
    double s_per_yr,
    double F_STAR7_MINI,
    double L_X_MINI,
    double dstarlya_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_MINI_R,
    double dstarlya_dt_prefactor_MINI_R,
    double dstarlya_cont_dt_prefactor_R,
    double dstarlya_inj_dt_prefactor_R,
    double dstarlya_cont_dt_prefactor_MINI_R,
    double dstarlya_inj_dt_prefactor_MINI_R
);

void download_spectral_integration_results(
    SpectralIntegDeviceData *dev,
    double *dxheat_dt_box,
    double *dxion_source_dt_box,
    double *dxlya_dt_box,
    double *dstarlya_dt_box,
    double *dstarlyLW_dt_box,
    double *dstarlya_cont_dt_box,
    double *dstarlya_inj_dt_box
);

void free_spectral_integration_gpu(SpectralIntegDeviceData *dev);

// Phase 11.6b: Spin temperature GPU kernel
void launch_spin_temperature_kernel(
    unsigned long long num_pixels,
    float zp, float dzp,
    double xray_prefactor, double volunit_inv, double lya_star_prefactor, double Nb_zp,
    double Trad, double Trad_inv, double Ts_prefactor, double xa_tilde_prefactor,
    double xc_inverse, double dcomp_dzp_prefactor, double hubble_zp, double N_zp,
    double growth_zp, double dgrowth_dzp, double dt_dzp,
    double growth_factor_zp, double inverse_growth_factor_z,
    double No_val, double N_b0_val, double H_FRAC_val, double HE_FRAC_val,
    double CLUMPING_FACTOR,
    double A10, double c_cms, double lambda_21, double k_B, double h_p, double T_21, double m_p,
    bool use_x_ray_heating, bool use_mini_halos,
    double *dxheat_dt_box, double *dxion_source_dt_box,
    double *dxlya_dt_box, double *dstarlya_dt_box, double *dstarlyLW_dt_box,
    float *density, float *prev_spin_temperature,
    float *prev_kinetic_temp, float *prev_xray_ionised_fraction,
    float *spin_temperature, float *kinetic_temp,
    float *xray_ionised_fraction, float *J_21_LW
);

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
