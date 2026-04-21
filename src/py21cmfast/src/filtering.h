#ifndef _FILTERING_H
#define _FILTERING_H

#include <complex.h>
#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

void filter_box(fftwf_complex *box, int box_dim[3], int filter_type, float R, float R_param,
                float R_star);
void filter_box_cpu(fftwf_complex *box, int box_dim[3], int filter_type, float R, float R_param,
                    float R_star);
void filter_box_gpu(fftwf_complex *box, int box_dim[3], int filter_type, float R, float R_param,
                    float R_star);
void fill_Rbox_table_gpu(float **result, fftwf_complex *unfiltered_box, double *R_array, int n_R,
                         double min_value, double const_factor, double *min_arr,
                         double *average_arr, double *max_arr);
void filter_and_transform_gpu(fftwf_complex *box, int box_dim[3], int filter_type, float R,
                              float R_param, int plan);
void filter_box_gpu_inplace(void *d_box, int box_dim[3], int filter_type, float R, float R_param);
int create_cufft_c2r_plan(int box_dim[3]);
void destroy_cufft_plan(int plan);

/* Device filter buffer management */
#define MAX_DEVICE_FILTER_FIELDS 8
struct DeviceFilterBuffers {
    void *d_fields[MAX_DEVICE_FILTER_FIELDS];
    void *h_fields[MAX_DEVICE_FILTER_FIELDS];
    int n_fields;
    void *d_working;
    int plan;
    unsigned long long kspace_size;
    unsigned long long real_padded_size;
    int box_dim[3];
    void *d_deltax_real; /* persistent copy of filtered deltax for Fcoll GPU path */
    void *d_xe_real;     /* persistent copy of filtered xe for Fcoll GPU path */
};
void device_memcpy(void *dst, void *src, unsigned long long size);
void init_device_filter_buffers(struct DeviceFilterBuffers *bufs, int box_dim[3],
                                fftwf_complex **h_unfiltered_fields, int n_fields);
void free_device_filter_buffers(struct DeviceFilterBuffers *bufs);
void filter_and_transform_device(void *d_source, void *d_working, fftwf_complex *h_output,
                                 int box_dim[3], int filter_type, float R, float R_param, int plan);
int test_filter(float *input_box, double R, double R_param, double R_star, int filter_flag,
                double *result);
int test_filter_cpu(float *input_box, double R, double R_param, double R_star, int filter_flag,
                    double *result);
int test_filter_gpu(float *input_box, double R, double R_param, double R_star, int filter_flag,
                    double *result);
double filter_function(double k, int filter_type);
double dwdm_filter(double k, double R, int filter_type);

#ifdef __cplusplus
}
#endif
#endif
