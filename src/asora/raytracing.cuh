#pragma once
#include <cuda_runtime.h>

// ========================================================================
// Header file for OCTA raytracing library.
// Functions defined and documented in raytracing_gpu.cu
// ========================================================================

// Modulo function with Fortran convention
inline int modulo(const int & a,const int & b);
inline __device__ int modulo_gpu(const int & a,const int & b);

// Device sign function
inline __device__ int sign_gpu(const double & x);

// Flat array index from i,j,k coordinates
inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N);

// Mapping from linear thread space to the cartesian coords of a q-shell in asora
__device__ void linthrd2cart(const int &,const int &,int&,int&);

// Raytrace all sources and compute photoionization rates
void do_all_sources_gpu(
    const double & R,
    double* coldensh_out,
    const double & sig,
    const double & dr,
    double* ndens,
    double* xh_av,
    double* phi_ion,
    const int & NumSrc,
    const int & m1,
    const double & minlogtau,
    const double & dlogtau,
    const int & NumTau);

// Raytracing kernel, called by do_all_sources
__global__ void evolve0D_gpu(
    const double Rmax_LLS,
    const int q,
    const int ns_start,
    const int num_src_par,
    const int NumSrc,
    int* src_pos,
    double* src_flux,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1,
    const double* photo_thin_table,
    const double* photo_thick_table,
    const double minlogtau,
    const double dlogtau,
    const int NumTau,
    const int last_l,
    const int last_r
);

// Short-characteristics interpolation function from C2Ray
__device__ void cinterp_gpu(
    const int i,
    const int j,
    const int k,
    const int i0,
    const int j0,
    const int k0,
    double & cdensi,
    double & path,
    double* coldensh_out,
    const double sigma_HI_at_ion_freq,
    const int & m1);


// Check if point is in domain (deprecated)
inline __device__ bool in_box_gpu(const int & i,const int & j,const int & k,const int & N)
{
    return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
}