#include <math.h>
// #include <cuComplex.h>
#include <cufft.h>
#include <stdio.h>
#include <fftw3.h>

// #include "logger.h"

#include "indexing.h"
#include "Constants.h"
#include "InputParameters.h"

#include "tiger_checks.h"

// device functions
__device__ double real_tophat_filter(double kR)
{
    // Second order taylor expansion around kR==0
    if (kR < 1e-4)
        return 1 - kR * kR / 10;
    return 3.0 * pow(kR, -3) * (sin(kR) - cos(kR) * kR);
}

__device__ double sharp_k_filter(double kR)
{
    // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
    if (kR * 0.413566994 > 1)
        return 0.;
    return 1;
}

__device__ double gaussian_filter(double kR_squared)
{
    return exp(-0.643 * 0.643 * kR_squared / 2.);
}

__device__ double exp_mfp_filter(double k, double R, double mfp, double exp_term)
{
    double f;

    double kR = k * R;
    double ratio = mfp / R;
    // Second order taylor expansion around kR==0
    // NOTE: the taylor coefficients could be stored and passed in
    //   but there aren't any super expensive operations here
    //   assuming the integer pow calls are optimized by the compiler
    //   test with the profiler
    if (kR < 1e-4)
    {
        double ts_0 = 6 * pow(ratio, 3) - exp_term * (6 * pow(ratio, 3) + 6 * pow(ratio, 2) + 3 * ratio);
        return ts_0 + (exp_term * (2 * pow(ratio, 2) + 0.5 * ratio) - 2 * ts_0 * pow(ratio, 2)) * kR * kR;
    }

    // Davies & Furlanetto MFP-eps(r) window function
    f = (kR * kR * pow(ratio, 2) + 2 * ratio + 1) * ratio * cos(kR);
    f += (kR * kR * (pow(ratio, 2) - pow(ratio, 3)) + ratio + 1) * sin(kR) / kR;
    f *= exp_term;
    f -= 2 * pow(ratio, 2);
    f *= -3 * ratio / pow(pow(kR * ratio, 2) + 1, 2);
    return f;
}

__device__ double spherical_shell_filter(double k, double R_outer, double R_inner)
{
    double kR_inner = k * R_inner;
    double kR_outer = k * R_outer;

    // Second order taylor expansion around kR_outer==0
    if (kR_outer < 1e-4)
        return 1. - kR_outer * kR_outer / 10 *
                        (pow(R_inner / R_outer, 5) - 1) /
                        (pow(R_inner / R_outer, 3) - 1);

    return 3.0 / (pow(kR_outer, 3) - pow(kR_inner, 3)) * (sin(kR_outer) - cos(kR_outer) * kR_outer - sin(kR_inner) + cos(kR_inner) * kR_inner);
}

// kernel function
__global__ void printComplexArray(cufftComplex *box, int num_elements, int dim_x, int dim_y, int dim_z, int midpoint, int RES, int filter_type, float R, float R_param, double delta_k, double delta_k_para)
{
    // global x, y, z index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // index dependent calculation
    float k_x, k_y, k_z, k_mag_sq, kR;
    if (z > midpoint)
    {
        k_x = (z - dim_z) * delta_k;
    }
    else
    {
        k_x = z * delta_k;
    }

    if (y > midpoint)
    {
        k_y = (y - dim_y) * delta_k;
    }
    else
    {
        k_y = y * delta_k;
    }

    k_z = x * delta_k_para;
    k_mag_sq = k_x * k_x + k_y * k_y + k_z * k_z;

    // setup constants if needed
    double R_const;
    if (filter_type == 3)
    {
        R_const = exp(-R / R_param);
    }

    // data index
    if (x < dim_x && y < dim_y && z < dim_z)
    {
        unsigned long long idx = x + dim_x * y + dim_x * dim_y * z;

        if (filter_type == 0)
        { // real space top-hat
            kR = sqrt(k_mag_sq) * R;
            box[idx].x *= real_tophat_filter(kR);
            box[idx].y *= real_tophat_filter(kR);
        }
        else if (filter_type == 1)
        { // k-space top hat
            // NOTE: why was this commented????
            //  This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
            //  kR = 0.17103765852*( k_x*k_x + k_y*k_y + k_z*k_z )*R*R;
            kR = sqrt(k_mag_sq) * R;
            box[idx].x *= sharp_k_filter(kR);
            box[idx].y *= sharp_k_filter(kR);
        }
        else if (filter_type == 2)
        { // gaussian
            // This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
            kR = k_mag_sq * R * R;
            box[idx].x *= gaussian_filter(kR);
            box[idx].y *= gaussian_filter(kR);
        }
        // The next two filters are not given by the HII_FILTER global, but used for specific grids
        else if (filter_type == 3)
        { // exponentially decaying tophat, param == scale of decay (MFP)
            // NOTE: This should be optimized, I havne't looked at it in a while
            box[idx].x *= exp_mfp_filter(sqrt(k_mag_sq), R, R_param, R_const);
            box[idx].y *= exp_mfp_filter(sqrt(k_mag_sq), R, R_param, R_const);
        }
        else if (filter_type == 4)
        { // spherical shell, R_param == inner radius
            box[idx].x *= spherical_shell_filter(sqrt(k_mag_sq), R, R_param);
            box[idx].y *= spherical_shell_filter(sqrt(k_mag_sq), R, R_param);
        }
        // else
        // {
        //     if ((x == 0) && (y == 0) && (z == 0))
        //         LOG_WARNING("Filter type %i is undefined. Box is unfiltered.", filter_type);
        // }

        // if (idx < num_elements)
        // {
        //     printf("Device Element %llu: (%f, %f)\n", idx, d_array[idx].x, d_array[idx].y);
        // }
    }
    
   
}

int checkComplextype(fftwf_complex *box, int total_elements, int xy_dim, int z_dim, int midpoint, int RES, int filter_type, float R, float R_param)
{
    const int num_elements = 16; // Number of elements to print

    // Print original host array for reference
    // printf("Original fftwf_complex host array:\n");
    // for (int i = 0; i < num_elements; i++)
    // {
    //     printf("Host Element %d: (%f, %f)\n", i, box[i][0], box[i][1]);
    // }
    // printf("The total number of elements: %d\n", total_elements);

    // Cast fftwf_complex to cufftComplex
    cufftComplex *h_cu_box = reinterpret_cast<cufftComplex *>(box);
    // Allocate device memory for cufftComplex array
    cufftComplex *d_cu_box;
    cudaMalloc((void **)&d_cu_box, sizeof(cufftComplex) * total_elements);

    // Copy the cuComplex array from host to device
    cudaMemcpy(d_cu_box, h_cu_box, sizeof(cufftComplex) * total_elements, cudaMemcpyHostToDevice);

    // Define threads layout. 
    int block_x = (z_dim + 3)/4;
    int block_y = (xy_dim +7)/8;
    int block_z = (xy_dim + 7)/8;
    dim3 blockGrid(block_x, block_y, block_z);
    dim3 threadsPerBlock(4,8,8);

    // pass the following macros as values
    double delta_k = DELTA_K;
    double delta_k_para = DELTA_K_PARA;

    // Launch the kernel to print the first few elements
    printComplexArray<<<blockGrid, threadsPerBlock>>>(d_cu_box, num_elements, z_dim, xy_dim, xy_dim, midpoint, RES, filter_type, R, R_param, delta_k, delta_k_para);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // copy the data from device to host
    cudaMemcpy(h_cu_box, d_cu_box, sizeof(cufftComplex) * total_elements, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_cu_box);

    // Free host memory
    // fftwf_free(box);

    return 0;
}
