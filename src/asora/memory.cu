#include "memory.cuh"
#include <iostream>

// ========================================================================
// Global variables. Pointers to GPU memory to store grid data
//
// To avoid uneccessary memory movement between host and device, we
// allocate dedicated memory on the device via a call to device_init at the
// beginning of the program. Data is copied to and from the host memory
// (typically numpy arrays) only when it changes and is required. For example:
//
// * The density field is copied to the device only when it
// actually changes, i.e. at the beginning of a timestep.
// * The photoionization rates for each source are computed and summed
// directly on the device and are copied to the host only when all sources
// have been passed.
// * The column density is NEVER copied back to the host, since it is only
// accessed on the device when computing ionization rates.
// ========================================================================
double* cdh_dev;                 // Outgoing column density of the cells
double* n_dev;                   // Density
double* x_dev;                   // Time-averaged ionized fraction
double* phi_dev;                 // Photoionization rates
double* photo_thin_table_dev;    // Thin Radiation table
double* photo_thick_table_dev;   // Thick Radiation table
int * src_pos_dev;
double * src_flux_dev;

int NUM_SRC_PAR;

// ========================================================================
// Initialization function to allocate device memory (pointers above)
// ========================================================================
void device_init(const int & N, const int & num_src_par)
{
    int dev_id = 0;

    cudaDeviceProp device_prop;
    cudaGetDevice(&dev_id);
    cudaGetDeviceProperties(&device_prop, dev_id);
    if (device_prop.computeMode == cudaComputeModeProhibited) {
        std::cerr << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice()" << std::endl;
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "cudaGetDeviceProperties returned error code " << error << ", line(" << __LINE__ << ")" << std::endl;
    } else {
        std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name << "\" with compute capability " << device_prop.major << "." << device_prop.minor << std::endl;
    }

    // Byte-size of grid data
    long unsigned int bytesize = N*N*N*sizeof(double);
    //std::cout << bytesize << std::endl;

    // Set the source batch size, i.e. the number of sources done in parallel (on the same GPU)
    NUM_SRC_PAR = num_src_par;

    // Allocate memory
    cudaMalloc(&cdh_dev, NUM_SRC_PAR * bytesize);
    cudaMalloc(&n_dev, bytesize);
    cudaMalloc(&x_dev, bytesize);
    cudaMalloc(&phi_dev, bytesize);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Couldn't allocate memory: " + std::to_string((3 + NUM_SRC_PAR)*bytesize/1e6) + std::string(cudaGetErrorName(error)) + " - " + std::string(cudaGetErrorString(error)));
        }    
    else {
        std::cout << "Successfully allocated " << (3 + NUM_SRC_PAR)*bytesize/1e6 << " Mb of device memory for grid of size N = " << N;
        std::cout << ", with source batch size " << NUM_SRC_PAR << std::endl;
    }
}

// ========================================================================
// Utility functions to copy data to device
// ========================================================================
void density_to_device(double* ndens,const int & N)
{
    cudaMemcpy(n_dev,ndens,N*N*N*sizeof(double),cudaMemcpyHostToDevice);
}

void photo_table_to_device(double* thin_table,double* thick_table,const int & NumTau)
{
    // Copy thin table
    cudaMalloc(&photo_thin_table_dev,NumTau*sizeof(double));
    cudaMemcpy(photo_thin_table_dev,thin_table,NumTau*sizeof(double),cudaMemcpyHostToDevice);
    // Copy thick table
    cudaMalloc(&photo_thick_table_dev,NumTau*sizeof(double));
    cudaMemcpy(photo_thick_table_dev,thick_table,NumTau*sizeof(double),cudaMemcpyHostToDevice);
}
void source_data_to_device(int* pos, double* flux, const int & NumSrc)
{   
    // Free arrays from previous evolve call
    cudaFree(src_pos_dev);
    cudaFree(src_flux_dev);

    // Allocate memory for sources of current evolve call
    cudaMalloc(&src_pos_dev,3*NumSrc*sizeof(int));
    cudaMalloc(&src_flux_dev,NumSrc*sizeof(double));

    // Copy source data (positions & strengths) to device
    cudaMemcpy(src_pos_dev,pos,3*NumSrc*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(src_flux_dev,flux,NumSrc*sizeof(double),cudaMemcpyHostToDevice);
}

// ========================================================================
// Deallocate device memory at the end of a run
// ========================================================================
void device_close()
{   
    printf("Deallocating device memory...\n");
    cudaFree(cdh_dev);
    cudaFree(n_dev);
    cudaFree(x_dev);
    cudaFree(phi_dev);
    cudaFree(photo_thin_table_dev);
    cudaFree(src_pos_dev);
    cudaFree(src_flux_dev);
}
