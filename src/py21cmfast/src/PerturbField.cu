// Re-write of perturb_field.c for being accessible within the MCMC
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <fftw3.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"
#include "Constants.h"
#include "indexing.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cosmology.h"
#include "dft.h"
#include "debugging.h"
#include "filtering.h"

#include "PerturbField.h"

void compute_perturbed_velocities(
    unsigned short axis,
    UserParams *user_params,
    fftwf_complex *HIRES_density_perturb,
    fftwf_complex *HIRES_density_perturb_saved,
    fftwf_complex *LOWRES_density_perturb,
    fftwf_complex *LOWRES_density_perturb_saved,
    float dDdt_over_D,
    int dimension,
    int switch_mid,
    float f_pixel_factor,
    float *velocity
){

    float k_x, k_y, k_z, k_sq;
    unsigned long long int n_x, n_y, n_z;
    unsigned long long int i,j,k;

    // ALICE: 3D vector for k-space coords
    float kvec[3];

    if(user_params->PERTURB_ON_HIGH_RES) {
        // We are going to generate the velocity field on the high-resolution perturbed
        // density grid
        // ALICE: Copy the saved k-space density field to HIRES_density_perturb.
        memcpy(
            HIRES_density_perturb,
            HIRES_density_perturb_saved,
            sizeof(fftwf_complex)*KSPACE_NUM_PIXELS
        );
    }
    else {
        // We are going to generate the velocity field on the low-resolution perturbed density grid
        // ALICE: Copy the saved k-space density field to LOWRES_density_perturb.
        memcpy(
            LOWRES_density_perturb,
            LOWRES_density_perturb_saved,
            sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS
        );
        LOG_SUPER_DEBUG("dDdt_over_D=%.6e, dimension=%d, switch_mid=%d, f_pixel_factor=%f", dDdt_over_D, dimension, switch_mid, f_pixel_factor);
    }

    // ALICE: Compute wave numbers (k_x, k_y, k_z) + compute velocity based on density perturbations.
    // ALICE: Wave numbers == frequencies of spatial oscillations (higher wave number=faster oscillations)
    #pragma omp parallel \
        shared(LOWRES_density_perturb,HIRES_density_perturb,dDdt_over_D,dimension,switch_mid) \
        private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq, kvec) \
        num_threads(user_params->N_THREADS)
    {
        #pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
            if (n_x > switch_mid)
                k_x = (n_x-dimension) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y=0; n_y<dimension; n_y++){
                if (n_y > switch_mid)
                    k_y = (n_y-dimension) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                for (n_z=0; n_z<=(unsigned long long)(user_params->NON_CUBIC_FACTOR*switch_mid); n_z++){
                    k_z = n_z * DELTA_K_PARA;

                    kvec[0] = k_x;
                    kvec[1] = k_y;
                    kvec[2] = k_z;

                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                    // now set the velocities
                    if ((n_x==0) && (n_y==0) && (n_z==0)) { // DC mode
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            HIRES_density_perturb[0] = 0;
                        }
                        else {
                            LOWRES_density_perturb[0] = 0;
                        }
                    }
                    else{
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            HIRES_density_perturb[C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*kvec[axis]*I/k_sq/(TOT_NUM_PIXELS+0.0);
                        }
                        else {
                            LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*kvec[axis]*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                        }
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("density_perturb after modification by dDdt: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    // ALICE: density field was already in k-space when passed in, so now filter (top-hat), inverse fft and copy to velocity field.
    if(user_params->PERTURB_ON_HIGH_RES) {

        // smooth the high resolution field ready for resampling
        // ALICE: RES=0 (dimension=DIM, midpoint=MIDDLE), filter_type=0 (real space top-hat filtering)
        if (user_params->DIM != user_params->HII_DIM)
            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));

        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, HIRES_density_perturb);

        // ALICE: Copy computed velocities to velocity field.
        #pragma omp parallel \
            shared(velocity,HIRES_density_perturb,f_pixel_factor) \
            private(i,j,k) \
            num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        *((float *)velocity + HII_R_INDEX(i,j,k)) = *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5), (unsigned long long)(j*f_pixel_factor+0.5), (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                }
            }
        }
    }
    // ALICE: LOWRES -> no top hat filtering, just inverse fft and copy to velocity field.
    else {
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, LOWRES_density_perturb);

        #pragma omp parallel \
            shared(velocity,LOWRES_density_perturb) \
            private(i,j,k) \
            num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        *((float *)velocity + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                    }
                }
            }
        }
    }
    LOG_SUPER_DEBUG("velocity: ");
    debugSummarizeBox(velocity, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

}

// ----------------------------------------------------------------------------------------------------------------------------

// #define R_INDEX(x,y,z)((unsigned long long)((z)+D_PARA*((y)+D*(x))))
__device__ inline double compute_R_INDEX(int i, int j, int k, int DIM, int D_PARA) {
    return k + D_PARA * (j + DIM * i)
}

// #define HII_R_INDEX(x,y,z)((unsigned long long)((z)+HII_D_PARA*((y)+HII_D*(x))))
__device__ inline double compute_HII_R_INDEX(int i, int j, int k, int HII_D_PARA, int HII_D) {
    return k + HII_D_PARA * (j + HII_D * i)
}

__global__ void perturb_density_field_kernel(
    double *resampled_box,
    const float* __restrict__ hires_density, // Is const needed as well as __restrict__?
    const float* __restrict__ hires_vx,
    const float* __restrict__ hires_vy,
    const float* __restrict__ hires_vz,
    const float* __restrict__ lowres_vx,
    const float* __restrict__ lowres_vy,
    const float* __restrict__ lowres_vz,
    const float* __restrict__ hires_vx_2LPT,
    const float* __restrict__ hires_vy_2LPT,
    const float* __restrict__ hires_vz_2LPT,
    const float* __restrict__ lowres_vx_2LPT,
    const float* __restrict__ lowres_vy_2LPT,
    const float* __restrict__ lowres_vz_2LPT,
    int dimension, int DIM, int D_PARA, int MID_PARA,
    int NON_CUBIC_FACTOR, float f_pixel_factor, float init_growth_factor,
    bool PERTURB_ON_HIGH_RES, bool USE_2LPT,
    ) {

    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get index of density cell
    int i = idx / (D_PARA * DIM);
    int j = (idx / D_PARA) % DIM;
    int k = idx % D_PARA;

    int r_index = compute_R_INDEX(i, j, k);
    
    // Map index to location in units of box size
    float xf = (i + 0.5) / DIM;
    float yf = (j + 0.5) / DIM;
    float zf = (k + 0.5) / D_PARA;

    // Update locations
    if (PERTURB_ON_HIGH_RES) {
        xf += __ldg(&hires_vx[r_index]);
        yf += __ldg(&hires_vy[r_index]);
        zf += __ldg(&hires_vz[r_index]);
    }
    else {
        unsigned long long HII_i = (unsigned long long)(i / f_pixel_factor);
        unsigned long long HII_j = (unsigned long long)(j / f_pixel_factor);
        unsigned long long HII_k = (unsigned long long)(k / f_pixel_factor);
        int HII_index = compute_HII_R_INDEX(HII_i, HII_j, HII_k, HII_D_PARA, HII_D);
        xf += __ldg(&lowres_vx[HII_index]);
        yf += __ldg(&lowres_vy[HII_index]);
        zf += __ldg(&lowres_vz[HII_index]);
    }

    // 2LPT (add second order corrections)
    if (USE_2LPT) {
        if (PERTURB_ON_HIGH_RES) {
            xf -= __ldg(&hires_vx_2LPT[r_index]);
            yf -= __ldg(&hires_vy_2LP[r_index]);
            zf -= __ldg(&hires_vz_2LPT[r_index]);
        }
        else {
            xf -= __ldg(&lowres_vx_2LPT[HII_index]);
            yf -= __ldg(&lowres_vy_2LPT[HII_index]);
            zf -= __ldg(&lowres_vz_2LPT[HII_index]);
        }
    }

    // Scale coordinates back to grid size
    xf *= (double)(dimension);
    yf *= (double)(dimension);
    zf *= (double)((unsigned long long)(NON_CUBIC_FACTOR * dimension));

    // Wrap coordinates to keep them within valid boundaries
    xf = fmod(fmod(xf, dimension) + dimension, dimension);
    yf = fmod(fmod(yf, dimension) + dimension, dimension);
    zf = fmod(fmod(zf, dimension * NCF) + dimension * NCF, dimension * NCF);

    // Get integer values for indices from floating point values
    int xi = xf;
    int yi = yf;
    int zi = zf;

    // Wrap index coordinates to ensure no out-of-bounds array access will be attempted
    xi = (xi % dimension + dimension) % dimension;
    yi = (yi % dimension + dimension) % dimension;
    zi = (zi % dimension * NCF + dimension * NCF) % dimension * NCF;

    // Determine the fraction of the perturbed cell which overlaps with the 8 nearest grid cells,
    // based on the grid cell which contains the centre of the perturbed cell
    float d_x = fabs(xf - (double)(xi + 0.5)); // Absolute distances from grid cell centre to perturbed cell centre
    float d_y = fabs(yf - (double)(yi + 0.5)); // (also) The fractions of mass which will be moved to neighbouring cells
    float d_z = fabs(zf - (double)(zi + 0.5));

    // 8 neighbour cells-of-interest will be shifted left/down/behind if perturbed midpoint is in left/bottom/back corner of cell.
    if (xf < (double)(xi + 0.5)) {
        // If perturbed cell centre is less than the mid-point then update fraction
        // of mass in the cell and determine the cell centre of neighbour to be the
        // lowest grid point index
        d_x = 1. - d_x;
        xi -= 1;
        xi += (xi + dimension) % dimension; // Only this critera is possible as iterate back by one (we cannot exceed DIM)
    }
    if(yf < (double)(yi + 0.5)) {
        d_y = 1. - d_y;
        yi -= 1;
        yi += (yi + dimension) % dimension;
    }
    if(zf < (double)(zi + 0.5)) {
        d_z = 1. - d_z;
        zi -= 1;
        zi += (zi + (unsigned long long)(NON_CUBIC_FACTOR * dimension)) % (unsigned long long)(NON_CUBIC_FACTOR * dimension);
    }
    // The fractions of mass which will remain with perturbed cell
    float t_x = 1. - d_x;
    float t_y = 1. - d_y;
    float t_z = 1. - d_z;

    // Determine the grid coordinates of the 8 neighbouring cells.
    // Neighbours will be in positive direction; front/right/above cells (-> 2x2 cube, with perturbed cell bottom/left/back)
    // Takes into account the offset based on cell centre determined above
    int xp1 = (xi + 1) % dimension;
    int yp1 = (yi + 1) % dimension;
    int zp1 = (zi + 1) % (unsigned long long)(NON_CUBIC_FACTOR * dimension);

    double scaled_density = 1 + init_growth_factor * __ldg(&hires_density[r_index]);

    if (PERTURB_ON_HIGH_RES) {
        // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
        // Cell mass = (1 + init_growth_factor * orig_density) * (proportion of mass to distribute)
        atomicAdd(&resampled_box[compute_R_INDEX(xi, yi, zi)], scaled_density * t_x * t_y * t_z);
        atomicAdd(&resampled_box[compute_R_INDEX(xp1, yi, zi)], scaled_density * d_x * t_y * t_z);
        atomicAdd(&resampled_box[compute_R_INDEX(xi, yp1, zi)], scaled_density * t_x * d_y * t_z);
        atomicAdd(&resampled_box[compute_R_INDEX(xp1, yp1, zi)], scaled_density * d_x * d_y * t_z);
        atomicAdd(&resampled_box[compute_R_INDEX(xi, yi, zp1)], scaled_density * t_x * t_y * d_z);
        atomicAdd(&resampled_box[compute_R_INDEX(xp1, yi, zp1)], scaled_density * d_x * t_y * d_z);
        atomicAdd(&resampled_box[compute_R_INDEX(xi, yp1, zp1)], scaled_density * t_x * d_y * d_z);
        atomicAdd(&resampled_box[compute_R_INDEX(xp1, yp1, zp1)], scaled_density * d_x * d_y * d_z);
    }
    else {
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yi, zi)], scaled_density * t_x * t_y * t_z);
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yi, zi)], scaled_density * d_x * t_y * t_z);
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yp1, zi)], scaled_density * t_x * d_y * t_z);
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yp1, zi)], scaled_density * d_x * d_y * t_z);
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yi, zp1)], scaled_density * t_x * t_y * d_z);
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yi, zp1)], scaled_density * d_x * t_y * d_z);
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yp1, zp1)], scaled_density * t_x * d_y * d_z);
        atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yp1, zp1)], scaled_density * d_x * d_y * d_z);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

int ComputePerturbField_gpu(
    float redshift, UserParams *user_params, CosmoParams *cosmo_params,
    InitialConditions *boxes, PerturbedField *perturbed_field
) {
    /*
     ComputePerturbField uses the first-order Langragian displacement field to move the
     masses in the cells of the density field. The high-res density field is extrapolated
     to some high-redshift (global_params.INITIAL_REDSHIFT), then uses the zeldovich
     approximation to move the grid "particles" onto the lower-res grid we use for the
     maps. Then we recalculate the velocity fields on the perturbed grid.
    */

    int status;
    Try{  // This Try{} wraps the whole function, so we don't indent.

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_noastro(user_params,cosmo_params);

    omp_set_num_threads(user_params->N_THREADS);

    fftwf_complex *HIRES_density_perturb, *HIRES_density_perturb_saved;
    fftwf_complex *LOWRES_density_perturb, *LOWRES_density_perturb_saved;

    float growth_factor, displacement_factor_2LPT, init_growth_factor, init_displacement_factor_2LPT;
    float mass_factor, dDdt, f_pixel_factor, velocity_displacement_factor, velocity_displacement_factor_2LPT;
    int i, j, k, dimension, switch_mid;

    // Function for deciding the dimensions of loops when we could
    // use either the low or high resolution grids.
    switch(user_params->PERTURB_ON_HIGH_RES) {
        case 0:
            dimension = user_params->HII_DIM;
            switch_mid = HII_MIDDLE;
            break;
        case 1:
            dimension = user_params->DIM;
            switch_mid = MIDDLE;
            break;
    }

    // ***************   BEGIN INITIALIZATION   ************************** //

    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        LOG_WARNING("Resolution is likely too low for accurate evolved density fields\n \
                It is recommended that you either increase the resolution (DIM/BOX_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }

    growth_factor = dicke(redshift);
    displacement_factor_2LPT = -(3.0/7.0) * growth_factor*growth_factor; // 2LPT eq. D8

    dDdt = ddickedt(redshift); // time derivative of the growth factor (1/s)
    init_growth_factor = dicke(global_params.INITIAL_REDSHIFT);
    init_displacement_factor_2LPT = -(3.0/7.0) * init_growth_factor*init_growth_factor; // 2LPT eq. D8

    // find factor of HII pixel size / deltax pixel size
    f_pixel_factor = user_params->DIM/(float)(user_params->HII_DIM);
    mass_factor = pow(f_pixel_factor, 3);

    // allocate memory for the updated density, and initialize
    LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    if(user_params->PERTURB_ON_HIGH_RES) {
        HIRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        HIRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    }

    double *resampled_box;

    //TODO: debugSummarizeIC is bugged when not all the fields are in memory
    // debugSummarizeIC(boxes, user_params->HII_DIM, user_params->DIM, user_params->NON_CUBIC_FACTOR);
    LOG_SUPER_DEBUG("growth_factor=%f, displacemet_factor_2LPT=%f, dDdt=%f, init_growth_factor=%f, init_displacement_factor_2LPT=%f, mass_factor=%f",
                    growth_factor, displacement_factor_2LPT, dDdt, init_growth_factor, init_displacement_factor_2LPT, mass_factor);

    // check if the linear evolution flag was set
    if (global_params.EVOLVE_DENSITY_LINEARLY){

        LOG_DEBUG("Linearly evolve density field");

#pragma omp parallel shared(growth_factor,boxes,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = growth_factor*boxes->hires_density[R_INDEX(i,j,k)];
                        }
                        else {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = growth_factor*boxes->lowres_density[HII_R_INDEX(i,j,k)];
                        }
                    }
                }
            }
        }
    }
    else {
        // Apply Zel'dovich/2LPT correction
        LOG_DEBUG("Apply Zel'dovich");

#pragma omp parallel shared(LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = 0.;
                        }
                        else {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                        }

                    }
                }
            }
        }

        velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;

        // now add the missing factor of D
#pragma omp parallel shared(boxes,velocity_displacement_factor,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->hires_vy[R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->hires_vz[R_INDEX(i,j,k)] *= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                        }
                        else {
                            boxes->lowres_vx[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->lowres_vy[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->lowres_vz[HII_R_INDEX(i,j,k)] *= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                        }
                    }
                }
            }
        }

        // * ************************************************************************* * //
        // *                           BEGIN 2LPT PART                                 * //
        // * ************************************************************************* * //
        // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        if(user_params->USE_2LPT){
            LOG_DEBUG("Apply 2LPT");

            // allocate memory for the velocity boxes and read them in
            velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;

            // now add the missing factor in eq. D9
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->hires_vy_2LPT[R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->hires_vz_2LPT[R_INDEX(i,j,k)] *= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                            }
                            else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] *= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                            }
                        }
                    }
                }
            }
        }


        // * ************************************************************************* * //
        // *                            END 2LPT PART                                  * //
        // * ************************************************************************* * //

        // ************  END INITIALIZATION **************************** //


        // ----------------------------------------------------------------------------------------------------------------------------

        // Box shapes from outputs.py and convenience macros
        if(user_params->PERTURB_ON_HIGH_RES) {
            int num_pixels = TOT_NUM_PIXELS;
            size_t size = TOT_NUM_PIXELS * sizeof(double);
        }
        else {
            int num_pixels = HII_TOT_NUM_PIXELS;
            size_t size = HII_TOT_NUM_PIXELS * sizeof(double);
        }

        // Allocate device memory for output box
        double* d_box;
        cudaMalloc(&d_box, size);
        cudaMemset(d_box, 0, sizeof(double) * size);

        // Allocate device memory for density field
        float* hires_density;
        cudaMalloc(&hires_density);
        cudaMemcpy(hires_density, boxes->hires_density, size, cudaMemcpyHostToDevice);

        // Allocate device memory and copy arrays to device as per user_params
        if (user_params->PERTURB_ON_HIGH_RES) {
            float* hires_vx;
            float* hires_vy;
            float* hires_vz;
            cudaMalloc(&hires_vx);
            cudaMalloc(&hires_vy);
            cudaMalloc(&hires_vz);
            cudaMemcpy(hires_vx, boxes->hires_vx, size, cudaMemcpyHostToDevice);
            cudaMemcpy(hires_vy, boxes->hires_vy, size, cudaMemcpyHostToDevice);
            cudaMemcpy(hires_vz, boxes->hires_vz, size, cudaMemcpyHostToDevice);
        }
        else {
            float* lowres_vx;
            float* lowres_vy;
            float* lowres_vz;
            cudaMalloc(&lowres_vx);
            cudaMalloc(&lowres_vy);
            cudaMalloc(&lowres_vz);
            cudaMemcpy(lowres_vx, boxes->lowres_vx, size, cudaMemcpyHostToDevice);
            cudaMemcpy(lowres_vy, boxes->lowres_vy, size, cudaMemcpyHostToDevice);
            cudaMemcpy(lowres_vz, boxes->lowres_vz, size, cudaMemcpyHostToDevice);
        }
        if (user_params->USE_2LPT) {
            if (user_params->PERTURB_ON_HIGH_RES) {
                float* hires_vx_2LPT;
                float* hires_vy_2LPT;
                float* hires_vz_2LPT;
                cudaMalloc(&hires_vx_2LPT);
                cudaMalloc(&hires_vy_2LPT);
                cudaMalloc(&hires_vz_2LPT);
                cudaMemcpy(hires_vx_2LPT, boxes->hires_vx_2LPT, size, cudaMemcpyHostToDevice);
                cudaMemcpy(hires_vy_2LPT, boxes->hires_vy_2LPT, size, cudaMemcpyHostToDevice);
                cudaMemcpy(hires_vz_2LPT, boxes->hires_vz_2LPT, size, cudaMemcpyHostToDevice);
            }
            else {
                float* lowres_vx_2LPT;
                float* lowres_vy_2LPT;
                float* lowres_vz_2LPT;
                cudaMalloc(&lowres_vx_2LPT);
                cudaMalloc(&lowres_vy_2LPT);
                cudaMalloc(&lowres_vz_2LPT);
                cudaMemcpy(lowres_vx_2LPT, boxes->lowres_vx_2LPT, size, cudaMemcpyHostToDevice);
                cudaMemcpy(lowres_vy_2LPT, boxes->lowres_vy_2LPT, size, cudaMemcpyHostToDevice);
                cudaMemcpy(lowres_vz_2LPT, boxes->lowres_vz_2LPT, size, cudaMemcpyHostToDevice);
            }
        }

        // Invoke kernel
        int threadsPerBlock = 256;
        int numBlocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;
        perturb_density_field_kernel<<<numBlocks, threadsPerBlock>>>(
            d_box, dimension, user_params->DIM, D_PARA, MID_PARA, user_params->NON_CUBIC_FACTOR,
            f_pixel_factor, init_growth_factor, user_params->PERTURB_ON_HIGH_RES, user_params->USE_2LPT,
        );

        // Only use during development!
        cudaError_t err = cudaDeviceSynchronize();
        CATCH_CUDA_ERROR(err);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("Kernel launch error: %s", cudaGetErrorString(err));
            Throw(CudaError); // Or the appropriate exception type
        }

        // Copy results from device to host
        double *resampled_box;
        // resampled_box = (double *)calloc(num_pixels, sizeof(double)); // is this needed?
        cudaMemcpy(resampled_box, d_box, size, cudaMemcpyDeviceToHost);

        // Deallocate device memory
        cudaFree(d_box);
        cudaFree(hires_density);

        if (user_params->PERTURB_ON_HIGH_RES) {
            cudaFree(hires_vx);
            cudaFree(hires_vy);
            cudaFree(hires_vz);
        }
        else {
            cudaFree(lowres_vx);
            cudaFree(lowres_vy);
            cudaFree(lowres_vz);
        }
        if (user_params->USE_2LPT) {
            if (user_params->PERTURB_ON_HIGH_RES) {
                cudaFree(hires_vx_2LPT);
                cudaFree(hires_vy_2LPT);
                cudaFree(hires_vz_2LPT);
            }
            else {
                cudaFree(lowres_vx_2LPT);
                cudaFree(lowres_vy_2LPT);
                cudaFree(lowres_vz_2LPT);
            }
        }

        // ----------------------------------------------------------------------------------------------------------------------------

        LOG_SUPER_DEBUG("resampled_box: ");
        debugSummarizeBoxDouble(resampled_box, dimension, user_params->NON_CUBIC_FACTOR, "  ");

        // Resample back to a float for remaining algorithm
        #pragma omp parallel \
            shared(LOWRES_density_perturb,HIRES_density_perturb,resampled_box,dimension) \
            private(i,j,k) \
            num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *( (float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) = (float)resampled_box[R_INDEX(i,j,k)];
                        }
                        else {
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) = (float)resampled_box[HII_R_INDEX(i,j,k)];
                        }
                    }
                }
            }
        }
        free(resampled_box);
        LOG_DEBUG("Finished perturbing the density field");

        LOG_SUPER_DEBUG("density_perturb: ");
        if(user_params->PERTURB_ON_HIGH_RES){
            debugSummarizeBoxComplex(HIRES_density_perturb, dimension, user_params->NON_CUBIC_FACTOR, "  ");
        }else{
            debugSummarizeBoxComplex(LOWRES_density_perturb, dimension, user_params->NON_CUBIC_FACTOR, "  ");
        }

        // deallocate
#pragma omp parallel shared(boxes,velocity_displacement_factor,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->hires_vy[R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->hires_vz[R_INDEX(i,j,k)] /= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                        }
                        else {
                            boxes->lowres_vx[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->lowres_vy[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->lowres_vz[HII_R_INDEX(i,j,k)] /= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                        }
                    }
                }
            }
        }

        if(user_params->USE_2LPT){
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->hires_vy_2LPT[R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->hires_vz_2LPT[R_INDEX(i,j,k)] /= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                            }
                            else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] /= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                            }
                        }
                    }
                }
            }
        }
        LOG_DEBUG("Cleanup velocities for perturb");
    }

    // Now, if I still have the high resolution density grid (HIRES_density_perturb) I need to downsample it to the low-resolution grid
    if(user_params->PERTURB_ON_HIGH_RES) {

        LOG_DEBUG("Downsample the high-res perturbed density");

        // Transform to Fourier space to sample (filter) the box
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, HIRES_density_perturb);

        // Need to save a copy of the high-resolution unfiltered density field for the velocities
        memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // Now filter the box
        // ALICE: RES=0 (dimension=DIM, midpoint=MIDDLE), filter_type=0 (real space top-hat filtering)
        if (user_params->DIM != user_params->HII_DIM) {
            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
        }

        // FFT back to real space
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, HIRES_density_perturb);

        // Renormalise the FFT'd box
#pragma omp parallel shared(HIRES_density_perturb,LOWRES_density_perturb,f_pixel_factor,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        // ALICE: Get corresponding high res indices, and normalise by total pixels.
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) =
                        *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)))/(float)TOT_NUM_PIXELS;

                        // ALICE: Add an offset.
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) -= 1.;

                        // ALICE: if less than -1, set slightly above -1.
                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1.+FRACT_FLOAT_ERR;
                        }
                    }
                }
            }
        }
    }
    else {

        if (!global_params.EVOLVE_DENSITY_LINEARLY){

#pragma omp parallel shared(LOWRES_density_perturb,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<HII_D_PARA; k++){
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1.;
                        }
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("LOWRES_density_perturb: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    // transform to k-space
    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, LOWRES_density_perturb);

    // smooth the field
    // ALICE: RES=1 (dimension=HII_DIM, midpoint=HII_MIDDLE), filter_type=2 (Gaussian filtering)
    if (!global_params.EVOLVE_DENSITY_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
        filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
    }

    LOG_SUPER_DEBUG("LOWRES_density_perturb after smoothing: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    // save a copy of the k-space density field
    memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, LOWRES_density_perturb);

    LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    // normalize after FFT
    // ALICE: divide by total pixels; if result < -1 changed it to just above -1.
    int bad_count=0;
#pragma omp parallel shared(LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS) reduction(+: bad_count)
    {
#pragma omp for
        for(i=0; i<user_params->HII_DIM; i++){
            for(j=0; j<user_params->HII_DIM; j++){
                for(k=0; k<HII_D_PARA; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;

                    if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1.0) { // shouldn't happen

                        if(bad_count<5) LOG_WARNING("LOWRES_density_perturb is <-1 for index %d %d %d (value=%f)", i,j,k, *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)));
                        if(bad_count==5) LOG_WARNING("Skipping further warnings for LOWRES_density_perturb.");
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                        bad_count++;
                    }
                }
            }
        }
    }
    if(bad_count>=5) LOG_WARNING("Total number of bad indices for LOW_density_perturb: %d", bad_count);
    LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space (normalized): ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

// ALICE: copy LOWRES_density_perturb cell values to density cells
#pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<HII_D_PARA; k++){
                    *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                }
            }
        }
    }

    // ****  Convert to velocities ***** //
    LOG_DEBUG("Generate velocity fields");

    float dDdt_over_D;

    dDdt_over_D = dDdt/growth_factor;


    if (user_params->KEEP_3D_VELOCITIES){
        compute_perturbed_velocities(
            0,
            user_params,
            HIRES_density_perturb,
            HIRES_density_perturb_saved,
            LOWRES_density_perturb,
            LOWRES_density_perturb_saved,
            dDdt_over_D,
            dimension,
            switch_mid,
            f_pixel_factor,
            perturbed_field->velocity_x
        );
        compute_perturbed_velocities(
            1,
            user_params,
            HIRES_density_perturb,
            HIRES_density_perturb_saved,
            LOWRES_density_perturb,
            LOWRES_density_perturb_saved,
            dDdt_over_D,
            dimension,
            switch_mid,
            f_pixel_factor,
            perturbed_field->velocity_y
        );
    }

    compute_perturbed_velocities(
        2,
        user_params,
        HIRES_density_perturb,
        HIRES_density_perturb_saved,
        LOWRES_density_perturb,
        LOWRES_density_perturb_saved,
        dDdt_over_D,
        dimension,
        switch_mid,
        f_pixel_factor,
        perturbed_field->velocity_z
    );

    fftwf_cleanup_threads();
    fftwf_cleanup();
    fftwf_forget_wisdom();

    // deallocate
    fftwf_free(LOWRES_density_perturb);
    fftwf_free(LOWRES_density_perturb_saved);
    if(user_params->PERTURB_ON_HIGH_RES) {
        fftwf_free(HIRES_density_perturb);
        fftwf_free(HIRES_density_perturb_saved);
    }
    fftwf_cleanup();

    } // End of Try{}
    Catch(status){
        return(status);
    }

    return(0);
}
