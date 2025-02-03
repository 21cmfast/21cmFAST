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

double *MapMass_cpu(
    UserParams *user_params,
    CosmoParams *cosmo_params,
    InitialConditions *boxes,
    double *resampled_box,
    int dimension,
    float f_pixel_factor,
    float init_growth_factor
){
    #pragma omp parallel \
        shared(init_growth_factor,boxes,f_pixel_factor,resampled_box,dimension) \
        private(i,j,k,xi,xf,yi,yf,zi,zf,HII_i,HII_j,HII_k,d_x,d_y,d_z,t_x,t_y,t_z,xp1,yp1,zp1) \
        num_threads(user_params->N_THREADS)
    {
        #pragma omp for
        unsigned long long int i,j,k;
        for (i=0; i<user_params->DIM;i++){
            for (j=0; j<user_params->DIM;j++){
                for (k=0; k<D_PARA;k++){

                    // map indeces to locations in units of box size
                    xf = (i+0.5)/((user_params->DIM)+0.0);
                    yf = (j+0.5)/((user_params->DIM)+0.0);
                    zf = (k+0.5)/((D_PARA)+0.0);

                    // update locations
                    if(user_params->PERTURB_ON_HIGH_RES) {
                        xf += (boxes->hires_vx)[R_INDEX(i, j, k)];
                        yf += (boxes->hires_vy)[R_INDEX(i, j, k)];
                        zf += (boxes->hires_vz)[R_INDEX(i, j, k)];
                    }
                    else {
                        HII_i = (unsigned long long)(i/f_pixel_factor);
                        HII_j = (unsigned long long)(j/f_pixel_factor);
                        HII_k = (unsigned long long)(k/f_pixel_factor);
                        xf += (boxes->lowres_vx)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                        yf += (boxes->lowres_vy)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                        zf += (boxes->lowres_vz)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                    }

                    // 2LPT PART
                    // add second order corrections
                    if(user_params->USE_2LPT){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            xf -= (boxes->hires_vx_2LPT)[R_INDEX(i,j,k)];
                            yf -= (boxes->hires_vy_2LPT)[R_INDEX(i,j,k)];
                            zf -= (boxes->hires_vz_2LPT)[R_INDEX(i,j,k)];
                        }
                        else {
                            xf -= (boxes->lowres_vx_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                            yf -= (boxes->lowres_vy_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                            zf -= (boxes->lowres_vz_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                        }
                    }
                    xf *= (double)(dimension);
                    yf *= (double)(dimension);
                    zf *= (double)((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));
                    while (xf >= (double)(dimension)){ xf -= (dimension);}
                    while (xf < 0){ xf += (dimension);}
                    while (yf >= (double)(dimension)){ yf -= (dimension);}
                    while (yf < 0){ yf += (dimension);}
                    while (zf >= (double)(user_params->NON_CUBIC_FACTOR*dimension)){ zf -= (user_params->NON_CUBIC_FACTOR*dimension);}
                    while (zf < 0){ zf += (user_params->NON_CUBIC_FACTOR*dimension);}
                    xi = xf;
                    yi = yf;
                    zi = zf;
                    if (xi >= (dimension)){ xi -= (dimension);}
                    if (xi < 0) {xi += (dimension);}
                    if (yi >= (dimension)){ yi -= (dimension);}
                    if (yi < 0) {yi += (dimension);}
                    if (zi >= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension))){ zi -= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}
                    if (zi < 0) {zi += ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}

                    // Determine the fraction of the perturbed cell which overlaps with the 8 nearest grid cells,
                    // based on the grid cell which contains the centre of the perturbed cell
                    d_x = fabs(xf - (double)(xi+0.5));
                    d_y = fabs(yf - (double)(yi+0.5));
                    d_z = fabs(zf - (double)(zi+0.5));
                    if(xf < (double)(xi+0.5)) {
                        // If perturbed cell centre is less than the mid-point then update fraction
                        // of mass in the cell and determine the cell centre of neighbour to be the
                        // lowest grid point index
                        d_x = 1. - d_x;
                        xi -= 1;
                        if (xi < 0) {xi += (dimension);} // Only this critera is possible as iterate back by one (we cannot exceed DIM)
                    }
                    if(yf < (double)(yi+0.5)) {
                        d_y = 1. - d_y;
                        yi -= 1;
                        if (yi < 0) {yi += (dimension);}
                    }
                    if(zf < (double)(zi+0.5)) {
                        d_z = 1. - d_z;
                        zi -= 1;
                        if (zi < 0) {zi += ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}
                    }
                    t_x = 1. - d_x;
                    t_y = 1. - d_y;
                    t_z = 1. - d_z;

                    // Determine the grid coordinates of the 8 neighbouring cells
                    // Takes into account the offset based on cell centre determined above
                    xp1 = xi + 1;
                    if(xp1 >= dimension) { xp1 -= (dimension);}
                    yp1 = yi + 1;
                    if(yp1 >= dimension) { yp1 -= (dimension);}
                    zp1 = zi + 1;
                    if(zp1 >= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension))) { zp1 -= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}

                    if(user_params->PERTURB_ON_HIGH_RES) {
                        // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
#pragma omp atomic
                            resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
#pragma omp atomic
                            resampled_box[R_INDEX(xp1,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
#pragma omp atomic
                            resampled_box[R_INDEX(xi,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
#pragma omp atomic
                            resampled_box[R_INDEX(xp1,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
#pragma omp atomic
                            resampled_box[R_INDEX(xi,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
#pragma omp atomic
                            resampled_box[R_INDEX(xp1,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
#pragma omp atomic
                            resampled_box[R_INDEX(xi,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
#pragma omp atomic
                            resampled_box[R_INDEX(xp1,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                    }
                    else {
                        // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xp1,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xi,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xp1,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xi,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xp1,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xi,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
#pragma omp atomic
                            resampled_box[HII_R_INDEX(xp1,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                    }
                }
            }
        }
    }
    return resampled_box;
}
