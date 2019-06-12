#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
//#include <pthread.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "21CMMC.h"
#include "logger.h"
#include "Constants.h"
#include "Globals.h"
#include "UsefulFunctions.c"
#include "ps.c"
#include "PerturbField.c"
#include "bubble_helper_progs.c"
#include "elec_interp.c"
#include "heating_helper_progs.c"
#include "recombinations.c"
#include "IonisationBox.c"
#include "SpinTemperatureBox.c"
#include "BrightnessTemperatureBox.c"



// Re-write of init.c for being accessible within the MCMC

int ComputeInitialConditions(unsigned long long random_seed, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes) {
    
    /*
     Generates the initial conditions: gaussian random density field (DIM^3) as well as the equal or lower resolution velocity fields, and smoothed density field (HII_DIM^3).
     See INIT_PARAMS.H and ANAL_PARAMS.H to set the appropriate parameters.
     Output is written to ../Boxes
     
     Author: Andrei Mesinger
     Date: 9/29/06
     */
    
    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    
    fftwf_plan plan;
    
    char wisdom_filename[500];
    
    unsigned long long ct;
    int n_x, n_y, n_z, i, j, k, ii;
    float k_x, k_y, k_z, k_mag, p, a, b, k_sq;
    double pixel_deltax;
    
    float f_pixel_factor;
    
    gsl_rng * r;
    
    // ************  INITIALIZATION ********************** //
    
    // Removed all references to threads as 21CMMC is always a single core implementation

    // seed the random number generators
    r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, random_seed);

    // allocate array for the k-space and real-space boxes
    fftwf_complex *HIRES_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    fftwf_complex *HIRES_box_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    // find factor of HII pixel size / deltax pixel size
    f_pixel_factor = user_params->DIM/(float)user_params->HII_DIM;
 
    // ************  END INITIALIZATION ****************** //
    
    // ************ CREATE K-SPACE GAUSSIAN RANDOM FIELD *********** //

    init_ps();

//    boxes->PSnormalisation = sigma_norm;

    for (n_x=0; n_x<user_params->DIM; n_x++){
        // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
        if (n_x>MIDDLE)
            k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
        else
            k_x = n_x * DELTA_K;
        
        for (n_y=0; n_y<user_params->DIM; n_y++){
            // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
            if (n_y>MIDDLE)
                k_y =(n_y-user_params->DIM) * DELTA_K;
            else
                k_y = n_y * DELTA_K;
            
            // since physical space field is real, only half contains independent modes
            for (n_z=0; n_z<=MIDDLE; n_z++){
                // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
                k_z = n_z * DELTA_K;
                
                // now get the power spectrum; remember, only the magnitude of k counts (due to issotropy)
                // this could be used to speed-up later maybe
                k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                p = power_in_k(k_mag);
                
                // ok, now we can draw the values of the real and imaginary part
                // of our k entry from a Gaussian distribution
                a = gsl_ran_ugaussian(r);
                b = gsl_ran_ugaussian(r);
                HIRES_box[C_INDEX(n_x, n_y, n_z)] = sqrt(VOLUME*p/2.0) * (a + b*I);
            }
        }
    }

    // *****  Adjust the complex conjugate relations for a real array  ***** //
    adj_complex_conj(HIRES_box,user_params,cosmo_params);
    // *** Let's also create a lower-resolution version of the density field  *** //

    memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    if (user_params->DIM != user_params->HII_DIM)
        filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));

    // FFT back to real space
    if(user_params->USE_FFTW_WISDOM) {
        // Check to see if the wisdom exists, create it if it doesn't
        sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->DIM);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_PATIENT);
            fftwf_execute(plan);
            
            // Store the wisdom for later use
            fftwf_export_wisdom_to_filename(wisdom_filename);
            
            // copy over unfiltered box
            memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
            
            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
    }
    else {
        plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
        fftwf_execute(plan);
    }
    
    // now sample the filtered box
    for (i=0; i<user_params->HII_DIM; i++){
        for (j=0; j<user_params->HII_DIM; j++){
            for (k=0; k<user_params->HII_DIM; k++){
                boxes->lowres_density[HII_R_INDEX(i,j,k)] =
                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                   (unsigned long long)(j*f_pixel_factor+0.5),
                                                   (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
            }
        }
    }
    
    // ******* PERFORM INVERSE FOURIER TRANSFORM ***************** //
    // add the 1/VOLUME factor when converting from k space to real space
    memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    for (ct=0; ct<KSPACE_NUM_PIXELS; ct++){
        HIRES_box[ct] /= VOLUME;
    }

    if(user_params->USE_FFTW_WISDOM) {
        plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
    }
    else {
        plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
    }
    fftwf_execute(plan);
    
    for (i=0; i<user_params->DIM; i++){
        for (j=0; j<user_params->DIM; j++){
            for (k=0; k<user_params->DIM; k++){
                *((float *)boxes->hires_density + R_INDEX(i,j,k)) = *((float *)HIRES_box + R_FFT_INDEX(i,j,k));
            }
        }
    }
    
    for(ii=0;ii<3;ii++) {

        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        // Now let's set the velocity field/dD/dt (in comoving Mpc)
        
        for (n_x=0; n_x<user_params->DIM; n_x++){
            if (n_x>MIDDLE)
                k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;
            
            for (n_y=0; n_y<user_params->DIM; n_y++){
                if (n_y>MIDDLE)
                    k_y =(n_y-user_params->DIM) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;
                
                for (n_z=0; n_z<=MIDDLE; n_z++){
                    k_z = n_z * DELTA_K;
                    
                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                    
                    // now set the velocities
                    if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                        HIRES_box[0] = 0;
                    }
                    else{
                        if(ii==0) {
                            HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq/VOLUME;
                        }
                        if(ii==1) {
                            HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq/VOLUME;
                        }
                        if(ii==2) {
                            HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq/VOLUME;
                        }
                    }
                }
            }
        }
    
        if (user_params->DIM != user_params->HII_DIM)
            filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
        
        if(user_params->USE_FFTW_WISDOM) {
            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
        }
        fftwf_execute(plan);
        
        // now sample to lower res
        // now sample the filtered box
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    if(ii==0) {
                        boxes->lowres_vx[HII_R_INDEX(i,j,k)] =
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                    if(ii==1) {
                        boxes->lowres_vy[HII_R_INDEX(i,j,k)] =
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                    if(ii==2) {
                        boxes->lowres_vz[HII_R_INDEX(i,j,k)] =
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                }
            }
        }
    }
    // write out file

    // * *************************************************** * //
    // *              BEGIN 2LPT PART                        * //
    // * *************************************************** * //

    // Generation of the second order Lagrangian perturbation theory (2LPT) corrections to the ZA
    // reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D

    // Parameter set in ANAL_PARAMS.H
    if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
        
        // use six supplementary boxes to store the gradients of phi_1 (eq. D13b)
        // Allocating the boxes
#define PHI_INDEX(i, j) ((int) ((i) - (j)) + 3*((j)) - ((int)(j))/2  )
        // ij -> INDEX
        // 00 -> 0
        // 11 -> 3
        // 22 -> 5
        // 10 -> 1
        // 20 -> 2
        // 21 -> 4
        
        fftwf_complex *phi_1[6];
        
        for(i = 0; i < 3; ++i){
            for(j = 0; j <= i; ++j){
                phi_1[PHI_INDEX(i, j)] = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
            }
        }
        
        for(i = 0; i < 3; ++i){
            for(j = 0; j <= i; ++j){
                
                // read in the box
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
                
                // generate the phi_1 boxes in Fourier transform
                for (n_x=0; n_x<user_params->DIM; n_x++){
                    if (n_x>MIDDLE)
                        k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                    else
                        k_x = n_x * DELTA_K;
                    
                    for (n_y=0; n_y<user_params->DIM; n_y++){
                        if (n_y>MIDDLE)
                            k_y =(n_y-user_params->DIM) * DELTA_K;
                        else
                            k_y = n_y * DELTA_K;
                        
                        for (n_z=0; n_z<=MIDDLE; n_z++){
                            k_z = n_z * DELTA_K;
                            
                            k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                            
                            float k[] = {k_x, k_y, k_z};
                            // now set the velocities
                            if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                                phi_1[PHI_INDEX(i, j)][0] = 0;
                            }
                            else{
                                phi_1[PHI_INDEX(i, j)][C_INDEX(n_x,n_y,n_z)] = -k[i]*k[j]*HIRES_box[C_INDEX(n_x, n_y, n_z)]/k_sq/VOLUME;
                                // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                            }
                        }
                    }
                }
                // Now we can generate the real phi_1[i,j]
                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)phi_1[PHI_INDEX(i, j)], (float *)phi_1[PHI_INDEX(i, j)], FFTW_ESTIMATE);
                fftwf_execute(plan);
            }
        }
        
        // Then we will have the laplacian of phi_2 (eq. D13b)
        // After that we have to return in Fourier space and generate the Fourier transform of phi_2
        int m, l;
        for (i=0; i<user_params->DIM; i++){
            for (j=0; j<user_params->DIM; j++){
                for (k=0; k<user_params->DIM; k++){
                    *( (float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i), (unsigned long long)(j), (unsigned long long)(k) )) = 0.0;
                    for(m = 0; m < 3; ++m){
                        for(l = m+1; l < 3; ++l){
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) += ( *((float *)(phi_1[PHI_INDEX(l, l)]) + R_FFT_INDEX((unsigned long long) (i),(unsigned long long) (j),(unsigned long long) (k)))  ) * (  *((float *)(phi_1[PHI_INDEX(m, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)))  );
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) -= ( *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long) (j),(unsigned long long)(k) ) )  ) * (  *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k) ))  );
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) /= TOT_NUM_PIXELS;
                        }
                    }
                }
            }
        }
        
        
        // Perform FFTs
        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if wisdom exists, if not create it
            sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
                
                // Repeating the above computation as the creating the wisdom overwrites the input data
                
                // Then we will have the laplacian of phi_2 (eq. D13b)
                // After that we have to return in Fourier space and generate the Fourier transform of phi_2
                int m, l;
                for (i=0; i<user_params->DIM; i++){
                    for (j=0; j<user_params->DIM; j++){
                        for (k=0; k<user_params->DIM; k++){
                            *( (float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i), (unsigned long long)(j), (unsigned long long)(k) )) = 0.0;
                            for(m = 0; m < 3; ++m){
                                for(l = m+1; l < 3; ++l){
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) += ( *((float *)(phi_1[PHI_INDEX(l, l)]) + R_FFT_INDEX((unsigned long long) (i),(unsigned long long) (j),(unsigned long long) (k)))  ) * (  *((float *)(phi_1[PHI_INDEX(m, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)))  );
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) -= ( *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long) (j),(unsigned long long)(k) ) )  ) * (  *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k) ))  );
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) /= TOT_NUM_PIXELS;
                                }
                            }
                        }
                    }
                }
                
                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        
        memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // Now we can store the content of box in a back-up file
        // Then we can generate the gradients of phi_2 (eq. D13b and D9)

        // ***** Write out back-up k-box RHS eq. D13b ***** //

        // For each component, we generate the velocity field (same as the ZA part)
        
        // Now let's set the velocity field/dD/dt (in comoving Mpc)
        
        // read in the box
        // TODO correct free of phi_1
        
        for(ii=0;ii<3;ii++) {
            
            if(ii>0) {
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
            }
        
            // set velocities/dD/dt
            for (n_x=0; n_x<user_params->DIM; n_x++){
                if (n_x>MIDDLE)
                    k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                else
                    k_x = n_x * DELTA_K;
                
                for (n_y=0; n_y<user_params->DIM; n_y++){
                    if (n_y>MIDDLE)
                        k_y =(n_y-user_params->DIM) * DELTA_K;
                    else
                        k_y = n_y * DELTA_K;
                    
                    for (n_z=0; n_z<=MIDDLE; n_z++){
                        k_z = n_z * DELTA_K;
                        
                        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                        
                        // now set the velocities
                        if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                            HIRES_box[0] = 0;
                        }
                        else{
                            if(ii==0) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq;
                            }
                            if(ii==1) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq;
                            }
                            if(ii==2) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq;
                            }
                        }
                    }
                    // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                }
            }
        
            if (user_params->DIM != user_params->HII_DIM)
                filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
            
            if(user_params->USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
            // now sample to lower res
            // now sample the filtered box
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        if(ii==0) {
                            boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] =
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)));
                        }
                        if(ii==1) {
                            boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] =
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)));
                        }
                        if(ii==2) {
                            boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] =
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)));
                        }
                    }
                }
            }
        }
    
        // deallocate the supplementary boxes
        for(i = 0; i < 3; ++i){
            for(j = 0; j <= i; ++j){
                fftwf_free(phi_1[PHI_INDEX(i,j)]);
            }
        }
    }

    // * *********************************************** * //
    // *               END 2LPT PART                     * //
    // * *********************************************** * //
    
    // deallocate
    fftwf_free(HIRES_box);
    fftwf_free(HIRES_box_saved);

    return(0);

}

/*****  Adjust the complex conjugate relations for a real array  *****/

void adj_complex_conj(fftwf_complex *HIRES_box, struct UserParams *user_params, struct CosmoParams *cosmo_params){
    int i, j, k;
    
    // corners
    HIRES_box[C_INDEX(0,0,0)] = 0;
    HIRES_box[C_INDEX(0,0,MIDDLE)] = crealf(HIRES_box[C_INDEX(0,0,MIDDLE)]);
    HIRES_box[C_INDEX(0,MIDDLE,0)] = crealf(HIRES_box[C_INDEX(0,MIDDLE,0)]);
    HIRES_box[C_INDEX(0,MIDDLE,MIDDLE)] = crealf(HIRES_box[C_INDEX(0,MIDDLE,MIDDLE)]);
    HIRES_box[C_INDEX(MIDDLE,0,0)] = crealf(HIRES_box[C_INDEX(MIDDLE,0,0)]);
    HIRES_box[C_INDEX(MIDDLE,0,MIDDLE)] = crealf(HIRES_box[C_INDEX(MIDDLE,0,MIDDLE)]);
    HIRES_box[C_INDEX(MIDDLE,MIDDLE,0)] = crealf(HIRES_box[C_INDEX(MIDDLE,MIDDLE,0)]);
    HIRES_box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)] = crealf(HIRES_box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)]);
    
    // do entire i except corners
    for (i=1; i<MIDDLE; i++){
        // just j corners
        for (j=0; j<=MIDDLE; j+=MIDDLE){
            for (k=0; k<=MIDDLE; k+=MIDDLE){
                HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX((user_params->DIM)-i,j,k)]);
            }
        }
        
        // all of j
        for (j=1; j<MIDDLE; j++){
            for (k=0; k<=MIDDLE; k+=MIDDLE){
                HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX((user_params->DIM)-i,(user_params->DIM)-j,k)]);
                HIRES_box[C_INDEX(i,(user_params->DIM)-j,k)] = conjf(HIRES_box[C_INDEX((user_params->DIM)-i,j,k)]);
            }
        }
    } // end loop over i
    
    // now the i corners
    for (i=0; i<=MIDDLE; i+=MIDDLE){
        for (j=1; j<MIDDLE; j++){
            for (k=0; k<=MIDDLE; k+=MIDDLE){
                HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX(i,(user_params->DIM)-j,k)]);
            }
        }
    } // end loop over remaining j

}

