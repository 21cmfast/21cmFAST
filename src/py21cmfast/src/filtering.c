
//filter_box, filter_box_annulus and filter_box_mfp should be combined in a better way, they require different inputs
//and they are run on different subsets of the boxes but they contain a lot of the same math

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>
#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "indexing.h"
#include "dft.h"

void filter_box_annulus(fftwf_complex *box, int RES, float R_inner, float R_outer){
    unsigned long long int n_x, n_z, n_y, dimension, midpoint;
    float k_x, k_y, k_z, k_mag, kRinner, kRouter;
    float f_inner, f_outer;

    switch(RES) {
        case 0:
            dimension = user_params_global->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = user_params_global->HII_DIM;
            midpoint = HII_MIDDLE;
            break;
    }
    // loop through k-box

#pragma omp parallel shared(box) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,kRinner,kRouter,f_inner,f_outer) num_threads(user_params_global->N_THREADS)
    {
#pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
            if (n_x>midpoint) {k_x =(n_x-dimension) * DELTA_K;}
            else {k_x = n_x * DELTA_K;}

            for (n_y=0; n_y<dimension; n_y++){
                if (n_y>midpoint) {k_y =(n_y-dimension) * DELTA_K;}
                else {k_y = n_y * DELTA_K;}

                for (n_z=0; n_z<=(unsigned long long)(user_params_global->NON_CUBIC_FACTOR*midpoint); n_z++){
                    k_z = n_z * DELTA_K_PARA;

                    k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                    kRinner = k_mag*R_inner;
                    kRouter = k_mag*R_outer;

                    if (kRinner > 1e-4){
                        f_inner = 3.0/(pow(kRouter, 3) - pow(kRinner, 3)) * (sin(kRinner) - cos(kRinner)*kRinner);
                        f_outer = 3.0/(pow(kRouter, 3) - pow(kRinner, 3)) * (sin(kRouter) - cos(kRouter)*kRouter);
                        if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= (f_outer - f_inner); }
                        if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= (f_outer - f_inner); }
                    }

                }
            }
        } // end looping through k box
    }
    return;
}

void filter_box(fftwf_complex *box, int RES, int filter_type, float R){
    unsigned long long int n_x, n_z, n_y, dimension,midpoint;
    float k_x, k_y, k_z, k_mag, kR;

    switch(RES) {
        case 0:
            dimension = user_params_global->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = user_params_global->HII_DIM;
            midpoint = HII_MIDDLE;
            break;
    }

    // loop through k-box

#pragma omp parallel shared(box) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,kR) num_threads(user_params_global->N_THREADS)
    {
#pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
            if (n_x>midpoint) {k_x =(n_x-dimension) * DELTA_K;}
            else {k_x = n_x * DELTA_K;}

            for (n_y=0; n_y<dimension; n_y++){
                if (n_y>midpoint) {k_y =(n_y-dimension) * DELTA_K;}
                else {k_y = n_y * DELTA_K;}
                for (n_z=0; n_z<=(unsigned long long)(user_params_global->NON_CUBIC_FACTOR*midpoint); n_z++){
                    k_z = n_z * DELTA_K_PARA;

                    if (filter_type == 0){ // real space top-hat

                        k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                        kR = k_mag*R; // real space top-hat

                        if (kR > 1e-4){
                            if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR); }
                            if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR); }
                        }
                    }
                    else if (filter_type == 1){ // k-space top hat

                        // This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
                        // as we don't need to evaluate the slower sqrt function
//                        kR = 0.17103765852*( k_x*k_x + k_y*k_y + k_z*k_z )*R*R;

                        k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                        kR = k_mag*R; // real space top-hat

                        kR *= 0.413566994; // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
                        if (kR > 1){
                            if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] = 0; }
                            if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] = 0; }
                        }
                    }
                    else if (filter_type == 2){ // gaussian
                        // This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
                        // as we don't need to evaluate the slower sqrt function
                        kR = 0.643*0.643*( k_x*k_x + k_y*k_y + k_z*k_z )*R*R;
//                        kR *= 0.643; // equates integrated volume to the real space top-hat
                        if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR/2.0); }
                        if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR/2.0); }
                    }
                    else{
                        if ( (n_x==0) && (n_y==0) && (n_z==0) )
                            LOG_WARNING("Filter type %i is undefined. Box is unfiltered.", filter_type);
                    }
                }
            }
        } // end looping through k box
    }

    return;
}

void filter_box_mfp(fftwf_complex *box, int RES, float R, float mfp){
    unsigned long long int n_x, n_z, n_y, dimension, midpoint;
    float k_x, k_y, k_z, k_mag, f, kR, kl;
    float const1;
    const1 = exp(-R/mfp); //independent of k, move it out of the loop
    // LOG_DEBUG("Filtering box with R=%.2e, L=%.2e",R,mfp);

    switch(RES) {
        case 0:
            dimension = user_params_global->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = user_params_global->HII_DIM;
            midpoint = HII_MIDDLE;
            break;
    }
    // loop through k-box

#pragma omp parallel shared(box) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,kR,kl,f) num_threads(user_params_global->N_THREADS)
    {
#pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
            if (n_x>midpoint) {k_x =(n_x-dimension) * DELTA_K;}
            else {k_x = n_x * DELTA_K;}

            for (n_y=0; n_y<dimension; n_y++){
                if (n_y>midpoint) {k_y =(n_y-dimension) * DELTA_K;}
                else {k_y = n_y * DELTA_K;}
                for (n_z=0; n_z<=(unsigned long long)(user_params_global->NON_CUBIC_FACTOR*midpoint); n_z++){
                    k_z = n_z * DELTA_K_PARA;

                    k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                    kR = k_mag*R;
                    kl = k_mag*mfp;

                    //Davies & Furlanetto MFP-eps(r) window function
                    //The filter no longer approaches 1 at k->0, so we can't use the limit
                    if (kR > 1e-4){
                        //build the filter
                        f = (kl*kl*R + 2*mfp + R)*kl*cos(kR);
                        f += (-kl*kl*mfp + kl*kl*R + mfp + R)*sin(kR);
                        f *= const1;
                        f -= 2*kl*mfp;
                        f *= -3.0*mfp/(kR*R*R*(kl*kl+1)*(kl*kl+1));
                    }
                    else{
                        // k-> 0 limit
                        f = 2*mfp*mfp + 2*mfp*R + R*R;
                        f *= -const1;
                        f += 2*mfp*mfp;
                        f *= 3*mfp/(R*R*R);
                    }
                    if(RES==1) { box[HII_C_INDEX(n_x, n_y, n_z)] *= f; }
                    if(RES==0) { box[C_INDEX(n_x, n_y, n_z)] *= f; }
                }
            }
        } // end looping through k box
    }
    return;
}

int test_mfp_filter(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double mfp, double *result){
    int i,j,k;
    unsigned long long int ii;
    //setup the box

    fftwf_complex *box_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                *((float *)box_unfiltered + HII_R_FFT_INDEX(i,j,k)) = input_box[HII_R_INDEX(i,j,k)];


    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_unfiltered);

    //QUESTION: why do this here instead of at the end?
    for(ii=0;ii<HII_KSPACE_NUM_PIXELS;ii++){
        box_unfiltered[ii] /= (double)HII_TOT_NUM_PIXELS;
    }

    memcpy(box_filtered,box_unfiltered,sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    if(flag_options->USE_EXP_FILTER)
        filter_box_mfp(box_filtered, 1, R, mfp);
    else
        filter_box(box_filtered,1,global_params.HII_FILTER,R);


    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_filtered);

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                    result[HII_R_INDEX(i,j,k)] = fmaxf(*((float *)box_filtered + HII_R_FFT_INDEX(i,j,k)) , 0.0);

    fftwf_free(box_unfiltered);
    fftwf_free(box_filtered);

    return 0;
}
