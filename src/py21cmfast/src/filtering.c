
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

double real_tophat_filter(double kR){
    if (kR < 1e-4)
        return 1;
    return 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR);
}

//TODO: TEST USING kR^2 INSTEAD FOR SPEED
//  ALSO TEST ASSIGNMENT vs MULTIPLICATION
double sharp_k_filter(double kR){
    // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
    if (kR*0.413566994 > 1)
       return 0.;
    return 1;
}

double gaussian_filter(double kR_squared){
    return exp(-0.643*0.643*kR_squared/2.);
}

double exp_mfp_filter(double k, double R, double mfp, double R_constant, double limit){
    double kR,kR2,f;

    kR = k*R;
    if(kR < 1e-4)
        return limit;

    kR2 = k*mfp;
    //Davies & Furlanetto MFP-eps(r) window function
    //The filter no longer approaches 1 at k->0, so we can't use the limit
    f = (kR2*kR2*R + 2*mfp + R)*kR2*cos(kR);
    f += (-kR2*kR2*mfp + kR2*kR2*R + mfp + R)*sin(kR);
    f *= R_constant;
    f -= 2*kR2*mfp;
    f *= -3.0*mfp/(kR*R*R*(kR2*kR2+1)*(kR2*kR2+1));
    return f;
}

double spherical_shell_filter(double k, double R_outer, double R_inner){
    double kR_inner = k*R_inner;
    double kR_outer = k*R_outer;

    if (kR_outer < 1e-4)
        return 1.;

    return 3.0/(pow(kR_outer, 3) - pow(kR_inner, 3)) \
        * (sin(kR_outer) - cos(kR_outer)*kR_outer \
        -  sin(kR_inner) + cos(kR_inner)*kR_inner);
}

void filter_box(fftwf_complex *box, int RES, int filter_type, float R, float R_param){
    int dimension, midpoint; //TODO: figure out why defining as ULL breaks this
    switch(RES) {
        case 0:
            dimension = user_params_global->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = user_params_global->HII_DIM;
            midpoint = HII_MIDDLE;
            break;
        default:
            LOG_ERROR("Resolution for filter functions must be 0(DIM) or 1(HII_DIM)");
            Throw(ValueError);
            break;
    }

    //setup constants if needed
    float R_constant_1, R_constant_2;
    if(filter_type == 3){
        R_constant_1 = exp(-R/R_param); //independent of k
        //k->0 limit of the mfp filter
        R_constant_2 = (2*R_param*R_param - R_constant_1*(2*R_param*R_param + 2*R_param*R + R*R))*3*R_param/(R*R*R);
    }

    // loop through k-box
    #pragma omp parallel num_threads(user_params_global->N_THREADS)
    {
        int n_x, n_z, n_y;
        float k_x, k_y, k_z, k_mag_sq, kR;
        unsigned long long grid_index;
        #pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
            if (n_x>midpoint) {k_x =(n_x-dimension) * DELTA_K;}
            else {k_x = n_x * DELTA_K;}

            for (n_y=0; n_y<dimension; n_y++){
                if (n_y>midpoint) {k_y =(n_y-dimension) * DELTA_K;}
                else {k_y = n_y * DELTA_K;}

                for (n_z=0; n_z<=(int)(user_params_global->NON_CUBIC_FACTOR*midpoint); n_z++){
                    k_z = n_z * DELTA_K_PARA;
                    k_mag_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                    grid_index = RES==1 ? HII_C_INDEX(n_x, n_y, n_z) : C_INDEX(n_x, n_y, n_z);

                    if (filter_type == 0){ // real space top-hat
                        kR = sqrt(k_mag_sq)*R;
                        box[grid_index] *= real_tophat_filter(kR);
                    }
                    else if (filter_type == 1){ // k-space top hat
                        //NOTE: why was this commented????
                        // This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
                        // kR = 0.17103765852*( k_x*k_x + k_y*k_y + k_z*k_z )*R*R;
                        kR = sqrt(k_mag_sq)*R;
                        box[grid_index] *= sharp_k_filter(kR);
                    }
                    else if (filter_type == 2){ // gaussian
                        // This is actually (kR^2) but since we zero the value and find kR > 1 this is more computationally efficient
                        kR = k_mag_sq*R*R;
                        box[grid_index] *= gaussian_filter(kR);
                    }
                    //The next two filters are not given by the HII_FILTER global, but used for specific grids
                    else if (filter_type == 3){ // exponentially decaying tophat, param == scale of decay (MFP)
                        //NOTE: This should be optimized, I havne't looked at it in a while
                        box[grid_index] *= exp_mfp_filter(sqrt(k_mag_sq),R,R_param,R_constant_1,R_constant_2);
                    }
                    else if (filter_type == 4){ //spherical shell, R_param == inner radius
                        box[grid_index] *= spherical_shell_filter(sqrt(k_mag_sq),R,R_param);
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

//Test function to filter a box without computing a whole output box
int test_filter(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double R_param, int filter_flag, double *result){
    int i,j,k;
    unsigned long long int ii;

    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);

    //setup the box
    fftwf_complex *box_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                *((float *)box_unfiltered + HII_R_FFT_INDEX(i,j,k)) = input_box[HII_R_INDEX(i,j,k)];

    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_unfiltered);

    for(ii=0;ii<HII_KSPACE_NUM_PIXELS;ii++){
        box_unfiltered[ii] /= (double)HII_TOT_NUM_PIXELS;
    }

    memcpy(box_filtered,box_unfiltered,sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    filter_box(box_filtered,1,filter_flag,R,R_param);

    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_filtered);

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                    result[HII_R_INDEX(i,j,k)] = *((float *)box_filtered + HII_R_FFT_INDEX(i,j,k));

    fftwf_free(box_unfiltered);
    fftwf_free(box_filtered);

    return 0;
}
