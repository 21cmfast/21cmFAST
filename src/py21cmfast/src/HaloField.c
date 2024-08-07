
// Re-write of find_halos.c from the original 21cmFAST


// ComputeHaloField takes in a k_space box of the linear overdensity field
// and filters it on decreasing scales in order to find virialized halos.
// Virialized halos are defined according to the linear critical overdensity.
// ComputeHaloField outputs a cube with non-zero elements containing the Mass of
// the virialized halos
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <fftw3.h>
#include <math.h>
#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cosmology.h"
#include "indexing.h"
#include "hmf.h"
#include "interp_tables.h"
#include "dft.h"
#include "filtering.h"
#include "debugging.h"
#include "hmf.h"
#include "Stochasticity.h"

#include "HaloField.h"

int check_halo(char * in_halo, UserParams *user_params, float R, int x, int y, int z, int check_type);
void init_halo_coords(HaloField *halos, long long unsigned int n_halos);
int pixel_in_halo(int grid_dim, int z_dim, int x, int x_index, int y, int y_index, int z, int z_index, float Rsq_curr_index );
void free_halo_field(HaloField *halos);

int ComputeHaloField(float redshift_desc, float redshift, UserParams *user_params, CosmoParams *cosmo_params,
                     AstroParams *astro_params, FlagOptions *flag_options,
                     InitialConditions *boxes, unsigned long long int random_seed,
                     HaloField * halos_desc, HaloField *halos) {

    int status;

    Try{ // This Try brackets the whole function, so we don't indent.

    //This happens if we are updating a halo field (no need to redo big halos)
    if(flag_options->HALO_STOCHASTICITY && redshift_desc > 0){
        LOG_DEBUG("Halo sampling switched on, bypassing halo finder to update %llu halos...",halos_desc->n_halos);
        //this would hold the two boxes used in the halo sampler, but here we are taking the sample from a catalogue so we define a dummy here
        float *dummy_box = NULL;
        stochastic_halofield(user_params, cosmo_params, astro_params, flag_options, random_seed, redshift_desc,
                            redshift, dummy_box, dummy_box, halos_desc, halos);
        return 0;
    }

LOG_DEBUG("input value:");
LOG_DEBUG("redshift=%f", redshift);
#if LOG_LEVEL >= DEBUG_LEVEL
        writeUserParams(user_params);
        writeCosmoParams(cosmo_params);
        writeAstroParams(flag_options, astro_params);
        writeFlagOptions(flag_options);
#endif

        // Makes the parameter structs visible to a variety of functions/macros
        // Do each time to avoid Python garbage collection issues
        Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);

        omp_set_num_threads(user_params->N_THREADS);

        fftwf_complex *density_field, *density_field_saved;

        float growth_factor, R, delta_m, M, Delta_R, delta_crit;
        char *in_halo, *forbidden;
        int i,j,k,x,y,z;
        long long unsigned int total_halo_num, r_halo_num;
        float R_temp, M_MIN;

        LOG_DEBUG("Begin Initialisation");

        // ***************** BEGIN INITIALIZATION ***************** //
        init_ps();

        growth_factor = dicke(redshift); // normalized to 1 at z=0
        delta_crit = Deltac; // for now set to spherical; check if we want elipsoidal later

        //store highly used parameters
        int grid_dim = user_params->DIM;
        int z_dim = D_PARA;

        //set minimum source mass
        M_MIN = minimum_source_mass(redshift, false, astro_params, flag_options);
        //if we use the sampler we want to stop at the HII cell mass
        if(flag_options->HALO_STOCHASTICITY)
            M_MIN = fmax(M_MIN,RtoM(L_FACTOR*user_params->BOX_LEN/user_params->HII_DIM));
        //otherwise we stop at the cell mass
        else
            M_MIN = fmax(M_MIN,RtoM(L_FACTOR*user_params->BOX_LEN/grid_dim));

        // allocate array for the k-space box
        density_field = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        density_field_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // allocate memory for the boolean in_halo box
        in_halo = (char *) malloc(sizeof(char)*TOT_NUM_PIXELS);

        // initialize
        memset(in_halo, 0, sizeof(char)*TOT_NUM_PIXELS);

        if(global_params.OPTIMIZE) {
            forbidden = (char *) malloc(sizeof(char)*TOT_NUM_PIXELS);
        }

        //Unused variables, for future threading
        unsigned long long int nhalo_threads[user_params->N_THREADS];
        unsigned long long int istart_threads[user_params->N_THREADS];
        //expected TOTAL halos in box from minimum source mass

        unsigned long long int arraysize_total = halos->buffer_size;
        unsigned long long int arraysize_local = arraysize_total / user_params->N_THREADS;

        if(LOG_LEVEL >= DEBUG_LEVEL){
            double Mmax_debug = 1e16;
            initialiseSigmaMInterpTable(M_MIN*0.9,Mmax_debug*1.1);
            double nhalo_debug = Nhalo_General(redshift, log(M_MIN), log(Mmax_debug)) * VOLUME;
            //expected halos above minimum filter mass
            LOG_DEBUG("DexM: We expect %.2f Halos between Masses [%.2e,%.2e]",nhalo_debug,M_MIN,Mmax_debug);
        }

#pragma omp parallel shared(boxes,density_field) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<grid_dim; i++){
                for (j=0; j<grid_dim; j++){
                    for (k=0; k<z_dim; k++){
                        *((float *)density_field + R_FFT_INDEX(i,j,k)) = *((float *)boxes->hires_density + R_INDEX(i,j,k));
                    }
                }
            }
        }

        dft_r2c_cube(user_params->USE_FFTW_WISDOM, grid_dim, z_dim, user_params->N_THREADS, density_field);

        // save a copy of the k-space density field
        memcpy(density_field_saved, density_field, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // ***************** END INITIALIZATION ***************** //

        LOG_DEBUG("Finalised Initialisation");

        // lets filter it now
        // set initial R value
        Delta_R = L_FACTOR*2.*user_params->BOX_LEN/(grid_dim+0.0);

        total_halo_num = 0;
        R = MtoR(M_MIN*1.01); // one percent higher for rounding

        LOG_DEBUG("Prepare to filter to find halos");

        while (R < L_FACTOR*user_params->BOX_LEN)
            R*=global_params.DELTA_R_FACTOR;

        HaloField *halos_dexm;
        if(flag_options->HALO_STOCHASTICITY){
            //To save memory, we allocate the smaller (large mass) halofield here instead of using halos_desc
            halos_dexm = malloc(sizeof(HaloField));
        }
        else{
            //assign directly to the output field instead
            halos_dexm = halos;
        }

        float *halo_field = calloc(TOT_NUM_PIXELS, sizeof(float));

        while ((R > 0.5*Delta_R) && (RtoM(R) >= M_MIN)){ // filter until we get to half the pixel size or M_MIN
            M = RtoM(R);
            LOG_SUPER_DEBUG("while loop for finding halos: R = %f 0.5*Delta_R = %f RtoM(R)=%e M_MIN=%e", R, 0.5*Delta_R, M, M_MIN);

            //Pending a serious deep-dive into this algorithm, I will force DexM to use the fitted parameters to the
            //      Sheth-Tormen mass function (as of right now, We do not even reproduce EPS results)
            delta_crit = growth_factor*sheth_delc_dexm(Deltac/growth_factor, sigma_z0(M));

            // if(global_params.DELTA_CRIT_MODE == 1){
            //     //This algorithm does not use the sheth tormen OR Jenkins parameters,
            //     //        rather it uses a reduced barrier to correct for some part of this algorithm,
            //     //        which would otherwise result in ~3x halos at redshift 6 (including with EPS).
            //     //        Once I Figure out the cause of the discrepancy I can adjust again but for now
            //     //        we will use the parameters that roughly give the ST mass function.
            //     // delta_crit = growth_factor*sheth_delc(Deltac/growth_factor, sigma_z0(M));
            //     if(user_params->HMF==1) {
            //         // use sheth tormen correction
            //         delta_crit = growth_factor*sheth_delc(Deltac/growth_factor, sigma_z0(M));
            //     }
            //     else if(user_params->HMF==4) {
            //         // use Delos 2023 flat barrier
            //         delta_crit = DELTAC_DELOS;
            //     }
            //     else if(user_params->HMF!=0){
            //         LOG_WARNING("Halo Finder: You have selected DELTA_CRIT_MODE==1 with HMF %d which does not have a barrier
            //                         , using EPS deltacrit = 1.68",user_params->HMF);
            //     }
            // }

            // first let's check if virialized halos of this size are rare enough
            // that we don't have to worry about them (let's define 7 sigma away, as in Mesinger et al 05)
            if ((sigma_z0(M)*growth_factor*7.) < delta_crit){
                LOG_SUPER_DEBUG("Haloes too rare for M = %e! Skipping...", M);
                R /= global_params.DELTA_R_FACTOR;
                continue;
            }

            memcpy(density_field, density_field_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

            // now filter the box on scale R
            // 0 = top hat in real space, 1 = top hat in k space
            filter_box(density_field, 0, global_params.HALO_FILTER, R);

            // do the FFT to get delta_m box
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, grid_dim, z_dim, user_params->N_THREADS, density_field);

            // *****************  BEGIN OPTIMIZATION ***************** //
            // to optimize speed, if the filter size is large (switch to collapse fraction criteria later)
            if(global_params.OPTIMIZE) {
                if(M > global_params.OPTIMIZE_MIN_MASS) {
                    memset(forbidden, 0, sizeof(char)*TOT_NUM_PIXELS);
                    // now go through the list of existing halos and paint on the no-go region onto <forbidden>

#pragma omp parallel shared(forbidden,R) private(x,y,z,R_temp) num_threads(user_params->N_THREADS)
                    {
                        float halo_buf;
#pragma omp for
                        for (x=0; x<grid_dim; x++){
                            for (y=0; y<grid_dim; y++){
                                for (z=0; z<z_dim; z++){
                                    halo_buf = halo_field[R_INDEX(x,y,z)];
                                    if(halo_buf > 0.) {
                                        R_temp = MtoR(halo_buf);
                                        check_halo(forbidden, user_params, R_temp+global_params.R_OVERLAP_FACTOR*R, x,y,z,2);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // *****************  END OPTIMIZATION ***************** //
            // now lets scroll through the box, flagging all pixels with delta_m > delta_crit
            r_halo_num=0;

            //THREADING: Fix the race condition propertly to thread: it doesn't matter which thread finds the halo first
            //  but if two threads find a halo in the same region simultaneously (before the first one updates in_halo) some halos could double-up
            //checking for overlaps in new halos after this loop could work, but I would have to calculate distances between all new halos which sounds slow
            for (x=0; x<grid_dim; x++){
                for (y=0; y<grid_dim; y++){
                    for (z=0; z<z_dim; z++){
                        delta_m = *((float *)density_field + R_FFT_INDEX(x,y,z)) * growth_factor / TOT_NUM_PIXELS;
                        // LOG_ULTRA_DEBUG("Cell (%d,%d,%d) idx %d got delta %.6e",x,y,z,R_INDEX(x,y,z),delta_m);
                        // LOG_ULTRA_DEBUG("in halo %c OPTIMIZE %d",in_halo[R_INDEX(x,y,z)],global_params.OPTIMIZE);
                        // if not within a larger halo, and radii don't overlap, update in_halo box
                        // *****************  BEGIN OPTIMIZATION ***************** //
                        if(global_params.OPTIMIZE && (M > global_params.OPTIMIZE_MIN_MASS)) {
                            if ( (delta_m > delta_crit) && !forbidden[R_INDEX(x,y,z)]){
                                check_halo(in_halo, user_params, R, x,y,z,2); // flag the pixels contained within this halo
                                check_halo(forbidden, user_params, (1.+global_params.R_OVERLAP_FACTOR)*R, x,y,z,2); // flag the pixels contained within this halo

                                halo_field[R_INDEX(x,y,z)] = M;

                                r_halo_num++; // keep track of the number of halos
                            }
                        }
                        // *****************  END OPTIMIZATION ***************** //
                        else {
                            if ((delta_m > delta_crit)){
                                // LOG_ULTRA_DEBUG("delta peak found at (%d,%d,%d), delta = %.4f",x,y,z,delta_m);
                                if(!in_halo[R_INDEX(x,y,z)]){
                                    // LOG_ULTRA_DEBUG("Not in halo",in_halo[R_INDEX(x,y,z)]);
                                    if(!check_halo(in_halo, user_params, R, x,y,z,1)){ // we found us a "new" halo!
                                        check_halo(in_halo, user_params, R, x,y,z,2); // flag the pixels contained within this halo
                                        // LOG_ULTRA_DEBUG("Flagged Pixels");

                                        halo_field[R_INDEX(x,y,z)] = M;

                                        r_halo_num++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            total_halo_num += r_halo_num;
            LOG_SUPER_DEBUG("n_halo = %llu, total = %llu , D = %.3f, delcrit = %.3f", r_halo_num, total_halo_num, growth_factor, delta_crit);

            R /= global_params.DELTA_R_FACTOR;
        }

        LOG_DEBUG("Obtained %llu halo masses and positions, now saving to HaloField struct.",total_halo_num);

        //Allocate the Halo Mass and Coordinate Fields (non-wrapper structure)
        if(flag_options->HALO_STOCHASTICITY)
            init_halo_coords(halos_dexm, total_halo_num);
        else
            halos_dexm->n_halos = total_halo_num;

        //Assign to the struct
        //NOTE: To thread this part, we would need to keep track of how many halos are in each thread before
        //      OR assign a buffer of size n_halo * n_thread (in case the last thread has all the halos),
        //      copy the structure from stochasticity.c with the assignment and condensing
        unsigned long long int count=0;
        float halo_buf = 0;
        for (x=0; x<grid_dim; x++){
            for (y=0; y<grid_dim; y++){
                for (z=0; z<z_dim; z++){
                    halo_buf = halo_field[R_INDEX(x,y,z)];
                    if(halo_buf > 0.) {
                        halos_dexm->halo_masses[count] = halo_buf;
                        halos_dexm->halo_coords[3*count + 0] = x;
                        halos_dexm->halo_coords[3*count + 1] = y;
                        halos_dexm->halo_coords[3*count + 2] = z;
                        count++;
                    }
                }
            }
        }

        add_properties_cat(random_seed, redshift, halos_dexm);
        LOG_DEBUG("Found %llu DexM halos",halos_dexm->n_halos);

        if(flag_options->HALO_STOCHASTICITY){
            LOG_DEBUG("Finding halos below grid resolution %.3e",M_MIN);
            //First we construct a grid which corresponds to how much of a HII_DIM cell is covered by halos
            //  This is used in the sampler
            //we don't need the density field anymore so we reuse it
#pragma omp parallel private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<grid_dim; i++){
                    for (j=0; j<grid_dim; j++){
                        for (k=0; k<z_dim; k++){
                            *((float *)density_field + R_FFT_INDEX(i,j,k)) = in_halo[R_INDEX(i,j,k)] ? 1. : 0.;
                        }
                    }
                }
            }

            dft_r2c_cube(user_params->USE_FFTW_WISDOM, grid_dim, z_dim, user_params->N_THREADS, density_field);
            if (user_params->DIM != user_params->HII_DIM) {
                //the tophat filter here will smoothe the grid to HII_DIM
                filter_box(density_field, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
            }
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, density_field);

            float *halo_overlap_box = calloc(HII_TOT_NUM_PIXELS,sizeof(float));
            float f_pixel_factor = user_params->DIM/(float)user_params->HII_DIM;
            //Now downsample the highres grid to get the lowres version
#pragma omp parallel private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<HII_D_PARA; k++){
                            halo_overlap_box[HII_R_INDEX(i,j,k)] =
                            *((float *)density_field + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                            (unsigned long long)(j*f_pixel_factor+0.5),
                                                            (unsigned long long)(k*f_pixel_factor+0.5)))/TOT_NUM_PIXELS;
                            halo_overlap_box[HII_R_INDEX(i,j,k)] = fmin(fmax(halo_overlap_box[HII_R_INDEX(i,j,k)],0.),1); //cannot be below zero or above one
                        }
                    }
                }
            }

            stochastic_halofield(user_params, cosmo_params, astro_params, flag_options, random_seed,
                                redshift_desc, redshift, boxes->lowres_density, halo_overlap_box, halos_dexm, halos);

            //Here, halos_dexm is allocated in the C, so free it
            free_halo_field(halos_dexm);
            free(halos_dexm);
            free(halo_overlap_box);
        }

        LOG_DEBUG("Finished halo processing.");

        free(in_halo);
        free(halo_field);

        if(global_params.OPTIMIZE) {
            free(forbidden);
        }

        fftwf_free(density_field);
        fftwf_free(density_field_saved);

        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();

LOG_DEBUG("Finished halo cleanup.");
LOG_DEBUG("Found %llu Halos", halos->n_halos);
if (halos->n_halos > 3)
    LOG_DEBUG("Halo Masses: %e %e %e %e", halos->halo_masses[0], halos->halo_masses[1], halos->halo_masses[2], halos->halo_masses[3]);

    } // End of Try()
    Catch(status){
        return(status);
    }
    return(0);

}

// Function check_halo combines the original two functions overlap_halo and update_in_halo
// from the original 21cmFAST. Lots of redundant code, hence reduced into a single function
int check_halo(char * in_halo, UserParams *user_params, float R, int x, int y, int z, int check_type) {

    // if check_type == 1 (perform original overlap halo)
    //          Funtion OVERLAP_HALO checks if the would be halo with radius R
    //          and centered on (x,y,z) overlaps with a pre-existing halo
    // if check_type == 2 (perform original update in halo)
    //          Funtion UPDATE_IN_HALO takes in a box <in_halo> and flags all points
    //          which fall within radius R of (x,y,z).

    int x_curr, y_curr, z_curr, x_min, x_max, y_min, y_max, z_min, z_max, R_index;
    float Rsq_curr_index, xsq, xplussq, xminsq, ysq, yplussq, yminsq, zsq, zplussq, zminsq;
    int x_index, y_index, z_index;
    long long unsigned int curr_index;

    if(check_type==1) {
        // scale R to a effective overlap size, using R_OVERLAP_FACTOR
        R *= global_params.R_OVERLAP_FACTOR;
    }

    int grid_dim = user_params->DIM;
    int z_dim = D_PARA;

    // convert R to index units
    R_index = ceil(R/user_params->BOX_LEN*grid_dim);
    Rsq_curr_index = pow(R/user_params->BOX_LEN*grid_dim, 2); // convert to index

    // set parameter range
    x_min = x-R_index;
    x_max = x+R_index;
    y_min = y-R_index;
    y_max = y+R_index;
    z_min = z-R_index;
    z_max = z+R_index;
    // LOG_ULTRA_DEBUG("Starting check from (%d,%d,%d) to (%d,%d,%d)",x_min,y_min,z_min,x_max,y_max,z_max);

    for (x_curr=x_min; x_curr<=x_max; x_curr++){
        for (y_curr=y_min; y_curr<=y_max; y_curr++){
            for (z_curr=z_min; z_curr<=z_max; z_curr++){
                x_index = x_curr;
                y_index = y_curr;
                z_index = z_curr;
                // adjust if we are outside of the box
                if (x_index<0) {x_index += grid_dim;}
                else if (x_index>=grid_dim) {x_index -= grid_dim;}
                if (y_index<0) {y_index += grid_dim;}
                else if (y_index>=grid_dim) {y_index -= grid_dim;}
                if (z_index<0) {z_index += z_dim;}
                else if (z_index>=z_dim) {z_index -= z_dim;}

                curr_index = R_INDEX(x_index,y_index,z_index);
                // LOG_ULTRA_DEBUG("current point (%d,%d,%d) idx %d",x_curr,y_curr,z_curr,curr_index);

                if(check_type==1) {
                    if ( in_halo[curr_index] &&
                        pixel_in_halo(grid_dim,z_dim,x,x_index,y,y_index,z,z_index,Rsq_curr_index) ) {
                            // this pixel already belongs to a halo, and would want to become part of this halo as well
                            return 1;
                    }
                }
                else if(check_type==2) {
                    // now check
                    if (!in_halo[curr_index]){
                        if(pixel_in_halo(grid_dim,z_dim,x,x_index,y,y_index,z,z_index,Rsq_curr_index)) {
                            // we are within the sphere defined by R, so change flag in in_halo array
                            in_halo[curr_index] = 1;
                        }
                    }
                }
                else {
                    LOG_ERROR("check_type must be 1 or 2, got %d", check_type);
                    Throw(ValueError);
                }
            }
        }
    }
    //if check_type==1, we found no halos
    //if check_type==2, we don't use the return value
    return 0;
}

void init_halo_coords(HaloField *halos, long long unsigned int n_halos){
    // Minimise memory usage by only storing the halo mass and positions
    halos->n_halos = n_halos;
    halos->halo_masses = (float *)calloc(n_halos,sizeof(float));
    halos->halo_coords = (int *)calloc(3*n_halos,sizeof(int));

    halos->star_rng = (float *) calloc(n_halos,sizeof(float));
    halos->sfr_rng = (float *) calloc(n_halos,sizeof(float));
    halos->xray_rng = (float *) calloc(n_halos,sizeof(float));
}

void free_halo_field(HaloField *halos){
    LOG_DEBUG("Freeing HaloField instance.");
    free(halos->halo_masses);
    free(halos->halo_coords);
    free(halos->star_rng);
    free(halos->sfr_rng);
    free(halos->xray_rng);
    halos->n_halos = 0;
}

int pixel_in_halo(int grid_dim, int z_dim, int x, int x_index, int y, int y_index, int z, int z_index, float Rsq_curr_index ) {

    float xsq, xplussq, xminsq, ysq, yplussq, yminsq, zsq, zplussq, zminsq;

    // remember to check all reflections
    xsq = pow(x-x_index, 2);
    ysq = pow(y-y_index, 2);
    zsq = pow(z-z_index, 2);
    xplussq = pow(x-x_index+grid_dim, 2);
    yplussq = pow(y-y_index+grid_dim, 2);
    zplussq = pow(z-z_index+z_dim, 2);
    xminsq = pow(x-x_index-grid_dim, 2);
    yminsq = pow(y-y_index-grid_dim, 2);
    zminsq = pow(z-z_index-z_dim, 2);

    //This checks the center, 6 faces, 12 edges and 8 corners of the cell == 27 points
    //NOTE:The center check is not really necessary
    if(
       ( (Rsq_curr_index > (xsq + ysq + zsq)) || // AND pixel is within this halo
        (Rsq_curr_index > (xsq + ysq + zplussq)) ||
        (Rsq_curr_index > (xsq + ysq + zminsq)) ||

        (Rsq_curr_index > (xsq + yplussq + zsq)) ||
        (Rsq_curr_index > (xsq + yplussq + zplussq)) ||
        (Rsq_curr_index > (xsq + yplussq + zminsq)) ||

        (Rsq_curr_index > (xsq + yminsq + zsq)) ||
        (Rsq_curr_index > (xsq + yminsq + zplussq)) ||
        (Rsq_curr_index > (xsq + yminsq + zminsq)) ||


        (Rsq_curr_index > (xplussq + ysq + zsq)) ||
        (Rsq_curr_index > (xplussq + ysq + zplussq)) ||
        (Rsq_curr_index > (xplussq + ysq + zminsq)) ||

        (Rsq_curr_index > (xplussq + yplussq + zsq)) ||
        (Rsq_curr_index > (xplussq + yplussq + zplussq)) ||
        (Rsq_curr_index > (xplussq + yplussq + zminsq)) ||

        (Rsq_curr_index > (xplussq + yminsq + zsq)) ||
        (Rsq_curr_index > (xplussq + yminsq + zplussq)) ||
        (Rsq_curr_index > (xplussq + yminsq + zminsq)) ||

        (Rsq_curr_index > (xminsq + ysq + zsq)) ||
        (Rsq_curr_index > (xminsq + ysq + zplussq)) ||
        (Rsq_curr_index > (xminsq + ysq + zminsq)) ||

        (Rsq_curr_index > (xminsq + yplussq + zsq)) ||
        (Rsq_curr_index > (xminsq + yplussq + zplussq)) ||
        (Rsq_curr_index > (xminsq + yplussq + zminsq)) ||

        (Rsq_curr_index > (xminsq + yminsq + zsq)) ||
        (Rsq_curr_index > (xminsq + yminsq + zplussq)) ||
        (Rsq_curr_index > (xminsq + yminsq + zminsq))
        )
       ) {
        return(1);
    }
    else {
        return(0);
    }
}
