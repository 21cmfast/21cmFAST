
// Re-write of find_halos.c from the original 21cmFAST


// ComputeHaloField takes in a k_space box of the linear overdensity field
// and filters it on decreasing scales in order to find virialized halos.
// Virialized halos are defined according to the linear critical overdensity.
// ComputeHaloField outputs a cube with non-zero elements containing the Mass of
// the virialized halos

int check_halo(char * in_halo, struct UserParams *user_params, float R, int x, int y, int z, int check_type);
void init_halo_coords(struct HaloField *halos, int n_halos);
int pixel_in_halo(int grid_dim, int z_dim, int x, int x_index, int y, int y_index, int z, int z_index, float Rsq_curr_index );
void free_halo_field(struct HaloField *halos);

int ComputeHaloField(float redshift_prev, float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, int random_seed, struct HaloField * halos_prev, struct HaloField *halos) {

    int status;

    Try{ // This Try brackets the whole function, so we don't indent.

    //This happens if we are updating a halo field (no need to redo big halos)
    if(flag_options->HALO_STOCHASTICITY && redshift_prev > 0){
        LOG_DEBUG("Halo sampling switched on, bypassing halo finder to update %d halos...",halos_prev->n_halos);
        stochastic_halofield(user_params, cosmo_params, astro_params, flag_options, random_seed, redshift_prev, redshift, boxes->lowres_density, halos_prev, halos);
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
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_UF(user_params,cosmo_params);

        omp_set_num_threads(user_params->N_THREADS);

        fftwf_complex *density_field, *density_field_saved;

        float growth_factor, R, delta_m, dm, dlnm, M, Delta_R, delta_crit;
        double fgtrm, dfgtrm;
        unsigned long long ct;
        char *in_halo, *forbidden;
        int i,j,k,x,y,z,dn,n;
        int total_halo_num;
        float R_temp, x_temp, y_temp, z_temp, dummy, M_MIN;

        LOG_DEBUG("Begin Initialisation");

        // ***************** BEGIN INITIALIZATION ***************** //
        init_ps();

        growth_factor = dicke(redshift); // normalized to 1 at z=0
        delta_crit = Deltac; // for now set to spherical; check if we want elipsoidal later

        //store highly used parameters
        int grid_dim = user_params->DIM;
        int z_dim = D_PARA;
        int num_pixels = TOT_NUM_PIXELS;
        int k_num_pixels = KSPACE_NUM_PIXELS;

        //set minimum source mass
        M_MIN = minimum_source_mass(redshift, astro_params, flag_options);
        //if we use the sampler we want to stop at the HII cell mass
        if(flag_options->HALO_STOCHASTICITY)
            M_MIN = fmax(M_MIN,RtoM(L_FACTOR*user_params->BOX_LEN/user_params->HII_DIM));
        //otherwise we stop at the cell mass
        else
            M_MIN = fmax(M_MIN,RtoM(L_FACTOR*user_params->BOX_LEN/grid_dim));

        // allocate array for the k-space box
        density_field = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*k_num_pixels);
        density_field_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*k_num_pixels);

        // allocate memory for the boolean in_halo box
        in_halo = (char *) malloc(sizeof(char)*num_pixels);

        // initialize
        memset(in_halo, 0, sizeof(char)*num_pixels);

        if(global_params.OPTIMIZE) {
            forbidden = (char *) malloc(sizeof(char)*num_pixels);
        }

        unsigned long long int nhalo_threads[user_params->N_THREADS];
        unsigned long long int istart_threads[user_params->N_THREADS];
        //expected TOTAL halos in box from minimum source mass

        unsigned long long int arraysize_total = halos->buffer_size;
        unsigned long long int arraysize_local = arraysize_total / user_params->N_THREADS;

        if(LOG_LEVEL >= DEBUG_LEVEL){
            double Mmax_debug = 1e16;
            initialiseSigmaMInterpTable(M_MIN*0.9,Mmax_debug*1.1);
            double nhalo_debug = VOLUME * IntegratedNdM(growth_factor,log(M_MIN),log(Mmax_debug),log(Mmax_debug),0,0,user_params->HMF,0);
            //expected halos above minimum filter mass
            LOG_DEBUG("DexM: We expect %.2f Halos between Masses [%.2e,%.2e] (%.2e)",nhalo_debug,M_MIN,Mmax_debug, RHOcrit * cosmo_params->OMm * VOLUME / TOT_NUM_PIXELS);
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
        memcpy(density_field_saved, density_field, sizeof(fftwf_complex)*k_num_pixels);

        // ***************** END INITIALIZATION ***************** //

        LOG_DEBUG("Finalised Initialisation");

        // lets filter it now
        // set initial R value

        fgtrm=dfgtrm=0;
        n=0;
        Delta_R = L_FACTOR*2.*user_params->BOX_LEN/(grid_dim+0.0);

        total_halo_num = 0;
        R = MtoR(M_MIN*1.01); // one percent higher for rounding

        LOG_DEBUG("Prepare to filter to find halos");

        while (R < L_FACTOR*user_params->BOX_LEN)
            R*=global_params.DELTA_R_FACTOR;

        struct HaloField *halos_dexm;
        if(flag_options->HALO_STOCHASTICITY){
            //To save memory, we allocate the smaller (large mass) halofield here instead of using halos_prev
            halos_dexm = malloc(sizeof(struct HaloField));
        }
        else{
            //assign directly to the output field instead
            halos_dexm = halos;
        }

        float *halo_field = calloc(num_pixels, sizeof(float));

        while ((R > 0.5*Delta_R) && (RtoM(R) >= M_MIN)){ // filter until we get to half the pixel size or M_MIN
            M = RtoM(R);
            LOG_SUPER_DEBUG("while loop for finding halos: R = %f 0.5*Delta_R = %f RtoM(R)=%e M_MIN=%e", R, 0.5*Delta_R, M, M_MIN);

            //TODO: throw in an init loop
            if(global_params.DELTA_CRIT_MODE == 1){
                if(user_params->HMF==1) {
                    // use sheth tormen correction
                    delta_crit = growth_factor*sheth_delc(Deltac/growth_factor, sigma_z0(M));
                }
                else if(user_params->HMF==6) {
                    // use Delos 2023 flat barrier
                    delta_crit = 1.5;
                }
                else if(user_params->HMF!=0){
                    LOG_WARNING("Halo Finder: You have selected DELTA_CRIT_MODE==1 with HMF %d which does not have a barrier\
                                    , using EPS deltacrit = 1.68",user_params->HMF);
                }
            }

            // first let's check if virialized halos of this size are rare enough
            // that we don't have to worry about them (let's define 7 sigma away, as in Mesinger et al 05)
            if ((sigma_z0(M)*growth_factor*7.) < delta_crit){
                LOG_SUPER_DEBUG("Haloes too rare for M = %e! Skipping...", M);
                R /= global_params.DELTA_R_FACTOR;
                continue;
            }

            memcpy(density_field, density_field_saved, sizeof(fftwf_complex)*k_num_pixels);

            // now filter the box on scale R
            // 0 = top hat in real space, 1 = top hat in k space
            filter_box(density_field, 0, global_params.HALO_FILTER, R);

            // do the FFT to get delta_m box
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, grid_dim, z_dim, user_params->N_THREADS, density_field);

            // *****************  BEGIN OPTIMIZATION ***************** //
            // to optimize speed, if the filter size is large (switch to collapse fraction criteria later)
            if(global_params.OPTIMIZE) {
                if(M > global_params.OPTIMIZE_MIN_MASS) {
                    memset(forbidden, 0, sizeof(char)*num_pixels);
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
            dn=0;

            //TODO: Fix the race condition propertly to thread: it doesn't matter which thread finds the halo first
            //  but if two threads find a halo in the same region simultaneously (before the first one updates in_halo) some halos could double-up
            //checking for overlaps in new halos after this loop could work, but I would have to calculate distances between all new halos which sounds slow
            for (x=0; x<grid_dim; x++){
                for (y=0; y<grid_dim; y++){
                    for (z=0; z<z_dim; z++){
                        delta_m = *((float *)density_field + R_FFT_INDEX(x,y,z)) * growth_factor / num_pixels;

                        // if not within a larger halo, and radii don't overlap, update in_halo box
                        //TODO: something to remove the criticals (see above note)
                        // *****************  BEGIN OPTIMIZATION ***************** //
                        if(global_params.OPTIMIZE && (M > global_params.OPTIMIZE_MIN_MASS)) {
                            if ( (delta_m > delta_crit) && !forbidden[R_INDEX(x,y,z)]){
                                check_halo(in_halo, user_params, R, x,y,z,2); // flag the pixels contained within this halo
                                check_halo(forbidden, user_params, (1.+global_params.R_OVERLAP_FACTOR)*R, x,y,z,2); // flag the pixels contained within this halo

                                halo_field[R_INDEX(x,y,z)] = M;

                                dn++; // keep track of the number of halos
                                n++;
                                total_halo_num++;
                            }
                        }
                        // *****************  END OPTIMIZATION ***************** //
                        else {
                            if ((delta_m > delta_crit) && !in_halo[R_INDEX(x,y,z)] && !check_halo(in_halo, user_params, R, x,y,z,1)){ // we found us a "new" halo!
                                // LOG_ULTRA_DEBUG("Halo found at (%d,%d,%d), delta = %.4f",x,y,z,delta_m);
                                check_halo(in_halo, user_params, R, x,y,z,2); // flag the pixels contained within this halo

                                halo_field[R_INDEX(x,y,z)] = M;

                                dn++; // keep track of the number of halos
                                n++;
                                total_halo_num++;
                            }
                        }
                    }
                }
            }

            LOG_SUPER_DEBUG("n_halo = %d, total = %d , D = %.3f, delcrit = %.3f", dn, n, growth_factor, delta_crit);

            R /= global_params.DELTA_R_FACTOR;
        }

        LOG_DEBUG("Obtained halo masses and positions, now saving to HaloField struct.");

        //Allocate the Halo Mass and Coordinate Fields (non-wrapper structure)
        if(flag_options->HALO_STOCHASTICITY)
            init_halo_coords(halos_dexm, total_halo_num);

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

        //add halo properties for ionisation TODO: add a flag
        add_properties_cat(user_params, cosmo_params, astro_params, flag_options, random_seed, redshift, halos_dexm);
        LOG_DEBUG("Found %d DexM halos",total_halo_num);

        if(flag_options->HALO_STOCHASTICITY){
            LOG_DEBUG("Finding halos below grid resolution %.3e",M_MIN);
            stochastic_halofield(user_params, cosmo_params, astro_params, flag_options, random_seed, redshift_prev, redshift, boxes->lowres_density, halos_dexm, halos);

            //Here, halos_dexm is allocated in the C, so free it
            free_halo_field(halos_dexm);
            free(halos_dexm);
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
LOG_DEBUG("Found %d Halos", halos->n_halos);
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
int check_halo(char * in_halo, struct UserParams *user_params, float R, int x, int y, int z, int check_type) {

    // if check_type == 1 (perform original overlap halo)
    //          Funtion OVERLAP_HALO checks if the would be halo with radius R
    //          and centered on (x,y,z) overlaps with a pre-existing halo
    // if check_type == 2 (perform original update in halo)
    //          Funtion UPDATE_IN_HALO takes in a box <in_halo> and flags all points
    //          which fall within radius R of (x,y,z).

    int x_curr, y_curr, z_curr, x_min, x_max, y_min, y_max, z_min, z_max, R_index;
    float Rsq_curr_index, xsq, xplussq, xminsq, ysq, yplussq, yminsq, zsq, zplussq, zminsq;
    int x_index, y_index, z_index;
    int curr_index;

    if(check_type==1) {
        // scale R to a effective overlap size, using R_OVERLAP_FACTOR
        R *= global_params.R_OVERLAP_FACTOR;
    }

    int grid_dim = user_params->DIM;
    int z_dim = D_PARA;
    int num_pixels = TOT_NUM_PIXELS;

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
    if(check_type==1) {
        return 0;
    }
}

void init_halo_coords(struct HaloField *halos, int n_halos){
    // Minimise memory usage by only storing the halo mass and positions
    halos->n_halos = n_halos;
    halos->halo_masses = (float *)calloc(n_halos,sizeof(float));
    halos->halo_coords = (int *)calloc(3*n_halos,sizeof(int));

    halos->star_rng = (float *) calloc(n_halos,sizeof(float));
    halos->sfr_rng = (float *) calloc(n_halos,sizeof(float));
}

void free_halo_field(struct HaloField *halos){
    LOG_DEBUG("Freeing HaloField instance.");
    free(halos->halo_masses);
    free(halos->halo_coords);
    free(halos->star_rng);
    free(halos->sfr_rng);
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
