
// Re-write of find_halos.c from the original 21cmFAST


// ComputeHaloField takes in a k_space box of the linear overdensity field
// and filters it on decreasing scales in order to find virialized halos.
// Virialized halos are defined according to the linear critical overdensity.
// ComputeHaloField outputs a cube with non-zero elements containing the Mass of
// the virialized halos

int check_halo(char * in_halo, struct UserParams *user_params, int res_flag, float R, int x, int y, int z, int check_type);
void init_halo_coords(struct HaloField *halos, int n_halos);
int pixel_in_halo(int grid_dim, int x, int x_index, int y, int y_index, int z, int z_index, float Rsq_curr_index );
void free_halo_field(struct HaloField *halos);
void init_hmf(struct HaloField *halos);
void trim_hmf(struct HaloField *halos);


int ComputeHaloField(float redshift_prev, float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, int random_seed, struct HaloField * halos_prev, struct HaloField *halos) {

    int status;

    Try{ // This Try brackets the whole function, so we don't indent.

    //This happens if we are updating a halo field (no need to redo big halos)
    if(flag_options->HALO_STOCHASTICITY && !(halos->first_box)){
        LOG_DEBUG("Halo sampling switched on, bypassing halo finder to update %d halos...",halos_prev->n_halos);
        stochastic_halofield(user_params, cosmo_params, astro_params, flag_options, random_seed, redshift_prev, redshift, boxes->lowres_density, halos_prev, halos);
        //unfortunately we cannot yet free a HaloField in python
        //TODO: implement proper allocation / freeing that guarantees halos_prev is allocated here
        //  Currently it is a dummy for non-first boxes
        free_halo_field(halos_prev);
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
        char filename[80], *in_halo, *forbidden;
        int i,j,k,x,y,z,dn,n,counter;
        int total_halo_num;
        float R_temp, x_temp, y_temp, z_temp, dummy, M_MIN;

LOG_DEBUG("Begin Initialisation");

        counter = 0;

        // ***************** BEGIN INITIALIZATION ***************** //
        init_ps();

        growth_factor = dicke(redshift); // normalized to 1 at z=0
        delta_crit = Deltac; // for now set to spherical; check if we want elipsoidal later

        //when using the halo sampler, this finds the large halos on HII_DIM, then the sampler runs below HII_DIM
        //otherwise it finds halos on DIM
        //I rename the flag here for clarity and in case we want some other conditions later
        int res_flag = flag_options->HALO_STOCHASTICITY ? 1 : 0;
        int grid_dim = res_flag ? user_params->HII_DIM : user_params->DIM;
        int num_pixels = res_flag ? HII_TOT_NUM_PIXELS : TOT_NUM_PIXELS;
        int k_num_pixels = res_flag ? HII_KSPACE_NUM_PIXELS : KSPACE_NUM_PIXELS;

        //set minimum source mass
        M_MIN = minimum_source_mass(redshift, astro_params, flag_options);

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

#pragma omp parallel shared(boxes,density_field) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<grid_dim; i++){
                for (j=0; j<grid_dim; j++){
                    for (k=0; k<grid_dim; k++){
                        //TODO: I want a cleaner way to approach the indexing with high/low res options
                        if(res_flag)
                            *((float *)density_field + HII_R_FFT_INDEX(i,j,k)) = *((float *)boxes->lowres_density + HII_R_INDEX(i,j,k));
                        else
                            *((float *)density_field + R_FFT_INDEX(i,j,k)) = *((float *)boxes->hires_density + R_INDEX(i,j,k));
                    }
                }
            }
        }

        dft_r2c_cube(user_params->USE_FFTW_WISDOM, grid_dim, user_params->N_THREADS, density_field);

        // save a copy of the k-space density field
        memcpy(density_field_saved, density_field, sizeof(fftwf_complex)*k_num_pixels);

        // ***************** END INITIALIZATION ***************** //

LOG_DEBUG("Finalised Initialisation");

        // lets filter it now
        // set initial R value
        R = MtoR(M_MIN*1.01); // one percent higher for rounding

LOG_DEBUG("Prepare to filter to find halos");

        while (R < L_FACTOR*user_params->BOX_LEN)
            R*=global_params.DELTA_R_FACTOR;

        fgtrm=dfgtrm=0;
        n=0;
        Delta_R = L_FACTOR*2.*user_params->BOX_LEN/(grid_dim+0.0);

        total_halo_num = 0;


        // This uses more memory than absolutely necessary, but is fastest.
        //TODO: I could just use halos_prev? which would make it exist in the wrapper
        //  but would take some tinkering
        struct HaloField *halos_large = malloc(sizeof(struct HaloField));
        halos_large->n_halos = 0;
        init_hmf(halos_large);
        float *halo_field = calloc(num_pixels, sizeof(float));

        while ((R > 0.5*Delta_R) && (RtoM(R) >= M_MIN)){ // filter until we get to half the pixel size or M_MIN

LOG_ULTRA_DEBUG("while loop for finding halos: R = %f 0.5*Delta_R = %f RtoM(R)=%e M_MIN=%e", R, 0.5*Delta_R, RtoM(R), M_MIN);

            M = RtoM(R);
            if(global_params.DELTA_CRIT_MODE == 1 && (user_params->HMF>0 && user_params->HMF<4)){
                if(user_params->HMF==1) {
                    // use sheth tormen correction
                    delta_crit = growth_factor*sheth_delc(Deltac/growth_factor, sigma_z0(M));
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
            filter_box(density_field, res_flag, global_params.HALO_FILTER, R);

            // do the FFT to get delta_m box
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, grid_dim, user_params->N_THREADS, density_field);

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
                                for (z=0; z<grid_dim; z++){
                                    if(res_flag)
                                        halo_buf = halo_field[HII_R_INDEX(x,y,z)];
                                    else
                                        halo_buf = halo_field[R_INDEX(x,y,z)];
                                    if(halo_buf > 0.) {
                                        R_temp = MtoR(halo_buf);
                                        check_halo(forbidden, user_params, res_flag, R_temp+global_params.R_OVERLAP_FACTOR*R, x,y,z,2);
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
            
//TODO: Fix the race condition: it doesn't matter which thread finds the halo first, but if two threads find a halo in the same region
//simultaneously (before the first one updates in_halo) some halos could double-up
//putting the conditional in a critical part would undo most of the threading benefit, think of something else
//checking for overlaps in new halos after this loop could work, but I would have to calculate distances between all new halos which sounds slow
//for now, I think it's good enough to set the static schedule so that the race condition only matters for large halos in low-res grids and small boxes.
#pragma omp parallel shared(boxes,density_field,in_halo,forbidden,halo_field,growth_factor,M,delta_crit) private(x,y,z,delta_m) num_threads(user_params->N_THREADS) reduction(+: dn,n,total_halo_num)
            {
                unsigned long long index;
#pragma omp for schedule(static)
                for (x=0; x<grid_dim; x++){
                    for (y=0; y<grid_dim; y++){
                        for (z=0; z<grid_dim; z++){
                            if(res_flag)
                                index = HII_R_FFT_INDEX(x,y,z);
                            else
                                index = R_FFT_INDEX(x,y,z);
                            
                            delta_m = *((float *)density_field + index) * growth_factor / num_pixels;

                            // if not within a larger halo, and radii don't overlap, update in_halo box
                            // *****************  BEGIN OPTIMIZATION ***************** //
                            if(global_params.OPTIMIZE && (M > global_params.OPTIMIZE_MIN_MASS)) {
                                if ( (delta_m > delta_crit) && !forbidden[index]){
                                    check_halo(in_halo, user_params, res_flag, R, x,y,z,2); // flag the pixels contained within this halo

                                    halo_field[index] = M;

                                    dn++; // keep track of the number of halos
                                    n++;
                                    total_halo_num++;
                                }
                            }
                            // *****************  END OPTIMIZATION ***************** //
                            else {
                                if ((delta_m > delta_crit) && !in_halo[index] && !check_halo(in_halo, user_params, res_flag, R, x,y,z,1)){ // we found us a "new" halo!

                                    check_halo(in_halo, user_params, res_flag, R, x,y,z,2); // flag the pixels contained within this halo

                                    halo_field[index] = M;

                                    dn++; // keep track of the number of halos
                                    n++;
                                    total_halo_num++;
                                }
                            }
                        }
                    }
                }
            }

            LOG_ULTRA_DEBUG("n_halo = %d, total = %d , D = %.3f, delcrit = %.3f", dn, n, growth_factor, delta_crit);

            if (dn > 0){
                // now lets keep the mass functions (FgrtR)
                fgtrm += M/(RHOcrit*cosmo_params->OMm)*dn/VOLUME;
                dfgtrm += pow(M/(RHOcrit*cosmo_params->OMm)*sqrt(dn)/VOLUME, 2);

                // and the dndlnm files
                dlnm = log(RtoM(global_params.DELTA_R_FACTOR*R)) - log(M);

                if (halos_large->n_mass_bins == halos_large->max_n_mass_bins){
                    // We've gone past the limit.
                    LOG_WARNING("Code has required more than 100 mass bins, and will no longer store masses.");
                }
                else{
                    halos_large->mass_bins[halos_large->n_mass_bins] = M;
                    halos_large->fgtrm[halos_large->n_mass_bins] = fgtrm;
                    halos_large->sqrt_dfgtrm[halos_large->n_mass_bins] = sqrt(dfgtrm);
                    halos_large->dndlm[halos_large->n_mass_bins] = dn/VOLUME/dlnm;
                    halos_large->sqrtdn_dlm[halos_large->n_mass_bins] = sqrt(dn)/VOLUME/dlnm;
                    halos_large->n_mass_bins++;
                }
            }

            R /= global_params.DELTA_R_FACTOR;
        }

        LOG_DEBUG("Obtained halo masses and positions, now saving to HaloField struct.");

        // Trim the mass function entries
        trim_hmf(halos_large);

        // Initialize the halo co-ordinate and mass arrays.
        init_halo_coords(halos_large, total_halo_num);

        int istart_local[user_params->N_THREADS];
        memset(istart_local,0,sizeof(int)*user_params->N_THREADS);
        float * local_masses;
        int * local_coords;
        int threadnum;

//I would expect the number of halos to be much less than the number of cells so defining local arrays here
//and then concatenating shouldn't increase memory too much
#pragma omp parallel shared(halos_large,halo_field,istart_local) private(x,y,z,counter,local_coords,local_masses,threadnum,i) num_threads(user_params->N_THREADS)
        {
            //TODO: find a way to allocate less, based on the local number of halos
            //this can be done in the previous loop if we can guarantee the scheduler allocates the same chunks
            threadnum = omp_get_thread_num();
            local_masses = calloc(total_halo_num,sizeof(float));
            local_coords = calloc(total_halo_num*3,sizeof(int));
            // reuse counter as its no longer needed
            counter = 0;
            float halo_buf = 0;
#pragma omp for
            for (x=0; x<grid_dim; x++){
                for (y=0; y<grid_dim; y++){
                    for (z=0; z<grid_dim; z++){
                        if(res_flag)
                            halo_buf = halo_field[HII_R_INDEX(x,y,z)];
                        else
                            halo_buf = halo_field[R_INDEX(x,y,z)];
                        if(halo_buf > 0.) {
                            local_masses[counter] = halo_buf;
                            local_coords[0 + counter*3] = x;
                            local_coords[1 + counter*3] = y;
                            local_coords[2 + counter*3] = z;
                            counter++;
                        }
                    }
                }
            }

//this loop exectuted on all threads, we need the start index of each local array
//i[0] == 0, i[1] == n_0, i[2] == n_0 + n_1 etc...
            for(i=user_params->N_THREADS-1;i>threadnum;i--){
#pragma omp atomic update
                istart_local[i] += counter;
            }
//we need each thread to be done here before copying the data
#pragma omp barrier

            LOG_SUPER_DEBUG("Thread %d has %d of %d halos, concatenating (starting at %d)...",threadnum,counter,total_halo_num,istart_local[threadnum]);
            
            //copy each local array into the struct
            memcpy(halos_large->halo_masses + istart_local[threadnum],local_masses,counter*sizeof(float));
            memcpy(halos_large->halo_coords + istart_local[threadnum]*3,local_coords,counter*sizeof(int)*3);

            free(local_coords);
            free(local_masses);
        }
        //add halo properties for ionisation TODO: add a flag
        add_properties_cat(user_params, cosmo_params, astro_params, flag_options, random_seed, redshift, halos_large);
        if(total_halo_num>=3)
        LOG_DEBUG("%d Large halos [%.3e,%.3e,%.3e]",halos_large->n_halos,halos_large->halo_masses[0],halos_large->halo_masses[1],halos_large->halo_masses[2]);
        
        //TODO: this probably should be in the wrapper but it's hard with each class only having one compute function
        //TODO: move later on so its possible to get halos above HII_DIM with DexM then sample below HII_DIM
        if(flag_options->HALO_STOCHASTICITY){
            LOG_DEBUG("Finding halos below grid resolution %.3e",RtoM(0.5*Delta_R));
            stochastic_halofield(user_params, cosmo_params, astro_params, flag_options, random_seed, redshift_prev, redshift, boxes->lowres_density, halos_large, halos);
            //stochastic_halofield allocates new memory for all the halos so we can free everything
            free_halo_field(halos_large);
        }
        else{
            //assign to output struct
            //I could pass in **HaloField to do this in one step but this should be fine
            //  since the fields are not allocated in the wrapper
            init_hmf(halos);
            halos->mass_bins = halos_large->mass_bins;
            halos->fgtrm = halos_large->fgtrm;
            halos->sqrt_dfgtrm = halos_large->fgtrm;
            halos->dndlm = halos_large->dndlm;
            halos->sqrtdn_dlm = halos_large->sqrtdn_dlm;
            halos->n_mass_bins = halos_large->n_mass_bins;

            halos->n_halos = halos_large->n_halos;
            halos->halo_masses = halos_large->halo_masses;
            halos->stellar_masses = halos_large->stellar_masses;
            halos->halo_sfr = halos_large->halo_sfr;
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
int check_halo(char * in_halo, struct UserParams *user_params, int res_flag, float R, int x, int y, int z, int check_type) {

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

    int grid_dim = res_flag ? user_params->HII_DIM : user_params->DIM;
    int num_pixels = res_flag ? HII_TOT_NUM_PIXELS : TOT_NUM_PIXELS;
    int k_num_pixels = res_flag ? HII_KSPACE_NUM_PIXELS : KSPACE_NUM_PIXELS;

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
                if (z_index<0) {z_index += grid_dim;}
                else if (z_index>=grid_dim) {z_index -= grid_dim;}

                if(res_flag)
                    curr_index = HII_R_INDEX(x_index,y_index,z_index);
                else
                    curr_index = R_INDEX(x_index,y_index,z_index);

                if(check_type==1) {
                    if ( in_halo[curr_index] &&
                        pixel_in_halo(grid_dim,x,x_index,y,y_index,z,z_index,Rsq_curr_index) ) {
                            // this pixel already belongs to a halo, and would want to become part of this halo as well
                            return 1;
                    }
                }
                else if(check_type==2) {
                    // now check
                    if (!in_halo[curr_index]){
                        if(pixel_in_halo(grid_dim,x,x_index,y,y_index,z,z_index,Rsq_curr_index)) {
                            // we are within the sphere defined by R, so change flag in in_halo array
//does the race condition matter here? if any thread marks the pixel it should be 1
#pragma omp atomic write
                            in_halo[curr_index] = 1;
                        }
                    }
                }
                else {
                    LOG_ERROR("check_type must be 1 or 2, got %d", check_type);
                    Throw ValueError;
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
    int i;
    halos->n_halos = n_halos;
    halos->halo_masses = (float *)calloc(n_halos,sizeof(float));
    halos->halo_coords = (int *)calloc(3*n_halos,sizeof(int));
    
    halos->stellar_masses = (float *) calloc(n_halos,sizeof(float));
    halos->halo_sfr = (float *) calloc(n_halos,sizeof(float));
}

void free_halo_field(struct HaloField *halos){
    LOG_DEBUG("Freeing HaloField instance.");
    free(halos->halo_masses);
    free(halos->halo_coords);
    free(halos->stellar_masses);
    free(halos->halo_sfr);
    halos->n_halos = 0;

    free(halos->mass_bins);
    free(halos->fgtrm);
    free(halos->sqrt_dfgtrm);
    free(halos->dndlm);
    free(halos->sqrtdn_dlm);
    halos->n_mass_bins = 0;
}
void init_hmf(struct HaloField *halos){
    // Initalize mass function array with an abitrary large number of elements.
    // We will trim it later.
    halos->max_n_mass_bins = 100;
    halos->mass_bins = (float *) malloc(sizeof(float) * halos->max_n_mass_bins);
    halos->fgtrm = (float *) malloc(sizeof(float) * halos->max_n_mass_bins);
    halos->sqrt_dfgtrm = (float *) malloc(sizeof(float) * halos->max_n_mass_bins);
    halos->dndlm = (float *) malloc(sizeof(float) * halos->max_n_mass_bins);
    halos->sqrtdn_dlm = (float *) malloc(sizeof(float) * halos->max_n_mass_bins);
    halos->n_mass_bins = 0;
}

void trim_hmf(struct HaloField *halos){
    // Trim hmf arrays down to actual number of mass bins.
    if (halos->n_mass_bins > 0){
        halos->mass_bins = (float *) realloc(halos->mass_bins, sizeof(float) * halos->n_mass_bins);
        halos->fgtrm = (float *) realloc(halos->fgtrm, sizeof(float)  * halos->n_mass_bins);
        halos->sqrt_dfgtrm = (float *) realloc(halos->sqrt_dfgtrm, sizeof(float)  * halos->n_mass_bins);
        halos->dndlm = (float *) realloc(halos->dndlm, sizeof(float)  * halos->n_mass_bins);
        halos->sqrtdn_dlm = (float *) realloc(halos->sqrtdn_dlm, sizeof(float)  * halos->n_mass_bins);
    }
}

int pixel_in_halo(int grid_dim, int x, int x_index, int y, int y_index, int z, int z_index, float Rsq_curr_index ) {

    float xsq, xplussq, xminsq, ysq, yplussq, yminsq, zsq, zplussq, zminsq;

    // remember to check all reflections
    xsq = pow(x-x_index, 2);
    ysq = pow(y-y_index, 2);
    zsq = pow(z-z_index, 2);
    xplussq = pow(x-x_index+grid_dim, 2);
    yplussq = pow(y-y_index+grid_dim, 2);
    zplussq = pow(z-z_index+grid_dim, 2);
    xminsq = pow(x-x_index-grid_dim, 2);
    yminsq = pow(y-y_index-grid_dim, 2);
    zminsq = pow(z-z_index-grid_dim, 2);

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
