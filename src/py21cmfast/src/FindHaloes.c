
// Re-write of find_halos.c from the original 21cmFAST


// ComputeHaloField takes in a k_space box of the linear overdensity field
// and filters it on decreasing scales in order to find virialized halos.
// Virialized halos are defined according to the linear critical overdensity.
// ComputeHaloField outputs a cube with non-zero elements containing the Mass of
// the virialized halos

int check_halo(char * in_halo, struct UserParams *user_params, float R, int x, int y, int z, int check_type);
int pixel_in_halo(struct UserParams *user_params, int x, int x_index, int y, int y_index, int z, int z_index, float Rsq_curr_index );
void init_halo_coords(struct HaloField *halos, int n_halos);
void free_halo_field(struct HaloField *halos);
void init_hmf(struct HaloField *halos);
void trim_hmf(struct HaloField *halos);


int ComputeHaloField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, struct HaloField *halos) {

    int status;

    Try{ // This Try brackets the whole function, so we don't indent.

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

        // ***************** END INITIALIZATION ***************** //
        init_ps();

        growth_factor = dicke(redshift); // normalized to 1 at z=0
        delta_crit = Deltac; // for now set to spherical; check if we want elipsoidal later

        //set the minimum source mass
        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
            M_MIN = astro_params->M_TURN;
        }
        else {
            if(flag_options->M_MIN_in_Mass) {
                M_MIN = (astro_params->M_TURN);
            }
            else {
                //set the minimum source mass
                if (astro_params->ION_Tvir_MIN < 9.99999e3) { // neutral IGM
                    M_MIN = TtoM(redshift, astro_params->ION_Tvir_MIN, 1.22);
                }
                else { // ionized IGM
                    M_MIN = TtoM(redshift, astro_params->ION_Tvir_MIN, 0.6);
                }
            }
        }

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

#pragma omp parallel shared(boxes,density_field) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->DIM; i++){
                for (j=0; j<user_params->DIM; j++){
                    for (k=0; k<D_PARA; k++){
                        *((float *)density_field + R_FFT_INDEX(i,j,k)) = *((float *)boxes->hires_density + R_INDEX(i,j,k));
                    }
                }
            }
        }

        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, density_field);

        // save a copy of the k-space density field
        memcpy(density_field_saved, density_field, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);


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
        Delta_R = L_FACTOR*2.*user_params->BOX_LEN/(user_params->DIM+0.0);

        total_halo_num = 0;


        // This uses more memory than absolutely necessary, but is fastest.
        init_hmf(halos);
        float *halo_field = calloc(TOT_NUM_PIXELS, sizeof(float));


        while ((R > 0.5*Delta_R) && (RtoM(R) >= M_MIN)){ // filter until we get to half the pixel size or M_MIN

LOG_ULTRA_DEBUG("while loop for finding halos: R = %f 0.5*Delta_R = %f RtoM(R)=%f M_MIN=%f", R, 0.5*Delta_R, RtoM(R), M_MIN);

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
LOG_DEBUG("Haloes too rare for M = %e! Skipping...", M);
                R /= global_params.DELTA_R_FACTOR;
                continue;
            }

            memcpy(density_field, density_field_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

            // now filter the box on scale R
            // 0 = top hat in real space, 1 = top hat in k space
            filter_box(density_field, 0, global_params.HALO_FILTER, R);

            // do the FFT to get delta_m box
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, density_field);

            // *****************  BEGIN OPTIMIZATION ***************** //
            // to optimize speed, if the filter size is large (switch to collapse fraction criteria later)
            if(global_params.OPTIMIZE) {
                if(M > global_params.OPTIMIZE_MIN_MASS) {
                    memset(forbidden, 0, sizeof(char)*TOT_NUM_PIXELS);
                    // now go through the list of existing halos and paint on the no-go region onto <forbidden>

                    for (x=0; x<user_params->DIM; x++){
                        for (y=0; y<user_params->DIM; y++){
                            for (z=0; z<D_PARA; z++){
                                if(halo_field[R_INDEX(x,y,z)] > 0.) {
                                    R_temp = MtoR(halo_field[R_INDEX(x,y,z)]);
                                    check_halo(forbidden, user_params, R_temp+global_params.R_OVERLAP_FACTOR*R, x,y,z,2);
                                }
                            }
                        }
                    }
                }
            }
            // *****************  END OPTIMIZATION ***************** //

            // now lets scroll through the box, flagging all pixels with delta_m > delta_crit
            dn=0;
            for (x=0; x<user_params->DIM; x++){
                for (y=0; y<user_params->DIM; y++){
                    for (z=0; z<D_PARA; z++){
                        delta_m = *((float *)density_field + R_FFT_INDEX(x,y,z)) * growth_factor / (float)TOT_NUM_PIXELS;       //since we didn't multiply k-space cube by V/N, we divide by 1/N here
                        // if not within a larger halo, and radii don't overlap, update in_halo box
                        // *****************  BEGIN OPTIMIZATION ***************** //
                        if(global_params.OPTIMIZE) {
                            if(M > global_params.OPTIMIZE_MIN_MASS) {
                                if ( (delta_m > delta_crit) && !forbidden[R_INDEX(x,y,z)]){

                                    check_halo(in_halo, user_params, R, x,y,z,2); // flag the pixels contained within this halo
                                    check_halo(forbidden, user_params, (1.+global_params.R_OVERLAP_FACTOR)*R, x,y,z,2); // flag the pixels contained within this halo

                                    halo_field[R_INDEX(x,y,z)] = M;

                                    dn++; // keep track of the number of halos
                                    n++;
                                    total_halo_num++;
                                }
                            }
                        }
                        // *****************  END OPTIMIZATION ***************** //
                        else {
                            if ((delta_m > delta_crit) && !in_halo[R_INDEX(x,y,z)] && !check_halo(in_halo, user_params, R, x,y,z,1)){ // we found us a "new" halo!

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

            if (dn > 0){
                // now lets keep the mass functions (FgrtR)
                fgtrm += M/(RHOcrit*cosmo_params->OMm)*dn/VOLUME;
                dfgtrm += pow(M/(RHOcrit*cosmo_params->OMm)*sqrt(dn)/VOLUME, 2);

                // and the dndlnm files
                dlnm = log(RtoM(global_params.DELTA_R_FACTOR*R)) - log(M);

                if (halos->n_mass_bins == halos->max_n_mass_bins){
                    // We've gone past the limit.
                    LOG_WARNING("Code has required more than 100 mass bins, and will no longer store masses.");
                }
                else{
                    halos->mass_bins[halos->n_mass_bins] = M;
                    halos->fgtrm[halos->n_mass_bins] = fgtrm;
                    halos->sqrt_dfgtrm[halos->n_mass_bins] = sqrt(dfgtrm);
                    halos->dndlm[halos->n_mass_bins] = dn/VOLUME/dlnm;
                    halos->sqrtdn_dlm[halos->n_mass_bins] = sqrt(dn)/VOLUME/dlnm;
                    halos->n_mass_bins++;
                }
            }

            R /= global_params.DELTA_R_FACTOR;
        }

LOG_DEBUG("Obtained halo masses and positions, now saving to HaloField struct.");

        // Trim the mass function entries
        trim_hmf(halos);

        // Initialize the halo co-ordinate and mass arrays.
        init_halo_coords(halos, total_halo_num);

        // reuse counter as its no longer needed
        counter = 0;

        for (x=0; x<user_params->DIM; x++){
            for (y=0; y<user_params->DIM; y++){
                for (z=0; z<D_PARA; z++){
                    if(halo_field[R_INDEX(x,y,z)] > 0.) {
                        halos->halo_masses[counter] = halo_field[R_INDEX(x,y,z)];
                        halos->halo_coords[0 + counter*3] = x;
                        halos->halo_coords[1 + counter*3] = y;
                        halos->halo_coords[2 + counter*3] = z;
                        counter++;
                    }
                }
            }
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

    if(check_type==1) {
        // scale R to a effective overlap size, using R_OVERLAP_FACTOR
        R *= global_params.R_OVERLAP_FACTOR;
    }

    // convert R to index units
    R_index = ceil(R/user_params->BOX_LEN*user_params->DIM);
    Rsq_curr_index = pow(R/user_params->BOX_LEN*user_params->DIM, 2); // convert to index

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
                if (x_index<0) {x_index += user_params->DIM;}
                else if (x_index>=user_params->DIM) {x_index -= user_params->DIM;}
                if (y_index<0) {y_index += user_params->DIM;}
                else if (y_index>=user_params->DIM) {y_index -= user_params->DIM;}
                if (z_index<0) {z_index += D_PARA;}
                else if (z_index>=D_PARA) {z_index -= D_PARA;}
                if(check_type==1) {
                    if ( in_halo[R_INDEX(x_index, y_index, z_index)] &&
                        pixel_in_halo(user_params,x,x_index,y,y_index,z,z_index,Rsq_curr_index) ) {
                            // this pixel already belongs to a halo, and would want to become part of this halo as well
                            return 1;
                    }
                }
                else if(check_type==2) {
                    // now check
                    if (!in_halo[R_INDEX(x_index, y_index, z_index)]){
                        if(pixel_in_halo(user_params,x,x_index,y,y_index,z,z_index,Rsq_curr_index)) {
                            // we are within the sphere defined by R, so change flag in in_halo array
                            in_halo[R_INDEX(x_index, y_index, z_index)] = 1;
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
}

void free_halo_field(struct HaloField *halos){
    LOG_DEBUG("Freeing HaloField instance.");
    free(halos->halo_masses);
    free(halos->halo_coords);
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

int pixel_in_halo(struct UserParams *user_params, int x, int x_index, int y, int y_index, int z, int z_index, float Rsq_curr_index ) {

    float xsq, xplussq, xminsq, ysq, yplussq, yminsq, zsq, zplussq, zminsq;

    // remember to check all reflections
    xsq = pow(x-x_index, 2);
    ysq = pow(y-y_index, 2);
    zsq = pow(z-z_index, 2);
    xplussq = pow(x-x_index+user_params->DIM, 2);
    yplussq = pow(y-y_index+user_params->DIM, 2);
    zplussq = pow(z-z_index+D_PARA, 2);
    xminsq = pow(x-x_index-user_params->DIM, 2);
    yminsq = pow(y-y_index-user_params->DIM, 2);
    zminsq = pow(z-z_index-D_PARA, 2);

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
