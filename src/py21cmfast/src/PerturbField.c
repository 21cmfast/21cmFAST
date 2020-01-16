
// Re-write of perturb_field.c for being accessible within the MCMC

int ComputePerturbField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes, struct PerturbedField *perturbed_field) {

    /*
     USAGE: perturb_field <REDSHIFT>

     PROGRAM PERTURB_FIELD uses the first-order Langragian displacement field to move the masses in the cells of the density field.
     The high-res density field is extrapolated to some high-redshift (INITIAL_REDSHIFT in ANAL_PARAMS.H), then uses the zeldovich approximation
     to move the grid "particles" onto the lower-res grid we use for the maps.  Then we recalculate the velocity fields on the perturbed grid.
     */

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    omp_set_num_threads(user_params->N_THREADS);
    fftwf_init_threads();
    fftwf_plan_with_nthreads(user_params->N_THREADS);
    fftwf_cleanup_threads();

    char wisdom_filename[500];

//    fftwf_complex *HIRES_density_perturb, *HIRES_density_perturb_saved; //cw
    fftwf_complex *LOWRES_density_perturb, *LOWRES_density_perturb_saved;
    fftwf_plan plan;

    float growth_factor, displacement_factor_2LPT, init_growth_factor, init_displacement_factor_2LPT, xf, yf, zf;
    float mass_factor, dDdt, f_pixel_factor, velocity_displacement_factor, velocity_displacement_factor_2LPT;
    unsigned long long ct, HII_i, HII_j, HII_k;
    int i,j,k, xi, yi, zi;
    double ave_delta, new_ave_delta;
    float k_x, k_y, k_z, k_sq, dDdt_over_D;
    int n_x, n_y, n_z;

    // ***************   BEGIN INITIALIZATION   ************************** //

    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        fprintf(stderr, "perturb_field.c: WARNING: Resolution is likely too low for accurate evolved density fields\n \
                It Is recommended that you either increase the resolution (DIM/Box_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }

    growth_factor = dicke(redshift);
    displacement_factor_2LPT = -(3.0/7.0) * growth_factor*growth_factor; // 2LPT eq. D8

    dDdt = ddickedt(redshift); // time derivative of the growth factor (1/s)
    init_growth_factor = dicke(global_params.INITIAL_REDSHIFT);
    init_displacement_factor_2LPT = -(3.0/7.0) * init_growth_factor*init_growth_factor; // 2LPT eq. D8

    // allocate memory for the updated density, and initialize
    LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    double *resampled_box;

    // check if the linear evolution flag was set
    if (global_params.EVOLVE_DENSITY_LINEARLY){
#pragma omp parallel shared(growth_factor,boxes,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
//=======
//    // find factor of HII pixel size / deltax pixel size
//    f_pixel_factor = user_params->DIM/(float)user_params->HII_DIM;
//    mass_factor = pow(f_pixel_factor, 3);
//
//    if (user_params->MOVE_DENSITY_HIGH_RES) { //cw updated high resolution density field generation
//        // allocate memory for the updated density, and initialize
//        HIRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//        HIRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//        // check if the linear evolution flag was set
//        if (global_params.EVOLVE_DENSITY_LINEARLY){
//            for (i=0; i<user_params->DIM; i++){ //cw converted everything to be high res
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<user_params->DIM; k++){
//
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = growth_factor*boxes->hires_density[R_INDEX(i,j,k)];
//                    }
//                }
//            }
//        }
//        // first order Zel'Dovich perturbation
//        else{
//
//            for (i=0; i<user_params->DIM; i++){ //cw converted everything to be high res
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<user_params->DIM; k++){
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = 0.;
//                    }
//                }
//            }
//
//            velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;
//
//            // now add the missing factor of D
//            for (ct=0; ct<TOT_NUM_PIXELS; ct++){ //cw converted everything to be high res
//                boxes->hires_vx[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//                boxes->hires_vy[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//                boxes->hires_vz[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//            }
//
//            // * ************************************************************************* * //
//            // *                           BEGIN 2LPT PART                                 * //
//            // * ************************************************************************* * //
//            // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
//            if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//
//                // allocate memory for the velocity boxes and read them in
//                velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;
//
//                // now add the missing factor in eq. D9
//                for (ct=0; ct<TOT_NUM_PIXELS; ct++){ //cw converted everything to be high res
//                    boxes->hires_vx_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                    boxes->hires_vy_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                    boxes->hires_vz_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                }
//            }
//
//            // * ************************************************************************* * //
//            // *                            END 2LPT PART                                  * //
//            // * ************************************************************************* * //
//
//            // ************  END INITIALIZATION **************************** //
//
//            // go through the high-res box, mapping the mass onto the low-res (updated) box
//            for (i=0; i<user_params->DIM;i++){
//                for (j=0; j<user_params->DIM;j++){
//                    for (k=0; k<user_params->DIM;k++){
//
//                        // map indeces to locations in units of box size
//                        xf = (i+0.5)/((user_params->DIM)+0.0);
//                        yf = (j+0.5)/((user_params->DIM)+0.0);
//                        zf = (k+0.5)/((user_params->DIM)+0.0);
//
//                        // update locations
//                        //cw kill HII_i = (unsigned long long)(i/f_pixel_factor);
//                        //cw kill HII_j = (unsigned long long)(j/f_pixel_factor);
//                        //cw kill HII_k = (unsigned long long)(k/f_pixel_factor);
//                        xf += (boxes->hires_vx)[R_INDEX(i, j, k)];
//                        yf += (boxes->hires_vy)[R_INDEX(i, j, k)];
//                        zf += (boxes->hires_vz)[R_INDEX(i, j, k)];
//
//                        // 2LPT PART
//                        // add second order corrections
//                        if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//                            xf -= (boxes->hires_vx_2LPT)[R_INDEX(i,j,k)];
//                            yf -= (boxes->hires_vy_2LPT)[R_INDEX(i,j,k)];
//                            zf -= (boxes->hires_vz_2LPT)[R_INDEX(i,j,k)];
//                        }
//
//                        xf *= (float)(user_params->DIM);
//                        yf *= (float)(user_params->DIM);
//                        zf *= (float)(user_params->DIM);
//                        while (xf >= (float)(user_params->DIM)){ xf -= (user_params->DIM);}
//                        while (xf < 0){ xf += (user_params->DIM);}
//                        while (yf >= (float)(user_params->DIM)){ yf -= (user_params->DIM);}
//                        while (yf < 0){ yf += (user_params->DIM);}
//                        while (zf >= (float)(user_params->DIM)){ zf -= (user_params->DIM);}
//                        while (zf < 0){ zf += (user_params->DIM);}
//                        xi = xf;
//                        yi = yf;
//                        zi = zf;
//                        if (xi >= (user_params->DIM)){ xi -= (user_params->DIM);}
//                        if (xi < 0) {xi += (user_params->DIM);}
//                        if (yi >= (user_params->DIM)){ yi -= (user_params->DIM);}
//                        if (yi < 0) {yi += (user_params->DIM);}
//                        if (zi >= (user_params->DIM)){ zi -= (user_params->DIM);}
//                        if (zi < 0) {zi += (user_params->DIM);}
//
//                        *( (float *)HIRES_density_perturb + R_FFT_INDEX(xi, yi, zi) ) +=
//                        (1 + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)]);
//                    }
//                }
//            }
//
//            // renormalize to the new pixel size, and make into delta //cw? QUESTION: do we need to do this here, surely better after downsampling?
//            /*/cw! I don't think it really matters
//            for (i=0; i<user_params->DIM; i++){
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<user_params->DIM; k++){
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) /= mass_factor;
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) -= 1;
//                    }
//                }
//            }*/
//
//            // deallocate
//            for (ct=0; ct<TOT_NUM_PIXELS; ct++){
//                boxes->hires_vx[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//                boxes->hires_vy[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//                boxes->hires_vz[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//            }
//
//            if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//                for (ct=0; ct<TOT_NUM_PIXELS; ct++){
//                    boxes->hires_vx_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                    boxes->hires_vy_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                    boxes->hires_vz_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                }
//            }
//        }
//
//        // Now smooth and sub sample from the high to the low resolution grids (everything that follows is just normalisations)
//
//        // ****  Print and convert to velocities ***** //
//        if (global_params.EVOLVE_DENSITY_LINEARLY){
//            memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//            // save the unfiltered density field
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if wisdom exists, if not create it
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                  }
//            }
//            else {
//                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//            memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS); //cw copy as will need it saved in complex form for wisdom c2r below
//            // filter the box before resampling it and loading into
//            if (user_params->DIM != user_params->HII_DIM)
//                filter_box( HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0) );
//
//            // FFT back to real space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                    }
//                else {
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box //cw QUESTION: Why are we copying over and FFTing the unfiltered box in the next two steps? Surely we want the smooted version of HIRES at this point?
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//                    // filter the box before resampling it and loading into
//                    if (user_params->DIM != user_params->HII_DIM) // filtering again since we've reloaded the unsmoothed version after Wisdom overwrote HIRES
//                        filter_box( HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0) );
//
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//            // resample the smoothed field onto the main density pointer
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)perturbed_field->density + R_INDEX(i,j,k)) = *( (float *)HIRES_density_perturb + R_FFT_INDEX( (unsigned long long)(i*f_pixel_factor+0.5),
//                                                          (unsigned long long)(j*f_pixel_factor+0.5),
//                                                          (unsigned long long)(k*f_pixel_factor+0.5) ) )/( (float)TOT_NUM_PIXELS ); //cw with normaliastion following c2r FFT
//
//                        if (*((float *)perturbed_field->density + R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen, but in case it does:
//                            *((float *)perturbed_field->density + R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
//                    }
//                }
//            }
//            //cw copy back the k-space unfiltered density field
//            memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//            // transform to k-space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//
//                    // save a copy of the k-space density field
//                    memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//
//            }
//            else {
//                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            // save a copy of the k-space density field */
//        }
//        else {
//
//            // transform to k-space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//
//                    // save a copy of the k-space density field
//                    memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            /*/smooth the field //cw this is no longer necessary since we're now instead filtering from hi to low resolution
//            if (!global_params.EVOLVE_DENSITY_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
//                filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
//            }*/
//
//
//            // save a copy of the k-space unsmoothed/unfiltered version of the perturbed density field
//            memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//            //cw filtering to lower resolution (resample below after ift)
//
//            if (user_params->DIM != user_params->HII_DIM)
//                filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
//
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//                    //cw QUESTION - Is this the right thing to do here, this is presumably saving to wisdom the real smoothed density field.
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//                    if (user_params->DIM != user_params->HII_DIM) //cw added another smoothing step here since we've returned HIRES to the saved unsmoothed field which ain't what we want for resampling later
//                        filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
//
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            /*/ normalize after FFT
//            for(i=0; i<user_params->DIM; i++){
//                for(j=0; j<user_params->DIM; j++){
//                    for(k=0; k<user_params->DIM; k++){
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) /= (float)TOT_NUM_PIXELS;
//                        if (*((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen
//                            *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
//                    }
//                }
//            }*/
//
//            for (i=0; i<user_params->DIM; i++){
//                for (j=0; j<user_params->DIM; j++){
//                    if (*((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
//                    for (k=0; k<user_params->DIM; k++){
//                        *((float *)perturbed_field->density + HII_R_INDEX( (unsigned long long)((i+0.0)/f_pixel_factor),(unsigned long long)((j+0.0)/f_pixel_factor),(unsigned long long)((k+0.0)/f_pixel_factor) ) )
//                            += ( (*( (float *)HIRES_density_perturb + R_FFT_INDEX( i, j, k ) )/(float)TOT_NUM_PIXELS - 1.0)/mass_factor );
//                    }
//                }
//            }
//            // copying back the k-space unsmoothed version of the perturbed density field ready for generating velocity fields
//            memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//        }
//
//        dDdt_over_D = dDdt/growth_factor;
//
//        for (n_x=0; n_x<user_params->DIM; n_x++){
//            if (n_x>MIDDLE)
//                k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
//            else
//                k_x = n_x * DELTA_K;
//
//            for (n_y=0; n_y<user_params->DIM; n_y++){
//                if (n_y>MIDDLE)
//                    k_y =(n_y-user_params->DIM) * DELTA_K;
//                else
//                    k_y = n_y * DELTA_K;
//
//                for (n_z=0; n_z<=MIDDLE; n_z++){
//                    k_z = n_z * DELTA_K;
//
//                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
//
//                    // now set the velocities
//                    if ((n_x==0) && (n_y==0) && (n_z==0)) // DC mode
//                        HIRES_density_perturb[0] = 0;
//                    else{
//                        HIRES_density_perturb[C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(TOT_NUM_PIXELS+0.0);
//                    }
//                }
//            }
//        }
//
//        memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//        //cw smooth the high resolution field ready for resampling
//        if (user_params->DIM != user_params->HII_DIM)
//            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
//
//        if(user_params->USE_FFTW_WISDOM) {
//            // Check to see if the wisdom exists, create it if it doesn't
//            sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->DIM);
//            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                fftwf_execute(plan);
//            }
//            else {
//
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//
//                // Store the wisdom for later use
//                fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                // copy over unfiltered box //cw QUESTION: Why are we copying over and FFTing the unfiltered box in the next two steps? Surely we want the smooted version of HIRES at this point?
//                memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//                if (user_params->DIM != user_params->HII_DIM)
//                    filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                fftwf_execute(plan);
//            }
//        }
//        else {
//            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
//            fftwf_execute(plan);
//        }
//
//        for (i=0; i<user_params->HII_DIM; i++){
//            for (j=0; j<user_params->HII_DIM; j++){
//                for (k=0; k<user_params->HII_DIM; k++){
//                    *((float *)perturbed_field->velocity + HII_R_INDEX(i,j,k)) = *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5), (unsigned long long)(j*f_pixel_factor+0.5), (unsigned long long)(k*f_pixel_factor+0.5)) );
//                }
//            }
//        }
//
//        // deallocate
//        fftwf_free(HIRES_density_perturb);
//        fftwf_free(HIRES_density_perturb_saved);
//
//    } //cw end update density field generation
//    else { //cw original density field perturbation from here
//
//        // allocate memory for the updated density, and initialize
//        LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//        LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//        //cw QUESTION - do we have to filter in this, or can we prefilter the density field for the same effect? probbers not, but depends on where discretisation effect kicks in and where it can be resolved.
//
//        // check if the linear evolution flag was set
//        if (global_params.EVOLVE_DENSITY_LINEARLY){
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = growth_factor*boxes->lowres_density[HII_R_INDEX(i,j,k)];
//    if (user_params->MOVE_DENSITY_HIGH_RES) { //cw updated high resolution density field generation
//        // allocate memory for the updated density, and initialize
//        HIRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//        HIRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//        // check if the linear evolution flag was set
//        if (global_params.EVOLVE_DENSITY_LINEARLY){
//            for (i=0; i<user_params->DIM; i++){ //cw converted everything to be high res
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<user_params->DIM; k++){
//
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = growth_factor*boxes->hires_density[R_INDEX(i,j,k)];
                    }
                }
            }
        }
    }
    // first order Zel'Dovich perturbation

    else {
#pragma omp parallel shared(LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)        
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                    }
                }
            }
        }

        velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;

        // now add the missing factor of D
#pragma omp parallel shared(boxes,velocity_displacement_factor) private(ct) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                boxes->lowres_vx[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                boxes->lowres_vy[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                boxes->lowres_vz[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
            }
        }

        // find factor of HII pixel size / deltax pixel size
        f_pixel_factor = user_params->DIM/(float)(user_params->HII_DIM);
        mass_factor = pow(f_pixel_factor, 3);

        // * ************************************************************************* * //
        // *                           BEGIN 2LPT PART                                 * //
        // * ************************************************************************* * //
        // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        if(global_params.SECOND_ORDER_LPT_CORRECTIONS){

            // allocate memory for the velocity boxes and read them in
            velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;

            // now add the missing factor in eq. D9
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                    boxes->lowres_vx_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                    boxes->lowres_vy_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                    boxes->lowres_vz_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                }
            }
        }

        // * ************************************************************************* * //
        // *                            END 2LPT PART                                  * //
        // * ************************************************************************* * //

        // ************  END INITIALIZATION **************************** //

        // Perturbing the density field required adding over multiple cells. Store intermediate result as a double to avoid rounding errors
        resampled_box = (double *)calloc(HII_TOT_NUM_PIXELS,sizeof(double));

        // go through the high-res box, mapping the mass onto the low-res (updated) box

#pragma omp parallel shared(LOWRES_density_perturb,init_growth_factor,boxes,f_pixel_factor,resampled_box) \
                    private(i,j,k,xi,xf,yi,yf,zi,zf,HII_i,HII_j,HII_k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
//
//        // first order Zel'Dovich perturbation
//        else{
//
//            for (i=0; i<user_params->DIM; i++){ //cw converted everything to be high res
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<user_params->DIM; k++){
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = 0.;
//                    }
//                }
//            }
//
//            velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;
//
//            // now add the missing factor of D
//            for (ct=0; ct<TOT_NUM_PIXELS; ct++){ //cw converted everything to be high res
//                boxes->hires_vx[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//                boxes->hires_vy[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//                boxes->hires_vz[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//            }
//
//            // * ************************************************************************* * //
//            // *                           BEGIN 2LPT PART                                 * //
//            // * ************************************************************************* * //
//            // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
//            if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//
//                // allocate memory for the velocity boxes and read them in
//                velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;
//
//                // now add the missing factor in eq. D9
//                for (ct=0; ct<TOT_NUM_PIXELS; ct++){ //cw converted everything to be high res
//                    boxes->hires_vx_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                    boxes->hires_vy_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                    boxes->hires_vz_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                }
//            }
//
//            // * ************************************************************************* * //
//            // *                            END 2LPT PART                                  * //
//            // * ************************************************************************* * //
//
//            // ************  END INITIALIZATION **************************** //
//
//            // go through the high-res box, mapping the mass onto the low-res (updated) box
            for (i=0; i<user_params->DIM;i++){
                for (j=0; j<user_params->DIM;j++){
                    for (k=0; k<user_params->DIM;k++){

                        // map indeces to locations in units of box size
                        xf = (i+0.5)/((user_params->DIM)+0.0);
                        yf = (j+0.5)/((user_params->DIM)+0.0);
                        zf = (k+0.5)/((user_params->DIM)+0.0);

                        // update locations
                        HII_i = (unsigned long long)(i/f_pixel_factor);
                        HII_j = (unsigned long long)(j/f_pixel_factor);
                        HII_k = (unsigned long long)(k/f_pixel_factor);
                        xf += (boxes->lowres_vx)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                        yf += (boxes->lowres_vy)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                        zf += (boxes->lowres_vz)[HII_R_INDEX(HII_i, HII_j, HII_k)];

//                        //cw kill HII_i = (unsigned long long)(i/f_pixel_factor);
//                        //cw kill HII_j = (unsigned long long)(j/f_pixel_factor);
//                        //cw kill HII_k = (unsigned long long)(k/f_pixel_factor);
//                        xf += (boxes->hires_vx)[R_INDEX(i, j, k)];
//                        yf += (boxes->hires_vy)[R_INDEX(i, j, k)];
//                        zf += (boxes->hires_vz)[R_INDEX(i, j, k)];

                        // 2LPT PART
                        // add second order corrections
                        if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
                            xf -= (boxes->lowres_vx_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                            yf -= (boxes->lowres_vy_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                            zf -= (boxes->lowres_vz_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                        }

                        xf *= (float)(user_params->HII_DIM);
                        yf *= (float)(user_params->HII_DIM);
                        zf *= (float)(user_params->HII_DIM);
                        while (xf >= (float)(user_params->HII_DIM)){ xf -= (user_params->HII_DIM);}
                        while (xf < 0){ xf += (user_params->HII_DIM);}
                        while (yf >= (float)(user_params->HII_DIM)){ yf -= (user_params->HII_DIM);}
                        while (yf < 0){ yf += (user_params->HII_DIM);}
                        while (zf >= (float)(user_params->HII_DIM)){ zf -= (user_params->HII_DIM);}
                        while (zf < 0){ zf += (user_params->HII_DIM);}
                        xi = xf;
                        yi = yf;
                        zi = zf;
                        if (xi >= (user_params->HII_DIM)){ xi -= (user_params->HII_DIM);}
                        if (xi < 0) {xi += (user_params->HII_DIM);}
                        if (yi >= (user_params->HII_DIM)){ yi -= (user_params->HII_DIM);}
                        if (yi < 0) {yi += (user_params->HII_DIM);}
                        if (zi >= (user_params->HII_DIM)){ zi -= (user_params->HII_DIM);}
                        if (zi < 0) {zi += (user_params->HII_DIM);}

#pragma omp atomic
                        resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)]);
                    }
                }
            }
        }

        // Resample back to a float for remaining algorithm
#pragma omp parallel shared(LOWRES_density_perturb,resampled_box) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) = (float)resampled_box[HII_R_INDEX(i,j,k)];
                    }
                }
            }
        }
        free(resampled_box);

        // renormalize to the new pixel size, and make into delta
#pragma omp parallel shared(LOWRES_density_perturb,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1;
                    }
//
//                            xf -= (boxes->hires_vx_2LPT)[R_INDEX(i,j,k)];
//                            yf -= (boxes->hires_vy_2LPT)[R_INDEX(i,j,k)];
//                            zf -= (boxes->hires_vz_2LPT)[R_INDEX(i,j,k)];
//                        }
//
//                        xf *= (float)(user_params->DIM);
//                        yf *= (float)(user_params->DIM);
//                        zf *= (float)(user_params->DIM);
//                        while (xf >= (float)(user_params->DIM)){ xf -= (user_params->DIM);}
//                        while (xf < 0){ xf += (user_params->DIM);}
//                        while (yf >= (float)(user_params->DIM)){ yf -= (user_params->DIM);}
//                        while (yf < 0){ yf += (user_params->DIM);}
//                        while (zf >= (float)(user_params->DIM)){ zf -= (user_params->DIM);}
//                        while (zf < 0){ zf += (user_params->DIM);}
//                        xi = xf;
//                        yi = yf;
//                        zi = zf;
//                        if (xi >= (user_params->DIM)){ xi -= (user_params->DIM);}
//                        if (xi < 0) {xi += (user_params->DIM);}
//                        if (yi >= (user_params->DIM)){ yi -= (user_params->DIM);}
//                        if (yi < 0) {yi += (user_params->DIM);}
//                        if (zi >= (user_params->DIM)){ zi -= (user_params->DIM);}
//                        if (zi < 0) {zi += (user_params->DIM);}
//
//                        *( (float *)HIRES_density_perturb + R_FFT_INDEX(xi, yi, zi) ) +=
//                        (1 + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)]);
//                    }
//                }
//            }
//
//            // renormalize to the new pixel size, and make into delta
//            for (i=0; i<user_params->DIM; i++){
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<user_params->DIM; k++){
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) /= mass_factor;
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) -= 1;
//                    }
//                }
//            }
//
//            // deallocate
//            for (ct=0; ct<TOT_NUM_PIXELS; ct++){
//                boxes->hires_vx[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//                boxes->hires_vy[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//                boxes->hires_vz[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//            }
//
//            if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//                for (ct=0; ct<TOT_NUM_PIXELS; ct++){
//                    boxes->hires_vx_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                    boxes->hires_vy_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                    boxes->hires_vz_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                }
            }
        }

        // deallocate
#pragma omp parallel shared(boxes,velocity_displacement_factor) private(ct) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                boxes->lowres_vx[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
                boxes->lowres_vy[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
                boxes->lowres_vz[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
            }
        }

        if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                    boxes->lowres_vx_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                    boxes->lowres_vy_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                    boxes->lowres_vz_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                }
            }
        }
    }

//        // Now smooth and sub sample from the high to the low resolution grids (everything that follows is just normalisations)
//
//        // ****  Print and convert to velocities ***** //
//        if (global_params.EVOLVE_DENSITY_LINEARLY){
//            // save the unfiltered density field
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if wisdom exists, if not create it
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                  }
//            }
//            else {
//                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//            memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//            // filter the box before resampling it and loading into
//            if (user_params->DIM != user_params->HII_DIM)
//                filter_box( HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0) );
//
//            // FFT back to real space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                    }
//                else {
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box //cw QUESTION: Why are we copying over and FFTing the unfiltered box in the next two steps? Surely we want the smooted version of HIRES at this point?
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//                    // filter the box before resampling it and loading into
//                    if (user_params->DIM != user_params->HII_DIM) // filtering again since we've reloaded the unsmoothed version after Wisdom overwrote HIRES
//                        filter_box( HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0) );
//
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//            // resample the smoothed field onto the main density pointer
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)perturbed_field->density + R_INDEX(i,j,k)) = *( (float *)HIRES_density_perturb + R_FFT_INDEX( (unsigned long long)(i*f_pixel_factor+0.5),
//                                                          (unsigned long long)(j*f_pixel_factor+0.5),
//                                                          (unsigned long long)(k*f_pixel_factor+0.5) ) )/( (float)TOT_NUM_PIXELS ); //cw with normaliastion following c2r FFT
//
//                        if (*((float *)perturbed_field->density + R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen, but in case it does:
//                            *((float *)perturbed_field->density + R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
//                    }
//                }
//            }
//            //cw copy back the k-space unfiltered density field
//            memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//            /* //cw removed this r2c section since we transformed earlier for filtering step and have reloaded the unsmoothed above
//            // transform to k-space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//
//                    // save a copy of the k-space density field
//                    memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//
//            }
//            else {
//                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            // save a copy of the k-space density field */
//        }
//        else {
//
//            // transform to k-space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//
//                    // save a copy of the k-space density field
//                    memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//>>>>>>> updated and compiling successfully - not yet tested
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM, (float *)HIRES_density_perturb, (fftwf_complex *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            /*/smooth the field //cw this is no longer necessary since we're now instead filtering from hi to low resolution
//            if (!global_params.EVOLVE_DENSITY_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
//                filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
//            }*/
//
//
//            // save a copy of the k-space unsmoothed version of the perturbed density field
//            memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//            //cw filtering to lower resolution (resample below after ift)
//
//            if (user_params->DIM != user_params->HII_DIM)
//                filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
//
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//                    //cw QUESTION - Is this the right thing to do here, this is presumably saving to wisdom the real smoothed density field.
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//                    if (user_params->DIM != user_params->HII_DIM) //cw added another smoothing step here since we've returned HIRES to the saved unsmoothed field which ain't what we want for resampling later
//                        filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
//
//                    plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            // normalize after FFT
//            for(i=0; i<user_params->DIM; i++){
//                for(j=0; j<user_params->DIM; j++){
//                    for(k=0; k<user_params->DIM; k++){
//                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) /= (float)TOT_NUM_PIXELS;
//                        if (*((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen
//                            *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
//                    }
//                }
//            }
//
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *(  (float *)HIRES_density_perturb
//                                                            + R_FFT_INDEX( (unsigned long long)(i*f_pixel_factor+0.5), (unsigned long long)(j*f_pixel_factor+0.5), (unsigned long long)(k*f_pixel_factor+0.5) )  );
//                    }
//                }
//            }
//            // copying back the unsmoothed version of the perturbed density field ready for generating velocity fields
//            memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//        }

    // ****  Print and convert to velocities ***** //
    if (global_params.EVOLVE_DENSITY_LINEARLY){
#pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));

//        dDdt_over_D = dDdt/growth_factor;
//
//        for (n_x=0; n_x<user_params->DIM; n_x++){
//            if (n_x>MIDDLE)
//                k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
//            else
//                k_x = n_x * DELTA_K;
//
//            for (n_y=0; n_y<user_params->DIM; n_y++){
//                if (n_y>MIDDLE)
//                    k_y =(n_y-user_params->DIM) * DELTA_K;
//                else
//                    k_y = n_y * DELTA_K;
//
//                for (n_z=0; n_z<=MIDDLE; n_z++){
//                    k_z = n_z * DELTA_K;
//
//                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
//
//                    // now set the velocities
//                    if ((n_x==0) && (n_y==0) && (n_z==0)) // DC mode
//                        HIRES_density_perturb[0] = 0;
//                    else{
//                        HIRES_density_perturb[C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(TOT_NUM_PIXELS+0.0);
                    }
                }
            }
        }

        // transform to k-space
        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"real_to_complex_DIM%d_NTHREADS%d.fftwf_wisdom",user_params->HII_DIM,user_params->N_THREADS);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);

//        memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//        //cw smooth the high resolution field ready for resampling
//        if (user_params->DIM != user_params->HII_DIM)
//            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
//
//        if(user_params->USE_FFTW_WISDOM) {
//            // Check to see if the wisdom exists, create it if it doesn't
//            sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->DIM);
//            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);

            }
            else {
                // save a copy of the k-space density field
                memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_PATIENT);
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
                fftwf_execute(plan);

                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);

                // copy over unfiltered box
                memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                fftwf_destroy_plan(plan);

                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);

//                // copy over unfiltered box //cw QUESTION: Why are we copying over and FFTing the unfiltered box in the next two steps? Surely we want the smooted version of HIRES at this point?
//                //memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
//
//                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        fftwf_destroy_plan(plan);

        // save a copy of the k-space density field
    }
    else{

        // transform to k-space
        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"real_to_complex_DIM%d_NTHREADS%d.fftwf_wisdom",user_params->HII_DIM,user_params->N_THREADS);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }

//            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM, (fftwf_complex *)HIRES_density_perturb, (float *)HIRES_density_perturb, FFTW_ESTIMATE);
//            fftwf_execute(plan);
//        }
//
//        for (i=0; i<user_params->HII_DIM; i++){
//            for (j=0; j<user_params->HII_DIM; j++){
//                for (k=0; k<user_params->HII_DIM; k++){
//                    *((float *)perturbed_field->velocity + HII_R_INDEX(i,j,k)) = *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5), (unsigned long long)(j*f_pixel_factor+0.5), (unsigned long long)(k*f_pixel_factor+0.5)) );
//                }
//            }
//        }
//
//        // deallocate
//        fftwf_free(HIRES_density_perturb);
//        fftwf_free(HIRES_density_perturb_saved);
//
//    } //cw end update density field generation
//    else { //cw original density field perturbation from here
//
//        // allocate memory for the updated density, and initialize
//        LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//        LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//        //cw QUESTION - do we have to filter in this, or can we prefilter the density field for the same effect? probbers not, but depends on where discretisation effect kicks in and where it can be resolved.
//
//        // check if the linear evolution flag was set
//        if (global_params.EVOLVE_DENSITY_LINEARLY){
//            for (i=0; i<user_params->DIM; i++){
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = growth_factor*boxes->lowres_density[HII_R_INDEX(i,j,k)];
//                    }
//                }
//            }
//        }
//        // first order Zel'Dovich perturbation
//        else{
//
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
//                    }
//                }
//            }
//
//            velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;
//
//            // now add the missing factor of D
//            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
//                boxes->lowres_vx[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//                boxes->lowres_vy[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//                boxes->lowres_vz[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
//            }
//
//            // * ************************************************************************* * //
//            // *                           BEGIN 2LPT PART                                 * //
//            // * ************************************************************************* * //
//            // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
//            if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//
//                // allocate memory for the velocity boxes and read them in
//                velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;
//
//                // now add the missing factor in eq. D9
//                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
//                    boxes->lowres_vx_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                    boxes->lowres_vy_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                    boxes->lowres_vz_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
//                }
//            }
//
//            // * ************************************************************************* * //
//            // *                            END 2LPT PART                                  * //
//            // * ************************************************************************* * //
//
//            // ************  END INITIALIZATION **************************** //
//
//            // go through the high-res box, mapping the mass onto the low-res (updated) box
//            for (i=0; i<user_params->DIM;i++){
//                for (j=0; j<user_params->DIM;j++){
//                    for (k=0; k<user_params->DIM;k++){
//
//                        // map indeces to locations in units of box size
//                        xf = (i+0.5)/((user_params->DIM)+0.0);
//                        yf = (j+0.5)/((user_params->DIM)+0.0);
//                        zf = (k+0.5)/((user_params->DIM)+0.0);
//
//                        // update locations
//                        HII_i = (unsigned long long)(i/f_pixel_factor);
//                        HII_j = (unsigned long long)(j/f_pixel_factor);
//                        HII_k = (unsigned long long)(k/f_pixel_factor);
//                        xf += (boxes->lowres_vx)[HII_R_INDEX(HII_i, HII_j, HII_k)];
//                        yf += (boxes->lowres_vy)[HII_R_INDEX(HII_i, HII_j, HII_k)];
//                        zf += (boxes->lowres_vz)[HII_R_INDEX(HII_i, HII_j, HII_k)];
//
//                        // 2LPT PART
//                        // add second order corrections
//                        if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//                            xf -= (boxes->lowres_vx_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
//                            yf -= (boxes->lowres_vy_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
//                            zf -= (boxes->lowres_vz_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
//                        }
//
//                        xf *= (float)(user_params->HII_DIM);
//                        yf *= (float)(user_params->HII_DIM);
//                        zf *= (float)(user_params->HII_DIM);
//                        while (xf >= (float)(user_params->HII_DIM)){ xf -= (user_params->HII_DIM);}
//                        while (xf < 0){ xf += (user_params->HII_DIM);}
//                        while (yf >= (float)(user_params->HII_DIM)){ yf -= (user_params->HII_DIM);}
//                        while (yf < 0){ yf += (user_params->HII_DIM);}
//                        while (zf >= (float)(user_params->HII_DIM)){ zf -= (user_params->HII_DIM);}
//                        while (zf < 0){ zf += (user_params->HII_DIM);}
//                        xi = xf;
//                        yi = yf;
//                        zi = zf;
//                        if (xi >= (user_params->HII_DIM)){ xi -= (user_params->HII_DIM);}
//                        if (xi < 0) {xi += (user_params->HII_DIM);}
//                        if (yi >= (user_params->HII_DIM)){ yi -= (user_params->HII_DIM);}
//                        if (yi < 0) {yi += (user_params->HII_DIM);}
//                        if (zi >= (user_params->HII_DIM)){ zi -= (user_params->HII_DIM);}
//                        if (zi < 0) {zi += (user_params->HII_DIM);}
//
//                        *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(xi, yi, zi) ) +=
//                        (1 + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)]);
//                    }
//                }
//            }
//
//            // renormalize to the new pixel size, and make into delta
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
//                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1;
//                    }
//                }
//            }
//
//            // deallocate
//            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
//                boxes->lowres_vx[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//                boxes->lowres_vy[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//                boxes->lowres_vz[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
//            }
//
//            if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
//                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
//                    boxes->lowres_vx_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                    boxes->lowres_vy_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                    boxes->lowres_vz_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
//                }
//            }
//        }
//
//        // ****  Print and convert to velocities ***** //
//        if (global_params.EVOLVE_DENSITY_LINEARLY){
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
//                    }
//                }
//            }
//
//            // transform to k-space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->HII_DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//
//                    // save a copy of the k-space density field
//                    memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            // save a copy of the k-space density field
//        }
//        else{
//
//            // transform to k-space
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->HII_DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//
//                    // save a copy of the k-space density field
//                    memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
            else {

                // save a copy of the k-space density field
                memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_PATIENT);
                fftwf_execute(plan);

                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);

                // copy over unfiltered box
                memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                fftwf_destroy_plan(plan);
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);

                fftwf_execute(plan);
            }
        }
//                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            //smooth the field
//            if (!global_params.EVOLVE_DENSITY_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
//                filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
//            }
//
//            // save a copy of the k-space density field
//            memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//
//            if(user_params->USE_FFTW_WISDOM) {
//                // Check to see if the wisdom exists, create it if it doesn't
//                sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->HII_DIM);
//                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
//                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//                else {
//
//                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_PATIENT);
//                    fftwf_execute(plan);
//
//                    // Store the wisdom for later use
//                    fftwf_export_wisdom_to_filename(wisdom_filename);
//
//                    // copy over unfiltered box
//                    memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//
//                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
//                    fftwf_execute(plan);
//                }
//            }
//            else {
//                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//
//            // normalize after FFT
//            for(i=0; i<user_params->HII_DIM; i++){
//                for(j=0; j<user_params->HII_DIM; j++){
//                    for(k=0; k<user_params->HII_DIM; k++){
//                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;
//                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen
//                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
//                    }
//                }
//            }
//
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<user_params->HII_DIM; k++){
//                        *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
//                    }
//                }
//            }
//
//            memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//        }
        else {
            plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        fftwf_destroy_plan(plan);

        //smooth the field
        if (!global_params.EVOLVE_DENSITY_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
            filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
        }

        // save a copy of the k-space density field

//        dDdt_over_D = dDdt/growth_factor;
//
//        for (n_x=0; n_x<user_params->HII_DIM; n_x++){
//            if (n_x>HII_MIDDLE)
//                k_x =(n_x-user_params->HII_DIM) * DELTA_K;  // wrap around for FFT convention
//            else
//                k_x = n_x * DELTA_K;
//
//            for (n_y=0; n_y<user_params->HII_DIM; n_y++){
//                if (n_y>HII_MIDDLE)
//                    k_y =(n_y-user_params->HII_DIM) * DELTA_K;
//                else
//                    k_y = n_y * DELTA_K;
//
//                for (n_z=0; n_z<=HII_MIDDLE; n_z++){
//                    k_z = n_z * DELTA_K;
//
//                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
//
//                    // now set the velocities
//                    if ((n_x==0) && (n_y==0) && (n_z==0)) // DC mode
//                        LOWRES_density_perturb[0] = 0;
//                    else{
//                        LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
//                    }
//                }
//            }
//        }
        memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"complex_to_real_DIM%d_NTHREADS%d.fftwf_wisdom",user_params->HII_DIM,user_params->N_THREADS);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {

                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_PATIENT);
                fftwf_execute(plan);

                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);

                // copy over unfiltered box
                memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                fftwf_destroy_plan(plan);
                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        fftwf_destroy_plan(plan);

        // normalize after FFT
#pragma omp parallel shared(LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for(i=0; i<user_params->HII_DIM; i++){
                for(j=0; j<user_params->HII_DIM; j++){
                    for(k=0; k<user_params->HII_DIM; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;
                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                    }
                }
            }
        }

#pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                    }
                }
            }
        }

        memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }

    float k_x, k_y, k_z, k_sq, dDdt_over_D;
    int n_x, n_y, n_z;

    dDdt_over_D = dDdt/growth_factor;

#pragma omp parallel shared(LOWRES_density_perturb,dDdt_over_D) private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (n_x=0; n_x<user_params->HII_DIM; n_x++){
            if (n_x>HII_MIDDLE)
                k_x =(n_x-user_params->HII_DIM) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y=0; n_y<user_params->HII_DIM; n_y++){
                if (n_y>HII_MIDDLE)
                    k_y =(n_y-user_params->HII_DIM) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                for (n_z=0; n_z<=HII_MIDDLE; n_z++){
                    k_z = n_z * DELTA_K;

                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                    // now set the velocities
                    if ((n_x==0) && (n_y==0) && (n_z==0)) // DC mode
                        LOWRES_density_perturb[0] = 0;
                    else{
                        LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                    }
                }
            }
        }
    }

    memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    if(user_params->USE_FFTW_WISDOM) {
        // If true, we haven't tried to read in the complex to real wisdom. Thus, check to see if we have it and create it if not
        if (global_params.EVOLVE_DENSITY_LINEARLY){

            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"complex_to_real_DIM%d_NTHREADS%d.fftwf_wisdom",user_params->HII_DIM,user_params->N_THREADS);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {

                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
                fftwf_execute(plan);

                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);

                // copy over unfiltered box
                memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                             (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            // We have already read in the complex to real wisdom for perturbing with the velocity field so just evaluate it
            plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
    }
    else {
        plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                     (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
        fftwf_execute(plan);
    }
    fftwf_destroy_plan(plan);

#pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    *((float *)perturbed_field->velocity + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                }
            }
        }
    }

    fftwf_cleanup_threads();
    fftwf_cleanup();
    fftwf_forget_wisdom();

    // deallocate
    fftwf_free(LOWRES_density_perturb);
    fftwf_free(LOWRES_density_perturb_saved);

//    fftwf_destroy_plan(plan);
    fftwf_cleanup();

    return(0);

}
