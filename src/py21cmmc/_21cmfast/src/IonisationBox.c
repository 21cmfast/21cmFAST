// Re-write of find_HII_bubbles.c for being accessible within the MCMC

int INIT_ERFC_INTERPOLATION = 1;
int INIT_RECOMBINATIONS = 1;

double *ERFC_VALS, *ERFC_VALS_DIFF;

int ComputeIonizedBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options,
                       struct PerturbedField *perturbed_field, struct IonizedBox *previous_ionize_box,
                       struct TsBox *spin_temp, struct IonizedBox *box) {

LOG_DEBUG("input values:");
LOG_DEBUG("redshift=%f, prev_redshift=%f", redshift, prev_redshift);
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
    
    omp_set_num_threads(1);
    
    char wisdom_filename[500];
    char filename[500];
    FILE *F;
    fftwf_plan plan;

    // Other parameters used in the code
    int i,j,k,ii, x,y,z, N_min_cell, LAST_FILTER_STEP, short_completely_ionised,skip_deallocate,first_step_R;
    int n_x, n_y, n_z,counter, N_halos_in_cell;
    unsigned long long ct;

    float growth_factor, pixel_mass, cell_length_factor, M_MIN, nf;
    float f_coll_crit, erfc_denom, erfc_denom_cell, res_xH, Splined_Fcoll, sqrtarg, xHI_from_xrays, curr_dens, massofscaleR, ION_EFF_FACTOR;
    float ave_M_coll_cell, ave_N_min_cell;
    
    double global_xH, ST_over_PS, mean_f_coll, f_coll, R, stored_R, f_coll_min;

    double t_ast, dfcolldt, Gamma_R_prefactor, rec, dNrec;
    float growth_factor_dz, fabs_dtdz, ZSTEP, Gamma_R, z_eff;
    const float dz = 0.01;

    float redshift_table_fcollz,redshift_table_fcollz_Xray;
    int redshift_int_fcollz,redshift_int_fcollz_Xray;
    
    float dens_val, overdense_small_min, overdense_small_bin_width, overdense_small_bin_width_inv, overdense_large_min, overdense_large_bin_width, overdense_large_bin_width_inv;
    
    int overdense_int, overdense_int_boundexceeded;
    int something_finite_or_infinite = 0;
    
    overdense_large_min = global_params.CRIT_DENS_TRANSITION*0.999;
    overdense_large_bin_width = 1./((double)NSFR_high-1.)*(Deltac-overdense_large_min);
    overdense_large_bin_width_inv = 1./overdense_large_bin_width;
    
    float Mlim_Fstar, Mlim_Fesc;
    
    float min_density, max_density;
    
    const gsl_rng_type * T;
    gsl_rng * r;
    
    init_ps();

LOG_SUPER_DEBUG("defined parameters");

    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;
    }
    else {
        ION_EFF_FACTOR = astro_params->HII_EFF_FACTOR;
    }
    
    // For recombinations
    if(flag_options->INHOMO_RECO) {
        
        if(INIT_RECOMBINATIONS) {
            init_MHR();
            INIT_RECOMBINATIONS=0;
        }
        
        ZSTEP = prev_redshift - redshift;
    
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++) {
            box->Gamma12_box[ct] = 0.0;
        }
    }
    else {
        ZSTEP = 0.2;
    }

    fabs_dtdz = fabs(dtdz(redshift));
    t_ast = astro_params->t_STAR * t_hubble(redshift);
    growth_factor_dz = dicke(redshift-dz);
    
    Splined_Fcoll = 0.;
    
    double ArgBinWidth, InvArgBinWidth, erfc_arg_val, erfc_arg_min, erfc_arg_max;
    int erfc_arg_val_index, ERFC_NUM_POINTS;
    
    erfc_arg_val = 0.;
    erfc_arg_val_index = 0;

    // Setup an interpolation table for the error function, helpful for calcluating the collapsed fraction (only for the default model, i.e. mass-independent ionising efficiency)
    erfc_arg_min = -15.0;
    erfc_arg_max = 15.0;
    
    ERFC_NUM_POINTS = 10000;
    
    ArgBinWidth = (erfc_arg_max - erfc_arg_min)/((double)ERFC_NUM_POINTS - 1.);
    InvArgBinWidth = 1./ArgBinWidth;

    if(INIT_ERFC_INTERPOLATION) {
        
        ERFC_VALS = calloc(ERFC_NUM_POINTS,sizeof(double));
        ERFC_VALS_DIFF = calloc(ERFC_NUM_POINTS,sizeof(double));
        
        for(i=0;i<ERFC_NUM_POINTS;i++) {
            
            erfc_arg_val = erfc_arg_min + ArgBinWidth*(double)i;
            
            ERFC_VALS[i] = splined_erfc(erfc_arg_val);
        }
        
        for(i=0;i<(ERFC_NUM_POINTS-1);i++) {
            ERFC_VALS_DIFF[i] = ERFC_VALS[i+1] - ERFC_VALS[i];
        }
        
        INIT_ERFC_INTERPOLATION = 0;
    }

LOG_SUPER_DEBUG("erfc interpolation done");

    /////////////////////////////////   BEGIN INITIALIZATION   //////////////////////////////////

    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        LOG_WARNING("Resolution is likely too low for accurate evolved density fields\n It Is recommended that you either increase the resolution (DIM/Box_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }

    // initialize power spectrum
    growth_factor = dicke(redshift);
    
    fftwf_complex *deltax_unfiltered, *deltax_unfiltered_original, *deltax_filtered, *xe_unfiltered, *xe_filtered, *N_rec_unfiltered, *N_rec_filtered;
    
    deltax_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deltax_unfiltered_original = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deltax_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    if(flag_options->USE_TS_FLUCT) {
        xe_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        xe_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }
    if (flag_options->INHOMO_RECO){
        N_rec_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS); // cumulative number of recombinations
        N_rec_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }
    
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        xi_SFR = calloc(NGL_SFR+1,sizeof(float));
        wi_SFR = calloc(NGL_SFR+1,sizeof(float));
    }

    float *Fcoll = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
     
    // Calculate the density field for this redshift if the initial conditions/cosmology are changing
    
    for (i=0; i<user_params->HII_DIM; i++){
        for (j=0; j<user_params->HII_DIM; j++){
            for (k=0; k<user_params->HII_DIM; k++){
                *((float *)deltax_unfiltered + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->density[HII_R_INDEX(i,j,k)];
            }
        }
    }

LOG_SUPER_DEBUG("density field calculated");

    // keep the unfiltered density field in an array, to save it for later
    memcpy(deltax_unfiltered_original, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    i=0;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    pixel_mass = RtoM(L_FACTOR*user_params->BOX_LEN/(float)(user_params->HII_DIM));
//    f_coll_crit = 1/HII_EFF_FACTOR;
    cell_length_factor = L_FACTOR;

    //set the minimum source mass
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        M_MIN = astro_params->M_TURN/50.;
        Mlim_Fstar = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_STAR, astro_params->F_STAR10);
        Mlim_Fesc = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_ESC, astro_params->F_ESC10);
    }
    else {
    
        //set the minimum source mass
        if (astro_params->ION_Tvir_MIN < 9.99999e3) // neutral IGM
            M_MIN = TtoM(redshift, astro_params->ION_Tvir_MIN, 1.22);
        else // ionized IGM
            M_MIN = TtoM(redshift, astro_params->ION_Tvir_MIN, 0.6);
        
    }

LOG_SUPER_DEBUG("minimum source mass has been set: %f", M_MIN);

    if(!flag_options->USE_TS_FLUCT) {
        if(initialiseSigmaMInterpTable(M_MIN,1e20)!=0) {
            LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
            return(2);
        }
    }

LOG_SUPER_DEBUG("sigma table has been initialised");

    // check for WDM

    if (global_params.P_CUTOFF && ( M_MIN < M_J_WDM())){
        LOG_WARNING("The default Jeans mass of %e Msun is smaller than the scale supressed by the effective pressure of WDM.", M_MIN);
        M_MIN = M_J_WDM();
        LOG_WARNING("Setting a new effective Jeans mass from WDM pressure supression of %e Msun", M_MIN);
    }

    // lets check if we are going to bother with computing the inhmogeneous field at all...
    global_xH = 0.0;
    
    // Determine the normalisation for the excursion set algorithm
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        mean_f_coll = Nion_General(redshift,astro_params->M_TURN,astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc);
    }
    else {
        mean_f_coll = FgtrM_General(redshift, M_MIN);
    }
    
    if(isfinite(mean_f_coll)==0) {
        LOG_ERROR("Mean collapse fraction is either finite or NaN!");
        return(2);
    }

LOG_SUPER_DEBUG("excursion set normalisation, mean_f_coll: %f", mean_f_coll);
    
    if (mean_f_coll * ION_EFF_FACTOR < global_params.HII_ROUND_ERR){ // way too small to ionize anything...
    //        printf( "The mean collapse fraction is %e, which is much smaller than the effective critical collapse fraction of %e\n I will just declare everything to be neutral\n", mean_f_coll, f_coll_crit);
        
        // find the neutral fraction
        if(flag_options->USE_TS_FLUCT) {
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                box->xH_box[ct] = 1-spin_temp->x_e_box[ct]; // convert from x_e to xH
                global_xH += box->xH_box[ct];
            }
            global_xH /= (double)HII_TOT_NUM_PIXELS;
        }
        else {
            init_heat();
            global_xH = 1. - xion_RECFAST(redshift, 0);
//            destruct_heat();
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                box->xH_box[ct] = global_xH;
            }
        }
    }
    else {
        
        // Take the ionisation fraction from the X-ray ionisations from Ts.c (only if the calculate spin temperature flag is set)
        if(flag_options->USE_TS_FLUCT) {
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)xe_unfiltered + HII_R_FFT_INDEX(i,j,k)) = spin_temp->x_e_box[HII_R_INDEX(i,j,k)];
                    }
                }
            }
        }

LOG_SUPER_DEBUG("calculated ionization fraction");

        if(flag_options->INHOMO_RECO) {
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)N_rec_unfiltered + HII_R_FFT_INDEX(i,j,k)) = previous_ionize_box->dNrec_box[HII_R_INDEX(i,j,k)];
                    }
                }
            }
        }

        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                // copy over unfiltered box
                memcpy(deltax_unfiltered, deltax_unfiltered_original, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }

LOG_SUPER_DEBUG("FFTs performed");

        if(flag_options->USE_TS_FLUCT) {
            if(user_params->USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)xe_unfiltered, (fftwf_complex *)xe_unfiltered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)xe_unfiltered, (fftwf_complex *)xe_unfiltered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
LOG_SUPER_DEBUG("more ffts performed");
        }


        if (flag_options->INHOMO_RECO){
            if(user_params->USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)N_rec_unfiltered, (fftwf_complex *)N_rec_unfiltered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)N_rec_unfiltered, (fftwf_complex *)N_rec_unfiltered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
LOG_SUPER_DEBUG("more ffts performed");
        }


        // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from
        //  real space to k-space
        // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
    
        for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
            deltax_unfiltered[ct] /= (HII_TOT_NUM_PIXELS+0.0);
        }
    
        if(flag_options->USE_TS_FLUCT) {
            for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                xe_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS;
            }
        }
    
        if (flag_options->INHOMO_RECO){
            for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                N_rec_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS;
            }
        }

LOG_SUPER_DEBUG("deltax unfiltered calculated");

        // ************************************************************************************* //
        // ***************** LOOP THROUGH THE FILTER RADII (in Mpc)  *************************** //
        // ************************************************************************************* //
        // set the max radius we will use, making sure we are always sampling the same values of radius
        // (this avoids aliasing differences w redshift)
        
        short_completely_ionised = 0;
        // loop through the filter radii (in Mpc)
        erfc_denom_cell=1; //dummy value
    
        R=fmax(R_BUBBLE_MIN, (cell_length_factor*user_params->BOX_LEN/(float)user_params->HII_DIM));
        
        while ((R - fmin(astro_params->R_BUBBLE_MAX, L_FACTOR*user_params->BOX_LEN)) <= FRACT_FLOAT_ERR ) {
            R*= global_params.DELTA_R_HII_FACTOR;
            if(R >= fmin(astro_params->R_BUBBLE_MAX, L_FACTOR*user_params->BOX_LEN)) {
                stored_R = R/(global_params.DELTA_R_HII_FACTOR);
            }
        }

LOG_DEBUG("set max radius: %f", R);
        
        R=fmin(astro_params->R_BUBBLE_MAX, L_FACTOR*user_params->BOX_LEN);
        LAST_FILTER_STEP = 0;
        
        first_step_R = 1;
        
        double R_temp = (double)(astro_params->R_BUBBLE_MAX);
        
        while (!LAST_FILTER_STEP && (M_MIN < RtoM(R)) ){
LOG_ULTRA_DEBUG("while loop for until RtoM(R)=%f reaches M_MIN=%f", RtoM(R), M_MIN);

            // Check if we are the last filter step
            if ( ((R/(global_params.DELTA_R_HII_FACTOR) - cell_length_factor*(user_params->BOX_LEN)/(float)(user_params->HII_DIM)) <= FRACT_FLOAT_ERR) || ((R/(global_params.DELTA_R_HII_FACTOR) - R_BUBBLE_MIN) <= FRACT_FLOAT_ERR) ) {
                LAST_FILTER_STEP = 1;
                R = fmax(cell_length_factor*user_params->BOX_LEN/(double)(user_params->HII_DIM), R_BUBBLE_MIN);
            }
            
            // Copy all relevant quantities from memory into new arrays to be smoothed and FFT'd.
            if(flag_options->USE_TS_FLUCT) {
                memcpy(xe_filtered, xe_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            }
            if (flag_options->INHOMO_RECO){
                memcpy(N_rec_filtered, N_rec_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            }
            memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            
            if (!LAST_FILTER_STEP || ((R - cell_length_factor*(user_params->BOX_LEN/(double)(user_params->HII_DIM))) > FRACT_FLOAT_ERR) ){
                if(flag_options->USE_TS_FLUCT) {
                    filter_box(xe_filtered, 1, global_params.HII_FILTER, R);
                }
                if (flag_options->INHOMO_RECO){
                    filter_box(N_rec_filtered, 1, global_params.HII_FILTER, R);
                }
                filter_box(deltax_filtered, 1, global_params.HII_FILTER, R);
            }

            // Perform FFTs
            if(user_params->USE_FFTW_WISDOM) {
                // Check to see if the wisdom exists, create it if it doesn't
                sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->HII_DIM);
                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_WISDOM_ONLY);
                    fftwf_execute(plan);
                }
                else {
                    
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_PATIENT);
                    fftwf_execute(plan);
                    
                    // Store the wisdom for later use
                    fftwf_export_wisdom_to_filename(wisdom_filename);
                    
                    // copy over unfiltered box
                    memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                    
                    // Repeat calculation as the FFTW WISDOM destroys the data
                    if (!LAST_FILTER_STEP || ((R - cell_length_factor*(user_params->BOX_LEN/(double)(user_params->HII_DIM))) > FRACT_FLOAT_ERR) ){
                        filter_box(deltax_filtered, 1, global_params.HII_FILTER, R);
                    }
                    
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_WISDOM_ONLY);
                    fftwf_execute(plan);
                }
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_ESTIMATE);
                fftwf_execute(plan);
            }

            if (flag_options->USE_TS_FLUCT) {
                if(user_params->USE_FFTW_WISDOM) {
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)xe_filtered, (float *)xe_filtered, FFTW_WISDOM_ONLY);
                }
                else {
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)xe_filtered, (float *)xe_filtered, FFTW_ESTIMATE);
                }
                fftwf_execute(plan);
            }
            
            if (flag_options->INHOMO_RECO){
                if(user_params->USE_FFTW_WISDOM) {
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)N_rec_filtered, (float *)N_rec_filtered, FFTW_WISDOM_ONLY);
                }
                else {
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)N_rec_filtered, (float *)N_rec_filtered, FFTW_ESTIMATE);
                }
                fftwf_execute(plan);
            }

            // Check if this is the last filtering scale.  If so, we don't need deltax_unfiltered anymore.
            // We will re-read it to get the real-space field, which we will use to set the residual neutral fraction
            ST_over_PS = 0;
            f_coll = 0;
            massofscaleR = RtoM(R);
            
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                
                min_density = max_density = 0.0;
                
                for (x=0; x<user_params->HII_DIM; x++){
                    for (y=0; y<user_params->HII_DIM; y++){
                        for (z=0; z<user_params->HII_DIM; z++){
                            // delta cannot be less than -1
                            *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) , -1.+FRACT_FLOAT_ERR);
                            
                            if( *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) < min_density ) {
                                min_density = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
                            }
                            if( *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) > max_density ) {
                                max_density = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
                            }
                        }
                    }
                }
                
                if(min_density < 0.) {
                    min_density = min_density*1.001;
                    if(min_density < -1.) {
                        // Use MIN_DENSITY_LOW_LIMIT as is it smaller than FRACT_FLOAT_ERR
                        min_density = -1. + global_params.MIN_DENSITY_LOW_LIMIT;
                    }
                }
                else {
                    min_density = min_density*0.999;
                }
                if(max_density < 0.) {
                    max_density = max_density*0.999;
                }
                else {
                    max_density = max_density*1.001;
                }
                
                if(global_params.HII_FILTER==1) {
                    if((0.413566994*R*2.*PI/user_params->BOX_LEN) > 1.) {
                        // The sharp k-space filter will set every cell to zero, and the interpolation table using a flexible min/max density will fail.
                        
                        min_density = -1. + global_params.MIN_DENSITY_LOW_LIMIT;
                        max_density = global_params.CRIT_DENS_TRANSITION*1.001;
                    }
                }
                
                overdense_small_min = log10(1. + min_density);
                if(max_density > global_params.CRIT_DENS_TRANSITION*1.001) {
                    overdense_small_bin_width = 1/((double)NSFR_low-1.)*(log10(1.+global_params.CRIT_DENS_TRANSITION*1.001)-overdense_small_min);
                }
                else {
                    overdense_small_bin_width = 1/((double)NSFR_low-1.)*(log10(1.+max_density)-overdense_small_min);
                }
                overdense_small_bin_width_inv = 1./overdense_small_bin_width;
                
                initialiseGL_Nion(NGL_SFR, astro_params->M_TURN,massofscaleR);
                
                if(initialise_Nion_General_spline(redshift,min_density,max_density,massofscaleR,astro_params->M_TURN,astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc)!=0) {
                    LOG_ERROR("I have encountered an infinite or a NaN value in initialise_Nion_General_spline");
                    return(2);
                }
            }
            else {
            
                erfc_denom = 2.*(pow(sigma_z0(M_MIN), 2) - pow(sigma_z0(massofscaleR), 2) );
                if (erfc_denom < 0) { // our filtering scale has become too small
                    break;
                }
                erfc_denom = sqrt(erfc_denom);
                erfc_denom = 1./( growth_factor * erfc_denom );
            
            }

            // Determine the global averaged f_coll for the overall normalisation
            
            overdense_int_boundexceeded = 0; // Reset value of int check to see if we are over-stepping our interpolation table
            
            // renormalize the collapse fraction so that the mean matches ST,
            // since we are using the evolved (non-linear) density field
            for (x=0; x<user_params->HII_DIM; x++){
                for (y=0; y<user_params->HII_DIM; y++){
                    for (z=0; z<user_params->HII_DIM; z++){
                        
                        // delta cannot be less than -1
                        *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) , -1.+FRACT_FLOAT_ERR);
                        
                        // <N_rec> cannot be less than zero
                        if (flag_options->INHOMO_RECO){
                            *((float *)N_rec_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)N_rec_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.0);
                        }
                        
                        // x_e has to be between zero and unity
                        if (flag_options->USE_TS_FLUCT){
                            *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.);
                            *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) = FMIN(*((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.999);
                        }
                        
                        curr_dens = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));

                        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
                            
                            if (curr_dens < global_params.CRIT_DENS_TRANSITION){
                                
                                if (curr_dens < -1.) {
                                    Splined_Fcoll = 0;
                                }
                                else {
                                    dens_val = (log10f(curr_dens+1.) - overdense_small_min)*overdense_small_bin_width_inv;
                                    overdense_int = (int)floorf( dens_val );
                                    
                                    if(overdense_int < 0 || (overdense_int + 1) > (NSFR_low - 1)) {
                                        overdense_int_boundexceeded = 1;
                                    }
                                    
                                    Splined_Fcoll = log10_Nion_spline[overdense_int]*( 1 + (float)overdense_int - dens_val ) + log10_Nion_spline[overdense_int+1]*( dens_val - (float)overdense_int );
                                    Splined_Fcoll = expf(Splined_Fcoll);
                                    
                                }
                            }
                            else {
                                if (curr_dens < 0.99*Deltac) {
                                    
                                    dens_val = (curr_dens - overdense_large_min)*overdense_large_bin_width_inv;
                                    
                                    overdense_int = (int)floorf( dens_val );

                                    if(overdense_int < 0 || (overdense_int + 1) > (NSFR_high - 1)) {
                                        overdense_int_boundexceeded = 1;
                                    }
                                    
                                    Splined_Fcoll = Nion_spline[overdense_int]*( 1 + (float)overdense_int - dens_val ) + Nion_spline[overdense_int+1]*( dens_val - (float)overdense_int );
                                }
                                else {
                                    Splined_Fcoll = 1.;
                                }
                            }
                            
                        }
                        else {
                        
                            erfc_arg_val = (Deltac - curr_dens)*erfc_denom;
                            if( erfc_arg_val < erfc_arg_min || erfc_arg_val > erfc_arg_max ) {
                                Splined_Fcoll = splined_erfc(erfc_arg_val);
                            }
                            else {
                                erfc_arg_val_index = (int)floor(( erfc_arg_val - erfc_arg_min )*InvArgBinWidth);
                                Splined_Fcoll = ERFC_VALS[erfc_arg_val_index] + (erfc_arg_val - (erfc_arg_min + ArgBinWidth*(double)erfc_arg_val_index))*ERFC_VALS_DIFF[erfc_arg_val_index]*InvArgBinWidth;
                            }
                        }
     
                        // save the value of the collasped fraction into the Fcoll array
                        Fcoll[HII_R_INDEX(x,y,z)] = Splined_Fcoll;
                        f_coll += Splined_Fcoll;
                    }
                }
            } //  end loop through Fcoll box
            
            if(overdense_int_boundexceeded==1) {
                LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables for the nion_splines");
                return(2);
            }
            
            
            if(isfinite(f_coll)==0) {
                LOG_ERROR("f_coll is either finite or NaN!");
                return(2);
            }

            f_coll /= (double) HII_TOT_NUM_PIXELS;
            
            // To avoid ST_over_PS becoms nan when f_coll = 0, I set f_coll = FRACT_FLOAT_ERR.
            if(flag_options->USE_MASS_DEPENDENT_ZETA) {
                if (f_coll <= f_coll_min) f_coll = f_coll_min;
            }
            else {
                if (f_coll <= FRACT_FLOAT_ERR) f_coll = FRACT_FLOAT_ERR;
            }
            
            ST_over_PS = mean_f_coll/f_coll;
            
            //////////////////////////////  MAIN LOOP THROUGH THE BOX ///////////////////////////////////
            // now lets scroll through the filtered box
            
            rec = 0.;
        
            xHI_from_xrays = 1;
            Gamma_R_prefactor = pow(1+redshift, 2) * (R*CMperMPC) * SIGMA_HI * global_params.ALPHA_UVB / (global_params.ALPHA_UVB+2.75) * N_b0 * ION_EFF_FACTOR / 1.0e-12;
            
            Gamma_R_prefactor /= t_ast;
            
            for (x=0; x<user_params->HII_DIM; x++){
                for (y=0; y<user_params->HII_DIM; y++){
                    for (z=0; z<user_params->HII_DIM; z++){
     
                        curr_dens = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
     
                        Splined_Fcoll = Fcoll[HII_R_INDEX(x,y,z)];
     
                        f_coll = ST_over_PS * Splined_Fcoll;
                        
                        if (LAST_FILTER_STEP){
                            ave_M_coll_cell = f_coll * pixel_mass * (1. + curr_dens);
                            ave_N_min_cell = ave_M_coll_cell / M_MIN; // ave # of M_MIN halos in cell
                            N_halos_in_cell = (int) gsl_ran_poisson(r, global_params.N_POISSON);
                        }
                        
                        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
                            if (f_coll <= f_coll_min) f_coll = f_coll_min;
                        }
                    
                        if (flag_options->INHOMO_RECO){
                            rec = (*((float *)N_rec_filtered + HII_R_FFT_INDEX(x,y,z))); // number of recombinations per mean baryon
                            rec /= (1. + curr_dens); // number of recombinations per baryon inside <R>
                        }
                    
                        // adjust the denominator of the collapse fraction for the residual electron fraction in the neutral medium
                        if (flag_options->USE_TS_FLUCT){
                            xHI_from_xrays = (1. - *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)));
                        }
                    
                        // check if fully ionized!
                        if ( (f_coll > (xHI_from_xrays/ION_EFF_FACTOR)*(1.0+rec)) ){ //IONIZED!!
                        
                            // if this is the first crossing of the ionization barrier for this cell (largest R), record the gamma
                            // this assumes photon-starved growth of HII regions...  breaks down post EoR
                            if (flag_options->INHOMO_RECO && (box->xH_box[HII_R_INDEX(x,y,z)] > FRACT_FLOAT_ERR) ){
                                box->Gamma12_box[HII_R_INDEX(x,y,z)] = Gamma_R_prefactor * f_coll;
                            }
                        
                            // keep track of the first time this cell is ionized (earliest time)
                            if (flag_options->INHOMO_RECO && (previous_ionize_box->z_re_box[HII_R_INDEX(x,y,z)] < 0)){
                                box->z_re_box[HII_R_INDEX(x,y,z)] = redshift;
                            }
                        
                            // FLAG CELL(S) AS IONIZED
                            if (global_params.FIND_BUBBLE_ALGORITHM == 2) // center method
                                box->xH_box[HII_R_INDEX(x,y,z)] = 0;
                            else if (global_params.FIND_BUBBLE_ALGORITHM == 1) // sphere method
                                update_in_sphere(box->xH_box, user_params->HII_DIM, R/(user_params->BOX_LEN), x/(user_params->HII_DIM+0.0), y/(user_params->HII_DIM+0.0), z/(user_params->HII_DIM+0.0));
                            else{
                                LOG_ERROR("Incorrect choice of find bubble algorithm: %i\nAborting...", global_params.FIND_BUBBLE_ALGORITHM);
                                box->xH_box[HII_R_INDEX(x,y,z)] = 0;
                                return(2);
                            }
                        } // end ionized
                        // If not fully ionized, then assign partial ionizations
                        else if (LAST_FILTER_STEP && (box->xH_box[HII_R_INDEX(x,y,z)] > TINY)){
                        
                            if (f_coll>1) f_coll=1;
                        
                            if(ave_N_min_cell < global_params.N_POISSON) {
                                f_coll = N_halos_in_cell * ( ave_M_coll_cell / (float)global_params.N_POISSON ) / (pixel_mass*(1. + curr_dens));
                            }
                            
                            if(ave_M_coll_cell < (M_MIN/5.)) {
                                f_coll = 0.;
                            }
                            
                            if (f_coll>1) f_coll=1;
                            res_xH = xHI_from_xrays - f_coll * ION_EFF_FACTOR;
                            
                            // and make sure fraction doesn't blow up for underdense pixels
                            if (res_xH < 0)
                                res_xH = 0;
                            else if (res_xH > 1)
                                res_xH = 1;
                        
                            box->xH_box[HII_R_INDEX(x,y,z)] = res_xH;

                        } // end partial ionizations at last filtering step
                    } // k
                } // j
            } // i
            
            if(first_step_R) {
                R = stored_R;
                first_step_R = 0;
            }
            else {
                R /= (global_params.DELTA_R_HII_FACTOR);
            }            
        }
     
        // find the neutral fraction
        if(LOG_LEVEL >= DEBUG_LEVEL){
            global_xH = 0;

            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                global_xH += box->xH_box[ct];
            }
            global_xH /= (float)HII_TOT_NUM_PIXELS;

        }
        
        if(isfinite(global_xH)==0) {
            LOG_ERROR("Neutral fraction is either infinite or a Nan. Something has gone wrong in the ionisation calculation!");
            return(2);
        }
        
        // update the N_rec field
        if (flag_options->INHOMO_RECO){
            
            for (x=0; x<user_params->HII_DIM; x++){
                for (y=0; y<user_params->HII_DIM; y++){
                    for (z=0; z<user_params->HII_DIM; z++){
                    
                        curr_dens = 1.0 + perturbed_field->density[HII_R_INDEX(x,y,z)];
                        z_eff = (1+redshift) * pow(curr_dens, 1.0/3.0) - 1;
                        dNrec = splined_recombination_rate(z_eff, box->Gamma12_box[HII_R_INDEX(x,y,z)]) * fabs_dtdz * ZSTEP * (1 - box->xH_box[HII_R_INDEX(x,y,z)]);
                        
                        if(isfinite(dNrec)==0) {
                            something_finite_or_infinite = 1;
                        }
                        
                        box->dNrec_box[HII_R_INDEX(x,y,z)] = previous_ionize_box->dNrec_box[HII_R_INDEX(x,y,z)] + dNrec;
                    }
                }
            }
            
            if(something_finite_or_infinite) {
                LOG_ERROR("Recombinations have returned either an infinite or NaN value.");
                return(2);
            }
            
        }
    }
    
    // deallocate
    gsl_rng_free (r);

LOG_DEBUG("global_xH = %e",global_xH);

    fftwf_free(deltax_unfiltered);
    fftwf_free(deltax_unfiltered_original);
    fftwf_free(deltax_filtered);
    if(flag_options->USE_TS_FLUCT) {
        fftwf_free(xe_unfiltered);
        fftwf_free(xe_filtered);
    }
    if (flag_options->INHOMO_RECO){
        fftwf_free(N_rec_unfiltered);
        fftwf_free(N_rec_filtered);
    }

LOG_SUPER_DEBUG("freed fftw boxes");

    free(Fcoll);
    
    free(xi_SFR);
    free(wi_SFR);

    if(!flag_options->USE_TS_FLUCT) {
        freeSigmaMInterpTable();
    }

    
//    fftwf_destroy_plan(plan);
    fftwf_cleanup();

    

LOG_DEBUG("finished!");

    return(0);
}

