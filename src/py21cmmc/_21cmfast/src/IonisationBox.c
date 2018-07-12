
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

int INIT_ERFC_INTERPOLATION = 1;

double erfc_arg_min = -15.0;
double erfc_arg_max = 15.0;
int ERFC_NUM_POINTS = 10000;

double *ERFC_VALS, *ERFC_VALS_DIFF;

void ComputeIonizedBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options,
                       struct PerturbedField *perturbed_field, struct IonizedBox *previous_ionize_box,
                       int do_spin_temp, struct TsBox *spin_temp, struct IonizedBox *box) {

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    
//    printf("EFF_FACTOR_PL_INDEX = %e HII_EFF_FACTOR = %e R_BUBBLE_MAX = %e ION_Tvir_MIN = %e L_X = %e\n",astro_params->EFF_FACTOR_PL_INDEX,astro_params->HII_EFF_FACTOR,
//           astro_params->R_BUBBLE_MAX,astro_params->ION_Tvir_MIN,astro_params->L_X);
//    printf("NU_X_THRESH = %e X_RAY_SPEC_INDEX = %e X_RAY_Tvir_MIN = %e\n",astro_params->NU_X_THRESH,astro_params->X_RAY_SPEC_INDEX,astro_params->X_RAY_Tvir_MIN);
//    printf("F_STAR = %e t_STAR = %e N_RSD_STEPS = %d\n",astro_params->F_STAR,astro_params->t_STAR,astro_params->N_RSD_STEPS);
//   
//    printf("NU_X_BAND_MAX = %e NU_X_MAX = %e\n",global_params.NU_X_BAND_MAX,global_params.NU_X_MAX);
//    printf("Z_HEAT_MAX = %e ZPRIME_STEP_FACTOR = %e\n",global_params.Z_HEAT_MAX,global_params.ZPRIME_STEP_FACTOR);
//    
//    printf("INCLUDE_ZETA_PL = %s SUBCELL_RSD = %s INHOMO_RECO = %s\n",flag_options->INCLUDE_ZETA_PL ?"true" : "false",flag_options->SUBCELL_RSD ?"true" : "false",flag_options->INHOMO_RECO ?"true" : "false");
    
    printf("low-res perturbed density; %e %e %e %e\n",perturbed_field->density[0],perturbed_field->density[100],perturbed_field->density[1000],perturbed_field->density[10000]);
    
    char filename[500];
    FILE *F;
    fftwf_plan plan;

    // Other parameters used in the code
    int i,j,k,ii, x,y,z, N_min_cell, LAST_FILTER_STEP, short_completely_ionised,skip_deallocate,first_step_R;
    int n_x, n_y, n_z,counter;
    unsigned long long ct;

    float growth_factor,MFEEDBACK, pixel_mass, cell_length_factor, ave_N_min_cell, M_MIN, nf;
    float f_coll_crit, erfc_denom, erfc_denom_cell, res_xH, Splined_Fcoll, sqrtarg, xHI_from_xrays, curr_dens, massofscaleR;

    double global_xH, global_step_xH, ST_over_PS, mean_f_coll_st, f_coll, R, stored_R;

    double t_ast, dfcolldt, Gamma_R_prefactor, rec, dNrec;
    float growth_factor_dz, fabs_dtdz, ZSTEP, Gamma_R, z_eff;
    const float dz = 0.01;

    const gsl_rng_type * T;
    gsl_rng * r;

    // For recombinations
    if(flag_options->INHOMO_RECO) {
        printf("With recombinations\n");
        ZSTEP = prev_redshift - redshift;
    
//        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++) {
//            box->Gamma12[ct] = 0.0;
//        }
    }
    else {
        printf("No recombinations\n");
        ZSTEP = 0.2;
    }
    fabs_dtdz = fabs(dtdz(redshift));
    t_ast = astro_params->t_STAR * t_hubble(redshift);
    growth_factor_dz = dicke(redshift-dz);
    
    double ArgBinWidth, InvArgBinWidth, erfc_arg_val;
    int erfc_arg_val_index;
    
    if(INIT_ERFC_INTERPOLATION) {
        
        // Setup an interpolation table for the error function, helpful for calcluating the collapsed fraction (only for the default model, i.e. mass-independent ionising efficiency)
        erfc_arg_min = -15.0;
        erfc_arg_max = 15.0;
        
        ERFC_NUM_POINTS = 10000;
        
        ERFC_VALS = calloc(ERFC_NUM_POINTS,sizeof(double));
        ERFC_VALS_DIFF = calloc(ERFC_NUM_POINTS,sizeof(double));
        
        ArgBinWidth = (erfc_arg_max - erfc_arg_min)/((double)ERFC_NUM_POINTS - 1.);
        InvArgBinWidth = 1./ArgBinWidth;
        
        for(i=0;i<ERFC_NUM_POINTS;i++) {
            
            erfc_arg_val = erfc_arg_min + ArgBinWidth*(double)i;
            
            ERFC_VALS[i] = splined_erfc(erfc_arg_val);
        }
        
        for(i=0;i<(ERFC_NUM_POINTS-1);i++) {
            ERFC_VALS_DIFF[i] = ERFC_VALS[i+1] - ERFC_VALS[i];
        }
        
        INIT_ERFC_INTERPOLATION = 0;
    }
    
    /////////////////////////////////   BEGIN INITIALIZATION   //////////////////////////////////

    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        printf("perturb_field.c: WARNING: Resolution is likely too low for accurate evolved density fields\n It Is recommended that you either increase the resolution (DIM/Box_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }

    // initialize power spectrum
    growth_factor = dicke(redshift);
    
    fftwf_complex *deltax_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *deltax_unfiltered_original = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *deltax_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *xe_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *xe_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    
    fftwf_complex *N_rec_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS); // cumulative number of recombinations
    fftwf_complex *N_rec_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    
//     float *deltax = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
     float *Fcoll = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
//     float *xH = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
     
    // Calculate the density field for this redshift if the initial conditions/cosmology are changing
    
    for (i=0; i<user_params->HII_DIM; i++){
        for (j=0; j<user_params->HII_DIM; j++){
            for (k=0; k<user_params->HII_DIM; k++){
                *((float *)deltax_unfiltered + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->density[HII_R_INDEX(i,j,k)];
            }
        }
    }
    
    printf("low-res perturbed density; %e %e\n",*((float *)deltax_unfiltered + HII_R_FFT_INDEX(0,0,0)),deltax_unfiltered[HII_R_FFT_INDEX(0,0,0)]);
    
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
/*    if (global_params.ION_Tvir_MIN > 0){ // use the virial temperature for Mmin
        if (global_params.ION_Tvir_MIN < 9.99999e3) // neutral IGM
            M_MIN = TtoM(redshift, global_params.ION_Tvir_MIN, 1.22);
        else // ionized IGM
            M_MIN = TtoM(redshift, global_params.ION_Tvir_MIN, 0.6);
    }
    else if (global_params.ION_Tvir_MIN < 0){ // use the mass
        M_MIN = ION_M_MIN;
    }
*/
    if (astro_params->ION_Tvir_MIN < 9.99999e3) // neutral IGM
        M_MIN = TtoM(redshift, astro_params->ION_Tvir_MIN, 1.22);
    else // ionized IGM
        M_MIN = TtoM(redshift, astro_params->ION_Tvir_MIN, 0.6);
    
    // check for WDM

    if (global_params.P_CUTOFF && ( M_MIN < M_J_WDM())){
        printf( "The default Jeans mass of %e Msun is smaller than the scale supressed by the effective pressure of WDM.\n", M_MIN);
        M_MIN = M_J_WDM();
        printf( "Setting a new effective Jeans mass from WDM pressure supression of %e Msun\n", M_MIN);
    }

//    for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
//        xH[ct] = 1.;
//    }

    // lets check if we are going to bother with computing the inhmogeneous field at all...

    global_xH = 0.0;

    MFEEDBACK = M_MIN;

    if(astro_params->EFF_FACTOR_PL_INDEX != 0.) {
        mean_f_coll_st = FgtrM_st_PL(redshift,M_MIN,MFEEDBACK,astro_params->EFF_FACTOR_PL_INDEX);
    }
    else {
        mean_f_coll_st = FgtrM_st(redshift, M_MIN);
    }
    
    printf("redshift = %e M_MIN = %e\n",redshift, M_MIN);
    printf("mean_f_coll_st = %e\n",mean_f_coll_st);
    
    if (mean_f_coll_st/(1./(astro_params->HII_EFF_FACTOR)) < global_params.HII_ROUND_ERR){ // way too small to ionize anything...
    //        printf( "The ST mean collapse fraction is %e, which is much smaller than the effective critical collapse fraction of %e\n I will just declare everything to be neutral\n", mean_f_coll_st, f_coll_crit);
    
        // find the neutral fraction
        if(flag_options->USE_TS_FLUCT) {
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                box->xH_box[ct] = 1-spin_temp->x_e_box[ct]; // convert from x_e to xH
                global_xH += box->xH_box[ct];
            }
            global_xH /= (double)HII_TOT_NUM_PIXELS;
        }
        else {
//            init_heat();
//            global_xH = 1. - xion_RECFAST(REDSHIFT_SAMPLE, 0);
//            destruct_heat();
//            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
//                box->xH_box[ct] = global_xH;
//            }
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
        
        if(flag_options->USE_TS_FLUCT) {
            plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)xe_unfiltered, (fftwf_complex *)xe_unfiltered, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
    
        if (flag_options->INHOMO_RECO){
            plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)N_rec_unfiltered, (fftwf_complex *)N_rec_unfiltered, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
    
        plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        fftwf_cleanup();
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
        
        R=fmin(astro_params->R_BUBBLE_MAX, L_FACTOR*user_params->BOX_LEN);
    
        LAST_FILTER_STEP = 0;
    
        initialiseSplinedSigmaM(M_MIN,1e16);
    
        first_step_R = 1;
        
        double R_temp = (double)(astro_params->R_BUBBLE_MAX);
        
        while (!LAST_FILTER_STEP && (M_MIN < RtoM(R)) ){
            
            // Check if we are the last filter step
            if ( ((R/(global_params.DELTA_R_HII_FACTOR) - cell_length_factor*(user_params->BOX_LEN)/(float)(user_params->HII_DIM)) <= FRACT_FLOAT_ERR) || ((R/(global_params.DELTA_R_HII_FACTOR) - R_BUBBLE_MIN) <= FRACT_FLOAT_ERR) ) {
                LAST_FILTER_STEP = 1;
                R = fmax(cell_length_factor*user_params->BOX_LEN/(double)(user_params->HII_DIM), R_BUBBLE_MIN);
            }
            
            // Copy all relevant quantities from memory into new arrays to be smoothed and FFT'd.
//            if(USE_TS_FLUCT) {
//                memcpy(xe_filtered, xe_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//            }
//            if (INHOMO_RECO){
//                memcpy(N_rec_filtered, N_rec_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//            }
            memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            
            if (!LAST_FILTER_STEP || ((R - cell_length_factor*(user_params->BOX_LEN/(double)(user_params->HII_DIM))) > FRACT_FLOAT_ERR) ){
//                if(USE_TS_FLUCT) {
//                    HII_filter(xe_filtered, HII_FILTER, R);
//                }
//                if (INHOMO_RECO){
//                    HII_filter(N_rec_filtered, HII_FILTER, R);
//                }
//                HII_filter(deltax_filtered, HII_FILTER, R);
                filter_box(deltax_filtered, 1, global_params.HII_FILTER, R);
            }
        
            // Perform FFTs
            plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_ESTIMATE);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
            fftwf_cleanup();
        
//            if (USE_TS_FLUCT) {
//                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)xe_filtered, (float *)xe_filtered, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
//        
//            if (INHOMO_RECO){
//                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)N_rec_filtered, (float *)N_rec_filtered, FFTW_ESTIMATE);
//                fftwf_execute(plan);
//            }
        
            // Check if this is the last filtering scale.  If so, we don't need deltax_unfiltered anymore.
            // We will re-read it to get the real-space field, which we will use to set the residual neutral fraction
            ST_over_PS = 0;
            f_coll = 0;
            massofscaleR = RtoM(R);
        
            erfc_denom = 2.*(pow(sigma_z0(M_MIN), 2) - pow(sigma_z0(massofscaleR), 2) );
            if (erfc_denom < 0) { // our filtering scale has become too small
                break;
            }
            erfc_denom = sqrt(erfc_denom);
            erfc_denom = 1./( growth_factor * erfc_denom );
        
            if((astro_params->EFF_FACTOR_PL_INDEX)!=0.) {
                initialiseGL_Fcoll(NGLlow,NGLhigh,M_MIN,massofscaleR);
                initialiseFcoll_spline(redshift,M_MIN,massofscaleR,massofscaleR,MFEEDBACK,astro_params->EFF_FACTOR_PL_INDEX);
            }
            
            // Determine the global averaged f_coll for the overall normalisation
                
            // renormalize the collapse fraction so that the mean matches ST,
            // since we are using the evolved (non-linear) density field
            for (x=0; x<user_params->HII_DIM; x++){
                for (y=0; y<user_params->HII_DIM; y++){
                    for (z=0; z<user_params->HII_DIM; z++){
                        
                        // delta cannot be less than -1
                        *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) , -1.+FRACT_FLOAT_ERR);
                        
                        // <N_rec> cannot be less than zero
//                        if (INHOMO_RECO){
//                            *((float *)N_rec_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)N_rec_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.0);
//                        }
                        
                        // x_e has to be between zero and unity
//                        if (USE_TS_IN_21CM){
//                            *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.);
//                            *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) = FMIN(*((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.999);
//                        }
                        
                        curr_dens = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
                        
                        if(astro_params->EFF_FACTOR_PL_INDEX!=0.) {
                            // Usage of 0.99*Deltac arises due to the fact that close to the critical density, the collapsed fraction becomes a little unstable
                            // However, such densities should always be collapsed, so just set f_coll to unity. Additionally, the fraction of points in this regime relative
                            // to the entire simulation volume is extremely small.
                            if(curr_dens <= 0.99*Deltac) {
                                FcollSpline(curr_dens,&(Splined_Fcoll));
                            }
                            else { // the entrire cell belongs to a collpased halo...  this is rare...
                                Splined_Fcoll =  1.0;
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
            
            f_coll /= (double) HII_TOT_NUM_PIXELS;
     
            ST_over_PS = mean_f_coll_st/f_coll;
        
            //////////////////////////////  MAIN LOOP THROUGH THE BOX ///////////////////////////////////
            // now lets scroll through the filtered box
            
            rec = 0.;
        
            xHI_from_xrays = 1;
            Gamma_R_prefactor = pow(1+redshift, 2) * (R*CMperMPC) * SIGMA_HI * global_params.ALPHA_UVB / (global_params.ALPHA_UVB+2.75) * N_b0 * (astro_params->HII_EFF_FACTOR) / 1.0e-12;
                
            for (x=0; x<user_params->HII_DIM; x++){
                for (y=0; y<user_params->HII_DIM; y++){
                    for (z=0; z<user_params->HII_DIM; z++){
     
                        curr_dens = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
     
                        Splined_Fcoll = Fcoll[HII_R_INDEX(x,y,z)];
     
                        f_coll = ST_over_PS * Splined_Fcoll;
                    
//                        if (INHOMO_RECO){
//                            dfcolldt = f_coll / t_ast;
//                            Gamma_R = Gamma_R_prefactor * dfcolldt;
//                            rec = (*((float *)N_rec_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC))); // number of recombinations per mean baryon
//                            rec /= (1. + curr_dens); // number of recombinations per baryon inside <R>
//                        }
                    
                        // adjust the denominator of the collapse fraction for the residual electron fraction in the neutral medium
//                        if (USE_TS_IN_21CM){
//                            xHI_from_xrays = (1. - *((float *)xe_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)));
//                        }
                    
                        // check if fully ionized!
                        if ( (f_coll > (xHI_from_xrays/(astro_params->HII_EFF_FACTOR))*(1.0+rec)) ){ //IONIZED!!
                        
                            // if this is the first crossing of the ionization barrier for this cell (largest R), record the gamma
                            // this assumes photon-starved growth of HII regions...  breaks down post EoR
//                            if (INHOMO_RECO && (xH[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] > FRACT_FLOAT_ERR) ){
//                                Gamma12[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = Gamma_R;
//                            }
                        
                            // keep track of the first time this cell is ionized (earliest time)
//                            if (INHOMO_RECO && (z_re[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] < 0)){
//                                z_re[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = REDSHIFT_SAMPLE;
//                            }
                        
                            // FLAG CELL(S) AS IONIZED
                            if (global_params.FIND_BUBBLE_ALGORITHM == 2) // center method
                                box->xH_box[HII_R_INDEX(x,y,z)] = 0;
                            else if (global_params.FIND_BUBBLE_ALGORITHM == 1) // sphere method
                                update_in_sphere(box->xH_box, user_params->HII_DIM, R/(user_params->BOX_LEN), x/(user_params->HII_DIM+0.0), y/(user_params->HII_DIM+0.0), z/(user_params->HII_DIM+0.0));
                            else{
                                printf( "Incorrect choice of find bubble algorithm: %i\nAborting...", global_params.FIND_BUBBLE_ALGORITHM);
                                box->xH_box[HII_R_INDEX(x,y,z)] = 0;
                            }
                        } // end ionized
                        // If not fully ionized, then assign partial ionizations
                        else if (LAST_FILTER_STEP && (box->xH_box[HII_R_INDEX(x,y,z)] > TINY)){
                        
                            if (f_coll>1) f_coll=1;
                        
                            ave_N_min_cell = f_coll * pixel_mass*(1. + curr_dens) / M_MIN; // ave # of M_MIN halos in cell
//                            printf("f_coll = %e pixel_mass = %e 1+del = %e dens = %e dens = %e M_MIN = %e ave_N_min_cell = %e\n",f_coll,pixel_mass,(1. + curr_dens),*((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)),*((float *)deltax_unfiltered_original + HII_R_FFT_INDEX(x,y,z)),M_MIN,ave_N_min_cell);
                            if (ave_N_min_cell < global_params.N_POISSON){
                                N_min_cell = (int) gsl_ran_poisson(r, ave_N_min_cell);
                                f_coll = N_min_cell * M_MIN / (pixel_mass*(1. + curr_dens));
                            }
                            
                            if (f_coll>1) f_coll=1;
                            res_xH = xHI_from_xrays - f_coll * (astro_params->HII_EFF_FACTOR);
//                            printf("xHI_from_xrays = %e f_coll = %e HII_EFF_FACTOR = %e res_xH = %e\n",xHI_from_xrays,f_coll,(astro_params->HII_EFF_FACTOR),res_xH);
                            // and make sure fraction doesn't blow up for underdense pixels
                            if (res_xH < 0)
                                res_xH = 0;
                            else if (res_xH > 1)
                                res_xH = 1;
                        
                            box->xH_box[HII_R_INDEX(x,y,z)] = res_xH;
//                            printf("x = %d y = %d z = %d res_xH = %e\n",x,y,z,res_xH,box->xH_box[HII_R_INDEX(x,y,z)]);
                        } // end partial ionizations at last filtering step
                    } // k
                } // j
            } // i
            
            global_step_xH = 0.;
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                global_step_xH += box->xH_box[ct];
            }
            global_step_xH /= (float)HII_TOT_NUM_PIXELS;
            
            if(global_step_xH==0.0) {
                short_completely_ionised = 1;
                break;
            }
        
            if(first_step_R) {
                R = stored_R;
                first_step_R = 0;
            }
            else {
                R /= (global_params.DELTA_R_HII_FACTOR);
            }
        }
     
        // find the neutral fraction
        global_xH = 0;
        
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
            global_xH += box->xH_box[ct];
        }
        global_xH /= (float)HII_TOT_NUM_PIXELS;
                
        printf("global_xH = %e\n",global_xH);
                
        /*
        // update the N_rec field
        if (INHOMO_RECO){
        
            //fft to get the real N_rec  and delta fields
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)N_rec_unfiltered, (float *)N_rec_unfiltered, FFTW_ESTIMATE);
            fftwf_execute(plan);
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_unfiltered, (float *)deltax_unfiltered, FFTW_ESTIMATE);
        
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        
            fftwf_cleanup();
        
            for (x=0; x<HII_DIM; x++){
                for (y=0; y<HII_DIM; y++){
                    for (z=0; z<HII_DIM; z++){
                    
                        curr_dens = 1.0 + (*((float *)deltax_unfiltered + HII_R_FFT_INDEX(x,y,z)));
                        z_eff = (1+REDSHIFT_SAMPLE) * pow(curr_dens, 1.0/3.0) - 1;
                        dNrec = splined_recombination_rate(z_eff, Gamma12[HII_R_INDEX(x,y,z)]) * fabs_dtdz * ZSTEP * (1 - xH[HII_R_INDEX(x,y,z)]);
                        *((float *)N_rec_unfiltered + HII_R_FFT_INDEX(x,y,z)) += dNrec;
                    }
                }
            }
        }
         */
    }

    // deallocate
    gsl_rng_free (r);

}

