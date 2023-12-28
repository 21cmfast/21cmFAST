// Re-write of find_HII_bubbles.c for being accessible within the MCMC

int INIT_ERFC_INTERPOLATION = 1;
int INIT_RECOMBINATIONS = 1;

double *ERFC_VALS, *ERFC_VALS_DIFF;

float absolute_delta_z;

float overdense_small_min, overdense_small_bin_width, overdense_small_bin_width_inv;
float overdense_large_min, overdense_large_bin_width, overdense_large_bin_width_inv;
float prev_overdense_small_min, prev_overdense_small_bin_width, prev_overdense_small_bin_width_inv;
float prev_overdense_large_min, prev_overdense_large_bin_width, prev_overdense_large_bin_width_inv;
float log10Mturn_min, log10Mturn_max, log10Mturn_bin_width, log10Mturn_bin_width_inv;
float log10Mturn_min_MINI, log10Mturn_max_MINI, log10Mturn_bin_width_MINI, log10Mturn_bin_width_inv_MINI;
float thistk;

int EvaluateSplineTable(bool MINI_HALOS, int dens_type, float curr_dens, float filtered_Mturn, float filtered_Mturn_MINI, float *Splined_Fcoll, float *Splined_Fcoll_MINI);
void InterpolationRange(int dens_type, float R, float L, float *min_density, float *max_density);

int ComputeIonizedBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options,
                       struct PerturbedField *perturbed_field,
                       struct PerturbedField *previous_perturbed_field,
                       struct IonizedBox *previous_ionize_box,
                       struct TsBox *spin_temp,
                       struct PerturbHaloField *halos,
                       struct InitialConditions *ini_boxes,
                       struct IonizedBox *box) {

    int status;

    Try{ // This Try brackets the whole function, so we don't indent.
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

    omp_set_num_threads(user_params->N_THREADS);

    // Other parameters used in the code
    int i,j,k,x,y,z, LAST_FILTER_STEP, first_step_R, short_completely_ionised,i_halo;
    int counter, N_halos_in_cell;
    unsigned long long ct;

    float growth_factor, pixel_mass, cell_length_factor, M_MIN, prev_growth_factor;
    float erfc_denom, erfc_denom_cell, res_xH, Splined_Fcoll, xHII_from_xrays, curr_dens, massofscaleR, ION_EFF_FACTOR, growth_factor_dz;
    float Splined_Fcoll_MINI, prev_dens, ION_EFF_FACTOR_MINI, prev_Splined_Fcoll, prev_Splined_Fcoll_MINI;
    float ave_M_coll_cell, ave_N_min_cell, pixel_volume, density_over_mean;

    float curr_vcb;

    double global_xH, ST_over_PS, f_coll, R, stored_R, f_coll_min;
    double ST_over_PS_MINI, f_coll_MINI, f_coll_min_MINI;

    double t_ast,  Gamma_R_prefactor, rec, dNrec, sigmaMmax;
    double Gamma_R_prefactor_MINI;
    float fabs_dtdz, ZSTEP, z_eff;
    const float dz = 0.01;

    float dens_val, prev_dens_val;

    int overdense_int,status_int;
    int something_finite_or_infinite = 0;
    int log10_Mturnover_MINI_int, log10_Mturnover_int;
    int *overdense_int_boundexceeded_threaded = calloc(user_params->N_THREADS,sizeof(int));

    if(user_params->USE_INTERPOLATION_TABLES) {
        overdense_large_min = global_params.CRIT_DENS_TRANSITION*0.999;
        overdense_large_bin_width = 1./((double)NSFR_high-1.)*(Deltac-overdense_large_min);
        overdense_large_bin_width_inv = 1./overdense_large_bin_width;

        prev_overdense_large_min = global_params.CRIT_DENS_TRANSITION*0.999;
        prev_overdense_large_bin_width = 1./((double)NSFR_high-1.)*(Deltac-prev_overdense_large_min);
        prev_overdense_large_bin_width_inv = 1./prev_overdense_large_bin_width;
    }

    double ave_log10_Mturnover, ave_log10_Mturnover_MINI;

    float Mlim_Fstar, Mlim_Fesc;
    float Mlim_Fstar_MINI, Mlim_Fesc_MINI;

    float Mcrit_atom, log10_Mcrit_atom, log10_Mcrit_mol;
    fftwf_complex *log10_Mturnover_unfiltered=NULL, *log10_Mturnover_filtered=NULL;
    fftwf_complex *log10_Mturnover_MINI_unfiltered=NULL, *log10_Mturnover_MINI_filtered=NULL;
    float log10_Mturnover, log10_Mturnover_MINI, Mcrit_LW, Mcrit_RE, Mturnover, Mturnover_MINI;

    float min_density, max_density;
    float prev_min_density, prev_max_density;

    float stored_redshift, adjustment_factor;

    gsl_rng * r[user_params->N_THREADS];

LOG_SUPER_DEBUG("initing heat");
    init_heat();
    float TK;
    TK = T_RECFAST(redshift,0);
    float cT_ad; //finding the adiabatic index at the initial redshift from 2302.08506 to fix adiabatic fluctuations.
    cT_ad = cT_approx(redshift);
LOG_SUPER_DEBUG("inited heat");

    init_ps();

LOG_SUPER_DEBUG("defined parameters");

    pixel_volume = pow(user_params->BOX_LEN/((float)(user_params->HII_DIM)), 3);


    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;
        ION_EFF_FACTOR_MINI = global_params.Pop3_ion * astro_params->F_STAR7_MINI * astro_params->F_ESC7_MINI;
    }
    else {
        ION_EFF_FACTOR = astro_params->HII_EFF_FACTOR;
        ION_EFF_FACTOR_MINI = 0.;
    }

    // For recombinations
    if(flag_options->INHOMO_RECO) {

        if(INIT_RECOMBINATIONS) {
            init_MHR();
            INIT_RECOMBINATIONS=0;
        }

     if (prev_redshift < 1) //deal with first redshift
		 ZSTEP = (1. + redshift) * (global_params.ZPRIME_STEP_FACTOR - 1.);
     else
        ZSTEP = prev_redshift - redshift;

#pragma omp parallel shared(box) private(ct) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++) {
                box->Gamma12_box[ct] = 0.0;
                box->MFP_box[ct] = 0.0;
            }
        }
    }
    else {
        ZSTEP = 0.2;
    }

#pragma omp parallel shared(box) private(ct) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++) {
            box->z_re_box[ct] = -1.0;
        }
    }

    LOG_SUPER_DEBUG("z_re_box init: ");
    debugSummarizeBox(box->z_re_box, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    fabs_dtdz = fabs(dtdz(redshift))/1e15; //reduce to have good precision
    t_ast = astro_params->t_STAR * t_hubble(redshift);
    growth_factor_dz = dicke(redshift-dz);

    // Modify the current sampled redshift to a redshift which matches the expected filling factor given our astrophysical parameterisation.
    // This is the photon non-conservation correction
    if(flag_options->PHOTON_CONS) {
        adjust_redshifts_for_photoncons(astro_params,flag_options,&redshift,&stored_redshift,&absolute_delta_z);
LOG_DEBUG("PhotonCons data:");
LOG_DEBUG("original redshift=%f, updated redshift=%f delta-z = %f", stored_redshift, redshift, absolute_delta_z);
        if(isfinite(redshift)==0 || isfinite(absolute_delta_z)==0) {
            LOG_ERROR("Updated photon non-conservation redshift is either infinite or NaN!");
            LOG_ERROR("This can sometimes occur when reionisation stalls (i.e. extremely low"\
                      "F_ESC or F_STAR or not enough sources)");
//            Throw(ParameterError);
            Throw(PhotonConsError);
        }
    }

    Splined_Fcoll = 0.;
    Splined_Fcoll_MINI = 0.;

    double ArgBinWidth, InvArgBinWidth, erfc_arg_val, erfc_arg_min, erfc_arg_max;
    int erfc_arg_val_index, ERFC_NUM_POINTS;

    erfc_arg_val = 0.;
    erfc_arg_val_index = 0;

    // Setup an interpolation table for the error function, helpful for calcluating the collapsed fraction
    // (only for the default model, i.e. mass-independent ionising efficiency)
    erfc_arg_min = -15.0;
    erfc_arg_max = 15.0;

    ERFC_NUM_POINTS = 10000;

    ArgBinWidth = (erfc_arg_max - erfc_arg_min)/((double)ERFC_NUM_POINTS - 1.);
    InvArgBinWidth = 1./ArgBinWidth;

    if(!flag_options->USE_MASS_DEPENDENT_ZETA && INIT_ERFC_INTERPOLATION) {

        ERFC_VALS = calloc(ERFC_NUM_POINTS,sizeof(double));
        ERFC_VALS_DIFF = calloc(ERFC_NUM_POINTS,sizeof(double));

#pragma omp parallel shared(ERFC_VALS,erfc_arg_min,ArgBinWidth) private(i,erfc_arg_val) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for(i=0;i<ERFC_NUM_POINTS;i++) {

                erfc_arg_val = erfc_arg_min + ArgBinWidth*(double)i;

                ERFC_VALS[i] = splined_erfc(erfc_arg_val);
            }
        }

#pragma omp parallel shared(ERFC_VALS_DIFF,ERFC_VALS) private(i) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for(i=0;i<(ERFC_NUM_POINTS-1);i++) {
                ERFC_VALS_DIFF[i] = ERFC_VALS[i+1] - ERFC_VALS[i];
            }
        }

        INIT_ERFC_INTERPOLATION = 0;
    }

LOG_SUPER_DEBUG("erfc interpolation done");

    /////////////////////////////////   BEGIN INITIALIZATION   //////////////////////////////////

    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        LOG_WARNING("Resolution is likely too low for accurate evolved density fields\n It Is recommended \
                    that you either increase the resolution (DIM/Box_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }

    // initialize power spectrum
    growth_factor = dicke(redshift);
    prev_growth_factor = dicke(prev_redshift);

    fftwf_complex *deltax_unfiltered, *deltax_unfiltered_original, *deltax_filtered;
    fftwf_complex *xe_unfiltered, *xe_filtered, *N_rec_unfiltered, *N_rec_filtered;
    fftwf_complex *prev_deltax_unfiltered, *prev_deltax_filtered;
    fftwf_complex *M_coll_unfiltered,*M_coll_filtered;

    deltax_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deltax_unfiltered_original = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deltax_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    if (flag_options->USE_MINI_HALOS){
        prev_deltax_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        prev_deltax_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }

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

        if(user_params->USE_INTERPOLATION_TABLES) {
            log10_overdense_spline_SFR = calloc(NSFR_low,sizeof(double));
            Overdense_spline_SFR = calloc(NSFR_high,sizeof(float));

            log10_Nion_spline = calloc(NSFR_low,sizeof(float));
            Nion_spline = calloc(NSFR_high,sizeof(float));

            if (flag_options->USE_MINI_HALOS){
                prev_log10_overdense_spline_SFR = calloc(NSFR_low,sizeof(double));
                prev_Overdense_spline_SFR = calloc(NSFR_high,sizeof(float));
                log10_Nion_spline = calloc(NSFR_low*NMTURN,sizeof(float));
                Nion_spline = calloc(NSFR_high*NMTURN,sizeof(float));
                log10_Nion_spline_MINI = calloc(NSFR_low*NMTURN,sizeof(float));
                Nion_spline_MINI = calloc(NSFR_high*NMTURN,sizeof(float));
                prev_log10_Nion_spline = calloc(NSFR_low*NMTURN,sizeof(float));
                prev_Nion_spline = calloc(NSFR_high*NMTURN,sizeof(float));
                prev_log10_Nion_spline_MINI = calloc(NSFR_low*NMTURN,sizeof(float));
                prev_Nion_spline_MINI = calloc(NSFR_high*NMTURN,sizeof(float));
            }
        }

        if (flag_options->USE_MINI_HALOS){
            Mturns = calloc(NMTURN,sizeof(float));
            Mturns_MINI = calloc(NMTURN,sizeof(float));
        }
    }

    // Calculate the density field for this redshift if the initial conditions/cosmology are changing

    if(flag_options->PHOTON_CONS) {
        adjustment_factor = dicke(redshift)/dicke(stored_redshift);
    }
    else {
        adjustment_factor = 1.;
    }

#pragma omp parallel shared(deltax_unfiltered,perturbed_field,adjustment_factor) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<HII_D_PARA; k++){
                    *((float *)deltax_unfiltered + HII_R_FFT_INDEX(i,j,k)) = (perturbed_field->density[HII_R_INDEX(i,j,k)])*adjustment_factor;
                }
            }
        }
    }

LOG_SUPER_DEBUG("density field calculated");

    // keep the unfiltered density field in an array, to save it for later
    memcpy(deltax_unfiltered_original, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    i=0;

    // Newer setup to be performed in parallel
    int thread_num;
    for(thread_num = 0; thread_num < user_params->N_THREADS; thread_num++){
        // Original defaults with gsl_rng_mt19937 and SEED = 0, thus start with this and iterate for all other threads by their thread number
        r[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(r[thread_num], thread_num);
    }

    pixel_mass = RtoM(L_FACTOR*user_params->BOX_LEN/(float)(user_params->HII_DIM));
    cell_length_factor = L_FACTOR;

    if(flag_options->USE_HALO_FIELD && (global_params.FIND_BUBBLE_ALGORITHM == 2) && ((user_params->BOX_LEN/(float)(user_params->HII_DIM) < 1))) {
        cell_length_factor = 1.;
    }

    if (prev_redshift < 1){
LOG_DEBUG("first redshift, do some initialization");
        previous_ionize_box->z_re_box    = (float *) calloc(HII_TOT_NUM_PIXELS, sizeof(float));
#pragma omp parallel shared(previous_ionize_box) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        previous_ionize_box->z_re_box[HII_R_INDEX(i, j, k)] = -1.0;
                    }
                }
            }
        }
        if (flag_options->INHOMO_RECO)
            previous_ionize_box->dNrec_box   = (float *) calloc(HII_TOT_NUM_PIXELS, sizeof(float));
    }
    //set the minimum source mass
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        if (flag_options->USE_MINI_HALOS){
            ave_log10_Mturnover = 0.;
            ave_log10_Mturnover_MINI = 0.;

            // this is the first z, and the previous_ionize_box  are empty
            if (prev_redshift < 1){
                previous_ionize_box->Gamma12_box = (float *) calloc(HII_TOT_NUM_PIXELS, sizeof(float));
                // really painful to get the length...
                counter = 1;
                R=fmax(global_params.R_BUBBLE_MIN, (cell_length_factor*user_params->BOX_LEN/(float)user_params->HII_DIM));
                while ((R - fmin(astro_params->R_BUBBLE_MAX, L_FACTOR*user_params->BOX_LEN)) <= FRACT_FLOAT_ERR ){
                    if(R >= fmin(astro_params->R_BUBBLE_MAX, L_FACTOR*user_params->BOX_LEN)) {
                        stored_R = R/(global_params.DELTA_R_HII_FACTOR);
                    }
                    R*= global_params.DELTA_R_HII_FACTOR;
                    counter += 1;
                }

                previous_ionize_box->Fcoll       = (float *) calloc(HII_TOT_NUM_PIXELS*counter, sizeof(float));
                previous_ionize_box->Fcoll_MINI  = (float *) calloc(HII_TOT_NUM_PIXELS*counter, sizeof(float));
                previous_ionize_box->mean_f_coll = 0.0;
                previous_ionize_box->mean_f_coll_MINI = 0.0;

#pragma omp parallel shared(prev_deltax_unfiltered) private(i,j,k) num_threads(user_params->N_THREADS)
                {
#pragma omp for
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<HII_D_PARA; k++){
                                *((float *)prev_deltax_unfiltered + HII_R_FFT_INDEX(i,j,k)) = -1.5;
                            }
                        }
                    }
                }
            }
            else{
#pragma omp parallel shared(prev_deltax_unfiltered,previous_perturbed_field) private(i,j,k) num_threads(user_params->N_THREADS)
                {
#pragma omp for
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<HII_D_PARA; k++){
                                *((float *)prev_deltax_unfiltered + HII_R_FFT_INDEX(i,j,k)) = previous_perturbed_field->density[HII_R_INDEX(i,j,k)];
                            }
                        }
                    }
                }
            }

LOG_SUPER_DEBUG("previous density field calculated");

            // fields added for minihalos
            Mcrit_atom              = atomic_cooling_threshold(redshift);
            log10_Mcrit_atom        = log10(Mcrit_atom);
            log10_Mcrit_mol         = log10(lyman_werner_threshold(redshift, 0.,0., astro_params));
            log10_Mturnover_unfiltered      = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            log10_Mturnover_filtered        = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            log10_Mturnover_MINI_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            log10_Mturnover_MINI_filtered   = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

            if (!log10_Mturnover_unfiltered || !log10_Mturnover_filtered || !log10_Mturnover_MINI_unfiltered || !log10_Mturnover_MINI_filtered){// || !Mcrit_RE_grid || !Mcrit_LW_grid)
                LOG_ERROR("Error allocating memory for Mturnover or Mturnover_MINI boxes");
                Throw(MemoryAllocError);
            }
LOG_SUPER_DEBUG("Calculating and outputting Mcrit boxes for atomic and molecular halos...");

#pragma omp parallel shared(redshift,previous_ionize_box,spin_temp,Mcrit_atom,log10_Mturnover_unfiltered,log10_Mturnover_MINI_unfiltered)\
                    private(x,y,z,Mcrit_RE,Mcrit_LW,Mturnover,Mturnover_MINI,log10_Mturnover,log10_Mturnover_MINI,curr_vcb) num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:ave_log10_Mturnover,ave_log10_Mturnover_MINI)
                for (x=0; x<user_params->HII_DIM; x++){
                    for (y=0; y<user_params->HII_DIM; y++){
                        for (z=0; z<HII_D_PARA; z++){

                            Mcrit_RE = reionization_feedback(redshift, previous_ionize_box->Gamma12_box[HII_R_INDEX(x, y, z)], previous_ionize_box->z_re_box[HII_R_INDEX(x, y, z)]);
                            if (flag_options->FIX_VCB_AVG){ //with this flag we ignore reading vcb box
                              curr_vcb = global_params.VAVG;
                            }
                            else{
                              if(user_params->USE_RELATIVE_VELOCITIES ){
                                curr_vcb = ini_boxes->lowres_vcb[HII_R_INDEX(x,y,z)];
                              }
                              else{ //set vcb to a constant, either zero or vavg.
                                curr_vcb = 0.0;
                              }
                            }

                            Mcrit_LW = lyman_werner_threshold(redshift, spin_temp->J_21_LW_box[HII_R_INDEX(x, y, z)], curr_vcb, astro_params);

                            //JBM: this only accounts for effect 3 (largest on minihaloes). Effects 1 and 2 affect both minihaloes (MCGs) and regular ACGs, but they're smaller ~10%. See Sec 2 of Muñoz+21 (2110.13919)


                            //*((float *)Mcrit_RE_grid + HII_R_FFT_INDEX(x,y,z)) = Mcrit_RE;
                            //*((float *)Mcrit_LW_grid + HII_R_FFT_INDEX(x,y,z)) = Mcrit_LW;
                            Mturnover            = Mcrit_RE > Mcrit_atom ? Mcrit_RE : Mcrit_atom;
                            Mturnover_MINI       = Mcrit_RE > Mcrit_LW   ? Mcrit_RE : Mcrit_LW;
                            log10_Mturnover      = log10(Mturnover);
                            log10_Mturnover_MINI = log10(Mturnover_MINI);

                            *((float *)log10_Mturnover_unfiltered      + HII_R_FFT_INDEX(x,y,z)) = log10_Mturnover;
                            *((float *)log10_Mturnover_MINI_unfiltered + HII_R_FFT_INDEX(x,y,z)) = log10_Mturnover_MINI;

                            ave_log10_Mturnover      += log10_Mturnover;
                            ave_log10_Mturnover_MINI += log10_Mturnover_MINI;
                        }
                    }
                }
            }

            box->log10_Mturnover_ave      = ave_log10_Mturnover/(double) HII_TOT_NUM_PIXELS;
            box->log10_Mturnover_MINI_ave = ave_log10_Mturnover_MINI/(double) HII_TOT_NUM_PIXELS;
            Mturnover                 = pow(10., box->log10_Mturnover_ave);
            Mturnover_MINI            = pow(10., box->log10_Mturnover_MINI_ave);
            M_MIN           = global_params.M_MIN_INTEGRAL;
            Mlim_Fstar_MINI = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_STAR_MINI, astro_params->F_STAR7_MINI * pow(1e3,astro_params->ALPHA_STAR_MINI));
            Mlim_Fesc_MINI  = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_ESC, astro_params->F_ESC7_MINI * pow(1e3, astro_params->ALPHA_ESC));
LOG_SUPER_DEBUG("average turnover masses are %.2f and %.2f for ACGs and MCGs", box->log10_Mturnover_ave, box->log10_Mturnover_MINI_ave);
        }
        else{
            M_MIN     = astro_params->M_TURN/50.;
            Mturnover = astro_params->M_TURN;
            box->log10_Mturnover_ave = log10(Mturnover);
            box->log10_Mturnover_MINI_ave = log10(Mturnover);
        }
        Mlim_Fstar = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_STAR, astro_params->F_STAR10);
        Mlim_Fesc  = Mass_limit_bisection(M_MIN, 1e16, astro_params->ALPHA_ESC, astro_params->F_ESC10);
    }
    else {

        //set the minimum source mass
        if (astro_params->ION_Tvir_MIN < 9.99999e3) { // neutral IGM
            M_MIN = (float)TtoM(redshift, astro_params->ION_Tvir_MIN, 1.22);
        }
        else { // ionized IGM
            M_MIN = (float)TtoM(redshift, astro_params->ION_Tvir_MIN, 0.6);
        }
    }

LOG_SUPER_DEBUG("minimum source mass has been set: %f", M_MIN);

    if(user_params->USE_INTERPOLATION_TABLES) {
        if(user_params->FAST_FCOLL_TABLES){
            initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
        }
        else if(flag_options->USE_MINI_HALOS){
            initialiseSigmaMInterpTable(global_params.M_MIN_INTEGRAL/50.,1e20);
        }else{
            initialiseSigmaMInterpTable(M_MIN,1e20);
        }
    }


LOG_SUPER_DEBUG("sigma table has been initialised");

    // check for WDM

    if (global_params.P_CUTOFF && ( M_MIN < M_J_WDM())){
        LOG_WARNING("The default Jeans mass of %e Msun is smaller than the scale supressed by the effective pressure of WDM.", M_MIN);
        M_MIN = M_J_WDM();
        LOG_WARNING("Setting a new effective Jeans mass from WDM pressure supression of %e Msun", M_MIN);
    }

    // ARE WE USING A DISCRETE HALO FIELD (identified in the ICs with FindHaloes.c and evolved  with PerturbHaloField.c)
    if(flag_options->USE_HALO_FIELD) {
        M_coll_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        M_coll_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

#pragma omp parallel shared(M_coll_unfiltered) private(ct) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (ct=0; ct<HII_TOT_FFT_NUM_PIXELS; ct++){
                *((float *)M_coll_unfiltered + ct) = 0;
            }
        }

#pragma omp parallel shared(M_coll_unfiltered,halos) \
                    private(i_halo,x,y,z) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i_halo=0; i_halo<halos->n_halos; i_halo++){
                x = halos->halo_coords[0+3*i_halo];
                y = halos->halo_coords[1+3*i_halo];
                z = halos->halo_coords[2+3*i_halo];

#pragma omp atomic
                *((float *)M_coll_unfiltered + HII_R_FFT_INDEX(x, y, z)) += halos->halo_masses[i_halo];
            }
        }
    } // end of the USE_HALO_FIELD option




    // lets check if we are going to bother with computing the inhmogeneous field at all...
    global_xH = 0.0;

    // Determine the normalisation for the excursion set algorithm
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        if (flag_options->USE_MINI_HALOS){
            if (previous_ionize_box->mean_f_coll * ION_EFF_FACTOR < 1e-4){
                box->mean_f_coll = Nion_General(redshift,M_MIN,Mturnover,astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,
                                                astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc);
            }
            else{
                box->mean_f_coll = previous_ionize_box->mean_f_coll + \
                                    Nion_General(redshift,M_MIN,Mturnover,astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,
                                                 astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc) - \
                                    Nion_General(prev_redshift,M_MIN,Mturnover,astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,
                                                 astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc);
            }
            if (previous_ionize_box->mean_f_coll_MINI * ION_EFF_FACTOR_MINI < 1e-4){
                box->mean_f_coll_MINI = Nion_General_MINI(redshift,M_MIN,Mturnover_MINI,Mcrit_atom,
                                                          astro_params->ALPHA_STAR_MINI,astro_params->ALPHA_ESC,astro_params->F_STAR7_MINI,
                                                          astro_params->F_ESC7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI);
            }
            else{
                box->mean_f_coll_MINI = previous_ionize_box->mean_f_coll_MINI + \
                                        Nion_General_MINI(redshift,M_MIN,Mturnover_MINI,Mcrit_atom,astro_params->ALPHA_STAR_MINI,
                                                          astro_params->ALPHA_ESC,astro_params->F_STAR7_MINI,astro_params->F_ESC7_MINI
                                                          ,Mlim_Fstar_MINI,Mlim_Fesc_MINI) - \
                                        Nion_General_MINI(prev_redshift,M_MIN,Mturnover_MINI,Mcrit_atom,astro_params->ALPHA_STAR_MINI,
                                                          astro_params->ALPHA_ESC,astro_params->F_STAR7_MINI,astro_params->F_ESC7_MINI,
                                                          Mlim_Fstar_MINI,Mlim_Fesc_MINI);
            }
            f_coll_min = Nion_General(global_params.Z_HEAT_MAX,M_MIN,Mturnover,astro_params->ALPHA_STAR,
                                      astro_params->ALPHA_ESC,astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc);
            f_coll_min_MINI = Nion_General_MINI(global_params.Z_HEAT_MAX,M_MIN,Mturnover_MINI,Mcrit_atom,
                                                astro_params->ALPHA_STAR_MINI,astro_params->ALPHA_ESC,astro_params->F_STAR7_MINI,
                                                astro_params->F_ESC7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI);
        }
        else{
            box->mean_f_coll = Nion_General(redshift,M_MIN,Mturnover,astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,
                                            astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc);
            box->mean_f_coll_MINI = 0.;
            f_coll_min = Nion_General(global_params.Z_HEAT_MAX,M_MIN,Mturnover,astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,
                                      astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc);
        }
    }
    else {
        LOG_DEBUG("Setting mean collapse fraction");
        box->mean_f_coll = FgtrM_General(redshift, M_MIN);
    }

    if(isfinite(box->mean_f_coll)==0) {
        LOG_ERROR("Mean collapse fraction is either infinite or NaN!");
//        Throw(ParameterError);
        Throw(InfinityorNaNError);
    }
LOG_SUPER_DEBUG("excursion set normalisation, mean_f_coll: %e", box->mean_f_coll);

    if (flag_options->USE_MINI_HALOS){
        if(isfinite(box->mean_f_coll_MINI)==0) {
            LOG_ERROR("Mean collapse fraction of MINI is either infinite or NaN!");
//            Throw(ParameterError);
            Throw(InfinityorNaNError);
        }
LOG_SUPER_DEBUG("excursion set normalisation, mean_f_coll_MINI: %e", box->mean_f_coll_MINI);
    }


    if (box->mean_f_coll * ION_EFF_FACTOR + box->mean_f_coll_MINI * ION_EFF_FACTOR_MINI< global_params.HII_ROUND_ERR){ // way too small to ionize anything...
    //        printf( "The mean collapse fraction is %e, which is much smaller than the effective critical collapse fraction of %e\n I will just declare everything to be neutral\n", mean_f_coll, f_coll_crit);

        // find the neutral fraction
        if(flag_options->USE_TS_FLUCT) {
#pragma omp parallel shared(box,spin_temp) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:global_xH)
                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                    box->xH_box[ct] = 1.-spin_temp->x_e_box[ct]; // convert from x_e to xH
                    global_xH += box->xH_box[ct];
                    box->temp_kinetic_all_gas[ct] = spin_temp->Tk_box[ct];
                }
            }
            global_xH /= (double)HII_TOT_NUM_PIXELS;
        }
        else {
            global_xH = 1. - xion_RECFAST(redshift, 0);

#pragma omp parallel shared(box,global_xH,TK,perturbed_field,cT_ad) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                    box->xH_box[ct] = global_xH;
                    box->temp_kinetic_all_gas[ct] = TK * (1.0 + cT_ad * perturbed_field->density[ct]); // Is perturbed_field defined already here? we need it for cT. I'm also assuming we don't need to multiply by other z here.
                }
            }
        }
    }
    else {

        // Take the ionisation fraction from the X-ray ionisations from Ts.c (only if the calculate spin temperature flag is set)
        if (flag_options->USE_TS_FLUCT) {
#pragma omp parallel shared(xe_unfiltered, spin_temp) private(i, j, k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i = 0; i < user_params->HII_DIM; i++) {
                    for (j = 0; j < user_params->HII_DIM; j++) {
                        for (k = 0; k < HII_D_PARA; k++) {
                            *((float *) xe_unfiltered + HII_R_FFT_INDEX(i, j, k)) = spin_temp->x_e_box[HII_R_INDEX(i, j, k)];
                        }
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("calculated ionization fraction");

        if (flag_options->INHOMO_RECO) {
#pragma omp parallel shared(N_rec_unfiltered, previous_ionize_box) private(i, j, k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i = 0; i < user_params->HII_DIM; i++) {
                    for (j = 0; j < user_params->HII_DIM; j++) {
                        for (k = 0; k < HII_D_PARA; k++) {
                            *((float *) N_rec_unfiltered +
                              HII_R_FFT_INDEX(i, j, k)) = previous_ionize_box->dNrec_box[HII_R_INDEX(i, j, k)];
                        }
                    }
                }
            }
        }

        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, deltax_unfiltered);

        LOG_SUPER_DEBUG("FFTs performed");

        if(flag_options->USE_MINI_HALOS){
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, prev_deltax_unfiltered);
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, log10_Mturnover_MINI_unfiltered);
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, log10_Mturnover_unfiltered);
            LOG_SUPER_DEBUG("MINI HALO ffts performed");
        }

        if (flag_options->USE_HALO_FIELD){
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, M_coll_unfiltered);
            LOG_SUPER_DEBUG("HALO_FIELD ffts performed");
        }

        if(flag_options->USE_TS_FLUCT) {
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, xe_unfiltered);
            LOG_SUPER_DEBUG("Ts ffts performed");
        }


        if (flag_options->INHOMO_RECO) {
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, N_rec_unfiltered);
        }

        // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from
        //  real space to k-space
        // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
#pragma omp parallel shared(deltax_unfiltered,xe_unfiltered,N_rec_unfiltered,prev_deltax_unfiltered,\
                            log10_Mturnover_unfiltered,log10_Mturnover_MINI_unfiltered,M_coll_unfiltered) \
                    private(ct) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                deltax_unfiltered[ct] /= (HII_TOT_NUM_PIXELS+0.0);
                if(flag_options->USE_TS_FLUCT) { xe_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS; }
                if (flag_options->INHOMO_RECO){ N_rec_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS; }
                if(flag_options->USE_HALO_FIELD) { M_coll_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS; }
                if(flag_options->USE_MINI_HALOS){
                    prev_deltax_unfiltered[ct]          /= (HII_TOT_NUM_PIXELS+0.0);
                    log10_Mturnover_unfiltered[ct]      /= (HII_TOT_NUM_PIXELS+0.0);
                    log10_Mturnover_MINI_unfiltered[ct] /= (HII_TOT_NUM_PIXELS+0.0);
                }
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
        erfc_denom_cell = 1; //dummy value

        R=fmax(global_params.R_BUBBLE_MIN, (cell_length_factor*user_params->BOX_LEN/(float)user_params->HII_DIM));

        while ((R - fmin(astro_params->R_BUBBLE_MAX, L_FACTOR * user_params->BOX_LEN)) <= FRACT_FLOAT_ERR) {
            R *= global_params.DELTA_R_HII_FACTOR;
            if (R >= fmin(astro_params->R_BUBBLE_MAX, L_FACTOR * user_params->BOX_LEN)) {
                stored_R = R / (global_params.DELTA_R_HII_FACTOR);
            }
        }

        LOG_DEBUG("set max radius: %f", R);

        R=fmin(astro_params->R_BUBBLE_MAX, L_FACTOR*user_params->BOX_LEN);

        LAST_FILTER_STEP = 0;

        first_step_R = 1;

        double R_temp = (double) (astro_params->R_BUBBLE_MAX);

        counter = 0;

        while (!LAST_FILTER_STEP && (M_MIN < RtoM(R)) ){
LOG_ULTRA_DEBUG("while loop for until RtoM(R)=%f reaches M_MIN=%f", RtoM(R), M_MIN);

            // Check if we are the last filter step
            if ( ((R/(global_params.DELTA_R_HII_FACTOR) - cell_length_factor*(user_params->BOX_LEN)/(float)(user_params->HII_DIM)) <= FRACT_FLOAT_ERR) || \
                    ((R/(global_params.DELTA_R_HII_FACTOR) - global_params.R_BUBBLE_MIN) <= FRACT_FLOAT_ERR) ) {
                LAST_FILTER_STEP = 1;
                R = fmax(cell_length_factor*user_params->BOX_LEN/(double)(user_params->HII_DIM), global_params.R_BUBBLE_MIN);
            }

            // Copy all relevant quantities from memory into new arrays to be smoothed and FFT'd.
            if (flag_options->USE_TS_FLUCT) {
                memcpy(xe_filtered, xe_unfiltered, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
            }
            if (flag_options->INHOMO_RECO) {
                memcpy(N_rec_filtered, N_rec_unfiltered, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
            }
            if (flag_options->USE_HALO_FIELD) {
                memcpy(M_coll_filtered, M_coll_unfiltered, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
            }

            memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

            if(flag_options->USE_MINI_HALOS){
                memcpy(prev_deltax_filtered, prev_deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                memcpy(log10_Mturnover_MINI_filtered, log10_Mturnover_MINI_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                memcpy(log10_Mturnover_filtered, log10_Mturnover_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            }

            if (!LAST_FILTER_STEP ||
                ((R - cell_length_factor * (user_params->BOX_LEN / (double) (user_params->HII_DIM))) >
                 FRACT_FLOAT_ERR)) {
                if (flag_options->USE_TS_FLUCT) {
                    filter_box(xe_filtered, 1, global_params.HII_FILTER, R);
                }
                if (flag_options->INHOMO_RECO) {
                    filter_box(N_rec_filtered, 1, global_params.HII_FILTER, R);
                }
                if (flag_options->USE_HALO_FIELD) {
                    filter_box(M_coll_filtered, 1, global_params.HII_FILTER, R);
                }
                filter_box(deltax_filtered, 1, global_params.HII_FILTER, R);
                if(flag_options->USE_MINI_HALOS){
                    filter_box(prev_deltax_filtered, 1, global_params.HII_FILTER, R);
                    filter_box(log10_Mturnover_MINI_filtered, 1, global_params.HII_FILTER, R);
                    filter_box(log10_Mturnover_filtered, 1, global_params.HII_FILTER, R);
                }
            }

            // Perform FFTs
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, deltax_filtered);

            if(flag_options->USE_MINI_HALOS){
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, prev_deltax_filtered);
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, log10_Mturnover_MINI_filtered);
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, log10_Mturnover_filtered);
            }

            if (flag_options->USE_HALO_FIELD) {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, M_coll_filtered);
            }

            if (flag_options->USE_TS_FLUCT) {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, xe_filtered);
            }

            if (flag_options->INHOMO_RECO) {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, N_rec_filtered);
            }

            // Check if this is the last filtering scale.  If so, we don't need deltax_unfiltered anymore.
            // We will re-read it to get the real-space field, which we will use to set the residual neutral fraction
            ST_over_PS = 0;
            ST_over_PS_MINI = 0;
            f_coll = 0;
            f_coll_MINI = 0;
            massofscaleR = RtoM(R);

            if(!user_params->USE_INTERPOLATION_TABLES) {
                sigmaMmax = sigma_z0(massofscaleR);
            }

            if (!flag_options->USE_HALO_FIELD) {
                if (flag_options->USE_MASS_DEPENDENT_ZETA) {

                    min_density = max_density = 0.0;

#pragma omp parallel shared(deltax_filtered) private(x, y, z) num_threads(user_params->N_THREADS)
                    {
#pragma omp for reduction(max:max_density) reduction(min:min_density)
                        for (x = 0; x < user_params->HII_DIM; x++) {
                            for (y = 0; y < user_params->HII_DIM; y++) {
                                for (z = 0; z < HII_D_PARA; z++) {
                                    // delta cannot be less than -1
                                    *((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z)) = fmaxf(
                                                *((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z)), -1. + FRACT_FLOAT_ERR);

                                    if (*((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z)) < min_density) {
                                                min_density = *((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z));
                                    }
                                    if (*((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z)) > max_density) {
                                                max_density = *((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z));
                                    }
                                }
                            }
                        }
                    }

                    if(user_params->USE_INTERPOLATION_TABLES) {
                        InterpolationRange(1,R,user_params->BOX_LEN,&min_density, &max_density);

                    }

                    if (flag_options->USE_MINI_HALOS){
                        // do the same for prev
                        prev_min_density = prev_max_density = 0.0;

#pragma omp parallel shared(prev_deltax_filtered) private(x, y, z) num_threads(user_params->N_THREADS)
                        {
#pragma omp for reduction(max:prev_max_density) reduction(min:prev_min_density)
                            for (x=0; x<user_params->HII_DIM; x++){
                                for (y=0; y<user_params->HII_DIM; y++){
                                    for (z=0; z<HII_D_PARA; z++){
                                        // delta cannot be less than -1
                                        *((float *)prev_deltax_filtered + HII_R_FFT_INDEX(x,y,z)) = \
                                                        fmaxf(*((float *)prev_deltax_filtered + HII_R_FFT_INDEX(x,y,z)) , -1.+FRACT_FLOAT_ERR);

                                        if( *((float *)prev_deltax_filtered + HII_R_FFT_INDEX(x,y,z)) < prev_min_density ) {
                                            prev_min_density = *((float *)prev_deltax_filtered + HII_R_FFT_INDEX(x,y,z));
                                        }
                                        if( *((float *)prev_deltax_filtered + HII_R_FFT_INDEX(x,y,z)) > prev_max_density ) {
                                            prev_max_density = *((float *)prev_deltax_filtered + HII_R_FFT_INDEX(x,y,z));
                                        }
                                    }
                                }
                            }
                        }

                        if(user_params->USE_INTERPOLATION_TABLES) {
                            InterpolationRange(2,R,user_params->BOX_LEN,&prev_min_density, &prev_max_density);
                        }

                        // do the same for logM
                        log10Mturn_min = 999;
                        log10Mturn_max = 0.0;
                        log10Mturn_min_MINI = 999;
                        log10Mturn_max_MINI = 0.0;

#pragma omp parallel shared(log10_Mturnover_filtered,log10_Mturnover_MINI_filtered,log10_Mcrit_atom,log10_Mcrit_mol) private(x, y, z) num_threads(user_params->N_THREADS)
                        {
#pragma omp for reduction(max:log10Mturn_max,log10Mturn_max_MINI) reduction(min:log10Mturn_min,log10Mturn_min_MINI)
                            for (x=0; x<user_params->HII_DIM; x++){
                                for (y=0; y<user_params->HII_DIM; y++){
                                    for (z=0; z<HII_D_PARA; z++){
                                        if (*((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z)) < log10_Mcrit_atom)
                                            *((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z)) = log10_Mcrit_atom;
                                        if (*((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z)) > LOG10_MTURN_MAX)
                                            *((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z)) = LOG10_MTURN_MAX;
                                        // Mturnover cannot be less than Mcrit_mol
                                        if (*((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) < log10_Mcrit_mol)
                                            *((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) = log10_Mcrit_mol;
                                        if (*((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) > LOG10_MTURN_MAX)
                                            *((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) = LOG10_MTURN_MAX;

                                        if (*((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z)) < log10Mturn_min)
                                            log10Mturn_min = *((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z));
                                        if (*((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z)) > log10Mturn_max)
                                            log10Mturn_max = *((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z));
                                        if (*((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) < log10Mturn_min_MINI)
                                            log10Mturn_min_MINI = *((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z));
                                        if (*((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) > log10Mturn_max_MINI)
                                            log10Mturn_max_MINI = *((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z));
                                    }
                                }
                            }
                        }

                        if(user_params->USE_INTERPOLATION_TABLES) {
                            log10Mturn_min = log10Mturn_min *0.99;
                            log10Mturn_max = log10Mturn_max *1.01;
                            log10Mturn_min_MINI = log10Mturn_min_MINI *0.99;
                            log10Mturn_max_MINI = log10Mturn_max_MINI *1.01;

                            log10Mturn_bin_width = (log10Mturn_max - log10Mturn_min) / NMTURN;
                            log10Mturn_bin_width_inv = 1./log10Mturn_bin_width;
                            log10Mturn_bin_width_MINI = (log10Mturn_max_MINI - log10Mturn_min_MINI) / NMTURN;
                            log10Mturn_bin_width_inv_MINI = 1./log10Mturn_bin_width_MINI;
                        }
                    }

                    initialiseGL_Nion(NGL_SFR, M_MIN,massofscaleR);

                    if(user_params->USE_INTERPOLATION_TABLES) {
                        if(flag_options->USE_MINI_HALOS){
                            initialise_Nion_General_spline_MINI(redshift,Mcrit_atom,min_density,max_density,massofscaleR,M_MIN,
                                                    log10Mturn_min,log10Mturn_max,log10Mturn_min_MINI,log10Mturn_max_MINI,
                                                    astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI,
                                                    astro_params->ALPHA_ESC,astro_params->F_STAR10,
                                                    astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc,astro_params->F_STAR7_MINI,
                                                    astro_params->F_ESC7_MINI,Mlim_Fstar_MINI, Mlim_Fesc_MINI, user_params->FAST_FCOLL_TABLES);

                            if (previous_ionize_box->mean_f_coll_MINI * ION_EFF_FACTOR_MINI + previous_ionize_box->mean_f_coll * ION_EFF_FACTOR > 1e-4){
                                    initialise_Nion_General_spline_MINI_prev(prev_redshift,Mcrit_atom,prev_min_density,prev_max_density,
                                                                    massofscaleR,M_MIN,log10Mturn_min,log10Mturn_max,log10Mturn_min_MINI,
                                                                    log10Mturn_max_MINI,astro_params->ALPHA_STAR,  astro_params->ALPHA_STAR_MINI,
                                                                    astro_params->ALPHA_ESC,
                                                                    astro_params->F_STAR10,astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc,
                                                                    astro_params->F_STAR7_MINI,astro_params->F_ESC7_MINI,
                                                                    Mlim_Fstar_MINI, Mlim_Fesc_MINI, user_params->FAST_FCOLL_TABLES);
                            }
                        }
                        else{
                            initialise_Nion_General_spline(redshift,min_density,max_density,massofscaleR,astro_params->M_TURN,
                                                        astro_params->ALPHA_STAR,astro_params->ALPHA_ESC,astro_params->F_STAR10,
                                                        astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc, user_params->FAST_FCOLL_TABLES);
                        }
                    }
                }
                else {

                    erfc_denom = 2. * (pow(sigma_z0(M_MIN), 2) - pow(sigma_z0(massofscaleR), 2));
                    if (erfc_denom < 0) { // our filtering scale has become too small
                        break;
                    }
                    erfc_denom = sqrt(erfc_denom);
                    erfc_denom = 1. / (growth_factor * erfc_denom);

                }
            }

            // Determine the global averaged f_coll for the overall normalisation

            // Reset value of int check to see if we are over-stepping our interpolation table
            for (i = 0; i < user_params->N_THREADS; i++) {
                overdense_int_boundexceeded_threaded[i] = 0;
            }

            // renormalize the collapse fraction so that the mean matches ST,
            // since we are using the evolved (non-linear) density field
#pragma omp parallel shared(deltax_filtered,N_rec_filtered,xe_filtered,overdense_int_boundexceeded_threaded,log10_Nion_spline,Nion_spline,erfc_denom,erfc_arg_min,\
                            erfc_arg_max,InvArgBinWidth,ArgBinWidth,ERFC_VALS_DIFF,ERFC_VALS,log10_Mturnover_filtered,log10Mturn_min,log10Mturn_bin_width_inv, \
                            log10_Mturnover_MINI_filtered,log10Mturn_bin_width_inv_MINI,log10_Nion_spline_MINI,prev_deltax_filtered,previous_ionize_box,ION_EFF_FACTOR,\
                            prev_overdense_small_bin_width, overdense_small_bin_width,overdense_small_bin_width_inv,\
                            prev_overdense_small_min,prev_overdense_small_bin_width_inv,prev_log10_Nion_spline,prev_log10_Nion_spline_MINI,prev_overdense_large_min,\
                            prev_overdense_large_bin_width_inv,prev_Nion_spline,prev_Nion_spline_MINI,box,counter,M_coll_filtered,massofscaleR,pixel_volume,sigmaMmax,\
                            M_MIN,growth_factor,Mlim_Fstar,Mlim_Fesc,Mcrit_atom,Mlim_Fstar_MINI,Mlim_Fesc_MINI,prev_growth_factor) \
                    private(x,y,z,curr_dens,Splined_Fcoll,Splined_Fcoll_MINI,dens_val,overdense_int,erfc_arg_val,erfc_arg_val_index,log10_Mturnover,\
                            log10_Mturnover_int,log10_Mturnover_MINI,log10_Mturnover_MINI_int,prev_dens,prev_Splined_Fcoll,prev_Splined_Fcoll_MINI,\
                            prev_dens_val,density_over_mean,status_int) \
                    num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:f_coll,f_coll_MINI)
                for (x = 0; x < user_params->HII_DIM; x++) {
                    for (y = 0; y < user_params->HII_DIM; y++) {
                        for (z = 0; z < HII_D_PARA; z++) {

                            // delta cannot be less than -1
                            *((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z)) = fmaxf(
                                                *((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z)), -1. + FRACT_FLOAT_ERR);

                            // <N_rec> cannot be less than zero
                            if (flag_options->INHOMO_RECO) {
                                *((float *) N_rec_filtered + HII_R_FFT_INDEX(x, y, z)) = fmaxf(*((float *) N_rec_filtered + HII_R_FFT_INDEX(x, y, z)), 0.0);
                            }

                            // x_e has to be between zero and unity
                            if (flag_options->USE_TS_FLUCT) {
                                *((float *) xe_filtered + HII_R_FFT_INDEX(x, y, z)) = fmaxf(*((float *) xe_filtered + HII_R_FFT_INDEX(x, y, z)), 0.);
                                *((float *) xe_filtered + HII_R_FFT_INDEX(x, y, z)) = fminf(*((float *) xe_filtered + HII_R_FFT_INDEX(x, y, z)), 0.999);
                            }

                            if(flag_options->USE_HALO_FIELD) {

                                // collapsed mass cannot be less than zero
                                *((float *)M_coll_filtered + HII_R_FFT_INDEX(x,y,z)) = fmaxf(
                                        *((float *)M_coll_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.0);

                                density_over_mean = 1.0 + *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));

                                Splined_Fcoll = *((float *)M_coll_filtered + HII_R_FFT_INDEX(x,y,z)) / (massofscaleR*density_over_mean);
                                Splined_Fcoll *= (4/3.0)*PI*pow(R,3) / pixel_volume;


                            }
                            else {

                                curr_dens = *((float *) deltax_filtered + HII_R_FFT_INDEX(x, y, z));

                                if (flag_options->USE_MASS_DEPENDENT_ZETA) {

                                    if (flag_options->USE_MINI_HALOS){

                                        log10_Mturnover = *((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z));
                                        log10_Mturnover_MINI = *((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z));

                                        if(user_params->USE_INTERPOLATION_TABLES) {

                                            status_int = EvaluateSplineTable(flag_options->USE_MINI_HALOS,1,curr_dens,log10_Mturnover,log10_Mturnover_MINI,
                                                                        &Splined_Fcoll,&Splined_Fcoll_MINI);

                                            if(status_int > 0) {
                                                overdense_int_boundexceeded_threaded[omp_get_thread_num()] = status_int;
                                                LOG_ULTRA_DEBUG("Broken 1059 in thread=%d", omp_get_thread_num());
                                            }
                                        }
                                        else {

                                            Splined_Fcoll = Nion_ConditionalM(growth_factor,log(M_MIN),log(massofscaleR),sigmaMmax,Deltac,curr_dens,
                                                                              pow(10.,log10_Mturnover),astro_params->ALPHA_STAR,
                                                                              astro_params->ALPHA_ESC,astro_params->F_STAR10,
                                                                              astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc, user_params->FAST_FCOLL_TABLES);

                                            Splined_Fcoll_MINI = Nion_ConditionalM_MINI(growth_factor,log(M_MIN),log(massofscaleR),sigmaMmax,Deltac,curr_dens,
                                                                                    pow(10.,log10_Mturnover_MINI),Mcrit_atom,astro_params->ALPHA_STAR_MINI,
                                                                                    astro_params->ALPHA_ESC,astro_params->F_STAR7_MINI,astro_params->F_ESC7_MINI,
                                                                                    Mlim_Fstar_MINI,Mlim_Fesc_MINI, user_params->FAST_FCOLL_TABLES);
                                        }

                                        prev_dens = *((float *)prev_deltax_filtered + HII_R_FFT_INDEX(x,y,z));

                                        if (previous_ionize_box->mean_f_coll_MINI * ION_EFF_FACTOR_MINI + previous_ionize_box->mean_f_coll * ION_EFF_FACTOR > 1e-4){

                                            if(user_params->USE_INTERPOLATION_TABLES) {

                                                status_int = EvaluateSplineTable(flag_options->USE_MINI_HALOS,2,prev_dens,log10_Mturnover,log10_Mturnover_MINI,
                                                                            &prev_Splined_Fcoll,&prev_Splined_Fcoll_MINI);

                                                if(status_int > 0) {
                                                    overdense_int_boundexceeded_threaded[omp_get_thread_num()] = status_int;
                                                    LOG_ULTRA_DEBUG("Broken 1086 in thread=%d", omp_get_thread_num());
                                                }
                                            }
                                            else {

                                                prev_Splined_Fcoll = Nion_ConditionalM(prev_growth_factor,log(M_MIN),log(massofscaleR),sigmaMmax,Deltac,prev_dens,
                                                                                       pow(10.,log10_Mturnover),astro_params->ALPHA_STAR,
                                                                                       astro_params->ALPHA_ESC,astro_params->F_STAR10,
                                                                                       astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc, user_params->FAST_FCOLL_TABLES);

                                                prev_Splined_Fcoll_MINI = Nion_ConditionalM_MINI(prev_growth_factor,log(M_MIN),log(massofscaleR),sigmaMmax,Deltac,prev_dens,
                                                                                        pow(10.,log10_Mturnover_MINI),Mcrit_atom,astro_params->ALPHA_STAR_MINI,
                                                                                        astro_params->ALPHA_ESC,astro_params->F_STAR7_MINI,astro_params->F_ESC7_MINI,
                                                                                        Mlim_Fstar_MINI,Mlim_Fesc_MINI, user_params->FAST_FCOLL_TABLES);
                                            }
                                        }
                                        else{
                                            prev_Splined_Fcoll = 0.;
                                            prev_Splined_Fcoll_MINI = 0.;
                                        }
                                    }
                                    else{

                                        if(user_params->USE_INTERPOLATION_TABLES) {

                                            status_int = EvaluateSplineTable(flag_options->USE_MINI_HALOS,1,curr_dens,0.,0.,&Splined_Fcoll,&Splined_Fcoll_MINI);

                                            if(status_int > 0) {
                                                overdense_int_boundexceeded_threaded[omp_get_thread_num()] = status_int;
                                                LOG_ULTRA_DEBUG("Broken 1115 in thread=%d", omp_get_thread_num());
                                            }


                                        }
                                        else {

                                            Splined_Fcoll = Nion_ConditionalM(growth_factor,log(M_MIN),log(massofscaleR),sigmaMmax,Deltac,curr_dens,
                                                                              astro_params->M_TURN,astro_params->ALPHA_STAR,
                                                                              astro_params->ALPHA_ESC,astro_params->F_STAR10,
                                                                              astro_params->F_ESC10,Mlim_Fstar,Mlim_Fesc, user_params->FAST_FCOLL_TABLES);

                                        }
                                    }
                                }
                                else {
                                    erfc_arg_val = (Deltac - curr_dens) * erfc_denom;
                                    if (erfc_arg_val < erfc_arg_min || erfc_arg_val > erfc_arg_max) {
                                        Splined_Fcoll = splined_erfc(erfc_arg_val);
                                    } else {
                                        erfc_arg_val_index = (int) floor((erfc_arg_val - erfc_arg_min) * InvArgBinWidth);

                                        Splined_Fcoll = ERFC_VALS[erfc_arg_val_index] + \
                                                (erfc_arg_val - (erfc_arg_min + ArgBinWidth * (double) erfc_arg_val_index)) * ERFC_VALS_DIFF[erfc_arg_val_index] *InvArgBinWidth;
                                    }
                                }
                            }

                            // save the value of the collasped fraction into the Fcoll array
                            if (flag_options->USE_MINI_HALOS){
                                if (Splined_Fcoll > 1.) Splined_Fcoll = 1.;
                                if (Splined_Fcoll < 0.) Splined_Fcoll = 1e-40;
                                if (prev_Splined_Fcoll > 1.) prev_Splined_Fcoll = 1.;
                                if (prev_Splined_Fcoll < 0.) prev_Splined_Fcoll = 1e-40;
                                box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = \
                                        previous_ionize_box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] + Splined_Fcoll - prev_Splined_Fcoll;

                                if (box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] >1.) box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = 1.;
                                //if (box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] <0.) box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = 1e-40;
                                //if (box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] < previous_ionize_box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)])
                                //    box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = previous_ionize_box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)];
                                f_coll += box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)];
                                if(isfinite(f_coll)==0) {
                                    LOG_ERROR("f_coll is either infinite or NaN!(%d,%d,%d)%g,%g,%g,%g,%g,%g,%g,%g,%g",\
                                            x,y,z,curr_dens,prev_dens,previous_ionize_box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)],\
                                            Splined_Fcoll, prev_Splined_Fcoll, curr_dens, prev_dens, \
                                            log10_Mturnover, *((float *)log10_Mturnover_filtered + HII_R_FFT_INDEX(x,y,z)));
//                                    Throw(ParameterError);
                                    Throw(InfinityorNaNError);
                                }

                                if (Splined_Fcoll_MINI > 1.) Splined_Fcoll_MINI = 1.;
                                if (Splined_Fcoll_MINI < 0.) Splined_Fcoll_MINI = 1e-40;
                                if (prev_Splined_Fcoll_MINI > 1.) prev_Splined_Fcoll_MINI = 1.;
                                if (prev_Splined_Fcoll_MINI < 0.) prev_Splined_Fcoll_MINI = 1e-40;
                                box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = \
                                            previous_ionize_box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] + Splined_Fcoll_MINI - prev_Splined_Fcoll_MINI;

                                if (box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] >1.) box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = 1.;
                                //if (box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] <0.) box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = 1e-40;
                                //if (box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] < previous_ionize_box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)])
                                //    box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = previous_ionize_box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)];
                                f_coll_MINI += box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)];
                                if(isfinite(f_coll_MINI)==0) {
                                    LOG_ERROR("f_coll_MINI is either infinite or NaN!(%d,%d,%d)%g,%g,%g,%g,%g,%g,%g",\
                                              x,y,z,curr_dens, prev_dens, previous_ionize_box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)],\
                                              Splined_Fcoll_MINI, prev_Splined_Fcoll_MINI, log10_Mturnover_MINI,\
                                              *((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)));
                                    LOG_DEBUG("%g,%g",previous_ionize_box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)],\
                                              previous_ionize_box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)]);
                                    LOG_DEBUG("%g,%g,%g,%g,%g,%g,%g,%g,",log10Mturn_min, log10Mturn_max, log10Mturn_bin_width, \
                                              log10Mturn_bin_width_inv, log10Mturn_max_MINI, log10Mturn_min_MINI, \
                                              log10Mturn_bin_width_MINI, log10Mturn_bin_width_inv_MINI);
                                    LOG_DEBUG("%g,%g,%g,%g,%d",curr_dens, overdense_small_min, overdense_small_bin_width_inv, dens_val, overdense_int);
                                    LOG_DEBUG("%d,%g,%g,%g",log10_Mturnover_MINI_int, log10_Mturnover_MINI, log10Mturn_min_MINI, log10Mturn_bin_width_inv_MINI);
                                    LOG_DEBUG("%g", *((float *)log10_Mturnover_MINI_filtered + HII_R_FFT_INDEX(x,y,z)));
                                    LOG_DEBUG("%d", counter);
                                    LOG_DEBUG("%g,%g,%g,%g",log10_Nion_spline_MINI[overdense_int   + NSFR_low* log10_Mturnover_MINI_int   ], \
                                              log10_Nion_spline_MINI[overdense_int +1+ NSFR_low* log10_Mturnover_MINI_int   ], \
                                              log10_Nion_spline_MINI[overdense_int   + NSFR_low*(log10_Mturnover_MINI_int+1)],  \
                                              log10_Nion_spline_MINI[overdense_int +1+ NSFR_low*(log10_Mturnover_MINI_int+1)]);
//                                    Throw(ParameterError);
                                    Throw(InfinityorNaNError);
                                }
                            }
                            else{
                                box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)] = Splined_Fcoll;
                                f_coll += Splined_Fcoll;
                            }
                        }
                    }
                }
            } //  end loop through Fcoll box

            for (i = 0; i < user_params->N_THREADS; i++) {
                if (overdense_int_boundexceeded_threaded[i] == 1) {
                    LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables for the nion_splines");
//                    Throw(ParameterError);
                    Throw(TableEvaluationError);
                }
            }
            if(isfinite(f_coll)==0) {
                LOG_ERROR("f_coll is either infinite or NaN!");
//                Throw(ParameterError);
                Throw(InfinityorNaNError);
            }
            f_coll /= (double) HII_TOT_NUM_PIXELS;

            if(isfinite(f_coll_MINI)==0) {
                LOG_ERROR("f_coll_MINI is either infinite or NaN!");
//                Throw(ParameterError);
                Throw(InfinityorNaNError);
            }

            f_coll_MINI /= (double) HII_TOT_NUM_PIXELS;

            // To avoid ST_over_PS becoming nan when f_coll = 0, I set f_coll = FRACT_FLOAT_ERR.
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                if (f_coll <= f_coll_min) f_coll = f_coll_min;
                if (flag_options->USE_MINI_HALOS){
                    if (f_coll_MINI <= f_coll_min_MINI) f_coll_MINI = f_coll_min_MINI;
                }
            }
            else {
                if (f_coll <= FRACT_FLOAT_ERR) f_coll = FRACT_FLOAT_ERR;
            }

            ST_over_PS = box->mean_f_coll/f_coll;
            ST_over_PS_MINI = box->mean_f_coll_MINI/f_coll_MINI;

            //////////////////////////////  MAIN LOOP THROUGH THE BOX ///////////////////////////////////
            // now lets scroll through the filtered box
            Gamma_R_prefactor = (R*CMperMPC) * SIGMA_HI * global_params.ALPHA_UVB / (global_params.ALPHA_UVB+2.75) * N_b0 * ION_EFF_FACTOR / 1.0e-12;
            Gamma_R_prefactor_MINI = (R*CMperMPC) * SIGMA_HI * global_params.ALPHA_UVB / (global_params.ALPHA_UVB+2.75) * N_b0 * ION_EFF_FACTOR_MINI / 1.0e-12;
            if(flag_options->PHOTON_CONS) {
                // Used for recombinations, which means we want to use the original redshift not the adjusted redshift
                Gamma_R_prefactor *= pow(1+stored_redshift, 2);
                Gamma_R_prefactor_MINI *= pow(1+stored_redshift, 2);
            }
            else {
                Gamma_R_prefactor *= pow(1+redshift, 2);
                Gamma_R_prefactor_MINI *= pow(1+redshift, 2);
            }

            Gamma_R_prefactor /= t_ast;
            Gamma_R_prefactor_MINI /= t_ast;

            if (global_params.FIND_BUBBLE_ALGORITHM != 2 && global_params.FIND_BUBBLE_ALGORITHM != 1) { // center method
                LOG_ERROR("Incorrect choice of find bubble algorithm: %i",
                          global_params.FIND_BUBBLE_ALGORITHM);
                Throw(ValueError);
            }


#pragma omp parallel shared(deltax_filtered,N_rec_filtered,xe_filtered,box,ST_over_PS,pixel_mass,M_MIN,r,f_coll_min,Gamma_R_prefactor,\
                            ION_EFF_FACTOR,ION_EFF_FACTOR_MINI,LAST_FILTER_STEP,counter,ST_over_PS_MINI,f_coll_min_MINI,Gamma_R_prefactor_MINI,TK,cT_ad,perturbed_field) \
                    private(x,y,z,curr_dens,Splined_Fcoll,f_coll,ave_M_coll_cell,ave_N_min_cell,N_halos_in_cell,rec,xHII_from_xrays,\
                            Splined_Fcoll_MINI,f_coll_MINI, res_xH) \
                    num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (x = 0; x < user_params->HII_DIM; x++) {
                    for (y = 0; y < user_params->HII_DIM; y++) {
                        for (z = 0; z < HII_D_PARA; z++) {

                            curr_dens = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));

                            Splined_Fcoll = box->Fcoll[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)];

                            f_coll = ST_over_PS * Splined_Fcoll;

                            if (flag_options->USE_MINI_HALOS){
                                Splined_Fcoll_MINI = box->Fcoll_MINI[counter * HII_TOT_NUM_PIXELS + HII_R_INDEX(x,y,z)];
                                f_coll_MINI = ST_over_PS_MINI * Splined_Fcoll_MINI;
                            }
                            else{
                                f_coll_MINI = 0.;
                            }

                            if (LAST_FILTER_STEP){
                                ave_M_coll_cell = (f_coll + f_coll_MINI) * pixel_mass * (1. + curr_dens);
                                ave_N_min_cell = ave_M_coll_cell / M_MIN; // ave # of M_MIN halos in cell
                                if(user_params->NO_RNG) {
                                    N_halos_in_cell = 1.;
                                }
                                else {
                                    N_halos_in_cell = (int) gsl_ran_poisson(r[omp_get_thread_num()],
                                                                            global_params.N_POISSON);
                                }
                            }

                            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                                if (f_coll <= f_coll_min) f_coll = f_coll_min;
                                if (flag_options->USE_MINI_HALOS){
                                    if (f_coll_MINI <= f_coll_min_MINI) f_coll_MINI = f_coll_min_MINI;
                                }
                            }

                            if (flag_options->INHOMO_RECO) {
                                rec = (*((float *) N_rec_filtered +
                                         HII_R_FFT_INDEX(x, y, z))); // number of recombinations per mean baryon
                                rec /= (1. + curr_dens); // number of recombinations per baryon inside <R>
                            } else {
                                rec = 0.;
                            }

                            // adjust the denominator of the collapse fraction for the residual electron fraction in the neutral medium
                            if (flag_options->USE_TS_FLUCT){
                                xHII_from_xrays = *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z));
                            } else {
                                xHII_from_xrays = 0.;
                            }

                            // check if fully ionized!
                            if ( (f_coll * ION_EFF_FACTOR + f_coll_MINI * ION_EFF_FACTOR_MINI> (1. - xHII_from_xrays)*(1.0+rec)) ){ //IONIZED!!
                                // if this is the first crossing of the ionization barrier for this cell (largest R), record the gamma
                                // this assumes photon-starved growth of HII regions...  breaks down post EoR
                                if (flag_options->INHOMO_RECO && (box->xH_box[HII_R_INDEX(x,y,z)] > FRACT_FLOAT_ERR) ){
                                    box->Gamma12_box[HII_R_INDEX(x,y,z)] = Gamma_R_prefactor * f_coll + Gamma_R_prefactor_MINI * f_coll_MINI;
                                    box->MFP_box[HII_R_INDEX(x,y,z)] = R;
                                }

                                // keep track of the first time this cell is ionized (earliest time)
                                if (previous_ionize_box->z_re_box[HII_R_INDEX(x,y,z)] < 0){
                                    box->z_re_box[HII_R_INDEX(x,y,z)] = redshift;
                                } else{
                                    box->z_re_box[HII_R_INDEX(x,y,z)] = previous_ionize_box->z_re_box[HII_R_INDEX(x,y,z)];
                                }

                                // FLAG CELL(S) AS IONIZED
                                if (global_params.FIND_BUBBLE_ALGORITHM == 2) // center method
                                    box->xH_box[HII_R_INDEX(x,y,z)] = 0;
                                if (global_params.FIND_BUBBLE_ALGORITHM == 1) // sphere method
                                    update_in_sphere(box->xH_box, user_params->HII_DIM, HII_D_PARA, R/(user_params->BOX_LEN), \
                                                     x/(user_params->HII_DIM+0.0), y/(user_params->HII_DIM+0.0), z/(HII_D_PARA+0.0));
                            } // end ionized
                                // If not fully ionized, then assign partial ionizations
                            else if (LAST_FILTER_STEP && (box->xH_box[HII_R_INDEX(x, y, z)] > TINY)) {

                                if (f_coll>1) f_coll=1;
                                if (f_coll_MINI>1) f_coll_MINI=1;

                                if (!flag_options->USE_HALO_FIELD){
                                    if(ave_N_min_cell < global_params.N_POISSON) {
                                        f_coll = N_halos_in_cell * ( ave_M_coll_cell / (float)global_params.N_POISSON ) / (pixel_mass*(1. + curr_dens));
                                        if (flag_options->USE_MINI_HALOS){
                                            f_coll_MINI = f_coll * (f_coll_MINI * ION_EFF_FACTOR_MINI) / (f_coll * ION_EFF_FACTOR + f_coll_MINI * ION_EFF_FACTOR_MINI);
                                            f_coll = f_coll - f_coll_MINI;
                                        }
                                        else{
                                            f_coll_MINI = 0.;
                                        }
                                    }

                                    if (ave_M_coll_cell < (M_MIN / 5.)) {
                                        f_coll = 0.;
                                        f_coll_MINI = 0.;
                                    }
                                }

                                if (f_coll>1) f_coll=1;
                                if (f_coll_MINI>1) f_coll_MINI=1;
                                res_xH = 1. - f_coll * ION_EFF_FACTOR - f_coll_MINI * ION_EFF_FACTOR_MINI;
                                // put the partial ionization here because we need to exclude xHII_from_xrays...
                                if (flag_options->USE_TS_FLUCT){
                                    box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)] = ComputePartiallyIoinizedTemperature(spin_temp->Tk_box[HII_R_INDEX(x,y,z)], res_xH);
                                }
                                else{
                                    box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)] = ComputePartiallyIoinizedTemperature(TK*(1 + cT_ad*perturbed_field->density[HII_R_INDEX(x,y,z)]), res_xH);
                                }
                                res_xH -= xHII_from_xrays;

                                // and make sure fraction doesn't blow up for underdense pixels
                                if (res_xH < 0)
                                    res_xH = 0;
                                else if (res_xH > 1)
                                    res_xH = 1;

                                box->xH_box[HII_R_INDEX(x, y, z)] = res_xH;

                            } // end partial ionizations at last filtering step
                        } // k
                    } // j
                } // i
            }

            LOG_SUPER_DEBUG("z_re_box after R=%f: ", R);
            debugSummarizeBox(box->z_re_box, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");


            if (first_step_R) {
                R = stored_R;
                first_step_R = 0;
            } else {
                R /= (global_params.DELTA_R_HII_FACTOR);
            }
            if (flag_options->USE_MINI_HALOS)
                counter += 1;
        }


#pragma omp parallel shared(box,spin_temp,redshift,deltax_unfiltered_original,TK) private(x,y,z,thistk) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (x=0; x<user_params->HII_DIM; x++){
                for (y=0; y<user_params->HII_DIM; y++){
                    for (z=0; z<HII_D_PARA; z++){
                        if ((box->z_re_box[HII_R_INDEX(x,y,z)]>0) && (box->xH_box[HII_R_INDEX(x,y,z)] < TINY)){
                            box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)] = ComputeFullyIoinizedTemperature(box->z_re_box[HII_R_INDEX(x,y,z)], \
                                                                        redshift, *((float *)deltax_unfiltered_original + HII_R_FFT_INDEX(x,y,z)));
                            // Below sometimes (very rare though) can happen when the density drops too fast and to below T_HI
                            if (flag_options->USE_TS_FLUCT){
                                if (box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)] < spin_temp->Tk_box[HII_R_INDEX(x,y,z)])
                                    box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)] = spin_temp->Tk_box[HII_R_INDEX(x,y,z)];
                                }
                            else{
                                thistk = TK*(1. + cT_ad*perturbed_field->density[HII_R_INDEX(x,y,z)]);
                                if (box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)] < thistk)
                                    box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)] = thistk;
                            }
                        }
                    }
                }
            }
        }

        for (x=0; x<user_params->HII_DIM; x++){
            for (y=0; y<user_params->HII_DIM; y++){
                for (z=0; z<HII_D_PARA; z++){
                    if(isfinite(box->temp_kinetic_all_gas[HII_R_INDEX(x,y,z)])==0){
                        LOG_ERROR("Tk after fully ioinzation is either infinite or a Nan. Something has gone wrong "\
                                  "in the temperature calculation: z_re=%.4f, redshift=%.4f, curr_dens=%.4e", box->z_re_box[HII_R_INDEX(x,y,z)], redshift, curr_dens);
//                        Throw(ParameterError);
                        Throw(InfinityorNaNError);
                    }
                }
            }
        }

        // find the neutral fraction
        if (LOG_LEVEL >= DEBUG_LEVEL) {
            global_xH = 0;

#pragma omp parallel shared(box) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:global_xH)
                for (ct = 0; ct < HII_TOT_NUM_PIXELS; ct++) {
                    global_xH += box->xH_box[ct];
                }
            }
            global_xH /= (float) HII_TOT_NUM_PIXELS;
        }

        if (isfinite(global_xH) == 0) {
            LOG_ERROR(
                    "Neutral fraction is either infinite or a Nan. Something has gone wrong in the ionisation calculation!");
//            Throw(ParameterError);
            Throw(InfinityorNaNError);
        }

        // update the N_rec field
        if (flag_options->INHOMO_RECO) {

#pragma omp parallel shared(perturbed_field, adjustment_factor, stored_redshift, redshift, box, previous_ionize_box, \
                            fabs_dtdz, ZSTEP, something_finite_or_infinite) \
                    private(x, y, z, curr_dens, z_eff, dNrec) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (x = 0; x < user_params->HII_DIM; x++) {
                    for (y = 0; y < user_params->HII_DIM; y++) {
                        for (z = 0; z < HII_D_PARA; z++) {

                            // use the original density and redshift for the snapshot (not the adjusted redshift)
                            // Only want to use the adjusted redshift for the ionisation field
                            curr_dens = 1.0 + (perturbed_field->density[HII_R_INDEX(x, y, z)]) / adjustment_factor;
                            z_eff = pow(curr_dens, 1.0 / 3.0);

                            if (flag_options->PHOTON_CONS) {
                                z_eff *= (1 + stored_redshift);
                            } else {
                                z_eff *= (1 + redshift);
                            }

                            dNrec = splined_recombination_rate(z_eff - 1., box->Gamma12_box[HII_R_INDEX(x, y, z)]) *
                                    fabs_dtdz * ZSTEP * (1. - box->xH_box[HII_R_INDEX(x, y, z)]);

                            if (isfinite(dNrec) == 0) {
                                something_finite_or_infinite = 1;
                            }

                            box->dNrec_box[HII_R_INDEX(x, y, z)] =
                                    previous_ionize_box->dNrec_box[HII_R_INDEX(x, y, z)] + dNrec;
                        }
                    }
                }
            }

            if (something_finite_or_infinite) {
                LOG_ERROR("Recombinations have returned either an infinite or NaN value.");
//                Throw(ParameterError);
                Throw(InfinityorNaNError);
            }
        }

        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();
    }

    destruct_heat();

    for (i=0; i<user_params->N_THREADS; i++) {
        gsl_rng_free (r[i]);
    }

LOG_DEBUG("global_xH = %e",global_xH);

    fftwf_free(deltax_unfiltered);
    fftwf_free(deltax_unfiltered_original);
    fftwf_free(deltax_filtered);
    if(flag_options->USE_MINI_HALOS){
        fftwf_free(prev_deltax_unfiltered);
        fftwf_free(prev_deltax_filtered);
    }
    if(flag_options->USE_TS_FLUCT) {
        fftwf_free(xe_unfiltered);
        fftwf_free(xe_filtered);
    }
    if (flag_options->INHOMO_RECO){
        fftwf_free(N_rec_unfiltered);
        fftwf_free(N_rec_filtered);
    }

    if(flag_options->USE_HALO_FIELD) {
        fftwf_free(M_coll_unfiltered);
        fftwf_free(M_coll_filtered);
    }


LOG_SUPER_DEBUG("freed fftw boxes");

    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        free(xi_SFR);
        free(wi_SFR);

        if(user_params->USE_INTERPOLATION_TABLES) {
            free(log10_overdense_spline_SFR);
            free(Overdense_spline_SFR);
            free(log10_Nion_spline);
            free(Nion_spline);

        }

        if(flag_options->USE_MINI_HALOS){
            free(Mturns);
            free(Mturns_MINI);
            fftwf_free(log10_Mturnover_unfiltered);
            fftwf_free(log10_Mturnover_filtered);
            fftwf_free(log10_Mturnover_MINI_unfiltered);
            fftwf_free(log10_Mturnover_MINI_filtered);

            if(user_params->USE_INTERPOLATION_TABLES) {
                free(prev_log10_overdense_spline_SFR);
                free(prev_Overdense_spline_SFR);
                free(prev_log10_Nion_spline);
                free(prev_Nion_spline);
                free(log10_Nion_spline_MINI);
                free(Nion_spline_MINI);
                free(prev_log10_Nion_spline_MINI);
                free(prev_Nion_spline_MINI);
            }
        }
        //fftwf_free(Mcrit_RE_grid);
        //fftwf_free(Mcrit_LW_grid);

    }

    if (prev_redshift < 1){
        free(previous_ionize_box->z_re_box);
        if (flag_options->USE_MASS_DEPENDENT_ZETA && flag_options->USE_MINI_HALOS){
            free(previous_ionize_box->Gamma12_box);
            free(previous_ionize_box->dNrec_box);
            free(previous_ionize_box->Fcoll);
            free(previous_ionize_box->Fcoll_MINI);
        }
    }

    if(!flag_options->USE_TS_FLUCT && user_params->USE_INTERPOLATION_TABLES) {
            freeSigmaMInterpTable();
    }

    free(overdense_int_boundexceeded_threaded);

    LOG_DEBUG("finished!\n");

    } // End of Try()

    Catch(status){
        return(status);
    }
    return(0);
}


int EvaluateSplineTable(bool MINI_HALOS, int dens_type, float curr_dens, float filtered_Mturn, float filtered_Mturn_MINI, float *Splined_Fcoll, float *Splined_Fcoll_MINI) {

    int overdense_int,overdense_int_status;
    float dens_val, small_bin_width, small_bin_width_inv, small_min;
    float log10_Mturnover, log10_Mturnover_MINI;
    int log10_Mturnover_int, log10_Mturnover_MINI_int;

    overdense_int_status = 0;

    if(MINI_HALOS) {
        log10_Mturnover = (filtered_Mturn - log10Mturn_min ) * log10Mturn_bin_width_inv;
        log10_Mturnover_int = (int)floorf( log10_Mturnover );
        log10_Mturnover_MINI = (filtered_Mturn_MINI - log10Mturn_min_MINI ) * log10Mturn_bin_width_inv_MINI;
        log10_Mturnover_MINI_int = (int)floorf( log10_Mturnover_MINI );
    }

    if(dens_type==1) {
        small_min = overdense_small_min;
        small_bin_width = overdense_small_bin_width;
        small_bin_width_inv = overdense_small_bin_width_inv;
    }
    if(dens_type==2) {
        small_min = prev_overdense_small_min;
        small_bin_width = prev_overdense_small_bin_width;
        small_bin_width_inv = prev_overdense_small_bin_width_inv;
    }

    if (curr_dens < global_params.CRIT_DENS_TRANSITION) {

        if (curr_dens <= -1.) {
            *Splined_Fcoll = 0;
            if(MINI_HALOS) {
                *Splined_Fcoll_MINI = 0;
            }
        } else {
            dens_val = (log10f(curr_dens + 1.) - small_min) * small_bin_width_inv;
            overdense_int = (int) floorf(dens_val);

            if (overdense_int < 0 || (overdense_int + 1) > (NSFR_low - 1)) {
                overdense_int_status = 1;
                LOG_INFO("overdense_int in thread %d got value %d (exceeded bounds). Current density=%g", omp_get_thread_num(), overdense_int, dens_val);
            }

            if(MINI_HALOS) {
                if(dens_type==1) {
                    *Splined_Fcoll = ( \
                                 log10_Nion_spline[overdense_int + NSFR_low*log10_Mturnover_int]*( 1 + (float)overdense_int - dens_val ) + \
                                 log10_Nion_spline[overdense_int + 1 + NSFR_low*log10_Mturnover_int]*( dens_val - (float)overdense_int ) \
                                 ) * (1 + (float)log10_Mturnover_int - log10_Mturnover) + \
                                ( \
                                 log10_Nion_spline[overdense_int + NSFR_low*(log10_Mturnover_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                 log10_Nion_spline[overdense_int + 1 + NSFR_low*(log10_Mturnover_int+1)]*( dens_val - (float)overdense_int ) \
                                 ) * (log10_Mturnover - (float)log10_Mturnover_int);

                    *Splined_Fcoll_MINI = ( \
                                      log10_Nion_spline_MINI[overdense_int + NSFR_low*log10_Mturnover_MINI_int]*( 1 + (float)overdense_int - dens_val ) + \
                                      log10_Nion_spline_MINI[overdense_int + 1 + NSFR_low*log10_Mturnover_MINI_int]*( dens_val - (float)overdense_int ) \
                                      ) * (1 + (float)log10_Mturnover_MINI_int - log10_Mturnover_MINI) + \
                                    ( \
                                     log10_Nion_spline_MINI[overdense_int + NSFR_low*(log10_Mturnover_MINI_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                     log10_Nion_spline_MINI[overdense_int + 1 + NSFR_low*(log10_Mturnover_MINI_int+1)]*( dens_val - (float)overdense_int ) \
                                     ) * (log10_Mturnover_MINI - (float)log10_Mturnover_MINI_int);
                }
                if(dens_type==2) {
                    *Splined_Fcoll = ( \
                                      prev_log10_Nion_spline[overdense_int + NSFR_low*log10_Mturnover_int]*( 1 + (float)overdense_int - dens_val ) + \
                                      prev_log10_Nion_spline[overdense_int + 1 + NSFR_low*log10_Mturnover_int]*( dens_val - (float)overdense_int ) \
                                      ) * (1 + (float)log10_Mturnover_int - log10_Mturnover) + \
                                    ( \
                                     prev_log10_Nion_spline[overdense_int + NSFR_low*(log10_Mturnover_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                     prev_log10_Nion_spline[overdense_int + 1 + NSFR_low*(log10_Mturnover_int+1)]*( dens_val - (float)overdense_int ) \
                                     ) * (log10_Mturnover - (float)log10_Mturnover_int);

                    *Splined_Fcoll_MINI = ( \
                                           prev_log10_Nion_spline_MINI[overdense_int + NSFR_low*log10_Mturnover_MINI_int]*( 1 + (float)overdense_int - dens_val ) + \
                                           prev_log10_Nion_spline_MINI[overdense_int + 1 + NSFR_low*log10_Mturnover_MINI_int]*( dens_val - (float)overdense_int ) \
                                           ) * (1 + (float)log10_Mturnover_MINI_int - log10_Mturnover_MINI) + \
                                    ( \
                                     prev_log10_Nion_spline_MINI[overdense_int + NSFR_low*(log10_Mturnover_MINI_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                     prev_log10_Nion_spline_MINI[overdense_int + 1 + NSFR_low*(log10_Mturnover_MINI_int+1)]*( dens_val - (float)overdense_int ) \
                                     ) * (log10_Mturnover_MINI - (float)log10_Mturnover_MINI_int);
                }

                *Splined_Fcoll_MINI = expf(*Splined_Fcoll_MINI);
            }
            else {
                *Splined_Fcoll = log10_Nion_spline[overdense_int] * (1 + (float) overdense_int - dens_val) + log10_Nion_spline[overdense_int + 1] * (dens_val - (float) overdense_int);
            }
            *Splined_Fcoll = expf(*Splined_Fcoll);
        }
    }
    else {
        if (curr_dens < 0.99 * Deltac) {

            if(dens_type==1) {
                dens_val = (curr_dens - overdense_large_min) * overdense_large_bin_width_inv;
              LOG_ULTRA_DEBUG("type=%d curr_dens=%e, overdense_large_min=%e, overdense_large_bin_width_inv=%e",\
              dens_type,curr_dens, overdense_large_min,overdense_large_bin_width_inv);
            }
            if(dens_type==2) {
                dens_val = (curr_dens - prev_overdense_large_min) * prev_overdense_large_bin_width_inv;
                LOG_ULTRA_DEBUG("type=%d curr_dens=%e, prev_overdense_large_min=%e, prev_overdense_large_bin_width_inv=%e",\
                dens_type,curr_dens, prev_overdense_large_min,prev_overdense_large_bin_width_inv);
              }



            overdense_int = (int) floorf(dens_val);

            if (overdense_int < 0 || (overdense_int + 1) > (NSFR_high - 1)) {
                overdense_int_status = 1;
                LOG_INFO("overdense_int in thread %d got value %d (exceeded bounds). Current density=%g", omp_get_thread_num(), overdense_int, dens_val);
            }

            if(MINI_HALOS) {
                if(dens_type==1) {
                    *Splined_Fcoll = ( \
                                 Nion_spline[overdense_int + NSFR_high* log10_Mturnover_int]*( 1 + (float)overdense_int - dens_val ) + \
                                 Nion_spline[overdense_int + 1 + NSFR_high* log10_Mturnover_int]*( dens_val - (float)overdense_int ) \
                                 ) * (1 + (float)log10_Mturnover_int - log10_Mturnover) + \
                                ( \
                                 Nion_spline[overdense_int + NSFR_high*(log10_Mturnover_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                 Nion_spline[overdense_int+ 1 + NSFR_high*(log10_Mturnover_int+1)]*( dens_val - (float)overdense_int ) \
                                 ) * (log10_Mturnover - (float)log10_Mturnover_int);

                    *Splined_Fcoll_MINI = ( \
                                      Nion_spline_MINI[overdense_int + NSFR_high* log10_Mturnover_MINI_int]*( 1 + (float)overdense_int - dens_val ) + \
                                      Nion_spline_MINI[overdense_int + 1 + NSFR_high* log10_Mturnover_MINI_int]*( dens_val - (float)overdense_int ) \
                                      ) * (1 + (float)log10_Mturnover_MINI_int - log10_Mturnover_MINI) + \
                                    ( \
                                     Nion_spline_MINI[overdense_int + NSFR_high*(log10_Mturnover_MINI_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                     Nion_spline_MINI[overdense_int + 1 + NSFR_high*(log10_Mturnover_MINI_int+1)]*( dens_val - (float)overdense_int ) \
                                     ) * (log10_Mturnover_MINI - (float)log10_Mturnover_MINI_int);
                }
                if(dens_type==2) {
                    *Splined_Fcoll = ( \
                                      prev_Nion_spline[overdense_int + NSFR_high* log10_Mturnover_int]*( 1 + (float)overdense_int - dens_val ) + \
                                      prev_Nion_spline[overdense_int + 1 + NSFR_high* log10_Mturnover_int]*( dens_val - (float)overdense_int ) \
                                      ) * (1 + (float)log10_Mturnover_int - log10_Mturnover) + \
                                    ( \
                                    prev_Nion_spline[overdense_int + NSFR_high*(log10_Mturnover_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                     prev_Nion_spline[overdense_int+ 1 + NSFR_high*(log10_Mturnover_int+1)]*( dens_val - (float)overdense_int ) \
                                     ) * (log10_Mturnover - (float)log10_Mturnover_int);

                    *Splined_Fcoll_MINI = ( \
                                           prev_Nion_spline_MINI[overdense_int + NSFR_high* log10_Mturnover_MINI_int]*( 1 + (float)overdense_int - dens_val ) + \
                                           prev_Nion_spline_MINI[overdense_int + 1 + NSFR_high* log10_Mturnover_MINI_int]*( dens_val - (float)overdense_int ) \
                                           ) * (1 + (float)log10_Mturnover_MINI_int - log10_Mturnover_MINI) + \
                                        ( \
                                         prev_Nion_spline_MINI[overdense_int + NSFR_high*(log10_Mturnover_MINI_int+1)]*( 1 + (float)overdense_int - dens_val ) + \
                                         prev_Nion_spline_MINI[overdense_int + 1 + NSFR_high*(log10_Mturnover_MINI_int+1)]*( dens_val - (float)overdense_int ) \
                                         ) * (log10_Mturnover_MINI - (float)log10_Mturnover_MINI_int);
                }
            }
            else {
                *Splined_Fcoll = Nion_spline[overdense_int] * (1 + (float) overdense_int - dens_val) + Nion_spline[overdense_int + 1] * (dens_val - (float) overdense_int);
            }
        }
        else {
            *Splined_Fcoll = 1.;
            if(MINI_HALOS) {
                *Splined_Fcoll_MINI = 1.;
            }
        }
    }
    return overdense_int_status;
}

void InterpolationRange(int dens_type, float R, float L, float *min_density, float *max_density) {

    float small_bin_width, small_bin_width_inv, small_min;

    if (*min_density < 0.) {
        *min_density = *min_density * 1.001;
        if (*min_density <= -1.) {
            // Use MIN_DENSITY_LOW_LIMIT as is it smaller than FRACT_FLOAT_ERR
            *min_density = -1. + global_params.MIN_DENSITY_LOW_LIMIT;
        }
    } else {
        *min_density = *min_density * 0.999;
    }

    if (*max_density < 0.) {
        *max_density = *max_density * 0.999;
    } else {
        *max_density = *max_density * 1.001;
    }

    if (global_params.HII_FILTER == 1) {
        if ((0.413566994 * R * 2. * PI / L) > 1.) {
            // The sharp k-space filter will set every cell to zero, and the interpolation table using a flexible min/max density will fail.

            *min_density = -1. + global_params.MIN_DENSITY_LOW_LIMIT;
            *max_density = global_params.CRIT_DENS_TRANSITION * 1.001;
        }
    }

    small_min = log10(1. + *min_density);

    if (*max_density > global_params.CRIT_DENS_TRANSITION * 1.001) {
        small_bin_width = 1 / ((double) NSFR_low - 1.) * (log10(1. + global_params.CRIT_DENS_TRANSITION * 1.001) - small_min);
    } else {
        small_bin_width = 1 / ((double) NSFR_low - 1.) * (log10(1. + *max_density) - small_min);
    }

    small_bin_width_inv = 1./small_bin_width;

    if(dens_type==1) {
        overdense_small_min = small_min;
        overdense_small_bin_width = small_bin_width;
        overdense_small_bin_width_inv = small_bin_width_inv;

LOG_ULTRA_DEBUG("R=%f, min_density=%f, max_density=%f, overdense_small_min=%f, overdense_small_bin_width=%f",\
                        R, *min_density, *max_density, small_min, small_bin_width);
    }
    if(dens_type==2) {
        prev_overdense_small_min = small_min;
        prev_overdense_small_bin_width = small_bin_width;
        prev_overdense_small_bin_width_inv = small_bin_width_inv;

LOG_ULTRA_DEBUG("R=%f, prev_min_density=%f, prev_max_density=%f, prev_overdense_small_min=%f, prev_overdense_small_bin_width=%f",\
                        R, *min_density, *max_density, small_min, small_bin_width);

    }
}
