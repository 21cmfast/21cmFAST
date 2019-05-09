
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

// Grids/arrays that only need to be initialised once (i.e. the lowest redshift density cube to be sampled)
double ***fcoll_R_grid, ***dfcoll_dz_grid;
double **grid_dens, **density_gridpoints;
double *Sigma_Tmin_grid, *ST_over_PS_arg_grid, *dstarlya_dt_prefactor, *zpp_edge, *sigma_atR;
float **delNL0_rev,**delNL0;
float *R_values, *delNL0_bw, *delNL0_Offset, *delNL0_LL, *delNL0_UL, *delNL0_ibw, *log10delNL0_diff, *log10delNL0_diff_UL,*min_densities, *max_densities, *zpp_interp_table;
short **dens_grid_int_vals;
short *SingleVal_int;

float *del_fcoll_Rct, *SFR_timescale_factor;

double *dxheat_dt_box, *dxion_source_dt_box, *dxlya_dt_box, *dstarlya_dt_box;

float *inverse_val_box;
int *m_xHII_low_box;

// Grids/arrays that are re-evaluated for each zp
double **fcoll_interp1, **fcoll_interp2, **dfcoll_interp1, **dfcoll_interp2;
double *fcoll_R_array, *sigma_Tmin, *ST_over_PS, *sum_lyn;
float *inverse_diff, *zpp_growth, *zpp_for_evolve_list;

// interpolation tables for the heating/ionisation integrals
double **freq_int_heat_tbl, **freq_int_ion_tbl, **freq_int_lya_tbl, **freq_int_heat_tbl_diff, **freq_int_ion_tbl_diff, **freq_int_lya_tbl_diff;

bool TsInterpArraysInitialised = false;
float initialised_redshift = 0.0;

int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options,
                  float perturbed_field_redshift,
                  struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp, struct TsBox *this_spin_temp) {

LOG_DEBUG("input values:");
LOG_DEBUG("redshift=%f, prev_redshift=%f perturbed_field_redshift=%f", redshift, prev_redshift, perturbed_field_redshift);
if (LOG_LEVEL >= DEBUG_LEVEL){
    writeAstroParams(flag_options, astro_params);
}

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_HF(user_params,cosmo_params,astro_params, flag_options);
    
    // This is an entire re-write of Ts.c from 21cmFAST. You can refer back to Ts.c in 21cmFAST if this become a little obtuse. The computation has remained the same //
    
    /////////////////// Defining variables for the computation of Ts.c //////////////
    char wisdom_filename[500];
    FILE *F, *OUT;
    fftwf_plan plan;
    
    unsigned long long ct, FCOLL_SHORT_FACTOR, box_ct;
    
    int R_ct,i,ii,j,k,i_z,COMPUTE_Ts,x_e_ct,m_xHII_low,m_xHII_high,n_ct, zpp_gridpoint1_int, zpp_gridpoint2_int,zpp_evolve_gridpoint1_int, zpp_evolve_gridpoint2_int,counter;
    
    short dens_grid_int;
    
    double Tk_ave, J_alpha_ave, xalpha_ave, J_alpha_tot, Xheat_ave, Xion_ave, nuprime, Ts_ave, lower_int_limit,Luminosity_converstion_factor,T_inv_TS_fast_inv;
    double dadia_dzp, dcomp_dzp, dxheat_dt, dxion_source_dt, dxion_sink_dt, T, x_e, dxe_dzp, n_b, dspec_dzp, dxheat_dzp, dxlya_dt, dstarlya_dt, fcoll_R;
    double Trad_fast,xc_fast,xc_inverse,TS_fast,TSold_fast,xa_tilde_fast,TS_prefactor,xa_tilde_prefactor,T_inv,T_inv_sq,xi_power,xa_tilde_fast_arg,Trad_fast_inv,TS_fast_inv,dcomp_dzp_prefactor;
    
    float growth_factor_z, inverse_growth_factor_z, R, R_factor, zp, mu_for_Ts, filling_factor_of_HI_zp, dzp, prev_zp, zpp, prev_zpp, prev_R, Tk_BC, xe_BC;
    float xHII_call, curr_xalpha, TK, TS, xe, deltax_highz;
    float zpp_for_evolve,dzpp_for_evolve, M_MIN;
    
    float determine_zpp_max, zpp_grid, zpp_gridpoint1, zpp_gridpoint2,zpp_evolve_gridpoint1, zpp_evolve_gridpoint2, grad1, grad2, grad3, grad4, delNL0_bw_val;
    float OffsetValue, DensityValueLow, min_density, max_density;
    
    double curr_delNL0, inverse_val,prefactor_1,prefactor_2,dfcoll_dz_val, density_eval1, density_eval2, grid_sigmaTmin, grid_dens_val, dens_grad, dens_width;
    
    double const_zp_prefactor, dt_dzp, x_e_ave, growth_factor_zp, dgrowth_factor_dzp;
    
    float M_MIN_WDM =  M_J_WDM();
    
    double ave_fcoll, ave_fcoll_inv, dfcoll_dz_val_ave, ION_EFF_FACTOR;
    
    float curr_dens, min_curr_dens, max_curr_dens;
    
    min_curr_dens = max_curr_dens = 0.;
    
    int fcoll_int_min, fcoll_int_max;
    
    fcoll_int_min = fcoll_int_max = 0;
    
    float Splined_Fcoll,Splined_Fcollzp_mean,Splined_SFRD_zpp, fcoll;
    float redshift_table_Nion_z,redshift_table_SFRD;
    float fcoll_interp_min, fcoll_interp_bin_width, fcoll_interp_bin_width_inv, fcoll_interp_val1, fcoll_interp_val2, dens_val;
    float fcoll_interp_high_min, fcoll_interp_high_bin_width, fcoll_interp_high_bin_width_inv;
    
    int fcoll_int;
    int redshift_int_Nion_z,redshift_int_SFRD;
    
    int table_int_boundexceeded = 0;
    int fcoll_int_boundexceeded = 0;
    
    double total_time, total_time2, total_time3, total_time4;
    float M_MIN_at_zp;
    
    int NO_LIGHT = 0;

    // Initialise arrays to be used for the Ts.c computation //
    fftwf_complex *box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *unfiltered_box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

LOG_SUPER_DEBUG("initialized");

    if(!TsInterpArraysInitialised) {
LOG_SUPER_DEBUG("initalising Ts Interp Arrays");

        // Grids/arrays that only need to be initialised once (i.e. the lowest redshift density cube to be sampled)
        
        zpp_edge = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        sigma_atR = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        R_values = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        
        min_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        max_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        
        zpp_interp_table = calloc(zpp_interp_points_SFR, sizeof(float));
        
        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
            
            SFR_timescale_factor = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            
            delNL0 = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                delNL0[i] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));
            }
            
            xi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));
            wi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));
            
            log10_SFRD_z_low_table = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
            for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                log10_SFRD_z_low_table[j] = (float *)calloc(NSFR_low,sizeof(float));
            }
            
            SFRD_z_high_table = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
            for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                SFRD_z_high_table[j] = (float *)calloc(NSFR_high,sizeof(float));
            }

            
            del_fcoll_Rct = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
            
            dxheat_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dxion_source_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dxlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dstarlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));

            m_xHII_low_box = (int *)calloc(HII_TOT_NUM_PIXELS,sizeof(int));
            inverse_val_box = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));

        }
        else {
            
            Sigma_Tmin_grid = (double *)calloc(zpp_interp_points_SFR,sizeof(double));
            
            fcoll_R_grid = (double ***)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double **));
            dfcoll_dz_grid = (double ***)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double **));
            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                fcoll_R_grid[i] = (double **)calloc(zpp_interp_points_SFR,sizeof(double *));
                dfcoll_dz_grid[i] = (double **)calloc(zpp_interp_points_SFR,sizeof(double *));
                for(j=0;j<zpp_interp_points_SFR;j++) {
                    fcoll_R_grid[i][j] = (double *)calloc(dens_Ninterp,sizeof(double));
                    dfcoll_dz_grid[i][j] = (double *)calloc(dens_Ninterp,sizeof(double));
                }
            }
            
            grid_dens = (double **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double *));
            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                grid_dens[i] = (double *)calloc(dens_Ninterp,sizeof(double));
            }
            
            density_gridpoints = (double **)calloc(dens_Ninterp,sizeof(double *));
            for(i=0;i<dens_Ninterp;i++) {
                density_gridpoints[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            }
            ST_over_PS_arg_grid = (double *)calloc(zpp_interp_points_SFR,sizeof(double));
            
            dens_grid_int_vals = (short **)calloc(HII_TOT_NUM_PIXELS,sizeof(short *));
            
            delNL0_bw = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            delNL0_Offset = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            delNL0_LL = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            delNL0_UL = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            delNL0_ibw = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            log10delNL0_diff = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            log10delNL0_diff_UL = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

            fcoll_interp1 = (double **)calloc(dens_Ninterp,sizeof(double *));
            fcoll_interp2 = (double **)calloc(dens_Ninterp,sizeof(double *));
            dfcoll_interp1 = (double **)calloc(dens_Ninterp,sizeof(double *));
            dfcoll_interp2 = (double **)calloc(dens_Ninterp,sizeof(double *));
            for(i=0;i<dens_Ninterp;i++) {
                fcoll_interp1[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                fcoll_interp2[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                dfcoll_interp1[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                dfcoll_interp2[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            }
            
            delNL0_rev = (float **)calloc(HII_TOT_NUM_PIXELS,sizeof(float *));
            for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
                dens_grid_int_vals[i] = (short *)calloc((float)global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(short));
                delNL0_rev[i] = (float *)calloc((float)global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            }
        }
        
        dstarlya_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        SingleVal_int = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(short));
        
        freq_int_heat_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_ion_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_lya_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_heat_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_ion_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_lya_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
        for(i=0;i<x_int_NXHII;i++) {
            freq_int_heat_tbl[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_ion_tbl[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_lya_tbl[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_heat_tbl_diff[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_ion_tbl_diff[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_lya_tbl_diff[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
        
        // Grids/arrays that are re-evaluated for each zp
        fcoll_R_array = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        inverse_diff = calloc(x_int_NXHII,sizeof(float));
        zpp_growth = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        
        sigma_Tmin = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        ST_over_PS = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        sum_lyn = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        zpp_for_evolve_list = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        
        TsInterpArraysInitialised = true;
LOG_SUPER_DEBUG("initalised Ts Interp Arrays");
    }
    
    ///////////////////////////////  BEGIN INITIALIZATION   //////////////////////////////
    growth_factor_z = dicke(perturbed_field_redshift);
    inverse_growth_factor_z = 1./growth_factor_z;
    
    //set the minimum ionizing source mass
    // In v1.4 the miinimum ionizing source mass does not depend on redshift.
    // For the constant ionizing efficiency parameter, M_MIN is set to be M_TURN which is a sharp cut-off.
    // For the new parametrization, the number of halos hosting active galaxies (i.e. the duty cycle) is assumed to
    // exponentially decrease below M_TURNOVER Msun, : fduty \propto e^(- M_TURNOVER / M)
    // In this case, we define M_MIN = M_TURN/50, i.e. the M_MIN is integration limit to compute follapse fraction.
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        M_MIN = (astro_params->M_TURN)/50.;
    }
    else {
        
        if(flag_options->M_MIN_in_Mass) {
            M_MIN = (astro_params->M_TURN)/50.;
        }
        else {
            //set the minimum source mass
            if (astro_params->X_RAY_Tvir_MIN < 9.99999e3) { // neutral IGM
                mu_for_Ts = 1.22;
            }
            else {  // ionized IGM
                mu_for_Ts = 0.6;
            }
            M_MIN = TtoM(redshift, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
        }
    }
    
    init_ps();

LOG_SUPER_DEBUG("Initialised PS");

    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;
    }
    else {
        ION_EFF_FACTOR = astro_params->HII_EFF_FACTOR;
    }
    
    // Initialize some interpolation tables
    if(this_spin_temp->first_box || (fabs(initialised_redshift - perturbed_field_redshift) > 0.0001) ) {
        LOG_SUPER_DEBUG("About to initialise heat");
        init_heat();
        LOG_SUPER_DEBUG("Initialised heat");

        if(flag_options->M_MIN_in_Mass || flag_options->USE_MASS_DEPENDENT_ZETA) {
            if(initialiseSigmaMInterpTable(M_MIN,1e20)!=0) {
                LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
                return(2);
            }
        }
        LOG_SUPER_DEBUG("Initialised sigmaM interp table");

    }
    
    if (redshift > global_params.Z_HEAT_MAX){
LOG_SUPER_DEBUG("redshift greater than Z_HEAT_MAX");
        xe = xion_RECFAST(redshift,0);
        TK = T_RECFAST(redshift,0);
        
        growth_factor_zp = dicke(redshift);

LOG_SUPER_DEBUG("growth factor zp = %f", growth_factor_zp);

        // read file
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    this_spin_temp->Tk_box[HII_R_INDEX(i,j,k)] = TK;
                    this_spin_temp->x_e_box[HII_R_INDEX(i,j,k)] = xe;
                    // compute the spin temperature
                    this_spin_temp->Ts_box[HII_R_INDEX(i,j,k)] = get_Ts(redshift, perturbed_field->density[HII_R_INDEX(i,j,k)]*inverse_growth_factor_z*growth_factor_zp,
                                                                        TK, xe, 0, &curr_xalpha);
                }
            }
        }

LOG_SUPER_DEBUG("read in file");

        if(!flag_options->M_MIN_in_Mass) {
            M_MIN = TtoM(redshift, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
LOG_DEBUG("Attempting to initialise sigmaM table with M_MIN=%e, Tvir_MIN=%e, mu=%e", M_MIN, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
            if(initialiseSigmaMInterpTable(M_MIN,1e20)!=0) {
                LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
                return(2);
            }
        }
LOG_SUPER_DEBUG("Initialised Sigma interp table");

    }
    else {
LOG_SUPER_DEBUG("redshift less than Z_HEAT_MAX");
        // Flag is set for previous spin temperature box as a previous spin temperature box must be passed, which makes it the initial condition
        if(this_spin_temp->first_box) {
LOG_SUPER_DEBUG("Treating as the first box");

            // set boundary conditions for the evolution equations->  values of Tk and x_e at Z_HEAT_MAX
            if (global_params.XION_at_Z_HEAT_MAX > 0) // user has opted to use his/her own value
                xe_BC = global_params.XION_at_Z_HEAT_MAX;
            else// will use the results obtained from recfast
                xe_BC = xion_RECFAST(global_params.Z_HEAT_MAX,0);
            if (global_params.TK_at_Z_HEAT_MAX > 0)
                Tk_BC = global_params.TK_at_Z_HEAT_MAX;
            else
                Tk_BC = T_RECFAST(global_params.Z_HEAT_MAX,0);
            
            // and initialize to the boundary values at Z_HEAT_END
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                previous_spin_temp->Tk_box[ct] = Tk_BC;
                previous_spin_temp->x_e_box[ct] = xe_BC;
            }
            x_e_ave = xe_BC;
            Tk_ave = Tk_BC;
        }
        else {
            x_e_ave = Tk_ave = 0.0;
            
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                x_e_ave += previous_spin_temp->x_e_box[ct];
                Tk_ave += previous_spin_temp->Tk_box[ct];
            }
            x_e_ave /= (float)HII_TOT_NUM_PIXELS;
            Tk_ave /= (float)HII_TOT_NUM_PIXELS;
        }
        
        /////////////// Create the z=0 non-linear density fields smoothed on scale R to be used in computing fcoll //////////////
        R = L_FACTOR*user_params->BOX_LEN/(float)user_params->HII_DIM;
        R_factor = pow(global_params.R_XLy_MAX/R, 1/((float)global_params.NUM_FILTER_STEPS_FOR_Ts));
        //      R_factor = pow(E, log(HII_DIM)/(float)NUM_FILTER_STEPS_FOR_Ts);
LOG_SUPER_DEBUG("Looping through R");

        if(this_spin_temp->first_box || (fabs(initialised_redshift - perturbed_field_redshift) > 0.0001) ) {
        
            // allocate memory for the nonlinear density field
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->density[HII_R_INDEX(i,j,k)];
                    }
                }
            }
            
            ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
            if(user_params->USE_FFTW_WISDOM) {
                // Check to see if the wisdom exists, create it if it doesn't
                sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->HII_DIM);
                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_WISDOM_ONLY);
                    fftwf_execute(plan);
                }
                else {
                    
                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_PATIENT);
                    fftwf_execute(plan);
                    
                    // Store the wisdom for later use
                    fftwf_export_wisdom_to_filename(wisdom_filename);
                    
                    // allocate memory for the nonlinear density field
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<user_params->HII_DIM; k++){
                                *((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->density[HII_R_INDEX(i,j,k)];
                            }
                        }
                    }
                    
                    plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_WISDOM_ONLY);
                    fftwf_execute(plan);
                }
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_ESTIMATE);
                fftwf_execute(plan);
            }
            
            // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
            // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
            for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
            }

            // Smooth the density field, at the same time store the minimum and maximum densities for their usage in the interpolation tables
            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            
                R_values[R_ct] = R;
                
                if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
                    sigma_atR[R_ct] = sigma_z0(RtoM(R));
                }
            
                // copy over unfiltered box
                memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            
                if (R_ct > 0){ // don't filter on cell size
                    filter_box(box, 1, global_params.HEAT_FILTER, R);
                }
                // now fft back to real space
                if(user_params->USE_FFTW_WISDOM) {
                    // Check to see if the wisdom exists, create it if it doesn't
                    sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->HII_DIM);
                    if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                        plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_WISDOM_ONLY);
                        fftwf_execute(plan);
                    }
                    else {
                        
                        plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_PATIENT);
                        fftwf_execute(plan);
                        
                        // Store the wisdom for later use
                        fftwf_export_wisdom_to_filename(wisdom_filename);
                        
                        // copy over unfiltered box
                        memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                        
                        if (R_ct > 0){ // don't filter on cell size
                            filter_box(box, 1, global_params.HEAT_FILTER, R);
                        }
                        
                        plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_WISDOM_ONLY);
                        fftwf_execute(plan);
                    }
                }
                else {
                    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
                    fftwf_execute(plan);
                }
                
                min_density = 0.0;
                max_density = 0.0;
            
                // copy over the values
                for (i=user_params->HII_DIM; i--;){
                    for (j=user_params->HII_DIM; j--;){
                        for (k=user_params->HII_DIM; k--;){
                            curr_delNL0 = *((float *)box + HII_R_FFT_INDEX(i,j,k));
                            
                            if (curr_delNL0 < -1){ // correct for alliasing in the filtering step
                                curr_delNL0 = -1+FRACT_FLOAT_ERR;
                            }
                            
                            // and linearly extrapolate to z=0
                            curr_delNL0 *= inverse_growth_factor_z;
                            
                            if(flag_options->USE_MASS_DEPENDENT_ZETA) {
                                delNL0[R_ct][HII_R_INDEX(i,j,k)] = curr_delNL0;
                            }
                            else {
                                delNL0_rev[HII_R_INDEX(i,j,k)][R_ct] = curr_delNL0;
                            }
                            
                            if(curr_delNL0 < min_density) {
                                min_density = curr_delNL0;
                            }
                            if(curr_delNL0 > max_density) {
                                max_density = curr_delNL0;
                            }
                        }
                    }
                }
                
                if(min_density < 0.0) {
                    min_density = min_density*1.001;
                    // min_density here can exceed -1. as it is always extrapolated back to the appropriate redshift
                }
                else {
                    min_density = min_density*0.999;
                }
                if(max_density < 0.0) {
                    max_density = max_density*0.999;
                }
                else {
                    max_density = max_density*1.001;
                }
                
                if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
                    delNL0_LL[R_ct] = min_density;
                    delNL0_Offset[R_ct] = 1.e-6 - (delNL0_LL[R_ct]);
                    delNL0_UL[R_ct] = max_density;
                }
                
                min_densities[R_ct] = min_density;
                max_densities[R_ct] = max_density;
                
                R *= R_factor;
            
            } //end for loop through the filter scales R
        }

LOG_SUPER_DEBUG("Finished loop through filter scales R");

        zp = perturbed_field_redshift*1.0001; //higher for rounding
        if(zp > global_params.Z_HEAT_MAX) {
            prev_zp = ((1+zp)/ global_params.ZPRIME_STEP_FACTOR - 1);
        }
        else {
            while (zp < global_params.Z_HEAT_MAX)
                zp = ((1+zp)*global_params.ZPRIME_STEP_FACTOR - 1);
            prev_zp = global_params.Z_HEAT_MAX;
            zp = ((1+zp)/ global_params.ZPRIME_STEP_FACTOR - 1);
        }
        
        // This sets the delta_z step for determining the heating/ionisation integrals. If first box is true,
        // it is only the difference between Z_HEAT_MAX and the redshift. Otherwise it is the difference between
        // the current and previous redshift (this behaviour is chosen to mimic 21cmFAST)
        if(this_spin_temp->first_box) {
            dzp = zp - prev_zp;
        }
        else {
            dzp = redshift - prev_redshift;
        }
        
        determine_zpp_min = perturbed_field_redshift*0.999;
        
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            if (R_ct==0){
                prev_zpp = zp;
                prev_R = 0;
            }
            else{
                prev_zpp = zpp_edge[R_ct-1];
                prev_R = R_values[R_ct-1];
            }
            zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
            zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
        }
        
        determine_zpp_max = zpp*1.001;
        
        if(!flag_options->M_MIN_in_Mass) {
            M_MIN = TtoM(determine_zpp_max, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
            
            if(initialiseSigmaMInterpTable(M_MIN,1e20)!=0) {
                LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
                return(2);
            }
        }

LOG_SUPER_DEBUG("Initialised sigma interp table");

        zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points_SFR-1.0);
        
        dens_width = 1./((double)dens_Ninterp - 1.);
        
        if(this_spin_temp->first_box || (fabs(initialised_redshift - perturbed_field_redshift) > 0.0001) ) {
        
            ////////////////////////////    Create and fill interpolation tables to be used by Ts.c   /////////////////////////////
            
            if(flag_options->USE_MASS_DEPENDENT_ZETA) {
            
                // generates an interpolation table for redshift
                for (i=0; i<zpp_interp_points_SFR;i++) {
                    zpp_interp_table[i] = determine_zpp_min + zpp_bin_width*(float)i;
                }            
                
                /* initialise interpolation of the mean collapse fraction for global reionization.*/
                if(initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10)!=0) {
                    LOG_ERROR("Detected either an infinite or NaN value in initialise_Nion_Ts_spline");
                    return(2);
                }
                
                if(initialise_SFRD_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->F_STAR10)!=0) {
                    LOG_ERROR("Detected either an infinite or NaN value in initialise_SFRD_spline");
                    return(2);
                }
                
            }
            else {
                // An interpolation table for f_coll (delta vs redshift)
                init_FcollTable(determine_zpp_min,determine_zpp_max);
        
                // Determine the sampling of the density values, for the various interpolation tables
                for(ii=0;ii<global_params.NUM_FILTER_STEPS_FOR_Ts;ii++) {
                    log10delNL0_diff_UL[ii] = log10( delNL0_UL[ii] + delNL0_Offset[ii] );
                    log10delNL0_diff[ii] = log10( delNL0_LL[ii] + delNL0_Offset[ii] );
                    delNL0_bw[ii] = ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width;
                    delNL0_ibw[ii] = 1./delNL0_bw[ii];
                }
        
                // Gridding the density values for the interpolation tables
                for(ii=0;ii<global_params.NUM_FILTER_STEPS_FOR_Ts;ii++) {
                    for(j=0;j<dens_Ninterp;j++) {
                        grid_dens[ii][j] = log10delNL0_diff[ii] + ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width*(double)j;
                        grid_dens[ii][j] = pow(10,grid_dens[ii][j]) - delNL0_Offset[ii];
                    }
                }
        
                // Calculate the sigma_z and Fgtr_M values for each point in the interpolation table
                for(i=0;i<zpp_interp_points_SFR;i++) {
                    zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points_SFR-1.0);
            
                    if(flag_options->M_MIN_in_Mass) {
                        Sigma_Tmin_grid[i] = sigma_z0(FMAX(M_MIN,  M_MIN_WDM));
                        ST_over_PS_arg_grid[i] = FgtrM_General(zpp_grid, FMAX(M_MIN,  M_MIN_WDM));
                    }
                    else {
                        Sigma_Tmin_grid[i] = sigma_z0(FMAX(TtoM(zpp_grid, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                        ST_over_PS_arg_grid[i] = FgtrM_General(zpp_grid, FMAX(TtoM(zpp_grid, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                    }
                }
        
                // Create the interpolation tables for the derivative of the collapsed fraction and the collapse fraction itself
                for(ii=0;ii<global_params.NUM_FILTER_STEPS_FOR_Ts;ii++) {
                    for(i=0;i<zpp_interp_points_SFR;i++) {
                
                        zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points_SFR-1.0);
                        grid_sigmaTmin = Sigma_Tmin_grid[i];
                
                        for(j=0;j<dens_Ninterp;j++) {
                    
                            grid_dens_val = grid_dens[ii][j];
                            fcoll_R_grid[ii][i][j] = sigmaparam_FgtrM_bias(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                            dfcoll_dz_grid[ii][i][j] = dfcoll_dz(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                        }
                    }
                }
        
                // Determine the grid point locations for solving the interpolation tables
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                    for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                        SingleVal_int[R_ct] = (short)floor( ( log10(delNL0_rev[box_ct][R_ct] + delNL0_Offset[R_ct]) - log10delNL0_diff[R_ct] )*delNL0_ibw[R_ct]);
                    }
                    memcpy(dens_grid_int_vals[box_ct],SingleVal_int,sizeof(short)*global_params.NUM_FILTER_STEPS_FOR_Ts);
                }
            
                // Evaluating the interpolated density field points (for using the interpolation tables for fcoll and dfcoll_dz)
                for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                    OffsetValue = delNL0_Offset[R_ct];
                    DensityValueLow = delNL0_LL[R_ct];
                    delNL0_bw_val = delNL0_bw[R_ct];

                    for(i=0;i<dens_Ninterp;i++) {
                        density_gridpoints[i][R_ct] = pow(10.,( log10( DensityValueLow + OffsetValue) + delNL0_bw_val*((float)i) )) - OffsetValue;
                    }
                }
            }
            
            initialised_redshift = perturbed_field_redshift;
        }

LOG_SUPER_DEBUG("got density gridpoints");

        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
            /* generate a table for interpolation of the collapse fraction with respect to the X-ray heating, as functions of
             filtering scale, redshift and overdensity.
             Note that at a given zp, zpp values depends on the filtering scale R, i.e. f_coll(z(R),delta).
             Compute the conditional mass function, but assume f_{esc10} = 1 and \alpha_{esc} = 0. */
            
            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                if (R_ct==0){
                    prev_zpp = redshift;
                    prev_R = 0;
                }
                else{
                    prev_zpp = zpp_edge[R_ct-1];
                    prev_R = R_values[R_ct-1];
                }
                zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
                zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
                zpp_growth[R_ct] = dicke(zpp);
            }
            
            if(initialise_SFRD_Conditional_table(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,max_densities,zpp_growth,R_values, astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->F_STAR10)!=0) {
                LOG_ERROR("Detected either an infinite or NaN value in initialise_SFRD_Conditional_table");
                return(2);
            };
            
        }

LOG_SUPER_DEBUG("Initialised SFRD table");

        zp = redshift;
        prev_zp = prev_redshift;
        
        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
         
            redshift_int_Nion_z = (int)floor( ( zp - determine_zpp_min )/zpp_bin_width );
            
            if(redshift_int_Nion_z < 0 || (redshift_int_Nion_z + 1) > (zpp_interp_points_SFR - 1)) {
                LOG_ERROR("I have overstepped my allocated memory for the interpolation table Nion_z_val");
                return(2);
            }
            
            redshift_table_Nion_z = determine_zpp_min + zpp_bin_width*(float)redshift_int_Nion_z;
            
            Splined_Fcollzp_mean = Nion_z_val[redshift_int_Nion_z] + ( zp - redshift_table_Nion_z )*( Nion_z_val[redshift_int_Nion_z+1] - Nion_z_val[redshift_int_Nion_z] )/(zpp_bin_width);
            
            if ( Splined_Fcollzp_mean < 1e-15 )
                NO_LIGHT = 1;
            else
                NO_LIGHT = 0;
            
            
            filling_factor_of_HI_zp = 1 - ION_EFF_FACTOR * Splined_Fcollzp_mean / (1.0 - x_e_ave);
            
        }
        else {
            
            if(flag_options->M_MIN_in_Mass) {
             
                if (FgtrM(zp, FMAX(M_MIN,  M_MIN_WDM)) < 1e-15 )
                    NO_LIGHT = 1;
                else
                    NO_LIGHT = 0;
                
                M_MIN_at_zp = M_MIN;
            }
            else {
            
                if (FgtrM(zp, FMAX(TtoM(zp, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM)) < 1e-15 )
                    NO_LIGHT = 1;
                else
                    NO_LIGHT = 0;
        
                M_MIN_at_zp = get_M_min_ion(zp);
            }
            filling_factor_of_HI_zp = 1 - ION_EFF_FACTOR * FgtrM_General(zp, M_MIN_at_zp) / (1.0 - x_e_ave);
        }
        
        if (filling_factor_of_HI_zp > 1) filling_factor_of_HI_zp=1;
        
        // let's initialize an array of redshifts (z'') corresponding to the
        // far edge of the dz'' filtering shells
        // and the corresponding minimum halo scale, sigma_Tmin,
        // as well as an array of the frequency integrals
LOG_SUPER_DEBUG("beginning loop over R_ct");

        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            if (R_ct==0){
                prev_zpp = zp;
                prev_R = 0;
            }
            else{
                prev_zpp = zpp_edge[R_ct-1];
                prev_R = R_values[R_ct-1];
            }

            zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
            zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
            
            zpp_for_evolve_list[R_ct] = zpp;
            if (R_ct==0){
                dzpp_for_evolve = zp - zpp_edge[0];
            }
            else{
                dzpp_for_evolve = zpp_edge[R_ct-1] - zpp_edge[R_ct];
            }
            zpp_growth[R_ct] = dicke(zpp);
            
            fcoll_R_array[R_ct] = 0.0;
            
            // let's now normalize the total collapse fraction so that the mean is the
            // Sheth-Torman collapse fraction
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation
                
                redshift_int_SFRD = (int)floor( ( zpp - determine_zpp_min )/zpp_bin_width );

                if(redshift_int_SFRD < 0 || (redshift_int_SFRD + 1) > (zpp_interp_points_SFR - 1)) {
                    LOG_ERROR("I have overstepped my allocated memory for the interpolation table SFRD_val");
                    return(2);
                }
                
                redshift_table_SFRD = determine_zpp_min + zpp_bin_width*(float)redshift_int_SFRD;

                Splined_SFRD_zpp = SFRD_val[redshift_int_SFRD] + ( zpp - redshift_table_SFRD )*( SFRD_val[redshift_int_SFRD+1] - SFRD_val[redshift_int_SFRD] )/(zpp_bin_width);

                ST_over_PS[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve);
                ST_over_PS[R_ct] *= Splined_SFRD_zpp;
                
                SFR_timescale_factor[R_ct] = hubble(zpp)*fabs(dtdz(zpp));
                
            }
            else {
                
                // Determining values for the evaluating the interpolation table
                zpp_gridpoint1_int = (int)floor((zpp - determine_zpp_min)/zpp_bin_width);
                zpp_gridpoint2_int = zpp_gridpoint1_int + 1;
                
                if(zpp_gridpoint1_int < 0 || (zpp_gridpoint1_int + 1) > (zpp_interp_points_SFR - 1)) {
                    LOG_ERROR("I have overstepped my allocated memory for the interpolation table fcoll_R_grid");
                    return(2);
                }
                
                zpp_gridpoint1 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint1_int;
                zpp_gridpoint2 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint2_int;
                
                grad1 = ( zpp_gridpoint2 - zpp )/( zpp_gridpoint2 - zpp_gridpoint1 );
                grad2 = ( zpp - zpp_gridpoint1 )/( zpp_gridpoint2 - zpp_gridpoint1 );
                
                sigma_Tmin[R_ct] = Sigma_Tmin_grid[zpp_gridpoint1_int] + grad2*( Sigma_Tmin_grid[zpp_gridpoint2_int] - Sigma_Tmin_grid[zpp_gridpoint1_int] );

                // Evaluating the interpolation table for the collapse fraction and its derivative
                for(i=0;i<(dens_Ninterp-1);i++) {
                    dens_grad = 1./( density_gridpoints[i+1][R_ct] - density_gridpoints[i][R_ct] );
                    
                    fcoll_interp1[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                    fcoll_interp2[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;
                    
                    dfcoll_interp1[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                    dfcoll_interp2[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;
                    
                }
                
                // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation
                ST_over_PS[R_ct] = dzpp_for_evolve * pow(1+zpp, -(astro_params->X_RAY_SPEC_INDEX));
                ST_over_PS[R_ct] *= ( ST_over_PS_arg_grid[zpp_gridpoint1_int] + grad2*( ST_over_PS_arg_grid[zpp_gridpoint2_int] - ST_over_PS_arg_grid[zpp_gridpoint1_int] ) );
                
            }

            lower_int_limit = FMAX(nu_tau_one_approx(zp, zpp, x_e_ave, filling_factor_of_HI_zp), (astro_params->NU_X_THRESH)*NU_over_EV);

            if(isfinite(lower_int_limit)==0) {
                LOG_ERROR("lower_int_limit has returned either an infinity or a NaN");
                return(2);
            }
            
            if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly

            // set up frequency integral table for later interpolation for the cell's x_e value
            for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);
            
                if(isfinite(freq_int_heat_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_ion_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_lya_tbl[x_e_ct][R_ct])==0) {
                    LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
                    return(2);
                }
            }
            
            // and create the sum over Lya transitions from direct Lyn flux
            sum_lyn[R_ct] = 0;
            for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                if (zpp > zmax(zp, n_ct))
                    continue;
                    
                nuprime = nu_n(n_ct)*(1+zpp)/(1.0+zp);
                sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0);
            }
            
        } // end loop over R_ct filter steps

LOG_SUPER_DEBUG("finished looping over R_ct filter steps");

        fcoll_interp_high_min = global_params.CRIT_DENS_TRANSITION;
        fcoll_interp_high_bin_width = 1./((float)NSFR_high-1.)*(Deltac - fcoll_interp_high_min);
        fcoll_interp_high_bin_width_inv = 1./fcoll_interp_high_bin_width;
        
        // Calculate fcoll for each smoothing radius
        if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
            for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                    
                    if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                        table_int_boundexceeded = 1;
                    }
                    
                    fcoll_R_array[R_ct] += ( fcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*( density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct] ) + fcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*( delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct] ) );
                }
            }
            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                ST_over_PS[R_ct] = ST_over_PS[R_ct]/(fcoll_R_array[R_ct]/(double)HII_TOT_NUM_PIXELS);
            }
         
            if(table_int_boundexceeded==1) {
                LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of fcoll");
                return(2);
            }

            
        }
        
        // scroll through each cell and update the temperature and residual ionization fraction
        growth_factor_zp = dicke(zp);
        dgrowth_factor_dzp = ddicke_dz(zp);
        dt_dzp = dtdz(zp);
            
        // Conversion of the input bolometric luminosity to a ZETA_X, as used to be used in Ts.c
        // Conversion here means the code otherwise remains the same as the original Ts.c
        if(fabs(astro_params->X_RAY_SPEC_INDEX - 1.0) < 0.000001) {
            Luminosity_converstion_factor = (astro_params->NU_X_THRESH)*NU_over_EV * log( global_params.NU_X_BAND_MAX/(astro_params->NU_X_THRESH) );
            Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
        }
        else {
            Luminosity_converstion_factor = pow( (global_params.NU_X_BAND_MAX)*NU_over_EV , 1. - (astro_params->X_RAY_SPEC_INDEX) ) - pow( (astro_params->NU_X_THRESH)*NU_over_EV , 1. - (astro_params->X_RAY_SPEC_INDEX) ) ;
            Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
            Luminosity_converstion_factor *= pow( (astro_params->NU_X_THRESH)*NU_over_EV, - (astro_params->X_RAY_SPEC_INDEX) )*(1 - (astro_params->X_RAY_SPEC_INDEX));
        }
        // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the definition of Luminosity)
        Luminosity_converstion_factor *= (3.1556226e7)/(hplank);
            
        // Leave the original 21cmFAST code for reference. Refer to Greig & Mesinger (2017) for the new parameterisation.
        const_zp_prefactor = ( (astro_params->L_X) * Luminosity_converstion_factor ) / ((astro_params->NU_X_THRESH)*NU_over_EV) * C * astro_params->F_STAR10 * cosmo_params->OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, astro_params->X_RAY_SPEC_INDEX+3);
        //          This line below is kept purely for reference w.r.t to the original 21cmFAST
        //            const_zp_prefactor = ZETA_X * X_RAY_SPEC_INDEX / NU_X_THRESH * C * F_STAR * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, X_RAY_SPEC_INDEX+3);
        
        //////////////////////////////  LOOP THROUGH BOX //////////////////////////////
        
        J_alpha_ave = xalpha_ave = Xheat_ave = Xion_ave = 0.;

        // Extra pre-factors etc. are defined here, as they are independent of the density field, and only have to be computed once per z' or R_ct, rather than each box_ct
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            dstarlya_dt_prefactor[R_ct]  = ( pow(1+zp,2)*(1+zpp_for_evolve_list[R_ct]) * sum_lyn[R_ct] )/( pow(1+zpp_for_evolve_list[R_ct], -(astro_params->X_RAY_SPEC_INDEX)) );
        }
            
        // Required quantities for calculating the IGM spin temperature
        // Note: These used to be determined in evolveInt (and other functions). But I moved them all here, into a single location.
        Trad_fast = T_cmb*(1.0+zp);
        Trad_fast_inv = 1.0/Trad_fast;
        TS_prefactor = pow(1.0e-7*(1.342881e-7 / hubble(zp))*No*pow(1+zp,3),1./3.);
        xa_tilde_prefactor = 1.66e11/(1.0+zp);
            
        xc_inverse =  pow(1.0+zp,3.0)*T21/( Trad_fast*A10_HYPERFINE );
            
        dcomp_dzp_prefactor = (-1.51e-4)/(hubble(zp)/Ho)/(cosmo_params->hlittle)*pow(Trad_fast,4.0)/(1.0+zp);
            
        prefactor_1 = N_b0 * pow(1+zp, 3);
        prefactor_2 = astro_params->F_STAR10 * C * N_b0 / FOURPI;
            
        x_e_ave = 0; Tk_ave = 0; Ts_ave = 0;
            
        // Note: I have removed the call to evolveInt, as is default in the original Ts.c. Removal of evolveInt and moving that computation below, removes unneccesary repeated computations
        // and allows for the interpolation tables that are now used to be more easily computed

        // Can precompute these quantities, independent of the density field (i.e. box_ct)
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            for (i=0; i<(x_int_NXHII-1); i++) {
                m_xHII_low = i;
                m_xHII_high = m_xHII_low + 1;
                
                inverse_diff[i] = 1./(x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
                freq_int_heat_tbl_diff[i][R_ct] = freq_int_heat_tbl[m_xHII_high][R_ct] - freq_int_heat_tbl[m_xHII_low][R_ct];
                freq_int_ion_tbl_diff[i][R_ct] = freq_int_ion_tbl[m_xHII_high][R_ct] - freq_int_ion_tbl[m_xHII_low][R_ct];
                freq_int_lya_tbl_diff[i][R_ct] = freq_int_lya_tbl[m_xHII_high][R_ct] - freq_int_lya_tbl[m_xHII_low][R_ct];
                
            }
        }

LOG_SUPER_DEBUG("looping over box...");

        // Main loop over the entire box for the IGM spin temperature and relevant quantities.
        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        
            for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                
                del_fcoll_Rct[box_ct] = 0.;
                
                dxheat_dt_box[box_ct] = 0.;
                dxion_source_dt_box[box_ct] = 0.;
                dxlya_dt_box[box_ct] = 0.;
                dstarlya_dt_box[box_ct] = 0.;
                
                xHII_call = previous_spin_temp->x_e_box[box_ct];
            
                // Check if ionized fraction is within boundaries; if not, adjust to be within
                if (xHII_call > x_int_XHII[x_int_NXHII-1]*0.999) {
                    xHII_call = x_int_XHII[x_int_NXHII-1]*0.999;
                } else if (xHII_call < x_int_XHII[0]) {
                    xHII_call = 1.001*x_int_XHII[0];
                }
                //interpolate to correct nu integral value based on the cell's ionization state
                
                m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
                
                inverse_val_box[box_ct] = (xHII_call - x_int_XHII[m_xHII_low_box[box_ct]])*inverse_diff[m_xHII_low_box[box_ct]];
                
            }
            
            for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                
                if( min_densities[R_ct]*zpp_growth[R_ct] < -1.) {
                    fcoll_interp_min = log10(global_params.MIN_DENSITY_LOW_LIMIT);
                }
                else {
                    fcoll_interp_min = log10(1. + min_densities[R_ct]*zpp_growth[R_ct]);
                }
                if( max_densities[R_ct]*zpp_growth[R_ct] > global_params.CRIT_DENS_TRANSITION ) {
                    fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+global_params.CRIT_DENS_TRANSITION)-fcoll_interp_min);
                }
                else {
                    fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+max_densities[R_ct]*zpp_growth[R_ct])-fcoll_interp_min);
                }
                fcoll_interp_bin_width_inv = 1./fcoll_interp_bin_width;
                
                ave_fcoll = ave_fcoll_inv = 0.0;
                
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                    
                    curr_dens = delNL0[R_ct][box_ct]*zpp_growth[R_ct];
            
                    if (!NO_LIGHT){
                        // Now determine all the differentials for the heating/ionisation rate equations
                        
                        if (curr_dens < global_params.CRIT_DENS_TRANSITION){
                            
                            if (curr_dens < -1.) {
                                fcoll = 0;
                            }
                            else {
                                dens_val = (log10f(curr_dens+1.) - fcoll_interp_min)*fcoll_interp_bin_width_inv;
                                
                                fcoll_int = (int)floorf( dens_val );
                                
                                if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_low - 1)) {
                                    fcoll_int_boundexceeded = 1;
                                }

                                fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int]*( 1 + (float)fcoll_int - dens_val ) + log10_SFRD_z_low_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );
                                
                                fcoll = expf(fcoll);
                            }
                        }
                        else {

                            if (curr_dens < 0.99*Deltac) {
                                
                                dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;
                                
                                fcoll_int = (int)floorf( dens_val );
                                
                                if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_high - 1)) {
                                    fcoll_int_boundexceeded = 1;
                                }
                                
                                fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );
                                
                            }
                            else {
                                fcoll = pow(10.,10.);
                            }
                        }
                        ave_fcoll += fcoll;
                        
                        del_fcoll_Rct[box_ct] = (1.+curr_dens)*fcoll;
                        
                    }
                }
                
                if(fcoll_int_boundexceeded==1) {
                    LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables for the fcoll/nion_splines");
                    return(2);
                }
                
                
                ave_fcoll /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);
                
                if(ave_fcoll!=0.) {
                    ave_fcoll_inv = 1./ave_fcoll;
                }
                
                dfcoll_dz_val = (ave_fcoll_inv/pow(10.,10.))*ST_over_PS[R_ct]*SFR_timescale_factor[R_ct]/astro_params->t_STAR;
                
                dstarlya_dt_prefactor[R_ct] *= dfcoll_dz_val;
                
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                    
                    // I've added the addition of zero just in case. It should be zero anyway, but just in case there is some weird
                    // numerical thing
                    if(ave_fcoll!=0.) {
                        dxheat_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( (freq_int_heat_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_heat_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                        dxion_source_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( (freq_int_ion_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_ion_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                    }
                    else {
                        dxheat_dt_box[box_ct] += 0.;
                        dxion_source_dt_box[box_ct] += 0.;
                    }
                    
                    if(ave_fcoll!=0.) {
                        dxlya_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( (freq_int_lya_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_lya_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                        dstarlya_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_dt_prefactor[R_ct];
                            
                    }
                    else {
                        dxlya_dt_box[box_ct] += 0.;
                        dstarlya_dt_box[box_ct] += 0.;
                    }
                    
                    // If R_ct == 0, as this is the final smoothing scale (i.e. it is reversed)
                    if(R_ct==0) {
                        
                        x_e = previous_spin_temp->x_e_box[box_ct];
                        T = previous_spin_temp->Tk_box[box_ct];
                        
                        // add prefactors
                        dxheat_dt_box[box_ct] *= const_zp_prefactor;
                        dxion_source_dt_box[box_ct] *= const_zp_prefactor;
                        
                        dxlya_dt_box[box_ct] *= const_zp_prefactor*prefactor_1 * (1.+delNL0[0][box_ct]*growth_factor_zp);
                        dstarlya_dt_box[box_ct] *= prefactor_2;
                        
                        // Now we can solve the evolution equations  //
                        
                        // First let's do dxe_dzp //
                        dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * (1.+delNL0[0][box_ct]*growth_factor_zp);
                        dxe_dzp = dt_dzp*(dxion_source_dt_box[box_ct] - dxion_sink_dt );
                        
                        // Next, let's get the temperature components //
                        // first, adiabatic term
                        dadia_dzp = 3/(1.0+zp);
                        if (fabs(delNL0[0][box_ct]) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                            dadia_dzp += dgrowth_factor_dzp/(1.0/delNL0[0][box_ct]+growth_factor_zp);
                        
                        dadia_dzp *= (2.0/3.0)*T;
                        
                        // next heating due to the changing species
                        dspec_dzp = - dxe_dzp * T / (1+x_e);
                        
                        // next, Compton heating
                        //                dcomp_dzp = dT_comp(zp, T, x_e);
                        dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e+f_He))*( Trad_fast - T );
                        
                        // lastly, X-ray heating
                        dxheat_dzp = dxheat_dt_box[box_ct] * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                        
                        //update quantities
                        x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                        if (x_e > 1) // can do this late in evolution if dzp is too large
                            x_e = 1 - FRACT_FLOAT_ERR;
                        else if (x_e < 0)
                            x_e = 0;
                        if (T < MAX_TK) {
                            T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp ) * dzp;
                        }
                        
                        if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                            T = T_cmb*(1+zp);
                        }
                        
                        this_spin_temp->x_e_box[box_ct] = x_e;
                        this_spin_temp->Tk_box[box_ct] = T;
                        
                        J_alpha_tot = ( dxlya_dt_box[box_ct] + dstarlya_dt_box[box_ct] ); //not really d/dz, but the lya flux
                            
                        // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                        // Algorithm is the same, but written to be more computationally efficient
                            
                        T_inv = expf((-1.)*logf(T));
                        T_inv_sq = expf((-2.)*logf(T));
                            
                        xc_fast = (1.0+delNL0[0][box_ct]*growth_factor_zp)*xc_inverse*( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                            
                        xi_power = TS_prefactor * cbrt((1.0+delNL0[0][box_ct]*growth_factor_zp)*(1.0-x_e)*T_inv_sq);
                        xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot*pow( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power, -1. );
                            
                        //if (J_alpha_tot > 1.0e-20) { // Must use WF effect
                        // New in v1.4
                        if (fabs(J_alpha_tot) > 1.0e-20) { // Must use WF effect
                            TS_fast = Trad_fast;
                            TSold_fast = 0.0;
                            while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {
                                    
                                TSold_fast = TS_fast;
                                    
                                xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - 0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;
                                    
                                TS_fast = (1.0+xa_tilde_fast+xc_fast)*pow(Trad_fast_inv+xa_tilde_fast*( T_inv + 0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                            }
                        } else { // Collisions only
                            TS_fast = (1.0 + xc_fast)/(Trad_fast_inv + xc_fast*T_inv);

                            xa_tilde_fast = 0.0;
                        }
                        
                        if(isfinite(TS_fast)==0) {
                            LOG_ERROR("Estimated spin temperature is either infinite of NaN!");
                            return(2);
                        }
                        
                        if(TS_fast < 0.) {
                            // It can very rarely result in a negative spin temperature. If negative, it is a very small number. Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                            TS_fast = fabs(TS_fast);
                        }
                            
                        this_spin_temp->Ts_box[box_ct] = TS_fast;

                        if(LOG_LEVEL >= DEBUG_LEVEL){
                            J_alpha_ave += J_alpha_tot;
                            xalpha_ave += xa_tilde_fast;
                            Xheat_ave += ( dxheat_dzp );
                            Xion_ave += ( dt_dzp*dxion_source_dt_box[box_ct] );
                            Ts_ave += TS_fast;
                            Tk_ave += T;
                        }

                        x_e_ave += x_e;
                    }
                }
            }
        }
        else {
            
            for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                
                x_e = previous_spin_temp->x_e_box[box_ct];
                T = previous_spin_temp->Tk_box[box_ct];

                xHII_call = x_e;
                
                // Check if ionized fraction is within boundaries; if not, adjust to be within
                if (xHII_call > x_int_XHII[x_int_NXHII-1]*0.999) {
                    xHII_call = x_int_XHII[x_int_NXHII-1]*0.999;
                } else if (xHII_call < x_int_XHII[0]) {
                    xHII_call = 1.001*x_int_XHII[0];
                }
                //interpolate to correct nu integral value based on the cell's ionization state
                
                m_xHII_low = locate_xHII_index(xHII_call);
                
                inverse_val = (xHII_call - x_int_XHII[m_xHII_low])*inverse_diff[m_xHII_low];
                
                // First, let's do the trapazoidal integration over zpp
                dxheat_dt = 0;
                dxion_source_dt = 0;
                dxlya_dt = 0;
                dstarlya_dt = 0;
                
                curr_delNL0 = delNL0_rev[box_ct][0];
                
                if (!NO_LIGHT){
                    // Now determine all the differentials for the heating/ionisation rate equations
                    for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                        
                        if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                            table_int_boundexceeded = 1;
                        }
                        
                        dfcoll_dz_val = ST_over_PS[R_ct]*(1.+delNL0_rev[box_ct][R_ct]*zpp_growth[R_ct])*( dfcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*(density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct]) + dfcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*(delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct]) );
                                            
                        dxheat_dt += dfcoll_dz_val * ( (freq_int_heat_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_heat_tbl[m_xHII_low][R_ct] );
                        dxion_source_dt += dfcoll_dz_val * ( (freq_int_ion_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_ion_tbl[m_xHII_low][R_ct] );
                    
                        dxlya_dt += dfcoll_dz_val * ( (freq_int_lya_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_lya_tbl[m_xHII_low][R_ct] );
                        dstarlya_dt += dfcoll_dz_val*dstarlya_dt_prefactor[R_ct];
                    }
                    
                    if(table_int_boundexceeded==1) {
                        LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of dfcoll_dz_val");
                        return(2);
                    }
                }
                
                // add prefactors
                dxheat_dt *= const_zp_prefactor;
                dxion_source_dt *= const_zp_prefactor;

                dxlya_dt *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                dstarlya_dt *= prefactor_2;
                
                // Now we can solve the evolution equations  //
                
                // First let's do dxe_dzp //
                dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                dxe_dzp = dt_dzp*(dxion_source_dt - dxion_sink_dt );
                
                // Next, let's get the temperature components //
                // first, adiabatic term
                dadia_dzp = 3/(1.0+zp);
                if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                    dadia_dzp += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);
                
                dadia_dzp *= (2.0/3.0)*T;
                
                // next heating due to the changing species
                dspec_dzp = - dxe_dzp * T / (1+x_e);
                
                // next, Compton heating
                dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e+f_He))*( Trad_fast - T );
            
                // lastly, X-ray heating
                dxheat_dzp = dxheat_dt * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                //update quantities
            
                x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                if (x_e > 1) // can do this late in evolution if dzp is too large
                    x_e = 1 - FRACT_FLOAT_ERR;
                else if (x_e < 0)
                    x_e = 0;
                if (T < MAX_TK) {
                    T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp ) * dzp;
                }
                
                if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                    T = T_cmb*(1+zp);
                }
                
                this_spin_temp->x_e_box[box_ct] = x_e;
                this_spin_temp->Tk_box[box_ct] = T;

                J_alpha_tot = ( dxlya_dt + dstarlya_dt ); //not really d/dz, but the lya flux
                    
                // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                // Algorithm is the same, but written to be more computationally efficient
                T_inv = pow(T,-1.);
                T_inv_sq = pow(T,-2.);
                    
                xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                xi_power = TS_prefactor * pow((1.0+curr_delNL0*growth_factor_zp)*(1.0-x_e)*T_inv_sq, 1.0/3.0);
                xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot*pow( 1.0 + 2.98394*xi_power + 1.53583*pow(xi_power,2.) + 3.85289*pow(xi_power,3.), -1. );
                    
                if (J_alpha_tot > 1.0e-20) { // Must use WF effect
                    TS_fast = Trad_fast;
                    TSold_fast = 0.0;
                    while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {
                            
                        TSold_fast = TS_fast;
                            
                        xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - 0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;
                            
                        TS_fast = (1.0+xa_tilde_fast+xc_fast)*pow(Trad_fast_inv+xa_tilde_fast*( T_inv + 0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                    }
                } else { // Collisions only
                    TS_fast = (1.0 + xc_fast)/(Trad_fast_inv + xc_fast*T_inv);
                    xa_tilde_fast = 0.0;
                }
                
                if(isfinite(TS_fast)==0) {
                    LOG_ERROR("Estimated spin temperature is either infinite of NaN!");
                    return(2);
                }
                
                if(TS_fast < 0.) {
                    // It can very rarely result in a negative spin temperature. If negative, it is a very small number. Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                    TS_fast = fabs(TS_fast);
                }
                
                this_spin_temp->Ts_box[box_ct] = TS_fast;

                if(LOG_LEVEL >= DEBUG_LEVEL){
                    J_alpha_ave += J_alpha_tot;
                    xalpha_ave += xa_tilde_fast;
                    Xheat_ave += ( dxheat_dzp );
                    Xion_ave += ( dt_dzp*dxion_source_dt );

                    Ts_ave += TS_fast;
                    Tk_ave += T;
                }


                x_e_ave += x_e;
            }
        }

LOG_SUPER_DEBUG("finished loop");

        /////////////////////////////  END LOOP ////////////////////////////////////////////
        // compute new average values
        if(LOG_LEVEL >= DEBUG_LEVEL){
            x_e_ave /= (double)HII_TOT_NUM_PIXELS;

            Ts_ave /= (double)HII_TOT_NUM_PIXELS;
            Tk_ave /= (double)HII_TOT_NUM_PIXELS;
            J_alpha_ave /= (double)HII_TOT_NUM_PIXELS;
            xalpha_ave /= (double)HII_TOT_NUM_PIXELS;
            Xheat_ave /= (double)HII_TOT_NUM_PIXELS;
            Xion_ave /= (double)HII_TOT_NUM_PIXELS;

            LOG_DEBUG("zp = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e xalpha_ave = %e Xheat_ave = %e Xion_ave = %e",zp,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave);
        }


    } // end main integral loop over z'

    return(0);
}


void free_TsCalcBoxes()
{
    int i,j;
    
    for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
        for(j=0;j<zpp_interp_points_SFR;j++) {
            free(fcoll_R_grid[i][j]);
            free(dfcoll_dz_grid[i][j]);
        }
        free(fcoll_R_grid[i]);
        free(dfcoll_dz_grid[i]);
    }
    free(fcoll_R_grid);
    free(dfcoll_dz_grid);
    
    for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
        free(grid_dens[i]);
    }
    free(grid_dens);
    
    for(i=0;i<dens_Ninterp;i++) {
        free(density_gridpoints[i]);
    }
    free(density_gridpoints);
    
    free(ST_over_PS_arg_grid);
    free(Sigma_Tmin_grid);
    free(fcoll_R_array);
    free(zpp_growth);
    free(inverse_diff);
    
    for(i=0;i<dens_Ninterp;i++) {
        free(fcoll_interp1[i]);
        free(fcoll_interp2[i]);
        free(dfcoll_interp1[i]);
        free(dfcoll_interp2[i]);
    }
    free(fcoll_interp1);
    free(fcoll_interp2);
    free(dfcoll_interp1);
    free(dfcoll_interp2);
    
    for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
        free(dens_grid_int_vals[i]);
        free(delNL0_rev[i]);
    }
    free(dens_grid_int_vals);
    free(delNL0_rev);
    
    free(delNL0_bw);
    free(zpp_for_evolve_list);
    free(R_values);
    free(delNL0_Offset);
    free(delNL0_LL);
    free(delNL0_UL);
    free(SingleVal_int);
    free(dstarlya_dt_prefactor);
    free(delNL0_ibw);
    free(log10delNL0_diff);
    free(log10delNL0_diff_UL);
    
    free(zpp_edge);
    free(sigma_atR);
    free(sigma_Tmin);
    free(ST_over_PS);
    free(sum_lyn);
    
    for(i=0;i<x_int_NXHII;i++) {
        free(freq_int_heat_tbl[i]);
        free(freq_int_ion_tbl[i]);
        free(freq_int_lya_tbl[i]);
        free(freq_int_heat_tbl_diff[i]);
        free(freq_int_ion_tbl_diff[i]);
        free(freq_int_lya_tbl_diff[i]);
    }
    free(freq_int_heat_tbl);
    free(freq_int_ion_tbl);
    free(freq_int_lya_tbl);
    free(freq_int_heat_tbl_diff);
    free(freq_int_ion_tbl_diff);
    free(freq_int_lya_tbl_diff);
    
}
