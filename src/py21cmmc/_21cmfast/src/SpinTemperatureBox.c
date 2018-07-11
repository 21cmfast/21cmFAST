
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

void ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options,
                       struct PerturbedField *p_cubes, struct TsBox *Ts_boxes) {

    // Makes the parameter structs visible to a variety of functions/macros
    if(StructInit==0) {
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_UF(user_params,cosmo_params);
        
        StructInit = 1;
    }
    
    printf("EFF_FACTOR_PL_INDEX = %e HII_EFF_FACTOR = %e R_BUBBLE_MAX = %e ION_Tvir_MIN = %e L_X = %e\n",astro_params->EFF_FACTOR_PL_INDEX,astro_params->HII_EFF_FACTOR,
           astro_params->R_BUBBLE_MAX,astro_params->ION_Tvir_MIN,astro_params->L_X);
    printf("NU_X_THRESH = %e X_RAY_SPEC_INDEX = %e X_RAY_Tvir_MIN = %e\n",astro_params->NU_X_THRESH,astro_params->X_RAY_SPEC_INDEX,astro_params->X_RAY_Tvir_MIN);
    printf("F_STAR = %e t_STAR = %e N_RSD_STEPS = %d\n",astro_params->F_STAR,astro_params->t_STAR,astro_params->N_RSD_STEPS);
   
    printf("NU_X_BAND_MAX = %e NU_X_MAX = %e\n",global_params.NU_X_BAND_MAX,global_params.NU_X_MAX);
    printf("Z_HEAT_MAX = %e ZPRIME_STEP_FACTOR = %e\n",global_params.Z_HEAT_MAX,global_params.ZPRIME_STEP_FACTOR);
    
    printf("INCLUDE_ZETA_PL = %s SUBCELL_RSD = %s INHOMO_RECO = %s\n",flag_options->INCLUDE_ZETA_PL ?"true" : "false",flag_options->SUBCELL_RSD ?"true" : "false",flag_options->INHOMO_RECO ?"true" : "false");
    
    /*
    
    // This is an entire re-write of Ts.c from 21cmFAST. You can refer back to Ts.c in 21cmFAST if this become a little obtuse. The computation has remained the same //
    
    /////////////////// Defining variables for the computation of Ts.c //////////////
    char filename[500];
    FILE *F, *OUT;
    fftwf_plan plan;
    
    unsigned long long ct, FCOLL_SHORT_FACTOR;
    
    int R_ct,i,ii,j,k,i_z,COMPUTE_Ts,x_e_ct,m_xHII_low,m_xHII_high,n_ct, zpp_gridpoint1_int, zpp_gridpoint2_int,zpp_evolve_gridpoint1_int, zpp_evolve_gridpoint2_int;
    
    short dens_grid_int;
    
    double Tk_ave, J_alpha_ave, xalpha_ave, J_alpha_tot, Xheat_ave, Xion_ave, nuprime, Ts_ave, lower_int_limit,Luminosity_converstion_factor,T_inv_TS_fast_inv;
    double dadia_dzp, dcomp_dzp, dxheat_dt, dxion_source_dt, dxion_sink_dt, T, x_e, dxe_dzp, n_b, dspec_dzp, dxheat_dzp, dxlya_dt, dstarlya_dt, fcoll_R;
    double Trad_fast,xc_fast,xc_inverse,TS_fast,TSold_fast,xa_tilde_fast,TS_prefactor,xa_tilde_prefactor,T_inv,T_inv_sq,xi_power,xa_tilde_fast_arg,Trad_fast_inv,TS_fast_inv,dcomp_dzp_prefactor;
    
    float growth_factor_z, inverse_growth_factor_z, R, R_factor, zp, mu_for_Ts, filling_factor_of_HI_zp, dzp, prev_zp, zpp, prev_zpp, prev_R, Tk_BC, xe_BC;
    float xHII_call, curr_xalpha, TK, TS, xe, deltax_highz;
    float zpp_for_evolve,dzpp_for_evolve;
    
    float determine_zpp_max, determine_zpp_min, zpp_grid, zpp_gridpoint1, zpp_gridpoint2,zpp_evolve_gridpoint1, zpp_evolve_gridpoint2, grad1, grad2, grad3, grad4, zpp_bin_width, delNL0_bw_val;
    float OffsetValue, DensityValueLow, min_density, max_density;
    
    double curr_delNL0, inverse_val,prefactor_1,prefactor_2,dfcoll_dz_val, density_eval1, density_eval2, grid_sigmaTmin, grid_dens_val, dens_grad, dens_width;
    
    float M_MIN_WDM =  M_J_WDM();
    
    double total_time, total_time2, total_time3, total_time4;
    
    int Tvir_min_int,Numzp_for_table,counter;
    double X_RAY_Tvir_BinWidth;
    
    X_RAY_Tvir_BinWidth = (X_RAY_Tvir_UPPERBOUND - X_RAY_Tvir_LOWERBOUND)/( (double)X_RAY_Tvir_POINTS - 1. );
    
    // Can speed up computation (~20%) by pre-sampling the fcoll field as a function of X_RAY_TVIR_MIN (performed by calling CreateFcollTable.
    // Can be helpful when HII_DIM > ~128, otherwise its easier to just do the full box
    // This table can be created using "CreateFcollTable.c". See this file for further details.
    if(SHORTEN_FCOLL) {
        
        Tvir_min_int = (int)floor( (log10(X_RAY_Tvir_MIN) - X_RAY_Tvir_LOWERBOUND)/X_RAY_Tvir_BinWidth );
        
        sprintf(filename, "FcollTvirTable_Numzp_ZPRIME_FACTOR%0.2f_logTvirmin%0.6f_logTvirmax%0.6f_XRAY_POINTS%d_z_end%06.6f_%0.2fMpc_%d.dat",ZPRIME_STEP_FACTOR,X_RAY_Tvir_LOWERBOUND,X_RAY_Tvir_UPPERBOUND,X_RAY_Tvir_POINTS,REDSHIFT,BOX_LEN,HII_DIM);
        F = fopen(filename, "rb");
        fread(&Numzp_for_table, sizeof(int),1,F);
        fclose(F);
    }
    else {
        // Need to take on some number for the memory allocation
        Numzp_for_table = 1;
    }
    
    // Allocate the memory for this interpolation table
    double ***Fcoll_R_Table = (double ***)calloc(Numzp_for_table,sizeof(double **));
    for(i=0;i<Numzp_for_table;i++) {
        Fcoll_R_Table[i] = (double **)calloc(X_RAY_Tvir_POINTS,sizeof(double *));
        for(j=0;j<X_RAY_Tvir_POINTS;j++) {
            Fcoll_R_Table[i][j] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
    }
    
    if(SHORTEN_FCOLL) {
        
        sprintf(filename, "FcollTvirTable_ZPRIME_FACTOR%0.2f_logTvirmin%0.6f_logTvirmax%0.6f_XRAY_POINTS%d_z_end%06.6f_%0.2fMpc_%d.dat",ZPRIME_STEP_FACTOR,X_RAY_Tvir_LOWERBOUND,X_RAY_Tvir_UPPERBOUND,X_RAY_Tvir_POINTS,REDSHIFT,BOX_LEN,HII_DIM);
        F = fopen(filename, "rb");
        for(i=0;i<Numzp_for_table;i++) {
            for(j=0;j<X_RAY_Tvir_POINTS;j++) {
                fread(Fcoll_R_Table[i][j], sizeof(double),NUM_FILTER_STEPS_FOR_Ts,F);
            }
        }
    }
    
    // Initialise arrays to be used for the Ts.c computation //
    init_21cmMC_Ts_arrays();
    
    ///////////////////////////////  BEGIN INITIALIZATION   //////////////////////////////
    growth_factor_z = dicke(REDSHIFT);
    inverse_growth_factor_z = 1./growth_factor_z;
    
    if (X_RAY_Tvir_MIN < 9.99999e3) // neutral IGM
        mu_for_Ts = 1.22;
    else // ionized IGM
        mu_for_Ts = 0.6;
    
    //set the minimum ionizing source mass
    M_MIN_at_z = get_M_min_ion(REDSHIFT);
    
    // Initialize some interpolation tables
    init_heat();
    
    // check if we are in the really high z regime before the first stars; if so, simple
    if (REDSHIFT > Z_HEAT_MAX){
        
        // NOTE: THIS NEEDS TO CHANGE. Though it'll only cause problems if this condition is met (which should never happen) //
        
        //(FgtrM(REDSHIFT, FMAX(TtoM(REDSHIFT, X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM)) < 1e-15 ){
        xe = xion_RECFAST(REDSHIFT,0);
        TK = T_RECFAST(REDSHIFT,0);
        
        // open input
        sprintf(filename, "../Boxes/updated_smoothed_deltax_z%06.2f_%i_%.0fMpc",REDSHIFT, HII_DIM, BOX_LEN);
        F = fopen(filename, "rb");
        
        // open output
        sprintf(filename, "../Boxes/Ts_z%06.2f_zetaX%.1e_alphaX%.1f_TvirminX%.1e_zetaIon%.2f_Pop%i_%i_%.0fMpc", REDSHIFT, HII_EFF_FACTOR, X_RAY_SPEC_INDEX, X_RAY_Tvir_MIN, R_BUBBLE_MAX, Pop, HII_DIM, BOX_LEN);
        
        // read file
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    fread(&deltax_highz, sizeof(float), 1, F);
                    
                    // compute the spin temperature
                    TS = get_Ts(REDSHIFT, deltax_highz, TK, xe, 0, &curr_xalpha);
                    
                    // and print it out
                    fwrite(&TS, sizeof(float), 1, OUT);
                }
            }
        }
        
        destruct_heat(); fclose(F); fclose(OUT);
    }
    else {
        
        // set boundary conditions for the evolution equations->  values of Tk and x_e at Z_HEAT_MAX
        if (XION_at_Z_HEAT_MAX > 0) // user has opted to use his/her own value
            xe_BC = XION_at_Z_HEAT_MAX;
        else// will use the results obtained from recfast
            xe_BC = xion_RECFAST(Z_HEAT_MAX,0);
        if (TK_at_Z_HEAT_MAX > 0)
            Tk_BC = TK_at_Z_HEAT_MAX;
        else
            Tk_BC = T_RECFAST(Z_HEAT_MAX,0);
        
        /////////////// Create the z=0 non-linear density fields smoothed on scale R to be used in computing fcoll //////////////
        R = L_FACTOR*BOX_LEN/(float)HII_DIM;
        R_factor = pow(R_XLy_MAX/R, 1/(float)NUM_FILTER_STEPS_FOR_Ts);
        //      R_factor = pow(E, log(HII_DIM)/(float)NUM_FILTER_STEPS_FOR_Ts);
        
        ///////////////////  Read in density box at z-prime  ///////////////
        if(GenerateNewICs) {
            
            // If GenerateNewICs == 1, we are generating a new set of initial conditions and density field. Hence, calculate the density field to be used for Ts.c
            
            ComputePerturbField(REDSHIFT);
            
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                        *((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k)) = LOWRES_density_REDSHIFT[HII_R_INDEX(i,j,k)];
                    }
                }
            }
        }
        else {
            
            // Read in a pre-computed density field which is stored in the "Boxes" folder
            
            // allocate memory for the nonlinear density field and open file
            sprintf(filename, "../Boxes/updated_smoothed_deltax_z%06.2f_%i_%.0fMpc",REDSHIFT, HII_DIM, BOX_LEN);
            F = fopen(filename, "rb");
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                        fread((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F);
                    }
                }
            }
            fclose(F);
        }
        
        ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
        plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        fftwf_cleanup();
        
        // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
        // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
        for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
            unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
        }
        
        // Smooth the density field, at the same time store the minimum and maximum densities for their usage in the interpolation tables
        for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            
            R_values[R_ct] = R;
            sigma_atR[R_ct] = sigma_z0(RtoM(R));
            
            // copy over unfiltered box
            memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            
            if (R_ct > 0){ // don't filter on cell size
                HII_filter(box, HEAT_FILTER, R);
            }
            // now fft back to real space
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
            fftwf_execute(plan);
            
            min_density = 0.0;
            max_density = 0.0;
            
            // copy over the values
            for (i=HII_DIM; i--;){
                for (j=HII_DIM; j--;){
                    for (k=HII_DIM; k--;){
                        curr_delNL0 = *((float *) box + HII_R_FFT_INDEX(i,j,k));
                        
                        if (curr_delNL0 < -1){ // correct for alliasing in the filtering step
                            curr_delNL0 = -1+FRACT_FLOAT_ERR;
                        }
                        
                        // and linearly extrapolate to z=0
                        curr_delNL0 *= inverse_growth_factor_z;
                        
                        delNL0_rev[HII_R_INDEX(i,j,k)][R_ct] = curr_delNL0;
                        
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
                delNL0_LL[R_ct] = min_density*1.001;
                delNL0_Offset[R_ct] = 1.e-6 - (delNL0_LL[R_ct]);
            }
            else {
                delNL0_LL[R_ct] = min_density*0.999;
                delNL0_Offset[R_ct] = 1.e-6 + (delNL0_LL[R_ct]);
            }
            if(max_density < 0.0) {
                delNL0_UL[R_ct] = max_density*0.999;
            }
            else {
                delNL0_UL[R_ct] = max_density*1.001;
            }
            
            R *= R_factor;
            
        } //end for loop through the filter scales R
        
        fftwf_destroy_plan(plan);
        fftwf_cleanup();
        
        // and initialize to the boundary values at Z_HEAT_END
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
            Tk_box[ct] = Tk_BC;
            x_e_box[ct] = xe_BC;
        }
        x_e_ave = xe_BC;
        Tk_ave = Tk_BC;
        
        ////////////////////////////    END INITIALIZATION   /////////////////////////////
        
        // main trapezoidal integral over z' (see eq. ? in Mesinger et al. 2009)
        zp = REDSHIFT*1.0001; //higher for rounding
        while (zp < Z_HEAT_MAX)
            zp = ((1+zp)*ZPRIME_STEP_FACTOR - 1);
        prev_zp = Z_HEAT_MAX;
        zp = ((1+zp)/ ZPRIME_STEP_FACTOR - 1);
        dzp = zp - prev_zp;
        COMPUTE_Ts = 0;
        
        determine_zpp_min = REDSHIFT*0.999;
        
        for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
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
        
        ////////////////////////////    Create and fill interpolation tables to be used by Ts.c   /////////////////////////////
        
        
        // An interpolation table for f_coll (delta vs redshift)
        init_FcollTable(determine_zpp_min,determine_zpp_max);
        
        zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points-1.0);
        
        dens_width = 1./((double)dens_Ninterp - 1.);
        
        // Determine the sampling of the density values, for the various interpolation tables
        for(ii=0;ii<NUM_FILTER_STEPS_FOR_Ts;ii++) {
            log10delNL0_diff_UL[ii] = log10( delNL0_UL[ii] + delNL0_Offset[ii] );
            log10delNL0_diff[ii] = log10( delNL0_LL[ii] + delNL0_Offset[ii] );
            delNL0_bw[ii] = ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width;
            delNL0_ibw[ii] = 1./delNL0_bw[ii];
        }
        
        // Gridding the density values for the interpolation tables
        for(ii=0;ii<NUM_FILTER_STEPS_FOR_Ts;ii++) {
            for(j=0;j<dens_Ninterp;j++) {
                grid_dens[ii][j] = log10delNL0_diff[ii] + ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width*(double)j;
                grid_dens[ii][j] = pow(10,grid_dens[ii][j]) - delNL0_Offset[ii];
            }
        }
        
        // Calculate the sigma_z and Fgtr_M values for each point in the interpolation table
        for(i=0;i<zpp_interp_points;i++) {
            zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points-1.0);
            
            Sigma_Tmin_grid[i] = sigma_z0(FMAX(TtoM(zpp_grid, X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
            ST_over_PS_arg_grid[i] = FgtrM_st(zpp_grid, FMAX(TtoM(zpp_grid, X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
        }
        
        // Create the interpolation tables for the derivative of the collapsed fraction and the collapse fraction itself
        for(ii=0;ii<NUM_FILTER_STEPS_FOR_Ts;ii++) {
            for(i=0;i<zpp_interp_points;i++) {
                
                zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points-1.0);
                grid_sigmaTmin = Sigma_Tmin_grid[i];
                
                for(j=0;j<dens_Ninterp;j++) {
                    
                    grid_dens_val = grid_dens[ii][j];
                    if(!SHORTEN_FCOLL) {
                        fcoll_R_grid[ii][i][j] = sigmaparam_FgtrM_bias(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                    }
                    dfcoll_dz_grid[ii][i][j] = dfcoll_dz(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                }
            }
        }
        
        // Determine the grid point locations for solving the interpolation tables
        for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
            for (R_ct=NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                SingleVal_int[R_ct] = (short)floor( ( log10(delNL0_rev[box_ct][R_ct] + delNL0_Offset[R_ct]) - log10delNL0_diff[R_ct] )*delNL0_ibw[R_ct]);
            }
            memcpy(dens_grid_int_vals[box_ct],SingleVal_int,sizeof(short)*NUM_FILTER_STEPS_FOR_Ts);
        }
        
        // Evaluating the interpolated density field points (for using the interpolation tables for fcoll and dfcoll_dz)
        for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            OffsetValue = delNL0_Offset[R_ct];
            DensityValueLow = delNL0_LL[R_ct];
            delNL0_bw_val = delNL0_bw[R_ct];
            
            for(i=0;i<dens_Ninterp;i++) {
                density_gridpoints[i][R_ct] = pow(10.,( log10( DensityValueLow + OffsetValue) + delNL0_bw_val*((float)i) )) - OffsetValue;
            }
        }
        
        counter = 0;
        
        // This is the main loop for calculating the IGM spin temperature. Structure drastically different from Ts.c in 21cmFAST, however algorithm and computation remain the same.
        while (zp > REDSHIFT){
            // check if we will next compute the spin temperature (i.e. if this is the final zp step)
            if (Ts_verbose || (((1+zp) / ZPRIME_STEP_FACTOR) < (REDSHIFT+1)) )
                COMPUTE_Ts = 1;
            
            // check if we are in the really high z regime before the first stars..
            if (FgtrM(zp, FMAX(TtoM(zp, X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM)) < 1e-15 )
                NO_LIGHT = 1;
            else
                NO_LIGHT = 0;
            
            M_MIN_at_zp = get_M_min_ion(zp);
            filling_factor_of_HI_zp = 1 - HII_EFF_FACTOR * FgtrM_st(zp, M_MIN_at_zp) / (1.0 - x_e_ave);
            if (filling_factor_of_HI_zp > 1) filling_factor_of_HI_zp=1;
            
            // let's initialize an array of redshifts (z'') corresponding to the
            // far edge of the dz'' filtering shells
            // and the corresponding minimum halo scale, sigma_Tmin,
            // as well as an array of the frequency integrals
            for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
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
                
                // Determining values for the evaluating the interpolation table
                zpp_gridpoint1_int = (int)floor((zpp - determine_zpp_min)/zpp_bin_width);
                zpp_gridpoint2_int = zpp_gridpoint1_int + 1;
                
                zpp_gridpoint1 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint1_int;
                zpp_gridpoint2 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint2_int;
                
                grad1 = ( zpp_gridpoint2 - zpp )/( zpp_gridpoint2 - zpp_gridpoint1 );
                grad2 = ( zpp - zpp_gridpoint1 )/( zpp_gridpoint2 - zpp_gridpoint1 );
                
                sigma_Tmin[R_ct] = Sigma_Tmin_grid[zpp_gridpoint1_int] + grad2*( Sigma_Tmin_grid[zpp_gridpoint2_int] - Sigma_Tmin_grid[zpp_gridpoint1_int] );
                
                // let's now normalize the total collapse fraction so that the mean is the
                // Sheth-Torman collapse fraction
                
                zpp_for_evolve_list[R_ct] = zpp;
                if (R_ct==0){
                    dzpp_for_evolve = zp - zpp_edge[0];
                }
                else{
                    dzpp_for_evolve = zpp_edge[R_ct-1] - zpp_edge[R_ct];
                }
                zpp_growth[R_ct] = dicke(zpp);
                
                // Evaluating the interpolation table for the collapse fraction and its derivative
                for(i=0;i<(dens_Ninterp-1);i++) {
                    dens_grad = 1./( density_gridpoints[i+1][R_ct] - density_gridpoints[i][R_ct] );
                    
                    if(!SHORTEN_FCOLL) {
                        fcoll_interp1[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                        fcoll_interp2[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;
                    }
                    
                    dfcoll_interp1[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                    dfcoll_interp2[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;
                }
                
                fcoll_R_array[R_ct] = 0.0;
                
                // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation
                ST_over_PS[R_ct] = dzpp_for_evolve * pow(1+zpp, -X_RAY_SPEC_INDEX);
                ST_over_PS[R_ct] *= ( ST_over_PS_arg_grid[zpp_gridpoint1_int] + grad2*( ST_over_PS_arg_grid[zpp_gridpoint2_int] - ST_over_PS_arg_grid[zpp_gridpoint1_int] ) );
                
                lower_int_limit = FMAX(nu_tau_one_approx(zp, zpp, x_e_ave, filling_factor_of_HI_zp), NU_X_THRESH);
                
                if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly
                
                // set up frequency integral table for later interpolation for the cell's x_e value
                for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                    freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                    freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                    if (COMPUTE_Ts)
                        freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);
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
            
            // Calculate fcoll for each smoothing radius
            
            // Can speed up computation (~20%) by pre-sampling the fcoll field as a function of X_RAY_TVIR_MIN (performed by calling CreateFcollTable.
            // Can be helpful when HII_DIM > ~128, otherwise its easier to just do the full box
            if(SHORTEN_FCOLL) {
                for(R_ct=0;R_ct<NUM_FILTER_STEPS_FOR_Ts;R_ct++) {
                    fcoll_R = Fcoll_R_Table[counter][Tvir_min_int][R_ct] + ( log10(X_RAY_Tvir_MIN) - ( X_RAY_Tvir_LOWERBOUND + (double)Tvir_min_int*X_RAY_Tvir_BinWidth ) )*( Fcoll_R_Table[counter][Tvir_min_int+1][R_ct] - Fcoll_R_Table[counter][Tvir_min_int][R_ct] )/X_RAY_Tvir_BinWidth;
                    
                    ST_over_PS[R_ct] = ST_over_PS[R_ct]/fcoll_R;
                }
            }
            else {
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                    for (R_ct=NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                        fcoll_R_array[R_ct] += ( fcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*( density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct] ) + fcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*( delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct] ) );
                    }
                }
                for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                    ST_over_PS[R_ct] = ST_over_PS[R_ct]/(fcoll_R_array[R_ct]/(double)HII_TOT_NUM_PIXELS);
                }
            }
            // scroll through each cell and update the temperature and residual ionization fraction
            growth_factor_zp = dicke(zp);
            dgrowth_factor_dzp = ddicke_dz(zp);
            dt_dzp = dtdz(zp);
            
            // Conversion of the input bolometric luminosity to a ZETA_X, as used to be used in Ts.c
            // Conversion here means the code otherwise remains the same as the original Ts.c
            if(fabs(X_RAY_SPEC_INDEX - 1.0) < 0.000001) {
                Luminosity_converstion_factor = NU_X_THRESH * log( NU_X_BAND_MAX/NU_X_THRESH );
                Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
            }
            else {
                Luminosity_converstion_factor = pow( NU_X_BAND_MAX , 1. - X_RAY_SPEC_INDEX ) - pow( NU_X_THRESH , 1. - X_RAY_SPEC_INDEX ) ;
                Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
                Luminosity_converstion_factor *= pow( NU_X_THRESH, - X_RAY_SPEC_INDEX )*(1 - X_RAY_SPEC_INDEX);
            }
            // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the definition of Luminosity)
            Luminosity_converstion_factor *= (3.1556226e7)/(hplank);
            
            // Leave the original 21cmFAST code for reference. Refer to Greig & Mesinger (2017) for the new parameterisation.
            const_zp_prefactor = ( L_X * Luminosity_converstion_factor ) / NU_X_THRESH * C * F_STAR * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, X_RAY_SPEC_INDEX+3);
            //          This line below is kept purely for reference w.r.t to the original 21cmFAST
            //            const_zp_prefactor = ZETA_X * X_RAY_SPEC_INDEX / NU_X_THRESH * C * F_STAR * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, X_RAY_SPEC_INDEX+3);
            
            //////////////////////////////  LOOP THROUGH BOX //////////////////////////////
            
            J_alpha_ave = xalpha_ave = Xheat_ave = Xion_ave = 0.;
            
            // Extra pre-factors etc. are defined here, as they are independent of the density field, and only have to be computed once per z' or R_ct, rather than each box_ct
            for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                dstarlya_dt_prefactor[R_ct]  = ( pow(1+zp,2)*(1+zpp_for_evolve_list[R_ct]) * sum_lyn[R_ct] )/( pow(1+zpp_for_evolve_list[R_ct], -X_RAY_SPEC_INDEX) );
            }
            
            // Required quantities for calculating the IGM spin temperature
            // Note: These used to be determined in evolveInt (and other functions). But I moved them all here, into a single location.
            Trad_fast = T_cmb*(1.0+zp);
            Trad_fast_inv = 1.0/Trad_fast;
            TS_prefactor = pow(1.0e-7*(1.342881e-7 / hubble(zp))*No*pow(1+zp,3),1./3.);
            xa_tilde_prefactor = 1.66e11/(1.0+zp);
            
            xc_inverse =  pow(1.0+zp,3.0)*T21/( Trad_fast*A10_HYPERFINE );
            
            dcomp_dzp_prefactor = (-1.51e-4)/(hubble(zp)/Ho)/hlittle*pow(Trad_fast,4.0)/(1.0+zp);
            
            prefactor_1 = N_b0 * pow(1+zp, 3);
            prefactor_2 = F_STAR * C * N_b0 / FOURPI;
            
            x_e_ave = 0; Tk_ave = 0; Ts_ave = 0;
            
            // Note: I have removed the call to evolveInt, as is default in the original Ts.c. Removal of evolveInt and moving that computation below, removes unneccesary repeated computations
            // and allows for the interpolation tables that are now used to be more easily computed
            
            // Can precompute these quantities, independent of the density field (i.e. box_ct)
            for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                for (i=0; i<(x_int_NXHII-1); i++) {
                    m_xHII_low = i;
                    m_xHII_high = m_xHII_low + 1;
                    
                    inverse_diff[i] = 1./(x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
                    freq_int_heat_tbl_diff[i][R_ct] = freq_int_heat_tbl[m_xHII_high][R_ct] - freq_int_heat_tbl[m_xHII_low][R_ct];
                    freq_int_ion_tbl_diff[i][R_ct] = freq_int_ion_tbl[m_xHII_high][R_ct] - freq_int_ion_tbl[m_xHII_low][R_ct];
                    freq_int_lya_tbl_diff[i][R_ct] = freq_int_lya_tbl[m_xHII_high][R_ct] - freq_int_lya_tbl[m_xHII_low][R_ct];
                }
            }
            // Main loop over the entire box for the IGM spin temperature and relevant quantities.
            for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                if (!COMPUTE_Ts && (Tk_box[box_ct] > MAX_TK)) //just leave it alone and go to next value
                    continue;
                
                x_e = x_e_box[box_ct];
                T = Tk_box[box_ct];
                
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
                    for (R_ct=NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                        
                        dfcoll_dz_val = ST_over_PS[R_ct]*(1.+delNL0_rev[box_ct][R_ct]*zpp_growth[R_ct])*( dfcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*(density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct]) + dfcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*(delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct]) );
                        
                        dxheat_dt += dfcoll_dz_val * ( (freq_int_heat_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_heat_tbl[m_xHII_low][R_ct] );
                        dxion_source_dt += dfcoll_dz_val * ( (freq_int_ion_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_ion_tbl[m_xHII_low][R_ct] );
                        
                        if (COMPUTE_Ts){
                            dxlya_dt += dfcoll_dz_val * ( (freq_int_lya_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_lya_tbl[m_xHII_low][R_ct] );
                            dstarlya_dt += dfcoll_dz_val*dstarlya_dt_prefactor[R_ct];
                        }
                    }
                }
                
                // add prefactors
                dxheat_dt *= const_zp_prefactor;
                dxion_source_dt *= const_zp_prefactor;
                if (COMPUTE_Ts){
                    dxlya_dt *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                    dstarlya_dt *= prefactor_2;
                }
                
                // Now we can solve the evolution equations  //
                
                // First let's do dxe_dzp //
                dxion_sink_dt = alpha_A(T) * CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
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
                //                dcomp_dzp = dT_comp(zp, T, x_e);
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
                
                x_e_box[box_ct] = x_e;
                Tk_box[box_ct] = T;
                
                if (COMPUTE_Ts){
                    J_alpha_tot = ( dxlya_dt + dstarlya_dt ); //not really d/dz, but the lya flux
                    
                    // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                    // Algorithm is the same, but written to be more computationally efficient
                    T_inv = pow(T,-1.);
                    T_inv_sq = pow(T,-2.);
                    
                    xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*( (1.0-x_e)*No*kappa_10_float(T,0) + x_e*N_b0*kappa_10_elec_float(T,0) + x_e*No*kappa_10_pH_float(T,0) );
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
                    if(TS_fast < 0.) {
                        // It can very rarely result in a negative spin temperature. If negative, it is a very small number. Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                        TS_fast = fabs(TS_fast);
                    }
                    
                    Ts[box_ct] = TS_fast;
                    
                    if(OUTPUT_AVE) {
                        J_alpha_ave += J_alpha_tot;
                        xalpha_ave += xa_tilde_fast;
                        Xheat_ave += ( dxheat_dzp );
                        Xion_ave += ( dt_dzp*dxion_source_dt );
                        
                        Ts_ave += TS_fast;
                        Tk_ave += T;
                    }
                }
                x_e_ave += x_e;
            }
            // For this redshift snapshot, we now determine the ionisation field and subsequently the 21cm brightness temperature map (also the 21cm PS)
            // Note the relatively small tolerance for zp and the input redshift. The user needs to be careful to provide the correct redshifts for evaluating this to high precision.
            // If the light-cone option is set, this criterion should automatically be met
            for(i_z=0;i_z<N_USER_REDSHIFT;i_z++) {
                if(fabs(redshifts[i_z] - zp)<0.001) {
                    
                    memcpy(Ts_z,Ts,sizeof(float)*HII_TOT_NUM_PIXELS);
                    memcpy(x_e_z,x_e_box,sizeof(float)*HII_TOT_NUM_PIXELS);
                    
                    if(i_z==0) {
                        // If in here, it doesn't matter what PREV_REDSHIFT is set to
                        // as the recombinations will not be calculated
                        ComputeIonisationBoxes(i_z,redshifts[i_z],redshifts[i_z]+0.2);
                    }
                    else {
                        ComputeIonisationBoxes(i_z,redshifts[i_z],redshifts[i_z-1]);
                    }
                }
            }
            
            /////////////////////////////  END LOOP ////////////////////////////////////////////
            
            // compute new average values
            x_e_ave /= (double)HII_TOT_NUM_PIXELS;
            
            if(OUTPUT_AVE) {
                Ts_ave /= (double)HII_TOT_NUM_PIXELS;
                Tk_ave /= (double)HII_TOT_NUM_PIXELS;
                J_alpha_ave /= (double)HII_TOT_NUM_PIXELS;
                xalpha_ave /= (double)HII_TOT_NUM_PIXELS;
                Xheat_ave /= (double)HII_TOT_NUM_PIXELS;
                Xion_ave /= (double)HII_TOT_NUM_PIXELS;
                
                printf("zp = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e xalpha_ave = %e Xheat_ave = %e Xion_ave = %e\n",zp,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave);
            }
            
            prev_zp = zp;
            zp = ((1+prev_zp) / ZPRIME_STEP_FACTOR - 1);
            dzp = zp - prev_zp;
            
            counter += 1;
        } // end main integral loop over z'
        
        destroy_21cmMC_Ts_arrays();
        destruct_heat();
    }
    
    for(i=0;i<Numzp_for_table;i++) {
        for(j=0;j<X_RAY_Tvir_POINTS;j++) {
            free(Fcoll_R_Table[i][j]);
        }
        free(Fcoll_R_Table[i]);
    }
    free(Fcoll_R_Table);
     
     
    
     */
}

