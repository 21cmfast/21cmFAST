
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

int ComputeBrightnessTemp(float redshift, int saturated_limit, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                           struct AstroParams *astro_params, struct FlagOptions *flag_options,
                           struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box) {

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    
    char wisdom_filename[500];
    
    double ave;
    
    ave = 0.;
    
    fftwf_plan plan;
    
    float *v = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
    float *vel_gradient = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
    
    float *x_pos = calloc(astro_params->N_RSD_STEPS,sizeof(float));
    float *x_pos_offset = calloc(astro_params->N_RSD_STEPS,sizeof(float));
    float *delta_T_RSD_LOS = calloc(user_params->HII_DIM,sizeof(float));
    
    int i, ii, j, k, n_x, n_y, n_z;
    float k_x, k_y, k_z;
    
    for (i=0; i<user_params->HII_DIM; i++){
        for (j=0; j<user_params->HII_DIM; j++){
            for (k=0; k<user_params->HII_DIM; k++){
                *((float *)v + HII_R_FFT_INDEX(i,j,k)) = perturb_field->velocity[HII_R_INDEX(i,j,k)];
            }
        }
    }
    
    float d1_low, d1_high, d2_low, d2_high, gradient_component, min_gradient_component, subcell_width, x_val1, x_val2, subcell_displacement;
    float RSD_pos_new, RSD_pos_new_boundary_low,RSD_pos_new_boundary_high, fraction_within, fraction_outside, cell_distance;
    
    double dvdx, max_v_deriv;
    float const_factor, T_rad, pixel_Ts_factor, pixel_x_HI, pixel_deltax, H;
    
    init_ps();
    
    T_rad = T_cmb*(1+redshift);
    H = hubble(redshift);
    const_factor = 27 * (cosmo_params->OMb*cosmo_params->hlittle*cosmo_params->hlittle/0.023) *
    sqrt( (0.15/(cosmo_params->OMm)/(cosmo_params->hlittle)/(cosmo_params->hlittle)) * (1.+redshift)/10.0 );
    
    ///////////////////////////////  END INITIALIZATION /////////////////////////////////////////////
    
    // ok, lets fill the delta_T box; which will be the same size as the bubble box
    
    for (i=0; i<user_params->HII_DIM; i++){
        for (j=0; j<user_params->HII_DIM; j++){
            for (k=0; k<user_params->HII_DIM; k++){
                
                pixel_deltax = perturb_field->density[HII_R_INDEX(i,j,k)];
                pixel_x_HI = ionized_box->xH_box[HII_R_INDEX(i,j,k)];
                
                box->brightness_temp[HII_R_INDEX(i,j,k)] = const_factor*pixel_x_HI*(1+pixel_deltax);
                
                if (flag_options->USE_TS_FLUCT) {
                    
                    if(flag_options->SUBCELL_RSD) {
                        // Converting the prefactors into the optical depth, tau. Factor of 1000 is the conversion of spin temperature from K to mK
                        box->brightness_temp[HII_R_INDEX(i,j,k)] *= (1. + redshift)/(1000.*spin_temp->Ts_box[HII_R_INDEX(i,j,k)]);
                    }
                    else {
                        pixel_Ts_factor = (1 - T_rad / spin_temp->Ts_box[HII_R_INDEX(i,j,k)]);
                        box->brightness_temp[HII_R_INDEX(i,j,k)] *= pixel_Ts_factor;
                    }
                }
                
                ave += box->brightness_temp[HII_R_INDEX(i,j,k)];
            }
        }
    }
    if(isfinite(ave)==0) {
        LOG_ERROR("Average brightness temperature is infinite or NaN!");
        return(2);
    }
    
    ave /= (float)HII_TOT_NUM_PIXELS;
    
    x_val1 = 0.;
    x_val2 = 1.;
    
    subcell_width = (user_params->BOX_LEN/(float)user_params->HII_DIM)/(float)(astro_params->N_RSD_STEPS);
    
    float max_cell_distance;
    
    max_cell_distance = 0.;
    
    // now write out the delta_T box
    if (global_params.T_USE_VELOCITIES){
        ave = 0.;
        
        memcpy(vel_gradient, v, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        
        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"real_to_complex_%d.fftwf_wisdom",user_params->HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)vel_gradient, (fftwf_complex *)vel_gradient, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)vel_gradient, (fftwf_complex *)vel_gradient, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                // copy over unfiltered box
                memcpy(vel_gradient, v, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)vel_gradient, (fftwf_complex *)vel_gradient, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)vel_gradient, (fftwf_complex *)vel_gradient, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        
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
                    
                    // take partial deriavative along the line of sight
                    *((fftwf_complex *) vel_gradient + HII_C_INDEX(n_x,n_y,n_z)) *= k_z*I/(float)HII_TOT_NUM_PIXELS;

                }
            }
        }
        
        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"complex_to_real_%d.fftwf_wisdom",user_params->HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)vel_gradient, (float *)vel_gradient, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                
                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)vel_gradient, (float *)vel_gradient, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                // copy over unfiltered box
                memcpy(vel_gradient, v, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                
                // re-perform calculation
                plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)vel_gradient, (fftwf_complex *)vel_gradient, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
                
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
                            
                            // take partial deriavative along the line of sight
                            *((fftwf_complex *) vel_gradient + HII_C_INDEX(n_x,n_y,n_z)) *= k_z*I/(float)HII_TOT_NUM_PIXELS;
                        }
                    }
                }
                
                plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)vel_gradient, (float *)vel_gradient, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)vel_gradient, (float *)vel_gradient, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        
        
        // now add the velocity correction to the delta_T maps (only used for T_S >> T_CMB case).
        max_v_deriv = fabs(global_params.MAX_DVDR*H);
        
        if(flag_options->SUBCELL_RSD) {
            
            // now add the velocity correction to the delta_T maps
            min_gradient_component = 1.0;
            
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        
                        gradient_component = fabs(vel_gradient[HII_R_FFT_INDEX(i,j,k)]/H + 1.0);
                        
                        if(flag_options->USE_TS_FLUCT) {
                            
                            // Calculate the brightness temperature, using the optical depth
                            if(gradient_component < FRACT_FLOAT_ERR) {
                                // Gradient component goes to zero, optical depth diverges. But, since we take exp(-tau), this goes to zero and (1 - exp(-tau)) goes to unity.
                                // Again, factors of 1000. are conversions from K to mK
                                box->brightness_temp[HII_R_INDEX(i,j,k)] = 1000.*(spin_temp->Ts_box[HII_R_INDEX(i,j,k)] - T_rad)/(1. + redshift);
                            }
                            else {
                                box->brightness_temp[HII_R_INDEX(i,j,k)] = (1. - exp(- box->brightness_temp[HII_R_INDEX(i,j,k)]/gradient_component ))*1000.*(spin_temp->Ts_box[HII_R_INDEX(i,j,k)] - T_rad)/(1. + redshift);
                            }
                        }
                        else {
                            
                            dvdx = vel_gradient[HII_R_FFT_INDEX(i,j,k)];
                            
                            // set maximum allowed gradient for this linear approximation
                            if (fabs(dvdx) > max_v_deriv){
                                if (dvdx < 0) dvdx = -max_v_deriv;
                                else dvdx = max_v_deriv;
                                //                               nonlin_ct++;
                            }
                            
                            box->brightness_temp[HII_R_INDEX(i,j,k)] /= (dvdx/H + 1.0);
                            
                        }
                    }
                }
            }
            
            // normalised units of cell length. 0 equals beginning of cell, 1 equals end of cell
            // These are the sub-cell central positions (x_pos_offset), and the corresponding normalised value (x_pos) between 0 and 1
            for(ii=0;ii<astro_params->N_RSD_STEPS;ii++) {
                x_pos_offset[ii] = subcell_width*(float)ii + subcell_width/2.;
                x_pos[ii] = x_pos_offset[ii]/( user_params->BOX_LEN/(float)user_params->HII_DIM );
            }
            
            // Note to convert the velocity v, to a displacement in redshift space, convert from s -> r + (1+z)*v/H(z)
            // To convert the velocity within the array v to km/s, it is a*dD/dt*delta. Where the scale factor a comes from the continuity equation
            // The array v as defined in 21cmFAST is (ik/k^2)*dD/dt*delta, as it is defined as a comoving quantity (scale factor is implicit).
            // However, the conversion between real and redshift space also picks up a scale factor, therefore the scale factors drop out and therefore
            // the displacement of the sub-cells is purely determined from the array, v and the Hubble factor: v/H.
            
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    
                    // Generate the optical-depth for the specific line-of-sight with R.S.D
                    for(k=0;k<user_params->HII_DIM;k++) {
                        delta_T_RSD_LOS[k] = 0.0;
                    }
                    
                    for (k=0; k<user_params->HII_DIM; k++){
                        
                        if((fabs(box->brightness_temp[HII_R_INDEX(i,j,k)]) >= FRACT_FLOAT_ERR) && (ionized_box->xH_box[HII_R_INDEX(i,j,k)] >= FRACT_FLOAT_ERR)) {
                            
                            if(k==0) {
                                d1_low = v[HII_R_FFT_INDEX(i,j,user_params->HII_DIM-1)]/H;
                                d2_low = v[HII_R_FFT_INDEX(i,j,k)]/H;
                            }
                            else {
                                d1_low = v[HII_R_FFT_INDEX(i,j,k-1)]/H;
                                d2_low = v[HII_R_FFT_INDEX(i,j,k)]/H;
                            }
                            // Displacements (converted from velocity) for the original cell centres straddling half of the sub-cells (cell after)
                            if(k==(user_params->HII_DIM-1)) {
                                d1_high = v[HII_R_FFT_INDEX(i,j,k)]/H;
                                d2_high = v[HII_R_FFT_INDEX(i,j,0)]/H;
                            }
                            else {
                                d1_high = v[HII_R_FFT_INDEX(i,j,k)]/H;
                                d2_high = v[HII_R_FFT_INDEX(i,j,k+1)]/H;
                            }
                            
                            for(ii=0;ii<astro_params->N_RSD_STEPS;ii++) {
                                
                                // linearly interpolate the displacements to determine the corresponding displacements of the sub-cells
                                // Checking of 0.5 is for determining if we are left or right of the mid-point of the original cell (for the linear interpolation of the displacement)
                                // to use the appropriate cell
                                
                                if(x_pos[ii] <= 0.5) {
                                    subcell_displacement = d1_low + ( (x_pos[ii] + 0.5 ) - x_val1)*( d2_low - d1_low )/( x_val2 - x_val1 );
                                }
                                else {
                                    subcell_displacement = d1_high + ( (x_pos[ii] - 0.5 ) - x_val1)*( d2_high - d1_high )/( x_val2 - x_val1 );
                                }
                                
                                // The new centre of the sub-cell post R.S.D displacement. Normalised to units of cell width for determining it's displacement
                                RSD_pos_new = (x_pos_offset[ii] + subcell_displacement)/( user_params->BOX_LEN/((float)user_params->HII_DIM) );
                                // The sub-cell boundaries of the sub-cell, for determining the fractional contribution of the sub-cell to neighbouring cells when
                                // the sub-cell straddles two cell positions
                                RSD_pos_new_boundary_low = RSD_pos_new - (subcell_width/2.)/( user_params->BOX_LEN/((float)user_params->HII_DIM) );
                                RSD_pos_new_boundary_high = RSD_pos_new + (subcell_width/2.)/( user_params->BOX_LEN/((float)user_params->HII_DIM) );
                                
                                if(RSD_pos_new_boundary_low >= 0.0 && RSD_pos_new_boundary_high < 1.0) {
                                    // sub-cell has remained in the original cell (just add it back to the original cell)
                                    
                                    delta_T_RSD_LOS[k] += box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                }
                                else if(RSD_pos_new_boundary_low < 0.0 && RSD_pos_new_boundary_high < 0.0) {
                                    // sub-cell has moved completely into a new cell (toward the observer)
                                    
                                    // determine how far the sub-cell has moved in units of original cell boundary
                                    cell_distance = ceil(fabs(RSD_pos_new_boundary_low))-1.;
                                    
                                    // Determine the location of the sub-cell relative to the original cell binning
                                    if(fabs(RSD_pos_new_boundary_high) > cell_distance) {
                                        // sub-cell is entirely contained within the new cell (just add it to the new cell)
                                        
                                        // check if the new cell position is at the edge of the box. If so, periodic boundary conditions
                                        if(k<((int)cell_distance+1)) {
                                            delta_T_RSD_LOS[k-((int)cell_distance+1) + user_params->HII_DIM] += box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[k-((int)cell_distance+1)] += box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                    }
                                    else {
                                        // sub-cell is partially contained within the cell
                                        
                                        // Determine the fraction of the sub-cell which is in either of the two original cells
                                        fraction_outside = (fabs(RSD_pos_new_boundary_low) - cell_distance)/(subcell_width/( user_params->BOX_LEN/((float)user_params->HII_DIM) ));
                                        fraction_within = 1. - fraction_outside;
                                        
                                        // Check if the first part of the sub-cell is at the box edge
                                        if(k<(((int)cell_distance))) {
                                            delta_T_RSD_LOS[k-((int)cell_distance) + user_params->HII_DIM] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[k-((int)cell_distance)] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        // Check if the second part of the sub-cell is at the box edge
                                        if(k<(((int)cell_distance + 1))) {
                                            delta_T_RSD_LOS[k-((int)cell_distance+1) + user_params->HII_DIM] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[k-((int)cell_distance+1)] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                    }
                                }
                                else if(RSD_pos_new_boundary_low < 0.0 && (RSD_pos_new_boundary_high > 0.0 && RSD_pos_new_boundary_high < 1.0)) {
                                    // sub-cell has moved partially into a new cell (toward the observer)
                                    
                                    // Determine the fraction of the sub-cell which is in either of the two original cells
                                    fraction_within = RSD_pos_new_boundary_high/(subcell_width/( user_params->BOX_LEN/((float)user_params->HII_DIM) ));
                                    fraction_outside = 1. - fraction_within;
                                    
                                    // Check the periodic boundaries conditions and move the fraction of each sub-cell to the appropriate new cell
                                    if(k==0) {
                                        delta_T_RSD_LOS[user_params->HII_DIM-1] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        delta_T_RSD_LOS[k] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                    }
                                    else {
                                        delta_T_RSD_LOS[k-1] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        delta_T_RSD_LOS[k] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                    }
                                }
                                else if((RSD_pos_new_boundary_low >= 0.0 && RSD_pos_new_boundary_low < 1.0) && (RSD_pos_new_boundary_high >= 1.0)) {
                                    // sub-cell has moved partially into a new cell (away from the observer)
                                    
                                    // Determine the fraction of the sub-cell which is in either of the two original cells
                                    fraction_outside = (RSD_pos_new_boundary_high - 1.)/(subcell_width/( user_params->BOX_LEN/((float)user_params->HII_DIM) ));
                                    fraction_within = 1. - fraction_outside;
                                    
                                    // Check the periodic boundaries conditions and move the fraction of each sub-cell to the appropriate new cell
                                    if(k==(user_params->HII_DIM-1)) {
                                        delta_T_RSD_LOS[k] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        delta_T_RSD_LOS[0] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                    }
                                    else {
                                        delta_T_RSD_LOS[k] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        delta_T_RSD_LOS[k+1] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                    }
                                }
                                else {
                                    // sub-cell has moved completely into a new cell (away from the observer)
                                    
                                    // determine how far the sub-cell has moved in units of original cell boundary
                                    cell_distance = floor(fabs(RSD_pos_new_boundary_high));
                                    
                                    if(RSD_pos_new_boundary_low >= cell_distance) {
                                        // sub-cell is entirely contained within the new cell (just add it to the new cell)
                                        
                                        // check if the new cell position is at the edge of the box. If so, periodic boundary conditions
                                        if(k>(user_params->HII_DIM - 1 - (int)cell_distance)) {
                                            delta_T_RSD_LOS[k+(int)cell_distance - user_params->HII_DIM] += box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[k+(int)cell_distance] += box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                    }
                                    else {
                                        // sub-cell is partially contained within the cell
                                        
                                        // Determine the fraction of the sub-cell which is in either of the two original cells
                                        fraction_outside = (RSD_pos_new_boundary_high - cell_distance)/(subcell_width/( user_params->BOX_LEN/((float)user_params->HII_DIM) ));
                                        fraction_within = 1. - fraction_outside;
                                        
                                        // Check if the first part of the sub-cell is at the box edge
                                        if(k>(user_params->HII_DIM - 1 - ((int)cell_distance-1))) {
                                            delta_T_RSD_LOS[k+(int)cell_distance-1 - user_params->HII_DIM] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[k+(int)cell_distance-1] += fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        // Check if the second part of the sub-cell is at the box edge
                                        if(k>(user_params->HII_DIM - 1 - ((int)cell_distance))) {
                                            delta_T_RSD_LOS[k+(int)cell_distance - user_params->HII_DIM] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[k+(int)cell_distance] += fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    for(k=0;k<user_params->HII_DIM;k++) {
                        box->brightness_temp[HII_R_INDEX(i,j,k)] = delta_T_RSD_LOS[k];
                            
                        ave += delta_T_RSD_LOS[k];
                    }
                }
            }
            
            ave /= (float)HII_TOT_NUM_PIXELS;
        }
        else {
            
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        
                        dvdx = vel_gradient[HII_R_FFT_INDEX(i,j,k)];
                        
                        // set maximum allowed gradient for this linear approximation
                        if (fabs(dvdx) > max_v_deriv){
                            if (dvdx < 0) dvdx = -max_v_deriv;
                            else dvdx = max_v_deriv;
                            //                               nonlin_ct++;
                        }
                        
                        box->brightness_temp[HII_R_INDEX(i,j,k)] /= (dvdx/H + 1.0);
                        
                        ave += box->brightness_temp[HII_R_INDEX(i,j,k)];
                    }
                }
            }
            ave /= (HII_TOT_NUM_PIXELS+0.0);
        }
    }
    
    if(isfinite(ave)==0) {
        LOG_ERROR("Average brightness temperature (after including velocities) is infinite or NaN!");
        return(2);
    }


LOG_DEBUG("ave Tb = %e", ave);

    free(v);
    free(vel_gradient);
    
    free(x_pos);
    free(x_pos_offset);
    free(delta_T_RSD_LOS);

    return(0);
}

