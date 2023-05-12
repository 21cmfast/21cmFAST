
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

int ComputeBrightnessTemp(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                           struct AstroParams *astro_params, struct FlagOptions *flag_options,
                           struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box) {

    int status;
    Try{ // Try block around whole function.
    LOG_DEBUG("Starting Brightness Temperature calculation for redshift %f", redshift);
    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    char wisdom_filename[500];
    int i, ii, j, k, n_x, n_y, n_z;
    float k_x, k_y, k_z;
    double ave;

    ave = 0.;

    omp_set_num_threads(user_params->N_THREADS);

    float *v = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
    float *vel_gradient = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));

    float *x_pos = calloc(astro_params->N_RSD_STEPS,sizeof(float));
    float *x_pos_offset = calloc(astro_params->N_RSD_STEPS,sizeof(float));
    float **delta_T_RSD_LOS = (float **)calloc(user_params->N_THREADS,sizeof(float *));
    for(i=0;i<user_params->N_THREADS;i++) {
        delta_T_RSD_LOS[i] = (float *)calloc(HII_D_PARA,sizeof(float));
    }


#pragma omp parallel shared(v,perturb_field) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<HII_D_PARA; k++){
                    *((float *)v + HII_R_FFT_INDEX(i,j,k)) = perturb_field->velocity[HII_R_INDEX(i,j,k)];
                }
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
    LOG_DEBUG("Performed Initialization.");

    // ok, lets fill the delta_T box; which will be the same size as the bubble box
#pragma omp parallel shared(const_factor,perturb_field,ionized_box,box,redshift,spin_temp,T_rad) \
            private(i,j,k,pixel_deltax,pixel_x_HI,pixel_Ts_factor) num_threads(user_params->N_THREADS)
    {
#pragma omp for reduction(+:ave)
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<HII_D_PARA; k++){

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
    }

    LOG_DEBUG("Filled delta_T.");

    if(isfinite(ave)==0) {
        LOG_ERROR("Average brightness temperature is infinite or NaN!");
//        Throw(ParameterError);
        Throw(InfinityorNaNError);
    }

    ave /= (float)HII_TOT_NUM_PIXELS;

    x_val1 = 0.;
    x_val2 = 1.;

    subcell_width = (user_params->BOX_LEN/(float)user_params->HII_DIM)/(float)(astro_params->N_RSD_STEPS);

    // now write out the delta_T box
    if (global_params.T_USE_VELOCITIES){
        ave = 0.;

        memcpy(vel_gradient, v, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, vel_gradient);

#pragma omp parallel shared(vel_gradient) private(n_x,n_y,n_z,k_x,k_y,k_z) num_threads(user_params->N_THREADS)
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

                    for (n_z=0; n_z<=HII_MIDDLE_PARA; n_z++){
                        k_z = n_z * DELTA_K_PARA;

                        // take partial deriavative along the line of sight
                        *((fftwf_complex *) vel_gradient + HII_C_INDEX(n_x,n_y,n_z)) *= k_z*I/(float)HII_TOT_NUM_PIXELS;
                    }
                }
            }
        }

        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, vel_gradient);

        // now add the velocity correction to the delta_T maps (only used for T_S >> T_CMB case).
        max_v_deriv = fabs(global_params.MAX_DVDR*H);

        if(flag_options->SUBCELL_RSD) {

            // now add the velocity correction to the delta_T maps
            min_gradient_component = 1.0;

#pragma omp parallel shared(vel_gradient,T_rad,redshift,spin_temp,box,max_v_deriv) \
                    private(i,j,k,gradient_component,dvdx) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<HII_D_PARA; k++){

                            gradient_component = fabs(vel_gradient[HII_R_FFT_INDEX(i,j,k)]/H + 1.0);

                            if(flag_options->USE_TS_FLUCT) {

                                // Calculate the brightness temperature, using the optical depth
                                if(gradient_component < FRACT_FLOAT_ERR) {
                                    // Gradient component goes to zero, optical depth diverges.
                                    // But, since we take exp(-tau), this goes to zero and (1 - exp(-tau)) goes to unity.
                                    // Again, factors of 1000. are conversions from K to mK
                                    box->brightness_temp[HII_R_INDEX(i,j,k)] = 1000.*(spin_temp->Ts_box[HII_R_INDEX(i,j,k)] - T_rad)/(1. + redshift);
                                }
                                else {
                                    box->brightness_temp[HII_R_INDEX(i,j,k)] = (1. - exp(- box->brightness_temp[HII_R_INDEX(i,j,k)]/gradient_component ))*\
                                                                                1000.*(spin_temp->Ts_box[HII_R_INDEX(i,j,k)] - T_rad)/(1. + redshift);
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
#pragma omp parallel shared(delta_T_RSD_LOS,box,ionized_box,v,x_val1,x_val2,x_pos,x_pos_offset,subcell_width) \
                    private(i,j,k,ii,d1_low,d2_low,d1_high,d2_high,subcell_displacement,RSD_pos_new,\
                            RSD_pos_new_boundary_low,RSD_pos_new_boundary_high,cell_distance,fraction_outside,fraction_within) \
                    num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:ave)
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){

                        // Generate the optical-depth for the specific line-of-sight with R.S.D
                        for(k=0;k<HII_D_PARA;k++) {
                            delta_T_RSD_LOS[omp_get_thread_num()][k] = 0.0;
                        }

                        for (k=0; k<HII_D_PARA; k++){

                            if((fabs(box->brightness_temp[HII_R_INDEX(i,j,k)]) >= FRACT_FLOAT_ERR) && \
                                        (ionized_box->xH_box[HII_R_INDEX(i,j,k)] >= FRACT_FLOAT_ERR)) {

                                if(k==0) {
                                    d1_low = v[HII_R_FFT_INDEX(i,j,HII_D_PARA-1)]/H;
                                    d2_low = v[HII_R_FFT_INDEX(i,j,k)]/H;
                                }
                                else {
                                    d1_low = v[HII_R_FFT_INDEX(i,j,k-1)]/H;
                                    d2_low = v[HII_R_FFT_INDEX(i,j,k)]/H;
                                }

                                // Displacements (converted from velocity) for the original cell centres straddling half of the sub-cells (cell after)
                                if(k==(HII_D_PARA-1)) {
                                    d1_high = v[HII_R_FFT_INDEX(i,j,k)]/H;
                                    d2_high = v[HII_R_FFT_INDEX(i,j,0)]/H;
                                }
                                else {
                                    d1_high = v[HII_R_FFT_INDEX(i,j,k)]/H;
                                    d2_high = v[HII_R_FFT_INDEX(i,j,k+1)]/H;
                                }

                                for(ii=0;ii<astro_params->N_RSD_STEPS;ii++) {

                                    // linearly interpolate the displacements to determine the corresponding displacements of the sub-cells
                                    // Checking of 0.5 is for determining if we are left or right of the mid-point
                                    // of the original cell (for the linear interpolation of the displacement)
                                    // to use the appropriate cell

                                    if(x_pos[ii] <= 0.5) {
                                        subcell_displacement = d1_low + ( (x_pos[ii] + 0.5 ) - x_val1)*( d2_low - d1_low )/( x_val2 - x_val1 );
                                    }
                                    else {
                                        subcell_displacement = d1_high + ( (x_pos[ii] - 0.5 ) - x_val1)*( d2_high - d1_high )/( x_val2 - x_val1 );
                                    }

                                    // The new centre of the sub-cell post R.S.D displacement.
                                    // Normalised to units of cell width for determining it's displacement
                                    RSD_pos_new = (x_pos_offset[ii] + subcell_displacement)/( user_params->BOX_LEN/((float)user_params->HII_DIM) );
                                    // The sub-cell boundaries of the sub-cell, for determining the fractional
                                    // contribution of the sub-cell to neighbouring cells when
                                    // the sub-cell straddles two cell positions
                                    RSD_pos_new_boundary_low = RSD_pos_new - (subcell_width/2.)/( user_params->BOX_LEN/((float)user_params->HII_DIM) );
                                    RSD_pos_new_boundary_high = RSD_pos_new + (subcell_width/2.)/( user_params->BOX_LEN/((float)user_params->HII_DIM) );

                                    if(RSD_pos_new_boundary_low >= 0.0 && RSD_pos_new_boundary_high < 1.0) {
                                        // sub-cell has remained in the original cell (just add it back to the original cell)

                                        delta_T_RSD_LOS[omp_get_thread_num()][k] += box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
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
                                                delta_T_RSD_LOS[omp_get_thread_num()][k-((int)cell_distance+1) + HII_D_PARA] += \
                                                                box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            else {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k-((int)cell_distance+1)] += \
                                                                box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                        }
                                        else {
                                            // sub-cell is partially contained within the cell

                                            // Determine the fraction of the sub-cell which is in either of the two original cells
                                            fraction_outside = (fabs(RSD_pos_new_boundary_low) - cell_distance)\
                                                        /(subcell_width/( user_params->BOX_LEN/((float)user_params->HII_DIM) ));
                                            fraction_within = 1. - fraction_outside;

                                            // Check if the first part of the sub-cell is at the box edge
                                            if(k<(((int)cell_distance))) {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k-((int)cell_distance) + HII_D_PARA] += \
                                                        fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            else {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k-((int)cell_distance)] += \
                                                        fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            // Check if the second part of the sub-cell is at the box edge
                                            if(k<(((int)cell_distance + 1))) {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k-((int)cell_distance+1) + HII_D_PARA] += \
                                                        fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            else {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k-((int)cell_distance+1)] += \
                                                        fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
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
                                            delta_T_RSD_LOS[omp_get_thread_num()][HII_D_PARA-1] += \
                                                    fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            delta_T_RSD_LOS[omp_get_thread_num()][k] += \
                                                    fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[omp_get_thread_num()][k-1] += \
                                                    fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            delta_T_RSD_LOS[omp_get_thread_num()][k] += \
                                                    fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                    }
                                    else if((RSD_pos_new_boundary_low >= 0.0 && RSD_pos_new_boundary_low < 1.0) && (RSD_pos_new_boundary_high >= 1.0)) {
                                        // sub-cell has moved partially into a new cell (away from the observer)

                                        // Determine the fraction of the sub-cell which is in either of the two original cells
                                        fraction_outside = (RSD_pos_new_boundary_high - 1.)/(subcell_width/( user_params->BOX_LEN/((float)user_params->HII_DIM) ));
                                        fraction_within = 1. - fraction_outside;

                                        // Check the periodic boundaries conditions and move the fraction of each sub-cell to the appropriate new cell
                                        if(k==(HII_D_PARA-1)) {
                                            delta_T_RSD_LOS[omp_get_thread_num()][k] += \
                                                    fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            delta_T_RSD_LOS[omp_get_thread_num()][0] += \
                                                    fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                        else {
                                            delta_T_RSD_LOS[omp_get_thread_num()][k] += \
                                                    fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            delta_T_RSD_LOS[omp_get_thread_num()][k+1] += \
                                                    fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                        }
                                    }
                                    else {
                                        // sub-cell has moved completely into a new cell (away from the observer)

                                        // determine how far the sub-cell has moved in units of original cell boundary
                                        cell_distance = floor(fabs(RSD_pos_new_boundary_high));

                                        if(RSD_pos_new_boundary_low >= cell_distance) {
                                            // sub-cell is entirely contained within the new cell (just add it to the new cell)

                                            // check if the new cell position is at the edge of the box. If so, periodic boundary conditions
                                            if(k>(HII_D_PARA - 1 - (int)cell_distance)) {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k+(int)cell_distance - HII_D_PARA] += \
                                                        box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            else {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k+(int)cell_distance] += \
                                                        box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                        }
                                        else {
                                            // sub-cell is partially contained within the cell

                                            // Determine the fraction of the sub-cell which is in either of the two original cells
                                            fraction_outside = (RSD_pos_new_boundary_high - cell_distance)/(subcell_width/( user_params->BOX_LEN/((float)user_params->HII_DIM) ));
                                            fraction_within = 1. - fraction_outside;

                                            // Check if the first part of the sub-cell is at the box edge
                                            if(k>(HII_D_PARA - 1 - ((int)cell_distance-1))) {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k+(int)cell_distance-1 - HII_D_PARA] += \
                                                        fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            else {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k+(int)cell_distance-1] += \
                                                        fraction_within*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            // Check if the second part of the sub-cell is at the box edge
                                            if(k>(HII_D_PARA - 1 - ((int)cell_distance))) {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k+(int)cell_distance - HII_D_PARA] += \
                                                        fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                            else {
                                                delta_T_RSD_LOS[omp_get_thread_num()][k+(int)cell_distance] += \
                                                        fraction_outside*box->brightness_temp[HII_R_INDEX(i,j,k)]/((float)astro_params->N_RSD_STEPS);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for(k=0;k<HII_D_PARA;k++) {
                            box->brightness_temp[HII_R_INDEX(i,j,k)] = delta_T_RSD_LOS[omp_get_thread_num()][k];

                            ave += delta_T_RSD_LOS[omp_get_thread_num()][k];
                        }
                    }
                }
            }

            ave /= (float)HII_TOT_NUM_PIXELS;
        }
        else {
#pragma omp parallel shared(vel_gradient,box) private(i,j,k,dvdx) num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:ave)
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<HII_D_PARA; k++){

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
            }
            ave /= (HII_TOT_NUM_PIXELS+0.0);
        }
    }

    LOG_DEBUG("Included velocities.");


    if(isfinite(ave)==0) {
        LOG_ERROR("Average brightness temperature (after including velocities) is infinite or NaN!");
//        Throw(ParameterError);
        Throw(InfinityorNaNError);
    }

    LOG_DEBUG("z = %.2f, ave Tb = %e", redshift, ave);

    free(v);
    free(vel_gradient);

    free(x_pos);
    free(x_pos_offset);
    for(i=0;i<user_params->N_THREADS;i++) {
        free(delta_T_RSD_LOS[i]);
    }
    free(delta_T_RSD_LOS);
    fftwf_cleanup_threads();
    fftwf_cleanup();
    LOG_DEBUG("Cleaned up.");

    } // End of try
    Catch(status){
        return(status);
    }

    return(0);
}
