double apply_subcell_rsds(
    struct UserParams *user_params,
    struct CosmoParams *cosmo_params,
    struct FlagOptions *flag_options,
    struct AstroParams *astro_params,
    struct IonizedBox *ionized_box,
    struct BrightnessTemp *box,
    float redshift,
    struct TsBox *spin_temp,
    float T_rad,
    float *v,
    float H
) {

    char wisdom_filename[500];
    int i, ii, j, k, n_x, n_y, n_z;
    float k_x, k_y, k_z;
    double ave;

    ave = 0.;

    omp_set_num_threads(user_params->N_THREADS);

    float *x_pos = calloc(astro_params->N_RSD_STEPS,sizeof(float));
    float *x_pos_offset = calloc(astro_params->N_RSD_STEPS,sizeof(float));
    float **delta_T_RSD_LOS = (float **)calloc(user_params->N_THREADS,sizeof(float *));
    for(i=0;i<user_params->N_THREADS;i++) {
        delta_T_RSD_LOS[i] = (float *)calloc(user_params->HII_DIM,sizeof(float));
    }

    float gradient_component, min_gradient_component;
    float d1_low, d1_high, d2_low, d2_high;
    float x_val1, x_val2, subcell_displacement;
    float RSD_pos_new, RSD_pos_new_boundary_low,RSD_pos_new_boundary_high;
    float fraction_within, fraction_outside, cell_distance;

    float cellsize = user_params->BOX_LEN/(float)user_params->HII_DIM;
    float subcell_width = cellsize/(float)(astro_params->N_RSD_STEPS);

    // normalised units of cell length. 0 equals beginning of cell, 1 equals end of cell
    // These are the sub-cell central positions (x_pos_offset), and the corresponding normalised value (x_pos) between 0 and 1
    for(ii=0;ii<astro_params->N_RSD_STEPS;ii++) {
        x_pos_offset[ii] = subcell_width*(float)ii + subcell_width/2.;
        x_pos[ii] = x_pos_offset[ii]/cellsize;
    }

    x_val1 = 0.0;
    x_val2 = 1.0;

    /*
    Note to convert the velocity v, to a displacement in redshift space, convert
    from s -> r + (1+z)*v/H(z). To convert the velocity within the array v to km/s, it
    is a*dD/dt*delta. Where the scale factor a comes from the continuity equation
    The array v as defined in 21cmFAST is (ik/k^2)*dD/dt*delta, as it is defined as a
    comoving quantity (scale factor is implicit). However, the conversion between real
    and redshift space also picks up a scale factor, therefore the scale factors drop
    out and therefore the displacement of the sub-cells is purely determined from the
    array, v and the Hubble factor: v/H.
    */
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
    return(ave);
}
