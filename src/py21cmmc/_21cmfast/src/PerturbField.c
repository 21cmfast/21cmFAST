
// Re-write of perturb_field.c for being accessible within the MCMC

void ComputePerturbField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes, struct PerturbedField *p_cubes) {
    
    /*
     USAGE: perturb_field <REDSHIFT>
     
     PROGRAM PERTURB_FIELD uses the first-order Langragian displacement field to move the masses in the cells of the density field.
     The high-res density field is extrapolated to some high-redshift (INITIAL_REDSHIFT in ANAL_PARAMS.H), then uses the zeldovich approximation
     to move the grid "particles" onto the lower-res grid we use for the maps.  Then we recalculate the velocity fields on the perturbed grid.
     */
    
    printf("high-res density; %e %e %e %e\n",boxes->hires_density[0],boxes->hires_density[100],boxes->hires_density[1000],boxes->hires_density[10000]);
    printf("low-res density; %e %e %e %e\n",boxes->lowres_density[0],boxes->lowres_density[100],boxes->lowres_density[1000],boxes->lowres_density[10000]);
    printf("low-res density (vx); %e %e %e %e\n",boxes->lowres_vx[0],boxes->lowres_vx[100],boxes->lowres_vx[1000],boxes->lowres_vx[10000]);
    printf("low-res density (vy); %e %e %e %e\n",boxes->lowres_vy[0],boxes->lowres_vy[100],boxes->lowres_vy[1000],boxes->lowres_vy[10000]);
    printf("low-res density (vz); %e %e %e %e\n",boxes->lowres_vz[0],boxes->lowres_vz[100],boxes->lowres_vz[1000],boxes->lowres_vz[10000]);
    printf("low-res density (vx 2LPT); %e %e %e %e\n",boxes->lowres_vx_2LPT[0],boxes->lowres_vx_2LPT[100],boxes->lowres_vx_2LPT[1000],boxes->lowres_vx_2LPT[10000]);
    printf("low-res density (vy 2LPT); %e %e %e %e\n",boxes->lowres_vy_2LPT[0],boxes->lowres_vy_2LPT[100],boxes->lowres_vy_2LPT[1000],boxes->lowres_vy_2LPT[10000]);
    printf("low-res density (vz 2LPT); %e %e %e %e\n",boxes->lowres_vz_2LPT[0],boxes->lowres_vz_2LPT[100],boxes->lowres_vz_2LPT[1000],boxes->lowres_vz_2LPT[10000]);
    
//    struct UserParams temp;
    
//    temp = boxes->user_params;
    
//    printf("value = %d\n",temp.DIM);
//    printf("value = %d\n",user_params->DIM);
    
//    printf("value = %d\n",boxes->user_params.DIM);
//    printf("value = %d\n",boxes->user_params.HII_DIM);
//    printf("value = %e\n",boxes->user_params.BOX_LEN);
    
//    printf("value = %e\n",boxes->cosmo_params.hlittle);
    
    fftwf_complex *LOWRES_density_perturb, *LOWRES_density_perturb_saved;
    fftwf_plan plan;
    
    float growth_factor, displacement_factor_2LPT, init_growth_factor, init_displacement_factor_2LPT, xf, yf, zf;
    float mass_factor, dDdt, f_pixel_factor, velocity_displacement_factor, velocity_displacement_factor_2LPT;
    unsigned long long ct, HII_i, HII_j, HII_k;
    int i,j,k, xi, yi, zi;
    double ave_delta, new_ave_delta;
    // ***************   BEGIN INITIALIZATION   ************************** //
    
    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        fprintf(stderr, "perturb_field.c: WARNING: Resolution is likely too low for accurate evolved density fields\n It Is recommended that you either increase the resolution (DIM/Box_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }
    
    growth_factor = dicke(redshift);
    displacement_factor_2LPT = -(3.0/7.0) * growth_factor*growth_factor; // 2LPT eq. D8
    
    dDdt = ddickedt(redshift); // time derivative of the growth factor (1/s)
    init_growth_factor = dicke(global_params.INITIAL_REDSHIFT);
    init_displacement_factor_2LPT = -(3.0/7.0) * init_growth_factor*init_growth_factor; // 2LPT eq. D8
    printf("gf = %e dDdt = %e displacement_factor_2LPT = %e init_growth_factor = %e init_displacement_factor_2LPT = %e\n",growth_factor,dDdt,displacement_factor_2LPT,init_growth_factor,init_displacement_factor_2LPT);
    
    
    // allocate memory for the updated density, and initialize
    LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    
    // check if the linear evolution flag was set
    if (global_params.EVOLVE_DENSITY_LINEARLY){
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = growth_factor*boxes->lowres_density[HII_R_INDEX(i,j,k)];
                }
            }
        }
    }
    // first order Zel'Dovich perturbation
    else{
        
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                }
            }
        }
        
        velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;
        
        // now add the missing factor of D
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
            boxes->lowres_vx[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
            boxes->lowres_vy[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
            boxes->lowres_vz[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
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
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                boxes->lowres_vx_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                boxes->lowres_vy_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                boxes->lowres_vz_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
            }
        }
        
        printf("low-res perturbed vx; %e %e %e %e\n",boxes->lowres_vx[0],boxes->lowres_vx[100],boxes->lowres_vx[1000],boxes->lowres_vx[10000]);
        printf("low-res perturbed vy; %e %e %e %e\n",boxes->lowres_vy[0],boxes->lowres_vy[100],boxes->lowres_vy[1000],boxes->lowres_vy[10000]);
        printf("low-res perturbed vz; %e %e %e %e\n",boxes->lowres_vz[0],boxes->lowres_vz[100],boxes->lowres_vz[1000],boxes->lowres_vz[10000]);
        printf("low-res perturbed vx (2LPT); %e %e %e %e\n",boxes->lowres_vx_2LPT[0],boxes->lowres_vx_2LPT[100],boxes->lowres_vx_2LPT[1000],boxes->lowres_vx_2LPT[10000]);
        printf("low-res perturbed vy (2LPT); %e %e %e %e\n",boxes->lowres_vy_2LPT[0],boxes->lowres_vy_2LPT[100],boxes->lowres_vy_2LPT[1000],boxes->lowres_vy_2LPT[10000]);
        printf("low-res perturbed vz (2LPT); %e %e %e %e\n",boxes->lowres_vz_2LPT[0],boxes->lowres_vz_2LPT[100],boxes->lowres_vz_2LPT[1000],boxes->lowres_vz_2LPT[10000]);
        
        // * ************************************************************************* * //
        // *                            END 2LPT PART                                  * //
        // * ************************************************************************* * //
        
        // ************  END INITIALIZATION **************************** //
        
        // go through the high-res box, mapping the mass onto the low-res (updated) box
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
                    
                    // now move the mass
//                    if(HII_R_FFT_INDEX(xi, yi, zi)==0) {
//                        printf("i = %d j = %d k = %d xi = %d yi = %d zi = %d xf = %e yf = %e zf = %e dens = %e val = %e sum = %e\n",i,j,k,xi,yi,zi,xf,yf,zf,(boxes->hires_density)[R_FFT_INDEX(i,j,k)],(1 + init_growth_factor*(boxes->hires_density)[R_FFT_INDEX(i,j,k)]),LOWRES_density_perturb[HII_R_FFT_INDEX(xi, yi, zi)]);
//                    }

                    *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(xi, yi, zi) ) +=
                    (1 + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)]);
                }
            }
        }
        
        printf("low-res density perturbed; %e %e %e %e\n",LOWRES_density_perturb[0],LOWRES_density_perturb[100],LOWRES_density_perturb[1000],LOWRES_density_perturb[10000]);
        
        // renormalize to the new pixel size, and make into delta
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1;
                }
            }
        }
        
        printf("low-res density perturbed (normalised); %e %e %e %e\n",LOWRES_density_perturb[0],LOWRES_density_perturb[100],LOWRES_density_perturb[1000],LOWRES_density_perturb[10000]);
        
        // deallocate
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
            boxes->lowres_vx[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
            boxes->lowres_vy[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
            boxes->lowres_vz[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
        }
        
        if(global_params.SECOND_ORDER_LPT_CORRECTIONS){
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                boxes->lowres_vx_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                boxes->lowres_vy_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                boxes->lowres_vz_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
            }
        }
    }
    
    // ****  Print and convert to velocities ***** //
    if (global_params.EVOLVE_DENSITY_LINEARLY){
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    *((float *)p_cubes->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                }
            }
        }
        
        // transform to k-space
        plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        fftwf_cleanup();
        
        // save a copy of the k-space density field
    }
    else{
        // transform to k-space
        plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        fftwf_cleanup();
        
        //smooth the field
        if (!global_params.EVOLVE_DENSITY_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
            filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
        }
        
        // save a copy of the k-space density field
        memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        
        plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        fftwf_cleanup();
        
        // normalize after FFT
        for(i=0; i<user_params->HII_DIM; i++){
            for(j=0; j<user_params->HII_DIM; j++){
                for(k=0; k<user_params->HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;
                    if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                }
            }
        }
        
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    *((float *)p_cubes->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                }
            }
        }
        memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }
    
    float k_x, k_y, k_z, k_sq, dDdt_over_D;
    int n_x, n_y, n_z;
    
    dDdt_over_D = dDdt/growth_factor;
    
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
    plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    fftwf_cleanup();
    
    for (i=0; i<user_params->HII_DIM; i++){
        for (j=0; j<user_params->HII_DIM; j++){
            for (k=0; k<user_params->HII_DIM; k++){
                *((float *)p_cubes->velocity + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
            }
        }
    }
    
    printf("low-res perturbed density; %e %e %e %e\n",p_cubes->density[0],p_cubes->density[100],p_cubes->density[1000],p_cubes->density[10000]);
    printf("low-res perturbed velocity (vz); %e %e %e %e\n",p_cubes->velocity[0],p_cubes->velocity[100],p_cubes->velocity[1000],p_cubes->velocity[10000]);
    
    // deallocate
    fftwf_free(LOWRES_density_perturb);
    fftwf_free(LOWRES_density_perturb_saved);

}
