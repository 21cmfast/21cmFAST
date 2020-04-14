
// Re-write of find_halos.c from the original 21cmFAST

int overlap_halo(char * in_halo, struct UserParams *user_params, float R, int x, int y, int z);
void update_in_halo(char * in_halo, struct UserParams *user_params, float R, int x, int y, int z);


int ComputeHaloField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct InitialConditions *boxes, struct HaloField *halos) {

LOG_DEBUG("input value:");
LOG_DEBUG("redshift=%f", redshift);
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
    fftwf_init_threads();
    fftwf_plan_with_nthreads(user_params->N_THREADS);
    fftwf_cleanup_threads();

    char wisdom_filename[500];
    fftwf_plan plan;
    fftwf_complex *density_field, *density_field_saved;

    float growth_factor, R, delta_m, dm, dlnm, M, Delta_R, delta_crit;
    double fgrtm, dfgrtm;
    unsigned long long ct;
    char filename[80], *in_halo, *in_halo_prevR, *forbidden;
    int i,j,k,x,y,z,dn,n;
    float R_temp, x_temp, y_temp, z_temp, dummy, M_MIN;

LOG_DEBUG("Begin Initialisation");

    // ***************** END INITIALIZATION ***************** //
    init_ps();

    growth_factor = dicke(redshift); // normalized to 1 at z=0
    delta_crit = Deltac; // for now set to spherical; check if we want elipsoidal later

    //set the minimum source mass
    M_MIN = astro_params->M_TURN/3.;

    // allocate array for the k-space box
    density_field = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    density_field_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    // allocate memory for the boolean in_halo box
    in_halo = (char *) malloc(sizeof(char)*TOT_NUM_PIXELS);
    in_halo_prevR = (char *) malloc(sizeof(char)*TOT_NUM_PIXELS);

    // initialize
    memset(in_halo, 0, sizeof(char)*TOT_NUM_PIXELS);
    memset(in_halo_prevR, 0, sizeof(char)*TOT_NUM_PIXELS);

    if(global_params.OPTIMIZE) {
        forbidden = (char *) malloc(sizeof(char)*TOT_NUM_PIXELS);
    }

#pragma omp parallel shared(boxes,density_field) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->DIM; i++){
            for (j=0; j<user_params->DIM; j++){
                for (k=0; k<user_params->DIM; k++){
                    *((float *)density_field + R_FFT_INDEX(i,j,k)) = *((float *)boxes->hires_density + R_INDEX(i,j,k));
                }
            }
        }
    }

    // Now need to convert the real space density to Fourier space
    if(user_params->USE_FFTW_WISDOM) {
        // Check to see if the wisdom exists, create it if it doesn't
        sprintf(wisdom_filename,"real_to_complex_DIM%d_NTHREADS%d.fftwf_wisdom",user_params->DIM,user_params->N_THREADS);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
            plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (float *)density_field, (fftwf_complex *)density_field, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);

        }
        else {

            plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (float *)density_field, (fftwf_complex *)density_field, FFTW_PATIENT);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);

            // Store the wisdom for later use
            fftwf_export_wisdom_to_filename(wisdom_filename);

            // copy back over the density cube
#pragma omp parallel shared(boxes,density_field) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->DIM; i++){
                    for (j=0; j<user_params->DIM; j++){
                        for (k=0; k<user_params->DIM; k++){
                            *((float *)density_field + R_FFT_INDEX(i,j,k)) = *((float *)boxes->hires_density + R_INDEX(i,j,k));
                        }
                    }
                }
            }

            plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (float *)density_field, (fftwf_complex *)density_field, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
    }
    else {
        plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                     (float *)density_field, (fftwf_complex *)density_field, FFTW_ESTIMATE);
        fftwf_execute(plan);
    }
    fftwf_destroy_plan(plan);

    // save a copy of the k-space density field
    memcpy(density_field_saved, density_field, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);


    // ***************** END INITIALIZATION ***************** //

LOG_DEBUG("Finalised Initialisation");

    // lets filter it now
    // set initial R value
    R = MtoR(M_MIN*1.01); // one percent higher for rounding

LOG_DEBUG("Prepare to filter to find halos");

    while (R < L_FACTOR*user_params->BOX_LEN)
        R*=global_params.DELTA_R_FACTOR;

    fgrtm=dfgrtm=0;
    n=0;
    Delta_R = L_FACTOR*2.*user_params->BOX_LEN/(user_params->DIM+0.0);

    while ((R > 0.5*Delta_R) && (RtoM(R) >= M_MIN)){ // filter until we get to half the pixel size or M_MIN

LOG_ULTRA_DEBUG("while loop for finding halos: R = %f 0.5*Delta_R = %f RtoM(R)=%f M_MIN=%f", R, 0.5*Delta_R, RtoM(R), M_MIN);

        M = RtoM(R);
        if(global_params.DELTA_CRIT_MODE == 1 && (user_params->HMF>0 && user_params->HMF<4)){
            if(user_params->HMF==1) {
                // use sheth tormen correction
                delta_crit = growth_factor*sheth_delc(Deltac/growth_factor, sigma_z0(M));
            }
            if(user_params->HMF==2) {
                // correction for Watson FoF.
                delta_crit = Deltac;
            }
            if(user_params->HMF==2) {
                // correction for Watson FoF-z.
                delta_crit = Deltac;
            }
        }
        printf("R = %e M = %e delta_crit = %e\n",R,M,delta_crit);

        // first let's check if virialized halos of this size are rare enough
        // that we don't have to worry about them (let's define 7 sigma away, as in Mesinger et al 05)
        if ((sigma_z0(M)*growth_factor*7.) < delta_crit){
LOG_DEBUG("Haloes too rare for M = %e! Skipping...");
            R /= global_params.DELTA_R_FACTOR;
            continue;
        }

        memcpy(density_field, density_field_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // now filter the box on scale R
        // 0 = top hat in real space, 1 = top hat in k space
        filter_box(density_field, 0, global_params.HALO_FILTER, R);

        // do the FFT to get delta_m box
        if(user_params->USE_FFTW_WISDOM) {
            // Check to see if the wisdom exists, create it if it doesn't
            sprintf(wisdom_filename,"complex_to_real_DIM%d_NTRHEADS%d.fftwf_wisdom",user_params->DIM,user_params->N_THREADS);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                             (fftwf_complex *)density_field, (float *)density_field, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                             (fftwf_complex *)density_field, (float *)density_field, FFTW_PATIENT);
                fftwf_execute(plan);

                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);

                // create the same filtered density field
                memcpy(density_field, density_field_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

                filter_box(density_field, 0, global_params.HALO_FILTER, R);

                fftwf_destroy_plan(plan);
                plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                             (fftwf_complex *)density_field, (float *)density_field, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (fftwf_complex *)density_field, (float *)density_field, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        fftwf_destroy_plan(plan);

        // *****************  BEGIN OPTIMIZATION ***************** //
        // to optimize speed, if the filter size is large (switch to collapse fraction criteria later)
        if(global_params.OPTIMIZE) {
            if(M > global_params.OPTIMIZE_MIN_MASS) {
                memset(forbidden, 0, sizeof(char)*TOT_NUM_PIXELS);
                // now go through the list of existing halos and paint on the no-go region onto <forbidden>

                // Leaving this blank for now!

            }
        }
        // *****************  BEGIN OPTIMIZATION ***************** //

        printf("%e %e %e\n",*((float *)density_field + R_FFT_INDEX(0,0,0)) * growth_factor / VOLUME,*((float *)density_field + R_FFT_INDEX(12,24,36)) * growth_factor / VOLUME,*((float *)density_field + R_FFT_INDEX(30,20,10)) * growth_factor / VOLUME);

        // now lets scroll through the box, flagging all pixels with delta_m > delta_crit
        dn=0;
        for (x=0; x<user_params->DIM; x++){
            for (y=0; y<user_params->DIM; y++){
                for (z=0; z<user_params->DIM; z++){
                    delta_m = *((float *)density_field + R_FFT_INDEX(x,y,z)) * growth_factor / VOLUME;       // don't forget the factor of 1/VOLUME!
                    // if not within a larger halo, and radii don't overlap print out stats, and update in_halo box
                    // *****************  BEGIN OPTIMIZATION ***************** //
                    if(global_params.OPTIMIZE) {
                        if(M > global_params.OPTIMIZE_MIN_MASS) {
                            if ( (delta_m > delta_crit) && !forbidden[R_INDEX(x,y,z)]){
//                                fprintf(stderr, "Found halo #%i, delta_m = %.3f at (x,y,z) = (%i,%i,%i)\n", n+1, delta_m, x,y,z);
//                                fprintf(OUT, "%e\t%f\t%f\t%f\n", M, x/(DIM+0.0), y/(DIM+0.0), z/(DIM+0.0));
//                                fflush(NULL);
                                update_in_halo(in_halo, user_params, R, x,y,z); // flag the pixels contained within this halo
                                update_in_halo(forbidden, user_params, (1.+global_params.R_OVERLAP_FACTOR)*R, x,y,z); // flag the pixels contained within this halo
                                dn++; // keep track of the number of halos
                                n++;

                                // Leaving empty for now
                            }
                        }
                    }
                    // *****************  END OPTIMIZATION ***************** //

                    if ((delta_m > delta_crit) && !in_halo[R_INDEX(x,y,z)] && !overlap_halo(in_halo, user_params, R, x,y,z)){ // we found us a "new" halo!

//                        printf("New Halo: x = %d y = %d z = %d Mass = %e\n",x,y,z,M);
                        update_in_halo(in_halo, user_params, R, x,y,z); // flag the pixels contained within this halo

                        dn++; // keep track of the number of halos
                        n++;
                    }
                }
            }
        }

        if (dn > 0){
            // now lets print out the mass functions (FgrtR)
            fgrtm += M/(RHOcrit*cosmo_params->OMm)*dn/VOLUME;
            dfgrtm += pow(M/(RHOcrit*cosmo_params->OMm)*sqrt(dn)/VOLUME, 2);


//            sprintf(filename, "../Output_files/FgtrM_files/hist_halos_z%.2f_%i_%.0fMpc_b%.3f_c%.3f", REDSHIFT, DIM, BOX_LEN, SHETH_b, SHETH_c);
//            F = fopen(filename, "a");
//            fprintf(F, "%e\t%e\t%e\t%e\t%e\t%e\n", M, fgrtm, sqrt(dfgrtm), FgtrM(REDSHIFT, M), FgtrM_st(REDSHIFT, M), FgtrM_bias(REDSHIFT, M, 0, sigma_z0(RtoM(L_FACTOR*BOX_LEN))) );
//            fclose(F);

            // and the dndlnm files
//            sprintf(filename, "../Output_files/DNDLNM_files/hist_halos_z%.2f_%i_%.0fMpc_b%.3f_c%.3f", REDSHIFT, DIM, BOX_LEN,  SHETH_b, SHETH_c);
//            F = fopen(filename, "a");
            //dm = RtoM(DELTA_R_FACTOR*R)-M;
            dlnm = log(RtoM(global_params.DELTA_R_FACTOR*R)) - log(M);
//            fprintf(F, "%e\t%e\t%e\t%e\t%e\t%e\n", M, dn/VOLUME/dlnm, sqrt(dn)/VOLUME/dlnm, M*dNdM(REDSHIFT, M), M*dNdM_st(REDSHIFT, M), M*dnbiasdM(M, REDSHIFT, RtoM(L_FACTOR*BOX_LEN), 0) );
            //      fprintf(F, "%e\t%e\t%e\t%e\t%e\t%e\n", M, M*dn/VOLUME/dm, M/dm/VOLUME*sqrt(dn), M*dNdM(REDSHIFT, M), M*dNdM_st(REDSHIFT, M), M*dnbiasdM(M, REDSHIFT, RtoM(BOX_LEN), 0) );
//            fclose(F);
        }

        R /= global_params.DELTA_R_FACTOR;
    }


    return(0);
}



// Funtion OVERLAP_HALO checks if the would be halo with radius R
// and centered on (x,y,z) overlaps with a preesisting halo
int overlap_halo(char * in_halo, struct UserParams *user_params, float R, int x, int y, int z) {

    int x_curr, y_curr, z_curr, x_min, x_max, y_min, y_max, z_min, z_max, R_index;
    float Rsq_curr_index, xsq, xplussq, xminsq, ysq, yplussq, yminsq, zsq, zplussq, zminsq;
    int x_index, y_index, z_index;

    // scale R to a effective overlap size, using R_OVERLAP_FACTOR
    R *= global_params.R_OVERLAP_FACTOR;

    // convert R to index units
    R_index = ceil(R/user_params->BOX_LEN*user_params->DIM);
    Rsq_curr_index = pow(R/user_params->BOX_LEN*user_params->DIM, 2); // convert to index

    // set parameter range
    x_min = x-R_index;
    x_max = x+R_index;
    y_min = y-R_index;
    y_max = y+R_index;
    z_min = z-R_index;
    z_max = z+R_index;

    //    printf("min %i, %i, %i\n", x_min, y_min, z_min);
    //printf("max %i, %i, %i\n", x_max, y_max, z_max);
    for (x_curr=x_min; x_curr<=x_max; x_curr++){
        for (y_curr=y_min; y_curr<=y_max; y_curr++){
            for (z_curr=z_min; z_curr<=z_max; z_curr++){
                x_index = x_curr;
                y_index = y_curr;
                z_index = z_curr;
                // adjust if we are outside of the box
                if (x_index<0) {x_index += user_params->DIM;}
                else if (x_index>=user_params->DIM) {x_index -= user_params->DIM;}
                if (y_index<0) {y_index += user_params->DIM;}
                else if (y_index>=user_params->DIM) {y_index -= user_params->DIM;}
                if (z_index<0) {z_index += user_params->DIM;}
                else if (z_index>=user_params->DIM) {z_index -= user_params->DIM;}

                // remember to check all reflections
                xsq = pow(x-x_index, 2);
                ysq = pow(y-y_index, 2);
                zsq = pow(z-z_index, 2);
                xplussq = pow(x-x_index+user_params->DIM, 2);
                yplussq = pow(y-y_index+user_params->DIM, 2);
                zplussq = pow(z-z_index+user_params->DIM, 2);
                xminsq = pow(x-x_index-user_params->DIM, 2);
                yminsq = pow(y-y_index-user_params->DIM, 2);
                zminsq = pow(z-z_index-user_params->DIM, 2);
                if ( in_halo[R_INDEX(x_index, y_index, z_index)] &&
                    ( (Rsq_curr_index > (xsq + ysq + zsq)) || // AND pixel is within this halo
                     (Rsq_curr_index > (xsq + ysq + zplussq)) ||
                     (Rsq_curr_index > (xsq + ysq + zminsq)) ||

                     (Rsq_curr_index > (xsq + yplussq + zsq)) ||
                     (Rsq_curr_index > (xsq + yplussq + zplussq)) ||
                     (Rsq_curr_index > (xsq + yplussq + zminsq)) ||

                     (Rsq_curr_index > (xsq + yminsq + zsq)) ||
                     (Rsq_curr_index > (xsq + yminsq + zplussq)) ||
                     (Rsq_curr_index > (xsq + yminsq + zminsq)) ||


                     (Rsq_curr_index > (xplussq + ysq + zsq)) ||
                     (Rsq_curr_index > (xplussq + ysq + zplussq)) ||
                     (Rsq_curr_index > (xplussq + ysq + zminsq)) ||

                     (Rsq_curr_index > (xplussq + yplussq + zsq)) ||
                     (Rsq_curr_index > (xplussq + yplussq + zplussq)) ||
                     (Rsq_curr_index > (xplussq + yplussq + zminsq)) ||

                     (Rsq_curr_index > (xplussq + yminsq + zsq)) ||
                     (Rsq_curr_index > (xplussq + yminsq + zplussq)) ||
                     (Rsq_curr_index > (xplussq + yminsq + zminsq)) ||


                     (Rsq_curr_index > (xminsq + ysq + zsq)) ||
                     (Rsq_curr_index > (xminsq + ysq + zplussq)) ||
                     (Rsq_curr_index > (xminsq + ysq + zminsq)) ||

                     (Rsq_curr_index > (xminsq + yplussq + zsq)) ||
                     (Rsq_curr_index > (xminsq + yplussq + zplussq)) ||
                     (Rsq_curr_index > (xminsq + yplussq + zminsq)) ||

                     (Rsq_curr_index > (xminsq + yminsq + zsq)) ||
                     (Rsq_curr_index > (xminsq + yminsq + zplussq)) ||
                     (Rsq_curr_index > (xminsq + yminsq + zminsq))
                     ) ){

                        // this pixel already belongs to a halo, and would want to become part of this halo as well
                        return 1;
                    }
            }
        }
    }

    return 0;
}



// Funtion UPDATE_IN_HALO takes in a box <in_halo> and flags all points
// which fall within radius R of (x,y,z).
void update_in_halo(char * in_halo, struct UserParams *user_params, float R, int x, int y, int z){
    int x_curr, y_curr, z_curr, x_min, x_max, y_min, y_max, z_min, z_max, R_index;
    float Rsq_curr_index, xsq, xplussq, xminsq, ysq, yplussq, yminsq, zsq, zplussq, zminsq;
    int x_index, y_index, z_index;

    // convert R to index units
    R_index = ceil(R/user_params->BOX_LEN*user_params->DIM);
    Rsq_curr_index = pow(R/user_params->BOX_LEN*user_params->DIM, 2); // convert to index

    // set parameter range
    x_min = x-R_index;
    x_max = x+R_index;
    y_min = y-R_index;
    y_max = y+R_index;
    z_min = z-R_index;
    z_max = z+R_index;

    //printf("min %i, %i, %i\n", x_min, y_min, z_min);
    //printf("max %i, %i, %i\n", x_max, y_max, z_max);
    for (x_curr=x_min; x_curr<=x_max; x_curr++){
        for (y_curr=y_min; y_curr<=y_max; y_curr++){
            for (z_curr=z_min; z_curr<=z_max; z_curr++){
                x_index = x_curr;
                y_index = y_curr;
                z_index = z_curr;
                // adjust if we are outside of the box
                if (x_index<0) {x_index += user_params->DIM;}
                else if (x_index>=user_params->DIM) {x_index -= user_params->DIM;}
                if (y_index<0) {y_index += user_params->DIM;}
                else if (y_index>=user_params->DIM) {y_index -= user_params->DIM;}
                if (z_index<0) {z_index += user_params->DIM;}
                else if (z_index>=user_params->DIM) {z_index -= user_params->DIM;}

                // now check
                if (!in_halo[R_INDEX(x_index, y_index, z_index)]){ // untaken pixel (not part of other halo)
                    // remember to check all reflections
                    xsq = pow(x-x_index, 2);
                    ysq = pow(y-y_index, 2);
                    zsq = pow(z-z_index, 2);
                    xplussq = pow(x-x_index+user_params->DIM, 2);
                    yplussq = pow(y-y_index+user_params->DIM, 2);
                    zplussq = pow(z-z_index+user_params->DIM, 2);
                    xminsq = pow(x-x_index-user_params->DIM, 2);
                    yminsq = pow(y-y_index-user_params->DIM, 2);
                    zminsq = pow(z-z_index-user_params->DIM, 2);
                    if ( (Rsq_curr_index > (xsq + ysq + zsq)) ||
                        (Rsq_curr_index > (xsq + ysq + zplussq)) ||
                        (Rsq_curr_index > (xsq + ysq + zminsq)) ||

                        (Rsq_curr_index > (xsq + yplussq + zsq)) ||
                        (Rsq_curr_index > (xsq + yplussq + zplussq)) ||
                        (Rsq_curr_index > (xsq + yplussq + zminsq)) ||

                        (Rsq_curr_index > (xsq + yminsq + zsq)) ||
                        (Rsq_curr_index > (xsq + yminsq + zplussq)) ||
                        (Rsq_curr_index > (xsq + yminsq + zminsq)) ||


                        (Rsq_curr_index > (xplussq + ysq + zsq)) ||
                        (Rsq_curr_index > (xplussq + ysq + zplussq)) ||
                        (Rsq_curr_index > (xplussq + ysq + zminsq)) ||

                        (Rsq_curr_index > (xplussq + yplussq + zsq)) ||
                        (Rsq_curr_index > (xplussq + yplussq + zplussq)) ||
                        (Rsq_curr_index > (xplussq + yplussq + zminsq)) ||

                        (Rsq_curr_index > (xplussq + yminsq + zsq)) ||
                        (Rsq_curr_index > (xplussq + yminsq + zplussq)) ||
                        (Rsq_curr_index > (xplussq + yminsq + zminsq)) ||


                        (Rsq_curr_index > (xminsq + ysq + zsq)) ||
                        (Rsq_curr_index > (xminsq + ysq + zplussq)) ||
                        (Rsq_curr_index > (xminsq + ysq + zminsq)) ||

                        (Rsq_curr_index > (xminsq + yplussq + zsq)) ||
                        (Rsq_curr_index > (xminsq + yplussq + zplussq)) ||
                        (Rsq_curr_index > (xminsq + yplussq + zminsq)) ||

                        (Rsq_curr_index > (xminsq + yminsq + zsq)) ||
                        (Rsq_curr_index > (xminsq + yminsq + zplussq)) ||
                        (Rsq_curr_index > (xminsq + yminsq + zminsq))
                        ){

                        // we are within the sphere defined by R, so change flag in in_halo array
                        in_halo[R_INDEX(x_index, y_index, z_index)] = 1;
                        //	    printf("%i, %i, %i\n", x_index, y_index, z_index);
                    }
                }
            }
        }
    }
}
