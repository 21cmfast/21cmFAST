// Re-write of perturb_field.c for being accessible within the MCMC
void compute_perturbed_velocities(
    unsigned short axis,
    struct UserParams *user_params,
    fftwf_complex *HIRES_density_perturb,
    fftwf_complex *HIRES_density_perturb_saved,
    fftwf_complex *LOWRES_density_perturb,
    fftwf_complex *LOWRES_density_perturb_saved,
    float dDdt_over_D,
    int dimension,
    int switch_mid,
    float f_pixel_factor,
    float *velocity
){

    float k_x, k_y, k_z, k_sq;
    int n_x, n_y, n_z;
    int i,j,k;

    float kvec[3];

    if(user_params->PERTURB_ON_HIGH_RES) {
        // We are going to generate the velocity field on the high-resolution perturbed
        // density grid
        memcpy(
            HIRES_density_perturb,
            HIRES_density_perturb_saved,
            sizeof(fftwf_complex)*KSPACE_NUM_PIXELS
        );
    }
    else {
        // We are going to generate the velocity field on the low-resolution perturbed density grid
        memcpy(
            LOWRES_density_perturb,
            LOWRES_density_perturb_saved,
            sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS
        );
        LOG_SUPER_DEBUG("dDdt_over_D=%.6e, dimension=%d, switch_mid=%d, f_pixel_factor=%f", dDdt_over_D, dimension, switch_mid, f_pixel_factor);
    }

    #pragma omp parallel \
        shared(LOWRES_density_perturb,HIRES_density_perturb,dDdt_over_D,dimension,switch_mid) \
        private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq, kvec) \
        num_threads(user_params->N_THREADS)
    {
        #pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
            if (n_x>switch_mid)
                k_x =(n_x-dimension) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y=0; n_y<dimension; n_y++){
                if (n_y>switch_mid)
                    k_y =(n_y-dimension) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                for (n_z=0; n_z<=(unsigned long long)(user_params->NON_CUBIC_FACTOR*switch_mid); n_z++){
                    k_z = n_z * DELTA_K_PARA;

                    kvec[0] = k_x;
                    kvec[1] = k_y;
                    kvec[2] = k_z;

                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                    // now set the velocities
                    if ((n_x==0) && (n_y==0) && (n_z==0)) { // DC mode
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            HIRES_density_perturb[0] = 0;
                        }
                        else {
                            LOWRES_density_perturb[0] = 0;
                        }
                    }
                    else{
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            HIRES_density_perturb[C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*kvec[axis]*I/k_sq/(TOT_NUM_PIXELS+0.0);
                        }
                        else {
                            LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*kvec[axis]*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                        }
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("density_perturb after modification by dDdt: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    if(user_params->PERTURB_ON_HIGH_RES) {

        // smooth the high resolution field ready for resampling
        if (user_params->DIM != user_params->HII_DIM)
            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));

        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, HIRES_density_perturb);

        #pragma omp parallel \
            shared(velocity,HIRES_density_perturb,f_pixel_factor) \
            private(i,j,k) \
            num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        *((float *)velocity + HII_R_INDEX(i,j,k)) = *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5), (unsigned long long)(j*f_pixel_factor+0.5), (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                }
            }
        }
    }
    else {
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, LOWRES_density_perturb);

        #pragma omp parallel \
            shared(velocity,LOWRES_density_perturb) \
            private(i,j,k) \
            num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        *((float *)velocity + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                    }
                }
            }
        }
    }
    LOG_SUPER_DEBUG("velocity: ");
    debugSummarizeBox(velocity, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

}

int ComputePerturbField(
    float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
    struct InitialConditions *boxes, struct PerturbedField *perturbed_field
){
    /*
     ComputePerturbField uses the first-order Langragian displacement field to move the
     masses in the cells of the density field. The high-res density field is extrapolated
     to some high-redshift (global_params.INITIAL_REDSHIFT), then uses the zeldovich
     approximation to move the grid "particles" onto the lower-res grid we use for the
     maps. Then we recalculate the velocity fields on the perturbed grid.
    */

    int status;
    Try{  // This Try{} wraps the whole function, so we don't indent.

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    omp_set_num_threads(user_params->N_THREADS);

    fftwf_complex *HIRES_density_perturb, *HIRES_density_perturb_saved;
    fftwf_complex *LOWRES_density_perturb, *LOWRES_density_perturb_saved;

    float growth_factor, displacement_factor_2LPT, init_growth_factor, init_displacement_factor_2LPT;
    double xf, yf, zf;
    float mass_factor, dDdt, f_pixel_factor, velocity_displacement_factor, velocity_displacement_factor_2LPT;
    unsigned long long ct, HII_i, HII_j, HII_k;
    int i,j,k,xi, yi, zi, dimension, switch_mid;
    double ave_delta, new_ave_delta;

    // Variables to perform cloud in cell re-distribution of mass for the perturbed field
    int xp1,yp1,zp1;
    float d_x,d_y,d_z,t_x,t_y,t_z;

    // Function for deciding the dimensions of loops when we could
    // use either the low or high resolution grids.
    switch(user_params->PERTURB_ON_HIGH_RES) {
        case 0:
            dimension = user_params->HII_DIM;
            switch_mid = HII_MIDDLE;
            break;
        case 1:
            dimension = user_params->DIM;
            switch_mid = MIDDLE;
            break;
    }

    // ***************   BEGIN INITIALIZATION   ************************** //

    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        LOG_WARNING("Resolution is likely too low for accurate evolved density fields\n \
                It is recommended that you either increase the resolution (DIM/BOX_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }

    growth_factor = dicke(redshift);
    displacement_factor_2LPT = -(3.0/7.0) * growth_factor*growth_factor; // 2LPT eq. D8

    dDdt = ddickedt(redshift); // time derivative of the growth factor (1/s)
    init_growth_factor = dicke(global_params.INITIAL_REDSHIFT);
    init_displacement_factor_2LPT = -(3.0/7.0) * init_growth_factor*init_growth_factor; // 2LPT eq. D8

    // find factor of HII pixel size / deltax pixel size
    f_pixel_factor = user_params->DIM/(float)(user_params->HII_DIM);
    mass_factor = pow(f_pixel_factor, 3);

    // allocate memory for the updated density, and initialize
    LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    if(user_params->PERTURB_ON_HIGH_RES) {
        HIRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        HIRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    }

    double *resampled_box;

    debugSummarizeIC(boxes, user_params->HII_DIM, user_params->DIM, user_params->NON_CUBIC_FACTOR);
    LOG_SUPER_DEBUG("growth_factor=%f, displacemet_factor_2LPT=%f, dDdt=%f, init_growth_factor=%f, init_displacement_factor_2LPT=%f, mass_factor=%f",
                    growth_factor, displacement_factor_2LPT, dDdt, init_growth_factor, init_displacement_factor_2LPT, mass_factor);

    // check if the linear evolution flag was set
    if (global_params.EVOLVE_DENSITY_LINEARLY){

        LOG_DEBUG("Linearly evolve density field");

#pragma omp parallel shared(growth_factor,boxes,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = growth_factor*boxes->hires_density[R_INDEX(i,j,k)];
                        }
                        else {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = growth_factor*boxes->lowres_density[HII_R_INDEX(i,j,k)];
                        }
                    }
                }
            }
        }
    }
    else {
        // Apply Zel'dovich/2LPT correction
        LOG_DEBUG("Apply Zel'dovich");

#pragma omp parallel shared(LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = 0.;
                        }
                        else {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                        }

                    }
                }
            }
        }

        velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;

        // now add the missing factor of D
#pragma omp parallel shared(boxes,velocity_displacement_factor,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->hires_vy[R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->hires_vz[R_INDEX(i,j,k)] *= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                        }
                        else {
                            boxes->lowres_vx[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->lowres_vy[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            boxes->lowres_vz[HII_R_INDEX(i,j,k)] *= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                        }
                    }
                }
            }
        }

        // * ************************************************************************* * //
        // *                           BEGIN 2LPT PART                                 * //
        // * ************************************************************************* * //
        // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        if(user_params->USE_2LPT){
            LOG_DEBUG("Apply 2LPT");

            // allocate memory for the velocity boxes and read them in
            velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;

            // now add the missing factor in eq. D9
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->hires_vy_2LPT[R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->hires_vz_2LPT[R_INDEX(i,j,k)] *= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                            }
                            else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] *= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // this is now comoving displacement in units of box size
                            }
                        }
                    }
                }
            }
        }


        // * ************************************************************************* * //
        // *                            END 2LPT PART                                  * //
        // * ************************************************************************* * //

        // ************  END INITIALIZATION **************************** //

        // Perturbing the density field required adding over multiple cells. Store intermediate result as a double to avoid rounding errors
        if(user_params->PERTURB_ON_HIGH_RES) {
            resampled_box = (double *)calloc(TOT_NUM_PIXELS,sizeof(double));
        }
        else {
            resampled_box = (double *)calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        }

        // go through the high-res box, mapping the mass onto the low-res (updated) box
        LOG_DEBUG("Perturb the density field");
        #pragma omp parallel \
            shared(init_growth_factor,boxes,f_pixel_factor,resampled_box,dimension) \
            private(i,j,k,xi,xf,yi,yf,zi,zf,HII_i,HII_j,HII_k,d_x,d_y,d_z,t_x,t_y,t_z,xp1,yp1,zp1) \
            num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (i=0; i<user_params->DIM;i++){
                for (j=0; j<user_params->DIM;j++){
                    for (k=0; k<D_PARA;k++){

                        // map indeces to locations in units of box size
                        xf = (i+0.5)/((user_params->DIM)+0.0);
                        yf = (j+0.5)/((user_params->DIM)+0.0);
                        zf = (k+0.5)/((D_PARA)+0.0);

                        // update locations
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            xf += (boxes->hires_vx)[R_INDEX(i, j, k)];
                            yf += (boxes->hires_vy)[R_INDEX(i, j, k)];
                            zf += (boxes->hires_vz)[R_INDEX(i, j, k)];
                        }
                        else {
                            HII_i = (unsigned long long)(i/f_pixel_factor);
                            HII_j = (unsigned long long)(j/f_pixel_factor);
                            HII_k = (unsigned long long)(k/f_pixel_factor);
                            xf += (boxes->lowres_vx)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                            yf += (boxes->lowres_vy)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                            zf += (boxes->lowres_vz)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                        }

                        // 2LPT PART
                        // add second order corrections
                        if(user_params->USE_2LPT){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                xf -= (boxes->hires_vx_2LPT)[R_INDEX(i,j,k)];
                                yf -= (boxes->hires_vy_2LPT)[R_INDEX(i,j,k)];
                                zf -= (boxes->hires_vz_2LPT)[R_INDEX(i,j,k)];
                            }
                            else {
                                xf -= (boxes->lowres_vx_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                yf -= (boxes->lowres_vy_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                zf -= (boxes->lowres_vz_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                            }
                        }
                        xf *= (double)(dimension);
                        yf *= (double)(dimension);
                        zf *= (double)((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));
                        while (xf >= (double)(dimension)){ xf -= (dimension);}
                        while (xf < 0){ xf += (dimension);}
                        while (yf >= (double)(dimension)){ yf -= (dimension);}
                        while (yf < 0){ yf += (dimension);}
                        while (zf >= (double)(user_params->NON_CUBIC_FACTOR*dimension)){ zf -= (user_params->NON_CUBIC_FACTOR*dimension);}
                        while (zf < 0){ zf += (user_params->NON_CUBIC_FACTOR*dimension);}
                        xi = xf;
                        yi = yf;
                        zi = zf;
                        if (xi >= (dimension)){ xi -= (dimension);}
                        if (xi < 0) {xi += (dimension);}
                        if (yi >= (dimension)){ yi -= (dimension);}
                        if (yi < 0) {yi += (dimension);}
                        if (zi >= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension))){ zi -= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}
                        if (zi < 0) {zi += ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}

                        // Determine the fraction of the perturbed cell which overlaps with the 8 nearest grid cells,
                        // based on the grid cell which contains the centre of the perturbed cell
                        d_x = fabs(xf - (double)(xi+0.5));
                        d_y = fabs(yf - (double)(yi+0.5));
                        d_z = fabs(zf - (double)(zi+0.5));
                        if(xf < (double)(xi+0.5)) {
                            // If perturbed cell centre is less than the mid-point then update fraction
                            // of mass in the cell and determine the cell centre of neighbour to be the
                            // lowest grid point index
                            d_x = 1. - d_x;
                            xi -= 1;
                            if (xi < 0) {xi += (dimension);} // Only this critera is possible as iterate back by one (we cannot exceed DIM)
                        }
                        if(yf < (double)(yi+0.5)) {
                            d_y = 1. - d_y;
                            yi -= 1;
                            if (yi < 0) {yi += (dimension);}
                        }
                        if(zf < (double)(zi+0.5)) {
                            d_z = 1. - d_z;
                            zi -= 1;
                            if (zi < 0) {zi += ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}
                        }
                        t_x = 1. - d_x;
                        t_y = 1. - d_y;
                        t_z = 1. - d_z;

                        // Determine the grid coordinates of the 8 neighbouring cells
                        // Takes into account the offset based on cell centre determined above
                        xp1 = xi + 1;
                        if(xp1 >= dimension) { xp1 -= (dimension);}
                        yp1 = yi + 1;
                        if(yp1 >= dimension) { yp1 -= (dimension);}
                        zp1 = zi + 1;
                        if(zp1 >= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension))) { zp1 -= ((unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension));}

                        if(user_params->PERTURB_ON_HIGH_RES) {
                            // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
#pragma omp atomic
                                resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
#pragma omp atomic
                                resampled_box[R_INDEX(xp1,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
#pragma omp atomic
                                resampled_box[R_INDEX(xi,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
#pragma omp atomic
                                resampled_box[R_INDEX(xp1,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
#pragma omp atomic
                                resampled_box[R_INDEX(xi,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
#pragma omp atomic
                                resampled_box[R_INDEX(xp1,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
#pragma omp atomic
                                resampled_box[R_INDEX(xi,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
#pragma omp atomic
                                resampled_box[R_INDEX(xp1,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                        }
                        else {
                            // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xp1,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xi,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xp1,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xi,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xp1,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xi,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
#pragma omp atomic
                                resampled_box[HII_R_INDEX(xp1,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                        }
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("resampled_box: ");
        debugSummarizeBoxDouble(resampled_box, dimension, user_params->NON_CUBIC_FACTOR, "  ");

        // Resample back to a float for remaining algorithm
        #pragma omp parallel \
            shared(LOWRES_density_perturb,HIRES_density_perturb,resampled_box,dimension) \
            private(i,j,k) \
            num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *( (float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) = (float)resampled_box[R_INDEX(i,j,k)];
                        }
                        else {
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) = (float)resampled_box[HII_R_INDEX(i,j,k)];
                        }
                    }
                }
            }
        }
        free(resampled_box);
        LOG_DEBUG("Finished perturbing the density field");

        LOG_SUPER_DEBUG("density_perturb: ");
        if(user_params->PERTURB_ON_HIGH_RES){
            debugSummarizeBoxComplex(HIRES_density_perturb, dimension, user_params->NON_CUBIC_FACTOR, "  ");
        }else{
            debugSummarizeBoxComplex(LOWRES_density_perturb, dimension, user_params->NON_CUBIC_FACTOR, "  ");
        }

        // deallocate
#pragma omp parallel shared(boxes,velocity_displacement_factor,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            boxes->hires_vx[R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->hires_vy[R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->hires_vz[R_INDEX(i,j,k)] /= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                        }
                        else {
                            boxes->lowres_vx[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->lowres_vy[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            boxes->lowres_vz[HII_R_INDEX(i,j,k)] /= (velocity_displacement_factor/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                        }
                    }
                }
            }
        }

        if(user_params->USE_2LPT){
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<(unsigned long long)(user_params->NON_CUBIC_FACTOR*dimension); k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->hires_vy_2LPT[R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->hires_vz_2LPT[R_INDEX(i,j,k)] /= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                            }
                            else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] /= (velocity_displacement_factor_2LPT/user_params->NON_CUBIC_FACTOR); // convert back to z = 0 quantity
                            }
                        }
                    }
                }
            }
        }
        LOG_DEBUG("Cleanup velocities for perturb");
    }

    // Now, if I still have the high resolution density grid (HIRES_density_perturb) I need to downsample it to the low-resolution grid
    if(user_params->PERTURB_ON_HIGH_RES) {

        LOG_DEBUG("Downsample the high-res perturbed density");

        // Transform to Fourier space to sample (filter) the box
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, HIRES_density_perturb);

        // Need to save a copy of the high-resolution unfiltered density field for the velocities
        memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // Now filter the box
        if (user_params->DIM != user_params->HII_DIM) {
            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
        }

        // FFT back to real space
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, D_PARA, user_params->N_THREADS, HIRES_density_perturb);

        // Renormalise the FFT'd box
#pragma omp parallel shared(HIRES_density_perturb,LOWRES_density_perturb,f_pixel_factor,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) =
                        *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)))/(float)TOT_NUM_PIXELS;

                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) -= 1.;

                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1.+FRACT_FLOAT_ERR;
                        }
                    }
                }
            }
        }
    }
    else {

        if (!global_params.EVOLVE_DENSITY_LINEARLY){

#pragma omp parallel shared(LOWRES_density_perturb,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<HII_D_PARA; k++){
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1.;
                        }
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("LOWRES_density_perturb: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    // transform to k-space
    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, LOWRES_density_perturb);

    //smooth the field
    if (!global_params.EVOLVE_DENSITY_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
        filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
    }

    LOG_SUPER_DEBUG("LOWRES_density_perturb after smoothing: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    // save a copy of the k-space density field
    memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, LOWRES_density_perturb);

    LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space: ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");

    // normalize after FFT
    int bad_count=0;
#pragma omp parallel shared(LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS) reduction(+: bad_count)
    {
#pragma omp for
        for(i=0; i<user_params->HII_DIM; i++){
            for(j=0; j<user_params->HII_DIM; j++){
                for(k=0; k<HII_D_PARA; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;

                    if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1.0) { // shouldn't happen

                        if(bad_count<5) LOG_WARNING("LOWRES_density_perturb is <-1 for index %d %d %d (value=%f)", i,j,k, *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)));
                        if(bad_count==5) LOG_WARNING("Skipping further warnings for LOWRES_density_perturb.");
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                        bad_count++;
                    }
                }
            }
        }
    }
    if(bad_count>=5) LOG_WARNING("Total number of bad indices for LOW_density_perturb: %d", bad_count);
    LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space (normalized): ");
    debugSummarizeBoxComplex(LOWRES_density_perturb, user_params->HII_DIM, user_params->NON_CUBIC_FACTOR, "  ");


#pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<HII_D_PARA; k++){
                    *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                }
            }
        }
    }

    // ****  Convert to velocities ***** //
    LOG_DEBUG("Generate velocity fields");

    float k_x, k_y, k_z, k_sq, dDdt_over_D;
    int n_x, n_y, n_z;

    dDdt_over_D = dDdt/growth_factor;


    if (user_params->KEEP_3D_VELOCITIES){
        compute_perturbed_velocities(
            0,
            user_params,
            HIRES_density_perturb,
            HIRES_density_perturb_saved,
            LOWRES_density_perturb,
            LOWRES_density_perturb_saved,
            dDdt_over_D,
            dimension,
            switch_mid,
            f_pixel_factor,
            perturbed_field->velocity_x
        );
        compute_perturbed_velocities(
            1,
            user_params,
            HIRES_density_perturb,
            HIRES_density_perturb_saved,
            LOWRES_density_perturb,
            LOWRES_density_perturb_saved,
            dDdt_over_D,
            dimension,
            switch_mid,
            f_pixel_factor,
            perturbed_field->velocity_y
        );
    }

    compute_perturbed_velocities(
        2,
        user_params,
        HIRES_density_perturb,
        HIRES_density_perturb_saved,
        LOWRES_density_perturb,
        LOWRES_density_perturb_saved,
        dDdt_over_D,
        dimension,
        switch_mid,
        f_pixel_factor,
        perturbed_field->velocity_z
    );

    fftwf_cleanup_threads();
    fftwf_cleanup();
    fftwf_forget_wisdom();

    // deallocate
    fftwf_free(LOWRES_density_perturb);
    fftwf_free(LOWRES_density_perturb_saved);
    if(user_params->PERTURB_ON_HIGH_RES) {
        fftwf_free(HIRES_density_perturb);
        fftwf_free(HIRES_density_perturb_saved);
    }
    fftwf_cleanup();

    } // End of Try{}
    Catch(status){
        return(status);
    }

    return(0);
}
