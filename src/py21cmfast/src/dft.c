int dft_c2r_cube(bool use_wisdom, int dim, int n_threads, fftwf_complex *box){
    char wisdom_filename[500];
    unsigned flag = FFTW_ESTIMATE;
    int status;
    fftwf_plan plan;

    Try{
        if(use_wisdom) {
            // Check to see if the wisdom exists
            sprintf(wisdom_filename,"%s/c2r_DIM%d_NTHREADS%d.",global_params.wisdoms_path, dim, n_threads);

            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                unsigned flag = FFTW_WISDOM_ONLY;
            }
            else {
                LOG_WARNING("Cannot locate FFTW Wisdom: %s file not found. Reverting to FFTW_ESTIMATE.", wisdom_filename);
            }
        }
        plan = fftwf_plan_dft_c2r_3d(dim, dim, dim, (fftwf_complex *)box, (float *)box, flag);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }
    Catch(status){
        return(status);
    }
    return(0);
}

int dft_r2c_cube(bool use_wisdom, int dim, int n_threads, fftwf_complex *box){
    char wisdom_filename[500];
    unsigned flag = FFTW_ESTIMATE;
    int status;
    fftwf_plan plan;

    Try{
        if(use_wisdom) {
            // Check to see if the wisdom exists
            sprintf(wisdom_filename,"%s/r2c_DIM%d_NTHREADS%d", global_params.wisdoms_path, dim, n_threads);

            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                unsigned flag = FFTW_WISDOM_ONLY;
            }
            else {
                LOG_WARNING("Cannot locate FFTW Wisdom: %s file not found. Reverting to FFTW_ESTIMATE.", wisdom_filename);
            }
        }
        plan = fftwf_plan_dft_r2c_3d(dim, dim, dim, (float *)box, (fftwf_complex *)box, flag);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }
    Catch(status){
        return(status);
    }
    return(0);
}

int CreateFFTWWisdoms(struct UserParams *user_params, struct CosmoParams *cosmo_params) {

    int status;
    char *wisdom_string;

    Try{ // This Try wraps the entire function so we don't indent.

        Broadcast_struct_global_UF(user_params,cosmo_params);

        fftwf_plan plan;

        char wisdom_filename[500];

        int i,j,k;

        omp_set_num_threads(user_params->N_THREADS);
        fftwf_init_threads();
        fftwf_plan_with_nthreads(user_params->N_THREADS);
        fftwf_cleanup_threads();

        // allocate array for the k-space and real-space boxes
        fftwf_complex *HIRES_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        fftwf_complex *LOWRES_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
//
//#pragma omp parallel shared(HIRES_box) private(i,j,k) num_threads(user_params->N_THREADS)
//        {
//#pragma omp for
//            for (i=0; i<user_params->DIM; i++){
//                for (j=0; j<user_params->DIM; j++){
//                    for (k=0; k<=MIDDLE; k++){
//                        HIRES_box[C_INDEX(i, j, k)] = 0.0;
//                    }
//                }
//            }
//        }
//
//#pragma omp parallel shared(LOWRES_box) private(i,j,k) num_threads(user_params->N_THREADS)
//        {
//#pragma omp for
//            for (i=0; i<user_params->HII_DIM; i++){
//                for (j=0; j<user_params->HII_DIM; j++){
//                    for (k=0; k<=HII_MIDDLE; k++){
//                        LOWRES_box[HII_C_INDEX(i, j, k)] = 0.0;
//                    }
//                }
//            }
//        }



        sprintf(wisdom_filename,"%s/r2c_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->DIM,user_params->N_THREADS);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftwf_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename,"%s/c2r_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->DIM,user_params->N_THREADS);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftwf_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (fftwf_complex *)HIRES_box, (float *)HIRES_box,  FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename,"r2c_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->HII_DIM,user_params->N_THREADS);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftwf_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (float *)LOWRES_box, (fftwf_complex *)LOWRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename,"c2r_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->HII_DIM,user_params->N_THREADS);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftwf_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (fftwf_complex *)LOWRES_box, (float *)LOWRES_box,  FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();

        // deallocate
        fftwf_free(HIRES_box);
        fftwf_free(LOWRES_box);


    } // End of Try{}

    Catch(status){
        return(status);
    }
    return(0);
}

//int CreateFFTWWisdoms(struct UserParams *user_params, struct CosmoParams *cosmo_params) {
//    fftwf_plan plan;
//    int dim = 75;
//
//    // allocate array for the k-space and real-space boxes
////    fftwf_complex *HIRES_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*dim*dim*(dim/2+1));
//    fftwf_complex *oned_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*dim);
//
//    // Going to need to construct an FFTW_Wisdom for this box
////    plan = fftwf_plan_dft_r2c_3d(dim,dim,dim, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_PATIENT);
//    plan = fftwf_plan_dft_1d(dim, oned_box, oned_box, FFTW_FORWARD, FFTW_PATIENT);
//    printf("Made a plan!\n");
//    fftwf_print_plan(plan);
//}
