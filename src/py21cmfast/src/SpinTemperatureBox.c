
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

// Grids/arrays that only need to be initialised once (i.e. the lowest redshift density cube to be sampled)
double ***fcoll_R_grid, ***dfcoll_dz_grid;
double **grid_dens, **density_gridpoints;
double *Sigma_Tmin_grid, *ST_over_PS_arg_grid, *dstarlya_dt_prefactor, *zpp_edge, *sigma_atR;
double *dstarlyLW_dt_prefactor, *dstarlya_dt_prefactor_MINI, *dstarlyLW_dt_prefactor_MINI;
float **delNL0_rev,**delNL0;
float *R_values, *delNL0_bw, *delNL0_Offset, *delNL0_LL, *delNL0_UL, *delNL0_ibw, *log10delNL0_diff;
float *log10delNL0_diff_UL,*min_densities, *max_densities, *zpp_interp_table;
short **dens_grid_int_vals;
short *SingleVal_int;

float *del_fcoll_Rct, *SFR_timescale_factor;
float *del_fcoll_Rct_MINI;

double *dxheat_dt_box, *dxion_source_dt_box, *dxlya_dt_box, *dstarlya_dt_box;
double *dxheat_dt_box_MINI, *dxion_source_dt_box_MINI, *dxlya_dt_box_MINI, *dstarlya_dt_box_MINI;
double *dstarlyLW_dt_box, *dstarlyLW_dt_box_MINI;

//Arrays needed for Heating calculations
double *dstarlya_cont_dt_box, *dstarlya_inj_dt_box, *dstarlya_cont_dt_prefactor, *dstarlya_inj_dt_prefactor, *sum_ly2, *sum_lynto2;
double *dstarlya_cont_dt_box_MINI, *dstarlya_inj_dt_box_MINI, *dstarlya_cont_dt_prefactor_MINI, *dstarlya_inj_dt_prefactor_MINI, *sum_ly2_MINI, *sum_lynto2_MINI;
//Variables needed for heating calculations
double prev_Ts, tau21, xCMB, eps_CMB, E_continuum, E_injected, Ndot_alpha_cont, Ndot_alpha_inj, Ndot_alpha_cont_MINI, Ndot_alpha_inj_MINI;
double ly2_store, ly2_store_MINI, lynto2_store, lynto2_store_MINI;
double dCMBheat_dzp, eps_Lya_cont, eps_Lya_inj, eps_Lya_cont_MINI, eps_Lya_inj_MINI, dstarlya_cont_dt, dstarlya_inj_dt;

double *log10_Mcrit_LW_ave_list;

float *inverse_val_box;
int *m_xHII_low_box;

// Grids/arrays that are re-evaluated for each zp
double **fcoll_interp1, **fcoll_interp2, **dfcoll_interp1, **dfcoll_interp2;
double *fcoll_R_array, *sigma_Tmin, *ST_over_PS, *sum_lyn;
float *inverse_diff, *zpp_growth, *zpp_for_evolve_list,*Mcrit_atom_interp_table;
double *ST_over_PS_MINI,*sum_lyn_MINI,*sum_lyLWn,*sum_lyLWn_MINI;

// interpolation tables for the heating/ionisation integrals
double **freq_int_heat_tbl, **freq_int_ion_tbl, **freq_int_lya_tbl, **freq_int_heat_tbl_diff;
double **freq_int_ion_tbl_diff, **freq_int_lya_tbl_diff;

bool TsInterpArraysInitialised = false;
float initialised_redshift = -1.0;

int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options,
                  float perturbed_field_redshift, short cleanup,
                  struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp,
                  struct InitialConditions *ini_boxes, struct TsBox *this_spin_temp) {
    int status;
    Try{ // This Try{} wraps the whole function.
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
    omp_set_num_threads(user_params->N_THREADS);

    /////////////////// Defining variables for the computation of Ts.c //////////////

    FILE *F, *OUT;

    unsigned long long ct, FCOLL_SHORT_FACTOR, box_ct;

    int R_ct,i,ii,j,k,i_z,COMPUTE_Ts,x_e_ct,m_xHII_low,m_xHII_high,n_ct, zpp_gridpoint1_int;
    int zpp_gridpoint2_int,zpp_evolve_gridpoint1_int, zpp_evolve_gridpoint2_int,counter;

    short dens_grid_int;

    double Tk_ave, J_alpha_ave, xalpha_ave, J_alpha_tot, Xheat_ave, Xion_ave, nuprime, Ts_ave;
    double lower_int_limit,Luminosity_converstion_factor,T_inv_TS_fast_inv;
    double J_LW_ave, J_alpha_tot_MINI, J_alpha_ave_MINI, J_LW_ave_MINI,dxheat_dzp_MINI,Xheat_ave_MINI;
    double dadia_dzp, dcomp_dzp, dxheat_dt, dxion_source_dt, dxion_sink_dt, T, x_e, dxe_dzp, n_b;
    double dspec_dzp, dxheat_dzp, dxlya_dt, dstarlya_dt, fcoll_R;
    double Trad_fast,xc_fast,xc_inverse,TS_fast,TSold_fast,xa_tilde_fast,TS_prefactor,xa_tilde_prefactor,gamma_alpha;
    double T_inv,T_inv_sq,xi_power,xa_tilde_fast_arg,Trad_fast_inv,TS_fast_inv,dcomp_dzp_prefactor;

    float growth_factor_z, inverse_growth_factor_z, R, R_factor, zp, mu_for_Ts, filling_factor_of_HI_zp;
    float dzp, prev_zp, zpp, prev_zpp, prev_R, Tk_BC, xe_BC;
    float xHII_call, curr_xalpha, TK, TS, xe, deltax_highz, cT_ad;
    float zpp_for_evolve,dzpp_for_evolve, M_MIN;
    float gdens, growthfac;

    float determine_zpp_max, zpp_grid, zpp_gridpoint1, zpp_gridpoint2,zpp_evolve_gridpoint1;
    float zpp_evolve_gridpoint2, grad1, grad2, grad3, grad4, delNL0_bw_val;
    float OffsetValue, DensityValueLow, min_density, max_density;

    double curr_delNL0, inverse_val,prefactor_1,prefactor_2,dfcoll_dz_val, density_eval1;
    double density_eval2, grid_sigmaTmin, grid_dens_val, dens_grad, dens_width;
    double prefactor_2_MINI, dfcoll_dz_val_MINI;

    double const_zp_prefactor, dt_dzp, x_e_ave, growth_factor_zp, dgrowth_factor_dzp, fcoll_R_for_reduction;
    double const_zp_prefactor_MINI;

    int n_pts_radii;
    double trial_zpp_min,trial_zpp_max,trial_zpp, weight;
    bool first_radii, first_zero;
    first_radii = true;
    first_zero = true;
    n_pts_radii = 1000;

    float M_MIN_WDM =  M_J_WDM();

    double ave_fcoll, ave_fcoll_inv, dfcoll_dz_val_ave, ION_EFF_FACTOR;
    double ave_fcoll_MINI, ave_fcoll_inv_MINI, dfcoll_dz_val_ave_MINI, ION_EFF_FACTOR_MINI;

    float curr_dens, min_curr_dens, max_curr_dens;

    float curr_vcb;
    min_curr_dens = max_curr_dens = 0.;

    int fcoll_int_min, fcoll_int_max;

    fcoll_int_min = fcoll_int_max = 0;

    float Splined_Fcoll,Splined_Fcollzp_mean,Splined_SFRD_zpp, fcoll;
    float redshift_table_Nion_z,redshift_table_SFRD, fcoll_interp_val1, fcoll_interp_val2, dens_val;
    float fcoll_interp_min, fcoll_interp_bin_width, fcoll_interp_bin_width_inv;
    float fcoll_interp_high_min, fcoll_interp_high_bin_width, fcoll_interp_high_bin_width_inv;

    float Splined_Fcoll_MINI,Splined_Fcollzp_mean_MINI_left,Splined_Fcollzp_mean_MINI_right;
    float Splined_Fcollzp_mean_MINI,Splined_SFRD_zpp_MINI_left,Splined_SFRD_zpp_MINI_right;
    float Splined_SFRD_zpp_MINI, fcoll_MINI,fcoll_MINI_right,fcoll_MINI_left;
    float fcoll_interp_min_MINI, fcoll_interp_bin_width_MINI, fcoll_interp_bin_width_inv_MINI;
    float fcoll_interp_val1_MINI, fcoll_interp_val2_MINI;
    float fcoll_interp_high_min_MINI, fcoll_interp_high_bin_width_MINI, fcoll_interp_high_bin_width_inv_MINI;

    int fcoll_int;
    int redshift_int_Nion_z,redshift_int_SFRD;
    float zpp_integrand, Mlim_Fstar, Mlim_Fesc, Mlim_Fstar_MINI, Mlim_Fesc_MINI, Mmax, sigmaMmax;

    double log10_Mcrit_LW_ave;
    float log10_Mcrit_mol;
    float log10_Mcrit_LW_ave_table_Nion_z, log10_Mcrit_LW_ave_table_SFRD;
    int  log10_Mcrit_LW_ave_int_Nion_z, log10_Mcrit_LW_ave_int_SFRD;
    double LOG10_MTURN_INT = (double) ((LOG10_MTURN_MAX - LOG10_MTURN_MIN)) / ((double) (NMTURN - 1.));
    float **log10_Mcrit_LW;
    int log10_Mcrit_LW_int;
    float log10_Mcrit_LW_diff, log10_Mcrit_LW_val;

    int table_int_boundexceeded = 0;
    int fcoll_int_boundexceeded = 0;

    int *fcoll_int_boundexceeded_threaded = calloc(user_params->N_THREADS,sizeof(int));
    int *table_int_boundexceeded_threaded = calloc(user_params->N_THREADS,sizeof(int));
    for(i=0;i<user_params->N_THREADS;i++) {
        fcoll_int_boundexceeded_threaded[i] = 0;
        table_int_boundexceeded_threaded[i] = 0;
    }

    double total_time, total_time2, total_time3, total_time4;
    float M_MIN_at_zp;

    int NO_LIGHT = 0;

    bool initialization_required = fabs(initialised_redshift - perturbed_field_redshift) > 0.0001;

    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;
        ION_EFF_FACTOR_MINI = global_params.Pop3_ion * astro_params->F_STAR7_MINI * astro_params->F_ESC7_MINI;
    }
    else {
        ION_EFF_FACTOR = astro_params->HII_EFF_FACTOR;
        ION_EFF_FACTOR_MINI = 0.;
    }

    // Initialise arrays to be used for the Ts.c computation //
    fftwf_complex *box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *unfiltered_box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *log10_Mcrit_LW_unfiltered, *log10_Mcrit_LW_filtered;
    if (flag_options->USE_MINI_HALOS){
        log10_Mcrit_LW_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        log10_Mcrit_LW_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        log10_Mcrit_LW = (float **) calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            log10_Mcrit_LW[R_ct] = (float *) calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        }
    }

LOG_SUPER_DEBUG("initialized");

    if(!TsInterpArraysInitialised) {
LOG_SUPER_DEBUG("initalising Ts Interp Arrays");

        // Grids/arrays that only need to be initialised once (i.e. the lowest redshift density cube to be sampled)

        zpp_edge = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        sigma_atR = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        R_values = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

        if(user_params->USE_INTERPOLATION_TABLES) {
            min_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            max_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

            zpp_interp_table = calloc(zpp_interp_points_SFR, sizeof(float));
        }

        if(flag_options->USE_MASS_DEPENDENT_ZETA) {

            SFR_timescale_factor = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

            if(user_params->MINIMIZE_MEMORY) {
                delNL0 = (float **)calloc(1,sizeof(float *));
                delNL0[0] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));
            }
            else {
                delNL0 = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                    delNL0[i] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));
                }
            }

            xi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));
            wi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));

            if(user_params->USE_INTERPOLATION_TABLES) {
                overdense_low_table = calloc(NSFR_low,sizeof(double));
                overdense_high_table = calloc(NSFR_high,sizeof(double));

                log10_SFRD_z_low_table = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    log10_SFRD_z_low_table[j] = (float *)calloc(NSFR_low,sizeof(float));
                }

                SFRD_z_high_table = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    SFRD_z_high_table[j] = (float *)calloc(NSFR_high,sizeof(float));
                }

                if(flag_options->USE_MINI_HALOS){
                    log10_SFRD_z_low_table_MINI = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                    for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                        log10_SFRD_z_low_table_MINI[j] = (float *)calloc(NSFR_low*NMTURN,sizeof(float));
                    }

                    SFRD_z_high_table_MINI = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                    for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                        SFRD_z_high_table_MINI[j] = (float *)calloc(NSFR_high*NMTURN,sizeof(float));
                    }
                }
            }

            if(flag_options->USE_MINI_HALOS){
                log10_Mcrit_LW_ave_list = (double *) calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            }

            del_fcoll_Rct = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));

            dxheat_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dxion_source_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dxlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dstarlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            //Allocate memory for Lya heating arrays
            if (flag_options->USE_LYA_HEATING){
                dstarlya_cont_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dstarlya_inj_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            }
            if (flag_options->USE_MINI_HALOS){
                del_fcoll_Rct_MINI = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));

                dstarlyLW_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dxheat_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dxion_source_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dxlya_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dstarlya_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dstarlyLW_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));

                if (flag_options->USE_LYA_HEATING){
                    dstarlya_cont_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                    dstarlya_inj_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                }
            }

            m_xHII_low_box = (int *)calloc(HII_TOT_NUM_PIXELS,sizeof(int));
            inverse_val_box = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));

        }
        else {

            if(user_params->USE_INTERPOLATION_TABLES) {
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

                dens_grid_int_vals = (short **)calloc(HII_TOT_NUM_PIXELS,sizeof(short *));
                for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
                    dens_grid_int_vals[i] = (short *)calloc((float)global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(short));
                }
            }

            delNL0_rev = (float **)calloc(HII_TOT_NUM_PIXELS,sizeof(float *));
            for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
                delNL0_rev[i] = (float *)calloc((float)global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            }
        }

        dstarlya_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        //Allocate memory for Lya heating arrays
        if (flag_options->USE_LYA_HEATING){
            dstarlya_cont_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dstarlya_inj_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
        if (flag_options->USE_MINI_HALOS){
            dstarlya_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dstarlyLW_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dstarlyLW_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));

            if (flag_options->USE_LYA_HEATING){
                dstarlya_cont_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                dstarlya_inj_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            }
        }
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
        //Allocate memory for Lya heating arrays
        if (flag_options->USE_LYA_HEATING){
            sum_ly2 = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lynto2 = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
        if (flag_options->USE_MINI_HALOS){
            Mcrit_atom_interp_table = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            ST_over_PS_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lyn_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lyLWn = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lyLWn_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));

            if (flag_options->USE_LYA_HEATING){
                sum_ly2_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                sum_lynto2_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            }
        }

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
        if (flag_options->USE_MINI_HALOS){
            M_MIN = (global_params.M_MIN_INTEGRAL)/50.;

            Mlim_Fstar = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR, astro_params->F_STAR10);
            Mlim_Fesc = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC, astro_params->F_ESC10);

            Mlim_Fstar_MINI = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR_MINI,
                                                   astro_params->F_STAR7_MINI * pow(1e3, astro_params->ALPHA_STAR_MINI));
            Mlim_Fesc_MINI = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC,
                                                  astro_params->F_ESC7_MINI * pow(1e3, astro_params->ALPHA_ESC));
        }
        else{
            M_MIN = (astro_params->M_TURN)/50.;

            Mlim_Fstar = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR, astro_params->F_STAR10);
            Mlim_Fesc = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC, astro_params->F_ESC10);
        }
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
        }
    }

    init_ps();

LOG_SUPER_DEBUG("Initialised PS");
LOG_SUPER_DEBUG("About to initialise heat");
    init_heat();
LOG_SUPER_DEBUG("Initialised heat");

    // Initialize some interpolation tables
    // if(initialization_required) {

        if(user_params->USE_INTERPOLATION_TABLES) {
            if(user_params->FAST_FCOLL_TABLES){
                initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
            }
          else{
            if(flag_options->M_MIN_in_Mass || flag_options->USE_MASS_DEPENDENT_ZETA) {
                if (flag_options->USE_MINI_HALOS){
                    initialiseSigmaMInterpTable(global_params.M_MIN_INTEGRAL/50.,1e20);
                }
                else{
                    initialiseSigmaMInterpTable(M_MIN,1e20);
                }
            }
            LOG_SUPER_DEBUG("Initialised sigmaM interp table");
          }
        }
    // }

    if (redshift >= global_params.Z_HEAT_MAX){
        LOG_SUPER_DEBUG("redshift %f >= Z_HEAT_MAX. Doing fast initial heating.", redshift);
        xe = xion_RECFAST(redshift,0);
        TK = T_RECFAST(redshift,0);
        cT_ad = cT_approx(redshift); //finding the adiabatic index at the initial redshift from 2302.08506 to fix adiabatic fluctuations.
        growth_factor_zp = dicke(redshift);

        growthfac = growth_factor_zp * inverse_growth_factor_z;
        // read file
        #pragma omp parallel shared(this_spin_temp,xe,TK,redshift,perturbed_field, \
                                    growthfac,cT_ad) \
                             private(i,j,k,ct,curr_xalpha,gdens) \
                             num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<HII_D_PARA; k++){
                        ct=HII_R_INDEX(i,j,k);
                        gdens = perturbed_field->density[ct]*growthfac;
                        this_spin_temp->Tk_box[ct] = TK *(1.0 + cT_ad * gdens);
                        this_spin_temp->x_e_box[ct] = xe;
                        // compute the spin temperature
                        this_spin_temp->Ts_box[ct] = get_Ts(redshift, gdens, TK, xe, 0, &curr_xalpha);
                    }
                }
            }
        }

        if(!flag_options->M_MIN_in_Mass) {
            M_MIN = (float)TtoM(redshift, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
            LOG_DEBUG("Attempting to initialise sigmaM table with M_MIN=%e, Tvir_MIN=%e, mu=%e",
                      M_MIN, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
            if(user_params->USE_INTERPOLATION_TABLES) {
              if(user_params->FAST_FCOLL_TABLES){
                initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
              }
              else{
                initialiseSigmaMInterpTable(M_MIN,1e20);
              }
            }
        }
        LOG_SUPER_DEBUG("Initialised Sigma interp table");

    }
    else {
        LOG_SUPER_DEBUG("Redshift %f less than Z_HEAT_MAX (%f)", redshift, global_params.Z_HEAT_MAX);
        // Note that in this case we NEED the previous_spin_temp to be allocated
        // and calculated (either in the fast way, or slow way below). If we get here
        // and it is not, then it will segfault. We should try to make the structs a
        // little more robust so we can check this and return a nicer error message.
        x_e_ave = Tk_ave = 0.0;

        #pragma omp parallel shared(previous_spin_temp) private(ct) num_threads(user_params->N_THREADS)
        {
            #pragma omp for reduction(+:x_e_ave,Tk_ave)
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                x_e_ave += previous_spin_temp->x_e_box[ct];
                Tk_ave += previous_spin_temp->Tk_box[ct];
            }
        }
        x_e_ave /= (float)HII_TOT_NUM_PIXELS;
        Tk_ave /= (float)HII_TOT_NUM_PIXELS;

        LOG_INFO("previous_spin_temp: %e", previous_spin_temp->Tk_box[0]);

        /////////////// Create the z=0 non-linear density fields smoothed on scale R to be used in computing fcoll //////////////
        R = L_FACTOR*user_params->BOX_LEN/(float)user_params->HII_DIM;
        R_factor = pow(global_params.R_XLy_MAX/R, 1/((float)global_params.NUM_FILTER_STEPS_FOR_Ts));
        LOG_SUPER_DEBUG("Looping through R");

        if(initialization_required) {

            // allocate memory for the nonlinear density field
            #pragma omp parallel shared(unfiltered_box,perturbed_field) private(i,j,k) num_threads(user_params->N_THREADS)
            {
                #pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<HII_D_PARA; k++){
                            *((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->density[HII_R_INDEX(i,j,k)];
                        }
                    }
                }
            }
            LOG_DEBUG("Allocated unfiltered box");

            ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, unfiltered_box);
            LOG_DEBUG("Done FFT on unfiltered box");

            // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
            // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
            #pragma omp parallel shared(unfiltered_box) private(ct) num_threads(user_params->N_THREADS)
            {
                #pragma omp for
                for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                    unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
                }
            }

            LOG_SUPER_DEBUG("normalised unfiltered box");

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
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box);
                LOG_ULTRA_DEBUG("Executed FFT for R=%f", R);

                min_density = 0.0;
                max_density = 0.0;

                // copy over the values
                #pragma omp parallel shared(box,inverse_growth_factor_z,delNL0,delNL0_rev) \
                                     private(i,j,k,curr_delNL0) \
                                     num_threads(user_params->N_THREADS)
                {
                    #pragma omp for reduction(max:max_density) reduction(min:min_density)
                    for (i=0;i<user_params->HII_DIM; i++){
                        for (j=0;j<user_params->HII_DIM; j++){
                            for (k=0;k<HII_D_PARA; k++){
                                curr_delNL0 = *((float *)box + HII_R_FFT_INDEX(i,j,k));

                                if (curr_delNL0 <= -1){ // correct for aliasing in the filtering step
                                    curr_delNL0 = -1+FRACT_FLOAT_ERR;
                                }

                                // and linearly extrapolate to z=0
                                curr_delNL0 *= inverse_growth_factor_z;

                                if(flag_options->USE_MASS_DEPENDENT_ZETA) {
                                    if(!user_params->MINIMIZE_MEMORY) {
                                        delNL0[R_ct][HII_R_INDEX(i,j,k)] = curr_delNL0;
                                    }
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
                }

                LOG_ULTRA_DEBUG("COPIED OVER VALUES");

                if(user_params->USE_INTERPOLATION_TABLES) {
                    if(min_density < 0.0) {
                        min_density = min_density*1.01;
                        // min_density here can exceed -1. as it is always extrapolated back to the appropriate redshift
                    }
                    else {
                        min_density = min_density*0.99;
                    }
                    if(max_density < 0.0) {
                        max_density = max_density*0.99;
                    }
                    else {
                        max_density = max_density*1.01;
                    }

                    if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
                        delNL0_LL[R_ct] = min_density;
                        delNL0_Offset[R_ct] = 1.e-6 - (delNL0_LL[R_ct]);
                        delNL0_UL[R_ct] = max_density;
                    }

                    min_densities[R_ct] = min_density;
                    max_densities[R_ct] = max_density;
                }

                R *= R_factor;
                LOG_ULTRA_DEBUG("FINISHED WITH THIS R, MOVING ON");
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

        // This sets the delta_z step for determining the heating/ionisation integrals.
        dzp = redshift - prev_redshift;

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
            M_MIN = (float)TtoM(determine_zpp_max, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
            if(user_params->USE_INTERPOLATION_TABLES) {
              if(user_params->FAST_FCOLL_TABLES){
                initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
              }
              else{
                initialiseSigmaMInterpTable(M_MIN,1e20);
              }
            }
        }

        LOG_SUPER_DEBUG("Initialised sigma interp table");

        if(user_params->USE_INTERPOLATION_TABLES) {
            zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points_SFR-1.0);

            dens_width = 1./((double)dens_Ninterp - 1.);
        }

        if(initialization_required) {

            ////////////////////////////    Create and fill interpolation tables to be used by Ts.c   /////////////////////////////

            if(user_params->USE_INTERPOLATION_TABLES) {

                if(flag_options->USE_MASS_DEPENDENT_ZETA) {

                    // generates an interpolation table for redshift
                    for (i=0; i<zpp_interp_points_SFR;i++) {
                        zpp_interp_table[i] = determine_zpp_min + zpp_bin_width*(float)i;
                    }

                    /* initialise interpolation of the mean collapse fraction for global reionization.*/
                    if (!flag_options->USE_MINI_HALOS){
                        initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                                 astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->ALPHA_ESC,
                                                 astro_params->F_STAR10, astro_params->F_ESC10);

                        initialise_SFRD_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                              astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->F_STAR10);
                    }
                    else{
                        initialise_Nion_Ts_spline_MINI(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                                      astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI, astro_params->ALPHA_ESC, astro_params->F_STAR10,
                                                      astro_params->F_ESC10, astro_params->F_STAR7_MINI, astro_params->F_ESC7_MINI);

                        initialise_SFRD_spline_MINI(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                                   astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI, astro_params->F_STAR10, astro_params->F_STAR7_MINI);
                    }
                    interpolation_tables_allocated = true;
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
                    #pragma omp parallel shared(determine_zpp_min,determine_zpp_max,\
                                                Sigma_Tmin_grid,ST_over_PS_arg_grid,\
                                                mu_for_Ts,M_MIN,M_MIN_WDM) \
                                         private(i,zpp_grid) \
                                         num_threads(user_params->N_THREADS)
                    {
                        #pragma omp for
                        for(i=0;i<zpp_interp_points_SFR;i++) {
                            zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points_SFR-1.0);

                            if(flag_options->M_MIN_in_Mass) {
                                Sigma_Tmin_grid[i] = sigma_z0(fmaxf(M_MIN,  M_MIN_WDM));
                                ST_over_PS_arg_grid[i] = FgtrM_General(zpp_grid, fmaxf(M_MIN,  M_MIN_WDM));
                            }
                            else {
                                Sigma_Tmin_grid[i] = sigma_z0(fmaxf((float)TtoM(zpp_grid, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                                ST_over_PS_arg_grid[i] = FgtrM_General(zpp_grid, fmaxf((float)TtoM(zpp_grid, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                            }
                        }
                    }

                    // Create the interpolation tables for the derivative of the collapsed fraction and the collapse fraction itself
                    #pragma omp parallel shared(fcoll_R_grid,dfcoll_dz_grid,Sigma_Tmin_grid,\
                                            determine_zpp_min,determine_zpp_max,\
                                            grid_dens,sigma_atR) \
                                     private(ii,i,j,zpp_grid,grid_sigmaTmin,grid_dens_val) \
                                     num_threads(user_params->N_THREADS)
                    {
                        #pragma omp for
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
                if (flag_options->USE_MINI_HALOS){
                    Mcrit_atom_interp_table[R_ct] = atomic_cooling_threshold(zpp);
                }
            }

            if(user_params->USE_INTERPOLATION_TABLES) {
                if (!flag_options->USE_MINI_HALOS){
                    initialise_SFRD_Conditional_table(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,
                                                     max_densities,zpp_growth,R_values, astro_params->M_TURN,
                                                     astro_params->ALPHA_STAR, astro_params->F_STAR10, user_params->FAST_FCOLL_TABLES);
                }
                else{
                    initialise_SFRD_Conditional_table_MINI(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,
                                                          max_densities,zpp_growth,R_values,Mcrit_atom_interp_table,
                                                          astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI, astro_params->F_STAR10,
                                                          astro_params->F_STAR7_MINI, user_params->FAST_FCOLL_TABLES);
                }
            }
        }

        LOG_SUPER_DEBUG("Initialised SFRD table");

        zp = redshift;
        prev_zp = prev_redshift;

        if(flag_options->USE_MASS_DEPENDENT_ZETA) {

            if(user_params->USE_INTERPOLATION_TABLES) {
                redshift_int_Nion_z = (int)floor( ( zp - determine_zpp_min )/zpp_bin_width );

                if(redshift_int_Nion_z < 0 || (redshift_int_Nion_z + 1) > (zpp_interp_points_SFR - 1)) {
                    LOG_ERROR("I have overstepped my allocated memory for the interpolation table Nion_z_val");
//                    Throw(ParameterError);
                    Throw(TableEvaluationError);
                }

                redshift_table_Nion_z = determine_zpp_min + zpp_bin_width*(float)redshift_int_Nion_z;

                Splined_Fcollzp_mean = Nion_z_val[redshift_int_Nion_z] + \
                        ( zp - redshift_table_Nion_z )*( Nion_z_val[redshift_int_Nion_z+1] - Nion_z_val[redshift_int_Nion_z] )/(zpp_bin_width);
            }
            else {

                if(flag_options->USE_MINI_HALOS) {
                    Splined_Fcollzp_mean = Nion_General(zp, global_params.M_MIN_INTEGRAL, atomic_cooling_threshold(zp), astro_params->ALPHA_STAR, astro_params->ALPHA_ESC,
                                                        astro_params->F_STAR10, astro_params->F_ESC10, Mlim_Fstar, Mlim_Fesc);
                }
                else {
                    Splined_Fcollzp_mean = Nion_General(zp, M_MIN, astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->ALPHA_ESC,
                                                    astro_params->F_STAR10, astro_params->F_ESC10, Mlim_Fstar, Mlim_Fesc);
                }
            }

            if (flag_options->USE_MINI_HALOS){
                log10_Mcrit_mol = log10(lyman_werner_threshold(zp, 0., 0.,astro_params));
                log10_Mcrit_LW_ave = 0.0;
                #pragma omp parallel shared(log10_Mcrit_LW_unfiltered,previous_spin_temp,zp)\
                                     private(i,j,k,curr_vcb) \
                                     num_threads(user_params->N_THREADS)
                {
                    #pragma omp for reduction(+:log10_Mcrit_LW_ave)
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<HII_D_PARA; k++){

                                if (flag_options->FIX_VCB_AVG){ //with this flag we ignore reading vcb box
                                    curr_vcb = global_params.VAVG;
                                }
                                else{
                                    if(user_params->USE_RELATIVE_VELOCITIES){
                                    curr_vcb = ini_boxes->lowres_vcb[HII_R_INDEX(i,j,k)];
                                    }
                                    else{ //set vcb to a constant, either zero or vavg.
                                    curr_vcb = 0.0;
                                    }
                                }

                                *((float *)log10_Mcrit_LW_unfiltered + HII_R_FFT_INDEX(i,j,k)) = \
                                              log10(lyman_werner_threshold(zp, previous_spin_temp->J_21_LW_box[HII_R_INDEX(i,j,k)],
                                              curr_vcb, astro_params) );

                                // This only accounts for effect 3 (only on minihaloes).
                                // Effects 1+2 also affects ACGs, but is included only on average.
                                log10_Mcrit_LW_ave += *((float *)log10_Mcrit_LW_unfiltered + HII_R_FFT_INDEX(i,j,k));
                            }
                        }
                    }
                }
                log10_Mcrit_LW_ave /= (double)HII_TOT_NUM_PIXELS;

                // NEED TO FILTER Mcrit_LW!!!
                /*** Transform unfiltered box to k-space to prepare for filtering ***/
                dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, log10_Mcrit_LW_unfiltered);

                #pragma omp parallel shared(log10_Mcrit_LW_unfiltered) \
                                     private(ct) \
                                     num_threads(user_params->N_THREADS)
                {
                    #pragma omp for
                    for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++) {
                        log10_Mcrit_LW_unfiltered[ct] /= (float)HII_TOT_NUM_PIXELS;
                    }
                }

                if(user_params->USE_INTERPOLATION_TABLES) {
                    log10_Mcrit_LW_ave_int_Nion_z = (int)floor( ( log10_Mcrit_LW_ave - LOG10_MTURN_MIN) / LOG10_MTURN_INT);
                    log10_Mcrit_LW_ave_table_Nion_z = LOG10_MTURN_MIN + LOG10_MTURN_INT * (float)log10_Mcrit_LW_ave_int_Nion_z;

                    Splined_Fcollzp_mean_MINI_left = Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_Nion_z] + \
                                                ( zp - redshift_table_Nion_z ) / (zpp_bin_width)*\
                                                  ( Nion_z_val_MINI[redshift_int_Nion_z + 1 + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_Nion_z] -\
                                                    Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_Nion_z] );
                    Splined_Fcollzp_mean_MINI_right = Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_Nion_z+1)] + \
                                                ( zp - redshift_table_Nion_z ) / (zpp_bin_width)*\
                                                  ( Nion_z_val_MINI[redshift_int_Nion_z + 1 + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_Nion_z+1)] -\
                                                    Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_Nion_z+1)] );
                    Splined_Fcollzp_mean_MINI = Splined_Fcollzp_mean_MINI_left + \
                                (log10_Mcrit_LW_ave - log10_Mcrit_LW_ave_table_Nion_z) / LOG10_MTURN_INT * (Splined_Fcollzp_mean_MINI_right - Splined_Fcollzp_mean_MINI_left);
                }
                else {
                    Splined_Fcollzp_mean_MINI = Nion_General_MINI(zp, global_params.M_MIN_INTEGRAL, pow(10.,log10_Mcrit_LW_ave), atomic_cooling_threshold(zp),
                                                                  astro_params->ALPHA_STAR_MINI, astro_params->ALPHA_ESC, astro_params->F_STAR7_MINI,
                                                                  astro_params->F_ESC7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI);
                }
            }
            else{
                Splined_Fcollzp_mean_MINI = 0;
            }

            if ( ( Splined_Fcollzp_mean < 1e-15 ) && (Splined_Fcollzp_mean_MINI < 1e-15))
                NO_LIGHT = 1;
            else
                NO_LIGHT = 0;

            filling_factor_of_HI_zp = 1 - ( ION_EFF_FACTOR * Splined_Fcollzp_mean + ION_EFF_FACTOR_MINI * Splined_Fcollzp_mean_MINI )/ (1.0 - x_e_ave);
        } else {
            if(flag_options->M_MIN_in_Mass) {

                if (FgtrM(zp, fmaxf(M_MIN,  M_MIN_WDM)) < 1e-15 )
                    NO_LIGHT = 1;
                else
                    NO_LIGHT = 0;

                M_MIN_at_zp = M_MIN;
            } else {

                if (FgtrM(zp, fmaxf((float)TtoM(zp, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM)) < 1e-15 )
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
            if (flag_options->USE_MINI_HALOS){
                Mcrit_atom_interp_table[R_ct] = atomic_cooling_threshold(zpp);
            }

            fcoll_R_array[R_ct] = 0.0;

            // let's now normalize the total collapse fraction so that the mean is the
            // Sheth-Torman collapse fraction
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation

                if(user_params->USE_INTERPOLATION_TABLES) {
                    redshift_int_SFRD = (int)floor( ( zpp - determine_zpp_min )/zpp_bin_width );

                    if(redshift_int_SFRD < 0 || (redshift_int_SFRD + 1) > (zpp_interp_points_SFR - 1)) {
                        LOG_ERROR("I have overstepped my allocated memory for the interpolation table SFRD_val");
//                        Throw(ParameterError);
                        Throw(TableEvaluationError);
                    }

                    redshift_table_SFRD = determine_zpp_min + zpp_bin_width*(float)redshift_int_SFRD;

                    Splined_SFRD_zpp = SFRD_val[redshift_int_SFRD] + \
                                    ( zpp - redshift_table_SFRD )*( SFRD_val[redshift_int_SFRD+1] - SFRD_val[redshift_int_SFRD] )/(zpp_bin_width);

                    ST_over_PS[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve);
                    ST_over_PS[R_ct] *= Splined_SFRD_zpp;
                }
                else {
                    ST_over_PS[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve); // Multiplied by Nion later
                }

                if(flag_options->USE_MINI_HALOS){
                    memcpy(log10_Mcrit_LW_filtered, log10_Mcrit_LW_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                    if (R_ct > 0){// don't filter on cell size
                        filter_box(log10_Mcrit_LW_filtered, 1, global_params.HEAT_FILTER, R_values[R_ct]);
                    }
                    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, log10_Mcrit_LW_filtered);

                    log10_Mcrit_LW_ave = 0; //recalculate it at this filtering scale
                    #pragma omp parallel shared(log10_Mcrit_LW,log10_Mcrit_LW_filtered,log10_Mcrit_mol) \
                                         private(i,j,k) num_threads(user_params->N_THREADS)
                    {
                        #pragma omp for reduction(+:log10_Mcrit_LW_ave)
                        for (i=0; i<user_params->HII_DIM; i++){
                            for (j=0; j<user_params->HII_DIM; j++){
                                for (k=0; k<HII_D_PARA; k++){
                                    log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = *((float *) log10_Mcrit_LW_filtered + HII_R_FFT_INDEX(i,j,k));
                                    if(log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] < log10_Mcrit_mol)
                                        log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = log10_Mcrit_mol;
                                    if (log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] > LOG10_MTURN_MAX)
                                        log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = LOG10_MTURN_MAX;
                                    log10_Mcrit_LW_ave += log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)];
                                }
                            }
                        }
                    }
                    log10_Mcrit_LW_ave /= (double)HII_TOT_NUM_PIXELS;

                    log10_Mcrit_LW_ave_list[R_ct] = log10_Mcrit_LW_ave;

                    if(user_params->USE_INTERPOLATION_TABLES) {
                        log10_Mcrit_LW_ave_int_SFRD = (int)floor( ( log10_Mcrit_LW_ave - LOG10_MTURN_MIN) / LOG10_MTURN_INT);
                        log10_Mcrit_LW_ave_table_SFRD = LOG10_MTURN_MIN + LOG10_MTURN_INT * (float)log10_Mcrit_LW_ave_int_SFRD;

                        Splined_SFRD_zpp_MINI_left = SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_SFRD] + \
                                                ( zpp - redshift_table_SFRD ) / (zpp_bin_width)*\
                                                  ( SFRD_val_MINI[redshift_int_SFRD + 1 + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_SFRD] -\
                                                    SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_SFRD] );
                        Splined_SFRD_zpp_MINI_right = SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_SFRD+1)] + \
                                                ( zpp - redshift_table_SFRD ) / (zpp_bin_width)*\
                                                  ( SFRD_val_MINI[redshift_int_SFRD + 1 + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_SFRD+1)] -\
                                                    SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_SFRD+1)] );
                        Splined_SFRD_zpp_MINI = Splined_SFRD_zpp_MINI_left + \
                            (log10_Mcrit_LW_ave - log10_Mcrit_LW_ave_table_SFRD) / LOG10_MTURN_INT * (Splined_SFRD_zpp_MINI_right - Splined_SFRD_zpp_MINI_left);

                        ST_over_PS_MINI[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve);
                        ST_over_PS_MINI[R_ct] *= Splined_SFRD_zpp_MINI;
                    }
                    else {
                        ST_over_PS_MINI[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve); // Multiplied by Nion later
                    }
                }

                SFR_timescale_factor[R_ct] = hubble(zpp)*fabs(dtdz(zpp));

            }
            else {

                if(user_params->USE_INTERPOLATION_TABLES) {
                    // Determining values for the evaluating the interpolation table
                    zpp_gridpoint1_int = (int)floor((zpp - determine_zpp_min)/zpp_bin_width);
                    zpp_gridpoint2_int = zpp_gridpoint1_int + 1;

                    if(zpp_gridpoint1_int < 0 || (zpp_gridpoint1_int + 1) > (zpp_interp_points_SFR - 1)) {
                        LOG_ERROR("I have overstepped my allocated memory for the interpolation table fcoll_R_grid");
//                        Throw(ParameterError);
                        Throw(TableEvaluationError);
                    }

                    zpp_gridpoint1 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint1_int;
                    zpp_gridpoint2 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint2_int;

                    grad1 = ( zpp_gridpoint2 - zpp )/( zpp_gridpoint2 - zpp_gridpoint1 );
                    grad2 = ( zpp - zpp_gridpoint1 )/( zpp_gridpoint2 - zpp_gridpoint1 );

                    sigma_Tmin[R_ct] = Sigma_Tmin_grid[zpp_gridpoint1_int] + grad2*( Sigma_Tmin_grid[zpp_gridpoint2_int] - Sigma_Tmin_grid[zpp_gridpoint1_int] );

                    // Evaluating the interpolation table for the collapse fraction and its derivative
                    for(i=0;i<(dens_Ninterp-1);i++) {
                        dens_grad = 1./( density_gridpoints[i+1][R_ct] - density_gridpoints[i][R_ct] );

                        fcoll_interp1[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + \
                                              ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                        fcoll_interp2[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + \
                                              ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;

                        dfcoll_interp1[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + \
                                               ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                        dfcoll_interp2[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + \
                                               ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;

                    }

                    // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation
                    ST_over_PS[R_ct] = dzpp_for_evolve * pow(1+zpp, -(astro_params->X_RAY_SPEC_INDEX));
                    ST_over_PS[R_ct] *= ( ST_over_PS_arg_grid[zpp_gridpoint1_int] + \
                                         grad2*( ST_over_PS_arg_grid[zpp_gridpoint2_int] - ST_over_PS_arg_grid[zpp_gridpoint1_int] ) );
                }
                else {
                    if(flag_options->M_MIN_in_Mass) {
                        sigma_Tmin[R_ct] = sigma_z0(fmaxf(M_MIN, M_MIN_WDM));
                    }
                    else {
                        sigma_Tmin[R_ct] = sigma_z0(fmaxf((float)TtoM(zpp, astro_params->X_RAY_Tvir_MIN, mu_for_Ts), M_MIN_WDM));
                    }

                    ST_over_PS[R_ct] = dzpp_for_evolve * pow(1+zpp, -(astro_params->X_RAY_SPEC_INDEX));
                }

            }

            if(user_params->USE_INTERPOLATION_TABLES) {
                if(flag_options->USE_MINI_HALOS){
                    lower_int_limit = fmax(nu_tau_one_MINI(zp, zpp, x_e_ave, filling_factor_of_HI_zp,
                                                              log10_Mcrit_LW_ave,LOG10_MTURN_INT), (astro_params->NU_X_THRESH)*NU_over_EV);
                }
                else{
                    lower_int_limit = fmax(nu_tau_one(zp, zpp, x_e_ave, filling_factor_of_HI_zp), (astro_params->NU_X_THRESH)*NU_over_EV);
                }

                if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly

                // set up frequency integral table for later interpolation for the cell's x_e value
                for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                    freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                    freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                    freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);

                    if(isfinite(freq_int_heat_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_ion_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_lya_tbl[x_e_ct][R_ct])==0) {
                        LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
//                        Throw(ParameterError);
                        Throw(TableGenerationError);
                    }
                }
            }

            // and create the sum over Lya transitions from direct Lyn flux
            sum_lyn[R_ct] = 0;
            //Lya flux for Lya heating
            if (flag_options->USE_LYA_HEATING){
                sum_ly2[R_ct] = 0;
                sum_lynto2[R_ct] = 0;
            }

            if (flag_options->USE_MINI_HALOS){
                sum_lyn_MINI[R_ct] = 0;
                sum_lyLWn[R_ct] = 0;
                sum_lyLWn_MINI[R_ct] = 0;

                if (flag_options->USE_LYA_HEATING) {
                    sum_ly2_MINI[R_ct] = 0;
                    sum_lynto2_MINI[R_ct] = 0;
                }
            }
            for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                if (zpp > zmax(zp, n_ct))
                    continue;

                nuprime = nu_n(n_ct)*(1+zpp)/(1.0+zp);
                if (flag_options->USE_MINI_HALOS){

                    //Separate out the continuum and injected flux contributions
                    if (flag_options->USE_LYA_HEATING){
                        ly2_store = 0.;
                        ly2_store_MINI = 0.;
                        lynto2_store = 0.;
                        lynto2_store_MINI = 0.;

                        if (n_ct==2){
                            ly2_store = frecycle(n_ct) * spectral_emissivity(nuprime, 0,2);
                            sum_ly2[R_ct] += ly2_store;

                            ly2_store_MINI = frecycle(n_ct) * spectral_emissivity(nuprime, 0,3);
                            sum_ly2_MINI[R_ct] += ly2_store_MINI;
                        }

                        if (n_ct>=3){
                            lynto2_store = frecycle(n_ct) * spectral_emissivity(nuprime, 0,2);
                            sum_lynto2[R_ct] += lynto2_store;

                            lynto2_store_MINI = frecycle(n_ct) * spectral_emissivity(nuprime, 0,3);
                            sum_lynto2_MINI[R_ct] += lynto2_store_MINI;
                        }

                        sum_lyn[R_ct] += (ly2_store + lynto2_store);
                        sum_lyn_MINI[R_ct] += (ly2_store_MINI + lynto2_store_MINI);
                    }
                    else{
                        sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
                        sum_lyn_MINI[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);
                    }

                    if (nuprime < NU_LW_THRESH / NUIONIZATION)
                        nuprime = NU_LW_THRESH / NUIONIZATION;
                    if (nuprime >= nu_n(n_ct + 1))
                        continue;
                    sum_lyLWn[R_ct]  += (1. - astro_params->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                    sum_lyLWn_MINI[R_ct] += (1. - astro_params->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
                }
                else{
                    //Separate out the continuum and injected flux contributions
                    if (flag_options->USE_LYA_HEATING){
                        ly2_store = 0.;
                        lynto2_store = 0.;

                        if (n_ct==2){
                            ly2_store = frecycle(n_ct) * spectral_emissivity(nuprime, 0, global_params.Pop);
                            sum_ly2[R_ct] += ly2_store;
                        }
                        if (n_ct>=3){
                            lynto2_store = frecycle(n_ct) * spectral_emissivity(nuprime, 0, global_params.Pop);
                            sum_lynto2[R_ct] += lynto2_store;
                        }

                        sum_lyn[R_ct] += (ly2_store + lynto2_store);

                    }
                    else{
                        sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, global_params.Pop);
                    }
                }
            }

            // Find if we need to add a partial contribution to a radii to avoid kinks in the Lyman-alpha flux
            // As we look at discrete radii (light-cone redshift, zpp) we can have two radii where one has a
            // contribution and the next (larger) radii has no contribution. However, if the number of filtering
            // steps were infinitely large, we would have contributions between these two discrete radii
            // Thus, this aims to add a weighted contribution to the first radii where this occurs to smooth out
            // kinks in the average Lyman-alpha flux.

            // Note: We do not apply this correction to the LW background as it is unaffected by this. It is only
            // the Lyn contribution that experiences the kink. Applying this correction to LW introduces kinks
            // into the otherwise smooth quantity
            if(R_ct > 1 && sum_lyn[R_ct]==0.0 && sum_lyn[R_ct-1]>0. && first_radii) {

                // The current zpp for which we are getting zero contribution
                trial_zpp_max = (prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp)+prev_zpp)*0.5;
                // The zpp for the previous radius for which we had a non-zero contribution
                trial_zpp_min = (zpp_edge[R_ct-2] - (R_values[R_ct-1] - R_values[R_ct-2])*CMperMPC / drdz(zpp_edge[R_ct-2])+zpp_edge[R_ct-2])*0.5;

                // Split the previous radii and current radii into n_pts_radii smaller radii (redshift) to have fine control of where
                // it transitions from zero to non-zero
                // This is a coarse approximation as it assumes that the linear sampling is a good representation of the different
                // volumes of the shells (from different radii).
                for(ii=0;ii<n_pts_radii;ii++) {
                    trial_zpp = trial_zpp_min + (trial_zpp_max - trial_zpp_min)*(float)ii/((float)n_pts_radii-1.);

                    counter = 0;
                    for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                        if (trial_zpp > zmax(zp, n_ct))
                            continue;

                        counter += 1;
                    }
                    if(counter==0&&first_zero) {
                        first_zero = false;
                        weight = (float)ii/(float)n_pts_radii;
                    }
                }

                // Now add a non-zero contribution to the previously zero contribution
                // The amount is the weight, multplied by the contribution from the previous radii
                sum_lyn[R_ct] = weight * sum_lyn[R_ct-1];
                //Apply same weight for sum_ly2 and sum_lynto2
                if (flag_options->USE_LYA_HEATING){
                    sum_ly2[R_ct] = weight * sum_ly2[R_ct-1];
                    sum_lynto2[R_ct] = weight * sum_lynto2[R_ct-1];
                }
                if (flag_options->USE_MINI_HALOS){
                    sum_lyn_MINI[R_ct] = weight * sum_lyn_MINI[R_ct-1];

                    if (flag_options->USE_LYA_HEATING){
                        sum_ly2_MINI[R_ct] = weight * sum_ly2_MINI[R_ct-1];
                        sum_lynto2_MINI[R_ct] = weight * sum_lynto2_MINI[R_ct-1];
                    }
                }
                first_radii = false;
            }


        } // end loop over R_ct filter steps


        // Throw the time intensive full calculations into a multiprocessing loop to get them evaluated faster
        if(!user_params->USE_INTERPOLATION_TABLES) {

            #pragma omp parallel shared(ST_over_PS,zpp_for_evolve_list,log10_Mcrit_LW_ave_list,\
                                        Mcrit_atom_interp_table,M_MIN,Mlim_Fstar,Mlim_Fstar_MINI,x_e_ave,\
                                        filling_factor_of_HI_zp,x_int_XHII,freq_int_heat_tbl,freq_int_ion_tbl,freq_int_lya_tbl,LOG10_MTURN_INT) \
                                 private(R_ct,x_e_ct,lower_int_limit)  \
                                 num_threads(user_params->N_THREADS)
            {
                #pragma omp for
                for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                        if(flag_options->USE_MINI_HALOS){
                            ST_over_PS[R_ct] *= Nion_General(zpp_for_evolve_list[R_ct], global_params.M_MIN_INTEGRAL, Mcrit_atom_interp_table[R_ct],
                                                             astro_params->ALPHA_STAR, 0., astro_params->F_STAR10, 1.,Mlim_Fstar,0.);
                            ST_over_PS_MINI[R_ct] *= Nion_General_MINI(zpp_for_evolve_list[R_ct], global_params.M_MIN_INTEGRAL, pow(10.,log10_Mcrit_LW_ave_list[R_ct]),
                                                                Mcrit_atom_interp_table[R_ct], astro_params->ALPHA_STAR_MINI, 0.,
                                                                astro_params->F_STAR7_MINI, 1.,Mlim_Fstar_MINI,0.);
                        }
                        else {
                            ST_over_PS[R_ct] *= Nion_General(zpp_for_evolve_list[R_ct], M_MIN, astro_params->M_TURN, astro_params->ALPHA_STAR, 0., astro_params->F_STAR10, 1.,Mlim_Fstar,0.);
                        }
                    }
                    else {
                        if(flag_options->M_MIN_in_Mass) {
                            ST_over_PS[R_ct] *= FgtrM_General(zpp_for_evolve_list[R_ct], fmaxf(M_MIN, M_MIN_WDM));
                        }
                        else {
                            ST_over_PS[R_ct] *= FgtrM_General(zpp_for_evolve_list[R_ct], fmaxf((float)TtoM(zpp_for_evolve_list[R_ct], astro_params->X_RAY_Tvir_MIN, mu_for_Ts), M_MIN_WDM));
                        }
                    }

                    if(flag_options->USE_MINI_HALOS){
                        lower_int_limit = fmax(nu_tau_one_MINI(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp,
                                                               log10_Mcrit_LW_ave_list[R_ct],LOG10_MTURN_INT), (astro_params->NU_X_THRESH)*NU_over_EV);
                    }
                    else{
                        lower_int_limit = fmax(nu_tau_one(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp), (astro_params->NU_X_THRESH)*NU_over_EV);
                    }

                    if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly

                    // set up frequency integral table for later interpolation for the cell's x_e value
                    for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                        freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                        freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                        freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);
                    }
                }
            }

            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                    if(isfinite(freq_int_heat_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_ion_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_lya_tbl[x_e_ct][R_ct])==0) {
                        LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
                        Throw(TableGenerationError);
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("finished looping over R_ct filter steps");

        if(user_params->USE_INTERPOLATION_TABLES) {
            fcoll_interp_high_min = global_params.CRIT_DENS_TRANSITION;
            fcoll_interp_high_bin_width = 1./((float)NSFR_high-1.)*(Deltac - fcoll_interp_high_min);
            fcoll_interp_high_bin_width_inv = 1./fcoll_interp_high_bin_width;
        }

        // Calculate fcoll for each smoothing radius
        if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
            if(user_params->N_THREADS==1) {
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                    for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){

                        if(user_params->USE_INTERPOLATION_TABLES) {
                            if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                                table_int_boundexceeded = 1;
                            }

                            fcoll_R_array[R_ct] += ( fcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                            ( density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct] ) + \
                                            fcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                            ( delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct] ) );
                        }
                        else {
                            fcoll_R_array[R_ct] += sigmaparam_FgtrM_bias(zpp_for_evolve_list[R_ct],sigma_Tmin[R_ct],delNL0_rev[box_ct][R_ct],sigma_atR[R_ct]);
                        }
                    }
                    if(table_int_boundexceeded==1) {
                        LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of fcoll");
                        Throw(TableEvaluationError);
                    }
                }
            }
            else {

                for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                    fcoll_R_for_reduction = 0.;

                    #pragma omp parallel shared(dens_grid_int_vals,R_ct,fcoll_interp1,density_gridpoints,\
                                                delNL0_rev,fcoll_interp2,table_int_boundexceeded_threaded,\
                                                zpp_for_evolve_list,sigma_Tmin,sigma_atR) \
                                         private(box_ct) num_threads(user_params->N_THREADS)
                    {
                        #pragma omp for reduction(+:fcoll_R_for_reduction)
                        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                            if(user_params->USE_INTERPOLATION_TABLES) {
                                if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                                    table_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                }

                                fcoll_R_for_reduction += ( fcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                                      ( density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct] ) + \
                                                      fcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                                      ( delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct] ) );
                            }
                            else {
                                fcoll_R_for_reduction += sigmaparam_FgtrM_bias(zpp_for_evolve_list[R_ct],sigma_Tmin[R_ct],delNL0_rev[box_ct][R_ct],sigma_atR[R_ct]);
                            }
                        }
                    }
                    fcoll_R_array[R_ct] = fcoll_R_for_reduction;
                }
                for(i=0;i<user_params->N_THREADS;i++) {
                    if(table_int_boundexceeded_threaded[i]==1) {
                        LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of fcoll");
                        Throw(TableEvaluationError);
                    }
                }
            }

            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                ST_over_PS[R_ct] = ST_over_PS[R_ct]/(fcoll_R_array[R_ct]/(double)HII_TOT_NUM_PIXELS);
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
            Luminosity_converstion_factor = pow( (global_params.NU_X_BAND_MAX)*NU_over_EV , 1. - (astro_params->X_RAY_SPEC_INDEX) ) - \
                                            pow( (astro_params->NU_X_THRESH)*NU_over_EV , 1. - (astro_params->X_RAY_SPEC_INDEX) ) ;
            Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
            Luminosity_converstion_factor *= pow( (astro_params->NU_X_THRESH)*NU_over_EV, - (astro_params->X_RAY_SPEC_INDEX) )*\
                                            (1 - (astro_params->X_RAY_SPEC_INDEX));
        }
        // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the definition of Luminosity)
        Luminosity_converstion_factor *= (3.1556226e7)/(hplank);

        // Leave the original 21cmFAST code for reference. Refer to Greig & Mesinger (2017) for the new parameterisation.
        const_zp_prefactor = ( (astro_params->L_X) * Luminosity_converstion_factor ) / ((astro_params->NU_X_THRESH)*NU_over_EV) \
                                * C * astro_params->F_STAR10 * cosmo_params->OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, astro_params->X_RAY_SPEC_INDEX+3);
        //          This line below is kept purely for reference w.r.t to the original 21cmFAST
        //            const_zp_prefactor = ZETA_X * X_RAY_SPEC_INDEX / NU_X_THRESH * C * F_STAR * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, X_RAY_SPEC_INDEX+3);

        if (flag_options->USE_MINI_HALOS){
            // do the same for MINI
            const_zp_prefactor_MINI = ( (astro_params->L_X_MINI) * Luminosity_converstion_factor ) / ((astro_params->NU_X_THRESH)*NU_over_EV) \
                                    * C * astro_params->F_STAR7_MINI * cosmo_params->OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, astro_params->X_RAY_SPEC_INDEX+3);
        }
        else{
            const_zp_prefactor_MINI = 0.;
        }
        //////////////////////////////  LOOP THROUGH BOX //////////////////////////////

        J_alpha_ave = xalpha_ave = Xheat_ave = Xion_ave = 0.;
        J_alpha_ave_MINI = J_LW_ave = J_LW_ave_MINI = Xheat_ave_MINI = 0.;

        // Extra pre-factors etc. are defined here, as they are independent of the density field,
        // and only have to be computed once per z' or R_ct, rather than each box_ct
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){

            zpp_integrand = ( pow(1+zp,2)*(1+zpp_for_evolve_list[R_ct]) )/( pow(1+zpp_for_evolve_list[R_ct], -(astro_params->X_RAY_SPEC_INDEX)) );
            dstarlya_dt_prefactor[R_ct]  = zpp_integrand * sum_lyn[R_ct];
            //Lya flux prefactors for Lya heating
            if (flag_options->USE_LYA_HEATING){
                dstarlya_cont_dt_prefactor[R_ct]  = zpp_integrand * sum_ly2[R_ct];
                dstarlya_inj_dt_prefactor[R_ct]  = zpp_integrand * sum_lynto2[R_ct];
            }
            if (flag_options->USE_MINI_HALOS){
                dstarlya_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lyn_MINI[R_ct];
                dstarlyLW_dt_prefactor[R_ct]  = zpp_integrand * sum_lyLWn[R_ct];
                dstarlyLW_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lyLWn_MINI[R_ct];

                if (flag_options->USE_LYA_HEATING){
                    dstarlya_cont_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_ly2_MINI[R_ct];
                    dstarlya_inj_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lynto2_MINI[R_ct];
                }
            }
        }

        // Required quantities for calculating the IGM spin temperature
        // Note: These used to be determined in evolveInt (and other functions). But I moved them all here, into a single location.
        Trad_fast = T_cmb*(1.0+zp);
        Trad_fast_inv = 1.0/Trad_fast;
        TS_prefactor = pow(1.0e-7*(1.342881e-7 / hubble(zp))*No*pow(1+zp,3),1./3.);

        gamma_alpha = f_alpha*pow(Ly_alpha_HZ*e_charge/(C/10.),2.); // division of C/10. is converstion of electric charge from esu to coulomb
        gamma_alpha /= 6.*(m_e/1000.)*pow(C/100.,3.)*vac_perm; //division by 1000. to convert gram to kg and division by 100. to convert cm to m

        xa_tilde_prefactor = 8.*PI*pow(Ly_alpha_ANG*1.e-8,2.)*gamma_alpha*T21; //1e-8 converts angstrom to cm.
        xa_tilde_prefactor /= 9.*A10_HYPERFINE*T_cmb*(1.0+zp);

        xc_inverse =  pow(1.0+zp,3.0)*T21/( Trad_fast*A10_HYPERFINE );

        dcomp_dzp_prefactor = (-1.51e-4)/(hubble(zp)/Ho)/(cosmo_params->hlittle)*pow(Trad_fast,4.0)/(1.0+zp);

        prefactor_1 = N_b0 * pow(1+zp, 3);
        prefactor_2 = astro_params->F_STAR10 * C * N_b0 / FOURPI;
        prefactor_2_MINI = astro_params->F_STAR7_MINI * C * N_b0 / FOURPI;

        x_e_ave = 0; Tk_ave = 0; Ts_ave = 0;

        // Note: I have removed the call to evolveInt, as is default in the original Ts.c.
        // Removal of evolveInt and moving that computation below, removes unneccesary repeated computations
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

            #pragma omp parallel shared(del_fcoll_Rct,dxheat_dt_box,dxion_source_dt_box,dxlya_dt_box,dstarlya_dt_box,previous_spin_temp,\
                            x_int_XHII,m_xHII_low_box,inverse_val_box,inverse_diff,dstarlyLW_dt_box,dstarlyLW_dt_box_MINI,\
                            dxheat_dt_box_MINI,dxion_source_dt_box_MINI,dxlya_dt_box_MINI,dstarlya_dt_box_MINI,\
                            dstarlya_cont_dt_box,dstarlya_inj_dt_box,dstarlya_cont_dt_box_MINI,dstarlya_inj_dt_box_MINI) \
                    private(box_ct,xHII_call) num_threads(user_params->N_THREADS)
            {
                #pragma omp for
                for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                    del_fcoll_Rct[box_ct] = 0.;

                    dxheat_dt_box[box_ct] = 0.;
                    dxion_source_dt_box[box_ct] = 0.;
                    dxlya_dt_box[box_ct] = 0.;
                    dstarlya_dt_box[box_ct] = 0.;

                    //Initialize Lya flux for Lya heating
                    if (flag_options->USE_LYA_HEATING){
                        dstarlya_cont_dt_box[box_ct] = 0.;
                        dstarlya_inj_dt_box[box_ct] = 0.;
                        }

                    if (flag_options->USE_MINI_HALOS){
                        dstarlyLW_dt_box[box_ct] = 0.;
                        dstarlyLW_dt_box_MINI[box_ct] = 0.;
                        dxheat_dt_box_MINI[box_ct] = 0.;
                        dxion_source_dt_box_MINI[box_ct] = 0.;
                        dxlya_dt_box_MINI[box_ct] = 0.;
                        dstarlya_dt_box_MINI[box_ct] = 0.;

                        if (flag_options->USE_LYA_HEATING){
                          dstarlya_cont_dt_box_MINI[box_ct] = 0.;
                          dstarlya_inj_dt_box_MINI[box_ct] = 0.;
                        }
                    }

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
            }

            for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){

                if(!user_params->USE_INTERPOLATION_TABLES) {
                    Mmax = RtoM(R_values[R_ct]);
                    sigmaMmax = sigma_z0(Mmax);
                }

                if(user_params->USE_INTERPOLATION_TABLES) {
                    if( min_densities[R_ct]*zpp_growth[R_ct] <= -1.) {
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
                }

                ave_fcoll = ave_fcoll_inv = 0.0;
                ave_fcoll_MINI = ave_fcoll_inv_MINI = 0.0;

                // If we are minimising memory usage, then we must smooth the box again
                // It's slower this way, but notably more memory efficient
                if(user_params->MINIMIZE_MEMORY) {

                    // copy over unfiltered box
                    memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                    if (R_ct > 0){ // don't filter on cell size
                        filter_box(box, 1, global_params.HEAT_FILTER, R_values[R_ct]);
                    }
                    // now fft back to real space
                    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box);
                    LOG_ULTRA_DEBUG("Executed FFT for R=%f", R_values[R_ct]);

                    // copy over the values
                    #pragma omp parallel shared(box,inverse_growth_factor_z,delNL0) \
                                         private(i,j,k,curr_delNL0) \
                                         num_threads(user_params->N_THREADS)
                    {
                        #pragma omp for
                        for (i=0;i<user_params->HII_DIM; i++){
                            for (j=0;j<user_params->HII_DIM; j++){
                                for (k=0;k<HII_D_PARA; k++){
                                    curr_delNL0 = *((float *)box + HII_R_FFT_INDEX(i,j,k));

                                    if (curr_delNL0 <= -1){ // correct for aliasing in the filtering step
                                        curr_delNL0 = -1+FRACT_FLOAT_ERR;
                                    }

                                    // and linearly extrapolate to z=0
                                    curr_delNL0 *= inverse_growth_factor_z;

                                    // Because we are FFT'ing again, just be careful that any rounding errors
                                    // don't cause the densities to exceed the bounds of the interpolation tables.
                                    if(user_params->USE_INTERPOLATION_TABLES) {
                                        if(curr_delNL0 > max_densities[R_ct]) {
                                            curr_delNL0 = max_densities[R_ct];
                                        }
                                        if(curr_delNL0 < min_densities[R_ct]) {
                                            curr_delNL0 = min_densities[R_ct];
                                        }
                                    }

                                    delNL0[0][HII_R_INDEX(i,j,k)] = curr_delNL0;
                                }
                            }
                        }
                    }
                }

                #pragma omp parallel shared(delNL0,zpp_growth,SFRD_z_high_table,fcoll_interp_high_min,\
                                            fcoll_interp_high_bin_width_inv,log10_SFRD_z_low_table,\
                                            fcoll_int_boundexceeded_threaded,log10_Mcrit_LW,SFRD_z_high_table_MINI,\
                                            log10_SFRD_z_low_table_MINI,del_fcoll_Rct,del_fcoll_Rct_MINI,Mmax,\
                                            sigmaMmax,Mcrit_atom_interp_table,Mlim_Fstar,Mlim_Fstar_MINI) \
                                     private(box_ct,curr_dens,fcoll,dens_val,fcoll_int,log10_Mcrit_LW_val,\
                                             log10_Mcrit_LW_int,log10_Mcrit_LW_diff,fcoll_MINI_left,\
                                             fcoll_MINI_right,fcoll_MINI) \
                                     num_threads(user_params->N_THREADS)
                {
                    #pragma omp for reduction(+:ave_fcoll,ave_fcoll_MINI)
                    for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                        if(user_params->MINIMIZE_MEMORY) {
                            curr_dens = delNL0[0][box_ct]*zpp_growth[R_ct];
                        }
                        else {
                            curr_dens = delNL0[R_ct][box_ct]*zpp_growth[R_ct];
                        }

                        if (flag_options->USE_MINI_HALOS && user_params->USE_INTERPOLATION_TABLES){
                            log10_Mcrit_LW_val = ( log10_Mcrit_LW[R_ct][box_ct] - LOG10_MTURN_MIN) / LOG10_MTURN_INT;
                            log10_Mcrit_LW_int = (int)floorf( log10_Mcrit_LW_val );
                            log10_Mcrit_LW_diff = log10_Mcrit_LW_val - (float)log10_Mcrit_LW_int;
                        }
                        if (!NO_LIGHT){
                            // Now determine all the differentials for the heating/ionisation rate equations

                            if(user_params->USE_INTERPOLATION_TABLES) {

                                if (curr_dens < global_params.CRIT_DENS_TRANSITION){

                                    if (curr_dens <= -1.) {
                                        fcoll = 0;
                                        fcoll_MINI = 0;
                                    }
                                    else {
                                        dens_val = (log10f(curr_dens+1.) - fcoll_interp_min)*fcoll_interp_bin_width_inv;

                                        fcoll_int = (int)floorf( dens_val );

                                        if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_low - 1)) {
                                            if(fcoll_int==(NSFR_low - 1)) {
                                                if(fabs(curr_dens - global_params.CRIT_DENS_TRANSITION) < 1e-4) {
                                                    // There can be instances where the numerical rounding causes it to go in here,
                                                    // rather than the curr_dens > global_params.CRIT_DENS_TRANSITION case
                                                    // This checks for this, and calculates f_coll in this instance, rather than causing it to error
                                                    dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;

                                                    fcoll_int = (int)floorf( dens_val );

                                                    fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + \
                                                            SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );
                                                    if (flag_options->USE_MINI_HALOS){
                                                        fcoll_MINI_left = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high * log10_Mcrit_LW_int]*\
                                                                    ( 1. + (float)fcoll_int - dens_val ) +\
                                                                    SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high * log10_Mcrit_LW_int]*\
                                                                    ( dens_val - (float)fcoll_int );

                                                        fcoll_MINI_right = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high * (log10_Mcrit_LW_int + 1)]*\
                                                                    ( 1. + (float)fcoll_int - dens_val ) +\
                                                                    SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high * (log10_Mcrit_LW_int + 1)]*\
                                                                    ( dens_val - (float)fcoll_int );

                                                        fcoll_MINI = fcoll_MINI_left * (1. - log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                                    }
                                                }
                                                else {

                                                    fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int];
                                                    fcoll = expf(fcoll);
                                                    if (flag_options->USE_MINI_HALOS){
                                                        fcoll_MINI_left = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low* log10_Mcrit_LW_int];
                                                        fcoll_MINI_right = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low *(log10_Mcrit_LW_int + 1)];
                                                        fcoll_MINI = fcoll_MINI_left * (1.-log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                                        fcoll_MINI = expf(fcoll_MINI);
                                                    }
                                                }
                                            }
                                            else {
                                                fcoll_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                            }
                                        }
                                        else {

                                            fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int]*( 1 + (float)fcoll_int - dens_val ) + \
                                                    log10_SFRD_z_low_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );

                                            fcoll = expf(fcoll);

                                            if (flag_options->USE_MINI_HALOS){
                                                fcoll_MINI_left = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low * log10_Mcrit_LW_int]*\
                                                                ( 1 + (float)fcoll_int - dens_val ) +\
                                                                    log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + 1 + NSFR_low * log10_Mcrit_LW_int]*\
                                                                ( dens_val - (float)fcoll_int );

                                                fcoll_MINI_right = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low * (log10_Mcrit_LW_int + 1)]*\
                                                                ( 1 + (float)fcoll_int - dens_val ) +\
                                                                log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + 1 + NSFR_low*(log10_Mcrit_LW_int + 1)]*\
                                                                ( dens_val - (float)fcoll_int );

                                                fcoll_MINI = fcoll_MINI_left * (1.-log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                                fcoll_MINI = expf(fcoll_MINI);
                                            }
                                        }
                                    }
                                }
                                else {

                                    if (curr_dens < 0.99*Deltac) {

                                        dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;

                                        fcoll_int = (int)floorf( dens_val );

                                        if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_high - 1)) {
                                            fcoll_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                        }

                                        fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + \
                                                SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );

                                        if (flag_options->USE_MINI_HALOS){
                                            fcoll_MINI_left = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high * log10_Mcrit_LW_int]*\
                                                        ( 1. + (float)fcoll_int - dens_val ) +\
                                                            SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high * log10_Mcrit_LW_int]*\
                                                        ( dens_val - (float)fcoll_int );

                                            fcoll_MINI_right = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high*(log10_Mcrit_LW_int + 1)]*\
                                                        ( 1. + (float)fcoll_int - dens_val ) +\
                                                            SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high*(log10_Mcrit_LW_int + 1)]*\
                                                        ( dens_val - (float)fcoll_int );

                                            fcoll_MINI = fcoll_MINI_left * (1.-log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                        }
                                    }
                                    else {
                                        fcoll = pow(10.,10.);
                                        fcoll_MINI =1e10;
                                    }
                                }
                            }
                            else {

                                if (flag_options->USE_MINI_HALOS){

                                    fcoll = Nion_ConditionalM(zpp_growth[R_ct],log(global_params.M_MIN_INTEGRAL),log(Mmax),sigmaMmax,Deltac,curr_dens,Mcrit_atom_interp_table[R_ct],
                                                              astro_params->ALPHA_STAR,0.,astro_params->F_STAR10,1.,Mlim_Fstar,0., user_params->FAST_FCOLL_TABLES);

                                    fcoll_MINI = Nion_ConditionalM_MINI(zpp_growth[R_ct],log(global_params.M_MIN_INTEGRAL),log(Mmax),sigmaMmax,Deltac,\
                                                           curr_dens,pow(10,log10_Mcrit_LW[R_ct][box_ct]),Mcrit_atom_interp_table[R_ct],\
                                                           astro_params->ALPHA_STAR_MINI,0.,astro_params->F_STAR7_MINI,1.,Mlim_Fstar_MINI, 0., user_params->FAST_FCOLL_TABLES);
                                    fcoll_MINI *= pow(10.,10.);

                                }
                                else {
                                    fcoll = Nion_ConditionalM(zpp_growth[R_ct],log(M_MIN),log(Mmax),sigmaMmax,Deltac,curr_dens,astro_params->M_TURN,
                                                              astro_params->ALPHA_STAR,0.,astro_params->F_STAR10,1.,Mlim_Fstar,0., user_params->FAST_FCOLL_TABLES);
                                }
                                fcoll *= pow(10.,10.);
                            }

                            ave_fcoll += fcoll;

                            del_fcoll_Rct[box_ct] = (1.+curr_dens)*fcoll;

                            if (flag_options->USE_MINI_HALOS){
                                ave_fcoll_MINI += fcoll_MINI;

                                del_fcoll_Rct_MINI[box_ct] = (1.+curr_dens)*fcoll_MINI;
                            }
                        }

                    }
                }

                for(i=0;i<user_params->N_THREADS;i++) {
                    if(fcoll_int_boundexceeded_threaded[omp_get_thread_num()]==1) {
                        LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables for the fcoll/nion_splines");
                        Throw(TableEvaluationError);
                    }
                }


                ave_fcoll /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);
                ave_fcoll_MINI /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);

                if(ave_fcoll!=0.) {
                    ave_fcoll_inv = 1./ave_fcoll;
                }

                if(ave_fcoll_MINI!=0.) {
                    ave_fcoll_inv_MINI = 1./ave_fcoll_MINI;
                }

                dfcoll_dz_val = (ave_fcoll_inv/pow(10.,10.))*ST_over_PS[R_ct]*SFR_timescale_factor[R_ct]/astro_params->t_STAR;

                dstarlya_dt_prefactor[R_ct] *= dfcoll_dz_val;

                //Calculate Lya flux for Lya heating
                if (flag_options->USE_LYA_HEATING){
                    dstarlya_cont_dt_prefactor[R_ct] *= dfcoll_dz_val;
                    dstarlya_inj_dt_prefactor[R_ct] *= dfcoll_dz_val;
                }

                if(flag_options->USE_MINI_HALOS){
                    dfcoll_dz_val_MINI = (ave_fcoll_inv_MINI/pow(10.,10.))*ST_over_PS_MINI[R_ct]*SFR_timescale_factor[R_ct]/astro_params->t_STAR;
                    dstarlya_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;
                    dstarlyLW_dt_prefactor[R_ct] *= dfcoll_dz_val;
                    dstarlyLW_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;

                    if (flag_options->USE_LYA_HEATING){
                        dstarlya_cont_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;
                        dstarlya_inj_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;
                    }
                }

                #pragma omp parallel shared(dxheat_dt_box,dxion_source_dt_box,dxlya_dt_box,dstarlya_dt_box,dfcoll_dz_val,del_fcoll_Rct,freq_int_heat_tbl_diff,\
                            m_xHII_low_box,inverse_val_box,freq_int_heat_tbl,freq_int_ion_tbl_diff,freq_int_ion_tbl,freq_int_lya_tbl_diff,\
                            freq_int_lya_tbl,dstarlya_dt_prefactor,R_ct,previous_spin_temp,this_spin_temp,const_zp_prefactor,prefactor_1,\
                            prefactor_2,delNL0,growth_factor_zp,dt_dzp,zp,dgrowth_factor_dzp,dcomp_dzp_prefactor,Trad_fast,dzp,TS_prefactor,\
                            xc_inverse,Trad_fast_inv,dstarlyLW_dt_box,dstarlyLW_dt_prefactor,dxheat_dt_box_MINI,dxion_source_dt_box_MINI,\
                            dxlya_dt_box_MINI,dstarlya_dt_box_MINI,dstarlyLW_dt_box_MINI,dfcoll_dz_val_MINI,del_fcoll_Rct_MINI,\
                            dstarlya_dt_prefactor_MINI,dstarlyLW_dt_prefactor_MINI,prefactor_2_MINI,const_zp_prefactor_MINI,\
                            dstarlya_cont_dt_box,dstarlya_inj_dt_box,dstarlya_cont_dt_prefactor,dstarlya_inj_dt_prefactor,\
                            dstarlya_cont_dt_box_MINI,dstarlya_inj_dt_box_MINI,dstarlya_cont_dt_prefactor_MINI,dstarlya_inj_dt_prefactor_MINI) \
                    private(box_ct,x_e,T,dxion_sink_dt,dxe_dzp,dadia_dzp,dspec_dzp,dcomp_dzp,dxheat_dzp,J_alpha_tot,T_inv,T_inv_sq,\
                            eps_CMB,dCMBheat_dzp,E_continuum,E_injected,Ndot_alpha_cont,Ndot_alpha_inj,eps_Lya_cont,eps_Lya_inj,\
                            Ndot_alpha_cont_MINI,Ndot_alpha_inj_MINI,eps_Lya_cont_MINI,eps_Lya_inj_MINI,prev_Ts,tau21,xCMB,\
                            xc_fast,xi_power,xa_tilde_fast_arg,TS_fast,TSold_fast,xa_tilde_fast,dxheat_dzp_MINI,J_alpha_tot_MINI,curr_delNL0) \
                    num_threads(user_params->N_THREADS)
                {
                    #pragma omp for reduction(+:J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave,Ts_ave,Tk_ave,x_e_ave,J_alpha_ave_MINI,Xheat_ave_MINI,J_LW_ave,J_LW_ave_MINI)
                    for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                        // I've added the addition of zero just in case. It should be zero anyway, but just in case there is some weird
                        // numerical thing
                        if(ave_fcoll!=0.) {
                            dxheat_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( \
                                                    (freq_int_heat_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                    freq_int_heat_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                            dxion_source_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( \
                                                    (freq_int_ion_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                    freq_int_ion_tbl[m_xHII_low_box[box_ct]][R_ct] ));

                            dxlya_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( \
                                                    (freq_int_lya_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                    freq_int_lya_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                            dstarlya_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_dt_prefactor[R_ct];

                            //Add Lya flux
                            if (flag_options->USE_LYA_HEATING){
                                dstarlya_cont_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_cont_dt_prefactor[R_ct];
                                dstarlya_inj_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_inj_dt_prefactor[R_ct];
                            }

                            if (flag_options->USE_MINI_HALOS){
                                dstarlyLW_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlyLW_dt_prefactor[R_ct];

                                if (flag_options->USE_LYA_HEATING){
                                    dstarlya_cont_dt_box_MINI[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_cont_dt_prefactor_MINI[R_ct];
                                    dstarlya_inj_dt_box_MINI[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_inj_dt_prefactor_MINI[R_ct];
                                }
                            }
                        }
                        else {
                            dxheat_dt_box[box_ct] += 0.;
                            dxion_source_dt_box[box_ct] += 0.;

                            dxlya_dt_box[box_ct] += 0.;
                            dstarlya_dt_box[box_ct] += 0.;

                            //Add Lya flux
                            if (flag_options->USE_LYA_HEATING){
                                dstarlya_cont_dt_box[box_ct] += 0.;
                                dstarlya_inj_dt_box[box_ct] += 0.;
                            }

                            if (flag_options->USE_MINI_HALOS){
                                dstarlyLW_dt_box[box_ct] += 0.;

                                if (flag_options->USE_LYA_HEATING){
                                    dstarlya_cont_dt_box_MINI[box_ct] += 0.;
                                    dstarlya_inj_dt_box_MINI[box_ct] += 0.;
                                }
                            }
                        }

                        if (flag_options->USE_MINI_HALOS){
                            if(ave_fcoll_MINI!=0.) {
                                dxheat_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)del_fcoll_Rct_MINI[box_ct]*( \
                                                            (freq_int_heat_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                        freq_int_heat_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                                dxion_source_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)del_fcoll_Rct_MINI[box_ct]*( \
                                                            (freq_int_ion_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                        freq_int_ion_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                                dxlya_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)del_fcoll_Rct_MINI[box_ct]*( \
                                                            (freq_int_lya_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                        freq_int_lya_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                                dstarlya_dt_box_MINI[box_ct] += (double)del_fcoll_Rct_MINI[box_ct]*dstarlya_dt_prefactor_MINI[R_ct];
                                dstarlyLW_dt_box_MINI[box_ct] += (double)del_fcoll_Rct_MINI[box_ct]*dstarlyLW_dt_prefactor_MINI[R_ct];
                            }
                        }

                        // If R_ct == 0, as this is the final smoothing scale (i.e. it is reversed)
                        if(R_ct==0) {

                            // Note here, that by construction it doesn't matter if using MINIMIZE_MEMORY as only need the R_ct = 0 box
                            curr_delNL0 = delNL0[0][box_ct];

                            x_e = previous_spin_temp->x_e_box[box_ct];
                            T = previous_spin_temp->Tk_box[box_ct];

                            // add prefactors
                            dxheat_dt_box[box_ct] *= const_zp_prefactor;
                            dxion_source_dt_box[box_ct] *= const_zp_prefactor;

                            dxlya_dt_box[box_ct] *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                            dstarlya_dt_box[box_ct] *= prefactor_2;

                            //Include pre-factors in Lya flux for Lya heating
                            if (flag_options->USE_LYA_HEATING){
                                dstarlya_cont_dt_box[box_ct] *= prefactor_2;
                                dstarlya_inj_dt_box[box_ct] *= prefactor_2;
                            }

                            if (flag_options->USE_MINI_HALOS){
                                dstarlyLW_dt_box[box_ct] *= prefactor_2 * (hplank * 1e21);

                                dxheat_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI;
                                dxion_source_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI;

                                dxlya_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                                dstarlya_dt_box_MINI[box_ct] *= prefactor_2_MINI;

                                dstarlyLW_dt_box_MINI[box_ct] *= prefactor_2_MINI * (hplank * 1e21);

                                if (flag_options->USE_LYA_HEATING){
                                    dstarlya_cont_dt_box_MINI[box_ct] *= prefactor_2_MINI;
                                    dstarlya_inj_dt_box_MINI[box_ct] *= prefactor_2_MINI;
                                }
                            }

                            //Added calculations of xCMB and tau_21 [eq. (2) and (4) of Reis et al. 2021]
                            prev_Ts = previous_spin_temp->Ts_box[box_ct];
                            tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)) /prev_Ts/hubble(zp);
                            xCMB = (1. - exp(-tau21))/tau21;

                            // Now we can solve the evolution equations  //

                            // First let's do dxe_dzp //
                            dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * \
                                            (1.+curr_delNL0*growth_factor_zp);
                            if (flag_options->USE_MINI_HALOS){
                                dxe_dzp = dt_dzp*(dxion_source_dt_box[box_ct] + dxion_source_dt_box_MINI[box_ct] - dxion_sink_dt );
                            }
                            else{
                                dxe_dzp = dt_dzp*(dxion_source_dt_box[box_ct] - dxion_sink_dt );
                            }

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

                            // next, X-ray heating
                            dxheat_dzp = dxheat_dt_box[box_ct] * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                            if (flag_options->USE_MINI_HALOS){
                                dxheat_dzp_MINI = dxheat_dt_box_MINI[box_ct] * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                            }

                            //next, CMB heating rate
                            dCMBheat_dzp = 0.;
                            if (flag_options->USE_CMB_HEATING) {
                                //Meiksin et al. 2021
                                eps_CMB = (3./4.) * (T_cmb*(1.+zp)/T21) * A10_HYPERFINE * f_H * (hplank*hplank/Lambda_21/Lambda_21/m_p) * (1.+2.*T/T21);
                                dCMBheat_dzp = 	-eps_CMB * (2./3./k_B/(1.+x_e))/hubble(zp)/(1.+zp);
                            }
                            //lastly, Ly-alpha heating rate
                            eps_Lya_cont = 0.;
                            eps_Lya_inj = 0.;
                            eps_Lya_cont_MINI = 0.;
                            eps_Lya_inj_MINI = 0.;
                            if (flag_options->USE_LYA_HEATING) {
                                E_continuum = Energy_Lya_heating(T, previous_spin_temp->Ts_box[box_ct], taugp(zp,curr_delNL0*growth_factor_zp,x_e), 2);
                                E_injected = Energy_Lya_heating(T, previous_spin_temp->Ts_box[box_ct], taugp(zp,curr_delNL0*growth_factor_zp,x_e), 3);
                                if (isnan(E_continuum) || isinf(E_continuum)){
                                    E_continuum = 0.;
                                }
                                if (isnan(E_injected) || isinf(E_injected)){
                                    E_injected = 0.;
                                }
                                Ndot_alpha_cont = (4.*PI*Ly_alpha_HZ) / (N_b0*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_cont_dt_box[box_ct];
                                Ndot_alpha_inj = (4.*PI*Ly_alpha_HZ) / (N_b0*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_inj_dt_box[box_ct];
                                eps_Lya_cont = - Ndot_alpha_cont * E_continuum * (2. / 3. /k_B/ (1.+x_e));
                                eps_Lya_inj = - Ndot_alpha_inj * E_injected * (2. / 3. /k_B/ (1.+x_e));
                                if (flag_options->USE_MINI_HALOS) {
                                    Ndot_alpha_cont_MINI = (4.*PI*Ly_alpha_HZ) / (N_b0*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_cont_dt_box_MINI[box_ct];
                                    Ndot_alpha_inj_MINI = (4.*PI*Ly_alpha_HZ) / (N_b0*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_inj_dt_box_MINI[box_ct];
                                    eps_Lya_cont_MINI = - Ndot_alpha_cont_MINI * E_continuum * (2. / 3. /k_B/ (1+x_e));
                                    eps_Lya_inj_MINI = - Ndot_alpha_inj_MINI * E_injected * (2. / 3. /k_B/ (1+x_e));
                                }
                            }

                            //update quantities
                            x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                            if (x_e > 1) // can do this late in evolution if dzp is too large
                                x_e = 1 - FRACT_FLOAT_ERR;
                            else if (x_e < 0)
                                x_e = 0;

                            //Add CMB and Lya heating rates, and evolve
                            if (T < MAX_TK) {
                                if (flag_options->USE_MINI_HALOS){
                                    T += ( dxheat_dzp + dxheat_dzp_MINI + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj + eps_Lya_cont_MINI + eps_Lya_inj_MINI) * dzp;
                                } else {
                                    T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj) * dzp;
                                }

                            }

                            if (isfinite(T) == 0) {
                                LOG_ERROR(
                                    "For box_ct=%d, got infinite value for Tk. dxheat_dzp=%g, dcomp_dzp=%g, dspec_dzp=%g, dadia_dzp=%g, dzp=%g, dxheat_dt_box=%g, dt_dzp=%g, dxe_dzp=%g, ",
                                    box_ct, dxheat_dzp, dcomp_dzp, dspec_dzp, dadia_dzp, dzp, dxheat_dt_box[box_ct], dt_dzp, dxe_dzp);
                                Throw(InfinityorNaNError);
                            }

                            if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                T = T_cmb*(1+zp);
                            }

                            this_spin_temp->x_e_box[box_ct] = x_e;
                            this_spin_temp->Tk_box[box_ct] = T;

                            J_alpha_tot = ( dxlya_dt_box[box_ct] + dstarlya_dt_box[box_ct] ); //not really d/dz, but the lya flux
                            if (flag_options->USE_MINI_HALOS){
                                J_alpha_tot_MINI = ( dxlya_dt_box_MINI[box_ct] + dstarlya_dt_box_MINI[box_ct] ); //not really d/dz, but the lya flux
                                this_spin_temp->J_21_LW_box[box_ct] = dstarlyLW_dt_box[box_ct] + dstarlyLW_dt_box_MINI[box_ct];
                            }

                            // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                            // Algorithm is the same, but written to be more computationally efficient
                            // Added corrections from xCMB [eq. (3) of Reis et al. 2021]
                            T_inv = expf((-1.)*logf(T));
                            T_inv_sq = expf((-2.)*logf(T));

                            xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*\
                                    ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );

                            xi_power = TS_prefactor * cbrt((1.0+curr_delNL0*growth_factor_zp)*(1.0-x_e)*T_inv_sq);

                            if (flag_options->USE_MINI_HALOS){
                                xa_tilde_fast_arg = xa_tilde_prefactor*(J_alpha_tot+J_alpha_tot_MINI)*\
                                                pow( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power, -1. );
                            }
                            else{
                                xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot*\
                                                pow( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power, -1. );
                            }

                            //if (J_alpha_tot > 1.0e-20) { // Must use WF effect
                            // New in v1.4
                            if (fabs(J_alpha_tot) > 1.0e-20) { // Must use WF effect
                                TS_fast = Trad_fast;
                                TSold_fast = 0.0;
                                while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {

                                    TSold_fast = TS_fast;

                                    xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - \
                                                     0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;

                                    TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+xa_tilde_fast*( T_inv + \
                                                    0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                                }
                            } else { // Collisions only
                                TS_fast = (xCMB + xc_fast)/(xCMB*Trad_fast_inv + xc_fast*T_inv);

                                xa_tilde_fast = 0.0;
                            }

                            if(TS_fast < 0.) {
                                // It can very rarely result in a negative spin temperature. If negative, it is a very small number.
                                //Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
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
                                if (flag_options->USE_MINI_HALOS){
                                    J_alpha_ave_MINI += J_alpha_tot_MINI;
                                    Xheat_ave_MINI += ( dxheat_dzp_MINI );
                                    J_LW_ave += dstarlyLW_dt_box[box_ct];
                                    J_LW_ave_MINI += dstarlyLW_dt_box_MINI[box_ct];
                                }
                            }

                            x_e_ave += x_e;
                        }
                    }
                }
            }
        }
        else {
            #pragma omp parallel shared(previous_spin_temp,x_int_XHII,inverse_diff,delNL0_rev,dens_grid_int_vals,ST_over_PS,zpp_growth,dfcoll_interp1,\
                            density_gridpoints,dfcoll_interp2,freq_int_heat_tbl_diff,freq_int_heat_tbl,freq_int_ion_tbl_diff,freq_int_ion_tbl,\
                            freq_int_lya_tbl_diff,freq_int_lya_tbl,dstarlya_dt_prefactor,const_zp_prefactor,prefactor_1,growth_factor_zp,dzp,\
                            dstarlya_cont_dt_prefactor, dstarlya_inj_dt_prefactor,\
                            dt_dzp,dgrowth_factor_dzp,dcomp_dzp_prefactor,this_spin_temp,xc_inverse,TS_prefactor,xa_tilde_prefactor,Trad_fast_inv,\
                            zpp_for_evolve_list,sigma_Tmin,sigma_atR) \
                    private(box_ct,x_e,T,xHII_call,m_xHII_low,inverse_val,dxheat_dt,dxion_source_dt,dxlya_dt,dstarlya_dt,curr_delNL0,R_ct,\
                            dstarlya_cont_dt,dstarlya_inj_dt,prev_Ts,tau21,xCMB,\
                            eps_CMB, dCMBheat_dzp, E_continuum, E_injected, Ndot_alpha_cont, Ndot_alpha_inj, eps_Lya_cont, eps_Lya_inj,\
                            dfcoll_dz_val,dxion_sink_dt,dxe_dzp,dadia_dzp,dspec_dzp,dcomp_dzp,dxheat_dzp,J_alpha_tot,T_inv,T_inv_sq,xc_fast,xi_power,\
                            xa_tilde_fast_arg,TS_fast,TSold_fast,xa_tilde_fast) \
                    num_threads(user_params->N_THREADS)
            {
                #pragma omp for reduction(+:J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave,Ts_ave,Tk_ave,x_e_ave)
                for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

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

                    //Initialized values for Ly-alpha heating
                    dstarlya_cont_dt = 0;
                    dstarlya_inj_dt = 0;

                    curr_delNL0 = delNL0_rev[box_ct][0];

                    if (!NO_LIGHT){
                        // Now determine all the differentials for the heating/ionisation rate equations
                        for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){

                            if(user_params->USE_INTERPOLATION_TABLES) {
                                if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                                    table_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                }

                                dfcoll_dz_val = ST_over_PS[R_ct]*(1.+delNL0_rev[box_ct][R_ct]*zpp_growth[R_ct])*( \
                                                    dfcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*\
                                                        (density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct]) + \
                                                    dfcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*\
                                                        (delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct]) );
                            }
                            else {
                                dfcoll_dz_val = ST_over_PS[R_ct]*(1.+delNL0_rev[box_ct][R_ct]*zpp_growth[R_ct])*( \
                                                dfcoll_dz(zpp_for_evolve_list[R_ct], sigma_Tmin[R_ct], delNL0_rev[box_ct][R_ct], sigma_atR[R_ct]) );
                            }

                            dxheat_dt += dfcoll_dz_val * \
                                        ( (freq_int_heat_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_heat_tbl[m_xHII_low][R_ct] );
                            dxion_source_dt += dfcoll_dz_val * \
                                        ( (freq_int_ion_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_ion_tbl[m_xHII_low][R_ct] );

                            dxlya_dt += dfcoll_dz_val * \
                                        ( (freq_int_lya_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_lya_tbl[m_xHII_low][R_ct] );
                            dstarlya_dt += dfcoll_dz_val*dstarlya_dt_prefactor[R_ct];
                            //Ly-alpha Heating
                            if (flag_options->USE_LYA_HEATING){
                              dstarlya_cont_dt += dfcoll_dz_val*dstarlya_cont_dt_prefactor[R_ct];
                              dstarlya_inj_dt += dfcoll_dz_val*dstarlya_inj_dt_prefactor[R_ct];
                            }
                        }
                    }

                    // add prefactors
                    dxheat_dt *= const_zp_prefactor;
                    dxion_source_dt *= const_zp_prefactor;

                    dxlya_dt *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                    dstarlya_dt *= prefactor_2;

                    //Ly-alpha Heating
                    if (flag_options->USE_LYA_HEATING){
                      dstarlya_cont_dt *= prefactor_2;
                      dstarlya_inj_dt *= prefactor_2;
                    }

                    //Added calculations of xCMB and tau_21 [eq. (2) and (4) of Reis et al. 2021]
                    prev_Ts = previous_spin_temp->Ts_box[box_ct];
                    tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)) /prev_Ts/hubble(zp);
                    xCMB = (1. - exp(-tau21))/tau21;

                    // Now we can solve the evolution equations  //

                    // First let's do dxe_dzp //
                    dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * \
                                    (1.+curr_delNL0*growth_factor_zp);
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

                    // next, X-ray heating
                    dxheat_dzp = dxheat_dt * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);

                    //next, CMB heating rate
                    dCMBheat_dzp = 0.;

                    if (flag_options->USE_CMB_HEATING) {
                        eps_CMB = (3./4.) * (T_cmb*(1.+zp)/T21) * A10_HYPERFINE * f_H * (hplank*hplank/Lambda_21/Lambda_21/m_p) * (1.+2.*T/T21);
                        dCMBheat_dzp = -eps_CMB * (2./3./k_B/(1.+x_e))/hubble(zp)/(1.+zp);
                    }

                    //lastly, Ly-alpha heating rate
                    eps_Lya_cont = 0.;
                    eps_Lya_inj = 0.;

                    if (flag_options->USE_LYA_HEATING) {
                        E_continuum = Energy_Lya_heating(T, previous_spin_temp->Ts_box[box_ct], taugp(zp,curr_delNL0*growth_factor_zp,x_e), 2);
                        E_injected = Energy_Lya_heating(T, previous_spin_temp->Ts_box[box_ct], taugp(zp,curr_delNL0*growth_factor_zp,x_e), 3);
                        if (isnan(E_continuum) || isinf(E_continuum)){
                            E_continuum = 0.;
                        }
                        if (isnan(E_injected) || isinf(E_injected)){
                            E_injected = 0.;
                        }
                        Ndot_alpha_cont = (4.*PI*Ly_alpha_HZ) / (N_b0*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_cont_dt;
                        Ndot_alpha_inj = (4.*PI*Ly_alpha_HZ) / (N_b0*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_inj_dt;
                        eps_Lya_cont = - Ndot_alpha_cont * E_continuum * (2. / 3. /k_B/ (1.+x_e));
                        eps_Lya_inj = - Ndot_alpha_inj * E_injected * (2. / 3. /k_B/ (1.+x_e));
                    }

                    //update quantities

                    x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                    if (x_e > 1) // can do this late in evolution if dzp is too large
                        x_e = 1 - FRACT_FLOAT_ERR;
                    else if (x_e < 0)
                        x_e = 0;
                    //Add CMB and Lya heating rates, and evolve
                    if (T < MAX_TK) {
                        T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj) * dzp;
                    }

                    if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                        T = T_cmb*(1+zp);
                    }

                    this_spin_temp->x_e_box[box_ct] = x_e;
                    this_spin_temp->Tk_box[box_ct] = T;

                    J_alpha_tot = ( dxlya_dt + dstarlya_dt ); //not really d/dz, but the lya flux

                    // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                    // Algorithm is the same, but written to be more computationally efficient
                    // Added corrections from xCMB [eq. (3) of Reis et al. 2021]
                    T_inv = pow(T,-1.);
                    T_inv_sq = pow(T,-2.);

                    xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*( (1.0-x_e)*No*kappa_10(T,0) + \
                                                                             x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                    xi_power = TS_prefactor * pow((1.0+curr_delNL0*growth_factor_zp)*(1.0-x_e)*T_inv_sq, 1.0/3.0);
                    xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot*\
                                        pow( 1.0 + 2.98394*xi_power + 1.53583*pow(xi_power,2.) + 3.85289*pow(xi_power,3.), -1. );

                    if (J_alpha_tot > 1.0e-20) { // Must use WF effect
                        TS_fast = Trad_fast;
                        TSold_fast = 0.0;
                        while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {

                            TSold_fast = TS_fast;

                            xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - \
                                            0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;

                            TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+xa_tilde_fast*\
                                                ( T_inv + 0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                        }
                    } else { // Collisions only
                        TS_fast = (xCMB + xc_fast)/(xCMB*Trad_fast_inv + xc_fast*T_inv);
                        xa_tilde_fast = 0.0;
                    }

                    if(TS_fast < 0.) {
                        // It can very rarely result in a negative spin temperature. If negative, it is a very small number.
                        // Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
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

            for(i=0;i<user_params->N_THREADS; i++) {
                if(table_int_boundexceeded_threaded[i]==1) {
                    LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of dfcoll_dz_val");
                    Throw(TableEvaluationError);
                }
            }
        }

        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
            if(isfinite(this_spin_temp->Ts_box[box_ct])==0) {
                LOG_ERROR("Estimated spin temperature is either infinite of NaN!");
                Throw(InfinityorNaNError);
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

            if (flag_options->USE_MINI_HALOS){
                J_alpha_ave_MINI /= (double)HII_TOT_NUM_PIXELS;
                Xheat_ave_MINI /= (double)HII_TOT_NUM_PIXELS;
                J_LW_ave /= (double)HII_TOT_NUM_PIXELS;
                J_LW_ave_MINI /= (double)HII_TOT_NUM_PIXELS;

                LOG_DEBUG("zp = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e(%e) xalpha_ave = %e \
                          Xheat_ave = %e(%e) Xion_ave = %e J_LW_ave = %e (%e)",zp,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,\
                          J_alpha_ave_MINI,xalpha_ave,Xheat_ave,Xheat_ave_MINI,Xion_ave,J_LW_ave/1e21,J_LW_ave_MINI/1e21);
            }
            else{
                LOG_DEBUG("zp = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e xalpha_ave = %e \
                          Xheat_ave = %e Xion_ave = %e",zp,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave);
            }
        }

    } // end main integral loop over z'

    fftwf_free(box);
    fftwf_free(unfiltered_box);

    if (flag_options->USE_MINI_HALOS){
        fftwf_free(log10_Mcrit_LW_unfiltered);
        fftwf_free(log10_Mcrit_LW_filtered);
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            free(log10_Mcrit_LW[R_ct]);
        }
        free(log10_Mcrit_LW);
    }

//    fftwf_destroy_plan(plan);
    fftwf_forget_wisdom();
    fftwf_cleanup_threads();
    fftwf_cleanup();

    // Free all the boxes. Ideally, we wouldn't do this, as almost always
    // the *next* call to ComputeTsBox will need the same memory. However,
    // we can't be sure that a call from python will not have changed the box size
    // without freeing, and get a segfault. The only way around this would be to
    // check (probably in python) every time spin() is called, whether the boxes
    // are already initialised _and_ whether they are of the right shape. This
    // seems difficult, so we leave that as future work.
    if(cleanup) free_TsCalcBoxes(user_params,flag_options);

    free(table_int_boundexceeded_threaded);
    free(fcoll_int_boundexceeded_threaded);
    } // End of try
    Catch(status){
        return(status);
    }
    return(0);
}


void free_TsCalcBoxes(struct UserParams *user_params, struct FlagOptions *flag_options)
{
    int i,j;

    free(zpp_edge);
    free(sigma_atR);
    free(R_values);

    if(user_params->USE_INTERPOLATION_TABLES) {
        free(min_densities);
        free(max_densities);

        free(zpp_interp_table);
    }

    free(SingleVal_int);
    free(dstarlya_dt_prefactor);
    free(fcoll_R_array);
    free(zpp_growth);
    free(inverse_diff);
    free(sigma_Tmin);
    free(ST_over_PS);
    free(sum_lyn);
    free(zpp_for_evolve_list);
    if (flag_options->USE_LYA_HEATING){
        free(dstarlya_cont_dt_prefactor);
        free(dstarlya_inj_dt_prefactor);
        free(sum_ly2);
        free(sum_lynto2);
    }
    if (flag_options->USE_MINI_HALOS){
        free(Mcrit_atom_interp_table);
        free(dstarlya_dt_prefactor_MINI);
        free(dstarlyLW_dt_prefactor);
        free(dstarlyLW_dt_prefactor_MINI);
        free(ST_over_PS_MINI);
        free(sum_lyn_MINI);
        free(sum_lyLWn);
        free(sum_lyLWn_MINI);
        if (flag_options->USE_LYA_HEATING){
            free(dstarlya_cont_dt_prefactor_MINI);
            free(dstarlya_inj_dt_prefactor_MINI);
            free(sum_ly2_MINI);
            free(sum_lynto2_MINI);
        }
    }

    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        free(SFR_timescale_factor);

        if(user_params->MINIMIZE_MEMORY) {
            free(delNL0[0]);
        }
        else {
            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                free(delNL0[i]);
            }
        }
        free(delNL0);

        free(xi_SFR_Xray);
        free(wi_SFR_Xray);

        if(user_params->USE_INTERPOLATION_TABLES) {
            free(overdense_low_table);
            free(overdense_high_table);
            for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                free(log10_SFRD_z_low_table[j]);
            }
            free(log10_SFRD_z_low_table);

            for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                free(SFRD_z_high_table[j]);
            }
            free(SFRD_z_high_table);
        }

        free(del_fcoll_Rct);
        free(dxheat_dt_box);
        free(dxion_source_dt_box);
        free(dxlya_dt_box);
        free(dstarlya_dt_box);
        free(m_xHII_low_box);
        free(inverse_val_box);

        if (flag_options->USE_LYA_HEATING){
            free(dstarlya_cont_dt_box);
            free(dstarlya_inj_dt_box);
        }

        if(flag_options->USE_MINI_HALOS){
            if(user_params->USE_INTERPOLATION_TABLES) {
                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    free(log10_SFRD_z_low_table_MINI[j]);
                }
                free(log10_SFRD_z_low_table_MINI);

                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    free(SFRD_z_high_table_MINI[j]);
                }
                free(SFRD_z_high_table_MINI);
            }

            free(del_fcoll_Rct_MINI);
            free(dstarlyLW_dt_box);
            free(dxheat_dt_box_MINI);
            free(dxion_source_dt_box_MINI);
            free(dxlya_dt_box_MINI);
            free(dstarlya_dt_box_MINI);
            free(dstarlyLW_dt_box_MINI);

            if (flag_options->USE_LYA_HEATING){
                free(dstarlya_cont_dt_box_MINI);
                free(dstarlya_inj_dt_box_MINI);
            }
        }
    }
    else {

        if(user_params->USE_INTERPOLATION_TABLES) {
            free(Sigma_Tmin_grid);
            free(ST_over_PS_arg_grid);
            free(delNL0_bw);
            free(delNL0_Offset);
            free(delNL0_LL);
            free(delNL0_UL);
            free(delNL0_ibw);
            free(log10delNL0_diff);
            free(log10delNL0_diff_UL);

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

            for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
                free(dens_grid_int_vals[i]);
            }
            free(dens_grid_int_vals);

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
        }

        for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
            free(delNL0_rev[i]);
        }
        free(delNL0_rev);

    }

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

    destruct_heat();
    TsInterpArraysInitialised = false;
}
