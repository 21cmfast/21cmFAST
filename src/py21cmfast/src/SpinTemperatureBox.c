// Re-write of find_HII_bubbles.c for being accessible within the MCMC

void ts_halos(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options, float perturbed_field_redshift, short cleanup,
                  struct PerturbedField *perturbed_field, struct XraySourceBox *source_box, struct TsBox *previous_spin_temp,
                  struct InitialConditions *ini_boxes, struct TsBox *this_spin_temp);

//Global arrays which have yet to be moved to structs
//R x box arrays
float **delNL0, **log10_Mcrit_LW;

//arrays for R-dependent prefactors
double *dstarlya_dt_prefactor, *dstarlya_dt_prefactor_MINI;
double *sum_lyn_MINI, *sum_lyLWn, *sum_lyLWn_MINI, *sum_lyn;
double *dstarlyLW_dt_prefactor, *dstarlyLW_dt_prefactor_MINI;
double *dstarlya_cont_dt_prefactor, *dstarlya_inj_dt_prefactor, *sum_ly2, *sum_lynto2;
double *dstarlya_cont_dt_prefactor_MINI, *dstarlya_inj_dt_prefactor_MINI, *sum_ly2_MINI, *sum_lynto2_MINI;

//boxes to hold stellar fraction integrals (Fcoll or SFRD)
float *del_fcoll_Rct, *del_fcoll_Rct_MINI;

//radiative term boxes which are summed over R
double *dxheat_dt_box, *dxion_source_dt_box, *dxlya_dt_box, *dstarlya_dt_box;
double *dxheat_dt_box_MINI, *dxion_source_dt_box_MINI, *dxlya_dt_box_MINI, *dstarlya_dt_box_MINI;
double *dstarlyLW_dt_box, *dstarlyLW_dt_box_MINI;
double *dstarlya_cont_dt_box, *dstarlya_inj_dt_box;
double *dstarlya_cont_dt_box_MINI, *dstarlya_inj_dt_box_MINI;

//x_e interpolation boxes / arrays (not a RGI)
float *inverse_val_box;
int *m_xHII_low_box;
float *inverse_diff;

// Grids/arrays that are re-evaluated for each zp
float *Mcrit_atom_interp_table;

// interpolation tables for the heating/ionisation integrals
double **freq_int_heat_tbl, **freq_int_ion_tbl, **freq_int_lya_tbl, **freq_int_heat_tbl_diff;
double **freq_int_ion_tbl_diff, **freq_int_lya_tbl_diff;

//R-dependent arrays which are set once
double *R_values, *dzpp_list, *dtdz_list, *zpp_growth, *zpp_for_evolve_list, *zpp_edge;
double *sigma_min, *sigma_max, *M_max_R, *M_min_R;
double *min_densities, *max_densities;

//lazy globals (things I should put elsewhere but are only set once based on parameters)
double Mlim_Fstar_g, Mlim_Fesc_g, Mlim_Fstar_MINI_g, Mlim_Fesc_MINI_g;

//Arrays which specify the Radii, distances, redshifts of each shell
//  They will have a global instance since they are reused a lot
//  However it is worth considering passing them into functions instead
// struct radii_spec{
//     double *R_values; //Radii edge of each shell
//     double *zpp_edge; //redshift of the shell edge
//     double *zpp_cen; //middle redshift of cell (z_inner + z_outer)/2
//     double *dzpp_list; //redshift difference between inner and outer edge
//     double *dtdz_list; //dtdz at zpp_cen
//     double *zpp_growth; //linear growth factor D(z) at zpp_cen
// }
// struct radii_spec r_s;

bool TsInterpArraysInitialised = false;
int debug_printed;
#pragma omp threadprivate(debug_printed)

int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options,
                  float perturbed_field_redshift, short cleanup,
                  struct PerturbedField *perturbed_field, struct XraySourceBox *source_box, struct TsBox *previous_spin_temp,
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
    Broadcast_struct_global_TS(user_params,cosmo_params,astro_params,flag_options);
    Broadcast_struct_global_IT(user_params);
    omp_set_num_threads(user_params->N_THREADS);
    debug_printed = 0;

    //TODO: rename to ts_main or move here with some allocation functions etc
    ts_halos(redshift,prev_redshift,user_params,cosmo_params,astro_params,flag_options,perturbed_field_redshift,
            cleanup,perturbed_field,source_box,previous_spin_temp,ini_boxes,this_spin_temp);
    
    destruct_heat();

    // This is an entire re-write of Ts.c from 21cmFAST. You can refer back to Ts.c in 21cmFAST if this become a little obtuse. The computation has remained the same //
    // JD: it became obtuse so I have attempted to modularise it

    } // End of try
    Catch(status){
        return(status);
    }
    return(0);
}

/* JDAVIES: I'm setting up a refactor here for the Halo option, but plan to move over the rest afterwards
 * This is convenient because calculating TS from the Halo Field won't use most of the above code, which is
 * dedicated to setting up the HMF integrals. I want to abstract into smaller parts e.g: filling tables, nu integrals
 * as well as have separate high-level functions (ts_halo,ts_fcoll,ts_massdep_zeta) so that each flag case can 
 * allocate and use what it needs while staying readable. Afterwards I would also want to replace most of the global
 * variables with static structs, so that things are scoped properly.*/
 
//OTHER NOTES:
//Assuming that the same redshift (or within 0.0001) isn't called twice in a row (which it shouldn't be because caching), the global tables (Nion,SFRD)
//  We may want to not use tables for global SFRD and Nion (would require a change in nu_tau_one)
//I rely on the default behaviour of OpenMP scoping (everything shared unless its a stack variable defined in the parallel region)
//  so I can avoid making massive directives, this breaks from the style of the remainder of the code but I find it much more readable
//  the only downside is when there are a lot of re-used private variables which is rarely the case and can be specially placed in a private clause
//The perturbed density field can be at a different redshift, it is linearly extrapolated to zp
//  I honestly don't know why this is an option, since perturbfields are almost always generated at the same redshift and it's
//  forced to be the same in _setup_redshift()
//z-INTEPROLATIONS: perturb field is linearly extrapolated to zp or zpp, local Nion calculations are based on this
//  globals are simply linearly interpolated to zpp
//Tau integrals are based on global Nion estimates. I want to change this to depend on the source field

static struct AstroParams *astro_params_ts;
static struct CosmoParams *cosmo_params_ts;
static struct UserParams *user_params_ts;
static struct FlagOptions *flag_options_ts;

//allocate the arrays that are always needed and defined globally, including frequency integrals, zp edges, etc
void Broadcast_struct_global_TS(struct UserParams *user_params, struct CosmoParams *cosmo_params,struct AstroParams *astro_params, struct FlagOption *flag_options){
    cosmo_params_ts = cosmo_params;
    user_params_ts = user_params;
    astro_params_ts = astro_params;
    flag_options_ts = flag_options;
}

void alloc_global_arrays(){
    int i,j;
    //z-edges
    zpp_for_evolve_list = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    zpp_growth = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    zpp_edge = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    dzpp_list = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    dtdz_list = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    R_values = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    
    sigma_min = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    sigma_max = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    M_min_R = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    M_max_R = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    
    //frequency integral tables
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
    inverse_diff = (float *)calloc(x_int_NXHII,sizeof(float));
    //actual heating term boxes
    dxheat_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
    dxion_source_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
    dxlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
    dstarlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
    if(flag_options_ts->USE_LYA_HEATING){
        dstarlya_cont_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dstarlya_inj_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
    }
    if(flag_options_ts->USE_MINI_HALOS){
        dstarlyLW_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
    }

    //spectral stuff
    //TODO: add LYA_HEATING
    dstarlya_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    if(flag_options_ts->USE_LYA_HEATING){
        dstarlya_cont_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        dstarlya_inj_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    }

    if(flag_options_ts->USE_MINI_HALOS){
        dstarlya_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        dstarlyLW_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        dstarlyLW_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        if(flag_options_ts->USE_LYA_HEATING){
            dstarlya_cont_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dstarlya_inj_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
    }
    
    
    //Nonhalo stuff
    //TODO: move interp tables to the other framework in interpolation.c
    int num_R_boxes = user_params_ts->MINIMIZE_MEMORY ? 1 : global_params.NUM_FILTER_STEPS_FOR_Ts;
    if(!flag_options_ts->USE_HALO_FIELD){
        delNL0 = (float **)calloc(num_R_boxes,sizeof(float *));
        for(i=0;i<num_R_boxes;i++) {
            delNL0[i] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));
        }
        if(flag_options_ts->USE_MINI_HALOS){
            log10_Mcrit_LW = (float **)calloc(num_R_boxes,sizeof(float *));
            for (i=0; i<num_R_boxes; i++){
                log10_Mcrit_LW[i] = (float *)calloc(HII_TOT_NUM_PIXELS, sizeof(float));
            }
        }
        
        del_fcoll_Rct =  calloc(HII_TOT_NUM_PIXELS,sizeof(float));
        if(flag_options_ts->USE_MINI_HALOS){
            del_fcoll_Rct_MINI = calloc(HII_TOT_NUM_PIXELS,sizeof(float));
        }

        xi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));
        wi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));

        if(user_params_ts->USE_INTERPOLATION_TABLES) {
            min_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            max_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));

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

            if(flag_options_ts->USE_MINI_HALOS){
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
    }
    
    //helpers for the interpolation
    m_xHII_low_box = (int *)calloc(HII_TOT_NUM_PIXELS,sizeof(int));
    inverse_val_box = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    if(flag_options_ts->USE_MINI_HALOS){
        Mcrit_atom_interp_table = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
    }
    
    TsInterpArraysInitialised = true;
}

void free_global_arrays(){
    int i,j;
    //frequency integrals
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
    free(inverse_diff);

    //z- edges
    free(zpp_growth);
    free(zpp_edge);
    free(zpp_for_evolve_list);
    free(dzpp_list);
    free(dtdz_list);
    free(R_values);

    free(sigma_min);
    free(sigma_max);
    free(M_min_R);
    free(M_max_R);

    //spectral
    free(dstarlya_dt_prefactor);
    if(flag_options_ts->USE_LYA_HEATING){
        free(dstarlya_cont_dt_prefactor);
        free(dstarlya_inj_dt_prefactor);
    }
    if(flag_options_ts->USE_MINI_HALOS){
        free(dstarlya_dt_prefactor_MINI);
        free(dstarlyLW_dt_prefactor);
        free(dstarlyLW_dt_prefactor_MINI);
        if(flag_options_ts->USE_LYA_HEATING){
            free(dstarlya_inj_dt_prefactor_MINI);
            free(dstarlya_cont_dt_prefactor_MINI);
        }
    }

    //boxes
    free(dxheat_dt_box);
    free(dxion_source_dt_box);
    free(dxlya_dt_box);
    free(dstarlya_dt_box);
    free(sum_lyn);
    if(flag_options_ts->USE_MINI_HALOS){
        free(dstarlyLW_dt_box);
    }
    if(flag_options_ts->USE_LYA_HEATING){
        free(dstarlya_cont_dt_box);
        free(dstarlya_inj_dt_box);
    }

    //interpolation helpers
    free(m_xHII_low_box);
    free(inverse_val_box);

    if(flag_options_ts->USE_MINI_HALOS){
        free(Mcrit_atom_interp_table);
    }

    //interp tables
    int num_R_boxes = user_params_ts->MINIMIZE_MEMORY ? 1 : global_params.NUM_FILTER_STEPS_FOR_Ts;
    if(!flag_options_ts->USE_HALO_FIELD){
        for(i=0;i<num_R_boxes;i++) {
            free(delNL0[i]);
        }
        free(delNL0);
        
        if(flag_options_ts->USE_MINI_HALOS){
            for(i=0;i<num_R_boxes;i++) {
                free(log10_Mcrit_LW[i]);
            }
            free(log10_Mcrit_LW);
        }

        free(del_fcoll_Rct);
        if(flag_options_ts->USE_MINI_HALOS){
            free(del_fcoll_Rct_MINI);
        }

        free(xi_SFR_Xray);
        free(wi_SFR_Xray);
        if(user_params_ts->USE_INTERPOLATION_TABLES) {
            free(min_densities);
            free(max_densities);
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
    }
    
    TsInterpArraysInitialised = false;
}

//This function should construct all the tables which depend on R
void setup_z_edges(double zp){
    double R, R_factor;
    double zpp, prev_zpp, prev_R;
    double dzpp_for_evolve;
    double determine_zpp_max; //this is not global like the minimum
    int R_ct;
    LOG_DEBUG("Starting z edges");

    R = L_FACTOR*user_params_ts->BOX_LEN/(float)user_params_ts->HII_DIM;
    R_factor = pow(global_params.R_XLy_MAX/R, 1/((float)global_params.NUM_FILTER_STEPS_FOR_Ts));
    
    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        R_values[R_ct] = R;
        if(R_ct==0){
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
        if(R_ct==0){
            dzpp_for_evolve = zp - zpp_edge[0];
        }
        else{
            dzpp_for_evolve = zpp_edge[R_ct-1] - zpp_edge[R_ct];
        }
        zpp_growth[R_ct] = dicke(zpp); //growth factors
        dzpp_list[R_ct] = dzpp_for_evolve; //z bin width
        dtdz_list[R_ct] = dtdz(zpp); // dt/dz''
        R *= R_factor;
        // LOG_DEBUG("edge: R: %.3e D: %.3e z: %.3e (%.3e) dt: %.3e"
        //    ,R_values[R_ct],zpp_growth[R_ct],zpp,zpp_edge[R_ct],dtdz_list[R_ct]);
    }
    LOG_DEBUG("%d steps R range [%.2e,%.2e] z range [%.2f,%.2f]",R_ct,R_values[0],R_values[R_ct-1],zp,zpp_edge[R_ct-1]);
}

void calculate_spectral_factors(double zp){
    double nuprime;
    bool first_radii=true, first_zero=true;
    double trial_zpp_max,trial_zpp_min,trial_zpp;
    int counter,ii;
    int n_pts_radii=1000;
    double weight;
    int R_ct, n_ct;
    double zpp,zpp_integrand;

    double sum_lyn_val, sum_lyn_val_MINI;
    double sum_lyLW_val, sum_lyLW_val_MINI;
    double sum_lynto2_val, sum_lynto2_val_MINI;
    double sum_ly2_val, sum_ly2_val_MINI;
    double sum_lyn_prev = 0., sum_lyn_prev_MINI = 0.;
    double sum_ly2_prev = 0., sum_ly2_prev_MINI = 0.;
    double sum_lynto2_prev = 0., sum_lynto2_prev_MINI = 0.;
    double prev_zpp;
    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        zpp = zpp_for_evolve_list[R_ct];
        //We need to set up prefactors for how much of Lyman-N radiation is recycled to Lyman-alpha
        sum_lyn_val = 0.;
        sum_lyn_val_MINI = 0.;
        sum_lyLW_val = 0.;
        sum_lyLW_val_MINI = 0.;
        sum_lynto2_val = 0.;
        sum_lynto2_val_MINI = 0.;
        sum_ly2_val = 0.;
        sum_ly2_val_MINI = 0.;

        //in case we use LYA_HEATING, we separate the ==2 and >2 cases
        nuprime = nu_n(2)*(1.+zpp)/(1.+zp);
        if(zpp < zmax(zp,2)){
            if(flag_options_ts->USE_MINI_HALOS){
                sum_ly2_val = frecycle(2) * spectral_emissivity(nuprime, 0, 2);
                sum_ly2_val_MINI = frecycle(2) * spectral_emissivity(nuprime, 0, 3);
            }
            else{
                sum_ly2_val = frecycle(2) * spectral_emissivity(nuprime, 0, global_params.Pop);
            }
        }

        for (n_ct=NSPEC_MAX; n_ct>=3; n_ct--){
            if (zpp > zmax(zp, n_ct))
                continue;

            nuprime = nu_n(n_ct)*(1+zpp)/(1.0+zp);

            if(flag_options_ts->USE_MINI_HALOS){
                sum_lynto2_val += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
                sum_lynto2_val_MINI += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);

                if (nuprime < NU_LW_THRESH / NUIONIZATION)
                    nuprime = NU_LW_THRESH / NUIONIZATION;
                if (nuprime >= nu_n(n_ct + 1))
                    continue;
                sum_lyLW_val  += (1. - astro_params_ts->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                sum_lyLW_val_MINI += (1. - astro_params_ts->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
            }
            else{
                //This is only useful if global_params.Pop is ever used, which I think is rare
                //TODO: It would be nice to remove the if-else otherwise
                sum_lynto2_val += frecycle(n_ct) * spectral_emissivity(nuprime, 0, global_params.Pop);
            }
        }
        sum_lyn_val = sum_ly2_val + sum_lynto2_val;
        sum_lyn_val_MINI = sum_ly2_val_MINI + sum_lynto2_val_MINI;
        
        //At the edge of the redshift limit, part of the shell will still contain a contribution
        //  This loop approximates the volume which contains the contribution
        //  and multiplies this by the previous shell's value.
        //TODO: this should probably be done separately for ly2, lyto2, OR each lyN?
        if(R_ct > 1 && sum_lyn_val==0.0 && sum_lyn_prev>0. && first_radii) {
            for(ii=0;ii<n_pts_radii;ii++) {
                trial_zpp = prev_zpp + (zpp - prev_zpp)*(float)ii/((float)n_pts_radii-1.);
                counter = 0;
                for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                    if (trial_zpp > zmax(zp, n_ct))
                        continue;
                    counter += 1;
                }
                //This is the first sub-radius which has no contribution
                //Use this distance to weigh contribution at previous R
                if(counter==0&&first_zero) {
                    first_zero = false;
                    weight = (float)ii/(float)n_pts_radii;
                }
            }
            sum_lyn_val = weight * sum_lyn_prev;
            sum_ly2_val = weight * sum_ly2_prev;
            sum_lynto2_val = weight * sum_lynto2_prev;
            if (flag_options_ts->USE_MINI_HALOS){
                sum_lyn_val_MINI = weight * sum_lyn_prev_MINI;
                sum_ly2_val_MINI = weight * sum_ly2_prev_MINI;
                sum_lynto2_val_MINI = weight * sum_lynto2_prev_MINI;
            }
            first_radii = false;
        }
        //TODO: compared to Mesinger+2011, which has (1+zpp)^3, same as const_zp_prefactor, figure out why
        zpp_integrand = ( pow(1+zp,2)*(1+zpp) );
        dstarlya_dt_prefactor[R_ct] = zpp_integrand * sum_lyn_val;
        LOG_SUPER_DEBUG("z: %.2e R: %.2e int %.2e starlya: %.4e",zpp,R_values[R_ct],zpp_integrand,dstarlya_dt_prefactor[R_ct]);
        
        if(flag_options_ts->USE_LYA_HEATING){
            dstarlya_cont_dt_prefactor[R_ct] = zpp_integrand * sum_ly2_val;
            dstarlya_inj_dt_prefactor[R_ct] = zpp_integrand * sum_lynto2_val;
            LOG_SUPER_DEBUG("cont %.2e inj %.2e",dstarlya_cont_dt_prefactor[R_ct],dstarlya_inj_dt_prefactor[R_ct]);
        }
        if(flag_options_ts->USE_MINI_HALOS){
            dstarlya_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lyn_val_MINI;
            dstarlyLW_dt_prefactor[R_ct] = zpp_integrand * sum_lyLW_val;
            dstarlyLW_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lyLW_val_MINI;
            if(flag_options_ts->USE_LYA_HEATING){
                dstarlya_cont_dt_prefactor_MINI[R_ct] = zpp_integrand * sum_ly2_val_MINI;
                dstarlya_inj_dt_prefactor_MINI[R_ct] = zpp_integrand * sum_lynto2_val_MINI;
            }
            
            LOG_SUPER_DEBUG("starmini: %.2e LW: %.2e LWmini: %.2e",dstarlya_dt_prefactor_MINI[R_ct],
                                                        dstarlyLW_dt_prefactor[R_ct],
                                                        dstarlyLW_dt_prefactor_MINI[R_ct]);
            LOG_SUPER_DEBUG("cont mini %.2e inj mini %.2e",dstarlya_cont_dt_prefactor_MINI[R_ct],dstarlya_inj_dt_prefactor_MINI[R_ct]);
        }

        sum_lyn_prev = sum_lyn_val;
        sum_lyn_prev_MINI = sum_lyn_val_MINI;
        sum_ly2_prev = sum_ly2_val;
        sum_ly2_prev_MINI = sum_ly2_val_MINI;
        sum_lynto2_prev = sum_lynto2_val;
        sum_lynto2_prev_MINI = sum_lynto2_val_MINI;
    }
    LOG_DEBUG("R0 starlya: %.4e",dstarlya_dt_prefactor[0]);
}

//fill fftwf boxes, do the r2c transform and normalise
void prepare_filter_boxes(double redshift, float *input_dens, float *input_vcb, float *input_j21, fftwf_complex *output_dens, fftwf_complex *output_LW){
    int i,j,k,ct;
    double curr_vcb,curr_j21,M_buf;
    
    //TODO: Meraxes just applies a pointer cast box = (fftwf_complex *) input. Figure out why this works,
    //      They pad the input by a factor of 2 to cover the complex part, but from the type I thought it would be stored [(r,c),(r,c)...]
    //      Not [(r,r,r,r....),(c,c,c....)] so the alignment should be wrong, right?
    #pragma omp parallel for private(i,j,k) num_threads(user_params_ts->N_THREADS) collapse(3)
    for(i=0;i<user_params_ts->HII_DIM;i++){
        for(j=0;j<user_params_ts->HII_DIM;j++){
            for(k=0;k<HII_D_PARA;k++){
                *((float *)output_dens + HII_R_FFT_INDEX(i,j,k)) = input_dens[HII_R_INDEX(i,j,k)];
            }
        }
    }
    ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
    dft_r2c_cube(user_params_ts->USE_FFTW_WISDOM, user_params_ts->HII_DIM, HII_D_PARA, user_params_ts->N_THREADS, output_dens);
    #pragma omp parallel for num_threads(user_params_ts->N_THREADS)
    for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
        output_dens[ct] /= (float)HII_TOT_NUM_PIXELS;
    }

    if(flag_options_ts->USE_MINI_HALOS){
        curr_vcb = flag_options_ts->FIX_VCB_AVG ? global_params.VAVG : 0;
        #pragma omp parallel for firstprivate(curr_vcb) private(i,j,k,curr_j21) num_threads(user_params_ts->N_THREADS) collapse(3)
        for(i=0;i<user_params_ts->HII_DIM;i++){
            for(j=0;j<user_params_ts->HII_DIM;j++){
                for(k=0;k<HII_D_PARA;k++){
                    if(!flag_options_ts->FIX_VCB_AVG && user_params_ts->USE_RELATIVE_VELOCITIES){
                        curr_vcb = input_vcb[HII_R_INDEX(i,j,k)];
                    }
                    curr_j21 = input_j21[HII_R_INDEX(i,j,k)];
                    //NOTE: we don't use reionization_feedback here, I assume it wouldn't do much but it's inconsistent
                    M_buf = log10(lyman_werner_threshold(redshift,
                                curr_j21, curr_vcb, astro_params_ts));
                    *((float *)output_LW + HII_R_FFT_INDEX(i,j,k)) = M_buf;
                }
            }
        }
        ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
        dft_r2c_cube(user_params_ts->USE_FFTW_WISDOM, user_params_ts->HII_DIM, HII_D_PARA, user_params_ts->N_THREADS, output_LW);
        #pragma omp parallel for num_threads(user_params_ts->N_THREADS)
        for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
            output_LW[ct] /= (float)HII_TOT_NUM_PIXELS;
        }
    }      
}


//fill a box[R_ct][box_ct] array for use in TS by filtering on different scales and storing results
void fill_Rbox_table(float **result, fftwf_complex *unfiltered_box, double * R_array, int n_R, double min_value, double const_factor, double *min_arr, double *average_arr, double *max_arr){
        //allocate table/grid memory
        //NOTE: if we aren't using minimize memory (in which case we don't call this function)
        //we can just allocate and free here.
        int i,j,k,ct,R_ct;
        double R;
        double ave_buffer, min_out_R, max_out_R;

        fftwf_complex *box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        // Smooth the density field, at the same time store the minimum and maximum densities for their usage in the interpolation tables
        for(R_ct=0; R_ct<n_R; R_ct++){
            R = R_array[R_ct];
            ave_buffer = 0;
            min_out_R = 1e20; //TODO:proper starting limits
            max_out_R = -1e20;
            // copy over unfiltered box
            memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            
            // don't filter on cell size
            if (R > L_FACTOR*(user_params_ts->BOX_LEN / user_params_ts->HII_DIM)){
                filter_box(box, 1, global_params.HEAT_FILTER, R);
            }

            // now fft back to real space
            dft_c2r_cube(user_params_ts->USE_FFTW_WISDOM, user_params_ts->HII_DIM, HII_D_PARA, user_params_ts->N_THREADS, box);
            // LOG_DEBUG("Executed FFT for R=%f", R);
            // copy over the values
#pragma omp parallel private(i,j,k) num_threads(user_params_ts->N_THREADS)
            {
                float curr;
#pragma omp for reduction(+:ave_buffer) reduction(max:max_out_R) reduction(min:min_out_R)
                for (i=0;i<user_params_ts->HII_DIM; i++){
                    for (j=0;j<user_params_ts->HII_DIM; j++){
                        for (k=0;k<HII_D_PARA; k++){
                            curr = *((float *)box + HII_R_FFT_INDEX(i,j,k));

                            //NOTE: Min value is on the grid BEFORE constant factor
                            // correct for aliasing in the filtering step
                            if (curr < min_value){ 
                                curr = min_value;
                            }

                            //constant factors (i.e linear extrapolation to z=0 for dens.)
                            curr = curr * const_factor;

                            ave_buffer += curr;
                            if(curr < min_out_R) min_out_R = curr;
                            if(curr > max_out_R) max_out_R = curr;
                            result[R_ct][HII_R_INDEX(i,j,k)] = curr;
                        }
                    }
                }
            }
            average_arr[R_ct] = ave_buffer/HII_TOT_NUM_PIXELS;
            min_arr[R_ct] = min_out_R;
            max_arr[R_ct] = max_out_R;
            // LOG_DEBUG("R %d [min,mean,max] = [%.2e,%.2e,%.2e]",R_ct,min_out_R,ave_buffer,max_out_R);
        }
    fftwf_free(box);
}

//fill a box[R_ct][box_ct] array for use in TS by filtering on different scales and storing results
//Similar to fill_Rbox_table but called using different redshifts for each scale
int UpdateXraySourceBox(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options, struct HaloBox *halobox,
                  double R_inner, double R_outer, int R_ct, struct XraySourceBox *source_box){
    int status;
    Try{
        int i,j,k,ct;
        fftwf_complex *filtered_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        fftwf_complex *unfiltered_box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        fftwf_complex *filtered_box_mini = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        fftwf_complex *unfiltered_box_mini = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        if(R_ct == 0){
            LOG_DEBUG("starting XraySourceBox");
        }
        double fsfr_avg = 0;
        double fsfr_avg_mini = 0;

    #pragma omp parallel private(i,j,k) num_threads(user_params->N_THREADS)
        {
    #pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k)) = halobox->halo_sfr[HII_R_INDEX(i,j,k)];
                        *((float *)unfiltered_box_mini + HII_R_FFT_INDEX(i,j,k)) = halobox->halo_sfr_mini[HII_R_INDEX(i,j,k)];
                    }
                }
            }
        }

        ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
        //this would normally only be done once but we're using a different redshift for each R now
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA,user_params->N_THREADS, unfiltered_box);
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA,user_params->N_THREADS, unfiltered_box_mini);

        // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
        // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
    #pragma omp parallel num_threads(user_params->N_THREADS)
        {
    #pragma omp for
            for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
                unfiltered_box_mini[ct] /= (float)HII_TOT_NUM_PIXELS;
            }
        }

        // Smooth the density field, at the same time store the minimum and maximum densities for their usage in the interpolation tables
        // copy over unfiltered box
        memcpy(filtered_box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        memcpy(filtered_box_mini, unfiltered_box_mini, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

        // Don't filter on the cell scale
        if(R_ct > 0){
            filter_box_annulus(filtered_box, 1, R_inner, R_outer);
            filter_box_annulus(filtered_box_mini, 1, R_inner, R_outer);
        }

        // now fft back to real space
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, filtered_box);
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, filtered_box_mini);

        // copy over the values
    #pragma omp parallel private(i,j,k) num_threads(user_params->N_THREADS) reduction(+:fsfr_avg)
        {
            float curr,curr_mini;
    #pragma omp for
            for (i=0;i<user_params->HII_DIM; i++){
                for (j=0;j<user_params->HII_DIM; j++){
                    for (k=0;k<user_params->HII_DIM; k++){
                        curr = *((float *)filtered_box + HII_R_FFT_INDEX(i,j,k));
                        curr_mini = *((float *)filtered_box_mini + HII_R_FFT_INDEX(i,j,k));
                        // correct for aliasing in the filtering step
                        if(curr < 0.) curr = 0.;
                        if(curr_mini < 0.) curr_mini = 0.;

                        source_box->filtered_sfr[R_ct * HII_TOT_NUM_PIXELS + HII_R_INDEX(i,j,k)] = curr;
                        source_box->filtered_sfr_mini[R_ct * HII_TOT_NUM_PIXELS + HII_R_INDEX(i,j,k)] = curr_mini;
                        fsfr_avg += curr;
                        fsfr_avg_mini += curr_mini;
                    }
                }
            }
        }
        source_box->mean_sfr[R_ct] = fsfr_avg;
        source_box->mean_sfr_mini[R_ct] = fsfr_avg_mini;
        source_box->mean_log10_Mcrit_LW[R_ct] = halobox->log10_Mcrit_LW_ave;
        if(R_ct == global_params.NUM_FILTER_STEPS_FOR_Ts - 1){
            LOG_DEBUG("finished XraySourceBox");
        }
        // LOG_DEBUG("R = %8.3f | mean sfr = %10.3e (%10.3e MINI)",R_outer,fsfr_avg/HII_TOT_NUM_PIXELS,fsfr_avg_mini/HII_TOT_NUM_PIXELS);
        
        fftwf_free(filtered_box);
        fftwf_free(unfiltered_box);
        fftwf_free(filtered_box_mini);
        fftwf_free(unfiltered_box_mini);

        fftwf_forget_wisdom();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    } // End of try
    Catch(status){
        return(status);
    }
    return(0);
}

//construct the [x_e][R_ct] tables
//NOTE: these have always been interpolation tables in x_e, regardless of flags
void fill_freqint_tables(double zp, double x_e_ave, double filling_factor_of_HI_zp, double *log10_Mcrit_LW_ave){
    double lower_int_limit;
    int x_e_ct,R_ct;
    double LOG10_MTURN_INT = (double) ((LOG10_MTURN_MAX - LOG10_MTURN_MIN)) / ((double) (NMTURN - 1.));
    //TODO: Move the bisections to some static context param struct so they're calculated once
    //  However since this is only used for non-interptable cases it's hardly going to affect much
#pragma omp parallel private(R_ct,x_e_ct,lower_int_limit) num_threads(user_params_ts->N_THREADS)
    {
#pragma omp for
        //In TauX we integrate Nion from zpp to zp using the LW turnover mass at zp (predending its at zpp)
        //  Calculated from the average smoothed zp grid (from previous LW field) at radius R
        //TODO: For now, I will (kind of) mimic this behaviour by providing average Mcrit_LW at zp from the HaloBox
        //  However I want to replace this with the REAL ionised fraction which occured at the previous timesteps
        //  i.e real global history structures rather than passing averages at zpp or zhat
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            if(flag_options_ts->USE_MINI_HALOS){
                lower_int_limit = fmax(nu_tau_one_MINI(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp,
                                                        log10_Mcrit_LW_ave[R_ct], Mlim_Fstar_g, Mlim_Fesc_g, Mlim_Fstar_MINI_g,
                                                        Mlim_Fesc_MINI_g), (astro_params_ts->NU_X_THRESH)*NU_over_EV);
            }
            else{
                lower_int_limit = fmax(nu_tau_one(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp, Mlim_Fstar_g, Mlim_Fesc_g),
                                             (astro_params_ts->NU_X_THRESH)*NU_over_EV);
            }
            // set up frequency integral table for later interpolation for the cell's x_e value
            for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);
        
                //we store these to avoid calculating them in the box_ct loop
                if(x_e_ct > 0){
                    freq_int_heat_tbl_diff[x_e_ct-1][R_ct] = freq_int_heat_tbl[x_e_ct][R_ct] - freq_int_heat_tbl[x_e_ct-1][R_ct];
                    freq_int_ion_tbl_diff[x_e_ct-1][R_ct] = freq_int_ion_tbl[x_e_ct][R_ct] - freq_int_ion_tbl[x_e_ct-1][R_ct];
                    freq_int_lya_tbl_diff[x_e_ct-1][R_ct] = freq_int_lya_tbl[x_e_ct][R_ct] - freq_int_lya_tbl[x_e_ct-1][R_ct];
                }
            }
            LOG_SUPER_DEBUG("%d of %d heat: %.3e %.3e %.3e ion: %.3e %.3e %.3e lya: %.3e %.3e %.3e lower %.3e"
                ,R_ct,global_params.NUM_FILTER_STEPS_FOR_Ts
                ,freq_int_heat_tbl[0][R_ct],freq_int_heat_tbl[x_int_NXHII/2][R_ct],freq_int_heat_tbl[x_int_NXHII-1][R_ct]
                ,freq_int_ion_tbl[0][R_ct],freq_int_ion_tbl[x_int_NXHII/2][R_ct],freq_int_ion_tbl[x_int_NXHII-1][R_ct]
                ,freq_int_lya_tbl[0][R_ct],freq_int_lya_tbl[x_int_NXHII/2][R_ct],freq_int_lya_tbl[x_int_NXHII-1][R_ct], lower_int_limit);
        }
//separating the inverse diff loop to prevent a race on different R_ct (shouldn't matter)
#pragma omp for
        for(x_e_ct=0;x_e_ct<x_int_NXHII-1;x_e_ct++){
            inverse_diff[x_e_ct] = 1./(x_int_XHII[x_e_ct+1] - x_int_XHII[x_e_ct]);
        }
    }

    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
            if(isfinite(freq_int_heat_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_ion_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_lya_tbl[x_e_ct][R_ct])==0) {
                LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
//                        Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }
}

//construct a Ts table above Z_HEAT_MAX, this can happen if we are computing the first box or if we
//request a redshift above Z_HEAT_MAX
void init_first_Ts(struct TsBox * box, float *dens, float z, float zp, double *x_e_ave, double *Tk_ave, bool prev_box){
    int box_ct;
    //zp is the requested redshift, z is the perturbed field redshift
    float growth_factor_zp;
    float inverse_growth_factor_z;
    double xe, TK;

    //TODO: its probably a good idea to force zp == Z_HEAT_MAX if prev_box
    //prev_box means we enable the user options for xe and Tk, and don't calculate Ts
    if(prev_box && global_params.XION_at_Z_HEAT_MAX > 0)
        xe = global_params.XION_at_Z_HEAT_MAX;
    else
        xe = xion_RECFAST(zp,0);
        
    if(prev_box && global_params.TK_at_Z_HEAT_MAX > 0)
        TK = global_params.TK_at_Z_HEAT_MAX;
    else
        TK = T_RECFAST(zp,0);

    //these could be moved out & passed as a ratio but surely this isn't a bottleneck
    if(!prev_box){
        growth_factor_zp = dicke(zp);
        inverse_growth_factor_z = 1/dicke(z);
    }
    *x_e_ave = xe;
    *Tk_ave = TK;

#pragma omp parallel private(box_ct) num_threads(user_params_ts->N_THREADS)
    {
        double curr_xalpha;
#pragma omp for
        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
            box->Tk_box[box_ct] = TK;
            box->x_e_box[box_ct] = xe;
            // compute the spin temperature
            if(!prev_box){
                box->Ts_box[box_ct] = get_Ts(z,dens[box_ct]*inverse_growth_factor_z*growth_factor_zp,
                                            TK, xe, 0, &curr_xalpha);
            }
        }
    }
}

//calculate the global properties used for making the frequency integrals,
//  used for filling factor, ST_OVER_PS, and NO_LIGHT
//TODO: in future, this function should calculate global expected values at each zpp
//  and be used in conjunction with a function which computes the box sums to do adjustment
//  e.g: global_reion -> if(!NO_LIGHT) -> sum_box -> source *= global/box_avg
//  either globally or at each R/zpp
int global_reion_properties(double zp, double x_e_ave, double *log10_Mcrit_LW_ave, double *Q_HI, double *mean_sfr_zpp, double *mean_sfr_zpp_mini){
    int R_ct;
    double sum_nion=0,sum_sfr=0,sum_mass=0,sum_nion_mini=0;
    double Q;
    double zpp;
    
    double log10_Mcrit_width = (double) ((LOG10_MTURN_MAX - LOG10_MTURN_MIN)) / ((double) (NMTURN - 1.));
    
    //For a lot of global evolution, this code uses Nion_general. We can replace this with the halo field
    //at the same snapshot, but the nu integrals go from zp to zpp to find the tau = 1 barrier
    //so it needs the QHII in a range [zp,zpp]. I want to replace this whole thing with a global history struct but
    //I will need to change the Tau function chain.
    double determine_zpp_max;
    if(user_params_ts->USE_INTERPOLATION_TABLES){
        determine_zpp_min = zp*0.999; //global
        //NOTE: must be called after setup_z_edges for this line
        determine_zpp_max = zpp_for_evolve_list[global_params.NUM_FILTER_STEPS_FOR_Ts-1]*1.001;
        zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points_SFR-1.0); //global

        //We need the tables for the frequency integrals & mean fixing
        //TODO: These global tables confuse me, we do ~400 (x50 for mini) integrals to build the table, despite only having
        //  ~100 redshifts. The benefit of interpolating here would only matter if we keep the same table
        //  over subsequent snapshots, which we don't seem to do.
        //  The Nion table is used in nu_tau_one a lot but I think there's a better way to do that
        if(flag_options_ts->USE_MASS_DEPENDENT_ZETA){
            /* initialise interpolation of the mean collapse fraction for global reionization.*/
            initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                        astro_params_ts->ALPHA_STAR, astro_params_ts->ALPHA_STAR_MINI, astro_params_ts->ALPHA_ESC,
                                        astro_params_ts->F_STAR10, astro_params_ts->F_ESC10, astro_params_ts->F_STAR7_MINI, astro_params_ts->F_ESC7_MINI,
                                        astro_params_ts->M_TURN, flag_options_ts->USE_MINI_HALOS);

            initialise_SFRD_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                    astro_params_ts->ALPHA_STAR, astro_params_ts->ALPHA_STAR_MINI,
                                    astro_params_ts->F_STAR10, astro_params_ts->F_STAR7_MINI,astro_params_ts->M_TURN,
                                    flag_options_ts->USE_MINI_HALOS);
        }
        else{
            init_FcollTable(determine_zpp_min,determine_zpp_max,astro_params_ts,flag_options_ts);
        }
    }

    //For consistency between halo and non-halo based, the NO_LIGHT and filling_factor_zp
    //  are based on the expected global Nion. as mentioned above it would be nice to
    //  change this to a saved reionisation/sfrd history from previous snapshots
    sum_nion = EvaluateNionTs(zp,Mlim_Fstar_g,Mlim_Fesc_g);
    if(flag_options_ts->USE_MINI_HALOS){
        sum_nion_mini = EvaluateNionTs_MINI(zp,log10_Mcrit_LW_ave[0],Mlim_Fstar_MINI_g,Mlim_Fesc_MINI_g);
    }

    LOG_DEBUG("nion zp = %.3e (%.3e MINI)",sum_nion,sum_nion_mini);

    //Now global SFRD at (R_ct) for the mean fixing
    for(R_ct=0;R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts;R_ct++){
        zpp = zpp_for_evolve_list[R_ct];
        mean_sfr_zpp[R_ct] = EvaluateSFRD(zpp,Mlim_Fstar_g);
        if(flag_options_ts->USE_MINI_HALOS){
            mean_sfr_zpp_mini[R_ct] = EvaluateSFRD_MINI(zpp,log10_Mcrit_LW_ave[R_ct],Mlim_Fstar_MINI_g);
        }
    }
    
    LOG_DEBUG("sfrd zpp = %.3e (%.3e MINI) --> %.3e (%.3e MINI)",mean_sfr_zpp[0],mean_sfr_zpp_mini[0],
                mean_sfr_zpp[R_ct],mean_sfr_zpp_mini[R_ct]);

    //TODO: change to use global_params.Pop in no minihalo case?, this variable is pretty inconsistently used
    //  throughout the rest of the code mostly just assuming Pop2
    double ION_EFF_FACTOR,ION_EFF_FACTOR_MINI;
    ION_EFF_FACTOR = astro_params_ts->F_STAR10 * astro_params_ts->F_ESC10 * global_params.Pop2_ion;
    ION_EFF_FACTOR_MINI = astro_params_ts->F_STAR7_MINI * astro_params_ts->F_ESC7_MINI * global_params.Pop3_ion;

    Q = 1 - ( ION_EFF_FACTOR * sum_nion + ION_EFF_FACTOR_MINI * sum_nion_mini )/ (1.0 - x_e_ave);
    //Q is only used without MASS_DEPENDENT_ZETA, else Nion_general / interpolation tables are called for each zhat in [zp,zpp]
    //TODO: Change this in the frequency integrals
    *Q_HI = Q;
    LOG_DEBUG("Done.");

    return sum_nion + sum_nion_mini > 1e-15 ? 0 : 1; //NO_LIGHT returned
}

//TODO: replace conditional tables with the new functions
//  Also probably reuse the input grids since they aren't used again
//  apart from unfiltered density
//TODO: this will likely need further modularisation
void calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
                             float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini){
    double ave_sfrd_buf=0;
    double ave_sfrd_buf_mini=0;
    double mturn_bin_width = (double) ((LOG10_MTURN_MAX - LOG10_MTURN_MIN)) / ((double) (NMTURN - 1.));
    double fcoll_interp_min,fcoll_interp_bin_width,fcoll_interp_bin_width_inv;
    double fcoll_interp_high_min,fcoll_interp_high_bin_width,fcoll_interp_high_bin_width_inv;

    int bound_exceeded[user_params_ts->N_THREADS];
    memset(bound_exceeded,0,user_params_ts->N_THREADS*sizeof(int));
    // LOG_DEBUG("Starting fcoll -> sfrd calculation");
    //TODO: MOVE ALL THESE MASS LIMITS TO R_ct ARRAYS
    if(user_params_ts->USE_INTERPOLATION_TABLES){
        if(flag_options_ts->USE_MASS_DEPENDENT_ZETA){
            if( min_densities[R_ct]*zpp_growth[R_ct] <= -1.) {
                fcoll_interp_min = log10(global_params.MIN_DENSITY_LOW_LIMIT);
            }
            else {
                fcoll_interp_min = log10(1. + min_densities[R_ct]*zpp_growth[R_ct]);
            }
            if( max_densities[R_ct]*zpp_growth[R_ct] > global_params.CRIT_DENS_TRANSITION*1.1 ) {
                fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+global_params.CRIT_DENS_TRANSITION*1.1)-fcoll_interp_min);
            }
            else {
                fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+max_densities[R_ct]*zpp_growth[R_ct])-fcoll_interp_min);
            }
            fcoll_interp_high_min = global_params.CRIT_DENS_TRANSITION;
            fcoll_interp_high_bin_width = 1./((float)NSFR_high-1.)*(Deltac - fcoll_interp_high_min);
            fcoll_interp_high_bin_width_inv = 1./fcoll_interp_high_bin_width;
            fcoll_interp_bin_width_inv = 1./fcoll_interp_bin_width;
        }
        else{
            fcoll_interp_bin_width = (max_densities[R_ct] - min_densities[R_ct])/(dens_Ninterp-1);
            fcoll_interp_bin_width_inv = 1./fcoll_interp_bin_width;
        }
    }
    
    // LOG_DEBUG("Got table limits / mass limits");
    #pragma omp parallel num_threads(user_params_ts->N_THREADS)
    {
        int box_ct;
        double curr_dens, dens_val, fcoll_MINI_left, fcoll_MINI_right;
        double log10_Mcrit_LW_val, log10_Mcrit_LW_diff;
        int log10_Mcrit_LW_int, fcoll_int;
        double fcoll, fcoll_MINI, dfcoll;

        #pragma omp for reduction(+:ave_sfrd_buf,ave_sfrd_buf_mini)
        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
            curr_dens = dens_R_grid[box_ct]*zpp_growth[R_ct];

            //boundary cases
            if(curr_dens <= -1.){
                sfrd_grid[box_ct] = 0;
                if(flag_options_ts->USE_MINI_HALOS) sfrd_grid_mini[box_ct] = 0;
                continue;
            }
            else if(curr_dens > Deltac*0.99){
                sfrd_grid[box_ct] = 1.;
                ave_sfrd_buf += 1.;
                if(flag_options_ts->USE_MINI_HALOS){
                    sfrd_grid_mini[box_ct] = 1.;
                    ave_sfrd_buf_mini += 1.;
                }
                continue;
            }

            if (flag_options_ts->USE_MINI_HALOS && user_params_ts->USE_INTERPOLATION_TABLES){
                log10_Mcrit_LW_val = ( Mcrit_R_grid[box_ct] - LOG10_MTURN_MIN) / mturn_bin_width;
                log10_Mcrit_LW_int = (int)floorf( log10_Mcrit_LW_val );
                log10_Mcrit_LW_diff = log10_Mcrit_LW_val - (float)log10_Mcrit_LW_int;
            }

            if(user_params_ts->USE_INTERPOLATION_TABLES){
                if(flag_options_ts->USE_MASS_DEPENDENT_ZETA){
                    if (curr_dens < global_params.CRIT_DENS_TRANSITION){
                        dens_val = (log10f(curr_dens+1.) - fcoll_interp_min)*fcoll_interp_bin_width_inv;
                        fcoll_int = (int)floorf(dens_val);
                        // LOG_DEBUG("curr_dens = %.2e, val = %.2e, idx = %d",curr_dens,dens_val,fcoll_int);

                        //There was a check here for dens < transition, but fcoll_int == NSFR_low - 1
                        //  I just added buffer in the table so that they overlap slightly
                        if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_low - 1)){
                            // LOG_ERROR("Bound Exceeded at  (%d,%d) dens %.2e (%.2e) [%.2e,%d]",R_ct,box_ct,curr_dens,log10f(curr_dens+1.),dens_val,fcoll_int);
                            // LOG_ERROR("Table limits [%.2e,%.2e,%.2e]",fcoll_interp_min,fcoll_interp_bin_width,
                            //                                 fcoll_interp_min + ((NSFR_low-1)*fcoll_interp_bin_width));
                            // LOG_ERROR("array value %.2e, growth %.2e",dens_R_grid[box_ct],zpp_growth[R_ct]);
                            // LOG_ERROR("global value %.2e, growth %.2e",user_params_ts->MINIMIZE_MEMORY ? delNL0[0][box_ct] : delNL0[R_ct][box_ct]);
                            bound_exceeded[omp_get_thread_num()] = 1;
                            continue;
                        }

                        fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int]*( 1 + (float)fcoll_int - dens_val ) + \
                                log10_SFRD_z_low_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );

                        fcoll = expf(fcoll);

                        if (flag_options_ts->USE_MINI_HALOS){
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
                    else{
                        dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;
                        fcoll_int = (int)floorf( dens_val );
                        // LOG_DEBUG("curr_dens = %.2e, val = %.2e, idx = %.2e",curr_dens,dens_val,fcoll_int);

                        if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_high - 1)) {
                            // LOG_ERROR("Bound Exceeded dens %.2e (%.2e,%d)",curr_dens,dens_val,fcoll_int);
                            // LOG_ERROR("Table limits [%.2e,%.2e,%.2e]",fcoll_interp_high_min,fcoll_interp_bin_width,
                            //                                 fcoll_interp_high_min + ((NSFR_high-1)*fcoll_interp_high_bin_width));
                            bound_exceeded[omp_get_thread_num()] = 1;
                            continue;
                        }

                        fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + \
                                SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );

                        if (flag_options_ts->USE_MINI_HALOS){
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
                }
                else{
                    // LOG_DEBUG("R %d %.2e | dens %.2e min %.2e wid %.2e",R_ct,R_values[R_ct],curr_dens,min_densities[R_ct],fcoll_interp_bin_width);
                    fcoll = EvaluateRGTable1D(curr_dens,F_table_val[R_ct],min_densities[R_ct],fcoll_interp_bin_width);
                    dfcoll = EvaluateRGTable1D(curr_dens,dF_table_val[R_ct],min_densities[R_ct],fcoll_interp_bin_width);
                }
            }
            else {
                if(flag_options_ts->USE_MASS_DEPENDENT_ZETA){
                    if (flag_options_ts->USE_MINI_HALOS){
                        fcoll = Nion_ConditionalM(zpp_growth[R_ct],log(global_params.M_MIN_INTEGRAL),log(M_max_R[R_ct]),sigma_max[R_ct],Deltac,curr_dens,Mcrit_atom_interp_table[R_ct],
                                                    astro_params_ts->ALPHA_STAR,0.,astro_params_ts->F_STAR10,1.,Mlim_Fstar_g,0., user_params_ts->FAST_FCOLL_TABLES);

                        fcoll_MINI = Nion_ConditionalM_MINI(zpp_growth[R_ct],log(global_params.M_MIN_INTEGRAL),log(M_max_R[R_ct]),sigma_max[R_ct],Deltac,\
                                                curr_dens,pow(10,Mcrit_R_grid[box_ct]),Mcrit_atom_interp_table[R_ct],\
                                                astro_params_ts->ALPHA_STAR_MINI,0.,astro_params_ts->F_STAR7_MINI,1.,Mlim_Fstar_MINI_g, 0., user_params_ts->FAST_FCOLL_TABLES);

                    }
                    else {
                        fcoll = Nion_ConditionalM(zpp_growth[R_ct],log(M_min_R[R_ct]),log(M_max_R[R_ct]),sigma_max[R_ct],Deltac,curr_dens,
                                                    astro_params_ts->M_TURN, astro_params_ts->ALPHA_STAR,0.,
                                                    astro_params_ts->F_STAR10,1.,Mlim_Fstar_g,0., user_params_ts->FAST_FCOLL_TABLES);
                    }
                }
                else{
                    fcoll = FgtrM_bias_fast(zpp_growth[R_ct],curr_dens,sigma_min[R_ct],sigma_max[R_ct]);
                    dfcoll = dfcoll_dz(zpp_for_evolve_list[R_ct],sigma_min[R_ct],curr_dens,sigma_max[R_ct]);
                }
            }
            //TODO: figure out why the average is not weighted by delta
            //  Effectively it means we match the volume-weighted stellar mass fraction to the Sheth-Tormen
            //  Instead of total stellar mass / SFRD (volume weighted sum instead of mass-weighted)
            //NOTE: The default case multiplied both of these by 1e10, but then divided by 1e10 before summing into the grids
            //      So I've removed the factor, this note is here in case that was there for a reason
            ave_sfrd_buf += fcoll;
            if(flag_options_ts->USE_MASS_DEPENDENT_ZETA){
                //the tables still multiply by 1e10 for some reason, will change
                sfrd_grid[box_ct] = (1.+curr_dens)*fcoll*1e-10;
                if (flag_options_ts->USE_MINI_HALOS){
                    ave_sfrd_buf_mini += fcoll_MINI;
                    sfrd_grid_mini[box_ct] = (1.+curr_dens)*fcoll_MINI*1e-10;
                }
            }
            else{
                sfrd_grid[box_ct] = (1.+curr_dens)*dfcoll;
            }
        }
    }
    //Throwing an error in a parralel region segfaults, so this is a way to handle it "properly"
    //TODO: don't go through the whole loop if something goes wrong
    int i;
    for(i=0;i<user_params_ts->N_THREADS;i++){
        if(bound_exceeded[i]){
            LOG_ERROR("Bound Exceeded on thread %d %d",i,bound_exceeded[i]);
            Throw(TableEvaluationError);
        }
    }
    *ave_sfrd = ave_sfrd_buf/HII_TOT_NUM_PIXELS;
    *ave_sfrd_mini = ave_sfrd_buf_mini/HII_TOT_NUM_PIXELS;

    //TODO: when you move the conditional tables remove this
    if(flag_options_ts->USE_MASS_DEPENDENT_ZETA){
        *ave_sfrd *= 1e-10;
        *ave_sfrd_mini *= 1e-10;
    }
}

//TODO: this could be further split into emissivity, spintemp calculation constants
struct Ts_zp_consts{
    double xray_prefactor; //convserion from SFRD to xray emissivity
    double Trad; //CMB temperature
    double Trad_inv; //inverse for acceleration (/ slower than * sometimes)
    double Ts_prefactor; //some volume factors
    double xa_tilde_prefactor; //lyman alpha prefactor
    double xc_inverse; //collisional prefactor
    double dcomp_dzp_prefactor; //compton prefactor
    double Nb_zp; //physical critical density
    double lya_star_prefactor; //converts SFR density -> stellar baryon density + prefactors
    double volunit_inv; //inverse volume unit for cm^-3 conversion
    double hubble_zp; //H(z)
    double growth_zp;
    double dgrowth_dzp;
    double dt_dzp;
};

void set_zp_consts(double zp, struct Ts_zp_consts *consts){
    //constant prefactors for the R_ct==0 part
    LOG_DEBUG("Setting zp constants");
    double Luminosity_converstion_factor;
    if(fabs(astro_params_ts->X_RAY_SPEC_INDEX - 1.0) < 1e-6) {
        Luminosity_converstion_factor = (astro_params_ts->NU_X_THRESH)*NU_over_EV * log( global_params.NU_X_BAND_MAX/(astro_params_ts->NU_X_THRESH) );
        Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
    }
    else {
        Luminosity_converstion_factor = pow( (global_params.NU_X_BAND_MAX)*NU_over_EV , 1. - (astro_params_ts->X_RAY_SPEC_INDEX) ) - \
                                        pow( (astro_params_ts->NU_X_THRESH)*NU_over_EV , 1. - (astro_params_ts->X_RAY_SPEC_INDEX) ) ;
        Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
        Luminosity_converstion_factor *= pow( (astro_params_ts->NU_X_THRESH)*NU_over_EV, - (astro_params_ts->X_RAY_SPEC_INDEX) )*\
                                        (1 - (astro_params_ts->X_RAY_SPEC_INDEX));
    }
    // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the definition of Luminosity)
    Luminosity_converstion_factor *= (3.1556226e7)/(hplank);

    //for halos, we just want the SFR -> X-ray part
    //NOTE: compared to Mesinger+11: (1+zpp)^2 (1+zp) -> (1+zp)^3
    consts->xray_prefactor = Luminosity_converstion_factor / ((astro_params_ts->NU_X_THRESH)*NU_over_EV) \
                            * C * pow(1+zp, astro_params_ts->X_RAY_SPEC_INDEX + 3); //(1+z)^3 is here because we don't want it in the star lya (already in zpp integrand)                     

    // Required quantities for calculating the IGM spin temperature
    // Note: These used to be determined in evolveInt (and other functions). But I moved them all here, into a single location.
    consts->Trad = T_cmb*(1.0+zp);
    consts->Trad_inv = 1.0/consts->Trad;
    consts->Ts_prefactor = pow(1.0e-7*(1.342881e-7 / hubble(zp))*No*pow(1+zp,3),1./3.);
    consts->xa_tilde_prefactor = 1.66e11/(1.0+zp);

    consts->xc_inverse =  pow(1.0+zp,3.0)*T21/(consts->Trad*A10_HYPERFINE);

    consts->dcomp_dzp_prefactor = (-1.51e-4)/(hubble(zp)/Ho)/(cosmo_params_ts->hlittle)*pow(consts->Trad,4.0)/(1.0+zp);

    consts->Nb_zp = N_b0 * (1+zp)*(1+zp)*(1+zp); //used for lya_X and sinks NOTE: the 2 density factors are from source & absorber since its downscattered x-ray
    consts->lya_star_prefactor = C / FOURPI * Msun / m_p * (1 - 0.75*global_params.Y_He); //converts SFR density -> stellar baryon density + prefactors
    
    //converts the grid emissivity unit to per cm-3
    if(flag_options_ts->USE_HALO_FIELD){
        consts->volunit_inv = pow(CMperMPC,-3); //changes to emissivity per cm-3
    }
    else{
        consts->volunit_inv = cosmo_params_ts->OMb * RHOcrit * pow(CMperMPC,-3);
    }   

    consts->growth_zp = dicke(zp);
    consts->hubble_zp = hubble(zp);
    consts->dgrowth_dzp = ddicke_dz(zp);
    consts->dt_dzp = dtdz(zp);

    LOG_DEBUG("Set zp consts xr %.2e Tr %.2e Ts %.2e xa %.2e xc %.2e cm %.2e",consts->xray_prefactor,consts->Trad,
                consts->Ts_prefactor,consts->xa_tilde_prefactor,consts->xc_inverse,consts->dcomp_dzp_prefactor);
    LOG_DEBUG("Nb %.2e la %.2e vi %.2e D %.2e H %.2e dD %.2e dt %.2e",consts->Nb_zp,consts->lya_star_prefactor,
                consts->volunit_inv,consts->growth_zp,consts->hubble_zp,consts->dgrowth_dzp,consts->dt_dzp);
}

//All the cell-dependent stuff needed to calculate Ts
struct Box_rad_terms{
    double dxion_dt;
    double dxheat_dt;
    double dxlya_dt;
    double dstarlya_dt;
    double dstarLW_dt;
    double dstarlya_cont_dt;
    double dstarlya_inj_dt;
    double delta;
    double prev_Ts;
    double prev_Tk;
    double prev_xe;
};

//outputs from the Ts calculation, to go into new boxes
struct Ts_cell{
    double Ts;
    double x_e;
    double Tk;
    double J_21_LW;
};

//Function for calculating the Ts box outputs quickly by using pre-calculated constants
//  as much as possible
//TODO: unpack the structures at the start?
//  BUT surely any compiler should optimise it....
//TODO: Move more constants out of the loop
struct Ts_cell get_Ts_fast(float zp, float dzp, struct Ts_zp_consts *consts, struct Box_rad_terms *rad){
    // Now we can solve the evolution equations  //
    struct Ts_cell output;
    double tau21,xCMB,dxion_sink_dt,dxe_dzp,dadia_dzp,dspec_dzp,dcomp_dzp,dxheat_dzp;
    double dCMBheat_dzp,eps_CMB,eps_Lya_cont,eps_Lya_inj,E_continuum,E_injected,Ndot_alpha_cont,Ndot_alpha_inj;
    tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-rad->prev_xe)*consts->Nb_zp)/rad->prev_Ts/consts->hubble_zp;
    xCMB = (1. - exp(-tau21))/tau21;
    
    // First let's do dxe_dzp //
    dxion_sink_dt = alpha_A(rad->prev_Tk) * global_params.CLUMPING_FACTOR * rad->prev_xe*rad->prev_xe * f_H * consts->Nb_zp * \
                    (1.+rad->delta);
                    
    dxe_dzp = consts->dt_dzp*(rad->dxion_dt - dxion_sink_dt);

    // Next, let's get the temperature components //
    // first, adiabatic term
    dadia_dzp = 3/(1.0+zp);
    if (fabs(rad->delta) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
        dadia_dzp += consts->dgrowth_dzp/(consts->growth_zp * (1.0/rad->delta+1.0));

    dadia_dzp *= (2.0/3.0)*rad->prev_Tk;

    // next heating due to the changing species
    dspec_dzp = - dxe_dzp * rad->prev_Tk / (1+rad->prev_xe);

    // next, Compton heating
    //                dcomp_dzp = dT_comp(zp, T, x_e);
    dcomp_dzp = consts->dcomp_dzp_prefactor*(rad->prev_xe/(1.0+rad->prev_xe+f_He))*(consts->Trad - rad->prev_Tk);

    // lastly, X-ray heating
    dxheat_dzp = rad->dxheat_dt * consts->dt_dzp * 2.0 / 3.0 / k_B / (1.0+rad->prev_xe);
    //next, CMB heating rate
    dCMBheat_dzp = 0.;
    if (flag_options_ts->USE_CMB_HEATING) {
        //Meiksin et al. 2021
        eps_CMB = (3./4.) * (consts->Trad/T21) * A10_HYPERFINE * f_H * (hplank*hplank/Lambda_21/Lambda_21/m_p) * (1. + 2.*rad->prev_Tk/T21);
        dCMBheat_dzp = 	-eps_CMB * (2./3./k_B/(1.+rad->prev_xe))/consts->hubble_zp/(1.+zp);
    }
    
    //lastly, Ly-alpha heating rate
    eps_Lya_cont = 0.;
    eps_Lya_inj = 0.;
    if (flag_options_ts->USE_LYA_HEATING) {
        E_continuum = Energy_Lya_heating(rad->prev_Tk, rad->prev_Ts, taugp(zp,rad->delta,rad->prev_xe), 2);
        E_injected = Energy_Lya_heating(rad->prev_Tk, rad->prev_Ts, taugp(zp,rad->delta,rad->prev_xe), 3);
        //TODO: look into the functions and make a better isfinite check
        if (isnan(E_continuum) || isinf(E_continuum)){
            E_continuum = 0.;
        }
        if (isnan(E_injected) || isinf(E_injected)){
            E_injected = 0.;
        }
        Ndot_alpha_cont = (4.*PI*Ly_alpha_HZ) / (consts->Nb_zp*(1.+rad->delta))/(1.+zp)/C * rad->dstarlya_cont_dt;
        Ndot_alpha_inj = (4.*PI*Ly_alpha_HZ) / (consts->Nb_zp*(1.+rad->delta))/(1.+zp)/C * rad->dstarlya_cont_dt;
        eps_Lya_cont = - Ndot_alpha_cont * E_continuum * (2./3./k_B/(1. + rad->prev_xe));
        eps_Lya_inj = - Ndot_alpha_inj * E_injected * (2./3./k_B/(1. + rad->prev_xe));
    }

    //update quantities
    double x_e,Tk;
    x_e = rad->prev_xe + (dxe_dzp*dzp); // remember dzp is negative
    if (x_e > 1) // can do this late in evolution if dzp is too large
        x_e = 1 - FRACT_FLOAT_ERR;
    else if (x_e < 0)
        x_e = 0;
    //NOTE: does this stop cooling if we ever go over the limit? I suppose that shouldn't happen but it's strange anyway
    Tk = rad->prev_Tk;
    if (Tk < MAX_TK){
        if(debug_printed==0){
            LOG_SUPER_DEBUG("Heating Terms: T %.4e | X %.4e | c %.4e | S %.4e | A %.4e | c %.4e | lc %.4e | li %.4e | dz %.4e",
                                        Tk, dxheat_dzp, dcomp_dzp, dspec_dzp, dadia_dzp, dCMBheat_dzp, eps_Lya_cont, eps_Lya_inj, dzp);
        }
        Tk += (dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj) * dzp;
        if(debug_printed==0){
            LOG_SUPER_DEBUG("--> T %.4e",Tk);
            debug_printed=1;
        }
    }
    //spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
    if (Tk<0)
        Tk = consts->Trad;

    output.x_e = x_e;
    output.Tk = Tk;

    double J_alpha_tot = rad->dstarlya_dt + rad->dxlya_dt; //not really d/dz, but the lya flux
    if(flag_options_ts->USE_MINI_HALOS){
        output.J_21_LW = rad->dstarLW_dt;
    }
    else{
        output.J_21_LW = 0.;
    }

    // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
    // Algorithm is the same, but written to be more computationally efficient

    //JD: I'm leaving these as comments in case I'm wrong, but there's NO WAY a compiler doesn't know the fastest way to invert a number
    // T_inv = expf((-1.)*logf(Tk));
    // T_inv_sq = expf((-2.)*logf(Tk));
    double T_inv, T_inv_sq, xc_fast,xi_power,xa_tilde_fast_arg;
    double xa_tilde_fast, TS_fast, TSold_fast;
    T_inv = 1/Tk;
    T_inv_sq = T_inv*T_inv;

    xc_fast = (1.0+rad->delta)*consts->xc_inverse*\
            ( (1.0-x_e)*No*kappa_10(Tk,0) + x_e*N_b0*kappa_10_elec(Tk,0) + x_e*No*kappa_10_pH(Tk,0) );

    xi_power = consts->Ts_prefactor * cbrt((1.0+rad->delta)*(1.0-x_e)*T_inv_sq);

    xa_tilde_fast_arg = consts->xa_tilde_prefactor*J_alpha_tot*pow( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power, -1. );
    
    if (J_alpha_tot > 1.0e-20) { // Must use WF effect
        TS_fast = consts->Trad;
        TSold_fast = 0.0;
        while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {
            TSold_fast = TS_fast;

            xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - \
                            0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;

            TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*consts->Trad_inv+xa_tilde_fast*\
                                ( T_inv + 0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
        }
    } else { // Collisions only
        TS_fast = (xCMB + xc_fast)/(xCMB*consts->Trad_inv + xc_fast*T_inv);
        xa_tilde_fast = 0.0;
    }
    if(TS_fast < 0.) {
        // It can very rarely result in a negative spin temperature. If negative, it is a very small number.
        // Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
        TS_fast = fabs(TS_fast);
    }

    output.Ts = TS_fast;
    
    return output;
}

//outer-level function for calculating Ts based on the Halo boxes
//TODO: make sure redshift (zp), perturbed_field_redshift, zpp all consistent
//THE !USE_MASS_DEPENDENT_ZETA case used to differ in a few ways, I'm logging them here just in case:
//  - The delNL0 array was reversed [box_ct][R_ct], i.e it was filled in a strided manner and the
//      R loop for the Ts calculation was inner. There are two implications, the first being that it's
//      likely slower to fill/sum this way (it would be ~100 byte strided), and the second is that it's incompatible with MINIMIZE_MEMORY,
//      since the dxdt[box_ct] grids can't function on an inner R-loop.
//  - There was a huge R_ct x 2D list of interpolation tables allocated, I'm guessing there was a time when this was allocated
//      once at the start of the run, but this no longer seems to be the case (we don't interpolate on zpp).
//      I replace this with R_ct x 1D tables here. The Fcoll table is used for the ST_over_PS sum and dFcolldz is used for the rates.
//  - Essentially, rather than being a totally separate program, I will make this flag simply the option to forgo all power-laws
//      and exponentials in order to fill in the SFRD tables with ERFC instead of integrating, speeding things up.
//  - There was a WDM mass cutoff parameter which has been replicated, I can implement this properly in minimum_source_mass when I modularise that part
//  - The density tables were spaced in log10 between 1e-6 and the maximum
void ts_halos(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options, float perturbed_field_redshift, short cleanup,
                  struct PerturbedField *perturbed_field, struct XraySourceBox *source_box, struct TsBox *previous_spin_temp,
                  struct InitialConditions *ini_boxes, struct TsBox *this_spin_temp){
    int box_ct, R_ct, i, j, k;
    double x_e_ave_p, Tk_ave_p;
    double growth_factor_z, growth_factor_zp;
    double inverse_growth_factor_z;
    double zp, dzp, prev_zp;

    LOG_DEBUG("starting halo spintemp");
    LOG_DEBUG("input values:");
    LOG_DEBUG("redshift=%f, prev_redshift=%f perturbed_field_redshift=%f", redshift, prev_redshift, perturbed_field_redshift);

    //TODO: IoniseBox expects this to be initialised even if it's not calculated here, make a cleaner way.
    init_ps();

    //allocate the global arrays we always use
    if(!TsInterpArraysInitialised){
        alloc_global_arrays();
    }
    //NOTE: For the code to work, previous_spin_temp MUST be allocated & calculated if redshift < Z_HEAT_MAX
    //TODO: move some of these zp factors to the const struct
    growth_factor_z = dicke(perturbed_field_redshift);
    inverse_growth_factor_z = 1./growth_factor_z;
    
    growth_factor_zp = dicke(redshift);
    dzp = redshift - prev_redshift;

    zp = redshift; //TODO: remove some of these aliases
    prev_zp = prev_redshift;
    
    //setup the R_ct 1D arrays
    setup_z_edges(zp);

    if(user_params->USE_INTERPOLATION_TABLES){
        double M_MIN;
        double z_var = flag_options->USE_MASS_DEPENDENT_ZETA ? \
            redshift : zpp_for_evolve_list[global_params.NUM_FILTER_STEPS_FOR_Ts - 1];
        M_MIN = minimum_source_mass(z_var,astro_params,flag_options);
        if(user_params->FAST_FCOLL_TABLES) M_MIN = fmin(MMIN_FAST,M_MIN);
        initialiseSigmaMInterpTable(M_MIN,1e20);
    }
    
    //As far as I can tell, the only thing used from this is the X_e array
    init_heat();
    //TODO: z ~> zmax case and first_box setting should be done in wrapper initialisation
    if(redshift > global_params.Z_HEAT_MAX){
        LOG_DEBUG("redshift greater than Z_HEAT_MAX");
        init_first_Ts(this_spin_temp,perturbed_field->density,perturbed_field_redshift,redshift,&x_e_ave_p,&Tk_ave_p,false);
        return;
    }
    
    calculate_spectral_factors(zp);
    
    //Fill the R_ct,box_ct fields
    //Since we use the average Mturn for the global tables this must be done first
    //NOTE: The filtered Mturn for the previous snapshot is used for Fcoll at ALL zpp
    //  regardless of distance from current reshift, this also goes for the averages
    //NOTE: Won't the average Mturn be the same for all R, since its just filtered?
    //TODO: These R_ct x box_ct grids are the subsitute for XraySourcebox for the no-halo case
    //  It should be more efficient to replace it with that structure in future, simply calculated
    //  from the density grid at one redshift (or maybe implement the annular filtering there too)
    //  This will involve the function which computes the SFRD from filtered density and Mcrit grids
    double ave_log10_MturnLW[global_params.NUM_FILTER_STEPS_FOR_Ts];
    double min_log10_MturnLW[global_params.NUM_FILTER_STEPS_FOR_Ts];
    double max_log10_MturnLW[global_params.NUM_FILTER_STEPS_FOR_Ts];
    double ave_dens[global_params.NUM_FILTER_STEPS_FOR_Ts];
    fftwf_complex *log10_Mcrit_LW_unfiltered;
    fftwf_complex *delta_unfiltered;
    double log10_Mcrit_mol, curr_vcb;
    double max_buf=-1e20, min_buf=1e20, curr_dens;
    curr_vcb = flag_options_ts->FIX_VCB_AVG ? global_params.VAVG : 0;

    //TODO: I want to move this part of the box assignment to an XraySourceBox for consistency between
    //  halo/nohalo flags and options to use the proper perturbfield/SFRD and annular filters
    if(!flag_options->USE_HALO_FIELD){
        //copy over to FFTW, do the forward FFTs and apply constants
        delta_unfiltered = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        if(flag_options->USE_MINI_HALOS)
            log10_Mcrit_LW_unfiltered = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

        prepare_filter_boxes(redshift,perturbed_field->density,ini_boxes->lowres_vcb,previous_spin_temp->J_21_LW_box,
                            delta_unfiltered,log10_Mcrit_LW_unfiltered);
        LOG_DEBUG("Forward FFT Done");
        //fill the filtered boxes if we are storing them all
        if(!user_params->MINIMIZE_MEMORY){
            fill_Rbox_table(delNL0,delta_unfiltered,R_values,global_params.NUM_FILTER_STEPS_FOR_Ts,-1,inverse_growth_factor_z,min_densities,ave_dens,max_densities);
            if(flag_options->USE_MINI_HALOS){
                //NOTE: we are using previous_zp LW threshold for all zpp, inconsistent with the halo model
                log10_Mcrit_mol = log10(lyman_werner_threshold(zp, 0., 0.,astro_params)); //minimum turnover NOTE: should be zpp?
                fill_Rbox_table(log10_Mcrit_LW,log10_Mcrit_LW_unfiltered,R_values,global_params.NUM_FILTER_STEPS_FOR_Ts,log10_Mcrit_mol,1,min_log10_MturnLW,ave_log10_MturnLW,max_log10_MturnLW);
            }
        }
        else{
            //Previously with MINIMIZE_MEMORY, the entire FFT sequence was done JUST to get the density limits at each R
            //  I can either: Just use the limits at R_ct == 0, mulitplied by the growth factor, in which case the tables
            //  will be coarser than required (filtering should never widen limits). OR: initialise one table within the R loop,
            //  which might be slower, but surely not as slow as doing the whole FFT loop
            //TODO: I'm trying the first but will revisit
            #pragma omp parallel for private(box_ct,curr_dens) num_threads(user_params->N_THREADS)
            for(box_ct=0;box_ct<HII_TOT_NUM_PIXELS;box_ct++){
                //TODO: I could definitely find these limits in prepare_filter_boxes(), and apply the constants there instead of in fill_Rbox_table()
                //  The only thing to worry about is that the minima (which should be applied each R after c2r) has been applied BEFORE the constant
                //  i.e delta has a minima of -1 at z=0 BEFORE the inverse growth factor is applied (TODO: check if this is right, surely the minima should be
                //  applied at perturbed_redshift, not z=0? since it's linear growth)
                curr_dens = perturbed_field->density[box_ct] * inverse_growth_factor_z;
                if(flag_options->USE_MINI_HALOS){
                    if(!flag_options_ts->FIX_VCB_AVG && user_params_ts->USE_RELATIVE_VELOCITIES){
                        curr_vcb = ini_boxes->lowres_vcb[box_ct];
                        log10_Mcrit_mol += log10(lyman_werner_threshold(zp, curr_vcb, previous_spin_temp->J_21_LW_box[box_ct],astro_params)); //minimum turnover NOTE: should be zpp?
                    }
                }
                if(max_buf < curr_dens)
                    max_buf = curr_dens;
                if(min_buf > curr_dens)
                    min_buf = curr_dens;
            }
            for(R_ct=0;R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts;R_ct++){
                max_densities[R_ct] = max_buf; //growth included in the init function
                min_densities[R_ct] = min_buf;
                if(flag_options->USE_MINI_HALOS){
                    ave_log10_MturnLW[R_ct] = log10_Mcrit_mol / HII_TOT_NUM_PIXELS; //similarly using R=0 box for avg
                }
            }
        }
        LOG_DEBUG("Constructed filtered boxes.");

        //set limits for the table
        if(user_params->USE_INTERPOLATION_TABLES){
            for(R_ct=0;R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts;R_ct++){
                //The density tables have edge issues due to the single precision log
                //  So here I will add a buffer to the min/max denstities
                //TODO: CHANGE IF DENSITY LIMITS ARE EVER USED FOR SOMETHING ELSE
                //  AND REMOVE IF/WHEN WE CHANGE THE LOW/HIGH TABLES
                min_densities[R_ct] *= 1.1; //negative
                max_densities[R_ct] *= 1.1;
                // LOG_DEBUG("R_ct %d [min,max] = [%.2e,%.2e] %.2e",R_ct,min_densities[R_ct]*zpp_growth[R_ct],max_densities[R_ct]*zpp_growth[R_ct],zpp_growth[R_ct]);
                if (flag_options->USE_MINI_HALOS){
                    Mcrit_atom_interp_table[R_ct] = atomic_cooling_threshold(zpp_for_evolve_list[R_ct]);
                }
            }
        }

        //mass limits required for no interp table case, also for FgtrM table limits
        //NOTE: no FAST_FCOLL here
        for(R_ct=0;R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts;R_ct++){
            //NOTE: not sure why this is the minimum mass here but I just moved it here
            M_min_R[R_ct] = minimum_source_mass(zpp_for_evolve_list[R_ct],astro_params,flag_options);
            M_max_R[R_ct] = RtoM(R_values[R_ct]);
            sigma_min[R_ct] = EvaluateSigma(log(M_min_R[R_ct]),0,NULL);
            sigma_max[R_ct] = EvaluateSigma(log(M_max_R[R_ct]),0,NULL);
            
            // LOG_DEBUG("R %d = %.2e z %.2e || M = [%.2e, %.2e] sig [%.2e %.2e]",R_ct,R_values[R_ct],
            //             zpp_for_evolve_list[R_ct],M_min_R[R_ct],M_max_R[R_ct],sigma_min[R_ct],sigma_max[R_ct]);
        }
        LOG_DEBUG("found limits");
        //currently, we need the density limits to setup the interp tables
        //  Since the lower conditional density table is in log10
        //TODO: consider the actual benefits of the log10 delta spacing
        //  and Either set a global limit for the table OR lienar spacing
        //  So we can re-use the two R_ct * box_ct grids as SFRD
        if(user_params->USE_INTERPOLATION_TABLES){
            if(flag_options->USE_MASS_DEPENDENT_ZETA){    
                if (!flag_options->USE_MINI_HALOS){
                    initialise_SFRD_Conditional_table(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,
                                                        max_densities,zpp_growth,R_values,astro_params->M_TURN,
                                                        astro_params->ALPHA_STAR, astro_params->F_STAR10, user_params->FAST_FCOLL_TABLES);
                }
                else{
                    initialise_SFRD_Conditional_table_MINI(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,
                                                            max_densities,zpp_growth,R_values,Mcrit_atom_interp_table,
                                                            astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI, astro_params->F_STAR10,
                                                            astro_params->F_STAR7_MINI, user_params->FAST_FCOLL_TABLES);
                }
            }
            else{
                initialise_FgtrM_delta_table(global_params.NUM_FILTER_STEPS_FOR_Ts, min_densities, max_densities, zpp_for_evolve_list, zpp_growth, sigma_min, sigma_max);
            }
        }
        else{
            //These are still re-calculated internally in each table initialisation
            Mlim_Fstar_g = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_ts->ALPHA_STAR, astro_params_ts->F_STAR10);
            Mlim_Fesc_g = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_ts->ALPHA_ESC, astro_params_ts->F_ESC10);
            if(flag_options_ts->USE_MINI_HALOS){
                Mlim_Fstar_MINI_g = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_ts->ALPHA_STAR_MINI,
                                                        astro_params_ts->F_STAR7_MINI * pow(1e3, astro_params_ts->ALPHA_STAR_MINI));
                Mlim_Fesc_MINI_g = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_ts->ALPHA_ESC,
                                                    astro_params_ts->F_ESC7_MINI * pow(1e3, astro_params_ts->ALPHA_ESC));
            }
        }
    }
    LOG_DEBUG("Initialised conditional tables.");

    //set the constants calculated once per snapshot
    struct Ts_zp_consts zp_consts;
    set_zp_consts(zp,&zp_consts);
    
    // LOG_DEBUG("Set consts.");

    x_e_ave_p = Tk_ave_p = 0.0;
    #pragma omp parallel num_threads(user_params->N_THREADS)
    {
    #pragma omp for reduction(+:x_e_ave_p,Tk_ave_p)
        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
            x_e_ave_p += previous_spin_temp->x_e_box[box_ct];
            Tk_ave_p += previous_spin_temp->Tk_box[box_ct];
        }
    }
    x_e_ave_p /= (float)HII_TOT_NUM_PIXELS;
    Tk_ave_p /= (float)HII_TOT_NUM_PIXELS; // not used?
    LOG_DEBUG("Prev Box: x_e_ave %.3e | TK_ave %.3e",x_e_ave_p,Tk_ave_p);

    int NO_LIGHT;
    double filling_factor_of_HI_zp;
    
    //this should initialise and use the global tables (given box average turnovers)
    //  and use them to give: Filling factor at zp (only used for !MASS_DEPENDENT_ZETA to get ion_eff)
    //  global SFRD at each filter radius (numerator of ST_over_PS factor)
    double mean_sfr_zpp[global_params.NUM_FILTER_STEPS_FOR_Ts];
    double mean_sfr_zpp_mini[global_params.NUM_FILTER_STEPS_FOR_Ts];
    
    //a bit of an awkward assignment, should be fixed when I move the no-halo filtering to an XraySourceBox
    double *log10_Mcrit_LW_ave_zpp;
    log10_Mcrit_LW_ave_zpp = flag_options_ts->USE_HALO_FIELD ? source_box->mean_log10_Mcrit_LW : ave_log10_MturnLW; 

    NO_LIGHT = global_reion_properties(redshift,x_e_ave_p,log10_Mcrit_LW_ave_zpp,&filling_factor_of_HI_zp,mean_sfr_zpp,mean_sfr_zpp_mini);

    //Initialise freq tables & prefactors (x_e by R tables)
    fill_freqint_tables(zp,x_e_ave_p,filling_factor_of_HI_zp,log10_Mcrit_LW_ave_zpp);
    LOG_DEBUG("done freqint.");

    //boxes that are independent of R (for interpolation of the nu integrals)
    //NOTE: Frequency integrals are based on PREVIOUS XHII
    //  The x_e tables are not regular, hence the precomputation of indices/interp points
    #pragma omp parallel private(box_ct) num_threads(user_params->N_THREADS)
    {
        float xHII_call;
        #pragma omp for
        for(box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
            xHII_call = previous_spin_temp->x_e_box[box_ct];
            // Check if ionized fraction is within boundaries; if not, adjust to be within
            if (xHII_call > x_int_XHII[x_int_NXHII-1]*0.999){
                xHII_call = x_int_XHII[x_int_NXHII-1]*0.999;
            }
            else if (xHII_call < x_int_XHII[0]) {
                xHII_call = 1.001*x_int_XHII[0];
            }
            //these are the index and interpolation term, moved outside the R loop and stored to not calculate them R times
            m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
            inverse_val_box[box_ct] = (xHII_call - x_int_XHII[m_xHII_low_box[box_ct]])*inverse_diff[m_xHII_low_box[box_ct]];

            //initialise += boxes (memory sometimes re-used)
            dxheat_dt_box[box_ct] = 0.;
            dxion_source_dt_box[box_ct] = 0.;
            dxlya_dt_box[box_ct] = 0.;
            dstarlya_dt_box[box_ct] = 0.;
            if(flag_options_ts->USE_MINI_HALOS)
                dstarlyLW_dt_box[box_ct] = 0.;
            if(flag_options->USE_LYA_HEATING){
                dstarlya_cont_dt_box[box_ct] = 0.;
                dstarlya_inj_dt_box[box_ct] = 0.;
            }
        }
    }
    LOG_DEBUG("done init.");

    //MAIN LOOP: SFR -> heating terms with freq integrals
    double z_edge_factor, dzpp_for_evolve, zpp, xray_R_factor;
    double J_alpha_ave,xheat_ave,xion_ave,Ts_ave,Tk_ave,x_e_ave;
    J_alpha_ave=xheat_ave=xion_ave=Ts_ave=Tk_ave=x_e_ave=0;
    double J_LW_ave=0., eps_lya_cont_ave=0, eps_lya_inj_ave=0;
    double lyacont_factor_mini=0.,lyainj_factor_mini=0.,starlya_factor_mini=0.;
    double ave_fcoll,ave_fcoll_MINI;
    double avg_fix_term=1.;
    double avg_fix_term_MINI=1.;
    double min_d_buf,ave_d_buf,max_d_buf;
    int R_index;
    float *delta_box_input;
    float *Mcrit_box_input;

    //if we have stars, fill in the heating term boxes
    if(!NO_LIGHT) {
        for(R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
            dzpp_for_evolve = dzpp_list[R_ct];
            zpp = zpp_for_evolve_list[R_ct];
            //TODO: check the edge factor again in the annular filter situation
            //  The integral of that filter is not 1
            //TODO: also remove the fabs and make sure signs are correct following
            //  dzpp is negative, as should dtdz be, look in get_Ts_fast()
            //TODO: hubble array instead of call
            if(flag_options->USE_HALO_FIELD)
                z_edge_factor = fabs(dzpp_for_evolve * dtdz_list[R_ct]); //dtdz'' dz'' -> dR for the radius sum (C included in constants)
            else if(flag_options->USE_MASS_DEPENDENT_ZETA)
                z_edge_factor = fabs(dzpp_for_evolve * dtdz_list[R_ct]) * hubble(zpp) / astro_params->t_STAR;
            else
                z_edge_factor = dzpp_for_evolve;

            xray_R_factor = pow(1+zpp,-(astro_params->X_RAY_SPEC_INDEX));
            //index for grids
            R_index = user_params->MINIMIZE_MEMORY ? 0 : R_ct;

            //TODO: we don't use the filtered density / Mcrit tables after this, It would be a good idea to re-use them
            // as SFR and SFR_MINI grids and move this outside the R loop if !MINIMIZE_MEMORY
            //  This should be solved by simply moving to XraySourceBox
            if(!flag_options->USE_HALO_FIELD){
                if(user_params->MINIMIZE_MEMORY) {
                    //we call the filtering functions once here per R
                    //This unnecessarily allocates and frees a fftwf box every time but surely that's not a bottleneck
                    fill_Rbox_table(delNL0,delta_unfiltered,&(R_values[R_ct]),1,-1,inverse_growth_factor_z,&min_d_buf,&ave_d_buf,&max_d_buf);
                    // LOG_DEBUG("R=%d, min,max,avg delta = (%.2e,%.2e,%.2e)",R_ct,min_d_buf,ave_d_buf,max_d_buf);
                    // LOG_DEBUG("Table bounds = (%.2e,%.2e)",min_densities[R_ct],max_densities[R_ct]);
                    if(flag_options->USE_MINI_HALOS){
                        fill_Rbox_table(log10_Mcrit_LW,log10_Mcrit_LW_unfiltered,&(R_values[R_ct]),1,0,1,min_log10_MturnLW,ave_log10_MturnLW,max_log10_MturnLW);
                    }
                }
                //set input pointers (doing things this way helps with flag flexibility)
                delta_box_input = delNL0[R_index];
                if(flag_options->USE_MINI_HALOS){
                    Mcrit_box_input = log10_Mcrit_LW[R_index];
                }
                calculate_sfrd_from_grid(R_ct,delta_box_input,Mcrit_box_input,del_fcoll_Rct,del_fcoll_Rct_MINI,&ave_fcoll,&ave_fcoll_MINI);
                avg_fix_term = mean_sfr_zpp[R_ct]/ave_fcoll; //THE SFRD table multiplies by 1e10 for some reason, which is hidden by this mean fixing
                if(flag_options->USE_MINI_HALOS) avg_fix_term_MINI = mean_sfr_zpp_mini[R_ct]/ave_fcoll_MINI;
                // LOG_DEBUG("z %6.2f ave sfrd val %.3e avg global %.3e (MINI %.2e %.2e)",zpp_for_evolve_list[R_ct],ave_fcoll,
                //                     mean_sfr_zpp[R_ct],ave_fcoll_MINI,mean_sfr_zpp_mini[R_ct]);
            }

            //minihalo factors should be separated since they may not be allocated
            //TODO: arrays < 100 should probably always be allocated on the stack
            if(flag_options->USE_MINI_HALOS){
                starlya_factor_mini = dstarlya_dt_prefactor_MINI[R_ct];
                lyacont_factor_mini = dstarlya_cont_dt_prefactor_MINI[R_ct];
                lyainj_factor_mini = dstarlya_inj_dt_prefactor_MINI[R_ct];
            }

            //in ComputeTS, there are prefactors which depend on the sum of stellar mass (to do the ST_OVER_PS part) so they have to be computed and stored separately
            //I don't need those here (although ST_OVER_PS hides some R-dependent factors which I define above)
            #pragma omp parallel private(box_ct) num_threads(user_params->N_THREADS)
            {
                //private variables
                int xidx;
                double ival, sfr_term, xray_sfr, x_e, T;
                double sfr_term_mini=0;
                #pragma omp for 
                for(box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
                    //sum each R contribution together
                    //NOTE: the original code had separate grids for minihalos, which were simply summed afterwards, I've combined them here since I can't
                    //  see a good reason for them to be separated (other than some floating point strangeness? i.e sum all the small numbers and big numbers separately)
                    //The dxdt boxes exist for two reasons. Firstly it allows the MINIMIZE_MEMORY to work (replaces ~40*NUM_PIXELS with ~4-16*NUM_PIXELS), as the FFT is done in the R-loop.
                    //  Secondly, it is likely faster to fill these boxes, convert to SFRD, and sum with a outer R loop.

                    if(flag_options->USE_HALO_FIELD){
                        sfr_term = source_box->filtered_sfr[R_index*HII_TOT_NUM_PIXELS + box_ct] * z_edge_factor;
                    }
                    else{
                        //NOTE: for !USE_MASS_DEPENDENT_ZETA, F_STAR10 is still used for constant stellar fraction
                        //TODO: check if this was intended since it is nowhere else in the code
                        sfr_term = del_fcoll_Rct[box_ct] * z_edge_factor * avg_fix_term * astro_params->F_STAR10;
                    }
                    if(flag_options->USE_MINI_HALOS){
                        if(flag_options->USE_HALO_FIELD){
                            sfr_term_mini = source_box->filtered_sfr_mini[R_ct*HII_TOT_NUM_PIXELS + box_ct] * z_edge_factor;
                        }
                        else{
                            sfr_term_mini = del_fcoll_Rct_MINI[box_ct] * z_edge_factor * avg_fix_term_MINI * astro_params_ts->F_STAR7_MINI;
                        }
                        dstarlyLW_dt_box[box_ct] += sfr_term*dstarlyLW_dt_prefactor[R_ct] + sfr_term_mini*dstarlyLW_dt_prefactor_MINI[R_ct];
                    }

                    xray_sfr = (sfr_term*astro_params->L_X + sfr_term_mini*astro_params->L_X_MINI);
                    xidx = m_xHII_low_box[box_ct];
                    ival = inverse_val_box[box_ct];
                    dxheat_dt_box[box_ct] += xray_sfr * xray_R_factor * (freq_int_heat_tbl_diff[xidx][R_ct] * ival + freq_int_heat_tbl[xidx][R_ct]);
                    dxion_source_dt_box[box_ct] += xray_sfr * xray_R_factor * (freq_int_ion_tbl_diff[xidx][R_ct] * ival + freq_int_ion_tbl[xidx][R_ct]);
                    dxlya_dt_box[box_ct] += xray_sfr * xray_R_factor * (freq_int_lya_tbl_diff[xidx][R_ct] * ival + freq_int_lya_tbl[xidx][R_ct]);
                    dstarlya_dt_box[box_ct] += sfr_term*dstarlya_dt_prefactor[R_ct] + sfr_term_mini*starlya_factor_mini; //the MINI factors might not be allocated

                    if(flag_options->USE_LYA_HEATING){
                        dstarlya_cont_dt_box[box_ct] += sfr_term*dstarlya_cont_dt_prefactor[R_ct] + sfr_term_mini*lyacont_factor_mini;
                        dstarlya_inj_dt_box[box_ct] += sfr_term*dstarlya_inj_dt_prefactor[R_ct] + sfr_term_mini*lyainj_factor_mini;
                    }
                    if(box_ct==0){
                        LOG_SUPER_DEBUG("Cell0 R=%.1f (%.3f) || xh %.2e | xi %.2e | xl %.2e | sl %.2e | ct %.2e | ij %.2e",R_values[R_ct],zpp_for_evolve_list[R_ct],dxheat_dt_box[box_ct],
                                        dxion_source_dt_box[box_ct],dxlya_dt_box[box_ct],dstarlya_dt_box[box_ct],dstarlya_cont_dt_box[box_ct],dstarlya_inj_dt_box[box_ct]);
                    }
                }
            }
        }
    }
    //R==0 part
#pragma omp parallel private(box_ct)
    {
        double curr_delta;
        struct Ts_cell ts_cell;
        struct Box_rad_terms rad;
        #pragma omp for reduction(+:J_alpha_ave,xheat_ave,xion_ave,Ts_ave,Tk_ave,x_e_ave,eps_lya_cont_ave,eps_lya_inj_ave)
        for(box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
            curr_delta = perturbed_field->density[box_ct] * growth_factor_zp * inverse_growth_factor_z; //map density to z' TODO: look at why this is an option
            //this corrected for aliasing before, but sometimes there are still some delta==-1 cells
            //  which breaks the adiabatic part, TODO: check out the perturbed field calculations to find out why
            if (curr_delta <= -1){
                curr_delta = -1+FRACT_FLOAT_ERR;
            }
            
            //NOTE: if the filtered grids have sfr density, and the factors convert to emissivity density, surely I need a 1/(1+delta_curr)
            //  factor here for absorber density (i.e overdense regions need more radiation to have the same effects)
            //  What am I missing? The ionisation box has a final delta dependence of (1+delta_source)/(1+delta_absorber) which makes sense
            //  But here it's just (1+delta_source).
            //  mean_sfr_zpp has no rhoc (divided out), The tables have no rhoc, there is only one in the constant factor, and a 1+delta_source in the grids

            //Add prefactors that don't depend on R
            rad.dxheat_dt = dxheat_dt_box[box_ct] * zp_consts.xray_prefactor * zp_consts.volunit_inv;
            rad.dxion_dt = dxion_source_dt_box[box_ct] * zp_consts.xray_prefactor * zp_consts.volunit_inv;
            rad.dxlya_dt = dxlya_dt_box[box_ct] * zp_consts.xray_prefactor * zp_consts.volunit_inv * zp_consts.Nb_zp * (1+curr_delta); //2 density terms from downscattering absorbers
            rad.dstarlya_dt = dstarlya_dt_box[box_ct] * zp_consts.lya_star_prefactor * zp_consts.volunit_inv;
            rad.delta = curr_delta;
            if(flag_options->USE_MINI_HALOS){
                rad.dstarLW_dt = dstarlyLW_dt_box[box_ct] * zp_consts.lya_star_prefactor * zp_consts.volunit_inv * hplank * 1e21;
            }
            if(flag_options->USE_LYA_HEATING){
                rad.dstarlya_cont_dt = dstarlya_cont_dt_box[box_ct] * zp_consts.lya_star_prefactor * zp_consts.volunit_inv;
                rad.dstarlya_inj_dt = dstarlya_inj_dt_box[box_ct] * zp_consts.lya_star_prefactor * zp_consts.volunit_inv;
            }
            rad.prev_Ts = previous_spin_temp->Ts_box[box_ct];
            rad.prev_Tk = previous_spin_temp->Tk_box[box_ct];
            rad.prev_xe = previous_spin_temp->x_e_box[box_ct];

            ts_cell = get_Ts_fast(zp,dzp,&zp_consts,&rad);
            this_spin_temp->Ts_box[box_ct] = ts_cell.Ts;
            this_spin_temp->Tk_box[box_ct] = ts_cell.Tk;
            this_spin_temp->x_e_box[box_ct] = ts_cell.x_e;
            this_spin_temp->J_21_LW_box[box_ct] = ts_cell.J_21_LW;

            // Single cell debug
            if(box_ct==0){
                LOG_SUPER_DEBUG("Cell0: delta: %.3e | xheat: %.3e | dxion: %.3e | dxlya: %.3e | dstarlya: %.3e",curr_delta
                    ,rad.dxheat_dt,rad.dxion_dt,rad.dxlya_dt,rad.dstarlya_dt);
                if(flag_options_ts->USE_LYA_HEATING){
                    LOG_SUPER_DEBUG("Lya inj %.3e | Lya cont %.3e",rad.dstarlya_inj_dt,rad.dstarlya_cont_dt);
                }
                if(flag_options_ts->USE_MINI_HALOS){
                    LOG_SUPER_DEBUG("LyW %.3e",rad.dstarLW_dt);
                }
                LOG_SUPER_DEBUG("Ts %.3e Tk %.3e x_e %.3e J_21_LW %.3e",ts_cell.Ts,ts_cell.Tk,ts_cell.x_e,ts_cell.J_21_LW);
            }
            
            if(LOG_LEVEL >= DEBUG_LEVEL){
                J_alpha_ave += rad.dxlya_dt + rad.dstarlya_dt;
                xheat_ave += rad.dxheat_dt;
                xion_ave += rad.dxion_dt;
                Ts_ave += ts_cell.Ts;
                Tk_ave += ts_cell.Tk;
                J_LW_ave += ts_cell.J_21_LW;
                eps_lya_inj_ave += rad.dstarlya_cont_dt;
                eps_lya_cont_ave += rad.dstarlya_inj_dt;
            }
            x_e_ave += ts_cell.x_e;
        }
    }

    if(LOG_LEVEL >= DEBUG_LEVEL){
        x_e_ave /= (double)HII_TOT_NUM_PIXELS;
        Ts_ave /= (double)HII_TOT_NUM_PIXELS;
        Tk_ave /= (double)HII_TOT_NUM_PIXELS;
        J_alpha_ave /= (double)HII_TOT_NUM_PIXELS;
        xheat_ave /= (double)HII_TOT_NUM_PIXELS;
        xion_ave /= (double)HII_TOT_NUM_PIXELS;

        LOG_DEBUG("AVERAGES zp = %.2e Ts = %.2e x_e = %.2e Tk %.2e",zp,Ts_ave,x_e_ave,Tk_ave);
        LOG_DEBUG("J_alpha = %.2e xheat = %.2e xion = %.2e",J_alpha_ave,xheat_ave,xion_ave);
        if (flag_options->USE_MINI_HALOS){
            J_LW_ave /= (double)HII_TOT_NUM_PIXELS;
            LOG_DEBUG("J_LW %.2e",J_LW_ave/1e21);
        }
        if (flag_options->USE_LYA_HEATING){
            eps_lya_cont_ave /= (double)HII_TOT_NUM_PIXELS;
            eps_lya_inj_ave /= (double)HII_TOT_NUM_PIXELS;
            LOG_DEBUG("eps_cont %.2e eps_inj %.2e",eps_lya_cont_ave,eps_lya_inj_ave);
        }
    }

    for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
        if(isfinite(this_spin_temp->Ts_box[box_ct])==0) {
            LOG_ERROR("Estimated spin temperature is either infinite of NaN!"
                "idx %d delta %.3e dxheat %.3e dxion %.3e dxlya %.3e dstarlya %.3e",box_ct,perturbed_field->density[box_ct]
                            ,dxheat_dt_box[box_ct],dxion_source_dt_box[box_ct],dxlya_dt_box[box_ct],dstarlya_dt_box[box_ct]);
//                Throw(ParameterError);
            Throw(InfinityorNaNError);
        }
    }

    if (flag_options->USE_MINI_HALOS){
        fftwf_free(log10_Mcrit_LW_unfiltered);
    }
    if(!flag_options->USE_HALO_FIELD){
        fftwf_free(delta_unfiltered);
        fftwf_forget_wisdom();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    }

    if(cleanup){    
        if(user_params->USE_INTERPOLATION_TABLES)
            FreeTsInterpolationTables(flag_options);
        
        free_global_arrays();
    }

    return;
}