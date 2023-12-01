//New file to store functions that deal with the interpolation tables, since they are used very often.
//  We use regular grid tables since they are faster to evaluate (we always know which bin we are in)
//  So I'm making a general function for the 1D and 2D cases
//TODO: find out if the compiler optimizes this if it's called with the same x or y, (first 3 points are the same)
//  Because there are some points such as the frequency integral tables where we evaluate an entire column of one 2D table
//  Otherwise write a function which saves the left_edge and right_edge for faster evaluation

//definitions from ps.c
//ps.c is currently included first
//MOVE THESE WHEN YOU MOVE THE COND TABLES
// #define NMTURN 50//100
// #define LOG10_MTURN_MAX ((double)(10))
// #define LOG10_MTURN_MIN ((double)(5.-9e-8))

static struct UserParams * user_params_it;

void Broadcast_struct_global_IT(struct UserParams *user_params){
    user_params_it = user_params;
}

struct RGTable1D{
    int n_bin;
    double x_min;
    double x_width;

    double *y_arr;
};

struct RGTable2D{
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    double **z_arr;

    double saved_ll, saved_ul; //for future acceleration
};

double EvaluateRGTable1D(double x, double *y_arr, double x_min, double x_width){
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(double)idx;
    double interp_point = (x - table_val)/x_width;
    // LOG_DEBUG("1D: x %.6e -> idx %d -> tbl %.6e -> itp %.6e",x,idx,table_val,interp_point);

    double result = y_arr[idx]*(1-interp_point) + y_arr[idx+1]*(interp_point);

    // LOG_DEBUG("-> result %.2e",result);

    return result;
}

//some tables are floats but I still need to return doubles
double EvaluateRGTable1D_f(double x, float *y_arr, float x_min, float x_width){
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(float)idx;
    double interp_point = (x - table_val)/x_width;

    return y_arr[idx]*(1-interp_point) + y_arr[idx+1]*(interp_point);
}

double EvaluateRGTable2D(double x, double y, double **z_arr, double x_min, double x_width, double y_min, double y_width){
    int x_idx = (int)floor((x - x_min)/x_width);
    int y_idx = (int)floor((y - y_min)/y_width);

    double x_table = x_min + x_width*(double)x_idx;
    double y_table = y_min + y_width*(double)y_idx;

    double interp_point_x = (x - x_table)/x_width;
    double interp_point_y = (y - y_table)/y_width;

    double left_edge, right_edge, result;

    // LOG_DEBUG("2D Interp: val (%.2e,%.2e) min (%.2e,%.2e) wid (%.2e,%.2e)",x,y,x_min,y_min,x_width,y_width);
    // LOG_DEBUG("2D Interp: idx (%d,%d) tbl (%.2e,%.2e) itp (%.2e,%.2e)",x_idx,y_idx,x_table,y_table,interp_point_x,interp_point_y);
    // LOG_DEBUG("2D Interp: table cornders (%.2e,%.2e,%.2e,%.2e)",z_arr[x_idx][y_idx],z_arr[x_idx][y_idx+1],z_arr[x_idx+1][y_idx],z_arr[x_idx+1][y_idx+1]);

    left_edge = z_arr[x_idx][y_idx]*(1-interp_point_y) + z_arr[x_idx][y_idx+1]*(interp_point_y);
    right_edge = z_arr[x_idx+1][y_idx]*(1-interp_point_y) + z_arr[x_idx+1][y_idx+1]*(interp_point_y);

    result = left_edge*(1-interp_point_x) + right_edge*(interp_point_x);
    // LOG_DEBUG("result %.6e",result);

    return result;
}

//I'm beginning to move the interpolation table initialisation here from ps.c
//  For now, we will keep them as globals but eventually moving to static structs will be ideal

//Global Nion (z,[Mcrit_LW]) tables
double *z_val, *Nion_z_val, **Nion_z_val_MINI;
//Global SFRD (z,[Mcrit_LW]) tables NOTE: spintemp assumes z_val == z_X_val
double *z_X_val, *SFRD_val, **SFRD_val_MINI;
//Fcoll table for old parametrisation (list of 1D tables)
double **F_table_dens, **F_table_val, **dF_table_val;

//NOTE: this table is initialised for up to N_redshift x N_Mturn, but only called N_filter times to assign ST_over_PS in Spintemp. 
//  It may be better to just do the integrals at each R
void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI, float mturn_a_const, int minihalos){
    int i,j;
    float Mmin, Mmax;
    Mmin = minihalos ? global_params.M_MIN_INTEGRAL : mturn_a_const/50.;
    Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fstar_MINI;

    LOG_DEBUG("initing SFRD spline from %.2f to %.2f",zmin,zmax);

    if (z_X_val == NULL){
        z_X_val = calloc(Nbin,sizeof(double));
        SFRD_val = calloc(Nbin,sizeof(double));
        if(minihalos){
            SFRD_val_MINI = calloc(Nbin,sizeof(double*));
            for(i=0;i<Nbin;i++){
                SFRD_val_MINI[i] = calloc(NMTURN,sizeof(double));
            }
        }
    }
    
    float MassTurnover[NMTURN];
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    if(minihalos){
        Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));
        for (j=0;j<NMTURN;j++){
            MassTurnover[j] = pow(10., LOG10_MTURN_MIN + (float)j/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
        }
    }

    #pragma omp parallel private(i,j) num_threads(user_params_it->N_THREADS)
    {
        float Mcrit_atom_val = mturn_a_const;
        #pragma omp for
        for (i=0; i<Nbin; i++){
            z_X_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin); //NOTE: currently breaks if z_X_val != z_val, due to the implementation in spintemp
            if(minihalos)Mcrit_atom_val = atomic_cooling_threshold(z_X_val[i]);
            SFRD_val[i] = Nion_General(z_X_val[i], Mmin, Mcrit_atom_val, Alpha_star, 0., Fstar10, 1.,Mlim_Fstar,0.);
            if(minihalos){
                for (j=0; j<NMTURN; j++){
                    SFRD_val_MINI[i][j] = Nion_General_MINI(z_X_val[i], Mmin, MassTurnover[j], Mcrit_atom_val, Alpha_star_mini, 0., Fstar7_MINI, 1.,Mlim_Fstar_MINI,0.);
                    // LOG_DEBUG("SFRD %d %d : z = %.2e Mt = %.2e val = %.2e",i,j,z_X_val[i],MassTurnover[j],SFRD_val_MINI[i][j]);
                }
            }
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(SFRD_val[i])==0) {
            i = Nbin;
            LOG_ERROR("Detected either an infinite or NaN value in SFRD_val");
            Throw(TableGenerationError);
        }
        if(minihalos){
            for (j=0; j<NMTURN; j++){
                if(isfinite(SFRD_val_MINI[i][j])==0) {
                    j = NMTURN;
                    LOG_ERROR("Detected either an infinite or NaN value in SFRD_val_MINI");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

//Unlike the SFRD spline, this one is used more due to the nu_tau_one() rootfind
void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10,
                                float Fesc10, float Fstar7_MINI, float Fesc7_MINI, float mturn_a_const, int minihalos){
    int i,j;
    //SigmaMInterp table has different limits with minihalos
    //TODO: make uniform
    float Mmin, Mmax;
    Mmin = minihalos ? global_params.M_MIN_INTEGRAL : mturn_a_const / 50;
    Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fesc, Mlim_Fstar_MINI, Mlim_Fesc_MINI;
    LOG_DEBUG("initing Nion spline from %.2f to %.2f",zmin,zmax);

    if (z_val == NULL){
        z_val = calloc(Nbin,sizeof(double));
        Nion_z_val = calloc(Nbin,sizeof(double));
        if(minihalos){
            Nion_z_val_MINI = calloc(Nbin,sizeof(double*));
            for(i=0;i<Nbin;i++){
                Nion_z_val_MINI[i] = calloc(NMTURN,sizeof(double));
            }
        }
    }
    LOG_DEBUG("allocd");

    float MassTurnover[NMTURN];
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fesc = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc10);
    if(minihalos){
        Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));
        Mlim_Fesc_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc7_MINI * pow(1e3, Alpha_esc));
        for (j=0;j<NMTURN;j++){
            MassTurnover[j] = pow(10., LOG10_MTURN_MIN + (float)j/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
        }
    }
    LOG_DEBUG("found mlim");

#pragma omp parallel private(i,j) num_threads(user_params_it->N_THREADS)
    {
        float Mcrit_atom_val = mturn_a_const;
#pragma omp for
        for (i=0; i<Nbin; i++){
            z_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin);
            if(minihalos) Mcrit_atom_val = atomic_cooling_threshold(z_val[i]);

            Nion_z_val[i] = Nion_General(z_val[i], Mmin, Mcrit_atom_val, Alpha_star, Alpha_esc, Fstar10, Fesc10, Mlim_Fstar, Mlim_Fesc);
            if(minihalos){
                for (j=0; j<NMTURN; j++){
                    // LOG_DEBUG("(%d,%d) -> z%.2e Ma%.2e Mm%.2e",i,j,z_val[i],Mcrit_atom_val,MassTurnover[j]);
                    Nion_z_val_MINI[i][j] = Nion_General_MINI(z_val[i], Mmin, MassTurnover[j], Mcrit_atom_val, Alpha_star_mini, Alpha_esc, Fstar7_MINI, Fesc7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI);
                }
            }
        }
    }
    LOG_DEBUG("filled tables");

    for (i=0; i<Nbin; i++){
        if(isfinite(Nion_z_val[i])==0) {
            i = Nbin;
            LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val");
            Throw(TableGenerationError);
        }
        if(minihalos){
            for (j=0; j<NMTURN; j++){
                if(isfinite(Nion_z_val_MINI[i][j])==0){
                    j = NMTURN;
                    LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val_MINI");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

//TODO: I'm not 100% sure the tables do much since there are no integrals (maybe erfc is slow?), look into it
//This table is a lot simpler than the one that existed previously, which was a R_ct x 2D table, with
//  loads of precomputed interpolation points. This may have been from a point where the table was generated only once
//  and needed both an R and zpp axis.
//NOTE: both here and for the conditional tables it says "log spacing is desired", but I can't see why.
//  TODO: make a plot of Fcoll vs delta, and Fcoll vs log(1+delta) to see which binning is better
//      but I would expect linear in delta to be fine
void initialise_FgtrM_delta_table(int n_radii, double *min_dens, double *max_dens, double *z_array, double *growth_array, double *smin_array, double *smax_array){
    int i,j;

    if(F_table_dens == NULL){
        F_table_dens = calloc(n_radii,sizeof(double*));
        for(i=0;i<n_radii;i++){
            F_table_dens[i] = calloc(dens_Ninterp,sizeof(double));
        }
        F_table_val = calloc(n_radii,sizeof(double*));
        for(i=0;i<n_radii;i++){
            F_table_val[i] = calloc(dens_Ninterp,sizeof(double));
        }
        dF_table_val = calloc(n_radii,sizeof(double*));
        for(i=0;i<n_radii;i++){
            dF_table_val[i] = calloc(dens_Ninterp,sizeof(double));
        }
    }

    for(i=0;i<n_radii;i++){
        // LOG_DEBUG("Starting R %d dens [%.2e,%.2e] D %.2e sig [%.2e,%.2e]",i,min_dens[i],max_dens[i],growth_array[i],smin_array[i],smax_array[i]);
        //dens_Ninterp is a global define, probably shouldn't be
        for(j=0;j<dens_Ninterp;j++){
            F_table_dens[i][j] = min_dens[i] + j*(max_dens[i] - min_dens[i])/(dens_Ninterp-1);
            F_table_val[i][j] = FgtrM_bias_fast(growth_array[i], F_table_dens[i][j], smin_array[i], smax_array[i]);
            dF_table_val[i][j] = dfcoll_dz(z_array[i], smin_array[i], F_table_dens[i][j], smax_array[i]);
        }
    }
    LOG_DEBUG("done");
}

int n_redshifts_1DTable;
double zmin_1DTable, zmax_1DTable, zbin_width_1DTable;
double *FgtrM_1DTable_linear;

//TODO: change to same z-bins as other global 1D tables
void init_FcollTable(double zmin, double zmax, struct AstroParams *astro_params, struct FlagOptions *flag_options)
{
    int i;
    double z_table;

    zmin_1DTable = zmin;
    zmax_1DTable = 1.2*zmax;

    zbin_width_1DTable = 0.1;

    n_redshifts_1DTable = (int)ceil((zmax_1DTable - zmin_1DTable)/zbin_width_1DTable);

    FgtrM_1DTable_linear = (double *)calloc(n_redshifts_1DTable,sizeof(double));
    
    LOG_DEBUG("initing Fcoll spline from %.2f to %.2f %d[%.2f %.2f]",zmin,zmax,n_redshifts_1DTable,
                        zmin_1DTable,zbin_width_1DTable);

    for(i=0;i<n_redshifts_1DTable;i++){
        z_table = zmin_1DTable + zbin_width_1DTable*(double)i;
        //NOTE: previously this divided Mturn by 50 which I think is a bug with M_MIN_in_Mass, since there is a sharp cutoff
        FgtrM_1DTable_linear[i] = FgtrM(z_table, minimum_source_mass(z_table,astro_params,flag_options));
    }
}


//frees Sigma and the global interp tables
//TODO: better organisation of the table allocation/free
//  also the 2D array freeing should be split loops / function
void FreeTsInterpolationTables(struct FlagOptions *flag_options) {
    LOG_DEBUG("Freeing some interpolation table memory.");
    //Since the sigma table in Ts and ion are linked, we cannot free here
    //TODO: better organisation of this table, we initilialise once unless using TtoM
    //  keeping in mind a user may change parameters in the same instance
	// freeSigmaMInterpTable();
    int i;
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        free(z_val); z_val = NULL;
        free(Nion_z_val);
        free(z_X_val); z_X_val = NULL;
        free(SFRD_val);
        if (flag_options->USE_MINI_HALOS){
            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++){
                free(Nion_z_val_MINI[i]);
                free(SFRD_val_MINI[i]);
            }
            free(Nion_z_val_MINI);
            free(SFRD_val_MINI);
        }
    }
    else{            
        for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++){
                free(F_table_dens[i]);
                free(F_table_val[i]);
                free(dF_table_val[i]);
        }
        free(F_table_dens);
        free(F_table_val);
        free(dF_table_val);
    }

    LOG_DEBUG("Done Freeing interpolation table memory.");
	interpolation_tables_allocated = false;
}