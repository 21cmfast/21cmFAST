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

    return y_arr[idx]*(1-interp_point) + y_arr[idx+1]*(interp_point);
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

//NOTE: this table is initialised for up to N_redshift x N_Mturn, but only called N_filter times to assign ST_over_PS in Spintemp. 
//  It may be better to just do the integrals at each R
void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI, float mturn_a_const, int minihalos){
    int i,j;
    float Mmin, Mmax;
    Mmin = minihalos ? global_params.M_MIN_INTEGRAL : mturn_a_const/50.;
    Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fstar_MINI;

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

//frees Sigma and the global interp tables
//TODO: better organisation of the table allocation/free
void FreeTsInterpolationTables(struct FlagOptions *flag_options) {
    LOG_DEBUG("Freeing some interpolation table memory.");
	freeSigmaMInterpTable();
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        free(z_val); z_val = NULL;
        free(Nion_z_val);
        free(z_X_val); z_X_val = NULL;
        free(SFRD_val);
        if (flag_options->USE_MINI_HALOS){
            free(Nion_z_val_MINI);
            free(SFRD_val_MINI);
        }
    }
    else{
        free(FgtrM_1DTable_linear);
    }

    LOG_DEBUG("Done Freeing interpolation table memory.");
	interpolation_tables_allocated = false;
}