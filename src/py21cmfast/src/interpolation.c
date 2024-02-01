//New file to store functions that deal with the interpolation tables, since they are used very often.
//  We use regular grid tables since they are faster to evaluate (we always know which bin we are in)
//  So I'm making a general function for the 1D and 2D cases
//TODO: find out if the compiler optimizes this if it's called with the same x or y, (first 3 points are the same)
//  Because there are some points such as the frequency integral tables where we evaluate an entire column of one 2D table
//  Otherwise write a function which saves the left_edge and right_edge for faster evaluation
//TODO: make a print_interp_error(array,x,y) function which prints the cell corners, interpolation points etc

//definitions from ps.c
//ps.c is currently included first
//MOVE THESE WHEN YOU MOVE THE COND TABLES
#define NDELTA 400
#define NMTURN 50//100
#define LOG10_MTURN_MAX ((double)(10))
#define LOG10_MTURN_MIN ((double)(5.-9e-8))

static struct UserParams * user_params_it;

void Broadcast_struct_global_IT(struct UserParams *user_params){
    user_params_it = user_params;
}

//TODO: for the moment the tables are still in global arrays, but will move to these structs soon
//TODO: sort out if we actually need single precision tables
struct RGTable1D{
    int n_bin;
    double x_min;
    double x_width;

    double *y_arr;
    int allocated;
};

struct RGTable2D{
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    double **z_arr;

    double saved_ll, saved_ul; //for future acceleration
    int allocated;
};

struct RGTable1D_f{
    int n_bin;
    double x_min;
    double x_width;

    float *y_arr;
    int allocated;
};

struct RGTable2D_f{
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    float **z_arr;

    double saved_ll, saved_ul; //for future acceleration
    int allocated;
};

void allocate_RGTable1D(int n_bin, struct RGTable1D * ptr){
    //allocate the struct itself
    ptr = malloc(sizeof(struct RGTable1D));
    ptr->n_bin = n_bin;
    ptr->y_arr = calloc(n_bin,sizeof(double));
}

void allocate_RGTable1D_f(int n_bin, struct RGTable1D_f * ptr){
    //allocate the struct itself
    ptr = malloc(sizeof(struct RGTable1D_f));
    ptr->n_bin = n_bin;
    //allocate the table array
    ptr->y_arr = calloc(n_bin,sizeof(float));
}

void free_RGTable1D(struct RGTable1D * ptr){
    free(ptr->y_arr);
    free(ptr);
    ptr = NULL;
}

void free_RGTable1D_f(struct RGTable1D_f * ptr){
    free(ptr->y_arr);
    free(ptr);
    ptr = NULL;
}

void allocate_RGTable2D(int n_x, int n_y, struct RGTable2D * ptr){
    //allocate the struct itself
    int i;
    ptr = malloc(sizeof(struct RGTable2D));
    ptr->nx_bin = n_x;
    ptr->ny_bin = n_y;

    //allocate the table 2D array
    ptr->z_arr = calloc(n_x,sizeof(double*));
    for(i=0;i<n_x;i++){
        ptr->z_arr[i] = calloc(n_y,sizeof(double));
    }
}

void allocate_RGTable2D_f(int n_x, int n_y, struct RGTable2D_f * ptr){
    //allocate the struct itself
    int i;
    ptr = malloc(sizeof(struct RGTable2D_f));
    ptr->nx_bin = n_x;
    ptr->ny_bin = n_y;

    //allocate the table 2D array
    ptr->z_arr = calloc(n_x,sizeof(float*));
    for(i=0;i<n_x;i++){
        ptr->z_arr[i] = calloc(n_y,sizeof(float));
    }
}

void free_RGTable2D_f(struct RGTable2D_f * ptr){
    int i;
    for(i=0;i<ptr->nx_bin;i++)
        free(ptr->z_arr[i]);
    free(ptr->z_arr);
    free(ptr);
    ptr = NULL;
}

void free_RGTable2D(struct RGTable2D * ptr){
    int i;
    for(i=0;i<ptr->nx_bin;i++)
        free(ptr->z_arr[i]);
    free(ptr->z_arr);
    free(ptr);
    ptr = NULL;
}

double EvaluateRGTable1D(double x, double *y_arr, double x_min, double x_width){
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(double)idx;
    double interp_point = (x - table_val)/x_width;
    // LOG_DEBUG("1D: x %.6e -> idx %d -> tbl %.6e -> itp %.6e",x,idx,table_val,interp_point);

    //a + f(a-b) is one fewer operation but less precise
    double result = y_arr[idx]*(1-interp_point) + y_arr[idx+1]*(interp_point);

    // LOG_DEBUG("-> result %.2e",result);

    return result;
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
    // LOG_DEBUG("2D Interp: table corners (%.2e,%.2e,%.2e,%.2e)",z_arr[x_idx][y_idx],z_arr[x_idx][y_idx+1],z_arr[x_idx+1][y_idx],z_arr[x_idx+1][y_idx+1]);

    left_edge = z_arr[x_idx][y_idx]*(1-interp_point_y) + z_arr[x_idx][y_idx+1]*(interp_point_y);
    right_edge = z_arr[x_idx+1][y_idx]*(1-interp_point_y) + z_arr[x_idx+1][y_idx+1]*(interp_point_y);

    result = left_edge*(1-interp_point_x) + right_edge*(interp_point_x);
    // LOG_DEBUG("result %.6e",result);

    return result;
}

//some tables are floats but I still need to return doubles
double EvaluateRGTable1D_f(double x, float *y_arr, float x_min, float x_width){
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(float)idx;
    double interp_point = (x - table_val)/x_width;

    return y_arr[idx]*(1-interp_point) + y_arr[idx+1]*(interp_point);
}

double EvaluateRGTable2D_f(double x, double y, float **z_arr, double x_min, double x_width, double y_min, double y_width){
    int x_idx = (int)floor((x - x_min)/x_width);
    int y_idx = (int)floor((y - y_min)/y_width);

    double x_table = x_min + x_width*(double)x_idx;
    double y_table = y_min + y_width*(double)y_idx;

    double interp_point_x = (x - x_table)/x_width;
    double interp_point_y = (y - y_table)/y_width;

    double left_edge, right_edge, result;

    // LOG_DEBUG("2D Interp: val (%.2e,%.2e) min (%.2e,%.2e) wid (%.2e,%.2e)",x,y,x_min,y_min,x_width,y_width);
    // LOG_DEBUG("2D Interp: idx (%d,%d) tbl (%.2e,%.2e) itp (%.2e,%.2e)",x_idx,y_idx,x_table,y_table,interp_point_x,interp_point_y);
    // LOG_DEBUG("2D Interp: table corners (%.2e,%.2e,%.2e,%.2e)",z_arr[x_idx][y_idx],z_arr[x_idx][y_idx+1],z_arr[x_idx+1][y_idx],z_arr[x_idx+1][y_idx+1]);

    left_edge = z_arr[x_idx][y_idx]*(1-interp_point_y) + z_arr[x_idx][y_idx+1]*(interp_point_y);
    right_edge = z_arr[x_idx+1][y_idx]*(1-interp_point_y) + z_arr[x_idx+1][y_idx+1]*(interp_point_y);

    result = left_edge*(1-interp_point_x) + right_edge*(interp_point_x);
    // LOG_DEBUG("result %.6e",result);

    return result;
}

//Specific interpolation tables are below

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

//NOTE: we only have one overdensity table, since currently all tables will use the same one
//   at any point in time. This may not be true in the future so be careful
//NOTE: since reionisation feedback is not included in the Ts calculation, the SFRD spline
//  is Rx1D unlike the Mini table, which is Rx2D
//NOTE: SFRD tables have fixed Mturn range, Nion tables vary
//TODO: fix the confusing Mmax=log(Mmax) naming
//TODO: it would be slightly less accurate but maybe faster to tabulate in linear delta, linear Fcoll
//  rather than linear-log, check the profiles
//I assign to NULL to keep track of allocation
float **ln_Nion_spline=NULL, **prev_ln_Nion_spline=NULL, **ln_SFRD_spline=NULL, *ln_Nion_spline_1D=NULL;
float **ln_Nion_spline_MINI=NULL, **prev_ln_Nion_spline_MINI=NULL, ***ln_SFRD_spline_MINI=NULL;

void initialise_Nion_General_spline(float z, float Mcrit_atom, float min_density, float max_density,
                                     float Mmax, float Mmin, float log10Mturn_min, float log10Mturn_max,
                                     float log10Mturn_min_MINI, float log10Mturn_max_MINI, float Alpha_star,
                                     float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10,
                                     float Mlim_Fstar, float Mlim_Fesc, float Fstar7_MINI, float Fesc7_MINI,
                                     float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, bool FAST_FCOLL_TABLES,
                                     bool minihalos, bool prev){
    double growthf, sigma2;
    int i,j;
    float **output_spline, **output_spline_MINI;
    double overdense_table[NDELTA];
    double mturns[NMTURN], mturns_MINI[NMTURN];

    if(prev){
        output_spline = prev_ln_Nion_spline;
        output_spline_MINI = prev_ln_Nion_spline_MINI;
    }
    else{
        output_spline = ln_Nion_spline;
        output_spline_MINI = ln_Nion_spline_MINI;
    }

    growthf = dicke(z);
    Mmin = log(Mmin);
    Mmax = log(Mmax);

    sigma2 = EvaluateSigma(Mmax,0,NULL);
    // Even when we use GL, this is done in Ionisationbox.c
    // initialiseGL_Nion(NGL_SFR, global_params.M_MIN_INTEGRAL, Mmax);

    for (i=0;i<NDELTA;i++) {
        overdense_table[i] = min_density + (float)i/((float)NDELTA-1.)*(max_density - min_density);
    }
    if(minihalos){
        for (i=0;i<NMTURN;i++){
            mturns[i] = pow(10., log10Mturn_min + (float)i/((float)NMTURN-1.)*(log10Mturn_max-log10Mturn_min));
            mturns_MINI[i] = pow(10., log10Mturn_min_MINI + (float)i/((float)NMTURN-1.)*(log10Mturn_max_MINI-log10Mturn_min_MINI));
        }
    }

    if (minihalos){
        if(ln_Nion_spline==NULL) {
            LOG_SUPER_DEBUG("allocating interp Nion2d");
            ln_Nion_spline = calloc(NDELTA,sizeof(float*));
            for(i=0;i<NDELTA;i++) ln_Nion_spline[i] = calloc(NMTURN,sizeof(float));
            ln_Nion_spline_MINI = calloc(NDELTA,sizeof(float*));
            for(i=0;i<NDELTA;i++) ln_Nion_spline_MINI[i] = calloc(NMTURN,sizeof(float));
            prev_ln_Nion_spline = calloc(NDELTA,sizeof(float*));
            for(i=0;i<NDELTA;i++) prev_ln_Nion_spline[i] = calloc(NMTURN,sizeof(float));
            prev_ln_Nion_spline_MINI = calloc(NDELTA,sizeof(float*));
            for(i=0;i<NDELTA;i++) prev_ln_Nion_spline_MINI[i] = calloc(NMTURN,sizeof(float));
        }
    }
    else{
        if(ln_Nion_spline_1D==NULL){
            LOG_SUPER_DEBUG("allocating interp Nion1d");
            //TODO: for some reason we don't use reion feedback without minihalos
            ln_Nion_spline_1D = calloc(NDELTA,sizeof(float));
        }
    }

#pragma omp parallel private(i,j) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for(i=0;i<NDELTA;i++){
            if(!minihalos){
                //pass constant M_turn as minimum
                ln_Nion_spline_1D[i] = log(Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,Deltac,
                                                overdense_table[i],Mcrit_atom,Alpha_star,Alpha_esc,
                                                Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,FAST_FCOLL_TABLES));
                if(ln_Nion_spline_1D[i]<-40)
                    ln_Nion_spline_1D[i]=-40;

                continue;
            }
            for (j=0; j<NMTURN; j++){
                output_spline[i][j] = log(Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,Deltac,
                                                overdense_table[i],mturns[j],Alpha_star,Alpha_esc,
                                                Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,FAST_FCOLL_TABLES));

                // output_spline[i][j] = log(GaussLegendreQuad_Nion(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,\
                                                        pow(10.,log10_overdense_spline_SFR[i])-1.,Mturns[j],Alpha_star,\
                                                        Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc, FAST_FCOLL_TABLES));

                if(output_spline[i][j]<-40)
                    output_spline[i][j]=-40;

                output_spline_MINI[i][j] = log(Nion_ConditionalM_MINI(growthf,Mmin,Mmax,sigma2,Deltac,overdense_table[i],
                                                    mturns_MINI[j],Mcrit_atom,Alpha_star_mini,Alpha_esc,Fstar7_MINI,Fesc7_MINI,
                                                    Mlim_Fstar_MINI,Mlim_Fesc_MINI, FAST_FCOLL_TABLES));

                // output_spline_MINI[i][j] = log(GaussLegendreQuad_Nion_MINI(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,\
                                                                pow(10.,log10_overdense_spline_SFR[i])-1.,Mturns_MINI[j],Mcrit_atom,\
                                                                Alpha_star_mini,Alpha_esc,Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI, FAST_FCOLL_TABLES));

                if(output_spline_MINI[i][j]<-40.)
                    output_spline_MINI[i][j]=-40.0;
            }
        }
    }

    for(i=0;i<NDELTA;i++) {
        if(!minihalos){
            if(isfinite(ln_Nion_spline_1D[i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline_1D");
                Throw(TableGenerationError);
            }
            continue;
        }
        for (j=0; j<NMTURN; j++){
            if(isfinite(output_spline[i][j])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline");
                Throw(TableGenerationError);
            }

            if(isfinite(output_spline_MINI[i][j])==0) {
               LOG_ERROR("Detected either an infinite or NaN value in Nion_spline_MINI");
                Throw(TableGenerationError);
            }
        }
    }
}

void initialise_SFRD_Conditional_table(int Nfilter, double min_density[], double max_density[], double growthf[],
                                    double R[], float Mcrit_atom[], double Mmin, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI, bool FAST_FCOLL_TABLES, bool minihalos){
    float Mmax,Mlim_Fstar,sigma2,Mlim_Fstar_MINI;
    int i,j,k,i_tot;

    Mmax = RtoM(R[Nfilter-1]);
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));

    float MassTurnover[NMTURN];
    for(i=0;i<NMTURN;i++){
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
    }

    //strictly I should check every pointer but they're allocated together
    //We still allocate the full spline even if Nfilter is less since it can be used elsewhere
    if(ln_SFRD_spline == NULL){
        LOG_SUPER_DEBUG("allocating interp SFRD");
        ln_SFRD_spline = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
        for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
            ln_SFRD_spline[j] = (float *)calloc(NDELTA,sizeof(float));
        }

        if(minihalos && ln_SFRD_spline_MINI == NULL){
            ln_SFRD_spline_MINI = (float ***)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float **));
            for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++){
                ln_SFRD_spline_MINI[j] = (float **)calloc(NDELTA,sizeof(float *));
                for(i=0;i<NDELTA;i++){
                    ln_SFRD_spline_MINI[j][i] = (float *)calloc(NMTURN,sizeof(float));
                }
            }
        }
    }

    LOG_DEBUG("Initialising SFRD conditional table");

    Mmin = log(Mmin);

    for(j=0; j < Nfilter; j++){
        Mmax = RtoM(R[j]);

        // initialiseGL_Nion_Xray(NGL_SFR, global_params.M_MIN_INTEGRAL, Mmax);
        Mmax = log(Mmax);
        sigma2 = EvaluateSigma(Mmax,0,NULL);

#pragma omp parallel private(i,k) num_threads(user_params_ps->N_THREADS)
        {
            double curr_dens;
#pragma omp for
            for (i=0; i<NDELTA; i++){
                curr_dens = min_density[j] + (float)i/((float)NDELTA-1.)*(max_density[j] - min_density[j]);
                ln_SFRD_spline[j][i] = log(Nion_ConditionalM(growthf[j],Mmin,Mmax,sigma2,Deltac,curr_dens,\
                                                            Mcrit_atom[j],Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES));
                // log_SFRD_spline[j][i] = log(GaussLegendreQuad_Nion(1,NGL_SFR,growthf[j],Mmax,sigma2,Deltac,curr_dens.,\
                                                                            Mcrit_atom[j],Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES));
                if(ln_SFRD_spline[j][i] < -50.)
                    ln_SFRD_spline[j][i] = -50.;

                if(!minihalos) continue;

                for (k=0; k<NMTURN; k++){
                    ln_SFRD_spline_MINI[j][i][k] = log(Nion_ConditionalM_MINI(growthf[j],Mmin,Mmax,sigma2,Deltac,\
                                                curr_dens,MassTurnover[k],Mcrit_atom[j],\
                                                Alpha_star_mini,0.,Fstar7_MINI,1.,Mlim_Fstar_MINI, 0., FAST_FCOLL_TABLES));
                    // ln_SFRD_spline_MINI[j][i][k] = log(GaussLegendreQuad_Nion_MINI(1,NGL_SFR,growthf[j],Mmax,sigma2,Deltac,overdense_array_R,\
                                                            MassTurnover[k], Mcrit_atom[j],Alpha_star_mini,0.,Fstar7_MINI,1.,Mlim_Fstar_MINI, 0., FAST_FCOLL_TABLES));
                    if(ln_SFRD_spline_MINI[j][i][k] < -50.)
                        ln_SFRD_spline_MINI[j][i][k] = -50.;
                }
            }
        }
    }
    for(j=0;j<Nfilter;j++){
        for (i=0; i<NDELTA; i++){
            if(isfinite(ln_SFRD_spline[j][i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in ACG SFRD conditional table");
                Throw(TableGenerationError);
            }

            if(!minihalos) continue;

            for (k=0; k<NMTURN; k++){
                if(isfinite(ln_SFRD_spline_MINI[j][i][k])==0) {
                    LOG_ERROR("Detected either an infinite or NaN value in MCG SFRD conditional table");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

void FreeNionConditionalTable(struct FlagOptions * flag_options){
    int i;
    LOG_SUPER_DEBUG("Freeing interp Nion");
    if(flag_options->USE_MINI_HALOS){
        for(i=0;i<NDELTA;i++) free(ln_Nion_spline[i]);
        free(ln_Nion_spline);
        ln_Nion_spline=NULL;
        for(i=0;i<NDELTA;i++) free(ln_Nion_spline_MINI[i]);
        free(ln_Nion_spline_MINI);
        ln_Nion_spline_MINI=NULL;
        for(i=0;i<NDELTA;i++) free(prev_ln_Nion_spline[i]);
        free(prev_ln_Nion_spline);
        prev_ln_Nion_spline=NULL;
        for(i=0;i<NDELTA;i++) free(prev_ln_Nion_spline_MINI[i]);
        free(prev_ln_Nion_spline_MINI);
        prev_ln_Nion_spline_MINI=NULL;
    }
    else{
        free(ln_Nion_spline_1D);
        ln_Nion_spline_1D = NULL;
    }
}

void FreeSFRDConditionalTable(struct FlagOptions *flag_options){
    int j,i;
    for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
        free(ln_SFRD_spline[j]);
    }
    free(ln_SFRD_spline);
    ln_SFRD_spline = NULL;

    if(flag_options->USE_MINI_HALOS){
        for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++){
            for(i=0;i<NDELTA;i++){
                free(ln_SFRD_spline_MINI[j][i]);
            }
            free(ln_SFRD_spline_MINI[j]);
        }
        free(ln_SFRD_spline_MINI);
        ln_SFRD_spline_MINI = NULL;
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
        free(z_val);
        z_val = NULL;
        free(Nion_z_val);
        free(z_X_val);
        z_X_val = NULL;
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
    if(!flag_options->USE_HALO_FIELD)
        FreeSFRDConditionalTable(flag_options);

    LOG_DEBUG("Done Freeing interpolation table memory.");
}
