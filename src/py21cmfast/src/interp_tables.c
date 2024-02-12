/* This file defines specific interpolation table initialisation functions, kept separate from the general interpolation table routines
   In order to allow them to use calculations based on other interpolation tables. Most importantly these fucntions require those from ps.c
   which requires the sigma(M) interpolation tables */

//MOVE THESE WHEN YOU MOVE THE COND TABLES
#define NDELTA 400
#define NMTURN 50//100
#define LOG10_MTURN_MAX ((double)(10))
#define LOG10_MTURN_MIN ((double)(5.-9e-8))

//we need to define a density minimum for the tables, since we are in lagrangian density / linear growth it's possible to go below -1
//so we explicitly set a minimum here which sets table limits and puts no halos in cells below that (Lagrangian) density
#define DELTA_MIN -1
#define MAX_DELTAC_FRAC (float)0.999 //max delta/deltac for interpolation tables / integrals


static struct UserParams * user_params_it;
static struct CosmoParams * cosmo_params_it;
static struct AstroParams * astro_params_it;
static struct FlagOptions * flag_options_it;

void Broadcast_struct_global_IT(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options){
    user_params_it = user_params;
    cosmo_params_it = cosmo_params;
    astro_params_it = astro_params;
    flag_options_it = flag_options;
}


//Tables for the grids
struct RGTable1D SFRD_z_table = {.allocated = false}, Nion_z_table = {.allocated = false};
struct RGTable2D SFRD_z_table_MINI = {.allocated = false}, Nion_z_table_MINI = {.allocated = false};
struct RGTable1D_f Nion_conditional_table1D = {.allocated = false}, SFRD_conditional_table = {.allocated = false};
struct RGTable2D_f Nion_conditional_table2D = {.allocated = false}, Nion_conditional_table_MINI = {.allocated = false}, SFRD_conditional_table_MINI = {.allocated = false};
struct RGTable2D_f Nion_conditional_table_prev = {.allocated = false}, Nion_conditional_table_MINI_prev = {.allocated = false};

//Tables for the catalogues
struct RGTable1D Nhalo_table = {.allocated = false}, Mcoll_table = {.allocated = false};
struct RGTable2D Nhalo_inv_table = {.allocated = false};

//Tables for the old parametrization
struct RGTable1D fcoll_z_table = {.allocated = false};
struct RGTable1D_f fcoll_conditional_table = {.allocated = false,};
struct RGTable1D_f dfcoll_conditional_table = {.allocated = false,};

//NOTE: this table is initialised for up to N_redshift x N_Mturn, but only called N_filter times to assign ST_over_PS in Spintemp.
//  It may be better to just do the integrals at each R
void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float Mmin, float Mmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI,
                             float mturn_a_const, bool minihalos){
    int i,j;
    float Mlim_Fstar, Mlim_Fstar_MINI;
    double z_val, mturn_val;

    LOG_DEBUG("initing SFRD spline from %.2f to %.2f",zmin,zmax);

    if (!SFRD_z_table.allocated){
        allocate_RGTable1D(Nbin,&SFRD_z_table);
    }
    if(minihalos && !SFRD_z_table_MINI.allocated){
        allocate_RGTable2D(Nbin,NMTURN,&SFRD_z_table_MINI);
    }

    SFRD_z_table.x_min = zmin;
    SFRD_z_table.x_width = (zmax - zmin)/((double)Nbin-1.);

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    if(minihalos){
        SFRD_z_table_MINI.x_min = zmin;
        SFRD_z_table_MINI.x_width = (zmax - zmin)/((double)Nbin-1.);
        SFRD_z_table_MINI.y_min = LOG10_MTURN_MIN;
        SFRD_z_table_MINI.y_width = (LOG10_MTURN_MAX-LOG10_MTURN_MIN)/((double)NMTURN-1.);
        Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));
    }

    #pragma omp parallel private(i,j) num_threads(user_params_it->N_THREADS)
    {
        float Mcrit_atom_val = mturn_a_const;
        #pragma omp for
        for (i=0; i<Nbin; i++){
            z_val = SFRD_z_table.x_min + i*SFRD_z_table.x_width; //both tables will have the same values here
            if(minihalos)Mcrit_atom_val = atomic_cooling_threshold(z_val);

            SFRD_z_table.y_arr[i] = Nion_General(z_val, Mmin, Mmax, Mcrit_atom_val, Alpha_star, 0., Fstar10, 1.,Mlim_Fstar,0.,user_params_it->INTEGRATION_METHOD_ATOMIC);
            if(minihalos){
                for (j=0; j<NMTURN; j++){
                    mturn_val = pow(10,SFRD_z_table_MINI.y_min + j*SFRD_z_table_MINI.y_width);
                    SFRD_z_table_MINI.z_arr[i][j] = Nion_General_MINI(z_val, Mmin, Mmax, mturn_val, Mcrit_atom_val, Alpha_star_mini,
                                                                 0., Fstar7_MINI, 1.,Mlim_Fstar_MINI,0.,user_params_it->INTEGRATION_METHOD_MINI);
                }
            }
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(SFRD_z_table.y_arr[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in SFRD table");
            Throw(TableGenerationError);
        }
        if(minihalos){
            for (j=0; j<NMTURN; j++){
                if(isfinite(SFRD_z_table_MINI.z_arr[i][j])==0) {
                    LOG_ERROR("Detected either an infinite or NaN value in SFRD_MINI table");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

//Unlike the SFRD spline, this one is used more due to the nu_tau_one() rootfind
void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, float Mmin, float Mmax, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10,
                                float Fesc10, float Fstar7_MINI, float Fesc7_MINI, float mturn_a_const, bool minihalos){
    int i,j;
    float Mlim_Fstar, Mlim_Fesc, Mlim_Fstar_MINI, Mlim_Fesc_MINI;
    LOG_DEBUG("initing Nion spline from %.2f to %.2f",zmin,zmax);
    double z_val, mturn_val;

    if (!Nion_z_table.allocated){
        allocate_RGTable1D(Nbin,&Nion_z_table);
    }
    if(minihalos && !Nion_z_table_MINI.allocated){
        allocate_RGTable2D(Nbin,NMTURN,&Nion_z_table_MINI);
    }
    Nion_z_table.x_min = zmin;
    Nion_z_table.x_width = (zmax - zmin)/((double)Nbin-1.);

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fesc = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc10);
    if(minihalos){
        Nion_z_table_MINI.x_min = zmin;
        Nion_z_table_MINI.x_width = (zmax - zmin)/((double)Nbin-1.);
        Nion_z_table_MINI.y_min = LOG10_MTURN_MIN;
        Nion_z_table_MINI.y_width = (LOG10_MTURN_MAX-LOG10_MTURN_MIN)/((double)NMTURN-1.);
        Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));
        Mlim_Fesc_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc7_MINI * pow(1e3, Alpha_esc));
    }

#pragma omp parallel private(i,j) num_threads(user_params_it->N_THREADS)
    {
        float Mcrit_atom_val = mturn_a_const;
#pragma omp for
        for (i=0; i<Nbin; i++){
            z_val = Nion_z_table.x_min + i*Nion_z_table.x_width; //both tables will have the same values here
            if(minihalos)Mcrit_atom_val = atomic_cooling_threshold(z_val);

            Nion_z_table.y_arr[i] = Nion_General(z_val, Mmin, Mmax, Mcrit_atom_val, Alpha_star, Alpha_esc, Fstar10, Fesc10,
                                             Mlim_Fstar, Mlim_Fesc, user_params_it->INTEGRATION_METHOD_ATOMIC);
            if(minihalos){
                for (j=0; j<NMTURN; j++){
                    mturn_val = pow(10,Nion_z_table_MINI.y_min + j*Nion_z_table_MINI.y_width);
                    Nion_z_table_MINI.z_arr[i][j] = Nion_General_MINI(z_val, Mmin, Mmax, mturn_val, Mcrit_atom_val, Alpha_star_mini, Alpha_esc,
                                                     Fstar7_MINI, Fesc7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI, user_params_it->INTEGRATION_METHOD_MINI);
                }
            }
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(Nion_z_table.y_arr[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val");
            Throw(TableGenerationError);
        }
        if(minihalos){
            for (j=0; j<NMTURN; j++){
                if(isfinite(Nion_z_table_MINI.z_arr[i][j])==0){
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
void initialise_FgtrM_delta_table(double min_dens, double max_dens, double zpp, double growth_zpp, double smin_zpp, double smax_zpp){
    int i,j;
    double dens;

    if(!fcoll_conditional_table.allocated){
        allocate_RGTable1D(dens_Ninterp,&fcoll_conditional_table);
    }
    fcoll_conditional_table.x_min = min_dens;
    fcoll_conditional_table.x_width = (max_dens - min_dens)/(dens_Ninterp - 1.);
    if(!dfcoll_conditional_table.allocated){
        allocate_RGTable1D(dens_Ninterp,&dfcoll_conditional_table);
    }
    dfcoll_conditional_table.x_min = min_dens;
    dfcoll_conditional_table.x_width = (max_dens - min_dens)/(dens_Ninterp - 1.);

    //dens_Ninterp is a global define, probably shouldn't be
    for(j=0;j<dens_Ninterp;j++){
        dens = fcoll_conditional_table.x_min + j*fcoll_conditional_table.x_width;
        fcoll_conditional_table.y_arr[j] = FgtrM_bias_fast(growth_zpp, dens, smin_zpp, smax_zpp);
        dfcoll_conditional_table.y_arr[j] = dfcoll_dz(zpp, smin_zpp, dens, smax_zpp);
    }
}

//TODO: change to same z-bins as other global 1D tables
void init_FcollTable(double zmin, double zmax, double M_min){
    int i;
    double z_val;

    fcoll_z_table.x_min = zmin;
    fcoll_z_table.x_width = 0.1;

    int n_z = (int)ceil((zmax - zmin)/fcoll_z_table.x_width);

    if(!fcoll_z_table.allocated){
        allocate_RGTable1D(n_z,&fcoll_z_table);
    }

    for(i=0;i<n_z;i++){
        z_val = fcoll_z_table.x_min + i*fcoll_z_table.x_width;
        //NOTE: previously this divided Mturn by 50 which I think is a bug with M_MIN_in_Mass, since there is a sharp cutoff
        fcoll_z_table.y_arr[i] = FgtrM(z_val, M_min);
    }
}

//NOTE: since reionisation feedback is not included in the Ts calculation, the SFRD spline
//  is Rx1D unlike the Mini table, which is Rx2D
//NOTE: SFRD tables have fixed Mturn range, Nion tables vary
//TODO: fix the confusing Mmax=log(Mmax) naming
//TODO: it would be slightly less accurate but maybe faster to tabulate in linear delta, linear Fcoll
//  rather than linear-log, check the profiles

void initialise_Nion_Conditional_spline(float z, float Mcrit_atom, float min_density, float max_density,
                                     float Mmin, float Mmax, float Mcond, float log10Mturn_min, float log10Mturn_max,
                                     float log10Mturn_min_MINI, float log10Mturn_max_MINI, float Alpha_star,
                                     float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10,
                                     float Mlim_Fstar, float Mlim_Fesc, float Fstar7_MINI, float Fesc7_MINI,
                                     float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, int method, int method_mini,
                                     bool minihalos, bool prev){
    double growthf, sigma2;
    int i,j;
    double overdense_table[NDELTA];
    double mturns[NMTURN], mturns_MINI[NMTURN];
    struct RGTable2D_f *table_2d, *table_mini;

    LOG_DEBUG("Initialising Nion conditional table at mass %.2e from delta %.2e to %.2e",Mcond,min_density,max_density);

    growthf = dicke(z);
    Mmin = log(Mmin);
    Mmax = log(Mmax);

    sigma2 = EvaluateSigma(log(Mcond),0,NULL);

    if(prev){
        table_2d = &Nion_conditional_table_prev;
        table_mini = &Nion_conditional_table_MINI_prev;
    }
    else{
        table_2d = &Nion_conditional_table2D;
        table_mini = &Nion_conditional_table_MINI;
    }

    //If we use minihalos, both tables are 2D (delta,mturn) due to reionisaiton feedback
    //otherwise, the Nion table is 1D, since reionsaiton feedback is only active with minihalos
    if (minihalos){
        if(!table_2d->allocated) {
            allocate_RGTable2D_f(NDELTA,NMTURN,table_2d);
        }
        if(!table_mini->allocated) {
            allocate_RGTable2D_f(NDELTA,NMTURN,table_mini);
        }
        table_2d->x_min = min_density;
        table_2d->x_width = (max_density - min_density)/(NDELTA-1.);
        table_2d->y_min = log10Mturn_min;
        table_2d->y_width = (log10Mturn_max - log10Mturn_min)/(NMTURN-1.);

        table_mini->x_min = min_density;
        table_mini->x_width = (max_density - min_density)/(NDELTA-1.);
        table_mini->y_min = log10Mturn_min_MINI;
        table_mini->y_width = (log10Mturn_max_MINI - log10Mturn_min_MINI)/(NMTURN-1.);
    }
    else{
        if(!Nion_conditional_table1D.allocated) {
            allocate_RGTable1D_f(NDELTA,&Nion_conditional_table1D);
        }
        Nion_conditional_table1D.x_min = min_density;
        Nion_conditional_table1D.x_width = (max_density - min_density)/(NDELTA-1.);
    }

    for (i=0;i<NDELTA;i++) {
        overdense_table[i] = min_density + (float)i/((float)NDELTA-1.)*(max_density - min_density);
    }
    if(minihalos){
        for (i=0;i<NMTURN;i++){
            mturns[i] = pow(10., log10Mturn_min + (float)i/((float)NMTURN-1.)*(log10Mturn_max-log10Mturn_min));
            mturns_MINI[i] = pow(10., log10Mturn_min_MINI + (float)i/((float)NMTURN-1.)*(log10Mturn_max_MINI-log10Mturn_min_MINI));
        }
    }

#pragma omp parallel private(i,j) num_threads(user_params_it->N_THREADS)
    {
#pragma omp for
        for(i=0;i<NDELTA;i++){
            if(!minihalos){
                //pass constant M_turn as minimum
                Nion_conditional_table1D.y_arr[i] = log(Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,
                                                overdense_table[i],Mcrit_atom,Alpha_star,Alpha_esc,
                                                Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,user_params_it->INTEGRATION_METHOD_ATOMIC));
                if(Nion_conditional_table1D.y_arr[i] < -40.)
                    Nion_conditional_table1D.y_arr[i] = -40.;

                continue;
            }
            for (j=0; j<NMTURN; j++){
                table_2d->z_arr[i][j] = log(Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,
                                                overdense_table[i],mturns[j],Alpha_star,Alpha_esc,
                                                Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,user_params_it->INTEGRATION_METHOD_ATOMIC));

                if(table_2d->z_arr[i][j] < -40.)
                    table_2d->z_arr[i][j] = -40.;

                table_mini->z_arr[i][j] = log(Nion_ConditionalM_MINI(growthf,Mmin,Mmax,sigma2,overdense_table[i],
                                                    mturns_MINI[j],Mcrit_atom,Alpha_star_mini,Alpha_esc,Fstar7_MINI,Fesc7_MINI,
                                                    Mlim_Fstar_MINI,Mlim_Fesc_MINI,user_params_it->INTEGRATION_METHOD_MINI));

                if(table_mini->z_arr[i][j] < -40.)
                    table_mini->z_arr[i][j] = -40.;
            }
        }
    }

    for(i=0;i<NDELTA;i++) {
        if(!minihalos){
            if(isfinite(Nion_conditional_table1D.y_arr[i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline_1D");
                Throw(TableGenerationError);
            }
            continue;
        }
        for (j=0; j<NMTURN; j++){
            if(isfinite(table_2d->z_arr[i][j])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline");
                Throw(TableGenerationError);
            }

            if(isfinite(table_2d->z_arr[i][j])==0) {
               LOG_ERROR("Detected either an infinite or NaN value in Nion_spline_MINI");
                Throw(TableGenerationError);
            }
        }
    }
}

//since SFRD is not used in Ionisationbox, and reionisation feedback is not included in the Ts calculation,
//    The non-minihalo table is always Rx1D and the minihalo table is always Rx2D

//This function initialises one table, for table Rx arrays I will call this function in a loop
void initialise_SFRD_Conditional_table(double min_density, double max_density, double growthf,
                                    float Mcrit_atom, double Mmin, double Mmax, double Mcond, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI, int method, int method_mini, bool minihalos){
    float Mlim_Fstar,sigma2,Mlim_Fstar_MINI;
    int i,j,k,i_tot;

    // LOG_DEBUG("Initialising SFRD conditional table at mass %.2e from delta %.2e to %.2e",Mcond,min_density,max_density);

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));

    double lnM_condition = log(Mcond);

    float MassTurnover[NMTURN];
    for(i=0;i<NMTURN;i++){
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
    }

    //NOTE: Here we use the constant Mturn limits instead of variables like in the Nion tables
    if(!SFRD_conditional_table.allocated){
        allocate_RGTable1D_f(NDELTA,&SFRD_conditional_table);
    }
    SFRD_conditional_table.x_min = min_density;
    SFRD_conditional_table.x_width = (max_density - min_density)/(NDELTA-1.);

    if(minihalos){
        if(!SFRD_conditional_table_MINI.allocated){
            allocate_RGTable2D_f(NDELTA,NMTURN,&SFRD_conditional_table_MINI);
        }
        SFRD_conditional_table_MINI.x_min = min_density;
        SFRD_conditional_table_MINI.x_width = (max_density - min_density)/(NDELTA-1.);
        SFRD_conditional_table_MINI.y_min = LOG10_MTURN_MIN;
        SFRD_conditional_table_MINI.y_width = (LOG10_MTURN_MAX - LOG10_MTURN_MIN)/(NMTURN-1.);
    }

    double lnMmin = log(Mmin);
    double lnMmax = log(Mmax);
    sigma2 = EvaluateSigma(lnM_condition,0,NULL); //sigma is always the condition, whereas lnMmax is just the integral limit

#pragma omp parallel private(i,k) num_threads(user_params_it->N_THREADS)
    {
        double curr_dens;
#pragma omp for
        for (i=0; i<NDELTA; i++){
            curr_dens = min_density + (float)i/((float)NDELTA-1.)*(max_density - min_density);

            // LOG_DEBUG("starting d %.2e M [%.2e %.2e] s %.2e",curr_dens, exp(lnMmin),exp(lnMmax),sigma2);
            SFRD_conditional_table.y_arr[i] = log(Nion_ConditionalM(growthf,lnMmin,lnMmax,sigma2,curr_dens,\
                                            Mcrit_atom,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., user_params_it->INTEGRATION_METHOD_ATOMIC));

            if(SFRD_conditional_table.y_arr[i] < -50.)
                SFRD_conditional_table.y_arr[i] = -50.;

            if(!minihalos) continue;

            for (k=0; k<NMTURN; k++){
                SFRD_conditional_table_MINI.z_arr[i][k] = log(Nion_ConditionalM_MINI(growthf,lnMmin,lnMmax,sigma2,\
                                            curr_dens,MassTurnover[k],Mcrit_atom,\
                                            Alpha_star_mini,0.,Fstar7_MINI,1.,Mlim_Fstar_MINI, 0., user_params_it->INTEGRATION_METHOD_MINI));

                if(SFRD_conditional_table_MINI.z_arr[i][k] < -50.)
                    SFRD_conditional_table_MINI.z_arr[i][k] = -50.;
            }
        }
    }
    for (i=0; i<NDELTA; i++){
        if(isfinite(SFRD_conditional_table.y_arr[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in ACG SFRD conditional table");
            Throw(TableGenerationError);
        }
        if(!minihalos) continue;

        for (k=0; k<NMTURN; k++){
            if(isfinite(SFRD_conditional_table_MINI.z_arr[i][k])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in MCG SFRD conditional table");
                Throw(TableGenerationError);
            }
        }
    }
}

//This table is N(>M | M_in), the CDF of dNdM_conditional
//NOTE: Assumes you give it ymin as the minimum mass TODO: add another argument for Mmin
void initialise_dNdM_tables(double xmin, double xmax, double ymin, double ymax, double growth1, double param, bool update){
    int nx,ny,np;
    double lnM_cond,delta_crit;
    int k_lim = update ? 1 : 0;
    double sigma_cond;
    LOG_DEBUG("Initialising dNdM Table from [[%.2e,%.2e],[%.2e,%.2e]]",xmin,xmax,ymin,ymax);
    LOG_DEBUG("D_out %.2e P %.2e up %d",growth1,param,update);

    if(!update){
        lnM_cond = param;
        sigma_cond = EvaluateSigma(lnM_cond,0,NULL);
        //current barrier at the condition for bounds checking
        delta_crit = get_delta_crit(user_params_it->HMF,sigma_cond,growth1);
        if(xmin < DELTA_MIN || xmax > MAX_DELTAC_FRAC*delta_crit){
            LOG_ERROR("Invalid delta [%.5f,%.5f] Either too close to critical density (> 0.999 * %.5f) OR negative mass",xmin,xmax,delta_crit);
            Throw(ValueError);
        }
    }
    nx = global_params.N_COND_INTERP;
    ny = global_params.N_MASS_INTERP;
    np = global_params.N_PROB_INTERP;

    double xa[nx], ya[ny], pa[np];

    int i,j,k;
    //set up coordinate grids
    for(i=0;i<nx;i++) xa[i] = xmin + (xmax - xmin)*((double)i)/((double)nx-1);
    for(j=0;j<ny;j++) ya[j] = ymin + (ymax - ymin)*((double)j)/((double)ny-1);
    for(k=0;k<np;k++){
        if(update)
            pa[k] = (double)k/(double)(np-1);
        else
            pa[k] = global_params.MIN_LOGPROB*(1 - (double)k/(double)(np-1));
    }

    //allocate tables
    if(!Nhalo_table.allocated)
        allocate_RGTable1D(nx,&Nhalo_table);

    Nhalo_table.x_min = xmin;
    Nhalo_table.x_width = (xmax - xmin)/((double)nx-1);

    if(!Mcoll_table.allocated)
        allocate_RGTable1D(nx,&Mcoll_table);

    Mcoll_table.x_min = xmin;
    Mcoll_table.x_width = (xmax - xmin)/((double)nx-1);

    if(!Nhalo_inv_table.allocated)
        allocate_RGTable2D(nx,np,&Nhalo_inv_table);

    Nhalo_inv_table.x_min = xmin;
    Nhalo_inv_table.x_width = (xmax - xmin)/((double)nx-1);
    Nhalo_inv_table.y_min = pa[0];
    Nhalo_inv_table.y_width = pa[1] - pa[0];
    struct parameters_gsl_MF_integrals integral_params = {
                .growthf = growth1,
                .HMF = user_params_it->HMF,
    };

    #pragma omp parallel num_threads(user_params_it->N_THREADS) private(i,j,k) firstprivate(delta_crit,integral_params,sigma_cond,lnM_cond)
    {
        double x,y,buf;
        double norm,fcoll;
        double lnM_prev,lnM_p;
        double prob;
        double p_prev,p_target;
        double k_next;
        double delta;

        #pragma omp for
        for(i=0;i<nx;i++){
            x = xa[i];
            //set the condition
            if(update){
                lnM_cond = x;
                //barrier at given mass
                sigma_cond = EvaluateSigma(lnM_cond,0,NULL);
                delta = get_delta_crit(user_params_it->HMF,sigma_cond,param)/param*growth1;
                // //current barrier at condition for bounds checking
                // delta_crit = get_delta_crit(user_params_it->HMF,EvaluateSigma(lnM_cond,0,NULL),growth1);
            }
            else{
                delta = x;
            }

            integral_params.delta = delta;
            integral_params.sigma_cond = sigma_cond;

            lnM_prev = ymin;
            p_prev = 0;

            if(ymin >= lnM_cond){
                Nhalo_table.y_arr[i] = 0.;
                Mcoll_table.y_arr[i] = 0.;
                for(k=1;k<np-1;k++)
                    Nhalo_inv_table.z_arr[i][k] = ymin;
                continue;
            }

            //BIG TODO: THIS IS SUPER INNEFICIENT, IF THE GL INTEGRATION WORKS FOR THE HALOS I WILL FIND A WAY TO ONLY INITIALISE WHEN I NEED TO
            if(user_params_it->INTEGRATION_METHOD_HALOS == 1)
                initialise_GL(NGL_INT, ymin, lnM_cond);

            norm = IntegratedNdM(ymin, lnM_cond, integral_params,-1, user_params_it->INTEGRATION_METHOD_HALOS);
            fcoll = IntegratedNdM(ymin, lnM_cond, integral_params, -2, user_params_it->INTEGRATION_METHOD_HALOS);
            Nhalo_table.y_arr[i] = norm;
            Mcoll_table.y_arr[i] = fcoll;
            // LOG_DEBUG("cond x: %.2e M [%.2e,%.2e] %.2e d %.2f D %.2f n %d ==> %.8e / %.8e",x,exp(ymin),exp(ymax),exp(lnM_cond),delta,growth1,i,norm,fcoll);

            //if the condition has no halos set the dndm table directly since norm==0 breaks things
            if(norm==0){
                for(k=1;k<np-1;k++)
                    Nhalo_inv_table.z_arr[i][k] = ymin;
                continue;
            }
            // //inverse table limits
            Nhalo_inv_table.z_arr[i][0] = lnM_cond; //will be overwritten in grid if we find MIN_LOGPROB within the mass range
            Nhalo_inv_table.z_arr[i][np-1] = ymin;

            //reset probability finding
            k=np-1;
            p_target = pa[k];
            p_prev = update ? 1. : 0; //start with p==1 from the ymin integral, (logp==0)
            for(j=1;j<ny;j++){
                //done with inverse table
                if(k < k_lim) break;

                //BIG TODO: THIS IS EVEN MORE INNEFICIENT, IF THE GL INTEGRATION WORKS FOR THE HALOS I WILL FIND A WAY TO ONLY INITIALISE WHEN I NEED TO
                if(user_params_it->INTEGRATION_METHOD_HALOS == 1)
                    initialise_GL(NGL_INT, y, lnM_cond);

                y = ya[j];
                if(lnM_cond <= y){
                    //setting to one guarantees samples at lower mass
                    //This fixes upper mass limits for the conditions
                    buf = 0.;
                }
                else{
                    buf = IntegratedNdM(y, lnM_cond, integral_params, -1, user_params_it->INTEGRATION_METHOD_HALOS); //Number density between ymin and y
                }

                prob = buf / norm;
                //catch some norm errors
                if(prob != prob){
                    LOG_ERROR("Normalisation error in table generation");
                    Throw(TableGenerationError);
                }

                //There are time where we have gone over the probability (machine precision) limit before reaching the mass limit
                //  we set the final point to be minimum probability at the maximum mass, which crosses the true CDF in the final bin
                //  but is the best we can do without a rootfind
                if(!update){
                    if(prob == 0.){
                        prob = global_params.MIN_LOGPROB;
                        y = lnM_cond;
                    }
                    else prob = log(prob);
                }
                // LOG_ULTRA_DEBUG("Int || x: %.2e (%d) y: %.2e (%d) ==> %.8e / %.8e",update ? exp(x) : x,i,exp(y),j,prob,p_prev);

                if(p_prev < p_target){
                        LOG_ERROR("Target moved up?");
                        Throw(TableGenerationError);
                }
                //loop through the remaining spaces in the inverse table and fill them
                while(prob <= p_target && k >= k_lim){
                    //since we go ascending in y, prob > prob_prev
                    //NOTE: linear interpolation in (lnMM,log(p)|p)
                    lnM_p = (p_prev-p_target)*(y - lnM_prev)/(p_prev-prob) + lnM_prev;
                    Nhalo_inv_table.z_arr[i][k] = lnM_p;

                    // LOG_ULTRA_DEBUG("Found c: %.2e p: (%.2e,%.2e,%.2e) (c %d, m %d, p %d) z: %.5e",update ? exp(x) : x,p_prev,p_target,prob,i,j,k,exp(lnM_p));

                    k--;
                    p_target=pa[k];
                }
                //keep the value at the previous mass bin for interpolation
                p_prev = prob;
                lnM_prev = y;
            }
        }
    }
    LOG_DEBUG("Done.");
}

//TODO: start rootfind tables again by copying the above and replacing the interpolation with a false positive rootfind

void free_dNdM_tables(){
    int i;
    free_RGTable2D(&Nhalo_inv_table);
    free_RGTable1D(&Nhalo_table);
    free_RGTable1D(&Mcoll_table);
}

//JD: moving the interp table evaluations here since some of them are needed in nu_tau_one
//TODO: Better organisation of the functions here and in spintemp, I should split the interptable case
double EvaluateNionTs(double redshift, double Mlim_Fstar, double Mlim_Fesc){
    //differences in turnover are handled by table setup
    if(user_params_it->USE_INTERPOLATION_TABLES){
        if(flag_options_it->USE_MASS_DEPENDENT_ZETA)
            return EvaluateRGTable1D(redshift,&Nion_z_table); //the correct table should be passed
        return EvaluateRGTable1D(redshift,&fcoll_z_table);
    }

    //minihalos uses a different turnover mass
    if(flag_options_it->USE_MINI_HALOS)
        return Nion_General(redshift, global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, atomic_cooling_threshold(redshift), astro_params_it->ALPHA_STAR, astro_params_it->ALPHA_ESC,
                            astro_params_it->F_STAR10, astro_params_it->F_ESC10, Mlim_Fstar, Mlim_Fesc,user_params_it->INTEGRATION_METHOD_ATOMIC);
    if(flag_options_it->USE_MASS_DEPENDENT_ZETA)
        return Nion_General(redshift, global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_it->M_TURN, astro_params_it->ALPHA_STAR, astro_params_it->ALPHA_ESC,
                            astro_params_it->F_STAR10, astro_params_it->F_ESC10, Mlim_Fstar, Mlim_Fesc,user_params_it->INTEGRATION_METHOD_ATOMIC);

    return FgtrM_General(redshift, minimum_source_mass(redshift,astro_params_it,flag_options_it), user_params_it->INTEGRATION_METHOD_ATOMIC);
}

double EvaluateNionTs_MINI(double redshift, double log10_Mturn_LW_ave, double Mlim_Fstar_MINI, double Mlim_Fesc_MINI){
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return EvaluateRGTable2D(redshift,log10_Mturn_LW_ave,&Nion_z_table_MINI);
    }

    return Nion_General_MINI(redshift, global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, pow(10.,log10_Mturn_LW_ave), atomic_cooling_threshold(redshift),
                            astro_params_it->ALPHA_STAR_MINI, astro_params_it->ALPHA_ESC, astro_params_it->F_STAR7_MINI,
                            astro_params_it->F_ESC7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI,user_params_it->INTEGRATION_METHOD_MINI);
}

double EvaluateSFRD(double redshift, double Mlim_Fstar){
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return EvaluateRGTable1D(redshift,&SFRD_z_table); //assuming you passed the right table
    }

    //minihalos uses a different turnover mass
    if(flag_options_it->USE_MINI_HALOS)
        return Nion_General(redshift, global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, atomic_cooling_threshold(redshift), astro_params_it->ALPHA_STAR, 0.,
                            astro_params_it->F_STAR10, 1., Mlim_Fstar, 0, user_params_it->INTEGRATION_METHOD_ATOMIC);

    if(flag_options_it->USE_MASS_DEPENDENT_ZETA)
        return Nion_General(redshift, global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_it->M_TURN, astro_params_it->ALPHA_STAR, astro_params_it->ALPHA_ESC,
                            astro_params_it->F_STAR10, astro_params_it->F_ESC10, Mlim_Fstar, 0, user_params_it->INTEGRATION_METHOD_ATOMIC);

    //NOTE: Previously, with M_MIN_IN_MASS, the FgtrM function used M_turn/50, which seems like a bug
    // since it goes against the assumption of sharp cutoff

    return FgtrM_General(redshift, minimum_source_mass(redshift,astro_params_it,flag_options_it),user_params_it->INTEGRATION_METHOD_ATOMIC);
}

double EvaluateSFRD_MINI(double redshift, double log10_Mturn_LW_ave, double Mlim_Fstar_MINI){
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return EvaluateRGTable2D(redshift,log10_Mturn_LW_ave,&SFRD_z_table_MINI);
    }
    return Nion_General_MINI(redshift, global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, pow(10.,log10_Mturn_LW_ave), atomic_cooling_threshold(redshift),
                            astro_params_it->ALPHA_STAR_MINI, 0., astro_params_it->F_STAR7_MINI,
                            1., Mlim_Fstar_MINI, 0., user_params_it->INTEGRATION_METHOD_MINI);
}

double EvaluateSFRD_Conditional(double delta, double growthf, double M_min, double M_max, double sigma_max, double Mturn_a, double Mlim_Fstar){
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return exp(EvaluateRGTable1D_f(delta,&SFRD_conditional_table));
    }

    return Nion_ConditionalM(growthf,log(M_min),log(M_max),sigma_max,delta,Mturn_a,
                                astro_params_it->ALPHA_STAR,0.,astro_params_it->F_STAR10,1.,Mlim_Fstar,0., user_params_it->INTEGRATION_METHOD_ATOMIC);
}

double EvaluateSFRD_Conditional_MINI(double delta, double log10Mturn_m, double growthf, double M_min, double M_max, double sigma_max, double Mturn_a, double Mlim_Fstar){
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return exp(EvaluateRGTable2D_f(delta,log10Mturn_m,&SFRD_conditional_table));
    }

    return Nion_ConditionalM_MINI(growthf,log(M_min),log(M_max),sigma_max,delta,pow(10,log10Mturn_m),Mturn_a,astro_params_it->ALPHA_STAR_MINI,
                                0.,astro_params_it->F_STAR7_MINI,1.,Mlim_Fstar, 0., user_params_it->INTEGRATION_METHOD_MINI);
}

double EvaluateNion_Conditional(double delta, double log10Mturn, double growthf, double M_min, double M_max, double sigma_max,
                                double Mlim_Fstar, double Mlim_Fesc, bool prev){
    // LOG_SUPER_DEBUG("Evaluating Nion C d %.2e min %.2e wid %.2e max %.2e",delta,Nion_conditional_table1D.x_min,Nion_conditional_table1D.x_width,
    //                 Nion_conditional_table1D.x_min + Nion_conditional_table1D.n_bin*Nion_conditional_table1D.x_width);
    struct RGTable2D_f *table = prev ? &Nion_conditional_table_prev : &Nion_conditional_table2D;
    if(user_params_it->USE_INTERPOLATION_TABLES){
        if(flag_options_it->USE_MINI_HALOS)
            return exp(EvaluateRGTable2D_f(delta, log10Mturn, table));
        return exp(EvaluateRGTable1D_f(delta, &Nion_conditional_table1D));
    }

    return Nion_ConditionalM(growthf,log(M_min),log(M_max),sigma_max,delta,pow(10,log10Mturn),
                                astro_params_it->ALPHA_STAR,astro_params_it->ALPHA_ESC,astro_params_it->F_STAR10,
                                astro_params_it->F_ESC10,Mlim_Fstar,Mlim_Fesc,user_params_it->INTEGRATION_METHOD_ATOMIC);
}

double EvaluateNion_Conditional_MINI(double delta, double log10Mturn_m, double growthf, double M_min, double M_max, double sigma_max,
                                    double Mturn_a, double Mlim_Fstar, double Mlim_Fesc, bool prev){
    struct RGTable2D_f *table = prev ? &Nion_conditional_table_MINI_prev : &Nion_conditional_table_MINI;
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return exp(EvaluateRGTable2D_f(delta,log10Mturn_m,table));
    }

    return Nion_ConditionalM_MINI(growthf,log(M_min),log(M_max),sigma_max,delta,pow(10,log10Mturn_m),Mturn_a,astro_params_it->ALPHA_STAR_MINI,
                                astro_params_it->ALPHA_ESC,astro_params_it->F_STAR7_MINI,astro_params_it->F_ESC7_MINI,Mlim_Fstar,
                                Mlim_Fesc,user_params_it->INTEGRATION_METHOD_MINI);
}

double EvaluateFcoll_delta(double delta, double growthf, double sigma_min, double sigma_max){
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return EvaluateRGTable1D_f(delta,&fcoll_conditional_table);
    }

    return FgtrM_bias_fast(growthf,delta,sigma_min,sigma_max);
}
double EvaluatedFcolldz(double delta, double redshift, double sigma_min, double sigma_max){
    if(user_params_it->USE_INTERPOLATION_TABLES){
        return EvaluateRGTable1D_f(delta,&dfcoll_conditional_table);
    }
    return dfcoll_dz(redshift,delta,sigma_min,sigma_max);
}

//These tables are always allocated so we do not need to combine tables and non-tables into a single evaluation function
double EvaluateNhalo(double condition){
    return EvaluateRGTable1D(condition,&Nhalo_table);
}

double EvaluateMcoll(double condition){
    return EvaluateRGTable1D(condition,&Mcoll_table);
}

double EvaluateNhaloInv(double condition, double prob){
    return EvaluateRGTable2D(condition,prob,&Nhalo_inv_table);
}
