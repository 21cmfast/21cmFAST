/* This file defines specific interpolation table initialisation functions, kept separate from the general interpolation table routines
   In order to allow them to use calculations based on other interpolation tables. Most importantly these fucntions require those from ps.c
   which requires the sigma(M) interpolation tables */

//MOVE THESE WHEN YOU MOVE THE COND TABLES
#define NDELTA 400
#define NMTURN 50//100
#define LOG10_MTURN_MAX ((double)(10))
#define LOG10_MTURN_MIN ((double)(5.-9e-8))

//NOTE: this table is initialised for up to N_redshift x N_Mturn, but only called N_filter times to assign ST_over_PS in Spintemp.
//  It may be better to just do the integrals at each R
void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI,
                             float mturn_a_const, int minihalos, struct RGTable1D * table, struct RGTable2D * table_mini){
    int i,j;
    float Mmin, Mmax;
    Mmin = minihalos ? global_params.M_MIN_INTEGRAL : mturn_a_const/50.;
    Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fstar_MINI;
    double z_val, mturn_val;

    LOG_DEBUG("initing SFRD spline from %.2f to %.2f",zmin,zmax);

    if (!table->allocated){
        allocate_RGTable1D(Nbin,table);
    }
    if(minihalos && !table_mini->allocated){
        allocate_RGTable2D(Nbin,NMTURN,table_mini);
    }
    table->x_min = zmin;
    table->x_width = (zmax - zmin)/((double)Nbin-1.);

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    if(minihalos){
        table_mini->x_min = zmin;
        table_mini->x_width = (zmax - zmin)/((double)Nbin-1.);
        table_mini->y_min = LOG10_MTURN_MIN;
        table_mini->y_width = (LOG10_MTURN_MAX-LOG10_MTURN_MIN)/((double)NMTURN-1.);
        Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));
    }

    #pragma omp parallel private(i,j) num_threads(user_params_it->N_THREADS)
    {
        float Mcrit_atom_val = mturn_a_const;
        #pragma omp for
        for (i=0; i<Nbin; i++){
            z_val = table->x_min + i*table->x_width; //both tables will have the same values here
            if(minihalos)Mcrit_atom_val = atomic_cooling_threshold(z_val);

            table->y_arr[i] = Nion_General(z_val, Mmin, Mmax, Mcrit_atom_val, Alpha_star, 0., Fstar10, 1.,Mlim_Fstar,0.);
            if(minihalos){
                for (j=0; j<NMTURN; j++){
                    mturn_val = pow(10,table_mini->y_min + j*table_mini->y_width);
                    table_mini->z_arr[i][j] = Nion_General_MINI(z_val, Mmin, Mmax, mturn_val, Mcrit_atom_val, Alpha_star_mini, 0., Fstar7_MINI, 1.,Mlim_Fstar_MINI,0.);
                }
            }
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(table->y_arr[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in SFRD table");
            Throw(TableGenerationError);
        }
        if(minihalos){
            for (j=0; j<NMTURN; j++){
                if(isfinite(table_mini->z_arr[i][j])==0) {
                    LOG_ERROR("Detected either an infinite or NaN value in SFRD_MINI table");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

//Unlike the SFRD spline, this one is used more due to the nu_tau_one() rootfind
void initialise_Nion_Ts_spline(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10,
                                float Fesc10, float Fstar7_MINI, float Fesc7_MINI, float mturn_a_const, int minihalos, struct RGTable1D * table,
                                struct RGTable2D * table_mini){
    int i,j;
    //SigmaMInterp table has different limits with minihalos
    //TODO: make uniform
    float Mmin, Mmax;
    Mmin = minihalos ? global_params.M_MIN_INTEGRAL : mturn_a_const / 50;
    Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fesc, Mlim_Fstar_MINI, Mlim_Fesc_MINI;
    LOG_DEBUG("initing Nion spline from %.2f to %.2f",zmin,zmax);
    double z_val, mturn_val;

    if (!table->allocated){
        allocate_RGTable1D(Nbin,table);
    }
    if(minihalos && !table_mini->allocated){
        allocate_RGTable2D(Nbin,NMTURN,table_mini);
    }
    table->x_min = zmin;
    table->x_width = (zmax - zmin)/((double)Nbin-1.);

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fesc = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc10);
    if(minihalos){
        table_mini->x_min = zmin;
        table_mini->x_width = (zmax - zmin)/((double)Nbin-1.);
        table_mini->y_min = LOG10_MTURN_MIN;
        table_mini->y_width = (LOG10_MTURN_MAX-LOG10_MTURN_MIN)/((double)NMTURN-1.);
        Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));
        Mlim_Fesc_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc7_MINI * pow(1e3, Alpha_esc));
    }

#pragma omp parallel private(i,j) num_threads(user_params_it->N_THREADS)
    {
        float Mcrit_atom_val = mturn_a_const;
#pragma omp for
        for (i=0; i<Nbin; i++){
            z_val = table->x_min + i*table->x_width; //both tables will have the same values here
            if(minihalos)Mcrit_atom_val = atomic_cooling_threshold(z_val);

            table->y_arr[i] = Nion_General(z_val, Mmin, Mmax, Mcrit_atom_val, Alpha_star, Alpha_esc, Fstar10, Fesc10, Mlim_Fstar, Mlim_Fesc);
            if(minihalos){
                for (j=0; j<NMTURN; j++){
                    mturn_val = pow(10,table_mini->y_min + j*table_mini->y_width);
                    table_mini->z_arr[i][j] = Nion_General_MINI(z_val, Mmin, Mmax, mturn_val, Mcrit_atom_val, Alpha_star_mini, Alpha_esc, Fstar7_MINI, Fesc7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI);
                }
            }
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(table->y_arr[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val");
            Throw(TableGenerationError);
        }
        if(minihalos){
            for (j=0; j<NMTURN; j++){
                if(isfinite(table_mini->z_arr[i][j])==0){
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
void initialise_FgtrM_delta_table_all(int n_radii, double *min_dens, double *max_dens, double *z_array, double *growth_array, double *smin_array, double *smax_array , struct RGTable1D *F_tables, struct RGTable1D *dF_tables){
    int i,j;
    double dens;

    for(i=0;i<n_radii;i++){
        if(!F_tables[i].allocated){
            allocate_RGTable1D(dens_Ninterp,&F_tables[i]);
            F_tables[i].x_min = min_dens[i];
            F_tables[i].x_width = (max_dens[i] - min_dens[i])/(dens_Ninterp - 1.);
        }
        if(!dF_tables[i].allocated){
            allocate_RGTable1D(dens_Ninterp,&dF_tables[i]);
            dF_tables[i].x_min = min_dens[i];
            dF_tables[i].x_width = (max_dens[i] - min_dens[i])/(dens_Ninterp - 1.);
        }
    }

    for(i=0;i<n_radii;i++){
        //dens_Ninterp is a global define, probably shouldn't be
        for(j=0;j<dens_Ninterp;j++){
            dens = F_tables[i].x_min + j*F_tables[i].x_width;
            F_tables[i].y_arr[j] = FgtrM_bias_fast(growth_array[i], dens, smin_array[i], smax_array[i]);
            dF_tables[i].y_arr[j] = dfcoll_dz(z_array[i], smin_array[i], dens, smax_array[i]);
        }
    }
    LOG_DEBUG("done");
}

void initialise_FgtrM_delta_table_one(double min_dens, double max_dens, double zpp, double growth_zpp, double smin_zpp, double smax_zpp, struct RGTable1D *F_table, struct RGTable1D *dF_table){
    int i,j;
    double dens;

    if(!F_table->allocated){
        allocate_RGTable1D(dens_Ninterp,F_table);
        F_table->x_min = min_dens;
        F_table->x_width = (max_dens - min_dens)/(dens_Ninterp - 1.);
    }
    if(!dF_table->allocated){
        allocate_RGTable1D(dens_Ninterp,dF_table);
        dF_table->x_min = min_dens;
        dF_table->x_width = (max_dens - min_dens)/(dens_Ninterp - 1.);
    }

    //dens_Ninterp is a global define, probably shouldn't be
    for(j=0;j<dens_Ninterp;j++){
        dens = F_table->x_min + j*F_table->x_width;
        F_table->y_arr[j] = FgtrM_bias_fast(growth_zpp, dens, smin_zpp, smax_zpp);
        dF_table->y_arr[j] = dfcoll_dz(zpp, smin_zpp, dens, smax_zpp);
    }
}

//TODO: change to same z-bins as other global 1D tables
void init_FcollTable(double zmin, double zmax, struct AstroParams *astro_params, struct FlagOptions *flag_options, struct RGTable1D *table)
{
    int i;
    double z_val;

    table->x_min = zmin;
    table->x_width = 0.1;

    int n_z = (int)ceil((zmax - zmin)/table->x_width);

    if(!table->allocated){
        allocate_RGTable1D(n_z,table);
    }

    for(i=0;i<n_z;i++){
        z_val = table->x_min + i*table->x_width;
        //NOTE: previously this divided Mturn by 50 which I think is a bug with M_MIN_in_Mass, since there is a sharp cutoff
        table->y_arr[i] = FgtrM(z_val, minimum_source_mass(z_val,astro_params,flag_options));
    }
}

//NOTE: since reionisation feedback is not included in the Ts calculation, the SFRD spline
//  is Rx1D unlike the Mini table, which is Rx2D
//NOTE: SFRD tables have fixed Mturn range, Nion tables vary
//TODO: fix the confusing Mmax=log(Mmax) naming
//TODO: it would be slightly less accurate but maybe faster to tabulate in linear delta, linear Fcoll
//  rather than linear-log, check the profiles

void initialise_Nion_Conditional_spline(float z, float Mcrit_atom, float min_density, float max_density,
                                     float Mmin, float Mmax, float log10Mturn_min, float log10Mturn_max,
                                     float log10Mturn_min_MINI, float log10Mturn_max_MINI, float Alpha_star,
                                     float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10,
                                     float Mlim_Fstar, float Mlim_Fesc, float Fstar7_MINI, float Fesc7_MINI,
                                     float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, bool FAST_FCOLL_TABLES,
                                     bool minihalos, struct RGTable1D_f *table_1d, struct RGTable2D_f *table_2d, struct RGTable2D_f *table_mini){
    double growthf, sigma2;
    int i,j;
    double overdense_table[NDELTA];
    double mturns[NMTURN], mturns_MINI[NMTURN];

    growthf = dicke(z);
    Mmin = log(Mmin);
    Mmax = log(Mmax);

    sigma2 = EvaluateSigma(Mmax,0,NULL);

    //If we use minihalos, both tables are 2D (delta,mturn) due to reionisaiton feedback
    //otherwise, the Nion table is 1D, since reionsaiton feedback is only active with minihalos
    if (minihalos){
        if(!table_2d->allocated) {
            LOG_SUPER_DEBUG("allocating interp Nion2d");
            allocate_RGTable2D_f(NDELTA,NMTURN,table_2d);
        }
        if(!table_mini->allocated) {
            LOG_SUPER_DEBUG("allocating interp Nion2d");
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
        if(!table_1d->allocated) {
            LOG_SUPER_DEBUG("allocating interp Nion2d");
            allocate_RGTable1D_f(NDELTA,table_1d);
        }
        table_1d->x_min = min_density;
        table_1d->x_width = (max_density - min_density)/(NDELTA-1.);
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

#pragma omp parallel private(i,j) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for(i=0;i<NDELTA;i++){
            if(!minihalos){
                //pass constant M_turn as minimum
                table_1d->y_arr[i] = log(Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,Deltac,
                                                overdense_table[i],Mcrit_atom,Alpha_star,Alpha_esc,
                                                Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,FAST_FCOLL_TABLES));
                if(table_1d->y_arr[i] < -40.)
                    table_1d->y_arr[i] = -40.;

                continue;
            }
            for (j=0; j<NMTURN; j++){
                table_2d->z_arr[i][j] = log(Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,Deltac,
                                                overdense_table[i],mturns[j],Alpha_star,Alpha_esc,
                                                Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,FAST_FCOLL_TABLES));

                if(table_2d->z_arr[i][j] < -40.)
                    table_2d->z_arr[i][j] = -40.;

                table_mini->z_arr[i][j] = log(Nion_ConditionalM_MINI(growthf,Mmin,Mmax,sigma2,Deltac,overdense_table[i],
                                                    mturns_MINI[j],Mcrit_atom,Alpha_star_mini,Alpha_esc,Fstar7_MINI,Fesc7_MINI,
                                                    Mlim_Fstar_MINI,Mlim_Fesc_MINI, FAST_FCOLL_TABLES));

                if(table_mini->z_arr[i][j] < -40.)
                    table_mini->z_arr[i][j] = -40.;
            }
        }
    }

    for(i=0;i<NDELTA;i++) {
        if(!minihalos){
            if(isfinite(table_1d->y_arr[i])==0) {
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
void initialise_SFRD_Conditional_table_all(int Nfilter, double min_density[], double max_density[], double growthf[],
                                    double R[], float Mcrit_atom[], double Mmin, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI, bool FAST_FCOLL_TABLES, bool minihalos, struct RGTable1D_f *tables, struct RGTable2D_f *tables_mini){
    float Mmax,Mlim_Fstar,sigma2,Mlim_Fstar_MINI;
    int i,j,k,i_tot;

    Mmax = RtoM(R[Nfilter-1]);
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));

    float MassTurnover[NMTURN];
    for(i=0;i<NMTURN;i++){
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
    }

    //NOTE: Here we use the constant Mturn limits instead of variables like in the Nion tables
    for(i=0;i<Nfilter;i++){
        if(!tables[i].allocated){
            allocate_RGTable1D_f(NDELTA,&tables[i]);
        }
        tables[i].x_min=min_density[i];
        tables[i].x_width=(max_density[i] - min_density[i])/(NDELTA-1.);

        if(!tables_mini[i].allocated){
            allocate_RGTable2D_f(NDELTA,NMTURN,&tables_mini[i]);
        }
        tables_mini[i].x_min = min_density[i];
        tables_mini[i].x_width = (max_density[i] - min_density[i])/(NDELTA-1.);
        tables_mini[i].y_min = LOG10_MTURN_MIN;
        tables_mini[i].x_width = (LOG10_MTURN_MAX - LOG10_MTURN_MIN)/(NMTURN-1.);
    }

    LOG_DEBUG("Initialising SFRD conditional table");

    Mmin = log(Mmin);

    for(j=0; j < Nfilter; j++){
        Mmax = RtoM(R[j]);

        Mmax = log(Mmax);
        sigma2 = EvaluateSigma(Mmax,0,NULL);

#pragma omp parallel private(i,k) num_threads(user_params_ps->N_THREADS)
        {
            double curr_dens;
#pragma omp for
            for (i=0; i<NDELTA; i++){
                curr_dens = min_density[j] + (float)i/((float)NDELTA-1.)*(max_density[j] - min_density[j]);
                tables[j].y_arr[i] = log(Nion_ConditionalM(growthf[j],Mmin,Mmax,sigma2,Deltac,curr_dens,\
                                                            Mcrit_atom[j],Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES));

                if(tables[j].y_arr[i] < -50.)
                    tables[j].y_arr[i] = -50.;

                if(!minihalos) continue;

                for (k=0; k<NMTURN; k++){
                    tables_mini[j].z_arr[i][k]  = log(Nion_ConditionalM_MINI(growthf[j],Mmin,Mmax,sigma2,Deltac,\
                                                curr_dens,MassTurnover[k],Mcrit_atom[j],\
                                                Alpha_star_mini,0.,Fstar7_MINI,1.,Mlim_Fstar_MINI, 0., FAST_FCOLL_TABLES));

                    if(tables_mini[j].z_arr[i][k] < -50.)
                        tables_mini[j].z_arr[i][k] = -50.;
                }
            }
        }
    }
    for(j=0;j<Nfilter;j++){
        for (i=0; i<NDELTA; i++){
            if(isfinite(tables[j].y_arr[i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in ACG SFRD conditional table");
                Throw(TableGenerationError);
            }

            if(!minihalos) continue;

            for (k=0; k<NMTURN; k++){
                if(isfinite(tables_mini[j].z_arr[i][k])==0) {
                    LOG_ERROR("Detected either an infinite or NaN value in MCG SFRD conditional table");
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

//To better localise the tables, I could either separate each radius like in Ionisationbox.c, OR calculate all SFRD at once ouside the R loop
//      For now I'm doing the former but if/when I move the non-halos to XraySourceBox I will use the _all version
void initialise_SFRD_Conditional_table_one(double min_density, double max_density, double growthf,
                                    double R, float Mcrit_atom, double Mmin, double Mmax, float Alpha_star, float Alpha_star_mini,
                                    float Fstar10, float Fstar7_MINI, bool FAST_FCOLL_TABLES, bool minihalos, struct RGTable1D_f *table, struct RGTable2D_f *table_mini){
    float Mlim_Fstar,sigma2,Mlim_Fstar_MINI;
    int i,j,k,i_tot;

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));

    float MassTurnover[NMTURN];
    for(i=0;i<NMTURN;i++){
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
    }

    //NOTE: Here we use the constant Mturn limits instead of variables like in the Nion tables
    if(!table->allocated){
        allocate_RGTable1D_f(NDELTA,table);
    }
    table->x_min = min_density;
    table->x_width = (max_density - min_density)/(NDELTA-1.);

    if(!table_mini->allocated){
        allocate_RGTable2D_f(NDELTA,NMTURN,table_mini);
    }
    table_mini->x_min = min_density;
    table_mini->x_width = (max_density - min_density)/(NDELTA-1.);
    table_mini->y_min = LOG10_MTURN_MIN;
    table_mini->x_width = (LOG10_MTURN_MAX - LOG10_MTURN_MIN)/(NMTURN-1.);

    double lnMmin = log(Mmin);
    double lnMmax = log(Mmax);
    sigma2 = EvaluateSigma(lnMmax,0,NULL);

#pragma omp parallel private(i,k) num_threads(user_params_ps->N_THREADS)
    {
        double curr_dens;
#pragma omp for
        for (i=0; i<NDELTA; i++){
            curr_dens = min_density + (float)i/((float)NDELTA-1.)*(max_density - min_density);
            table->y_arr[i] = log(Nion_ConditionalM(growthf,lnMmin,lnMmax,sigma2,Deltac,curr_dens,\
                                                        Mcrit_atom,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES));

            if(table->y_arr[i] < -50.)
                table->y_arr[i] = -50.;

            if(!minihalos) continue;

            for (k=0; k<NMTURN; k++){
                table_mini->z_arr[i][k]  = log(Nion_ConditionalM_MINI(growthf,lnMmin,lnMmax,sigma2,Deltac,\
                                            curr_dens,MassTurnover[k],Mcrit_atom,\
                                            Alpha_star_mini,0.,Fstar7_MINI,1.,Mlim_Fstar_MINI, 0., FAST_FCOLL_TABLES));

                if(table_mini->z_arr[i][k] < -50.)
                    table_mini->z_arr[i][k] = -50.;
            }
        }
    }
    for (i=0; i<NDELTA; i++){
        if(isfinite(table->y_arr[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in ACG SFRD conditional table");
            Throw(TableGenerationError);
        }
        if(!minihalos) continue;

        for (k=0; k<NMTURN; k++){
            if(isfinite(table_mini->z_arr[i][k])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in MCG SFRD conditional table");
                Throw(TableGenerationError);
            }
        }
    }
}
