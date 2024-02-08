
//calculates halo properties from astro parameters plus the correlated rng
//TODO: make the input and output labeled structs
//The inputs include all properties with a separate RNG
//The outputs include all halo properties PLUS all properties which cannot be recovered when mixing all the halos together
//  i.e escape fraction weighting, minihalo stuff that has separate parameters
//Since there are so many spectral terms in the spin temperature calculation, it will be most efficient to split SFR into regular and minihalos
//  BUT not split the ionisedbox fields. i.e
//INPUT ARRAY || 0: Stellar mass RNG, 1: SFR RNG
//OUTPUTS FOR RADIATIVE BACKGROUNDS: 0: SFR, 1: SFR_MINI, 2: n_ion 3: f_esc and N_ion weighted SFR (for gamma)
//OUTPUTS FOR HALO BOXES: 4: Stellar mass, 5: Stellar mass (minihalo)
//in order to remain consistent with the minihalo treatment in default (Nion_a * exp(-M/M_a) + Nion_m * exp(-M/M_m - M_a/M))
//  we treat the minihalos as a shift in the mean, where each halo will have both components, representing a smooth
//  transition in halo mass from one set of SFR/emmissivity parameters to the other.
void set_halo_properties(float halo_mass, float M_turn_a, float M_turn_m, float t_h, double norm_esc_var, double alpha_esc_var, float * input, float * output){
    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;
    double sigma_sfr = astro_params_stoc->SIGMA_SFR;

    double f7 = astro_params_stoc->F_STAR7_MINI;
    double fa_m = astro_params_stoc->ALPHA_STAR_MINI;
    double fesc7 = astro_params_stoc->F_ESC7_MINI;

    double fstar_mean, fstar_mean_mini, sfr_mean, sfr_mean_mini;
    double f_sample, f_sample_mini, sm_sample, n_ion_sample, sfr_sample, wsfr_sample;
    double f_rng, sfr_rng;
    double sm_sample_mini, sfr_sample_mini;
    double fesc,fesc_mini;

    //TODO: It could save some `pow` calls with F_esc if I compute a mass limit for fesc outside the loop
    //NOTE: I can only do this if f_esc remains non-stochastic, this will also be irrelevant with mean property interptables
    fesc = fmin(norm_esc_var*pow(halo_mass/1e10,alpha_esc_var),1);

    //A flattening of the high-mass FSTAR, HACKY VERSION FOR NOW
    //TODO: code it properly with new parameters and pivot point defined somewhere
    //NOTE: we don't want an upturn even with a negative ALPHA_STAR
    if(astro_params_stoc->ALPHA_STAR > -0.61){
        fstar_mean = f10 * exp(-M_turn_a/halo_mass) * pow(2.6e11/1e10,astro_params_stoc->ALPHA_STAR);
        fstar_mean /= pow(halo_mass/2.8e11,-astro_params_stoc->ALPHA_STAR) + pow(halo_mass/2.8e11,0.61);
    }
    else{
        fstar_mean = f10 * pow(halo_mass/1e10,fa) * exp(-M_turn_a/halo_mass);
    }

    //TODO: apply some version of the Mcrit smoothing which happens in the minihalo model
    //NOTE: that smoothing is a trapezoidal integration (assuming constant Mturn in the step)
    //  This has some implications for my model, the halo history SHOULD matter in terms of Nion
    //  i.e the turnover mass HISTORY of a halo should change its total n_ion.
    //  However it is not clear to me how this can be implemented, since I would need not only the
    //  previous M_turn grids but also a previous halo mass (with possibly many progenitors)
    if(flag_options_stoc->USE_MINI_HALOS){
        fesc_mini = fmin(fesc7*pow(halo_mass/1e7,alpha_esc_var),1);
        fstar_mean_mini = f7 * pow(halo_mass/1e7,fa_m) * exp(-M_turn_m/halo_mass - halo_mass/M_turn_a);
    }
    else{
        fstar_mean_mini = 0;
        fesc_mini = 0.;
    }

    /* Simply adding lognormal scatter to a delta increases the mean (2* is as likely as 0.5*)
    * We multiply by exp(-sigma^2/2) so that X = exp(mu + N(0,1)*sigma) has the desired mean */
    f_rng = exp(-sigma_star*sigma_star/2 + input[0]*sigma_star);

    //This clipping is normally done with the mass_limit_bisection root find. hard to do with stochastic
    //TODO: Interpolation tables for all the mean relations? (is this really faster than a coulple pow and exp calls?)
    f_sample = fmin(fstar_mean * f_rng,1);
    f_sample_mini = fmin(fstar_mean_mini * f_rng,1);

    sm_sample = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * f_sample; //f_star is galactic GAS/star fraction, so OMb is needed
    sm_sample_mini = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * f_sample_mini; //f_star is galactic GAS/star fraction, so OMb is needed

    sfr_mean = sm_sample / (astro_params_stoc->t_STAR * t_h);
    sfr_mean_mini = sm_sample_mini / (astro_params_stoc->t_STAR * t_h);

    //Since there's no clipping on t_STAR, we can apply the lognormal to SFR directly instead of t_STAR
    sfr_rng = exp(-sigma_sfr*sigma_sfr/2 + input[1]*sigma_sfr);
    sfr_sample = sfr_mean * sfr_rng;
    sfr_sample_mini = sfr_mean_mini * sfr_rng;

    n_ion_sample = sm_sample*global_params.Pop2_ion*fesc + sm_sample_mini*global_params.Pop3_ion*fesc_mini;
    wsfr_sample = sfr_sample*global_params.Pop2_ion*fesc + sfr_sample_mini*global_params.Pop3_ion*fesc_mini;

    //LOG_ULTRA_DEBUG("HM %.3e | SM %.3e | SFR %.3e (%.3e) | F* %.3e (%.3e) | duty %.3e",halo_mass,sm_sample,sfr_sample,sfr_mean,f_sample,fstar_mean,dutycycle_term);

    output[0] = sfr_sample;
    output[1] = sfr_sample_mini;
    output[2] = n_ion_sample;
    output[3] = wsfr_sample;
    output[4] = sm_sample;
    output[5] = sm_sample_mini;
    return 0;
}

//Fixed halo grids, where each property is set as the integral of the CMF on the EULERIAN cell scale
//As per default 21cmfast (strange pretending that the lagrangian density is eulerian and then *(1+delta))
//This outputs the UN-NORMALISED grids (before mean-adjustment)
//TODO: add minihalos
//TODO: use the interpolation tables (Fixed grids are currently slow but a debug case so this is low priority)
int set_fixed_grids(double redshift, double norm_esc, double alpha_esc, double M_min, double M_max, struct InitialConditions * ini_boxes,
                    struct PerturbedField * perturbed_field, struct TsBox *previous_spin_temp,
                    struct IonizedBox *previous_ionize_box, struct HaloBox *grids, double *averages){
    //There's quite a bit of re-calculation here but this only happens once per snapshot
    double cell_volume = VOLUME / HII_TOT_NUM_PIXELS;
    double M_cell = RHOcrit * cosmo_params_stoc->OMm * cell_volume; //mass in cell of mean dens
    double growth_z = dicke(redshift);
    double alpha_star = astro_params_stoc->ALPHA_STAR;
    double norm_star = astro_params_stoc->F_STAR10;
    double t_h = t_hubble(redshift);

    double alpha_star_mini = astro_params_stoc->ALPHA_STAR_MINI;
    double norm_star_mini = astro_params_stoc->F_STAR7_MINI;
    double norm_esc_mini = astro_params_stoc->F_ESC7_MINI;

    double t_star = astro_params_stoc->t_STAR;

    double lnMmin = log(M_min);
    double lnMcell = log(M_cell);
    double lnMmax = log(M_max);

    double sigma_cell = EvaluateSigma(lnMcell,0,NULL);

    double prefactor_mass = RHOcrit * cosmo_params_stoc->OMm;
    double prefactor_nion = RHOcrit * cosmo_params_stoc->OMb * norm_star * norm_esc * global_params.Pop2_ion;
    double prefactor_nion_mini = RHOcrit * cosmo_params_stoc->OMb * norm_star_mini * norm_esc_mini * global_params.Pop3_ion;
    double prefactor_sfr = RHOcrit * cosmo_params_stoc->OMb * norm_star / t_star / t_h;
    double prefactor_sfr_mini = RHOcrit * cosmo_params_stoc->OMb * norm_star_mini / t_star / t_h;

    double Mlim_Fstar = Mass_limit_bisection(M_min, M_cell, alpha_star, norm_star);
    double Mlim_Fesc = Mass_limit_bisection(M_min, M_cell, alpha_esc, norm_esc);

    double Mlim_Fstar_mini = Mass_limit_bisection(M_min, M_cell, alpha_star, norm_star * pow(1e3,alpha_esc));
    double Mlim_Fesc_mini = Mass_limit_bisection(M_min, M_cell, alpha_esc, norm_esc_mini * pow(1e3,alpha_esc));

    double hm_avg=0, nion_avg=0, sfr_avg=0, sfr_avg_mini=0, wsfr_avg=0;
    double Mlim_a_avg=0, Mlim_m_avg=0;

    //interptable variables
    double min_density = -1;
    double max_density = MAX_DELTAC_FRAC*Deltac;

    double curr_vcb = flag_options_stoc->FIX_VCB_AVG ? global_params.VAVG : 0;
    double delta_crit = get_delta_crit(user_params_stoc->HMF,sigma_cell,growth_z);


    struct parameters_gsl_MF_integrals params = {
            .redshift = redshift,
            .growthf = growth_z,
            .sigma_cond = sigma_cell,
            .HMF = user_params_stoc->HMF,
    };

    if(user_params_stoc->INTEGRATION_METHOD_ATOMIC == 1 || user_params_stoc->INTEGRATION_METHOD_MINI == 1)
        initialise_GL(NGL_INT, lnMmin, lnMmax);

    //store initial values so we don't recompute
    double M_turn_a_store = flag_options_stoc->USE_MINI_HALOS ? atomic_cooling_threshold(redshift) : astro_params_stoc->M_TURN;
    double M_turn_m_store = lyman_werner_threshold(redshift, 0., curr_vcb, astro_params_stoc);
    double M_turn_a = M_turn_a_store;
    double M_turn_m = M_turn_m_store;
    double M_turn_r = 0.;

    struct RGTable1D_f SFRD_conditional_table = {.allocated=false};
    struct RGTable2D_f SFRD_conditional_table_MINI = {.allocated=false};
    struct RGTable1D_f Nion_Conditional_Table1D = {.allocated=false};
    struct RGTable2D_f Nion_Conditional_Table2D = {.allocated=false}, Nion_Conditional_Table_MINI = {.allocated=false};

    //TODO: These tables are coarser than needed, I should do an initial loop to find delta and Mturn limits
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialise_SFRD_Conditional_table(min_density,max_density,growth_z,M_turn_a,M_min,M_max,M_cell,
                                                astro_params_stoc->ALPHA_STAR, astro_params_stoc->ALPHA_STAR_MINI, astro_params_stoc->F_STAR10,
                                                astro_params_stoc->F_STAR7_MINI, user_params_stoc->INTEGRATION_METHOD_ATOMIC,
                                                user_params_stoc->INTEGRATION_METHOD_MINI,
                                                flag_options_stoc->USE_MINI_HALOS,&SFRD_conditional_table,&SFRD_conditional_table_MINI);

        //note: we do not yet have the previous ion table here
        initialise_Nion_Conditional_spline(redshift,M_turn_a,min_density,max_density,M_min,M_max,M_cell,
                                LOG10_MTURN_MIN,LOG10_MTURN_MAX,LOG10_MTURN_MIN,LOG10_MTURN_MAX,
                                astro_params_stoc->ALPHA_STAR, astro_params_stoc->ALPHA_STAR_MINI,
                                alpha_esc,astro_params_stoc->F_STAR10,
                                norm_esc,Mlim_Fstar,Mlim_Fesc,astro_params_stoc->F_STAR7_MINI,
                                astro_params_stoc->F_ESC7_MINI,Mlim_Fstar_mini, Mlim_Fesc_mini,  user_params_stoc->INTEGRATION_METHOD_ATOMIC,
                                user_params_stoc->INTEGRATION_METHOD_MINI,
                                flag_options_stoc->USE_MINI_HALOS, &Nion_Conditional_Table1D, &Nion_Conditional_Table2D, &Nion_Conditional_Table_MINI);

        //TODO: disable inverse table generation here with a flag or split up the functions
        initialise_dNdM_tables(min_density, max_density, lnMmin, lnMmax, growth_z, lnMcell, false);
    }

    LOG_DEBUG("Mean halo boxes || M = [%.2e %.2e] | Mcell = %.2e (s=%.2e) | z = %.2e | D = %.2e | cellvol = %.2e",M_min,M_max,M_cell,sigma_cell,redshift,growth_z,cell_volume);
#pragma omp parallel num_threads(user_params_stoc->N_THREADS) firstprivate(M_turn_m,M_turn_a,M_turn_r,curr_vcb)
    {
        int i;
        double dens=0;
        double mass=0, nion=0, sfr=0, h_count=0;
        double nion_mini=0, sfr_mini=0;
        double wsfr=0;
#pragma omp for reduction(+:hm_avg,nion_avg,sfr_avg,sfr_avg_mini,wsfr_avg,Mlim_a_avg,Mlim_m_avg)
        for(i=0;i<HII_TOT_NUM_PIXELS;i++){
            dens = perturbed_field->density[i];
            params.delta = dens;
            if(!flag_options_stoc->FIX_VCB_AVG && user_params_stoc->USE_RELATIVE_VELOCITIES){
                curr_vcb = ini_boxes->lowres_vcb[i];
            }

            if(flag_options_stoc->USE_MINI_HALOS){
                M_turn_m = lyman_werner_threshold(redshift, previous_spin_temp->J_21_LW_box[i], curr_vcb, astro_params_stoc);
                M_turn_r = reionization_feedback(redshift, previous_ionize_box->Gamma12_box[i], previous_ionize_box->z_re_box[i]);
                if(M_turn_r > M_turn_a) M_turn_a = M_turn_r;
                if(M_turn_r > M_turn_m) M_turn_m = M_turn_r;
            }

            //ignore very low density NOTE:not using DELTA_MIN since it's perturbed (Eulerian)
            if(dens <= -1){
                mass = 0.;
                nion = 0.;
                sfr = 0.;
                h_count = 0;
            }
            //turn into one large halo if we exceed the critical
            //Since these are perturbed (Eulerian) grids, I use the total cell mass (1+dens)
            else if(dens>=MAX_DELTAC_FRAC*delta_crit){
                if(M_cell <= M_max){
                    mass = M_cell * (1+dens) / cell_volume;
                    nion = prefactor_nion * M_cell * (1+dens) / cosmo_params_stoc->OMm / RHOcrit * pow(M_cell*(1+dens)/1e10,alpha_star) * pow(M_cell*(1+dens)/1e10,alpha_esc) / cell_volume;
                    sfr = prefactor_sfr * M_cell * (1+dens) / cosmo_params_stoc->OMm / RHOcrit * pow(M_cell*(1+dens)/1e10,alpha_star) / cell_volume;
                    h_count = 1;
                }
                else{
                    //here the cell delta is above critical, but the integrals do not include the cell mass, so we set to zero
                    //NOTE: this function gives integral [M_min,M_limit] given a cell
                    mass = 0.;
                    nion = 0.;
                    sfr = 0.;
                    h_count = 0;
                }
            }
            else{
                //calling IntegratedNdM with star and SFR need special care for the f*/fesc clipping, and calling NionConditionalM for mass includes duty cycle
                //neither of which I want
                if(user_params_stoc->USE_INTERPOLATION_TABLES){
                    h_count = EvaluateRGTable1D(dens,&Nhalo_table);
                    mass = EvaluateRGTable1D(dens,&Mcoll_table);
                    sfr = exp(EvaluateRGTable1D_f(dens,&SFRD_conditional_table));
                    if(flag_options_stoc->USE_MINI_HALOS){
                        sfr_mini = exp(EvaluateRGTable2D_f(dens,log10(M_turn_m),&SFRD_conditional_table_MINI));
                        nion_mini = exp(EvaluateRGTable2D_f(dens,log10(M_turn_m),&Nion_Conditional_Table_MINI));
                        nion = exp(EvaluateRGTable2D_f(dens,log10(M_turn_a),&Nion_Conditional_Table2D));
                    }
                    else{
                        nion = exp(EvaluateRGTable1D_f(dens,&Nion_Conditional_Table1D));
                    }
                }
                else{
                    //NOTE: we use the atomic method for all halo mass/count here
                    h_count = IntegratedNdM(lnMmin,lnMmax,params,-1,user_params_stoc->INTEGRATION_METHOD_HALOS); //FF doesn't work for halo number yet
                    mass = IntegratedNdM(lnMmin,lnMmax,params,-2,user_params_stoc->INTEGRATION_METHOD_ATOMIC);

                    nion = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_cell, dens, M_turn_a
                                            , alpha_star, alpha_esc, norm_star, norm_esc
                                            , Mlim_Fstar, Mlim_Fesc,  user_params_stoc->INTEGRATION_METHOD_ATOMIC);

                    sfr = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_cell, dens, M_turn_a
                                            , alpha_star, 0., norm_star, 1., Mlim_Fstar, 0.
                                            , user_params_stoc->INTEGRATION_METHOD_ATOMIC);

                    //Same integral as Nion
                    // wsfr = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_max, delta_crit, dens, M_turn_a
                    //                         , astro_params_stoc->ALPHA_STAR, alpha_esc, astro_params_stoc->F_STAR10, norm_esc, Mlim_Fstar, Mlim_Fesc
                    //                         , user_params_stoc->FAST_FCOLL_TABLES);
                    if(flag_options_stoc->USE_MINI_HALOS){
                        nion_mini = Nion_ConditionalM_MINI(growth_z, lnMmin, lnMmax, sigma_cell,
                                                            dens, M_turn_m, M_turn_a, alpha_star_mini,
                                                            alpha_esc, norm_star_mini, norm_esc, Mlim_Fstar_mini,
                                                            Mlim_Fesc_mini, user_params_stoc->INTEGRATION_METHOD_MINI);

                        sfr_mini = Nion_ConditionalM_MINI(growth_z, lnMmin, lnMmax, sigma_cell,
                                                            dens, M_turn_m, M_turn_a, alpha_star_mini,
                                                            0., norm_star_mini, 1., Mlim_Fstar_mini, 0.,
                                                            user_params_stoc->INTEGRATION_METHOD_MINI);
                    }
                }
            }
            grids->halo_mass[i] = mass * prefactor_mass * (1+dens);
            grids->n_ion[i] = (nion*prefactor_nion + nion_mini*prefactor_nion_mini) * (1+dens);
            grids->halo_sfr[i] = (sfr*prefactor_sfr) * (1+dens);
            grids->whalo_sfr[i] = grids->n_ion[i] / t_star / t_h; //no stochasticity so they're the same to a constant
            grids->count[i] = (int)(h_count * prefactor_mass * (1+dens)); //NOTE: truncated
            grids->halo_sfr_mini[i] = sfr_mini*prefactor_sfr_mini * (1+dens); //zero if !Minihalos

            if(i==0 && user_params_stoc->USE_INTERPOLATION_TABLES){
                LOG_SUPER_DEBUG("Cell 0 tables: count %.2e mass %.2e nion %.2e sfr %.2e delta %.2f",h_count,mass,nion,sfr,dens);
                LOG_SUPER_DEBUG("Cell 0 intgrl: count %.2e mass %.2e nion %.2e sfr %.2e",
                                IntegratedNdM(lnMmin,lnMmax,params,-1,user_params_stoc->INTEGRATION_METHOD_ATOMIC),
                                IntegratedNdM(lnMmin,lnMmax,params,-2,user_params_stoc->INTEGRATION_METHOD_ATOMIC),
                                Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_cell, dens, M_turn_a
                                            , astro_params_stoc->ALPHA_STAR, alpha_esc, astro_params_stoc->F_STAR10, norm_esc
                                            , Mlim_Fstar, Mlim_Fesc, user_params_stoc->INTEGRATION_METHOD_ATOMIC),
                                Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_cell, dens, M_turn_a
                                            , alpha_star, 0., norm_star, 1., Mlim_Fstar, 0.
                                            , user_params_stoc->INTEGRATION_METHOD_ATOMIC));
                LOG_SUPER_DEBUG("Cell 0 grids: count %d mass %.2e nion %.2e sfr %.2e", grids->count[i],
                                    grids->halo_mass[i],grids->n_ion[i],grids->halo_sfr[i]);
            }

            hm_avg += grids->halo_mass[i];
            nion_avg += grids->n_ion[i];
            sfr_avg += grids->halo_sfr[i];
            wsfr_avg += grids->whalo_sfr[i];
            sfr_avg_mini += grids->halo_sfr_mini[i];
            Mlim_a_avg += M_turn_a;
            Mlim_m_avg += M_turn_m;
        }
    }

    free_RGTable1D_f(&Nion_Conditional_Table1D);
    free_RGTable2D_f(&Nion_Conditional_Table2D);
    free_RGTable2D_f(&Nion_Conditional_Table_MINI);
    free_RGTable1D_f(&SFRD_conditional_table);
    free_RGTable2D_f(&SFRD_conditional_table_MINI);

    hm_avg /= HII_TOT_NUM_PIXELS;
    nion_avg /= HII_TOT_NUM_PIXELS;
    sfr_avg /= HII_TOT_NUM_PIXELS;
    sfr_avg_mini /= HII_TOT_NUM_PIXELS;
    wsfr_avg /= HII_TOT_NUM_PIXELS;
    Mlim_a_avg /= HII_TOT_NUM_PIXELS;
    Mlim_m_avg /= HII_TOT_NUM_PIXELS;

    averages[0] = hm_avg;
    averages[1] = sfr_avg;
    averages[2] = sfr_avg_mini;
    averages[3] = nion_avg;
    averages[4] = wsfr_avg;
    averages[5] = Mlim_a_avg;
    averages[6] = Mlim_m_avg;

    return 0;
}

//Expected global averages for box quantities for mean adjustment
//TODO: use the global interpolation tables (only one integral per property per snapshot so this is low priority)
//WARNING: THESE AVERAGE BOXES ARE WRONG, CHECK THEM
//TODO: Use the functions from SpinTemperature.c Instead with the tables
int get_box_averages(double redshift, double norm_esc, double alpha_esc, double M_min, double M_max, double M_turn_a, double M_turn_m, double *averages){
    double alpha_star = astro_params_stoc->ALPHA_STAR;
    double norm_star = astro_params_stoc->F_STAR10;
    double t_star = astro_params_stoc->t_STAR;
    double t_h = t_hubble(redshift);

    LOG_SUPER_DEBUG("Getting Box averages z=%.2f M [%.2e %.2e] Mt [%.2e %.2e]",redshift,M_min,M_max,M_turn_a,M_turn_m);

    double alpha_star_mini = astro_params_stoc->ALPHA_STAR_MINI;
    double norm_star_mini = astro_params_stoc->F_STAR7_MINI;
    double norm_esc_mini = astro_params_stoc->F_ESC7_MINI;
    //There's quite a bit of re-calculation here but this only happens once per snapshot
    double lnMmax = log(M_max);
    double lnMmin = log(M_min);
    double growth_z = dicke(redshift);

    // double prefactor_mass = RHOcrit * cosmo_params_stoc->OMm;
    double prefactor_nion = RHOcrit * cosmo_params_stoc->OMb * norm_star * norm_esc * global_params.Pop2_ion;
    double prefactor_nion_mini = RHOcrit * cosmo_params_stoc->OMb * norm_star_mini * norm_esc_mini * global_params.Pop3_ion;
    double prefactor_sfr = RHOcrit * cosmo_params_stoc->OMb * norm_star / t_star / t_h;
    double prefactor_sfr_mini = RHOcrit * cosmo_params_stoc->OMb * norm_star_mini / t_star / t_h;

    double hm_expected,nion_expected,sfr_expected,wsfr_expected,sfr_expected_mini=0;

    double Mlim_Fstar = Mass_limit_bisection(M_min, M_max, alpha_star, norm_star);
    double Mlim_Fesc = Mass_limit_bisection(M_min, M_max, alpha_esc, norm_esc);
    double Mlim_Fstar_mini = Mass_limit_bisection(M_min, M_max, alpha_star, norm_star * pow(1e3,alpha_esc));
    double Mlim_Fesc_mini = Mass_limit_bisection(M_min, M_max, alpha_esc, norm_esc_mini * pow(1e3,alpha_esc));

    struct parameters_gsl_MF_integrals params = {
            .redshift = redshift,
            .growthf = growth_z,
            .HMF = user_params_stoc->HMF,
    };

    if(user_params_stoc->INTEGRATION_METHOD_ATOMIC == 1 || user_params_stoc->INTEGRATION_METHOD_MINI == 1)
        initialise_GL(NGL_INT, lnMmin, lnMmax);

    //NOTE: we use the atomic method for all halo mass/count here
    hm_expected = IntegratedNdM(lnMmin,lnMmax,params,1,user_params_stoc->INTEGRATION_METHOD_ATOMIC);
    nion_expected = Nion_General(redshift, M_min, M_max, M_turn_a, alpha_star, alpha_esc, norm_star,
                                 norm_esc, Mlim_Fstar, Mlim_Fesc,user_params_stoc->INTEGRATION_METHOD_ATOMIC) * prefactor_nion;
    sfr_expected = Nion_General(redshift, M_min, M_max, M_turn_a, alpha_star, 0., norm_star, 1.,
                                 Mlim_Fstar, 0.,user_params_stoc->INTEGRATION_METHOD_ATOMIC) * prefactor_sfr;
    // wsfr_expected = Nion_General(redshift, M_min, M_turn_a, alpha_star, alpha_esc, norm_star, norm_esc, Mlim_Fstar, Mlim_Fesc);
    if(flag_options_stoc->USE_MINI_HALOS){
        nion_expected += Nion_General_MINI(redshift, M_min, M_max, M_turn_m, M_turn_a,
                                            alpha_star_mini, alpha_esc, norm_star_mini,
                                            norm_esc_mini, Mlim_Fstar_mini, Mlim_Fesc_mini,
                                            user_params_stoc->INTEGRATION_METHOD_MINI) * prefactor_nion_mini;

        sfr_expected_mini = Nion_General_MINI(redshift, M_min, M_max, M_turn_m, M_turn_a,
                                            alpha_star_mini, 0., norm_star_mini,
                                            1., Mlim_Fstar_mini, 0.,
                                            user_params_stoc->INTEGRATION_METHOD_MINI) * prefactor_sfr_mini;
    }

    // hm_expected *= prefactor_mass; //for non-CMF, the factors are already there
    wsfr_expected = nion_expected / t_star / t_h; //same integral, different prefactors, different in the stochastic grids due to scatter

    averages[0] = hm_expected;
    averages[1] = sfr_expected;
    averages[2] = sfr_expected_mini;
    averages[3] = nion_expected;
    averages[4] = wsfr_expected;

    return 0;
}

//This, for the moment, grids the PERTURBED halo catalogue.
//TODO: make a way to output both types by making 2 wrappers to this function that pass in arrays rather than structs
//NOTE: this function is quite slow to generate fixed halo boxes, however I don't mind since it's a debug case
//  If we want to make it faster just replace the integrals with the existing interpolation tables
//TODO: I should also probably completely separate the fixed and sampled grids into two functions which this calls
int ComputeHaloBox(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
                    struct FlagOptions * flag_options, struct InitialConditions *ini_boxes, struct PerturbedField * perturbed_field, struct PerturbHaloField *halos,
                    struct TsBox *previous_spin_temp, struct IonizedBox *previous_ionize_box, struct HaloBox *grids){
    int status;
    Try{

        int idx;
        //TODO: Check if this initialisation is necessary. aren't they already zero'd in Python?
#pragma omp parallel for num_threads(user_params->N_THREADS) private(idx)
        for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
            grids->halo_mass[idx] = 0.0;
            grids->n_ion[idx] = 0.0;
            grids->halo_sfr[idx] = 0.0;
            grids->whalo_sfr[idx] = 0.0;
            grids->count[idx] = 0;
        }
        grids->log10_Mcrit_LW_ave = log10(lyman_werner_threshold(redshift, 0., 0.,astro_params));

        LOG_DEBUG("Gridding %d halos...",halos->n_halos);

        //get parameters
        Broadcast_struct_global_UF(user_params,cosmo_params);
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);

        double alpha_esc = astro_params->ALPHA_ESC;
        double norm_esc = astro_params->F_ESC10;
        if(flag_options->PHOTON_CONS_ALPHA){
            norm_esc = get_alpha_fit(redshift);
        }
        double hm_avg=0,nion_avg=0,sfr_avg=0,wsfr_avg=0,sfr_avg_mini=0;

        double M_min = minimum_source_mass(redshift,astro_params,flag_options);
        double t_h = t_hubble(redshift);
        double cell_volume = VOLUME/HII_TOT_NUM_PIXELS;
        double M_cell = cosmo_params->OMm*RHOcrit*cell_volume;
        double M_turn_a_avg = 0, M_turn_m_avg = 0, M_turn_r_avg = 0.;
        double M_turn_a_avg_cell=0, M_turn_m_avg_cell=0, M_turn_r_avg_cell=0;

        double curr_vcb = flag_options->FIX_VCB_AVG ? global_params.VAVG : 0;

        //store initial values so we don't recompute
        double M_turn_a_store = flag_options->USE_MINI_HALOS ? atomic_cooling_threshold(redshift) : astro_params->M_TURN;
        double M_turn_m_store = lyman_werner_threshold(redshift, 0., curr_vcb, astro_params_stoc);
        double M_turn_a = M_turn_a_store;
        double M_turn_m = M_turn_m_store;
        double M_turn_r = 0.;

        double averages_box[7], averages_global[5], averages_subsampler[5];

        init_ps();
        if(user_params->USE_INTERPOLATION_TABLES){
            initialiseSigmaMInterpTable(M_min/2, global_params.M_MAX_INTEGRAL); //this needs to be initialised above MMax because of Nion_General
        }
        //do the mean HMF box
        //The default 21cmFAST has a strange behaviour where the nonlinear density is used as linear,
        //the condition mass is at mean density, but the total cell mass is multiplied by delta
        //This part mimics that behaviour
        //Since we need the average turnover masses before we can calculate the global means, we do the CMF integrals first
        //Then we calculate the expected UMF integrals before doing the adjustment
        if(flag_options->FIXED_HALO_GRIDS){
            set_fixed_grids(redshift, norm_esc, alpha_esc, M_min, M_cell, ini_boxes, perturbed_field, previous_spin_temp, previous_ionize_box, grids, averages_box);
            M_turn_a_avg = averages_box[5];
            M_turn_m_avg = averages_box[6];
            get_box_averages(redshift, norm_esc, alpha_esc, M_min, M_cell, M_turn_a_avg, M_turn_m_avg, averages_global);
            //This is the mean adjustment that happens in the rest of the code
            int i;

            //NOTE: in the default mode, global averages are fixed separately for minihalo and regular parts
#pragma omp parallel for num_threads(user_params->N_THREADS)
            for(i=0;i<HII_TOT_NUM_PIXELS;i++){
                grids->halo_mass[i] *= averages_global[0]/averages_box[0];
                grids->halo_sfr[i] *= averages_global[1]/averages_box[1];
                grids->halo_sfr_mini[i] *= averages_global[2]/averages_box[2];
                grids->n_ion[i] *= averages_global[3]/averages_box[3];
                grids->whalo_sfr[i] *= averages_global[4]/averages_box[4];
            }

            hm_avg = averages_global[0];
            sfr_avg = averages_global[1];
            sfr_avg_mini = averages_global[2];
            nion_avg = averages_global[3];
            wsfr_avg = averages_global[4];
        }
        else{
            //set below-resolution properties
            if(global_params.AVG_BELOW_SAMPLER && M_min < global_params.SAMPLER_MIN_MASS){
                set_fixed_grids(redshift, norm_esc, alpha_esc, M_min, global_params.SAMPLER_MIN_MASS, ini_boxes, perturbed_field, previous_spin_temp, previous_ionize_box, grids, averages_box);
                //TODO: This is pretty redundant, but since the fixed grids have density units (X Mpc-3) I have to re-multiply before adding the halos.
                //      I should instead have a flag to output the summed values in cell. (2*N_pixel > N_halo so generally i don't want to do it in the halo loop)
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] *= cell_volume;
                    grids->n_ion[idx] *= cell_volume;
                    grids->halo_sfr[idx] *= cell_volume;
                    grids->halo_sfr_mini[idx] *= cell_volume;
                    grids->whalo_sfr[idx] *= cell_volume;
                }
            }
#pragma omp parallel num_threads(user_params->N_THREADS) firstprivate(M_turn_a,M_turn_m,M_turn_r,curr_vcb,idx)
            {
                int i_halo,x,y,z;
                double m,nion,sfr,wsfr,sfr_mini,stars_mini,stars;

                float in_props[2];
                float out_props[6];

#pragma omp for reduction(+:hm_avg,nion_avg,sfr_avg,wsfr_avg,M_turn_a_avg,M_turn_m_avg,M_turn_r_avg)
                for(i_halo=0; i_halo<halos->n_halos; i_halo++){
                    x = halos->halo_coords[0+3*i_halo]; //NOTE:PerturbedHaloField is on HII_DIM, HaloField is on DIM
                    y = halos->halo_coords[1+3*i_halo];
                    z = halos->halo_coords[2+3*i_halo];
                    //TODO: figure out if its faster to do these calculations n_halo times OR search a grid cell for halos and do them n_cell times
                    if(!flag_options_stoc->FIX_VCB_AVG && user_params->USE_RELATIVE_VELOCITIES){
                        curr_vcb = ini_boxes->lowres_vcb[HII_R_INDEX(x,y,z)];
                    }

                    //set values before reionisation feedback
                    M_turn_a = M_turn_a_store;
                    if(flag_options->USE_MINI_HALOS){
                        M_turn_m = lyman_werner_threshold(redshift, previous_spin_temp->J_21_LW_box[HII_R_INDEX(x,y,z)], curr_vcb, astro_params);
                        M_turn_r = reionization_feedback(redshift, previous_ionize_box->Gamma12_box[HII_R_INDEX(x, y, z)], previous_ionize_box->z_re_box[HII_R_INDEX(x, y, z)]);
                    }

                    //if reion feedback is higher, replace the values
                    if(M_turn_r > M_turn_a) M_turn_a = M_turn_r;
                    if(M_turn_r > M_turn_m) M_turn_m = M_turn_r;

                    m = halos->halo_masses[i_halo];

                    //these are the halo property RNG sequences
                    in_props[0] = halos->star_rng[i_halo];
                    in_props[1] = halos->sfr_rng[i_halo];

                    set_halo_properties(m,M_turn_a,M_turn_m,t_h,norm_esc,alpha_esc,in_props,out_props);

                    sfr = out_props[0];
                    sfr_mini = out_props[1];
                    nion = out_props[2];
                    wsfr = out_props[3];
                    stars = out_props[4];
                    stars_mini = out_props[5];

                    if(x+y+z == 0){
                        LOG_SUPER_DEBUG("Cell 0 Halo %d: HM: %.2e SM: %.2e (%.2e) NI: %.2e SF: %.2e (%.2e) WS: %.2e",i_halo,m,stars,stars_mini,nion,sfr,sfr_mini,wsfr);
                    }

                    //feed back the calculated properties to PerturbHaloField
                    //TODO: move set_halo_properties to PertburbHaloField and move it forward in time
                    //  This will require EITHER separating mini and regular halo components OR the ternary halo model (inactive,moleculer,atomic)
                    //  OR directly storing all the grid components

                    //TODO: is it possible to apply some sort of array reduction here with OpenMP instead of atomics?
#pragma omp atomic update
                    grids->halo_mass[HII_R_INDEX(x, y, z)] += m;
#pragma omp atomic update
                    grids->halo_stars[HII_R_INDEX(x, y, z)] += stars;
#pragma omp atomic update
                    grids->halo_stars_mini[HII_R_INDEX(x, y, z)] += stars_mini;
#pragma omp atomic update
                    grids->n_ion[HII_R_INDEX(x, y, z)] += nion;
#pragma omp atomic update
                    grids->halo_sfr[HII_R_INDEX(x, y, z)] += sfr;
#pragma omp atomic update
                    grids->halo_sfr_mini[HII_R_INDEX(x, y, z)] += sfr_mini;
#pragma omp atomic update
                    grids->whalo_sfr[HII_R_INDEX(x, y, z)] += wsfr;
                    //It can be convenient to remove halos from a catalogue by setting them to zero, don't count those here
                    if(m>0){
#pragma omp atomic update
                        grids->count[HII_R_INDEX(x, y, z)] += 1;
                    }

                    M_turn_m_avg += M_turn_m;
                    if(LOG_LEVEL >= DEBUG_LEVEL){
                        hm_avg += m;
                        sfr_avg += sfr;
                        sfr_avg_mini += sfr_mini;
                        wsfr_avg += wsfr;
                        nion_avg += nion;
                        M_turn_a_avg += M_turn_a;
                        M_turn_r_avg += M_turn_r;
                    }
                }
#pragma omp for reduction(+:M_turn_a_avg_cell,M_turn_m_avg_cell,M_turn_r_avg_cell)
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] /= cell_volume;
                    grids->n_ion[idx] /= cell_volume;
                    grids->halo_sfr[idx] /= cell_volume;
                    grids->halo_sfr_mini[idx] /= cell_volume;
                    grids->whalo_sfr[idx] /= cell_volume;

                    //cell averages for debug
                    if(LOG_LEVEL >= DEBUG_LEVEL && flag_options_stoc->USE_MINI_HALOS){
                        M_turn_r = reionization_feedback(redshift, previous_ionize_box->Gamma12_box[idx], previous_ionize_box->z_re_box[idx]);
                        M_turn_a = atomic_cooling_threshold(redshift);
                        M_turn_m = lyman_werner_threshold(redshift, previous_spin_temp->J_21_LW_box[idx], ini_boxes->lowres_vcb[idx], astro_params);
                        M_turn_a_avg_cell += M_turn_a > M_turn_r ? M_turn_a : M_turn_r;
                        M_turn_m_avg_cell += M_turn_m > M_turn_r ? M_turn_m : M_turn_r;
                        M_turn_r_avg_cell += M_turn_r;
                    }
                }
            }
            LOG_SUPER_DEBUG("Cell 0: HM: %.2e SM: %.2e (%.2e) NI: %.2e SF: %.2e (%.2e) WS: %.2e ct : %d",grids->halo_mass[HII_R_INDEX(0,0,0)],
                                grids->halo_stars[HII_R_INDEX(0,0,0)],grids->halo_stars_mini[HII_R_INDEX(0,0,0)],
                                grids->n_ion[HII_R_INDEX(0,0,0)],grids->halo_sfr[HII_R_INDEX(0,0,0)],grids->halo_sfr_mini[HII_R_INDEX(0,0,0)],
                                grids->whalo_sfr[HII_R_INDEX(0,0,0)],grids->count[HII_R_INDEX(0,0,0)]);
            M_turn_m_avg /= halos->n_halos;
            grids->log10_Mcrit_LW_ave = log10(M_turn_m_avg);
            if(LOG_LEVEL >= DEBUG_LEVEL){
                hm_avg /= VOLUME;
                sfr_avg /= VOLUME;
                sfr_avg_mini /= VOLUME;
                wsfr_avg /= VOLUME;
                nion_avg /= VOLUME;

                //NOTE: There is an inconsistency here, the sampled grids use a halo-averaged turnover mass
                //  whereas the fixed grids / default 21cmfast uses the volume averaged LOG10(turnover mass).
                //  Neither of these are a perfect representation due to the nonlinear way turnover mass affects N_ion
                M_turn_a_avg /= halos->n_halos;
                M_turn_a_avg_cell /= HII_TOT_NUM_PIXELS;
                M_turn_m_avg_cell /= HII_TOT_NUM_PIXELS;
                M_turn_r_avg /= halos->n_halos;
                M_turn_r_avg_cell /= HII_TOT_NUM_PIXELS;

                //If we have no halos, assume the turnover has no reion feedback & no LW for the debug printing
                if(halos->n_halos == 0){
                    M_turn_a_avg = M_turn_a_store;
                    M_turn_m_avg = M_turn_m_store;
                    M_turn_r_avg = 0.;
                }

                get_box_averages(redshift, norm_esc, alpha_esc, global_params.SAMPLER_MIN_MASS, global_params.M_MAX_INTEGRAL, M_turn_a_avg, M_turn_m_avg, averages_global);
                get_box_averages(redshift, norm_esc, alpha_esc, M_min, global_params.SAMPLER_MIN_MASS, M_turn_a_avg, M_turn_m_avg, averages_subsampler);
            }
        }

        if(user_params->USE_INTERPOLATION_TABLES && (flag_options->FIXED_HALO_GRIDS || LOG_LEVEL >= DEBUG_LEVEL)){
                freeSigmaMInterpTable();
        }

        LOG_DEBUG("HALO BOXES REDSHIFT %.2f",redshift);
        LOG_DEBUG("Exp. averages: (HM %11.3e, NION %11.3e, SFR %11.3e, SFR_MINI %11.3e, WSFR %11.3e)",averages_global[0],averages_global[3],averages_global[1],
                                                                                                    averages_global[2],averages_global[4]);
        LOG_DEBUG("Box  averages: (HM %11.3e, NION %11.3e, SFR %11.3e, SFR_MINI %11.3e, WSFR %11.3e)",hm_avg,nion_avg,sfr_avg,sfr_avg_mini,wsfr_avg);
        LOG_DEBUG("Ratio:         (HM %11.3e, NION %11.3e, SFR %11.3e, SFR_MINI %11.3e, WSFR %11.3e)",hm_avg/averages_global[0],nion_avg/averages_global[3],
                                                                                                    sfr_avg/averages_global[1],sfr_avg_mini/averages_global[2],
                                                                                                    wsfr_avg/averages_global[4]);
        if(global_params.AVG_BELOW_SAMPLER && M_min < global_params.SAMPLER_MIN_MASS){
            LOG_DEBUG("SUB-SAMPLER",redshift);
            LOG_DEBUG("Exp. averages: (HM %11.3e, NION %11.3e, SFR %11.3e, SFR_MINI %11.3e, WSFR %11.3e)",averages_subsampler[0],averages_subsampler[3],averages_subsampler[1],
                                                                                                        averages_subsampler[2],averages_subsampler[4]);
            LOG_DEBUG("Box  averages: (HM %11.3e, NION %11.3e, SFR %11.3e, SFR_MINI %11.3e, WSFR %11.3e)",averages_box[0],averages_box[3],
                                                                                                        averages_box[1],averages_box[2],averages_box[4]);
            LOG_DEBUG("Ratio:         (HM %11.3e, NION %11.3e, SFR %11.3e, SFR_MINI %11.3e, WSFR %11.3e)",averages_box[0]/averages_subsampler[0],
                                                                                                        averages_box[3]/averages_subsampler[3],
                                                                                                        averages_box[1]/averages_subsampler[1],
                                                                                                        averages_box[2]/averages_subsampler[2],
                                                                                                        averages_box[4]/averages_subsampler[4]);
        }
        LOG_DEBUG("Turnovers: ACG %11.3e global %11.3e Cell %11.3e",M_turn_a_avg,atomic_cooling_threshold(redshift),M_turn_a_avg_cell);
        LOG_DEBUG("MCG  %11.3e Cell %11.3e",M_turn_m_avg,M_turn_m_avg_cell);
        LOG_DEBUG("Reion %11.3e Cell %11.3e",M_turn_r_avg,M_turn_r_avg_cell);
    }
    Catch(status){
        return(status);
    }
    LOG_DEBUG("Done.");
    return(0);
}
