/* This file contains fucntions for calculating the HaloBox output for 21cmfast, containing the gridded
 * source properties, either from integrating the conditional mass functions in a cell or from the halo sampler */

//Parameters for the halo box calculations
//  These are just the values which are calculated at the beginning of ComputeHaloBox and don't change
//  using this reduces the use of the global parameter structs and allows fewer exp/log unit changes
struct HaloBoxConstants{
    double redshift;

    double fstar_10;
    double alpha_star;
    double sigma_star;

    double alpha_upper;
    double pivot_upper;
    double upper_pivot_ratio;

    double fstar_7;
    double alpha_star_mini;

    double t_h;
    double t_star;
    double sigma_sfr_lim;
    double sigma_sfr_idx;

    double l_x;
    double l_x_mini;
    double sigma_xray;

    double fesc_10;
    double alpha_esc;
    double fesc_7;

    double vcb_norel;
    double mturn_a_nofb;
    double mturn_m_nofb;

    double Mlim_Fstar;
    double Mlim_Fesc;
    double Mlim_Fstar_mini;
    double Mlim_Fesc_mini;
};

//struct holding each halo property we currently need.
//This is only used for both averages over the box/catalogues
//  as well as an individual halo's properties
struct HaloProperties{
    double halo_mass;
    double stellar_mass;
    double halo_sfr;
    double stellar_mass_mini;
    double sfr_mini;
    double fescweighted_sfr;
    double n_ion;
    double halo_xray;
    double metallicity;
    double m_turn_acg;
    double m_turn_mcg;
    double m_turn_reion;
};

void set_hbox_constants(double redshift, struct AstroParams *astro_params, struct FlagOptions *flag_options, struct HaloBoxConstants *consts){
    consts->redshift = redshift;

    consts->fstar_10 = astro_params->F_STAR10;
    consts->alpha_star = astro_params->ALPHA_STAR;
    consts->sigma_star = astro_params->SIGMA_STAR;

    consts->alpha_upper = astro_params->UPPER_STELLAR_TURNOVER_INDEX;
    consts->pivot_upper = astro_params->UPPER_STELLAR_TURNOVER_MASS;
    consts->upper_pivot_ratio = pow(astro_params->UPPER_STELLAR_TURNOVER_MASS/1e10,astro_params->ALPHA_STAR);

    consts->fstar_7 = astro_params->F_STAR7_MINI;
    consts->alpha_star_mini = astro_params->ALPHA_STAR_MINI;

    consts->t_h = t_hubble(redshift);
    consts->t_star = astro_params->t_STAR;
    consts->sigma_sfr_lim = astro_params->SIGMA_SFR_LIM;
    consts->sigma_sfr_idx = astro_params->SIGMA_SFR_INDEX;

    consts->l_x = astro_params->L_X * 1e-40; //setting units to 1e40 erg s -1 so we can store in float
    consts->l_x_mini = astro_params->L_X_MINI * 1e-40;
    consts->sigma_xray = astro_params->SIGMA_LX;

    consts->alpha_esc = astro_params->ALPHA_ESC;
    consts->fesc_10= astro_params->F_ESC10;
    consts->fesc_7 = astro_params->F_ESC7_MINI;

    if(flag_options->PHOTON_CONS_TYPE == 2)
        consts->alpha_esc = get_fesc_fit(redshift);
    else if(flag_options->PHOTON_CONS_TYPE == 3)
        consts->fesc_10 = get_fesc_fit(redshift);

    consts->mturn_a_nofb = flag_options->USE_MINI_HALOS ? atomic_cooling_threshold(redshift) : astro_params->M_TURN;

    consts->mturn_m_nofb = 0.;
    if(flag_options->USE_MINI_HALOS){
        consts->mturn_m_nofb = lyman_werner_threshold(redshift, 0., consts->vcb_norel, astro_params);
        consts->vcb_norel = flag_options->FIX_VCB_AVG ? global_params.VAVG : 0;
    }

    if(flag_options->FIXED_HALO_GRIDS || user_params_stoc->AVG_BELOW_SAMPLER){
        consts->Mlim_Fstar = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, consts->alpha_star, consts->fstar_10);
        consts->Mlim_Fesc = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, consts->alpha_esc, consts->fesc_10);

        if(flag_options->USE_MINI_HALOS){
            consts->Mlim_Fstar_mini = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, consts->alpha_star_mini,
                                                            consts->fstar_7 * pow(1e3,consts->alpha_star_mini));
            consts->Mlim_Fesc_mini = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, consts->alpha_esc,
                                                            consts->fesc_7 * pow(1e3,consts->alpha_esc));
        }
    }
}

void get_halo_stellarmass(double halo_mass, double mturn_acg, double mturn_mcg, double star_rng,
                             struct HaloBoxConstants *consts, double *star_acg, double *star_mcg){
    //low-mass ACG power-law parameters
    double f_10 = consts->fstar_10;
    double f_a = consts->alpha_star;
    double sigma_star = consts->sigma_star;

    //high-mass ACG power-law parameters
    double fu_a = consts->alpha_upper;
    double fu_p = consts->pivot_upper;

    //MCG parameters
    double f_7 = consts->fstar_7;
    double f_a_mini = consts->alpha_star_mini;

    //intermediates
    double fstar_mean;
    double f_sample, f_sample_mini;
    double sm_sample,sm_sample_mini;

    double baryon_ratio = cosmo_params_stoc->OMb / cosmo_params_stoc->OMm;

    //We don't want an upturn even with a negative ALPHA_STAR
    if(flag_options_stoc->USE_UPPER_STELLAR_TURNOVER && (f_a > fu_a)){
        fstar_mean = consts->upper_pivot_ratio / (pow(halo_mass/fu_p,-f_a)+pow(halo_mass/fu_p,-fu_a));
    }
    else{
        fstar_mean = pow(halo_mass/1e10,f_a); //PL term
    }
    f_sample = f_10 * fstar_mean * exp(-mturn_acg/halo_mass + star_rng*sigma_star); //1e10 normalisation of stellar mass
    if(f_sample > 1.) f_sample = 1.;

    sm_sample = f_sample * halo_mass * baryon_ratio;
    *star_acg = sm_sample;

    if(!flag_options_stoc->USE_MINI_HALOS){
        *star_mcg = 0.;
        return;
    }

    f_sample_mini = pow(halo_mass/1e7,f_a_mini) * f_7 * exp(-mturn_mcg/halo_mass - halo_mass/mturn_acg + star_rng);
    if(f_sample_mini > 1.) f_sample_mini = 1.;

    sm_sample_mini = f_sample_mini * halo_mass * baryon_ratio;
    *star_mcg = sm_sample_mini;
}

void get_halo_sfr(double stellar_mass, double stellar_mass_mini, double sfr_rng,
                     struct HaloBoxConstants *consts, double *sfr, double *sfr_mini){
    double sfr_mean, sfr_mean_mini;
    double sfr_sample, sfr_sample_mini;

    double sigma_sfr_lim = consts->sigma_sfr_lim;
    double sigma_sfr_idx = consts->sigma_sfr_idx;

    //set the scatter based on the total Stellar mass
    //We use the total stellar mass (MCG + ACG) NOTE: it might be better to separate later
    double sigma_sfr=0.;
    if(sigma_sfr_lim > 0.){
        sigma_sfr = sigma_sfr_idx * log10((stellar_mass+stellar_mass_mini)/1e10) + sigma_sfr_lim;
        if(sigma_sfr < sigma_sfr_lim) sigma_sfr = sigma_sfr_lim;
    }
    sfr_mean = stellar_mass / (consts->t_star * consts->t_h);

    sfr_sample = sfr_mean * exp(sfr_rng*sigma_sfr);
    *sfr = sfr_sample;

    if(!flag_options_stoc->USE_MINI_HALOS){
        *sfr_mini = 0.;
        return;
    }

    sfr_mean_mini = stellar_mass_mini / (consts->t_star * consts->t_h);
    sfr_sample_mini = sfr_mean_mini * exp(sfr_rng*sigma_sfr);
    *sfr_mini = sfr_sample_mini;
}

void get_halo_metallicity(double sfr, double stellar, double redshift, double *z_out){
    //Hardcoded for now: 6 extra fit parameters in the equation
    double z_denom, z_result;
    z_denom = (1.28825e10 * pow(sfr*SperYR,0.56));
    z_result = 0.296 * pow(1 + pow(stellar/z_denom,-2.1),-0.148) * pow(10,-0.056*redshift + 0.064);

    *z_out = z_result;
}

void get_halo_xray(double sfr, double sfr_mini, double metallicity, double xray_rng, struct HaloBoxConstants *consts, double *xray_out){
    //Hardcoded for now (except the lx normalisation and the scatter): 3 extra fit parameters in the equation
    double xray, xray_mini;
    double l10z = log10(metallicity);
    xray = (-0.11*l10z + 1.30)*log10(sfr*SperYR) - 0.31*l10z;
    xray = exp(LN10*xray + xray_rng*consts->sigma_xray);

    xray_mini = 0.;
    if(flag_options_stoc->USE_MINI_HALOS){
        xray_mini = (-0.11*l10z + 1.30)*log10(sfr_mini*SperYR) - 0.31*l10z;
        xray_mini = exp(LN10*xray_mini + xray_rng*consts->sigma_xray);
    }

    *xray_out = xray*consts->l_x + xray_mini*consts->l_x_mini;
}

//calculates halo properties from astro parameters plus the correlated rng
//The inputs include all properties with a separate RNG
//The outputs include all sampled halo properties PLUS all properties which cannot be recovered when mixing all the halos together
//  i.e escape fraction weighting, minihalo stuff that has separate parameters
//Since there are so many spectral terms in the spin temperature calculation, it will be most efficient to split SFR into regular and minihalos
//  BUT not split the ionisedbox fields.
//in order to remain consistent with the minihalo treatment in default (Nion_a * exp(-M/M_a) + Nion_m * exp(-M/M_m - M_a/M))
//  we treat the minihalos as a shift in the mean, where each halo will have both components, representing a smooth
//  transition in halo mass from one set of SFR/emmissivity parameters to the other.
void set_halo_properties(double halo_mass, double M_turn_a, double M_turn_m, double M_turn_r,
                         struct HaloBoxConstants *consts, double *input_rng, struct HaloProperties *output){
    double n_ion_sample, wsfr_sample;
    double fesc;
    double fesc_mini=0.;

    //do reionization feedback
    if(flag_options_stoc->USE_MINI_HALOS){
        if(M_turn_a < M_turn_r)M_turn_a = M_turn_r;
        if(M_turn_m < M_turn_r)M_turn_m = M_turn_r;
    }

    double stellar_mass, stellar_mass_mini;
    get_halo_stellarmass(halo_mass,M_turn_a,M_turn_m,input_rng[0],consts,&stellar_mass,&stellar_mass_mini);

    double sfr,sfr_mini;
    get_halo_sfr(stellar_mass,stellar_mass_mini,input_rng[1],consts,&sfr,&sfr_mini);

    double metallicity=0;
    double xray_lum=0;
    if(flag_options_stoc->USE_TS_FLUCT){
        get_halo_metallicity(sfr+sfr_mini,stellar_mass+stellar_mass_mini,consts->redshift,&metallicity);
        get_halo_xray(sfr,sfr_mini,metallicity,input_rng[2],consts,&xray_lum);
    }

    //no rng for escape fraction yet
    fesc = fmin(consts->fesc_10*pow(halo_mass/1e10,consts->alpha_esc),1);
    if(flag_options_stoc->USE_MINI_HALOS)
        fesc_mini = fmin(consts->fesc_7*pow(halo_mass/1e7,consts->alpha_esc),1);

    n_ion_sample = stellar_mass*global_params.Pop2_ion*fesc + stellar_mass_mini*global_params.Pop3_ion*fesc_mini;
    wsfr_sample = sfr*global_params.Pop2_ion*fesc + sfr_mini*global_params.Pop3_ion*fesc_mini;

    output->halo_mass = halo_mass;
    output->stellar_mass = stellar_mass;
    output->stellar_mass_mini = stellar_mass_mini;
    output->halo_sfr = sfr;
    output->sfr_mini = sfr_mini;
    output->fescweighted_sfr = wsfr_sample;
    output->n_ion = n_ion_sample;
    output->metallicity = metallicity;
    output->halo_xray = xray_lum;

    output->m_turn_acg = M_turn_a;
    output->m_turn_mcg = M_turn_m;
    output->m_turn_reion = M_turn_r;
}

//Fixed halo grids, where each property is set as the integral of the CMF on the EULERIAN cell scale
//As per default 21cmfast (strange pretending that the lagrangian density is eulerian and then *(1+delta))
//This outputs the UN-NORMALISED grids (before mean-adjustment)
int set_fixed_grids(double M_min, double M_max, struct InitialConditions *ini_boxes,
                    struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp,
                    struct IonizedBox *previous_ionize_box, struct HaloBoxConstants *consts, struct HaloBox *grids, struct HaloProperties *averages){
    double M_cell = RHOcrit * cosmo_params_stoc->OMm * VOLUME / HII_TOT_NUM_PIXELS; //mass in cell of mean dens
    double growth_z = dicke(consts->redshift);

    double lnMmin = log(M_min);
    double lnMcell = log(M_cell);
    double lnMmax = log(M_max);

    double sigma_cell = EvaluateSigma(lnMcell);

    double prefactor_mass = RHOcrit * cosmo_params_stoc->OMm;
    double prefactor_stars = RHOcrit * cosmo_params_stoc->OMb * consts->fstar_10;
    double prefactor_stars_mini = RHOcrit * cosmo_params_stoc->OMb * consts->fstar_7;
    double prefactor_sfr = prefactor_stars / consts->t_star / consts->t_h;
    double prefactor_sfr_mini = prefactor_stars_mini / consts->t_star / consts->t_h;
    double prefactor_nion = prefactor_stars * consts->fesc_10 * global_params.Pop2_ion;
    double prefactor_nion_mini = prefactor_stars_mini * consts->fesc_7 * global_params.Pop3_ion;
    double prefactor_wsfr = prefactor_sfr * consts->fesc_10;
    double prefactor_wsfr_mini = prefactor_sfr_mini * consts->fesc_7;
    double prefactor_xray = prefactor_sfr * SperYR * consts->l_x;
    double prefactor_xray_mini = prefactor_sfr_mini * consts->l_x_mini * SperYR;

    double hm_avg=0, nion_avg=0, sfr_avg=0, sfr_avg_mini=0, wsfr_avg=0, xray_avg=0;
    double Mlim_a_avg=0, Mlim_m_avg=0, Mlim_r_avg=0, sm_avg=0, sm_avg_mini=0;

    //find density grid limits
    double min_density = 0.;
    double max_density = 0.;
    #pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        int i;
        double dens;
        #pragma omp for reduction(min:min_density) reduction(max:max_density)
        for(i=0;i<HII_TOT_NUM_PIXELS;i++){
            dens = perturbed_field->density[i];
            if(dens > max_density) max_density = dens;
            if(dens < min_density) min_density = dens;
        }
    }

    struct parameters_gsl_MF_integrals params = {
            .redshift = consts->redshift,
            .growthf = growth_z,
            .sigma_cond = sigma_cell,
            .HMF = user_params_stoc->HMF,
    };

    double M_turn_a_nofb = consts->mturn_a_nofb;

    LOG_DEBUG("Mean halo boxes || M = [%.2e %.2e] | Mcell = %.2e (s=%.2e) | z = %.2e | D = %.2e",M_min,M_max,M_cell,sigma_cell,
                consts->redshift,growth_z);

    //These tables are coarser than needed, an initial loop for Mturn to find limits may help
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        if(user_params_stoc->INTEGRATION_METHOD_ATOMIC == 1 || user_params_stoc->INTEGRATION_METHOD_MINI == 1){
            initialise_GL(NGL_INT, lnMmin, lnMmax);
        }

        //This table assumes no reionisation feedback
        initialise_SFRD_Conditional_table(min_density,max_density,growth_z,consts->mturn_a_nofb,M_min,M_max,M_cell,
                                                consts->alpha_star, consts->alpha_star_mini, consts->fstar_10,
                                                consts->fstar_7, user_params_stoc->INTEGRATION_METHOD_ATOMIC,
                                                user_params_stoc->INTEGRATION_METHOD_MINI,
                                                flag_options_stoc->USE_MINI_HALOS);

        //This table includes reionisation feedback, but takes the atomic turnover anyway for the upper turnover
        initialise_Nion_Conditional_spline(consts->redshift,consts->mturn_a_nofb,min_density,max_density,M_min,M_max,M_cell,
                                LOG10_MTURN_MIN,LOG10_MTURN_MAX,LOG10_MTURN_MIN,LOG10_MTURN_MAX,
                                consts->alpha_star, consts->alpha_star_mini,
                                consts->alpha_esc, consts->fstar_10,
                                consts->fesc_10,consts->Mlim_Fstar,consts->Mlim_Fesc,consts->fstar_7,
                                consts->fesc_7,consts->Mlim_Fstar_mini, consts->Mlim_Fesc_mini,
                                user_params_stoc->INTEGRATION_METHOD_ATOMIC,
                                user_params_stoc->INTEGRATION_METHOD_MINI,
                                flag_options_stoc->USE_MINI_HALOS, false);

        initialise_dNdM_tables(min_density, max_density, lnMmin, lnMmax, growth_z, lnMcell, false);
    }

#pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        int i;
        double dens;
        double J21_val, Gamma12_val, zre_val;
        double curr_vcb = consts->vcb_norel;
        double M_turn_m = consts->mturn_m_nofb;
        double M_turn_a = consts->mturn_a_nofb;
        double M_turn_r;
        double mass_intgrl, h_count;
        double intgrl_fesc_weighted, intgrl_stars_only;
        double intgrl_fesc_weighted_mini=0., intgrl_stars_only_mini=0.;

#pragma omp for reduction(+:hm_avg,sm_avg,sfr_avg,xray_avg,nion_avg,wsfr_avg,sm_avg_mini,Mlim_a_avg,Mlim_m_avg,Mlim_r_avg)
        for(i=0;i<HII_TOT_NUM_PIXELS;i++){
            dens = perturbed_field->density[i];
            params.delta = dens;

            if(flag_options_stoc->USE_MINI_HALOS){
                if(!flag_options_stoc->FIX_VCB_AVG && user_params_stoc->USE_RELATIVE_VELOCITIES){
                    curr_vcb = ini_boxes->lowres_vcb[i];
                }
                J21_val=Gamma12_val=zre_val=0.;
                if(consts->redshift < global_params.Z_HEAT_MAX){
                    J21_val = previous_spin_temp->J_21_LW_box[i];
                    Gamma12_val = previous_ionize_box->Gamma12_box[i];
                    zre_val = previous_ionize_box->z_re_box[i];
                }
                M_turn_a = M_turn_a_nofb;
                M_turn_m = lyman_werner_threshold(consts->redshift, J21_val, curr_vcb, astro_params_stoc);
                M_turn_r = reionization_feedback(consts->redshift, Gamma12_val, zre_val);
                if(M_turn_a < M_turn_r)M_turn_a = M_turn_r;
                if(M_turn_m < M_turn_r)M_turn_m = M_turn_r;
            }

            h_count = EvaluateNhalo(dens, growth_z, lnMmin, lnMmax, M_cell, sigma_cell, dens);
            mass_intgrl = EvaluateMcoll(dens, growth_z, lnMmin, lnMmax, M_cell, sigma_cell, dens);
            intgrl_fesc_weighted = EvaluateNion_Conditional(dens,log10(M_turn_a),growth_z,M_min,M_max,M_cell,sigma_cell,
                                            consts->Mlim_Fstar,consts->Mlim_Fesc,false);
            intgrl_stars_only = EvaluateSFRD_Conditional(dens,growth_z,M_min,M_max,M_cell,sigma_cell,
                                            log10(M_turn_a),consts->Mlim_Fstar);
            if(flag_options_stoc->USE_MINI_HALOS){
                intgrl_fesc_weighted_mini = EvaluateSFRD_Conditional_MINI(dens,log10(M_turn_m),growth_z,M_min,M_max,M_cell,sigma_cell,
                                                            log10(M_turn_a),consts->Mlim_Fstar);
                intgrl_stars_only_mini = EvaluateNion_Conditional_MINI(dens,log10(M_turn_m),growth_z,M_min,M_max,M_cell,sigma_cell,
                                                            log10(M_turn_a),consts->Mlim_Fstar,consts->Mlim_Fesc,false);
            }

            grids->count[i] = (int)(h_count * M_cell * (1+dens)); //NOTE: truncated
            grids->halo_mass[i] = mass_intgrl * prefactor_mass * (1+dens);
            grids->halo_sfr[i] = (intgrl_stars_only*prefactor_sfr) * (1+dens);
            grids->halo_sfr_mini[i] = intgrl_stars_only_mini*prefactor_sfr_mini * (1+dens);
            grids->n_ion[i] = (intgrl_fesc_weighted*prefactor_nion + intgrl_fesc_weighted_mini*prefactor_nion_mini) * (1+dens);
            grids->whalo_sfr[i] = (intgrl_fesc_weighted*prefactor_wsfr + intgrl_fesc_weighted_mini*prefactor_wsfr_mini) * (1+dens);
            grids->halo_xray[i] = (intgrl_stars_only*prefactor_xray + intgrl_stars_only_mini*prefactor_xray_mini) * (1+dens);
            grids->halo_stars[i] = intgrl_stars_only*prefactor_stars * (1+dens);
            grids->halo_stars_mini[i] = intgrl_stars_only_mini*prefactor_stars_mini * (1+dens);

            hm_avg += grids->halo_mass[i];
            nion_avg += grids->n_ion[i];
            sfr_avg += grids->halo_sfr[i];
            wsfr_avg += grids->whalo_sfr[i];
            sfr_avg_mini += grids->halo_sfr_mini[i];
            xray_avg += grids->halo_xray[i];
            sm_avg += grids->halo_stars[i];
            sm_avg_mini += grids->halo_stars_mini[i];
            Mlim_a_avg += M_turn_a;
            Mlim_m_avg += M_turn_m;
            Mlim_r_avg += M_turn_r;
        }
    }

    free_conditional_tables();

    hm_avg /= HII_TOT_NUM_PIXELS;
    sm_avg /= HII_TOT_NUM_PIXELS;
    sm_avg_mini /= HII_TOT_NUM_PIXELS;
    nion_avg /= HII_TOT_NUM_PIXELS;
    sfr_avg /= HII_TOT_NUM_PIXELS;
    sfr_avg_mini /= HII_TOT_NUM_PIXELS;
    wsfr_avg /= HII_TOT_NUM_PIXELS;
    xray_avg /= HII_TOT_NUM_PIXELS;
    Mlim_a_avg /= HII_TOT_NUM_PIXELS;
    Mlim_m_avg /= HII_TOT_NUM_PIXELS;
    Mlim_r_avg /= HII_TOT_NUM_PIXELS;

    averages->halo_mass = hm_avg;
    averages->stellar_mass = sm_avg;
    averages->stellar_mass_mini = sm_avg_mini;
    averages->halo_sfr = sfr_avg;
    averages->sfr_mini = sfr_avg_mini;
    averages->n_ion = nion_avg;
    averages->halo_xray = xray_avg;
    averages->fescweighted_sfr = wsfr_avg;
    averages->m_turn_acg = Mlim_a_avg;
    averages->m_turn_mcg = Mlim_m_avg;
    averages->m_turn_reion = Mlim_r_avg;

    return 0;
}

//Expected global averages for box quantities for mean adjustment
//WARNING: THESE AVERAGE BOXES ARE WRONG, CHECK THEM
int get_box_averages(double M_min, double M_max, double M_turn_a, double M_turn_m, struct HaloBoxConstants *consts, struct HaloProperties *averages){
    LOG_SUPER_DEBUG("Getting Box averages z=%.2f M [%.2e %.2e] Mt [%.2e %.2e]",consts->redshift,M_min,M_max,M_turn_a,M_turn_m);
    double t_h = consts->t_h;
    double lnMmax = log(M_max);
    double lnMmin = log(M_min);

    double prefactor_mass = RHOcrit * cosmo_params_stoc->OMm;
    double prefactor_stars = RHOcrit * cosmo_params_stoc->OMb * consts->fstar_10;
    double prefactor_stars_mini = RHOcrit * cosmo_params_stoc->OMb * consts->fstar_7;
    double prefactor_sfr = prefactor_stars / consts->t_star / t_h;
    double prefactor_sfr_mini = prefactor_stars_mini / consts->t_star / t_h;
    double prefactor_nion = prefactor_stars * consts->fesc_10 * global_params.Pop2_ion;
    double prefactor_nion_mini = prefactor_stars_mini * consts->fesc_7 * global_params.Pop3_ion;
    double prefactor_wsfr = prefactor_sfr * consts->fesc_10;
    double prefactor_wsfr_mini = prefactor_sfr_mini * consts->fesc_7;
    double prefactor_xray = prefactor_sfr * SperYR * consts->l_x;
    double prefactor_xray_mini = prefactor_sfr_mini * consts->l_x_mini * SperYR;

    double mass_intgrl, h_count;
    double intgrl_fesc_weighted, intgrl_stars_only;
    double intgrl_fesc_weighted_mini=0., intgrl_stars_only_mini=0.;

    if(user_params_stoc->INTEGRATION_METHOD_ATOMIC == 1 || user_params_stoc->INTEGRATION_METHOD_MINI == 1)
        initialise_GL(NGL_INT, lnMmin, lnMmax);

    //NOTE: we use the atomic method for all halo mass/count here
    mass_intgrl = Fcoll_General(consts->redshift,lnMmin,lnMmax);
    intgrl_fesc_weighted = Nion_General(consts->redshift, lnMmin, lnMmax, M_turn_a, consts->alpha_star, consts->alpha_esc, consts->fstar_10,
                                 consts->fesc_10, consts->Mlim_Fstar, consts->Mlim_Fesc);
    intgrl_stars_only = Nion_General(consts->redshift, lnMmin, lnMmax, M_turn_a, consts->alpha_star, 0., consts->fstar_10, 1.,
                                 consts->Mlim_Fstar, 0.);
    if(flag_options_stoc->USE_MINI_HALOS){
        intgrl_fesc_weighted_mini = Nion_General_MINI(consts->redshift, lnMmin, lnMmax, M_turn_m, M_turn_a,
                                            consts->alpha_star_mini, consts->alpha_esc, consts->fstar_7,
                                            consts->fesc_7, consts->Mlim_Fstar_mini, consts->Mlim_Fesc_mini);

        intgrl_stars_only_mini = Nion_General_MINI(consts->redshift, lnMmin, lnMmax, M_turn_m, M_turn_a,
                                            consts->alpha_star_mini, 0., consts->fstar_7,
                                            1., consts->Mlim_Fstar_mini, 0.);
    }

    averages->halo_mass = mass_intgrl * prefactor_mass;
    averages->stellar_mass = intgrl_stars_only * prefactor_stars;
    averages->halo_sfr = intgrl_stars_only * prefactor_sfr;
    averages->stellar_mass_mini = intgrl_stars_only_mini * prefactor_stars_mini;
    averages->sfr_mini = intgrl_stars_only_mini * prefactor_sfr_mini;
    averages->n_ion = intgrl_fesc_weighted*prefactor_nion + intgrl_fesc_weighted_mini*prefactor_nion_mini;
    averages->fescweighted_sfr = intgrl_fesc_weighted*prefactor_wsfr + intgrl_fesc_weighted_mini*prefactor_wsfr_mini;
    averages->halo_xray = intgrl_stars_only*prefactor_xray + intgrl_stars_only*prefactor_xray_mini;
    averages->m_turn_acg = M_turn_a;
    averages->m_turn_mcg = M_turn_m;

    return 0;
}

void halobox_debug_print_avg(struct HaloProperties *averages_box, struct HaloProperties *averages_subsampler, struct HaloBoxConstants *consts, double M_min, double M_cell){
    if(LOG_LEVEL < DEBUG_LEVEL)
        return;
    struct HaloProperties averages_sub_expected, averages_global;
    LOG_DEBUG("HALO BOXES REDSHIFT %.2f [%.2e %.2e]",consts->redshift,M_min,M_cell);
    if(flag_options_stoc->FIXED_HALO_GRIDS){
        get_box_averages(M_min, M_cell, averages_box->m_turn_acg, averages_box->m_turn_mcg, consts, &averages_global);
    }
    else{
        get_box_averages(user_params_stoc->SAMPLER_MIN_MASS, M_cell, averages_box->m_turn_acg, averages_box->m_turn_mcg, consts, &averages_global);
        if(user_params_stoc->AVG_BELOW_SAMPLER && M_min < user_params_stoc->SAMPLER_MIN_MASS){
            get_box_averages(M_min, user_params_stoc->SAMPLER_MIN_MASS, averages_box->m_turn_acg, averages_box->m_turn_mcg, consts, &averages_sub_expected);
        }
    }

    LOG_DEBUG("Exp. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY %11.3e, NION %11.3e)",
                averages_global.halo_mass,averages_global.stellar_mass, averages_global.stellar_mass_mini, averages_global.halo_sfr,
                averages_global.sfr_mini,averages_global.halo_xray,averages_global.n_ion);
    LOG_DEBUG("Box. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY %11.3e, NION %11.3e)",
                averages_box->halo_mass,averages_box->stellar_mass,averages_box->stellar_mass_mini,averages_box->halo_sfr,
                averages_box->sfr_mini,averages_box->halo_xray,averages_box->n_ion);

    if(!flag_options_stoc->FIXED_HALO_GRIDS && user_params_stoc->AVG_BELOW_SAMPLER && M_min < user_params_stoc->SAMPLER_MIN_MASS){
        LOG_DEBUG("of which SUB-SAMPLER",consts->redshift);
        LOG_DEBUG("Exp. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY %11.3e, NION %11.3e)",
                    averages_sub_expected.halo_mass,averages_sub_expected.stellar_mass, averages_sub_expected.stellar_mass_mini, averages_sub_expected.halo_sfr,
                    averages_sub_expected.sfr_mini,averages_sub_expected.halo_xray,averages_sub_expected.n_ion);
        LOG_DEBUG("Box. averages: (HM %11.3e, SM %11.3e SM_MINI %11.3e SFR %11.3e, SFR_MINI %11.3e, XRAY %11.3e, NION %11.3e)",
                    averages_subsampler->halo_mass,averages_subsampler->stellar_mass, averages_subsampler->stellar_mass_mini, averages_subsampler->halo_sfr,
                    averages_subsampler->sfr_mini,averages_subsampler->halo_xray,averages_subsampler->n_ion);
    }
    LOG_DEBUG("Turnovers: ACG %11.3e global %11.3e",averages_box->m_turn_acg,consts->mturn_a_nofb);
    LOG_DEBUG("MCG  %11.3e nofb %11.3e",averages_box->m_turn_mcg,consts->mturn_m_nofb);
    LOG_DEBUG("Reion %11.3e",averages_box->m_turn_reion);
}

void sum_halos_onto_grid(struct InitialConditions *ini_boxes, struct TsBox *previous_spin_temp, struct IonizedBox * previous_ionize_box,
                          struct PerturbHaloField *halos, struct HaloBoxConstants *consts, struct HaloBox *grids, struct HaloProperties *averages){
    double redshift = consts->redshift;
    //averages
    double hm_avg=0.,sm_avg=0.,sfr_avg=0.;
    double sm_avg_mini=0.,sfr_avg_mini=0.;
    double M_turn_a_avg=0.,M_turn_m_avg=0.,M_turn_r_avg=0.;
    double n_ion_avg=0., wsfr_avg=0., xray_avg=0.;
    //counts
    unsigned long long int total_n_halos, n_halos_cut=0.;

    double cell_volume = VOLUME / HII_TOT_NUM_PIXELS;
    #pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        int x,y,z;
        unsigned long long int i_halo,i_cell;
        double hmass,nion,sfr,wsfr,sfr_mini,stars_mini,stars,xray;
        double J21_val, Gamma12_val, zre_val;

        double curr_vcb = consts->vcb_norel;
        double M_turn_m = consts->mturn_m_nofb;
        double M_turn_a = consts->mturn_a_nofb;
        double M_turn_r = 0.;

        double in_props[3];
        struct HaloProperties out_props;

    #pragma omp for reduction(+:hm_avg,sm_avg,sm_avg_mini,sfr_avg,sfr_avg_mini,n_ion_avg,xray_avg,wsfr_avg,M_turn_a_avg,M_turn_m_avg,M_turn_r_avg,n_halos_cut)
        for(i_halo=0; i_halo<halos->n_halos; i_halo++){
            hmass = halos->halo_masses[i_halo];
            //It is sometimes useful to make cuts to the halo catalogues before gridding.
            //  We implement this in a simple way, if the user sets a halo's mass to zero we skip it
            if(hmass == 0.){
                n_halos_cut++;
                continue;
            }
            //NOTE:Unlike HaloField, PerturbHaloField is on HII_DIM so we don't need to correct here
            x = halos->halo_coords[0+3*i_halo];
            y = halos->halo_coords[1+3*i_halo];
            z = halos->halo_coords[2+3*i_halo];
            i_cell = HII_R_INDEX(x,y,z);

            //set values before reionisation feedback
            //NOTE: I could easily apply reionization feedback without minihalos but this was not done previously
            if(flag_options_stoc->USE_MINI_HALOS){
                if(!flag_options_stoc->FIX_VCB_AVG && user_params_stoc->USE_RELATIVE_VELOCITIES)
                    curr_vcb = ini_boxes->lowres_vcb[i_cell];

                J21_val=Gamma12_val=zre_val=0.;
                if(consts->redshift < global_params.Z_HEAT_MAX){
                    J21_val = previous_spin_temp->J_21_LW_box[i_cell];
                    Gamma12_val = previous_ionize_box->Gamma12_box[i_cell];
                    zre_val = previous_ionize_box->z_re_box[i_cell];
                }
                M_turn_m = lyman_werner_threshold(redshift, J21_val, curr_vcb, astro_params_stoc);
                M_turn_r = reionization_feedback(redshift, Gamma12_val, zre_val);
            }

            //these are the halo property RNG sequences
            in_props[0] = halos->star_rng[i_halo];
            in_props[1] = halos->sfr_rng[i_halo];
            in_props[2] = halos->xray_rng[i_halo];

            set_halo_properties(hmass,M_turn_a,M_turn_m,M_turn_r,consts,in_props,&out_props);

            sfr = out_props.halo_sfr;
            sfr_mini = out_props.sfr_mini;
            nion = out_props.n_ion;
            wsfr = out_props.fescweighted_sfr;
            stars = out_props.stellar_mass;
            stars_mini = out_props.stellar_mass_mini;
            xray = out_props.halo_xray;

            #if LOG_LEVEL >= ULTRA_DEBUG_LEVEL
            if(x+y+z == 0){
                LOG_ULTRA_DEBUG("Cell 0 Halo %d: HM: %.2e SM: %.2e (%.2e) SF: %.2e (%.2e) X: %.2e NI: %.2e WS: %.2e Z %.2e",i_halo,m,stars,stars_mini,sfr,sfr_mini,xray,nion,wsfr,out_props.metallicity);
                LOG_ULTRA_DEBUG("Mturn_a %.2e Mturn_m %.2e",M_turn_a,M_turn_m);
            }
            #endif

            //update the grids
            #pragma omp atomic update
            grids->halo_mass[i_cell] += hmass;
            #pragma omp atomic update
            grids->halo_stars[i_cell] += stars;
            #pragma omp atomic update
            grids->halo_stars_mini[i_cell] += stars_mini;
            #pragma omp atomic update
            grids->n_ion[i_cell] += nion;
            #pragma omp atomic update
            grids->halo_sfr[i_cell] += sfr;
            #pragma omp atomic update
            grids->halo_sfr_mini[i_cell] += sfr_mini;
            #pragma omp atomic update
            grids->whalo_sfr[i_cell] += wsfr;
            #pragma omp atomic update
            grids->halo_xray[i_cell] += xray;
            #pragma omp atomic update
            grids->count[i_cell] += 1;

            M_turn_m_avg += M_turn_m;
            hm_avg += hmass;
            sfr_avg += sfr;
            sfr_avg_mini += sfr_mini;
            sm_avg += stars;
            sm_avg_mini += stars_mini;
            xray_avg += xray;
            n_ion_avg += nion;
            wsfr_avg += wsfr;
            M_turn_a_avg += M_turn_a;
            M_turn_r_avg += M_turn_r;
            M_turn_m_avg += M_turn_m;
        }

        #pragma omp for
        for (i_cell=0; i_cell<HII_TOT_NUM_PIXELS; i_cell++) {
            grids->halo_mass[i_cell] /= cell_volume;
            grids->halo_sfr[i_cell] /= cell_volume;
            grids->halo_sfr_mini[i_cell] /= cell_volume;
            grids->halo_stars[i_cell] /= cell_volume;
            grids->halo_stars_mini[i_cell] /= cell_volume;
            grids->halo_xray[i_cell] /= cell_volume;
            grids->n_ion[i_cell] /= cell_volume;
            grids->whalo_sfr[i_cell] /= cell_volume;
        }
    }
    total_n_halos = halos->n_halos - n_halos_cut;
    LOG_SUPER_DEBUG("Cell 0 Halo %d: HM: %.2e SM: %.2e (%.2e) SF: %.2e (%.2e) X: %.2e NI: %.2e WS: %.2e ct : %d",grids->halo_mass[HII_R_INDEX(0,0,0)],
                        grids->halo_stars[HII_R_INDEX(0,0,0)],grids->halo_stars_mini[HII_R_INDEX(0,0,0)],
                        grids->halo_sfr[HII_R_INDEX(0,0,0)],grids->halo_sfr_mini[HII_R_INDEX(0,0,0)],
                        grids->halo_xray[HII_R_INDEX(0,0,0)],grids->n_ion[HII_R_INDEX(0,0,0)],
                        grids->whalo_sfr[HII_R_INDEX(0,0,0)],grids->count[HII_R_INDEX(0,0,0)]);

    //NOTE: There is an inconsistency here, the sampled grids use a halo-averaged turnover mass
    //  whereas the fixed grids / default 21cmfast uses the volume averaged LOG10(turnover mass).
    //  Neither of these are a perfect representation due to the nonlinear way turnover mass affects N_ion
    if(total_n_halos > 0){
        M_turn_r_avg /= total_n_halos;
        M_turn_a_avg /= total_n_halos;
        M_turn_m_avg /= total_n_halos;
    }
    //If we have no halos, assume the turnover has no reion feedback & no LW
    if(total_n_halos == 0){
        M_turn_m_avg = lyman_werner_threshold(redshift, 0., consts->vcb_norel, astro_params_stoc);
    }
    grids->log10_Mcrit_LW_ave = log10(M_turn_m_avg);

    hm_avg /= VOLUME;
    sm_avg /= VOLUME;
    sm_avg_mini /= VOLUME;
    sfr_avg /= VOLUME;
    sfr_avg_mini /= VOLUME;
    n_ion_avg /= VOLUME;
    xray_avg /= VOLUME;

    averages->halo_mass = hm_avg;
    averages->stellar_mass = sm_avg;
    averages->halo_sfr = sfr_avg;
    averages->stellar_mass_mini = sm_avg_mini;
    averages->sfr_mini = sfr_avg_mini;
    averages->halo_xray = xray_avg;
    averages->n_ion = n_ion_avg;
    averages->m_turn_acg = M_turn_a_avg;
    averages->m_turn_mcg = M_turn_m_avg;
    averages->m_turn_reion = M_turn_r_avg;
}

//We grid a PERTURBED halofield into the necessary quantities for calculating radiative backgrounds
int ComputeHaloBox(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
                    struct FlagOptions * flag_options, struct InitialConditions *ini_boxes, struct PerturbedField * perturbed_field, struct PerturbHaloField *halos,
                    struct TsBox *previous_spin_temp, struct IonizedBox *previous_ionize_box, struct HaloBox *grids){
    int status;
    Try{
        //get parameters
        Broadcast_struct_global_UF(user_params,cosmo_params);
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);
        Broadcast_struct_global_IT(user_params,cosmo_params,astro_params,flag_options);

        unsigned long long int idx;
#pragma omp parallel for num_threads(user_params->N_THREADS) private(idx)
        for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
            grids->halo_mass[idx] = 0.0;
            grids->n_ion[idx] = 0.0;
            grids->halo_sfr[idx] = 0.0;
            grids->halo_sfr_mini[idx] = 0.0;
            grids->halo_stars[idx] = 0.0;
            grids->halo_stars_mini[idx] = 0.0;
            grids->whalo_sfr[idx] = 0.0;
            grids->count[idx] = 0;
        }

        struct HaloBoxConstants hbox_consts;

        set_hbox_constants(redshift,astro_params,flag_options,&hbox_consts);

        LOG_DEBUG("Gridding %llu halos...",halos->n_halos);

        double M_min = minimum_source_mass(redshift,false,astro_params,flag_options);
        double cell_volume = VOLUME/HII_TOT_NUM_PIXELS;
        double M_cell = cosmo_params->OMm*RHOcrit*cell_volume;

        double M_turn_a_global, M_turn_m_global;

        struct HaloProperties averages_box, averages_subsampler, averages_global;

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
            set_fixed_grids(M_min, M_cell, ini_boxes, perturbed_field, previous_spin_temp, previous_ionize_box, &hbox_consts, grids, &averages_box);
            M_turn_a_global = averages_box.m_turn_acg;
            M_turn_m_global = averages_box.m_turn_mcg;
            get_box_averages(M_min, M_cell, M_turn_a_global, M_turn_m_global, &hbox_consts, &averages_global);
            //This is the mean adjustment that happens in the rest of the code
            //NOTE: in the default mode, global averages are fixed separately for minihalo and regular parts
            #pragma omp parallel for num_threads(user_params->N_THREADS)
            for(idx=0;idx<HII_TOT_NUM_PIXELS;idx++){
                grids->halo_mass[idx] *= averages_global.halo_mass/averages_box.halo_mass;
                grids->halo_stars[idx] *= averages_global.stellar_mass/averages_box.stellar_mass;
                grids->halo_stars_mini[idx] *= averages_global.stellar_mass_mini/averages_box.stellar_mass_mini;
                grids->halo_sfr[idx] *= averages_global.halo_sfr/averages_box.halo_sfr;
                grids->halo_sfr_mini[idx] *= averages_global.sfr_mini/averages_box.sfr_mini;
                grids->halo_xray[idx] *= averages_global.halo_xray/averages_box.halo_xray;
                grids->n_ion[idx] *= averages_global.n_ion/averages_box.n_ion;
                grids->whalo_sfr[idx] *= averages_global.fescweighted_sfr/averages_box.fescweighted_sfr;
            }
        }
        else{
            //set below-resolution properties
            if(user_params->AVG_BELOW_SAMPLER && M_min < user_params->SAMPLER_MIN_MASS){
                set_fixed_grids(M_min, user_params->SAMPLER_MIN_MASS, ini_boxes,
                                perturbed_field, previous_spin_temp, previous_ionize_box,
                                &hbox_consts, grids, &averages_subsampler);
                //This is pretty redundant, but since the fixed grids have density units (X Mpc-3) I have to re-multiply before adding the halos.
                //      I should instead have a flag to output the summed values in cell. (2*N_pixel > N_halo so generally i don't want to do it in the halo loop)
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] *= cell_volume;
                    grids->halo_stars[idx] *= cell_volume;
                    grids->halo_stars_mini[idx] *= cell_volume;
                    grids->halo_xray[idx] *= cell_volume;
                    grids->n_ion[idx] *= cell_volume;
                    grids->halo_sfr[idx] *= cell_volume;
                    grids->halo_sfr_mini[idx] *= cell_volume;
                    grids->whalo_sfr[idx] *= cell_volume;
                }
                LOG_DEBUG("finished subsampler M[%.2e %.2e]",M_min,user_params->SAMPLER_MIN_MASS);
            }
            sum_halos_onto_grid(ini_boxes, previous_spin_temp, previous_ionize_box,
                                halos, &hbox_consts, grids, &averages_box);
        }

        if(user_params->USE_INTERPOLATION_TABLES){
                freeSigmaMInterpTable();
        }
        halobox_debug_print_avg(&averages_box,&averages_subsampler,&hbox_consts,M_min,M_cell);
    }
    Catch(status){
        return(status);
    }
    LOG_DEBUG("Done.");
    return 0;
}

//test function for getting halo properties from the wrapper, can use a lot of memory for large catalogs
int test_halo_props(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
                    struct FlagOptions * flag_options, struct InitialConditions *ini_boxes, struct PerturbedField * perturbed_field, struct PerturbHaloField *halos,
                    struct TsBox *previous_spin_temp, struct IonizedBox *previous_ionize_box, float *halo_props){
    int status;
    Try{
        //get parameters
        Broadcast_struct_global_UF(user_params,cosmo_params);
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);

        struct HaloBoxConstants hbox_consts;
        set_hbox_constants(redshift,astro_params,flag_options,&hbox_consts);

        LOG_DEBUG("Getting props for %llu halos...",halos->n_halos);

        #pragma omp parallel num_threads(user_params_stoc->N_THREADS)
        {
            int x,y,z;
            unsigned long long int i_halo, i_cell;
            double m,nion,sfr,wsfr,sfr_mini,stars_mini,stars,xray;
            double J21_val, Gamma12_val, zre_val;

            double curr_vcb = hbox_consts.vcb_norel;
            double M_turn_m = hbox_consts.mturn_m_nofb;
            double M_turn_a = hbox_consts.mturn_a_nofb;
            double M_turn_r = 0.;

            double in_props[3];
            struct HaloProperties out_props;

        #pragma omp for
            for(i_halo=0; i_halo<halos->n_halos; i_halo++){
                m = halos->halo_masses[i_halo];
                //It is sometimes useful to make cuts to the halo catalogues before gridding.
                //  We implement this in a simple way, if the user sets a halo's mass to zero we skip it
                if(m == 0.){
                    continue;
                }
                x = halos->halo_coords[0+3*i_halo];
                y = halos->halo_coords[1+3*i_halo];
                z = halos->halo_coords[2+3*i_halo];
                i_cell = HII_R_INDEX(x,y,z);

                //set values before reionisation feedback
                //NOTE: I could easily apply reionization feedback without minihalos but this was not done previously
                if(flag_options_stoc->USE_MINI_HALOS){
                    if(!flag_options_stoc->FIX_VCB_AVG && user_params_stoc->USE_RELATIVE_VELOCITIES)
                        curr_vcb = ini_boxes->lowres_vcb[i_cell];

                    J21_val=Gamma12_val=zre_val=0.;
                    if(redshift < global_params.Z_HEAT_MAX){
                        J21_val = previous_spin_temp->J_21_LW_box[i_cell];
                        Gamma12_val = previous_ionize_box->Gamma12_box[i_cell];
                        zre_val = previous_ionize_box->z_re_box[i_cell];
                    }
                    M_turn_m = lyman_werner_threshold(redshift, J21_val, curr_vcb, astro_params_stoc);
                    M_turn_r = reionization_feedback(redshift, Gamma12_val, zre_val);
                }

                //these are the halo property RNG sequences
                in_props[0] = halos->star_rng[i_halo];
                in_props[1] = halos->sfr_rng[i_halo];
                in_props[2] = halos->xray_rng[i_halo];

                set_halo_properties(m,M_turn_a,M_turn_m,M_turn_r,&hbox_consts,in_props,&out_props);

                halo_props[12*i_halo +  0] = out_props.halo_mass;
                halo_props[12*i_halo +  1] = out_props.stellar_mass;
                halo_props[12*i_halo +  2] = out_props.halo_sfr;

                halo_props[12*i_halo +  3] = out_props.halo_xray;
                halo_props[12*i_halo +  4] = out_props.n_ion;
                halo_props[12*i_halo +  5] = out_props.fescweighted_sfr;

                halo_props[12*i_halo +  6] = out_props.stellar_mass_mini;
                halo_props[12*i_halo +  7] = out_props.sfr_mini;

                halo_props[12*i_halo +  8] = out_props.m_turn_acg;
                halo_props[12*i_halo +  9] = out_props.m_turn_mcg;
                halo_props[12*i_halo + 10] = out_props.m_turn_reion;
                halo_props[12*i_halo + 11] = out_props.metallicity;

                // LOG_ULTRA_DEBUG("HM %.2e SM %.2e SF %.2e NI %.2e LX %.2e",m,out_props.stellar_mass,out_props.halo_sfr,out_props.n_ion,out_props.halo_xray);
                // LOG_ULTRA_DEBUG("MINI: SM %.2e SF %.2e WSF %.2e Mta %.2e Mtm %.2e Mtr %.2e",out_props.stellar_mass_mini,out_props.sfr_mini,out_props.fescweighted_sfr,
                //                 out_props.m_turn_acg,out_props.m_turn_mcg,out_props.m_turn_reion);
                // LOG_ULTRA_DEBUG("RNG: STAR %.2e SFR %.2e XRAY %.2e",in_props[0],in_props[1],in_props[2]);
            }
        }

        if(user_params->USE_INTERPOLATION_TABLES){
            freeSigmaMInterpTable();
        }
    }
    Catch(status){
        return(status);
    }
    LOG_DEBUG("Done.");
    return 0;

}
