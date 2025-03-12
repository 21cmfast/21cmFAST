/* This file will contain functions which contain methods to directly calculate integrals from the frontend */
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "Constants.h"
#include "cosmology.h"
#include "hmf.h"
#include "scaling_relations.h"
#include "interp_tables.h"
#include "InputParameters.h"
#include "Stochasticity.h"

void get_sigma(UserParams *user_params, CosmoParams *cosmo_params, int n_masses, double *mass_values, double *sigma_out, double *dsigmasqdm_out){
    Broadcast_struct_global_noastro(user_params,cosmo_params);
    init_ps();

    if(user_params->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_MIN_INTEGRAL, M_MAX_INTEGRAL);

    int i;
    for(i=0;i<n_masses;i++){
        sigma_out[i] = EvaluateSigma(log(mass_values[i]));
        dsigmasqdm_out[i] = EvaluatedSigmasqdm(log(mass_values[i]));
    }
}

//integrates at fixed (set by parameters) mass range for many conditions
void get_condition_integrals(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double z_prev, int n_conditions, double *cond_values,
                        double *out_n_exp, double *out_m_exp){

    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    struct HaloSamplingConstants hs_const_struct;

    //unneccessarily creates the inverse table (a few seconds) but much cleaner this way
    stoc_set_consts_z(&hs_const_struct,redshift,z_prev);

    int i;
    for(i=0;i<n_conditions;i++){
        stoc_set_consts_cond(&hs_const_struct,cond_values[i]);
        out_n_exp[i] = hs_const_struct.expected_N;
        out_m_exp[i] = hs_const_struct.expected_M;
    }
}

//integrates a single condition over many mass ranges
//returns probability a sampled mass is within the range (integral / full integral range determined by parameters)
void integrate_chmf_masslims(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double z_prev, double cond_value, int n_masslim, double *lnM_lo, double *lnM_hi,
                        double *out_n){

    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    struct HaloSamplingConstants hs_const_struct;

    //unneccessarily creates tables if flags are set (a few seconds)
    stoc_set_consts_z(&hs_const_struct,redshift,z_prev);
    stoc_set_consts_cond(&hs_const_struct,cond_value);

    double exp_n_total = hs_const_struct.expected_N;

    int i;
    for(i=0;i<n_masslim;i++){
        out_n[i] = Nhalo_Conditional(
            hs_const_struct.growth_out,
            lnM_lo[i],
            lnM_hi[i],
            hs_const_struct.lnM_cond,
            hs_const_struct.sigma_cond,
            hs_const_struct.delta,
            0 //QAG
        ) / exp_n_total;
    }
}

void get_halomass_at_probability(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double z_prev, int n_conditions, double *cond_values, double *probabilities,
                        double *out_mass){

    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    struct HaloSamplingConstants hs_const_struct;

    stoc_set_consts_z(&hs_const_struct,redshift,z_prev);

    int i;
    for(i=0;i<n_conditions;i++){
        out_mass[i] = EvaluateNhaloInv(cond_values[i],probabilities[i]) * hs_const_struct.M_cond;
    }
}

void get_global_SFRD_z(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        int n_redshift, double *redshifts, double *log10_turnovers_mcg, double *out_sfrd, double *out_sfrd_mini){
    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    init_ps();

    //a bit hacky, but we need a lower limit for the tables
    double M_min = minimum_source_mass(redshifts[0],true,astro_params,flag_options);
    if(user_params->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min,M_MAX_INTEGRAL);

    struct ScalingConstants sc;

    int i;
    double z_min = user_params->Z_HEAT_MAX;
    double z_max = 0.;
    for(i=0;i<n_redshift;i++){
        if(redshifts[i] < z_min)
            z_min = redshifts[i];
        if(redshifts[i] > z_max)
            z_max = redshifts[i];
    }

    if(user_params->USE_INTERPOLATION_TABLES > 1){
        initialise_SFRD_spline(
            zpp_interp_points_SFR,
            z_min,
            z_max,
            &sc
        );
    }

    for(i=0;i<n_redshift;i++){
        set_scaling_constants(redshifts[0],astro_params,flag_options,&sc,false);
        out_sfrd[i] = EvaluateSFRD(redshifts[i],&sc);
        if(flag_options->USE_MINI_HALOS)
            out_sfrd_mini[i] = EvaluateSFRD_MINI(redshifts[i],log10_turnovers_mcg[i],&sc);
    }
}

void get_global_Nion_z(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        int n_redshift, double *redshifts, double *log10_turnovers_mcg, double *out_nion, double *out_nion_mini){
    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    init_ps();

    //a bit hacky, but we need a lower limit for the tables
    double M_min = minimum_source_mass(redshifts[0],true,astro_params,flag_options);
    if(user_params->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min,M_MAX_INTEGRAL);

    struct ScalingConstants sc;

    int i;
    double z_min = user_params->Z_HEAT_MAX;
    double z_max = 0.;
    for(i=0;i<n_redshift;i++){
        if(redshifts[i] < z_min)
            z_min = redshifts[i];
        if(redshifts[i] > z_max)
            z_max = redshifts[i];
    }

    if(user_params->USE_INTERPOLATION_TABLES > 1){
        initialise_Nion_Ts_spline(
            zpp_interp_points_SFR,
            z_min,
            z_max,
            &sc
        );
    }
    for(i=0;i<n_redshift;i++){
        set_scaling_constants(redshifts[0],astro_params,flag_options,&sc,false);
        out_nion[i] = EvaluateNionTs(redshifts[i],&sc);
        if(flag_options->USE_MINI_HALOS)
            out_nion_mini[i] = EvaluateNionTs_MINI(redshifts[i],log10_turnovers_mcg[i],&sc);
    }
}

void get_conditional_FgtrM(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double *out_fcoll, double *out_dfcoll){
    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    init_ps();

    double M_min = minimum_source_mass(redshift,true,astro_params,flag_options);
    if(user_params->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min,M_MAX_INTEGRAL);
    double sigma_min = EvaluateSigma(log(M_min));
    double sigma_cond = EvaluateSigma(log(RtoM(R)));
    double growthf = dicke(redshift);

    int i;
    double min_dens=-1;
    double max_dens=10;
    double dens;
    for(i=0;i<n_densities;i++){
        dens = densities[i];
        if(dens < min_dens) min_dens = dens;
        if(dens > max_dens) max_dens = dens;
    }
    if(user_params->USE_INTERPOLATION_TABLES > 1){
        initialise_FgtrM_delta_table(
            min_dens,
            max_dens,
            redshift,
            growthf,
            sigma_min,
            sigma_cond
        );
    }
    for(i=0;i<n_densities;i++){
        out_fcoll[i] = EvaluateFcoll_delta(densities[i],growthf,sigma_min,sigma_cond);
        out_dfcoll[i] = EvaluatedFcolldz(densities[i],growthf,sigma_min,sigma_cond);
    }
}

void get_conditional_SFRD(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double *log10_mturns,
                        double *out_sfrd, double *out_sfrd_mini){

    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    init_ps();

    double M_min = minimum_source_mass(redshift,true,astro_params,flag_options);
    if(user_params->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min,M_MAX_INTEGRAL);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    struct ScalingConstants sc;
    set_scaling_constants(redshift,astro_params,flag_options,&sc,false);

    int i;
    double min_dens=-1;
    double max_dens=10;
    double dens;
    for(i=0;i<n_densities;i++){
        dens = densities[i];
        if(dens < min_dens) min_dens = dens;
        if(dens > max_dens) max_dens = dens;
    }

    if(user_params->USE_INTERPOLATION_TABLES > 1){
        initialise_SFRD_Conditional_table(
            redshift,
            min_dens,
            max_dens,
            M_min,
            M_cond,
            M_cond,
            &sc
        );
    }
    for(i=0;i<n_densities;i++)
        out_sfrd[i] = EvaluateSFRD_Conditional(densities[i],growthf,M_min,M_cond,M_cond,sigma_cond,&sc);
    if(flag_options->USE_MINI_HALOS){
        for(i=0;i<n_densities;i++)
            out_sfrd_mini[i] = EvaluateSFRD_Conditional_MINI(densities[i],log10_mturns[i],
                                                growthf,M_min,M_cond,M_cond,sigma_cond,&sc);
    }
}

void get_conditional_Nion(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double *log10_mturns_acg, double *log10_mturns_mcg,
                        double *out_nion, double *out_nion_mini){

    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    init_ps();

    double M_min = minimum_source_mass(redshift,true,astro_params,flag_options);
    if(user_params->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min,M_MAX_INTEGRAL);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    struct ScalingConstants sc;
    set_scaling_constants(redshift,astro_params,flag_options,&sc,false);

    int i;
    double min_dens=-1;
    double max_dens=10;
    double min_l10mturn_acg=10.;
    double min_l10mturn_mcg=10.;
    double max_l10mturn_acg=5.;
    double max_l10mturn_mcg=5.;
    double dens, l10mturn_a, l10mturn_m;
    for(i=0;i<n_densities;i++){
        dens = densities[i];
        l10mturn_a = log10_mturns_acg[i];
        l10mturn_m = log10_mturns_mcg[i];
        if(dens < min_dens) min_dens = dens;
        if(dens > max_dens) max_dens = dens;
        if(l10mturn_a < min_l10mturn_acg) min_l10mturn_acg = l10mturn_a;
        if(l10mturn_a > max_l10mturn_acg) max_l10mturn_acg = l10mturn_a;
        if(l10mturn_m < min_l10mturn_mcg) min_l10mturn_mcg = l10mturn_m;
        if(l10mturn_m > max_l10mturn_mcg) max_l10mturn_mcg = l10mturn_m;
    }

    if(user_params->USE_INTERPOLATION_TABLES > 1){
        initialise_Nion_Conditional_spline(
            redshift,
            min_dens,
            max_dens,
            M_min,
            M_cond,
            M_cond,
            min_l10mturn_acg,
            max_l10mturn_acg,
            min_l10mturn_mcg,
            max_l10mturn_mcg,
            &sc,
            false
        );
    }
    for(i=0;i<n_densities;i++)
        out_nion[i] = EvaluateNion_Conditional(densities[i],log10_mturns_acg[i],growthf,M_min,M_cond,M_cond,sigma_cond,&sc,false);
    if(flag_options->USE_MINI_HALOS){
        for(i=0;i<n_densities;i++)
            out_nion_mini[i] = EvaluateNion_Conditional_MINI(densities[i],log10_mturns_mcg[i],
                                                growthf,M_min,M_cond,M_cond,sigma_cond,&sc,false);
    }
}

void get_conditional_Xray(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        double redshift, double R, int n_densities, double *densities, double *log10_mturns,
                        double *out_xray){

    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    init_ps();

    double M_min = minimum_source_mass(redshift,true,astro_params,flag_options);
    if(user_params->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_min,M_MAX_INTEGRAL);
    double M_cond = RtoM(R);
    double sigma_cond = EvaluateSigma(log(M_cond));
    double growthf = dicke(redshift);

    struct ScalingConstants sc;
    set_scaling_constants(redshift,astro_params,flag_options,&sc,false);

    int i;
    double min_dens=-1;
    double max_dens=10;
    double dens;
    for(i=0;i<n_densities;i++){
        dens = densities[i];
        if(dens < min_dens) min_dens = dens;
        if(dens > max_dens) max_dens = dens;
    }

    if(user_params->USE_INTERPOLATION_TABLES > 1){
        initialise_Xray_Conditional_table(
            redshift,
            min_dens,
            max_dens,
            M_min,
            M_cond,
            M_cond,
            &sc
        );
    }
    for(i=0;i<n_densities;i++)
        out_xray[i] = EvaluateXray_Conditional(densities[i],log10_mturns[i],redshift,growthf,M_min,M_cond,M_cond,sigma_cond,&sc);
}
