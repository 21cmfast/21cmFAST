/*This file contains the halo scaling relations which can be used by
    the integrals in hmf.c or the sampled halos in HaloBox.c*/
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "cosmology.h"
#include "photoncons.h"
#include "hmf.h"
#include "thermochem.h"

#include "scaling_relations.h"

void set_scaling_constants(double redshift, AstroParams *astro_params, FlagOptions *flag_options, struct ScalingConstants *consts){
    consts->redshift = redshift;

    //Set on for the fixed grid case since we are missing halos above the cell mass
    consts->fix_mean = flag_options->FIXED_HALO_GRIDS;
    //whether to fix *integrated* (not sampled) galaxy properties to the expected mean
    consts->scaling_median = flag_options->HALO_SCALING_RELATIONS_MEDIAN;

    consts->fstar_10 = astro_params->F_STAR10;
    consts->alpha_star = astro_params->ALPHA_STAR;
    consts->sigma_star = astro_params->SIGMA_STAR;

    consts->alpha_upper = astro_params->UPPER_STELLAR_TURNOVER_INDEX;
    consts->pivot_upper = astro_params->UPPER_STELLAR_TURNOVER_MASS;
    consts->upper_pivot_ratio = pow(consts->pivot_upper/1e10,consts->alpha_star) + \
                                pow(consts->pivot_upper/1e10,consts->alpha_upper);

    consts->fstar_7 = astro_params->F_STAR7_MINI;
    consts->alpha_star_mini = astro_params->ALPHA_STAR_MINI;

    consts->t_h = t_hubble(redshift);
    consts->t_star = astro_params->t_STAR;
    consts->sigma_sfr_lim = astro_params->SIGMA_SFR_LIM;
    consts->sigma_sfr_idx = astro_params->SIGMA_SFR_INDEX;

    consts->l_x = astro_params->L_X * 1e-38; //setting units to 1e38 erg s -1 so we can store in float
    consts->l_x_mini = astro_params->L_X_MINI * 1e-38;
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
        consts->vcb_norel = flag_options->FIX_VCB_AVG ? global_params.VAVG : 0;
        consts->mturn_m_nofb = lyman_werner_threshold(redshift, 0., consts->vcb_norel, astro_params);
    }

    if(flag_options->FIXED_HALO_GRIDS || user_params_global->AVG_BELOW_SAMPLER){
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


/*
General Scaling realtions used in mass function integrals and sampling.

These are usually integrated over log(M) and can avoid a few of pow/exp calls
by keeping them in log. Since they are called within integrals they don't use the
ScalingConstants. Instead pulling from the GSL integration parameter structs
*/

double scaling_single_PL(double M, double alpha, double pivot){
    return pow(M/pivot,alpha);
}

double log_scaling_single_PL(double lnM, double alpha, double ln_pivot){
    return alpha*(lnM - ln_pivot);
}

//single power-law with provided limit for scaling == 1, returns f(M)/f(pivot) == norm
//limit is provided ahead of time for efficiency
double scaling_PL_limit(double M, double norm, double alpha, double pivot, double limit){
    if ((alpha > 0. && M > limit) || (alpha < 0. && M < limit))
        return 1/norm;

    //if alpha is zero, this returns 1 as expected (note strict inequalities above)
    return scaling_single_PL(M,alpha,pivot);
}

//log version for possible acceleration
double log_scaling_PL_limit(double lnM, double ln_norm, double alpha, double ln_pivot, double ln_limit){
    if ((alpha > 0. && lnM > ln_limit) || (alpha < 0. && lnM < ln_limit))
        return -ln_norm;

    //if alpha is zero, this returns log(1) as expected (note strict inequalities above)
    return log_scaling_single_PL(lnM,alpha,ln_pivot);
}

//concave-down double power-law, we pass pivot_ratio == denominator(M=pivot_lo)
//  to save pow() calls. Due to the double power-law we gain little from a log verison
//  returns f(M)/f(M==pivot_lo)
double scaling_double_PL(double M, double alpha_lo, double pivot_ratio,
                        double alpha_hi, double pivot_hi){
    //if alpha is zero, this returns 1 as expected (note strict inequalities above)
    return pivot_ratio / (pow(M/pivot_hi,-alpha_lo) + pow(M/pivot_hi,-alpha_hi));
}

/*
Scaling relations used in the halo sampler. Quantities calculated for a single halo mass
and are summed onto grids.
*/

//The mean Lx_over_SFR given by Lehmer+2021, by integrating analyitically over their double power-law + exponential Luminosity function
//NOTE: this relation is fit to low-z, and there is a PEAK of Lx/SFR around 10% solar due to the critical L term
//NOTE: this relation currently also has no normalisation, and is fixed to the Lehmer values for now
double lx_on_sfr_Lehmer(double metallicity){
    double l10z = log10(metallicity);

    //LF parameters from Lehmer+2021
    double slope_low = 1.74;
    double slope_high = 1.16 + 1.34*l10z;
    double xray_norm = 1.29;
    double l10break_L = 38.54 - 38.; //the -38 written explicitly for clarity on our units
    double l10crit_L = 39.98 - 38. + 0.6*l10z;
    double L_ratio = pow(10,l10break_L - l10crit_L);

    //The double power-law + exponential integrates to an upper and lower incomplete Gamma function
    //since the slope is < 2 we don't need to set a lower limit, but this can be done by replacing
    //gamma(2-y_low) with gamma_inc(2-y_low,L_lower/L_crit)
    double prefactor_low = pow(10,l10crit_L*(2-slope_low));
    double prefactor_high = pow(10,l10crit_L*(2-slope_high) + l10break_L*(slope_high-slope_low));
    double gamma_low = gsl_sf_gamma(2-slope_low) - gsl_sf_gamma_inc(2-slope_low,L_ratio);
    double gamma_high = gsl_sf_gamma_inc(2-slope_high,L_ratio);

    double lx_over_sfr = xray_norm*(prefactor_low*gamma_low + prefactor_high*gamma_high);

    return lx_over_sfr;
}

//double power-law in Z with the low-metallicity PL fixed as constant
double lx_on_sfr_doublePL(double metallicity, double lx_constant){
    double hi_z_index = -0.64; //power-law index of LX/SFR at high-z
    double lo_z_index = 0.;
    double z_pivot = 0.05; // Z at which LX/SFR == lx_constant / 2

    return lx_constant*scaling_double_PL(metallicity,lo_z_index,1.,hi_z_index,z_pivot);
}

//first order power law Lx with cross-term (e.g Kaur+22)
//here the constant defines the value at 1 Zsun and 1 Msun yr-1
double lx_on_sfr_PL_Kaur(double sfr, double metallicity, double lx_constant){
    //Hardcoded for now (except the lx normalisation and the scatter): 3 extra fit parameters in the equation
    //taking values from Kaur+22, constant factors controlled by astro_params->L_X
    double sfr_index = 0.03;
    double z_index = -0.64;
    double cross_index = 0.0;
    double l10z = log10(metallicity);

    double lx_over_sfr = (cross_index*l10z + sfr_index)*log10(sfr*SperYR) + z_index*l10z;
    return pow(10,lx_over_sfr) * lx_constant;
}

//Schechter function from Kaur+22
//Here the constant defines the value minus 1 at the turnover Z
double lx_on_sfr_Schechter(double metallicity, double lx_constant){
    //Hardcoded for now (except the lx normalisation and the scatter): 3 extra fit parameters in the equation
    //taking values from Kaur+22, constant factors controlled by astro_params->L_X
    double z_turn = 8e-3/0.02; //convert to solar
    double logz_index = 0.3;
    double l10z = log10(metallicity/z_turn);

    double lx_over_sfr = logz_index*l10z - metallicity/z_turn;
    return pow(10,lx_over_sfr) * lx_constant;
}

double get_lx_on_sfr(double sfr, double metallicity, double lx_constant){
    //Future TODO: experiment more with these models and parameterise properly
    // return lx_on_sfr_Lehmer(metallicity);
    // return lx_on_sfr_Schechter(metallicity, lx_constant);
    // return lx_on_sfr_PL_Kaur(sfr,metallicity, lx_constant);
    //HACK: new/old model switch with upperstellar flag
    if(flag_options_global->USE_UPPER_STELLAR_TURNOVER)
        return lx_on_sfr_doublePL(metallicity, lx_constant);
    return lx_constant;
}

void get_halo_stellarmass(double halo_mass, double mturn_acg, double mturn_mcg, double star_rng,
                             struct ScalingConstants *consts, double *star_acg, double *star_mcg){
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

    double baryon_ratio = cosmo_params_global->OMb / cosmo_params_global->OMm;
    //adjustment to the mean for lognormal scatter
    double stoc_adjustment_term = consts->scaling_median ? 0 : sigma_star * sigma_star / 2.;

    //We don't want an upturn even with a negative ALPHA_STAR
    if(flag_options_global->USE_UPPER_STELLAR_TURNOVER && (f_a > fu_a)){
        fstar_mean = scaling_double_PL(halo_mass,f_a,consts->upper_pivot_ratio,fu_a,fu_p);
    }
    else{
        fstar_mean = scaling_single_PL(halo_mass,consts->alpha_star,1e10); //PL term
    }
    f_sample = f_10 * fstar_mean * exp(-mturn_acg/halo_mass + star_rng*sigma_star - stoc_adjustment_term); //1e10 normalisation of stellar mass
    if(f_sample > 1.) f_sample = 1.;

    sm_sample = f_sample * halo_mass * baryon_ratio;
    *star_acg = sm_sample;

    if(!flag_options_global->USE_MINI_HALOS){
        *star_mcg = 0.;
        return;
    }

    f_sample_mini = pow(halo_mass/1e7,f_a_mini) * f_7;
    f_sample_mini *= exp(-mturn_mcg/halo_mass - halo_mass/mturn_acg + star_rng*sigma_star - stoc_adjustment_term);
    if(f_sample_mini > 1.) f_sample_mini = 1.;

    sm_sample_mini = f_sample_mini * halo_mass * baryon_ratio;
    *star_mcg = sm_sample_mini;
}

void get_halo_sfr(double stellar_mass, double stellar_mass_mini, double sfr_rng,
                     struct ScalingConstants *consts, double *sfr, double *sfr_mini){
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

    //adjustment to the mean for lognormal scatter
    double stoc_adjustment_term = consts->scaling_median ? 0 : sigma_sfr * sigma_sfr / 2.;
    sfr_sample = sfr_mean * exp(sfr_rng*sigma_sfr - stoc_adjustment_term);
    *sfr = sfr_sample;

    if(!flag_options_global->USE_MINI_HALOS){
        *sfr_mini = 0.;
        return;
    }

    sfr_mean_mini = stellar_mass_mini / (consts->t_star * consts->t_h);
    sfr_sample_mini = sfr_mean_mini * exp(sfr_rng*sigma_sfr - stoc_adjustment_term);
    *sfr_mini = sfr_sample_mini;
}

void get_halo_metallicity(double sfr, double stellar, double redshift, double *z_out){
    //Hardcoded for now: 6 extra fit parameters in the equation
    double z_denom, z_result;
    double redshift_scaling = pow(10,-0.056*redshift + 0.064);
    double stellar_term = 1.;
    if(stellar > 0.){
      stellar_term = pow(1 + pow(stellar/z_denom,-2.1),-0.148);
      z_denom = (1.28825e10 * pow(sfr*SperYR,0.56));
    }

    z_result = 1.23 * stellar_term * redshift_scaling;

    *z_out = z_result;
}

void get_halo_xray(double sfr, double sfr_mini, double metallicity, double xray_rng, struct ScalingConstants *consts, double *xray_out){
    double sigma_xray = consts->sigma_xray;

    //adjustment to the mean for lognormal scatter
    double stoc_adjustment_term = consts->scaling_median ? 0 : sigma_xray * sigma_xray / 2.;
    double rng_factor = exp(xray_rng*consts->sigma_xray - stoc_adjustment_term);

    double lx_over_sfr = get_lx_on_sfr(sfr,metallicity,consts->l_x);
    double xray = lx_over_sfr*(sfr*SperYR)*rng_factor;

    if(flag_options_global->USE_MINI_HALOS){
        lx_over_sfr = get_lx_on_sfr(sfr_mini,metallicity,consts->l_x_mini); //Since there *are* some SFR-dependent models, this is done separately
        xray += lx_over_sfr*(sfr_mini*SperYR)*rng_factor;
    }

    *xray_out = xray;
}
