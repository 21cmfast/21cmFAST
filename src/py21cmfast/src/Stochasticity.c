/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"
#include "Constants.h"
#include "indexing.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "interp_tables.h"
#include "hmf.h"
#include "cosmology.h"
#include "InitialConditions.h"

#include "Stochasticity.h"
//buffer size (per cell of arbitrary size) in the sampling function
#define MAX_HALO_CELL (int)1e5

//parameters for the halo mass->stars calculations
//Note: ideally I would split this into constants set per snapshot and
//  constants set per condition, however some variables (delta or Mass)
//  can be set with differing frequencies depending on the condition type
struct HaloSamplingConstants{
    //calculated per redshift
    int from_catalog; //flag for first box or updating halos
    double corr_sfr;
    double corr_star;
    double corr_xray;

    double z_in;
    double z_out;
    double growth_in;
    double growth_out;
    double M_min;
    double lnM_min;
    double M_max_tables;
    double lnM_max_tb;
    double sigma_min;

    //per-condition/redshift depending on from_catalog or not
    double delta;
    double M_cond;
    double lnM_cond;
    double sigma_cond;

    //calculated per condition
    double cond_val; //This is the table x value (density for grids, log mass for progenitors)
    double expected_N;
    double expected_M;
};

void print_hs_consts(struct HaloSamplingConstants * c){
    LOG_DEBUG("Printing halo sampler constants....");
    LOG_DEBUG("from_catalog %d z_in %.2f z_out %.2f d_in %.2f d_out %.2f",c->from_catalog,c->z_in,c->z_out,c->growth_in,c->growth_out);
    LOG_DEBUG("M_min %.2e (%.2e) (%.2f) M_max %.2e (%.2e)",c->M_min,c->lnM_min,c->sigma_min,c->M_max_tables,c->lnM_max_tb);
    LOG_DEBUG("Corr Star %.2e SFR %.2e",c->corr_star,c->corr_sfr);
    LOG_DEBUG("CONDITION DEPENDENT STUFF (may not be set)");
    LOG_DEBUG("delta %.2e M_c %.2e (%.2e) (%.2e) cond %.2e",c->delta,c->M_cond,c->lnM_cond,c->sigma_cond,c->cond_val);
    LOG_DEBUG("exp N %.2f exp M %.2e",c->expected_N,c->expected_M);
    return;
}

//This function, designed to be used in the wrapper to estimate Halo catalogue size, takes the parameters and returns average number of halos within the entire box
double expected_nhalo(double redshift, UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions * flag_options){
    //minimum sampled mass
    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);
    double M_min = user_params->SAMPLER_MIN_MASS;
    //maximum sampled mass
    double M_max = RHOcrit * cosmo_params->OMm * VOLUME / HII_TOT_NUM_PIXELS;
    double result;

    init_ps();
    if(user_params->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(M_min,M_max);

    result = Nhalo_General(redshift, log(M_min), log(M_max)) * VOLUME * cosmo_params->OMm * RHOcrit;
    LOG_DEBUG("Expected %.2e Halos in the box from masses %.2e to %.2e at z=%.2f",result,M_min,M_max,redshift);

    if(user_params->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();

    return result;
}

double sample_dndM_inverse(double condition, struct HaloSamplingConstants * hs_constants, gsl_rng * rng){
    double p_in, min_prob, result;
    p_in = gsl_rng_uniform(rng);
    result = EvaluateNhaloInv(condition,p_in);
    result = fmin(1,fmax(0,result)); //clip in case of extrapolation
    result = result * hs_constants->M_cond;
    return result;
}

//Set the constants that are calculated once per snapshot
void stoc_set_consts_z(struct HaloSamplingConstants *const_struct, double redshift, double redshift_desc){
    LOG_DEBUG("Setting z constants z=%.2f z_desc=%.2f",redshift,redshift_desc);
    const_struct->growth_out = dicke(redshift);
    const_struct->z_out = redshift;
    const_struct->z_in = redshift_desc;

    const_struct->M_min = user_params_global->SAMPLER_MIN_MASS / user_params_global->SAMPLER_BUFFER_FACTOR;
    const_struct->lnM_min = log(const_struct->M_min);
    const_struct->M_max_tables = global_params.M_MAX_INTEGRAL;
    const_struct->lnM_max_tb = log(const_struct->M_max_tables);

    init_ps();
    if(user_params_global->USE_INTERPOLATION_TABLES){
        if(user_params_global->SAMPLE_METHOD == 3)
            initialiseSigmaMInterpTable(const_struct->M_min/2,const_struct->M_max_tables); //the binary split needs to go below the resolution
        else
            initialiseSigmaMInterpTable(const_struct->M_min,const_struct->M_max_tables);

        if(user_params_global->SAMPLE_METHOD == 2)
            InitialiseSigmaInverseTable();
    }

    const_struct->sigma_min = EvaluateSigma(const_struct->lnM_min);

    if(redshift_desc >= 0){
        const_struct->growth_in = dicke(redshift_desc);
        if(astro_params_global->CORR_SFR > 0)
            const_struct->corr_sfr = exp(-(redshift - redshift_desc)/astro_params_global->CORR_SFR);
        else
            const_struct->corr_sfr = 0;
        if(astro_params_global->CORR_STAR > 0)
            const_struct->corr_star = exp(-(redshift - redshift_desc)/astro_params_global->CORR_STAR);
        else
            const_struct->corr_star = 0;
        if(astro_params_global->CORR_LX > 0)
            const_struct->corr_xray = exp(-(redshift - redshift_desc)/astro_params_global->CORR_LX);
        else
            const_struct->corr_xray = 0;

        const_struct->from_catalog = 1;
        initialise_dNdM_tables(log(user_params_global->SAMPLER_MIN_MASS), const_struct->lnM_max_tb, const_struct->lnM_min, const_struct->lnM_max_tb,
                                const_struct->growth_out, const_struct->growth_in, true);
        if(user_params_global->SAMPLE_METHOD == 0 || user_params_global->SAMPLE_METHOD == 1){
            initialise_dNdM_inverse_table(log(user_params_global->SAMPLER_MIN_MASS), const_struct->lnM_max_tb, const_struct->lnM_min,
                                    const_struct->growth_out, const_struct->growth_in, true);
        }
        if(user_params_global->SAMPLE_METHOD == 3){
            initialise_J_split_table(200,1e-4,20.,0.2);
        }
    }
    else {
        double M_cond = RHOcrit * cosmo_params_global->OMm * VOLUME / HII_TOT_NUM_PIXELS;
        const_struct->M_cond = M_cond;
        const_struct->lnM_cond = log(M_cond);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond);
        //for the table limits
        double delta_crit = get_delta_crit(user_params_global->HMF,const_struct->sigma_cond,const_struct->growth_out);
        const_struct->from_catalog = 0;
        //TODO: determine the minimum density in the field and pass it in (<-1 is fine for Lagrangian)
        initialise_dNdM_tables(DELTA_MIN, MAX_DELTAC_FRAC*delta_crit, const_struct->lnM_min, const_struct->lnM_max_tb,
                                const_struct->growth_out, const_struct->lnM_cond, false);
        initialise_dNdM_inverse_table(DELTA_MIN, MAX_DELTAC_FRAC*delta_crit, const_struct->lnM_min,
                                const_struct->growth_out, const_struct->lnM_cond, false);
    }

    LOG_DEBUG("Done.");
    return;
}

//set the constants which are calculated once per condition
void stoc_set_consts_cond(struct HaloSamplingConstants *const_struct, double cond_val){
    double m_exp,n_exp;

    //Here the condition is a mass, volume is the Lagrangian volume and delta_l is set by the
    //redshift difference which represents the difference in delta_crit across redshifts
    if(const_struct->from_catalog){
        const_struct->M_cond = cond_val;
        const_struct->lnM_cond = log(cond_val);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond);
        //mean stellar mass of this halo mass, used for stellar z correlations
        const_struct->cond_val = const_struct->lnM_cond;
        //condition delta is the previous delta crit
        const_struct->delta = get_delta_crit(user_params_global->HMF,const_struct->sigma_cond,const_struct->growth_in)\
                                                / const_struct->growth_in * const_struct->growth_out;
    }
    //Here the condition is a cell of a given density, the volume/mass is given by the grid parameters
    else{
        //since the condition mass/sigma is already set all we need is delta
        const_struct->delta = cond_val;
        const_struct->cond_val = cond_val;
    }

    //Get expected N and M from interptables
    //the splines don't work well for cells above Deltac, but there CAN be cells above deltac, since this calculation happens
    //before the overlap, and since the smallest dexm mass is M_cell*(1.01^3) there *could* be a cell above Deltac not in a halo
    //NOTE: all this does is prevent integration errors below since these cases are also dealt with in stoc_sample
    if(const_struct->delta > MAX_DELTAC_FRAC*get_delta_crit(user_params_global->HMF,const_struct->sigma_cond,const_struct->growth_out)){
        const_struct->expected_M = const_struct->M_cond;
        const_struct->expected_N = 1;
    }
    else if(const_struct->delta <= DELTA_MIN){
        const_struct->expected_M = 0;
        const_struct->expected_N = 0;
    }
    else{
        n_exp = EvaluateNhalo(const_struct->cond_val,const_struct->growth_out,const_struct->lnM_min,
                                const_struct->lnM_max_tb,const_struct->M_cond,const_struct->sigma_cond,const_struct->delta);
        m_exp = EvaluateMcoll(const_struct->cond_val,const_struct->growth_out,const_struct->lnM_min,
                                const_struct->lnM_max_tb,const_struct->M_cond,const_struct->sigma_cond,const_struct->delta);
        const_struct->expected_N = n_exp * const_struct->M_cond;
        const_struct->expected_M = m_exp * const_struct->M_cond;
    }
    return;
}

void place_on_hires_grid(int x, int y, int z, int *crd_hi, gsl_rng * rng){
    //we want to randomly place each halo within each lores cell,then map onto hires
    //this is so halos are on DIM grids to match HaloField and Perturb options
    int x_hi,y_hi,z_hi;
    double randbuf;
    int lo_dim = user_params_global->HII_DIM;
    int hi_dim = user_params_global->DIM;
    randbuf = gsl_rng_uniform(rng);
    x_hi = (int)((x + randbuf) / (double)(lo_dim) * (double)(hi_dim));
    randbuf = gsl_rng_uniform(rng);
    y_hi = (int)((y + randbuf) / (double)(lo_dim) * (double)(hi_dim));
    randbuf = gsl_rng_uniform(rng);
    z_hi = (int)((z + randbuf) / (double)(HII_D_PARA) * (double)(D_PARA));
    crd_hi[0] = x_hi;
    crd_hi[1] = y_hi;
    crd_hi[2] = z_hi;
}

//This function adds stochastic halo properties to an existing halo
void set_prop_rng(gsl_rng *rng, bool from_catalog, double *interp, double * input, double * output){
    double rng_star,rng_sfr,rng_xray;

    //Correlate properties by interpolating between the sampled and descendant gaussians
    rng_star = astro_params_global->SIGMA_STAR > 0. ? gsl_ran_ugaussian(rng) : 0.;
    rng_sfr = astro_params_global->SIGMA_SFR_LIM > 0. ? gsl_ran_ugaussian(rng) : 0.;
    rng_xray = astro_params_global->SIGMA_LX > 0. ? gsl_ran_ugaussian(rng) : 0.;

    if(from_catalog){
        rng_star = (1-interp[0])*rng_star + interp[0]*input[0];
        rng_sfr = (1-interp[1])*rng_sfr + interp[1]*input[1];
        rng_xray = (1-interp[2])*rng_xray + interp[2]*input[2];
    }

    output[0] = rng_star;
    output[1] = rng_sfr;
    output[2] = rng_xray;
    return;
}

//This is the function called to assign halo properties to an entire catalogue, used for DexM halos
int add_properties_cat(unsigned long long int seed, float redshift, HaloField *halos){
    //set up the rng
    gsl_rng * rng_stoc[user_params_global->N_THREADS];
    seed_rng_threads(rng_stoc,seed);

    LOG_DEBUG("computing rng for %llu halos",halos->n_halos);

    //loop through the halos and assign properties
    unsigned long long int i;
    double buf[3];
    double dummy[3]; //we don't need interpolation here
#pragma omp parallel for private(i,buf)
    for(i=0;i<halos->n_halos;i++){
        // LOG_ULTRA_DEBUG("halo %d hm %.2e crd %d %d %d",i,halos->halo_masses[i],halos->halo_coords[3*i+0],halos->halo_coords[3*i+1],halos->halo_coords[3*i+2]);
        set_prop_rng(rng_stoc[omp_get_thread_num()], false, dummy, dummy, buf);
        // LOG_ULTRA_DEBUG("stars %.2e sfr %.2e",buf[0],buf[1]);
        halos->star_rng[i] = buf[0];
        halos->sfr_rng[i] = buf[1];
        halos->xray_rng[i] = buf[2];
    }

    free_rng_threads(rng_stoc);

    LOG_DEBUG("Done.");
    return 0;
}

/* Creates a realisation of halo properties by sampling the halo mass function and
 * conditional property PDFs, the number of halos is poisson sampled from the integrated CMF*/
int stoc_halo_sample(struct HaloSamplingConstants *hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    double exp_N = hs_constants->expected_N;

    double hm_sample;
    int ii, nh;
    int halo_count=0;

    double tbl_arg = hs_constants->cond_val;

    nh = gsl_ran_poisson(rng,exp_N);
    for(ii=0;ii<nh;ii++){
        M_out[halo_count++] = sample_dndM_inverse(tbl_arg,hs_constants,rng);;
    }

    *n_halo_out = halo_count;
    return 0;
}

double remove_random_halo(gsl_rng * rng, int n_halo, int *idx, double *M_prog, float *M_out){
    double last_M_del;
    int random_idx;
    do {
        random_idx = gsl_rng_uniform_int(rng,n_halo);
    } while(M_out[random_idx] == 0);
    last_M_del = M_out[random_idx];
    *M_prog -= last_M_del;
    M_out[random_idx] = 0; //zero mass halos are skipped and not counted

    *idx = random_idx;
    return last_M_del;
}

/*Function which "corrects" a mass sample after it exceeds the expected mass*/
//CURRENT IMPLEMENTATION: half the time I keep/throw away the last halo based on which sample is closer to the expected mass.
// However this introduces a bias since the last halo is likely larger than average So the other half the time,
// I throw away random halos until we are again below exp_M, effectively the same process in reverse. which has the opposite bias
void fix_mass_sample(gsl_rng * rng, double exp_M, int *n_halo_pt, double *M_tot_pt, float *M_out){
    //Keep the last halo if it brings us closer to the expected mass
    //This is done by addition or subtraction over the limit to balance
    //the bias of the last halo being larger
    int random_idx;
    double last_M_del;
    bool sel = gsl_rng_uniform_int(rng,2);
    // int sel = 1;
    if(sel){
        if(fabs(*M_tot_pt - M_out[*n_halo_pt-1] - exp_M) < fabs(*M_tot_pt - exp_M)){
            *M_tot_pt -= M_out[*n_halo_pt-1];
            //here we remove by setting the counter one lower so it isn't read
            (*n_halo_pt)--; //increment has preference over dereference
        }
    }
    else{
        while(*M_tot_pt > exp_M){
            //here we remove by setting halo mass to zero, skipping it during the consolidation
            last_M_del = remove_random_halo(rng,*n_halo_pt,&random_idx,M_tot_pt,M_out);
        }

        // if the sample with the last subtracted halo is closer to the expected mass, keep it
        // LOG_ULTRA_DEBUG("Deciding to keep last halo M %.3e tot %.3e exp %.3e",last_M_del,*M_tot_pt,exp_M);
        if(fabs(*M_tot_pt + last_M_del - exp_M) < fabs(*M_tot_pt - exp_M)){
            M_out[random_idx] = last_M_del;
            *M_tot_pt += last_M_del;
        }
    }
}

/* Creates a realisation of halo properties by sampling the halo mass function and
 * conditional property PDFs, Sampling is done until there is no more mass in the condition
 * Stochasticity is ignored below a certain mass threshold */
int stoc_mass_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    double exp_M = hs_constants->expected_M;

    //The mass-limited sampling as-is has a slight bias to producing too many halos,
    //  which is independent of density or halo mass,
    //  this factor reduces the total expected mass to bring it into line with the CMF
    exp_M *= user_params_global->HALOMASS_CORRECTION;

    int n_halo_sampled=0;
    double M_prog=0;
    double M_sample;

    double tbl_arg = hs_constants->cond_val;

    while(M_prog < exp_M){
        M_sample = sample_dndM_inverse(tbl_arg,hs_constants,rng);

        M_prog += M_sample;
        M_out[n_halo_sampled++] = M_sample;
    }
    //The above sample is above the expected mass, by up to 100%. I wish to make the average mass equal to exp_M
    fix_mass_sample(rng,exp_M,&n_halo_sampled,&M_prog,M_out);

    *n_halo_out = n_halo_sampled;
    return 0;
}

bool partition_rejection(double sigma_m, double sigma_min, double sigma_cond, double del_c, double growthf, gsl_rng * rng){
    //no rejection in EPS
    double test1,test2,randval;
    if(user_params_global->HMF == 0){
        return false;
    }
    else if(user_params_global->HMF == 1){
        test1 = st_taylor_factor(sigma_m,sigma_cond,growthf,NULL) - del_c; //maximum barrier term in mass range
        test2 = st_taylor_factor(sigma_min,sigma_cond,growthf,NULL) - del_c;
        randval = gsl_rng_uniform(rng);
        return randval > (test2/test1);
    }
    else{
        LOG_ERROR("Partition sampling currently only works using EPS or SMT CMF");
        Throw(ValueError);
    }
}

//Sheth & Lemson 1999 partition model, Changes delta,M in the CMF as you sample
//This avoids discretization issues in the mass sampling, however.....
//it has been noted to overproduce small halos (McQuinn+ 2007)
//I don't know why sampling from Fcoll(M) is correct?
//  Do we not sample the same `particle` multiple times? (i.e 10x more samples for a single 10x mass halo)
//  How does the reduction of mass after each sample *exactly* cancel this 1/M effect
//If you want a non-barrier-based CMF, I don't know how to implement it here
int stoc_partition_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    //lnMmin only used for sampling, apply factor here
    double exp_M = hs_constants->expected_M;
    double M_cond = hs_constants->M_cond;
    double d_cond = hs_constants->delta;
    double growthf = hs_constants->growth_out;
    double sigma_min = hs_constants->sigma_min;

    int n_halo_sampled;
    double nu_sample, sigma_sample, M_sample, M_remaining, delta_current;
    double lnM_remaining, sigma_r, del_term;

    double tbl_arg = hs_constants->cond_val;
    n_halo_sampled = 0;
    double nu_fudge_factor = user_params_global->HALOMASS_CORRECTION;

    //set initial amount
    // M_remaining = M_cond; // full condition
    M_remaining = exp_M; //subtract unresolved mass
    lnM_remaining = log(M_remaining);

    double nu_min;
    while(M_remaining > user_params_global->SAMPLER_MIN_MASS){
        sigma_r = EvaluateSigma(lnM_remaining);

        delta_current = (get_delta_crit(user_params_global->HMF,sigma_r,growthf) - d_cond)/(M_remaining/M_cond);
        del_term = delta_current*delta_current/growthf/growthf;

        nu_min = sqrt(del_term/(sigma_min*sigma_min - sigma_r*sigma_r)); //nu at minimum progenitor

        //we use the gaussian tail distribution to enforce our Mmin limit from the sigma tables
        do{
            nu_sample = gsl_ran_ugaussian_tail(rng,nu_min)*nu_fudge_factor;
            sigma_sample = sqrt(del_term/(nu_sample*nu_sample) + sigma_r*sigma_r);
        } while(partition_rejection(sigma_sample,sigma_min,sigma_r,delta_current/growthf,growthf,rng));

        M_sample = EvaluateSigmaInverse(sigma_sample);
        M_sample = exp(M_sample);

        M_out[n_halo_sampled++] = M_sample;
        M_remaining -= M_sample;
        lnM_remaining = log(M_remaining);
    }

    *n_halo_out = n_halo_sampled;
    return 0;
}

double ComputeFraction_split(
    double sigma_start, double sigmasq_start, double sigmasq_res,
    double G1, double dd, double gamma1
) {
    double u_res = sigma_start*pow(sigmasq_res - sigmasq_start, -.5);
    return sqrt(2./PI)*EvaluateJ(u_res,gamma1)*G1/sigma_start*dd;
}

//binary splitting with small internal steps based on Parkinson+08, Bensen+16, Qiu+20 (Darkforest)
//This code was modified from the tree generation function in Darkforest (Qiu et al 2020. ArXiv: 2007.14624)
int stoc_split_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    //define constants
    double G0 = user_params_global->PARKINSON_G0;
    double gamma1 = user_params_global->PARKINSON_y1;
    double gamma2 = user_params_global->PARKINSON_y2;
    double m_res = hs_constants->M_min;
    double lnm_res = hs_constants->lnM_min;
    double eps1 = 0.1;
    double eps2 = 0.1;

    //declare intermediate variables
    double mu, G1, G2;
    double d_start, d_target;
    double q_res;
    double m_start, m_half, lnm_half, lnm_start;
    double sigma_start, sigma_half, sigma_res;
    double sigmasq_start, sigmasq_half, sigmasq_res;
    double alpha_half,alpha_q;
    double V_res, V_half;
    double B, beta, eta, pow_diff;
    double dd1, dd2, dd, dd_target, dN_dd, N_upper, F;
    double q, m_q, sigma_q, sigmasq_q, R_q, factor1, factor2;
    double m_prog1, m_prog2;
    int save;

    float d_points[MAX_HALO_CELL], m_points[MAX_HALO_CELL];

    //set initial points
    d_points[0] = Deltac / hs_constants->growth_in;
    m_points[0] = hs_constants->M_cond;
    d_target = Deltac / hs_constants->growth_out;

    //counters for total mass, number at target z, active index, and total number in sub-tree
    double M_total = 0;
    int n_out = 0;
    int idx = 0;
    int n_points = 1;

    sigma_res = EvaluateSigma(lnm_res);
    sigmasq_res = sigma_res*sigma_res;
    while(idx < n_points){
        //define the starting condition
        d_start = d_points[idx];
        m_start = m_points[idx];
        lnm_start = log(m_start);
        dd_target = d_target - d_start;
        save = 0;

        // Compute useful quantites
        m_half = 0.5*m_start;
        lnm_half = log(m_half);
        sigma_start = EvaluateSigma(lnm_start);
        sigmasq_start = sigma_start*sigma_start;
        sigma_half = EvaluateSigma(lnm_half);
        sigmasq_half = sigma_half*sigma_half;

        G1 = G0*pow(d_start/sigma_start, gamma2);
        q_res = m_res/m_start;
        q = 0.;
        if (q_res >= 0.5){
            //No split is possible within our resolution,
            //  so we only consider the timestep restriction to make the EPS limit valid
            //  and take away the average mass below the resolution
            dd = eps1*sqrt(2)*sqrt(sigmasq_half - sigmasq_start);
            if(dd >= dd_target){
                dd = dd_target;
                save = 1;
            }
            F = ComputeFraction_split(sigma_start, sigmasq_start, sigmasq_res, G1, dd, gamma1);
        }
        else{
            alpha_half = EvaluatedSigmasqdm(lnm_half); //d(sigma^2)/dm
            alpha_half = -m_half/(2*sigma_half*sigma_half)*alpha_half; //-d(lnsigma)/d(lnm)
            // Compute B and beta
            V_res = sigmasq_res*pow(sigmasq_res - sigmasq_start, -1.5);
            V_half = sigmasq_half*pow(sigmasq_half - sigmasq_start, -1.5);
            beta = log(V_res/V_half)/log(2.*q_res);
            B = pow(2., beta)*V_half;

            // Compute ddelta1, the timestep limit ensuring the exponent in the EPS MF is small (limit as time -> 0 is valid)
            dd1 = eps1*sqrt(2)*sqrt(sigmasq_half - sigmasq_start);

            // Compute ddelta2, the timestep limit ensuring the assumption of maximum 1 split is valid
            mu = gamma1 < 0. ? -log(sigma_res/sigma_half)/log(2.*q_res) : alpha_half;
            eta = beta - 1 - gamma1*mu;
            pow_diff = pow(.5, eta) - pow(q_res, eta);
            G2 = G1*pow(sigma_half/sigma_start, gamma1)*pow(0.5, mu*gamma1);
            dN_dd = sqrt(2./PI)*B*pow_diff/eta*alpha_half*G2; //this is the number of progenitors expected per unit increase in the barrier
            dd2 = eps2/dN_dd; //barrier change which results in average of at most eps2 progenitors

            // Choose the minimum of the two timestep limits
            dd = fmin(dd1,dd2);
            if(dd >= dd_target){
                dd = dd_target;
                save = 1;
            }

            N_upper = dN_dd*dd;
            // Compute F
            F = ComputeFraction_split(sigma_start, sigmasq_start, sigmasq_res, G1, dd, gamma1);
            // Generate random numbers and split the tree
            if (gsl_rng_uniform(rng) < N_upper) {
                q = pow(pow(q_res, eta) + pow_diff*gsl_rng_uniform(rng), 1./eta);
                m_q = q*m_start;
                sigma_q = EvaluateSigma(log(m_q));
                alpha_q = EvaluatedSigmasqdm(log(m_q)); //d(sigma^2)/dm
                alpha_q = -m_q/(2*sigma_q*sigma_q)*alpha_q; //-d(lnsigma)/d(lnm)
                sigmasq_q = sigma_q*sigma_q;

                factor1 = alpha_q/alpha_half;
                factor2 = sigmasq_q*pow(sigmasq_q - sigmasq_start, -1.5)/(B*pow(q, beta));
                R_q = factor1*factor2;
                if (gsl_rng_uniform(rng) > R_q)
                    q = 0.; // No split
            }
        }
        // LOG_ULTRA_DEBUG("split i %d n %d m %.2e d %.2e",idx,n_points,m_start,d_start);
        // LOG_ULTRA_DEBUG("q %.2e F %.2e dd %.2e (%.2e %.2e) of %.2e",q,F,dd,dd1,dd2,dd_target);
        // LOG_ULTRA_DEBUG("dNdd %.2e B %.2e pow %.2e eta %.2e ah %.2e G2 %.2e b %.2e",dN_dd,B,pow_diff,eta,alpha_half,G2,beta);

        // Compute progenitor mass from fraction q, always subtract the below-resolution mass from the largest
        m_prog1 = (1 - F - q)*m_start;
        m_prog2 = q*m_start;
        //if this branch is finished, add to the output array
        if (save) {
            if (m_prog1 > m_res) {
                M_out[n_out++] = m_prog1;
                M_total += m_prog1;
            }
            if (m_prog2 > m_res) {
                M_out[n_out++] = m_prog2;
                M_total += m_prog2;
            }
        }
        //if not finished yet, add them to the internal arrays
        //We don't need the point at idx anymore, so we can put the first progenitor
        //at the start point, and the second at the end
        //since the first is always more massive, this saves memory
        //NOTE: this still drifts by the number of saved halos, figure out how to
        //   keep the active halo at zero until the end, but that's minor as it should only drift a few dozen
        else {
            if (m_prog1 > m_res){
                //replace current halo with the larger progenitor
                d_points[idx] = dd + d_start;
                m_points[idx] = m_prog1;
                //since we replaced, do not advance the index
                idx--;
            }
            if (m_prog2 > m_res){
                //add the smaller progenitor to the end
                d_points[n_points] = dd + d_start;
                m_points[n_points++] = m_prog2;
            }
        }
        idx++;
    }
    *n_halo_out = n_out;
    // LOG_ULTRA_DEBUG("Total M = %.4e (%.4e) N = %d",M_total,M_total/hs_constants->M_cond,n_out);
    return 0;
}

int stoc_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    //TODO: really examine the case for number/mass sampling
    //The poisson sample fails spectacularly for high delta (from_catalogs or dense cells)
    //  and excludes the correlation between number and mass (e.g many small halos or few large ones)
    //The mass sample underperforms at low exp_M/M_max by excluding stochasticity in the total collapsed fraction
    //  and excluding larger halos (e.g if exp_M is 0.1*M_max we can effectively never sample the large halos)
    //i.e there is some case for a delta cut between these two methods however I have no intuition for the exact levels

    int err;
    //If the expected mass is below our minimum saved mass, don't bother calculating
    //NOTE: some of these conditions are redundant with set_consts_cond()
    if(hs_constants->delta <= DELTA_MIN || hs_constants->expected_M < user_params_global->SAMPLER_MIN_MASS){
        *n_halo_out = 0;
        return 0;
    }
    //if delta is above critical, form one big halo
    if(hs_constants->delta >= MAX_DELTAC_FRAC*get_delta_crit(user_params_global->HMF,hs_constants->sigma_cond,hs_constants->growth_out)){
        *n_halo_out = 1;

        //Expected mass takes into account potential dexm overlap
        M_out[0] = hs_constants->expected_M;
        return 0;
    }

    //We always use Number-Limited sampling for grid-based cases
    if(user_params_global->SAMPLE_METHOD == 1 || !hs_constants->from_catalog){
        err = stoc_halo_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(user_params_global->SAMPLE_METHOD == 0){
        err = stoc_mass_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(user_params_global->SAMPLE_METHOD == 2){
        err = stoc_partition_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(user_params_global->SAMPLE_METHOD == 3){
        err = stoc_split_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else{
        LOG_ERROR("Invalid sampling method");
        Throw(ValueError);
    }
    if(*n_halo_out > MAX_HALO_CELL){
        LOG_ERROR("too many halos in condition, buffer overflow");
        Throw(ValueError);
    }
    return err;
}

// will have to add properties here and output grids, instead of in perturbed
int sample_halo_grids(gsl_rng **rng_arr, double redshift, float *dens_field, float *halo_overlap_box, HaloField *halofield_large, HaloField *halofield_out, struct HaloSamplingConstants *hs_constants){
    int lo_dim = user_params_global->HII_DIM;
    double boxlen = user_params_global->BOX_LEN;

    double Mcell = hs_constants->M_cond;
    double Mmin = hs_constants->M_min;
    double lnMmin = hs_constants->lnM_min;
    double growthf = hs_constants->growth_out;

    unsigned long long int nhalo_in = halofield_large->n_halos;
    unsigned long long int nhalo_threads[user_params_global->N_THREADS];
    unsigned long long int istart_threads[user_params_global->N_THREADS];

    unsigned long long int arraysize_total = halofield_out->buffer_size;
    unsigned long long int arraysize_local = arraysize_total / user_params_global->N_THREADS;

    LOG_DEBUG("Beginning stochastic halo sampling on %d ^3 grid",lo_dim);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e,volume = %.3e, D = %.3e",redshift,Mmin,Mcell,Mcell/RHOcrit/cosmo_params_global->OMm,growthf);
    LOG_DEBUG("Total Array Size %llu, array size per thread %llu (~%.3e GB total)",arraysize_total,arraysize_local,6.*arraysize_total*sizeof(int)/1e9);

    double total_volume_excluded=0.;
    double total_volume_dexm=0.;
    double cell_volume = VOLUME / pow((double)user_params_global->HII_DIM,3);

#pragma omp parallel num_threads(user_params_global->N_THREADS)
    {
        //PRIVATE VARIABLES
        int x,y,z,i;
        unsigned long long int halo_idx;
        int threadnum = omp_get_thread_num();

        int nh_buf;
        double delta;
        double prop_buf[3], prop_dummy[3];
        int crd_hi[3];

        double mass_defc;

        //buffer per cell
        float hm_buf[MAX_HALO_CELL];

        unsigned long long int count=0;
        unsigned long long int istart = threadnum * arraysize_local;
        //debug total
        double M_tot_cell=0.;

        //we need a private version
        struct HaloSamplingConstants hs_constants_priv;
        hs_constants_priv = *hs_constants;

        //assign big halos into list first (split amongst ranks)
#pragma omp for reduction(+:total_volume_dexm)
        for (halo_idx=0;halo_idx<nhalo_in;halo_idx++){
            halofield_out->halo_masses[istart+count] = halofield_large->halo_masses[halo_idx];
            halofield_out->star_rng[istart+count] = halofield_large->star_rng[halo_idx];
            halofield_out->sfr_rng[istart+count] = halofield_large->sfr_rng[halo_idx];
            halofield_out->halo_coords[0 + 3*(istart+count)] = halofield_large->halo_coords[0 + 3*halo_idx];
            halofield_out->halo_coords[1 + 3*(istart+count)] = halofield_large->halo_coords[1 + 3*halo_idx];
            halofield_out->halo_coords[2 + 3*(istart+count)] = halofield_large->halo_coords[2 + 3*halo_idx];

            total_volume_dexm += halofield_large->halo_masses[halo_idx] / (RHOcrit * cosmo_params_global->OMm * cell_volume);
            count++;
        }

#pragma omp for reduction(+:total_volume_excluded)
        for (x=0; x<lo_dim; x++){
            for (y=0; y<lo_dim; y++){
                for (z=0; z<HII_D_PARA; z++){
                    delta = dens_field[HII_R_INDEX(x,y,z)] * growthf;
                    stoc_set_consts_cond(&hs_constants_priv,delta);
                    if((x+y+z) == 0){
                        print_hs_consts(&hs_constants_priv);
                    }

                    mass_defc = halo_overlap_box[HII_R_INDEX(x,y,z)];
                    total_volume_excluded += mass_defc;

                    hs_constants_priv.expected_M *= (1.-mass_defc);
                    hs_constants_priv.expected_N *= (1.-mass_defc);

                    stoc_sample(&hs_constants_priv, rng_arr[threadnum], &nh_buf, hm_buf);
                    //output total halo number, catalogues of masses and positions
                    M_tot_cell = 0;
                    for(i=0;i<nh_buf;i++){
                        //sometimes halos are subtracted from the sample (set to zero)
                        //we do not want to save these
                        if(hm_buf[i] < user_params_global->SAMPLER_MIN_MASS) continue;

                        if(count >= arraysize_local){
                            LOG_ERROR("More than %llu halos (expected %.1e) with buffer size factor %.1f",
                                        arraysize_local,arraysize_local/user_params_global->MAXHALO_FACTOR,user_params_global->MAXHALO_FACTOR);
                            LOG_ERROR("If you expected to have an above average halo number try raising user_params->MAXHALO_FACTOR");
                            Throw(ValueError);
                        }

                        set_prop_rng(rng_arr[threadnum], 0, prop_dummy, prop_dummy, prop_buf);
                        place_on_hires_grid(x,y,z,crd_hi,rng_arr[threadnum]);

                        halofield_out->halo_masses[istart + count] = hm_buf[i];
                        halofield_out->halo_coords[3*(istart + count) + 0] = crd_hi[0];
                        halofield_out->halo_coords[3*(istart + count) + 1] = crd_hi[1];
                        halofield_out->halo_coords[3*(istart + count) + 2] = crd_hi[2];

                        halofield_out->star_rng[istart + count] = prop_buf[0];
                        halofield_out->sfr_rng[istart + count] = prop_buf[1];
                        halofield_out->xray_rng[istart + count] = prop_buf[2];
                        count++;

                        M_tot_cell += hm_buf[i];
                        if((x+y+z) == 0){
                            LOG_ULTRA_DEBUG("Halo %d Mass %.2e Stellar %.2e SFR %.2e",i,hm_buf[i],prop_buf[0],prop_buf[1]);
                        }
                    }
                    if((x+y+z) == 0){
                        LOG_SUPER_DEBUG("Cell (%d %d %d): delta %.2f | N %d (exp. %.2e) | Total M %.2e (exp. %.2e) overlap defc %.2f",x,y,z,
                                        delta,nh_buf,hs_constants_priv.expected_N,M_tot_cell,hs_constants_priv.expected_M,mass_defc);
                    }
                }
            }
        }
        LOG_SUPER_DEBUG("Thread %d found %llu halos",threadnum,count);

        istart_threads[threadnum] = istart;
        nhalo_threads[threadnum] = count;
    }

    LOG_SUPER_DEBUG("Total dexm volume %.6e Total volume excluded %.6e (In units of HII_DIM cells)",total_volume_dexm,total_volume_excluded);

    //Condense the sparse array (serial)
    int i=0;
    unsigned long long int count_total = 0;
    for(i=0;i<user_params_global->N_THREADS;i++){
        memmove(&halofield_out->halo_masses[count_total],&halofield_out->halo_masses[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->star_rng[count_total],&halofield_out->star_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->sfr_rng[count_total],&halofield_out->sfr_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->xray_rng[count_total],&halofield_out->xray_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->halo_coords[3*count_total],&halofield_out->halo_coords[3*istart_threads[i]],sizeof(int)*3*nhalo_threads[i]);
        LOG_SUPER_DEBUG("Moved array (start,count) (%llu, %llu) to position %llu",istart_threads[i],nhalo_threads[i],count_total);
        count_total += nhalo_threads[i];
    }
    halofield_out->n_halos = count_total;

    //replace the rest with zeros for clarity
    memset(&halofield_out->halo_masses[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->halo_coords[3*count_total],0,3*(arraysize_total-count_total)*sizeof(int));
    memset(&halofield_out->star_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->sfr_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->xray_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    LOG_SUPER_DEBUG("Set %llu elements beyond %llu to zero",arraysize_total-count_total,count_total);
    return 0;
}

//NOTE: there's a lot of repeated code here and in build_halo_cats, find a way to merge
int sample_halo_progenitors(gsl_rng ** rng_arr, double z_in, double z_out, HaloField *halofield_in,
                             HaloField *halofield_out, struct HaloSamplingConstants *hs_constants){
    if(z_in >= z_out){
        LOG_ERROR("halo progenitors must go backwards in time!!! z_in = %.1f, z_out = %.1f",z_in,z_out);
        Throw(ValueError);
    }

    double growth_in = hs_constants->growth_in;
    double growth_out = hs_constants->growth_out;
    int lo_dim = user_params_global->HII_DIM;
    int hi_dim = user_params_global->DIM;
    double boxlen = user_params_global->BOX_LEN;
    //cell size for smoothing / CMF calculation
    double lnMmax_tb = hs_constants->lnM_max_tb;
    double Mmax_tb = hs_constants->M_max_tables;
    double Mmin = hs_constants->M_min;
    double lnMmin = hs_constants->lnM_min;
    double delta = hs_constants->delta;

    unsigned long long int nhalo_in = halofield_in->n_halos;
    unsigned long long int nhalo_threads[user_params_global->N_THREADS];
    unsigned long long int istart_threads[user_params_global->N_THREADS];

    unsigned long long int arraysize_total = halofield_out->buffer_size;
    unsigned long long int arraysize_local = arraysize_total / user_params_global->N_THREADS;

    LOG_DEBUG("Beginning stochastic halo sampling of progenitors on %llu halos",nhalo_in);
    LOG_DEBUG("z = %f, Mmin = %e, d = %.3e",z_out,Mmin,delta);
    LOG_DEBUG("Total Array Size %llu, array size per thread %llu (~%.3e GB total)",arraysize_total,arraysize_local,6.*arraysize_total*sizeof(int)/1e9);

    double corr_arr[3] = {hs_constants->corr_star,hs_constants->corr_sfr,hs_constants->corr_xray};

#pragma omp parallel num_threads(user_params_global->N_THREADS)
    {
        float prog_buf[MAX_HALO_CELL];
        int n_prog;
        double M_prog;

        double propbuf_in[3];
        double propbuf_out[3];

        int threadnum = omp_get_thread_num();
        double M2;
        int jj;
        unsigned long long int ii;
        unsigned long long int count=0;
        unsigned long long int istart = threadnum * arraysize_local;

        //we need a private version
        //also the naming convention should be better between structs/struct pointers
        struct HaloSamplingConstants hs_constants_priv;
        hs_constants_priv = *hs_constants;

#pragma omp for
        for(ii=0;ii<nhalo_in;ii++){
            M2 = halofield_in->halo_masses[ii];
            if(M2 < Mmin || M2 > Mmax_tb){
                LOG_ERROR("Input Mass = %.2e at %llu of %llu, something went wrong in the input catalogue",M2,ii,nhalo_in);
                Throw(ValueError);
            }
            //set condition-dependent variables for sampling
            stoc_set_consts_cond(&hs_constants_priv,M2);

            //Sample the CMF set by the descendant
            stoc_sample(&hs_constants_priv,rng_arr[threadnum],&n_prog,prog_buf);

            propbuf_in[0] = halofield_in->star_rng[ii];
            propbuf_in[1] = halofield_in->sfr_rng[ii];
            propbuf_in[2] = halofield_in->xray_rng[ii];

            //place progenitors in local list
            M_prog = 0;
            for(jj=0;jj<n_prog;jj++){
                //sometimes halos are subtracted from the sample (set to zero)
                //we do not want to save these
                if(prog_buf[jj] < user_params_global->SAMPLER_MIN_MASS) continue;

                if(count >= arraysize_local){
                    LOG_ERROR("More than %llu halos (expected %.1e) with buffer size factor %.1f",
                                arraysize_local,arraysize_local/user_params_global->MAXHALO_FACTOR,user_params_global->MAXHALO_FACTOR);
                    LOG_ERROR("If you expected to have an above average halo number try raising user_params_global->MAXHALO_FACTOR");
                    Throw(ValueError);
                }

                set_prop_rng(rng_arr[threadnum], 1, corr_arr, propbuf_in, propbuf_out);

                halofield_out->halo_masses[istart + count] = prog_buf[jj];
                halofield_out->halo_coords[3*(istart + count) + 0] = halofield_in->halo_coords[3*ii+0];
                halofield_out->halo_coords[3*(istart + count) + 1] = halofield_in->halo_coords[3*ii+1];
                halofield_out->halo_coords[3*(istart + count) + 2] = halofield_in->halo_coords[3*ii+2];

                halofield_out->star_rng[istart + count] = propbuf_out[0];
                halofield_out->sfr_rng[istart + count] = propbuf_out[1];
                halofield_out->xray_rng[istart + count] = propbuf_out[2];
                count++;

                if(ii==0){
                    M_prog += prog_buf[jj];
                    LOG_ULTRA_DEBUG("First Halo Prog %d: Mass %.2e Stellar %.2e SFR %.2e e_d %.3f",jj,prog_buf[jj],propbuf_out[0],propbuf_out[1],Deltac*growth_out/growth_in);
                }
            }
            if(ii==0){
                LOG_ULTRA_DEBUG(" HMF %d delta %.3f delta_coll %.3f delta_desc %.3f adjusted %.3f",user_params_global->HMF,
                                                                                hs_constants_priv.delta,
                                                                                get_delta_crit(user_params_global->HMF,hs_constants_priv.sigma_cond,growth_out),
                                                                                get_delta_crit(user_params_global->HMF,hs_constants_priv.sigma_cond,growth_in),
                                                                                get_delta_crit(user_params_global->HMF,hs_constants_priv.sigma_cond,growth_in)*growth_out/growth_in);
                print_hs_consts(&hs_constants_priv);
                LOG_SUPER_DEBUG("First Halo: Mass %.2f | N %d (exp. %.2e) | Total M %.2e (exp. %.2e)",
                                        M2,n_prog,hs_constants_priv.expected_N,M_prog,hs_constants_priv.expected_M);
            }
        }
        istart_threads[threadnum] = istart;
        nhalo_threads[threadnum] = count;
    }

    //Condense the sparse array
    int i=0;
    unsigned long long int count_total = 0;
    for(i=0;i<user_params_global->N_THREADS;i++){
        LOG_SUPER_DEBUG("Thread %d found %llu Halos",i,nhalo_threads[i]);
        memmove(&halofield_out->halo_masses[count_total],&halofield_out->halo_masses[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->star_rng[count_total],&halofield_out->star_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->sfr_rng[count_total],&halofield_out->sfr_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->halo_coords[3*count_total],&halofield_out->halo_coords[3*istart_threads[i]],sizeof(int)*3*nhalo_threads[i]);
        count_total += nhalo_threads[i];
    }
    //replace the rest with zeros for clarity
    memset(&halofield_out->halo_masses[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->halo_coords[3*count_total],0,3*(arraysize_total-count_total)*sizeof(int));
    memset(&halofield_out->star_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->sfr_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    halofield_out->n_halos = count_total;

    return 0;
}

//function that talks between the structures (Python objects) and the sampling functions
int stochastic_halofield(UserParams *user_params, CosmoParams *cosmo_params,
                         AstroParams *astro_params, FlagOptions *flag_options,
                         unsigned long long int seed, float redshift_desc, float redshift,
                         float *dens_field, float *halo_overlap_box, HaloField *halos_desc, HaloField *halos){
    Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);

    int n_halo_stoc;
    int i_start,i;

    if(redshift_desc > 0 && halos_desc->n_halos == 0){
        LOG_DEBUG("No halos to sample from redshifts %.2f to %.2f, continuing...",redshift_desc,redshift);
        return 0;
    }

    //set up the rng
    gsl_rng * rng_stoc[user_params->N_THREADS];
    seed_rng_threads(rng_stoc,seed);

    struct HaloSamplingConstants hs_constants;
    stoc_set_consts_z(&hs_constants,redshift,redshift_desc);

    //Fill them
    //NOTE:Halos prev in the first box corresponds to the large DexM halos
    if(redshift_desc < 0.){
        LOG_DEBUG("building first halo field at z=%.1f", redshift);
        sample_halo_grids(rng_stoc,redshift,dens_field,halo_overlap_box,halos_desc,halos,&hs_constants);
    }
    else{
        LOG_DEBUG("Calculating halo progenitors from z=%.1f to z=%.1f | %llu", redshift_desc,redshift,halos_desc->n_halos);
        sample_halo_progenitors(rng_stoc,redshift_desc,redshift,halos_desc,halos,&hs_constants);
    }

    LOG_DEBUG("Found %llu Halos", halos->n_halos);

    if(halos->n_halos > 3){
        LOG_DEBUG("First few Masses:  %11.3e %11.3e %11.3e",halos->halo_masses[0],halos->halo_masses[1],halos->halo_masses[2]);
        LOG_DEBUG("First few Stellar RNG: %11.3e %11.3e %11.3e",halos->star_rng[0],halos->star_rng[1],halos->star_rng[2]);
        LOG_DEBUG("First few SFR RNG:     %11.3e %11.3e %11.3e",halos->sfr_rng[0],halos->sfr_rng[1],halos->sfr_rng[2]);
    }

    if(user_params_global->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
    }
    free_dNdM_tables();

    free_rng_threads(rng_stoc);
    LOG_DEBUG("Done.");
    return 0;
}

//This is a test function which takes a list of conditions (cells or halos) and samples them to produce a descendant list
//      as well as per-condition number and mass counts
int single_test_sample(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options,
                        unsigned long long int seed, int n_condition, float *conditions, int *cond_crd, double z_out, double z_in,
                        int *out_n_tot, int *out_n_cell, double *out_n_exp,
                        double *out_m_cell, double *out_m_exp, float *out_halo_masses, int *out_halo_coords){
    int status;
    Try{
        //make the global structs
        Broadcast_struct_global_all(user_params,cosmo_params,astro_params,flag_options);

        omp_set_num_threads(user_params->N_THREADS);

        //set up the rng
        gsl_rng * rng_stoc[user_params->N_THREADS];
        seed_rng_threads(rng_stoc,seed);

        if(z_in > 0 && z_out <= z_in){
            LOG_DEBUG("progenitor sampling must go back in time z_out=%.2f z_in=%.2f",z_out,z_in);
            Throw(ValueError);
        }
        int i,j;

        struct HaloSamplingConstants hs_const_struct;
        struct HaloSamplingConstants *hs_constants = &hs_const_struct;

        LOG_DEBUG("Setting z constants. %.3f %.3f",z_out,z_in);
        stoc_set_consts_z(hs_constants,z_out,z_in);

        LOG_DEBUG("SINGLE SAMPLE: z = (%.2f,%.2f), Mmin = %.3e, cond(%d)=[%.2e,%.2e,%.2e...]",z_out,z_in,hs_constants->M_min,
                                                                        n_condition,conditions[0],conditions[1],conditions[2]);

        //halo catalogues + cell sums from multiple conditions, given M as cell descendant halos/cells
        //the result mapping is n_halo_total (1) (exp_n,exp_m,n_prog,m_prog) (n_desc) M_cat (n_prog_total)
        int n_halo_tot=0;
        #pragma omp parallel num_threads(user_params->N_THREADS) private(i,j)
        {
            float out_hm[MAX_HALO_CELL];
            double M_prog;
            int out_crd[3];
            int n_halo,n_halo_cond;
            double cond;
            //we need a private version
            struct HaloSamplingConstants hs_constants_priv;
            hs_constants_priv = *hs_constants;
            #pragma omp for
            for(j=0;j<n_condition;j++){
                cond = conditions[j];
                stoc_set_consts_cond(&hs_constants_priv,cond);
                if(j==0) print_hs_consts(&hs_constants_priv);
                stoc_sample(&hs_constants_priv, rng_stoc[omp_get_thread_num()], &n_halo, out_hm);

                n_halo_cond = 0;
                M_prog = 0;
                for(i=0;i<n_halo;i++){
                    if(out_hm[i] < user_params->SAMPLER_MIN_MASS) continue;
                    M_prog += out_hm[i];
                    n_halo_cond++;

                    //critical is bad, but this is a test function so it doesn't matter much
                    #pragma omp critical
                    {
                        out_halo_masses[n_halo_tot] = out_hm[i];
                        if(hs_constants_priv.from_catalog){
                            out_halo_coords[3*n_halo_tot + 0] = cond_crd[3*j+0];
                            out_halo_coords[3*n_halo_tot + 1] = cond_crd[3*j+1];
                            out_halo_coords[3*n_halo_tot + 2] = cond_crd[3*j+2];
                        }
                        else{
                            place_on_hires_grid(cond_crd[3*j+0],cond_crd[3*j+1],cond_crd[3*j+2],out_crd,rng_stoc[omp_get_thread_num()]);
                            out_halo_coords[3*n_halo_tot + 0] = out_crd[0];
                            out_halo_coords[3*n_halo_tot + 1] = out_crd[1];
                            out_halo_coords[3*n_halo_tot + 2] = out_crd[2];
                        }
                        n_halo_tot++;
                    }
                }
                //output descendant statistics
                out_n_cell[j] = n_halo_cond;
                out_m_cell[j] = M_prog;
            }
        }

        out_n_tot[0] = n_halo_tot;

        //get expected values from the saved mass range
        if(hs_constants->from_catalog){
            initialise_dNdM_tables(log(user_params->SAMPLER_MIN_MASS), hs_constants->lnM_max_tb, log(user_params->SAMPLER_MIN_MASS),
                                 hs_constants->lnM_max_tb, hs_constants->growth_out, hs_constants->growth_in, true);
        }
        else{
            double delta_crit = get_delta_crit(user_params_global->HMF,hs_constants->sigma_cond,hs_constants->growth_out);
            initialise_dNdM_tables(DELTA_MIN, MAX_DELTAC_FRAC*delta_crit, log(user_params->SAMPLER_MIN_MASS), hs_constants->lnM_max_tb,
                                 hs_constants->growth_out, hs_constants->lnM_cond, false);
        }
        #pragma omp parallel
        {
            struct HaloSamplingConstants hs_constants_exp;
            hs_constants_exp = *hs_constants;
            double cond;

            #pragma omp for
            for(j=0;j<n_condition;j++){
                cond = conditions[j];
                stoc_set_consts_cond(&hs_constants_exp,cond);
                out_n_exp[j] = hs_constants_exp.expected_N;
                out_m_exp[j] = hs_constants_exp.expected_M;
            }
        }

        if(user_params_global->USE_INTERPOLATION_TABLES){
            freeSigmaMInterpTable();
        }
        free_dNdM_tables();

        free_rng_threads(rng_stoc);
    } //end of try

    Catch(status){
        return(status);
    }
    LOG_DEBUG("Done.");
    return(0);
}
