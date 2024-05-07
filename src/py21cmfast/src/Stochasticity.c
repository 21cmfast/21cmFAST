/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//TODO: Don't have every error be a ValueError

//max number of attempts for mass tolerance before failure
#define MAX_ITERATIONS 1e4
#define MAX_ITER_N 1e2 //for stoc_halo_sample (select N halos) how many tries for one N, this should be large to enforce a near-possion p(N)
#define MMAX_TABLES 1e14

//buffer size (per cell of arbitrary size) in the sampling function
#define MAX_HALO_CELL (int)1e5

//NOTE: because the .c files are directly included in GenerateIC.c, the static doesn't really do anything :(
static struct AstroParams *astro_params_stoc;
static struct CosmoParams *cosmo_params_stoc;
static struct UserParams *user_params_stoc;
static struct FlagOptions *flag_options_stoc;

//parameters for the halo mass->stars calculations
//Note: ideally I would split this into constants set per snapshot and
//  constants set per condition, however some variables (delta or Mass)
//  can be set with differing frequencies depending on the condition type
struct HaloSamplingConstants{
    //calculated per redshift
    int from_catalog; //flag for first box or updating halos
    double corr_sfr;
    double corr_star;

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

//The sigma interp table is regular in log mass, not sigma so we need to loop ONLY FOR METHOD=3
double EvaluateSigmaInverse(double sigma, struct RGTable1D_f *s_table){
    int idx;
    for(idx=0;idx<NMass;idx++){
        if(sigma < s_table->y_arr[idx]) break;
    }
    if(idx == NMass){
        LOG_ERROR("sigma inverse out of bounds.");
        Throw(TableEvaluationError);
    }
    double table_val_0 = s_table->x_min + idx*s_table->x_width;
    double table_val_1 = s_table->x_min + (idx-1)*s_table->x_width;
    double interp_point = (sigma - table_val_0)/(table_val_1-table_val_0);

    return table_val_0*(1-interp_point) + table_val_1*(interp_point);
}

void Broadcast_struct_global_STOC(struct UserParams *user_params, struct CosmoParams *cosmo_params,struct AstroParams *astro_params, struct FlagOptions *flag_options){
    cosmo_params_stoc = cosmo_params;
    user_params_stoc = user_params;
    astro_params_stoc = astro_params;
    flag_options_stoc = flag_options;
}

//This function, designed to be used in the wrapper to estimate Halo catalogue size, takes the parameters and returns average number of halos within the entire box
double expected_nhalo(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions * flag_options){
    //minimum sampled mass
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    double M_min = global_params.SAMPLER_MIN_MASS;
    //maximum sampled mass
    double M_max = RHOcrit * cosmo_params->OMm * VOLUME / HII_TOT_NUM_PIXELS;
    double growthf = dicke(redshift);
    double result;

    init_ps();
    if(user_params->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(M_min,M_max);

    struct parameters_gsl_MF_integrals params = {
        .redshift = redshift,
        .growthf = growthf,
        .HMF = user_params->HMF,
    };

    result = IntegratedNdM(log(M_min), log(M_max), params, 1, 0) * VOLUME;
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

    const_struct->M_min = global_params.SAMPLER_MIN_MASS / global_params.SAMPLER_BUFFER_FACTOR;
    const_struct->lnM_min = log(const_struct->M_min);
    const_struct->M_max_tables = global_params.M_MAX_INTEGRAL;
    const_struct->lnM_max_tb = log(const_struct->M_max_tables);

    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        if(global_params.SAMPLE_METHOD == 3)
            initialiseSigmaMInterpTable(const_struct->M_min/2,const_struct->M_max_tables); //the binary split needs to go below the resolution
        else
            initialiseSigmaMInterpTable(const_struct->M_min,const_struct->M_max_tables);
    }

    const_struct->sigma_min = EvaluateSigma(const_struct->lnM_min);

    if(redshift_desc >= 0){
        const_struct->growth_in = dicke(redshift_desc);
        if(astro_params_stoc->CORR_SFR > 0)
            const_struct->corr_sfr = exp(-(redshift - redshift_desc)/astro_params_stoc->CORR_SFR);
        else
            const_struct->corr_sfr = 0;
        if(astro_params_stoc->CORR_STAR > 0)
            const_struct->corr_star = exp(-(redshift - redshift_desc)/astro_params_stoc->CORR_STAR);
        else
            const_struct->corr_star = 0;

        const_struct->from_catalog = 1;
        initialise_dNdM_tables(log(global_params.SAMPLER_MIN_MASS), const_struct->lnM_max_tb, const_struct->lnM_min, const_struct->lnM_max_tb,
                                const_struct->growth_out, const_struct->growth_in, true);
        if(global_params.SAMPLE_METHOD < 2){
            initialise_dNdM_inverse_table(log(global_params.SAMPLER_MIN_MASS), const_struct->lnM_max_tb, const_struct->lnM_min,
                                    const_struct->growth_out, const_struct->growth_in, true);
        }
        if(global_params.SAMPLE_METHOD == 3){
            initialise_J_split_table(200,1e-4,20.,0.2);
        }
    }
    else {
        double M_cond = RHOcrit * cosmo_params_stoc->OMm * VOLUME / HII_TOT_NUM_PIXELS;
        const_struct->M_cond = M_cond;
        const_struct->lnM_cond = log(M_cond);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond);
        //for the table limits
        double delta_crit = get_delta_crit(user_params_stoc->HMF,const_struct->sigma_cond,const_struct->growth_out);
        const_struct->from_catalog = 0;
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
        const_struct->delta = get_delta_crit(user_params_stoc->HMF,const_struct->sigma_cond,const_struct->growth_in)\
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
    if(const_struct->delta > MAX_DELTAC_FRAC*get_delta_crit(user_params_stoc->HMF,const_struct->sigma_cond,const_struct->growth_out)){
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
    int lo_dim = user_params_stoc->HII_DIM;
    int hi_dim = user_params_stoc->DIM;
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
int set_prop_rng(gsl_rng *rng, int from_catalog, double *interp, float * input, float * output){
    //find log(property/variance) / mean
    double prop1 = gsl_ran_ugaussian(rng);
    double prop2 = gsl_ran_ugaussian(rng);

    //Correlate properties by interpolating between the sampled and descendant gaussians
    //THIS ASSUMES THAT THE SELF-CORRELATION IS IN THE LOG PROPRETY, NOT THE PROPERTY ITSELF
    //IF IT SHOULD BE IN LINEAR SPACE, EXPONENTIATE THE RANDOM VARIABLES
    if(from_catalog){
        prop1 = (1-interp[0])*prop1 + interp[0]*input[0];
        prop2 = (1-interp[1])*prop1 + interp[1]*input[1];
    }

    output[0] = prop1;
    output[1] = prop2;

    return;
}

//This is the function called to assign halo properties to an entire catalogue, used for DexM halos
int add_properties_cat(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options
                        , int seed, float redshift, struct HaloField *halos){
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);
    //set up the rng
    gsl_rng * rng_stoc[user_params->N_THREADS];
    seed_rng_threads(rng_stoc,seed);

    int nhalos = halos->n_halos;
    LOG_DEBUG("adding stars to %d halos",halos->n_halos);

    struct HaloSamplingConstants hs_constants;
    hs_constants.z_out = redshift;

    //loop through the halos and assign properties
    int i;
    float buf[2];
    //dummy
    float inbuf[2];
#pragma omp parallel for private(buf)
    for(i=0;i<nhalos;i++){
        // LOG_ULTRA_DEBUG("halo %d hm %.2e crd %d %d %d",i,halos->halo_masses[i],halos->halo_coords[3*i+0],halos->halo_coords[3*i+1],halos->halo_coords[3*i+2]);
        set_prop_rng(rng_stoc[omp_get_thread_num()], 0, inbuf, inbuf, buf);
        // LOG_ULTRA_DEBUG("stars %.2e sfr %.2e",buf[0],buf[1]);
        halos->star_rng[i] = buf[0];
        halos->sfr_rng[i] = buf[1];
    }

    free_rng_threads(rng_stoc);

    LOG_DEBUG("Done.");
    return 0;
}

/* Creates a realisation of halo properties by sampling the halo mass function and
 * conditional property PDFs, the number of halos is poisson sampled from the integrated CMF*/
int stoc_halo_sample(struct HaloSamplingConstants *hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    double M_min = hs_constants->M_min;
    double mass_tol = global_params.STOC_MASS_TOL;
    double exp_N = hs_constants->expected_N;
    double exp_M = hs_constants->expected_M;
    double M_cond = hs_constants->M_cond;

    double hm_sample, M_prog;
    int ii, nh;
    int n_attempts=0, n_failures=0;
    int halo_count=0;

    double tbl_arg = hs_constants->cond_val;

    for(n_failures=0;n_failures<MAX_ITERATIONS;n_failures++){
        n_attempts = 0;
        nh = gsl_ran_poisson(rng,exp_N);
        //find an N which *could* fit the mass tolerance
        while((nh*M_cond < exp_M*(1-mass_tol))||(M_min*nh > exp_M*(1+mass_tol))){
            nh = gsl_ran_poisson(rng,exp_N);
        }
        for(n_attempts=0;n_attempts<MAX_ITER_N;n_attempts++){
            M_prog = 0;
            halo_count = 0;
            for(ii=0;ii<nh;ii++){
                hm_sample = sample_dndM_inverse(tbl_arg,hs_constants,rng);
                M_prog += hm_sample;
                M_out[halo_count++] = hm_sample;
                // LOG_ULTRA_DEBUG("Sampled %.3e | %.3e",hm_sample,M_prog);
            }
            // LOG_ULTRA_DEBUG("attempt %d (%d halo) M=%.3e [%.3e, %.3e]",n_attempts,nh,M_prog,exp_M*(1-mass_tol),exp_M*(1+mass_tol));
            if((M_prog < exp_M*(1+mass_tol)) && (M_prog > exp_M*(1-mass_tol))){
                //using goto to break double loop
                goto found_halo_sample;
            }
        }
    }

    //technically I don't need the if statement but it might be confusing otherwise
    if(n_failures >= MAX_ITERATIONS){
        LOG_ERROR("passed max iter in sample, last attempt cond %.3e M=%.3e [%.3e, %.3e] Me %.3e Mt %.3e",tbl_arg,M_prog,exp_M*(1-mass_tol),exp_M*(1+mass_tol),exp_M,mass_tol);
        Throw(ValueError);
    }

    found_halo_sample: *n_halo_out = halo_count;
    // LOG_ULTRA_DEBUG("Got %d (exp. %.2e) halos mass %.2e (exp. %.2e) %.2f | (%d,%d) att.",
    //                 nh,exp_N,M_prog,exp_M,M_prog/exp_M - 1, n_failures, n_attempts);
    return 0;
}

double remove_random_halo(gsl_rng * rng, int n_halo, int *idx, double *M_prog, float *M_out){
    double last_M_del;
    int random_idx;
    do {
        random_idx = gsl_rng_uniform_int(rng,n_halo);
        //LOG_ULTRA_DEBUG("Want to remove halo %d of %d Mass %.3e",random_idx,M_out[random_idx],n_halo);
    } while(M_out[random_idx] == 0);
    last_M_del = M_out[random_idx];
    *M_prog -= last_M_del;
    M_out[random_idx] = 0; //zero mass halos are skipped and not counted

    *idx = random_idx;
    return last_M_del;
}

/*Convenience function to store all of the mass-based sample "corrections" i.e what we decide to do when the mass samples go above the trigger*/
//CURRENT IMPLEMENTATION: half the time I keep/throw away the last halo based on which sample is closer to the expected mass.
// However this introduces a bias since the last halo is likely larger than average So the other half the time,
// I throw away random halos until we are again below exp_M, effectively the same process in reverse. which has the (exact?) opposite bias
void fix_mass_sample(gsl_rng * rng, double exp_M, int *n_halo_pt, double *M_tot_pt, float *M_out){
    //Keep the last halo if it brings us closer to the expected mass
    //This is done by addition or subtraction over the limit to balance
    //the bias of the last halo being larger
    int random_idx;
    double last_M_del;
    bool sel = gsl_rng_uniform_int(rng,2);
    // int sel = 1;
    if(sel){
        //LOG_ULTRA_DEBUG("Deciding to keep last halo M %.3e tot %.3e exp %.3e",M_out[*n_halo_pt-1],*M_tot_pt,exp_M);
        if(fabs(*M_tot_pt - M_out[*n_halo_pt-1] - exp_M) < fabs(*M_tot_pt - exp_M)){
            //LOG_ULTRA_DEBUG("removed");
            *M_tot_pt -= M_out[*n_halo_pt-1];
            //here we remove by setting the counter one lower so it isn't read
            (*n_halo_pt)--; //increment has preference over dereference
        }
    }
    else{
        while(*M_tot_pt > exp_M){
            //here we remove by setting halo mass to zero, skipping it during the consolidation
            last_M_del = remove_random_halo(rng,*n_halo_pt,&random_idx,M_tot_pt,M_out);
            //LOG_ULTRA_DEBUG("Removed halo %d M %.3e tot %.3e",random_idx,last_M_del,*M_tot_pt);
        }

        // if the sample with the last subtracted halo is closer to the expected mass, keep it
        // LOG_ULTRA_DEBUG("Deciding to keep last halo M %.3e tot %.3e exp %.3e",last_M_del,*M_tot_pt,exp_M);
        if(fabs(*M_tot_pt + last_M_del - exp_M) < fabs(*M_tot_pt - exp_M)){
            M_out[random_idx] = last_M_del;
            // LOG_ULTRA_DEBUG("kept.");
            *M_tot_pt += last_M_del;
        }
    }
}

/* Creates a realisation of halo properties by sampling the halo mass function and
 * conditional property PDFs, Sampling is done until there is no more mass in the condition
 * Stochasticity is ignored below a certain mass threshold */
int stoc_mass_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    //lnMmin only used for sampling, apply factor here
    double mass_tol = global_params.STOC_MASS_TOL;
    double exp_M = hs_constants->expected_M;
    //fudge factor for assuming that internal lagrangian volumes are independent
    exp_M *= 0.90;

    int n_halo_sampled=0;
    double M_prog=0;
    double M_sample;

    double tbl_arg = hs_constants->cond_val;

    while(M_prog < exp_M){
        M_sample = sample_dndM_inverse(tbl_arg,hs_constants,rng);

        M_prog += M_sample;
        M_out[n_halo_sampled++] = M_sample;
        // LOG_ULTRA_DEBUG("Sampled %.3e | %.3e %d",M_sample,M_prog,n_halo_sampled);
    }
    // LOG_ULTRA_DEBUG("Before fix: %d %.3e",n_halo_sampled,M_prog);
    //The above sample is above the expected mass, by up to 100%. I wish to make the average mass equal to exp_M
    fix_mass_sample(rng,exp_M,&n_halo_sampled,&M_prog,M_out);
    // LOG_ULTRA_DEBUG("After fix: %d %.3e",n_halo_sampled,M_prog);

    // LOG_ULTRA_DEBUG("Got %d (exp.%.2e) halos mass %.2e (exp. %.2e) %.2f",
                    // n_halo_sampled,hs_constants->expected_N,M_prog,exp_M,M_prog/exp_M - 1);
    *n_halo_out = n_halo_sampled;
    return 0;
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
    double M_min = hs_constants->M_min;
    double sigma_min = hs_constants->sigma_min;
    double lnM_min = hs_constants->lnM_min;

    int n_halo_sampled;
    double x_sample, sigma_sample, M_sample, M_remaining, delta_current;
    double lnM_remaining, sigma_r, del_term;

    double tbl_arg = hs_constants->cond_val;
    n_halo_sampled = 0;
    // LOG_ULTRA_DEBUG("Start: M %.2e (%.2e) d %.2e",M_cond,exp_M,d_cond);

    //set initial amount (subtracted unresolved Mass)
    M_remaining = exp_M;
    lnM_remaining = log(M_remaining);

    double x_min;

    while(M_remaining > global_params.SAMPLER_MIN_MASS){
        sigma_r = EvaluateSigma(lnM_remaining);
        delta_current = (get_delta_crit(user_params_stoc->HMF,sigma_r,growthf) - d_cond)/(M_remaining/M_cond);
        del_term = delta_current*delta_current/growthf/growthf;

        //Low x --> high sigma --> low mass, high x --> low sigma --> high mass
        //|x| required to sample the smallest halo on the tables
        x_min = sqrt(del_term/(sigma_min*sigma_min - sigma_r*sigma_r));

        //LOG_ULTRA_DEBUG("M_rem %.2e d %.2e sigma %.2e min %.2e xmin %.2f",M_remaining,delta_current,sigma_r,sigma_min,x_min);

        //we use the gaussian tail distribution to enforce our Mmin limit from the sigma tables
        x_sample = gsl_ran_ugaussian_tail(rng,x_min);
        sigma_sample = sqrt(del_term/(x_sample*x_sample) + sigma_r*sigma_r);
        M_sample = EvaluateSigmaInverse(sigma_sample,&Sigma_InterpTable);
        M_sample = exp(M_sample);

        //LOG_ULTRA_DEBUG("found Mass %d %.2e",n_halo_sampled,M_sample);
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
    // LOG_ULTRA_DEBUG("Frac: u_res = %.2e, dd=%.2e, J=%.2e",u_res,dd,EvaluateJ(u_res,gamma1));
    return sqrt(2./PI)*EvaluateJ(u_res,gamma1)*G1/sigma_start*dd;
}

//binary splitting with small internal steps based on Parkinson+08, Bensen+16, Qiu+20 (Darkforest)
//This code was mostly taken from Darkforest (Qiu+20)
//NOTE: some unused variables here
//Only works with adjusted EPS
int stoc_split_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    double G0 = 0.8;
    double gamma1 = 0.35;
    double gamma2 = -0.35;
    double m_res = hs_constants->M_min;
    double lnm_res = hs_constants->lnM_min;
    double eps1 = 0.1;
    double eps2 = 0.1;
    // Load interpolation tables for sigma_m
    // Initialise the binary tree
    // Plant tree
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
    int n_out = 0;
    int idx = 0;

    float d_points[MAX_HALO_CELL], m_points[MAX_HALO_CELL];
    int n_points;
    //set initial points
    d_points[0] = Deltac / hs_constants->growth_in;
    m_points[0] = hs_constants->M_cond;
    d_target = Deltac / hs_constants->growth_out;
    n_points = 1;
    double M_total = 0;

    sigma_res = EvaluateSigma(lnm_res);
    sigmasq_res = sigma_res*sigma_res;

    while(idx < n_points) {
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
        alpha_half = EvaluatedSigmasqdm(lnm_half);
        //convert from d(sigma^2)/dm to -d(lnsigma)/d(lnm)
        alpha_half = -m_half/(2*sigma_half*sigma_half)*alpha_half;
        sigmasq_half = sigma_half*sigma_half;
        G1 = G0*pow(d_start/sigma_start, gamma2);
        //
        q = 0.;
        q_res = m_res/m_start;
        if (q_res >= 0.5) {
            // No split
            dd = eps1*sqrt(2)*sqrt(sigmasq_half - sigmasq_start);
            if(dd >= dd_target){
                dd = dd_target;
                save = 1;
            }
            F = ComputeFraction_split(sigma_start, sigmasq_start, sigmasq_res, G1, dd, gamma1);
        }
        else {
            // Compute B and beta
            V_res = sigmasq_res*pow(sigmasq_res - sigmasq_start, -1.5);
            V_half = sigmasq_half*pow(sigmasq_half - sigmasq_start, -1.5);
            beta = log(V_res/V_half)/log(2.*q_res);
            B = pow(2., beta)*V_half;
            // Compute ddelta1
            dd1 = eps1*sqrt(2)*sqrt(sigmasq_half - sigmasq_start);
            // Compute ddelta2
            mu = gamma1 < 0. ? -log(sigma_res/sigma_half)/log(2.*q_res) : alpha_half;
            eta = beta - 1 - gamma1*mu;
            pow_diff = pow(.5, eta) - pow(q_res, eta);
            G2 = G1*pow(sigma_half/sigma_start, gamma1)*pow(0.5, mu*gamma1);
            dN_dd = sqrt(2./PI)*B*pow_diff/eta*alpha_half*G2;
            dd2 = eps2/dN_dd;
            // Choose
            if (dd1 < dd2)
                dd = dd1;
            else
                dd = dd2;
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
                alpha_q = EvaluatedSigmasqdm(log(m_q));
                //convert from d(sigma^2)/dm to -d(lnsigma)/d(lnm)
                alpha_q = -m_q/(2*sigma_q*sigma_q)*alpha_q;
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
        // Compute progenitor mass
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
    if(hs_constants->delta <= DELTA_MIN || hs_constants->expected_M < global_params.SAMPLER_MIN_MASS){
        *n_halo_out = 0;
        return 0;
    }
    //if delta is above critical, form one big halo
    if(hs_constants->delta >= MAX_DELTAC_FRAC*get_delta_crit(user_params_stoc->HMF,hs_constants->sigma_cond,hs_constants->growth_out)){
        *n_halo_out = 1;

        //Expected mass takes into account potential dexm overlap
        M_out[0] = hs_constants->expected_M;
        return 0;
    }

    //We always use Number-Limited sampling for grid-based cases
    if(global_params.SAMPLE_METHOD == 1 || !hs_constants->from_catalog){
        err = stoc_halo_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(global_params.SAMPLE_METHOD == 0){
        err = stoc_mass_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(global_params.SAMPLE_METHOD == 2){
        err = stoc_partition_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(global_params.SAMPLE_METHOD == 3){
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
int sample_halo_grids(gsl_rng **rng_arr, double redshift, float *dens_field, float *halo_overlap_box, struct HaloField *halofield_large, struct HaloField *halofield_out, struct HaloSamplingConstants *hs_constants){
    int lo_dim = user_params_stoc->HII_DIM;
    int hi_dim = user_params_stoc->DIM;
    double boxlen = user_params_stoc->BOX_LEN;

    double Mcell = hs_constants->M_cond;
    double lnMcell = hs_constants->lnM_cond;
    double Mmin = hs_constants->M_min;
    double lnMmin = hs_constants->lnM_min;

    int nhalo_in = halofield_large->n_halos;
    double growthf = hs_constants->growth_out;

    unsigned long long int nhalo_threads[user_params_stoc->N_THREADS];
    unsigned long long int istart_threads[user_params_stoc->N_THREADS];

    unsigned long long int arraysize_total = halofield_out->buffer_size;
    unsigned long long int arraysize_local = arraysize_total / user_params_stoc->N_THREADS;

    LOG_DEBUG("Beginning stochastic halo sampling on %d ^3 grid",lo_dim);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e,volume = %.3e, D = %.3e",redshift,Mmin,Mcell,Mcell/RHOcrit/cosmo_params_stoc->OMm,growthf);
    LOG_DEBUG("Total Array Size %llu, array size per thread %llu (~%.3e GB total)",arraysize_total,arraysize_local,6.*arraysize_total*sizeof(int)/1e9);

    double *dexm_radii = calloc(nhalo_in,sizeof(double));
    double *dexm_pos = calloc(nhalo_in*3,sizeof(double));
    double total_volume_excluded=0.;
    double total_volume_dexm=0.;
    double vol_conversion = (double)user_params_stoc->HII_DIM / (double)user_params_stoc->DIM;

#pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        //PRIVATE VARIABLES
        int x,y,z,i,j;
        int threadnum = omp_get_thread_num();

        int nh_buf=0;
        double delta;
        float prop_buf[2], prop_dummy[2];
        int crd_hi[3];

        double mass_defc;

        //buffers per cell
        float hm_buf[MAX_HALO_CELL];

        unsigned long long int count=0;
        unsigned long long int istart = threadnum * arraysize_local;
        //debug total
        double M_cell=0.;

        //we need a private version
        struct HaloSamplingConstants hs_constants_priv;
        hs_constants_priv = *hs_constants;

        //assign big halos into list first (split amongst ranks)
#pragma omp for reduction(+:total_volume_dexm)
        for (j=0;j<nhalo_in;j++){
            halofield_out->halo_masses[istart+count] = halofield_large->halo_masses[j];
            halofield_out->star_rng[istart+count] = halofield_large->star_rng[j];
            halofield_out->sfr_rng[istart+count] = halofield_large->sfr_rng[j];
            halofield_out->halo_coords[0 + 3*(istart+count)] = halofield_large->halo_coords[0 + 3*j];
            halofield_out->halo_coords[1 + 3*(istart+count)] = halofield_large->halo_coords[1 + 3*j];
            halofield_out->halo_coords[2 + 3*(istart+count)] = halofield_large->halo_coords[2 + 3*j];

            //To accelerate, I store the large-halo radii (units of HII_DIM cells)
            total_volume_dexm += halofield_large->halo_masses[j] / (RHOcrit * cosmo_params_stoc->OMm) * pow(vol_conversion,3);
            dexm_radii[j] = MtoR(halofield_large->halo_masses[j])/boxlen*lo_dim;

            //to compare the centres of the cells at DIM and HII_DIM, we add 0.5 to the coordinate at each resolution
            //subtracting 0.5 means we are in the coordinates where (0,0,0) is the CENTRE of the first cell,
            // to directly compare with (x,y,z) on HII_DIM
            dexm_pos[0] = (halofield_large->halo_coords[0 + 3*j] + 0.5) * vol_conversion - 0.5;
            dexm_pos[1] = (halofield_large->halo_coords[1 + 3*j] + 0.5) * vol_conversion - 0.5;
            dexm_pos[2] = (halofield_large->halo_coords[2 + 3*j] + 0.5) * vol_conversion - 0.5;
            count++;
        }

//I need the full radii list before the loop starts
#pragma omp barrier

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
                    M_cell = 0;
                    for(i=0;i<nh_buf;i++){
                        //sometimes halos are subtracted from the sample (set to zero)
                        //we do not want to save these
                        if(hm_buf[i] < global_params.SAMPLER_MIN_MASS) continue;

                        if(count >= arraysize_local){
                            LOG_ERROR("More than %d halos (expected %d) with buffer size factor %d",
                                        arraysize_local,arraysize_local/global_params.MAXHALO_FACTOR,global_params.MAXHALO_FACTOR);
                            LOG_ERROR("If you expected to have an above average halo number try raising global_params.MAXHALO_FACTOR");
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
                        count++;

                        M_cell += hm_buf[i];
                        if((x+y+z) == 0){
                            LOG_ULTRA_DEBUG("Halo %d Mass %.2e Stellar %.2e SFR %.2e",i,hm_buf[i],prop_buf[0],prop_buf[1]);
                        }
                    }
                    if((x+y+z) == 0){
                        LOG_SUPER_DEBUG("Cell (%d %d %d): delta %.2f | N %d (exp. %.2e) | Total M %.2e (exp. %.2e) overlap defc %.2f",x,y,z,
                                        delta,nh_buf,hs_constants_priv.expected_N,M_cell,hs_constants_priv.expected_M,mass_defc);
                    }
                }
            }
        }
        LOG_SUPER_DEBUG("Thread %d found %d halos",threadnum,count);

        istart_threads[threadnum] = istart;
        nhalo_threads[threadnum] = count;
    }

    LOG_SUPER_DEBUG("Total dexm volume %.6e Total volume excluded %.6e (In units of HII_DIM cells)",total_volume_dexm,total_volume_excluded);
    free(dexm_radii);
    free(dexm_pos);

    //Condense the sparse array (serial)
    int i=0;
    unsigned long long int count_total = 0;
    for(i=0;i<user_params_stoc->N_THREADS;i++){
        memmove(&halofield_out->halo_masses[count_total],&halofield_out->halo_masses[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->star_rng[count_total],&halofield_out->star_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->sfr_rng[count_total],&halofield_out->sfr_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->halo_coords[3*count_total],&halofield_out->halo_coords[3*istart_threads[i]],sizeof(int)*3*nhalo_threads[i]);
        LOG_SUPER_DEBUG("Moved array (start,count) (%d, %d) to position %d",istart_threads[i],nhalo_threads[i],count_total);
        count_total += nhalo_threads[i];
    }
    halofield_out->n_halos = count_total;

    //replace the rest with zeros for clarity
    memset(&halofield_out->halo_masses[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->halo_coords[3*count_total],0,3*(arraysize_total-count_total)*sizeof(int));
    memset(&halofield_out->star_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->sfr_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    LOG_SUPER_DEBUG("Set %d elements beyond %d to zero",arraysize_total-count_total,count_total);
    return 0;
}

//NOTE: there's a lot of repeated code here and in build_halo_cats, find a way to merge
int sample_halo_progenitors(gsl_rng ** rng_arr, double z_in, double z_out, struct HaloField *halofield_in,
                             struct HaloField *halofield_out, struct HaloSamplingConstants *hs_constants){
    int nhalo_in = halofield_in->n_halos;
    if(z_in >= z_out){
        LOG_ERROR("halo progenitors must go backwards in time!!! z_in = %.1f, z_out = %.1f",z_in,z_out);
        Throw(ValueError);
    }

    double growth_in = hs_constants->growth_in;
    double growth_out = hs_constants->growth_out;
    int lo_dim = user_params_stoc->HII_DIM;
    int hi_dim = user_params_stoc->DIM;
    double boxlen = user_params_stoc->BOX_LEN;
    //cell size for smoothing / CMF calculation
    double lnMmax_tb = hs_constants->lnM_max_tb;
    double Mmax_tb = hs_constants->M_max_tables;
    double Mmin = hs_constants->M_min;
    double lnMmin = hs_constants->lnM_min;
    double delta = hs_constants->delta;

    unsigned long long int nhalo_threads[user_params_stoc->N_THREADS];
    unsigned long long int istart_threads[user_params_stoc->N_THREADS];

    unsigned long long int arraysize_total = halofield_out->buffer_size;
    unsigned long long int arraysize_local = arraysize_total / user_params_stoc->N_THREADS;

    LOG_DEBUG("Beginning stochastic halo sampling of progenitors on %d halos",nhalo_in);
    LOG_DEBUG("z = %f, Mmin = %e, d = %.3e",z_out,Mmin,delta);
    LOG_DEBUG("Total Array Size %llu, array size per thread %llu (~%.3e GB total)",arraysize_total,arraysize_local,6.*arraysize_total*sizeof(int)/1e9);

    double corr_arr[2] = {hs_constants->corr_star,hs_constants->corr_sfr};

#pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        float prog_buf[MAX_HALO_CELL];
        int n_prog;
        double M_prog;

        float propbuf_in[2];
        float propbuf_out[2];

        int threadnum = omp_get_thread_num();
        double M2;
        int ii,jj;
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
                LOG_ERROR("Input Mass = %.2e at %d of %d, something went wrong in the input catalogue",M2,ii,nhalo_in);
                Throw(ValueError);
            }
            //set condition-dependent variables for sampling
            stoc_set_consts_cond(&hs_constants_priv,M2);

            //Sample the CMF set by the descendant
            stoc_sample(&hs_constants_priv,rng_arr[threadnum],&n_prog,prog_buf);

            propbuf_in[0] = halofield_in->star_rng[ii];
            propbuf_in[1] = halofield_in->sfr_rng[ii];

            //place progenitors in local list
            M_prog = 0;
            for(jj=0;jj<n_prog;jj++){
                //sometimes halos are subtracted from the sample (set to zero)
                //we do not want to save these
                if(prog_buf[jj] < global_params.SAMPLER_MIN_MASS) continue;

                if(count >= arraysize_local){
                    LOG_ERROR("More than %d halos (expected %d) with buffer size factor %d",
                                arraysize_local,arraysize_local/global_params.MAXHALO_FACTOR,global_params.MAXHALO_FACTOR);
                    LOG_ERROR("If you expected to have an above average halo number try raising global_params.MAXHALO_FACTOR");
                    Throw(ValueError);
                }

                set_prop_rng(rng_arr[threadnum], 1, corr_arr, propbuf_in, propbuf_out);

                halofield_out->halo_masses[istart + count] = prog_buf[jj];
                halofield_out->halo_coords[3*(istart + count) + 0] = halofield_in->halo_coords[3*ii+0];
                halofield_out->halo_coords[3*(istart + count) + 1] = halofield_in->halo_coords[3*ii+1];
                halofield_out->halo_coords[3*(istart + count) + 2] = halofield_in->halo_coords[3*ii+2];

                halofield_out->star_rng[istart + count] = propbuf_out[0];
                halofield_out->sfr_rng[istart + count] = propbuf_out[1];
                count++;

                if(ii==0){
                    M_prog += prog_buf[jj];
                    LOG_ULTRA_DEBUG("First Halo Prog %d: Mass %.2e Stellar %.2e SFR %.2e e_d %.3f",jj,prog_buf[jj],propbuf_out[0],propbuf_out[1],Deltac*growth_out/growth_in);
                }
            }
            if(ii==0){
                LOG_ULTRA_DEBUG(" HMF %d delta %.3f delta_coll %.3f delta_desc %.3f adjusted %.3f",user_params_stoc->HMF,
                                                                                hs_constants_priv.delta,
                                                                                get_delta_crit(user_params_stoc->HMF,hs_constants_priv.sigma_cond,growth_out),
                                                                                get_delta_crit(user_params_stoc->HMF,hs_constants_priv.sigma_cond,growth_in),
                                                                                get_delta_crit(user_params_stoc->HMF,hs_constants_priv.sigma_cond,growth_in)*growth_out/growth_in);
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
    for(i=0;i<user_params_stoc->N_THREADS;i++){
        LOG_SUPER_DEBUG("Thread %d found %d Halos",i,nhalo_threads[i]);
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
int stochastic_halofield(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options
                        , int seed, float redshift_desc, float redshift, float *dens_field, float *halo_overlap_box, struct HaloField *halos_desc, struct HaloField *halos){
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);
    Broadcast_struct_global_IT(user_params,cosmo_params,astro_params,flag_options);

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
        LOG_DEBUG("Calculating halo progenitors from z=%.1f to z=%.1f | %d", redshift_desc,redshift,halos_desc->n_halos);
        sample_halo_progenitors(rng_stoc,redshift_desc,redshift,halos_desc,halos,&hs_constants);
    }

    LOG_DEBUG("Found %d Halos", halos->n_halos);

    if(halos->n_halos > 3){
        LOG_DEBUG("First few Masses:  %11.3e %11.3e %11.3e",halos->halo_masses[0],halos->halo_masses[1],halos->halo_masses[2]);
        LOG_DEBUG("First few Stellar: %11.3e %11.3e %11.3e",halos->star_rng[0],halos->star_rng[1],halos->star_rng[2]);
        LOG_DEBUG("First few SFR:     %11.3e %11.3e %11.3e",halos->sfr_rng[0],halos->sfr_rng[1],halos->sfr_rng[2]);
    }

    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
    }
    free_dNdM_tables();

    free_rng_threads(rng_stoc);
    LOG_DEBUG("Done.");
    return 0;
}

int test_mfp_filter(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options
                    , float *input_box, double R, double mfp, double *result){
    int i,j,k;
    //setup the box

    fftwf_complex *box_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    LOG_DEBUG("Allocated");

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                *((float *)box_unfiltered + HII_R_FFT_INDEX(i,j,k)) = input_box[HII_R_INDEX(i,j,k)];
    LOG_DEBUG("Inited");


    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_unfiltered);

    LOG_DEBUG("FFT'd");

    //QUESTION: why do this here instead of at the end?
    for(i=0;i<HII_KSPACE_NUM_PIXELS;i++){
        box_unfiltered[i] /= (double)HII_TOT_NUM_PIXELS;
    }

    memcpy(box_filtered,box_unfiltered,sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    LOG_DEBUG("Copied");

    if(flag_options->USE_EXP_FILTER)
        filter_box_mfp(box_filtered, 1, R, mfp);
    else
        filter_box(box_filtered,1,global_params.HII_FILTER,R);


    LOG_DEBUG("Filtered");
    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_filtered);
    LOG_DEBUG("IFFT'd");

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                    result[HII_R_INDEX(i,j,k)] = fmaxf(*((float *)box_filtered + HII_R_FFT_INDEX(i,j,k)) , 0.0);

    LOG_DEBUG("Assigned");

    fftwf_free(box_unfiltered);
    fftwf_free(box_filtered);

    return 0;
}

//This is a test function which takes a list of conditions (cells or halos) and samples them to produce a descendant list
//      as well as per-condition number and mass counts
int single_test_sample(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options,
                        int seed, int n_condition, float *conditions, int *cond_crd, double z_out, double z_in, int *out_n_tot, int *out_n_cell, double *out_n_exp,
                        double *out_m_cell, double *out_m_exp, float *out_halo_masses, int *out_halo_coords){
    int status;
    Try{
        //make the global structs
        Broadcast_struct_global_UF(user_params,cosmo_params);
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);
        Broadcast_struct_global_IT(user_params,cosmo_params,astro_params,flag_options);

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

        struct parameters_gsl_MF_integrals integral_params = {
                .redshift = z_out,
                .growthf = hs_constants->growth_out,
                .HMF = user_params->HMF,
        };

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
                    if(out_hm[i] < global_params.SAMPLER_MIN_MASS) continue;
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
                // out_n_exp[j] = hs_constants_priv.expected_N;
                // out_m_exp[j] = hs_constants_priv.expected_M;
                out_n_cell[j] = n_halo_cond;
                out_m_cell[j] = M_prog;
            }
        }

        out_n_tot[0] = n_halo_tot;

        //get expected values from the saved mass range
        if(hs_constants->from_catalog){
            initialise_dNdM_tables(log(global_params.SAMPLER_MIN_MASS), hs_constants->lnM_max_tb, log(global_params.SAMPLER_MIN_MASS),
                                 hs_constants->lnM_max_tb, hs_constants->growth_out, hs_constants->growth_in, true);
        }
        else{
            double delta_crit = get_delta_crit(user_params_stoc->HMF,hs_constants->sigma_cond,hs_constants->growth_out);
            initialise_dNdM_tables(DELTA_MIN, MAX_DELTAC_FRAC*delta_crit, log(global_params.SAMPLER_MIN_MASS), hs_constants->lnM_max_tb,
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

        if(user_params_stoc->USE_INTERPOLATION_TABLES){
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
