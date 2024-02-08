/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//BIG TODO: sort out single/double precision all the way through
//TODO: the USE_INTERPOLATION_TABLES flag may need to be forced, it's in a strange position here
//  where it only affects the sigma table.
//NOTE: for the future discrete tables this does make sense
//TODO: Don't have every error be a ValueError

//max number of attempts for mass tolerance before failure
#define MAX_ITERATIONS 1e2
#define MAX_ITER_N 1e2 //for stoc_halo_sample (select N halos) how many tries for one N, this should be large to enforce a near-possion p(N)
#define MMAX_TABLES 1e14

//buffer size (per cell of arbitrary size) in the sampling function
#define MAX_HALO_CELL (int)1e5
#define MAX_DELTAC_FRAC (float)0.999 //max delta/deltac for interpolation tables / integrals

//we need to define a density minimum for the tables, since we are in lagrangian density / linear growth it's possible to go below -1
//so we explicitly set a minimum here which sets table limits and puts no halos in cells below that (Lagrangian) density
#define DELTA_MIN -1

//NOTE: because the .c files are directly included in GenerateIC.c, the static doesn't really do anything :(
static struct AstroParams *astro_params_stoc;
static struct CosmoParams *cosmo_params_stoc;
static struct UserParams *user_params_stoc;
static struct FlagOptions *flag_options_stoc;

//TODO: move these tables to the below struct
struct RGTable2D Nhalo_inv_table = {.allocated=false};
struct RGTable1D Nhalo_table = {.allocated=false};
struct RGTable1D Mcoll_table = {.allocated=false};

//parameters for the halo mass->stars calculations
//Note: ideally I would split this into constants set per snapshot and
//  constants set per condition, however some variables (delta or Mass)
//  can be set with differing frequencies depending on the condition type
struct HaloSamplingConstants{
    //calculated per redshift
    int update; //flag for first box or updating halos
    double t_h;
    double t_h_prev;
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

    //per-condition/redshift depending on update or not
    double delta;
    double M_cond;
    double lnM_cond;
    double sigma_cond;

    //calculated per condition
    double cond_val; //need to specify for the tables
    //can't do SFR since it depends on the sampled stellar
    //although there's no pow calls so it should be faster
    double expected_N;
    double expected_M;

    //calculated per sample
    //Nothing here since it's the lowest level and there's no point saving
};

void print_hs_consts(struct HaloSamplingConstants * c){
    LOG_DEBUG("Printing halo sampler constants....");
    LOG_DEBUG("update %d z_in %.2f z_out %.2f d_in %.2f d_out %.2f",c->update,c->z_in,c->z_out,c->growth_in,c->growth_out);
    LOG_DEBUG("t_h %.2e t_h_prev %.2e M_min %.2e (%.2e) (%.2f) M_max %.2e (%.2e)",c->t_h,c->t_h_prev,c->M_min,c->lnM_min,c->sigma_min,c->M_max_tables,c->lnM_max_tb);
    LOG_DEBUG("Corr Star %.2e SFR %.2e",c->corr_star,c->corr_sfr);
    //TODO: change formatting based on c->update to make it clear what is z-dependent or condition-dependent
    LOG_DEBUG("CONDITION DEPENDENT STUFF (may not be set)");
    LOG_DEBUG("delta %.2e M_c %.2e (%.2e) (%.2e) cond %.2e",c->delta,c->M_cond,c->lnM_cond,c->sigma_cond,c->cond_val);
    LOG_DEBUG("exp N %.2f exp M %.2e",c->expected_N,c->expected_M);
    return;
}

//The sigma interp table is regular in log mass, not sigma so we need to loop ONLY FOR METHOD=3
//TODO: make a new RGI from the sigma tables if we take the partition method seriously.
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
    double table_val_1 = s_table->x_min + idx-1*s_table->x_width; //TODO:CHECK
    double interp_point = (sigma - table_val_0)/(table_val_1-table_val_0);

    return table_val_0*(1-interp_point) + table_val_1*(interp_point);
}

void Broadcast_struct_global_STOC(struct UserParams *user_params, struct CosmoParams *cosmo_params,struct AstroParams *astro_params, struct FlagOptions *flag_options){
    cosmo_params_stoc = cosmo_params;
    user_params_stoc = user_params;
    astro_params_stoc = astro_params;
    flag_options_stoc = flag_options;
}

// TODO: this should probably be in UsefulFunctions.c
void seed_rng_threads(gsl_rng * rng_arr[], int seed){
    // setting tbe random seeds (copied from GenerateICs.c)
    gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

    gsl_rng_set(rseed, seed);

    unsigned int seeds[user_params_stoc->N_THREADS];

    // For multithreading, seeds for the RNGs are generated from an initial RNG (based on the input random_seed) and then shuffled (Author: Fred Davies)
    int num_int = INT_MAX/256; //JD: this was taking a few seconds per snapshot so i reduced the number TODO: init the RNG once
    int i, thread_num;
    unsigned int *many_ints = (unsigned int *)malloc((size_t)(num_int*sizeof(unsigned int))); // Some large number of possible integers
    for (i=0; i<num_int; i++) {
        many_ints[i] = i;
    }

    gsl_ran_choose(rseed, seeds, user_params_stoc->N_THREADS, many_ints, num_int, sizeof(unsigned int)); // Populate the seeds array from the large list of integers
    gsl_ran_shuffle(rseed, seeds, user_params_stoc->N_THREADS, sizeof(unsigned int)); // Shuffle the randomly selected integers

    int checker;

    checker = 0;
    // seed the random number generators
    for (thread_num = 0; thread_num < user_params_stoc->N_THREADS; thread_num++){
        switch (checker){
            case 0:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 1:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_gfsr4);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 2:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_cmrg);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 3:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_mrg);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 4:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_taus2);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
        } // end switch

        checker += 1;

        if(checker==5) {
            checker = 0;
        }
    }

    gsl_rng_free(rseed);
    free(many_ints);
}

void free_rng_threads(gsl_rng * rng_arr[]){
    int ii;
    for(ii=0;ii<user_params_stoc->N_THREADS;ii++){
        gsl_rng_free(rng_arr[ii]);
    }
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
        initialiseSigmaMInterpTable(M_min/2,M_max);

    struct parameters_gsl_MF_integrals params = {
        .redshift = redshift,
        .growthf = growthf,
        .HMF = user_params_stoc->HMF,
    };

    if(user_params_stoc->INTEGRATION_METHOD_HALOS == 1)
        initialise_GL(NGL_INT, log(M_min), log(M_max));

    result = IntegratedNdM(log(M_min), log(M_max), params, 1, user_params->INTEGRATION_METHOD_HALOS) * VOLUME;
    LOG_DEBUG("Expected %.2e Halos in the box from masses %.2e to %.2e at z=%.2f",result,M_min,M_max,redshift);

    if(user_params->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();

    return result;
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
        delta_crit = get_delta_crit(user_params_stoc->HMF,sigma_cond,growth1);
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
                .HMF = user_params_stoc->HMF,
    };

    #pragma omp parallel num_threads(user_params_stoc->N_THREADS) private(i,j,k) firstprivate(delta_crit,integral_params,sigma_cond,lnM_cond)
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
                delta = get_delta_crit(user_params_stoc->HMF,sigma_cond,param)/param*growth1;
                // //current barrier at condition for bounds checking
                // delta_crit = get_delta_crit(user_params_stoc->HMF,EvaluateSigma(lnM_cond,0,NULL),growth1);
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
            if(user_params_stoc->INTEGRATION_METHOD_HALOS == 1)
                initialise_GL(NGL_INT, ymin, lnM_cond);

            norm = IntegratedNdM(ymin, lnM_cond, integral_params,-1, user_params_stoc->INTEGRATION_METHOD_HALOS);
            fcoll = IntegratedNdM(ymin, lnM_cond, integral_params, -2, user_params_stoc->INTEGRATION_METHOD_HALOS);
            Nhalo_table.y_arr[i] = norm;
            Mcoll_table.y_arr[i] = fcoll;
            // LOG_ULTRA_DEBUG("cond x: %.2e M [%.2e,%.2e] %.2e d %.2f D %.2f n %d ==> %.8e / %.8e",x,exp(ymin),exp(ymax),exp(lnM_cond),delta,growth1,i,norm,fcoll);

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
                if(user_params_stoc->INTEGRATION_METHOD_HALOS == 1)
                    initialise_GL(NGL_INT, y, lnM_cond);

                y = ya[j];
                if(lnM_cond <= y){
                    //setting to one guarantees samples at lower mass
                    //This fixes upper mass limits for the conditions
                    buf = 0.;
                }
                else{
                    buf = IntegratedNdM(y, lnM_cond, integral_params, -1, user_params_stoc->INTEGRATION_METHOD_HALOS); //Number density between ymin and y
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

//TODO: Speedtest the RGI interpolation present in Spintemp etc...
//  Save the X/Y/Z from the table builder and apply the Minihalo 2D interpolation
//NOTE: if p(x) is uniform, p(log(1-x)) follows the exponential distribution
//  But the gsl_ran_exponential function does the exact same thing but adds a mean
double sample_dndM_inverse(double condition, struct HaloSamplingConstants * hs_constants, gsl_rng * rng){
    double p_in, min_prob;
    p_in = gsl_rng_uniform(rng);
    if(!hs_constants->update){
        p_in = log(p_in);
        if(p_in < global_params.MIN_LOGPROB) p_in = global_params.MIN_LOGPROB; //we assume that M(min_logprob) ~ M_cond
    }
    return EvaluateRGTable2D(condition,p_in,&Nhalo_inv_table);
}

//Set the constants that are calculated once per snapshot
void stoc_set_consts_z(struct HaloSamplingConstants *const_struct, double redshift, double redshift_prev){
    LOG_DEBUG("Setting z constants z=%.2f z_prev=%.2f",redshift,redshift_prev);
    const_struct->t_h = t_hubble(redshift);
    const_struct->growth_out = dicke(redshift);
    const_struct->z_out = redshift;
    const_struct->z_in = redshift_prev;

    const_struct->M_min = global_params.SAMPLER_MIN_MASS;
    const_struct->lnM_min = log(const_struct->M_min);
    const_struct->M_max_tables = global_params.M_MAX_INTEGRAL;
    const_struct->lnM_max_tb = log(const_struct->M_max_tables);


    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(const_struct->M_min / 2,const_struct->M_max_tables);
    }

    const_struct->sigma_min = EvaluateSigma(const_struct->lnM_min,0,NULL);

    if(redshift_prev >= 0){
        const_struct->t_h_prev = t_hubble(redshift_prev);
        const_struct->growth_in = dicke(redshift_prev);
        if(astro_params_stoc->CORR_SFR > 0)
            const_struct->corr_sfr = exp(-(redshift - redshift_prev)/astro_params_stoc->CORR_SFR);
        else
            const_struct->corr_sfr = 0;
        if(astro_params_stoc->CORR_STAR > 0)
            const_struct->corr_star = exp(-(redshift - redshift_prev)/astro_params_stoc->CORR_STAR);
        else
            const_struct->corr_star = 0;

        const_struct->update = 1;
        //TODO: change the table functions to accept the structure
        initialise_dNdM_tables(const_struct->lnM_min, const_struct->lnM_max_tb,const_struct->lnM_min, const_struct->lnM_max_tb,
                                const_struct->growth_out, const_struct->growth_in, true);
    }
    else {
        double M_cond = RHOcrit * cosmo_params_stoc->OMm * VOLUME / HII_TOT_NUM_PIXELS;
        const_struct->M_cond = M_cond;
        const_struct->lnM_cond = log(M_cond);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond,0,NULL);
        //for the table limits
        double delta_crit = get_delta_crit(user_params_stoc->HMF,const_struct->sigma_cond,const_struct->growth_out);
        const_struct->update = 0;
        initialise_dNdM_tables(DELTA_MIN, MAX_DELTAC_FRAC*delta_crit, const_struct->lnM_min, const_struct->lnM_max_tb, const_struct->growth_out, const_struct->lnM_cond, false);
    }

    LOG_DEBUG("Done.");
    return;
}

//set the constants which are calculated once per condition
void stoc_set_consts_cond(struct HaloSamplingConstants *const_struct, double cond_val){
    double m_exp,n_exp;

    //Here the condition is a mass, volume is the Lagrangian volume and delta_l is set by the
    //redshift difference which represents the difference in delta_crit across redshifts
    if(const_struct->update){
        const_struct->M_cond = cond_val;
        const_struct->lnM_cond = log(cond_val);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond,0,NULL);
        //mean stellar mass of this halo mass, used for stellar z correlations
        const_struct->cond_val = const_struct->lnM_cond;
        //condition delta is the previous delta crit
        const_struct->delta = get_delta_crit(user_params_stoc->HMF,const_struct->sigma_cond,const_struct->growth_in)\
                                                / const_struct->growth_in * const_struct->growth_out;
    }
    //Here the condition is a cell of a given density, the volume/mass is given by the grid parameters
    else{
        const_struct->delta = cond_val;
        const_struct->cond_val = cond_val;

        //the splines don't work well for cells above Deltac, but there CAN be cells above deltac, since this calculation happens
        //before the overlap, and since the smallest dexm mass is M_cell*(1.01^3) there *could* be a cell above Deltac not in a halo
        //NOTE: all this does is prevent integration errors below since these cases are also dealt with in stoc_sample
        if(cond_val > MAX_DELTAC_FRAC*get_delta_crit(user_params_stoc->HMF,const_struct->sigma_cond,const_struct->growth_out)){
            const_struct->expected_M = const_struct->M_cond;
            const_struct->expected_N = 1;
            return;
        }
        if(cond_val < DELTA_MIN){
            const_struct->expected_M = 0;
            const_struct->expected_N = 0;
            return;
        }
    }

    //Get expected N from interptable
    n_exp = EvaluateRGTable1D(const_struct->cond_val,&Nhalo_table);
    //NOTE: while the most common mass functions have simpler expressions for f(<M) (erfc based) this will be general, and shouldn't impact compute time much
    m_exp = EvaluateRGTable1D(const_struct->cond_val,&Mcoll_table);
    const_struct->expected_N = n_exp * const_struct->M_cond;
    const_struct->expected_M = m_exp * const_struct->M_cond;
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
int set_prop_rng(gsl_rng *rng, int update, double *interp, float * input, float * output){
    //find log(property/variance) / mean
    double prop1 = gsl_ran_ugaussian(rng);
    double prop2 = gsl_ran_ugaussian(rng);

    //Correlate properties by interpolating between the sampled and descendant gaussians
    //THIS ASSUMES THAT THE SELF-CORRELATION IS IN THE LOG PROPRETY, NOT THE PROPERTY ITSELF
    //IF IT SHOULD BE IN LINEAR SPACE, EXPONENTIATE THE RANDOM VARIABLES
    if(update){
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

    //QUICK HACK: setup a fast t_h for the catalogue since this is called from FindHaloes.c
    struct HaloSamplingConstants hs_constants;
    hs_constants.z_out = redshift;
    hs_constants.t_h = t_hubble(redshift);

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
                hm_sample = exp(hm_sample);
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
        LOG_ERROR("passed max iter in sample, last attempt M=%.3e [%.3e, %.3e] Me %.3e Mt %.3e cond %.3e",tbl_arg,M_prog,exp_M*(1-mass_tol),exp_M*(1+mass_tol),exp_M);
        Throw(ValueError);
    }

    found_halo_sample: *n_halo_out = halo_count;
    // LOG_ULTRA_DEBUG("Got %d (exp. %.2e) halos mass %.2e (exp. %.2e) %.2f | (%d,%d) att.",
    //                 nh,exp_N,M_prog,exp_M,M_prog/exp_M - 1, n_failures, n_attempts);
    return 0;
}

//TODO: this may not need its own function and passing around the pointers can be annoying
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
int fix_mass_sample(gsl_rng * rng, double exp_M, int *n_halo_pt, double *M_tot_pt, float *M_out){
    //Keep the last halo if it brings us closer to the expected mass
    //This is done by addition or subtraction over the limit to balance
    //the bias of the last halo being larger
    int random_idx;
    int n_removed = 0;
    double last_M_del;
    if(gsl_rng_uniform_int(rng,2)){
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
            n_removed++;
        }

        // if the sample with the last subtracted halo is closer to the expected mass, keep it
        // LOG_ULTRA_DEBUG("Deciding to keep last halo M %.3e tot %.3e exp %.3e",last_M_del,*M_tot_pt,exp_M);
        if(fabs(*M_tot_pt + last_M_del - exp_M) < fabs(*M_tot_pt - exp_M)){
            M_out[random_idx] = last_M_del;
            // LOG_ULTRA_DEBUG("kept.");
            *M_tot_pt += last_M_del;
            n_removed--;
        }
    }
    return n_removed;
    //FUTURE "fixes"
    //TODO: try different target / trigger, which could overcome the M > exp_M issues better than a delta
    //  e.g: set trigger for the check at exp_M - (M_Max - exp_M), target at exp_M
    //  This provides a range of trigger points which can be absorbed into the below-resolution values
    // Obviously breaks for exp_M < 0.5 M, think of other thresholds.

}

/* Creates a realisation of halo properties by sampling the halo mass function and
 * conditional property PDFs, Sampling is done until there is no more mass in the condition
 * Stochasticity is ignored below a certain mass threshold*/
int stoc_mass_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    //lnMmin only used for sampling, apply factor here
    double mass_tol = global_params.STOC_MASS_TOL;
    double exp_M = hs_constants->expected_M;
    //TODO:make this a globalparam
    if(hs_constants->update)exp_M *= 1.; //fudge factor for assuming that internal lagrangian volumes are independent

    int n_halo_sampled, n_failures=0;
    double M_prog=0;
    double M_sample;
    int n_removed;

    double tbl_arg = hs_constants->cond_val;

    for(n_failures=0;n_failures<MAX_ITERATIONS;n_failures++){
        n_halo_sampled = 0;
        M_prog = 0;
        while(M_prog < exp_M){
            M_sample = sample_dndM_inverse(tbl_arg,hs_constants,rng);
            M_sample = exp(M_sample);

            M_prog += M_sample;
            M_out[n_halo_sampled++] = M_sample;
            // LOG_ULTRA_DEBUG("Sampled %.3e | %.3e %d",M_sample,M_prog,n_halo_sampled);
        }

        //The above sample is above the expected mass, by up to 100%. I wish to make the average mass equal to exp_M
        // LOG_ULTRA_DEBUG("Before fix: %d %.3e",n_halo_sampled,M_prog);
        n_removed = fix_mass_sample(rng,exp_M,&n_halo_sampled,&M_prog,M_out);
        // LOG_ULTRA_DEBUG("After fix: %d (-%d) %.3e",n_halo_sampled,n_removed,M_prog);

        //LOG_ULTRA_DEBUG("attempt %d M=%.3e [%.3e, %.3e]",n_failures,M_prog,exp_M*(1-mass_tol),exp_M*(1+mass_tol));
        // //enforce some level of mass conservation
        if((M_prog <= exp_M*(1+mass_tol)) && (M_prog >= exp_M*(1-mass_tol))){
                //using goto to break double loop
                goto found_halo_sample;
        }
    }

    //technically I don't need the if statement but it might be confusing otherwise
    if(n_failures >= MAX_ITERATIONS){
        LOG_ERROR("passed max iter in sample, last attempt M=%.3e [%.3e, %.3e] cond %.3e",M_prog,exp_M*(1-mass_tol),exp_M*(1+mass_tol),tbl_arg);
        Throw(ValueError);
    }

    found_halo_sample: *n_halo_out = n_halo_sampled;
    // LOG_ULTRA_DEBUG("Got %d (exp.%.2e) halos mass %.2e (exp. %.2e) %.2f | %d att",
    //                 n_halo_sampled,hs_constants->expected_N,M_prog,exp_M,M_prog/exp_M - 1,n_failures);
    return 0;
}

//Sheth & Lemson partition model, Changes delta,M in the CMF as you sample
//This avoids discretization issues in the mass sampling, however.....
//it has been noted to overproduce small halos (McQuinn+ 2007)
//I don't know why sampling from Fcoll(M) is correct?
//  Do we not sample the same `particle` multiple times? (i.e 10x more samples for a single 10x mass halo)
//  How does the reduction of mass after each sample *exactly* cancel this 1/M effect
//If you want a non-barrier-based CMF, I don't know how to implement it here
int stoc_sheth_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    //lnMmin only used for sampling, apply factor here
    double exp_M = hs_constants->expected_M;
    double M_cond = hs_constants->M_cond;
    double d_cond = hs_constants->delta;
    double growthf = hs_constants->growth_out;
    double M_min = hs_constants->M_min;
    double sigma_min = hs_constants->sigma_min;
    double lnM_min = hs_constants->lnM_min;
    double lnM_max_tb = hs_constants->lnM_max_tb;

    int n_halo_sampled;
    double x_sample, sigma_sample, M_sample, M_remaining, delta_current;
    double lnM_remaining, sigma_r, del_term;

    double tbl_arg = hs_constants->cond_val;
    n_halo_sampled = 0;
    // LOG_ULTRA_DEBUG("Start: M %.2e (%.2e) d %.2e",M_cond,exp_M,d_cond);
    //set initial amount (subtract unresolved)
    //TODO: check if I should even do this
    M_remaining = exp_M;
    lnM_remaining = log(M_remaining);

    double x_min;

    while(M_remaining > global_params.SAMPLER_MIN_MASS){
        sigma_r = EvaluateSigma(lnM_remaining,0,NULL);
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

//binary splitting with small internal steps based on Parkinson+08, Bensen+16, Qiu+20 (Darkforest)
//This code was mostly taken from Darkforest (Qiu+20)
//NOTE: some unused variables here
//TODO: optimize sqrt 2 etc
//TODO: make it work with non-EPS HMF, either with parameter setting OR implement directly
int stoc_split_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    double G0 = 1;
    double gamma1 = 0;
    double gamma2 = 0;
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
    int idx_first = -1;
    int idx = 0;

    double growthf = hs_constants->growth_out;
    //TODO_finish growth of d (i.e use the J function to speed up)
    double growth_d = 0;
    float d_points[MAX_HALO_CELL], m_points[MAX_HALO_CELL];
    int n_points;
    //set initial points
    d_points[0] = hs_constants->delta / growthf;
    m_points[0] = hs_constants->M_cond;
    d_target = Deltac / growthf;
    n_points = 1;

    sigma_res = EvaluateSigma(lnm_res, 0, NULL);
    sigmasq_res = sigma_res*sigma_res;

    // LOG_DEBUG("Starting split %.2e %.2e",d_points[0],m_points[0]);

    while(idx < n_points) {
        d_start = d_points[idx];
        m_start = m_points[idx];
        lnm_start = log(m_start);
        dd_target = d_target - d_start;
        save = 0;
        // Compute useful quantites
        m_half = 0.5*m_start;
        lnm_half = log(m_half);
        sigma_start = EvaluateSigma(lnm_start,0,NULL);
        sigmasq_start = sigma_start*sigma_start;
        sigma_half = EvaluateSigma(lnm_half,1,&alpha_half);
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
            //TODO: look at the J function in Parkinson+08 for a speedup
            //F = ComputeFraction(sigma_start, sigmasq_start, sigmasq_res, G1, dd, kit_sp);
            growth_d = Deltac/(d_start + dd);
            F = 1 - FgtrM_bias_fast(growth_d,d_start,sigma_res,sigma_start);
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
            //TODO: look at the J function in Parkinson+08 for a speedup
            growth_d = Deltac/(d_start + dd);
            F = 1 - FgtrM_bias_fast(growth_d,d_start,sigma_res,sigma_start);
            // Generate random numbers and split the tree
            if (gsl_rng_uniform(rng) < N_upper) {
                q = pow(pow(q_res, eta) + pow_diff*gsl_rng_uniform(rng), 1./eta);
                m_q = q*m_start;
                sigma_q = EvaluateSigma(log(m_q),1,&alpha_q);
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
        // LOG_DEBUG("split i %d n %d m %.2e d %.2e",idx,n_points,m_start,d_start);
        // LOG_DEBUG("q %.2e dd %.2e (%.2e %.2e) of %.2e",q,dd,dd1,dd2,dd_target);
        // LOG_DEBUG("dNdd %.2e B %.2e pow %.2e eta %.2e ah %.2e G2 %.2e b %.2e",dN_dd,B,pow_diff,eta,alpha_half,G2,beta);
        // Compute progenitor mass
        m_prog1 = (1 - F - q)*m_start;
        m_prog2 = q*m_start;
        //if this branch is finished, add to the output array
        if (save) {
            if (m_prog1 > m_res) {
                M_out[n_out++] = m_prog1;
            }
            if (m_prog2 > m_res) {
                M_out[n_out++] = m_prog2;
            }
        }
        //if not finished yet, add them to the internal arrays
        //We don't need the point at idx anymore, so we can put the first progenitor
        //at the start point, and the second at the end
        //since the first is always more massive, this saves memory
        //TODO: this still drifts by the number of saved halos, figure out how to
        //  keep the active halo at zero until the end, but that's minor as it should only drift a few dozen
        else {
            if (m_prog1 > m_res){
                d_points[idx] = dd + d_start;
                m_points[idx] = m_prog1;
                idx--;
            }
            if (m_prog2 > m_res){
                d_points[n_points] = dd + d_start;
                m_points[n_points++] = m_prog2;
            }
        }
        idx++;
    }
    *n_halo_out = n_out;
    return 0;
}

int stoc_sample(struct HaloSamplingConstants * hs_constants, gsl_rng * rng, int *n_halo_out, float *M_out){
    // LOG_ULTRA_DEBUG("Condition M = %.2e exp n,M (%.2e,%.2e), Mmin = %.2e, delta = %.2f upd %d",
    //                 hs_constants->M_cond,hs_constants->expected_N,hs_constants->expected_M,
    //                 hs_constants->M_min,hs_constants->delta,hs_constants->update);
    //TODO: really examine the case for number/mass sampling
    //The poisson sample fails spectacularly for high delta (updates or dense cells)
    //  and excludes the correlation between number and mass (e.g many small halos or few large ones)
    //The mass sample underperforms at low exp_M/M_max by excluding stochasticity in the total collapsed fraction
    //  and excluding larger halos (e.g if exp_M is 0.1*M_max we can effectively never sample the large halos)
    //i.e there is some case for a delta cut between these two methods however I have no intuition for the exact levels
    //TODO: try something like if(exp_N > 10 && exp_M < 0.9*M_cond) stoc_halo_sample();

    int err;
    //If the expected mass is below our minimum saved mass, don't bother calculating
    //NOTE: some of these conditions are redundant with set_consts_cond()
    if(hs_constants->delta <= DELTA_MIN || hs_constants->expected_M < global_params.SAMPLER_MIN_MASS){
        *n_halo_out = 0;
        return 0;
    }
    //if delta is above critical, form one big halo
    if(hs_constants->delta > MAX_DELTAC_FRAC*get_delta_crit(user_params_stoc->HMF,hs_constants->sigma_cond,hs_constants->growth_out)){
        *n_halo_out = 1;

        //using expected instead of overlap here, since the expected fraction approaches 100% for delta -> Detlac
        // the only time this matters is when the cell has overlap with a DexM halo
        //hm_buf[0] = hs_constants->M_cond;
        M_out[0] = hs_constants->expected_M;
        return 0;
    }

    if(global_params.SAMPLE_METHOD == 0 || (global_params.SAMPLE_METHOD == 3 && !hs_constants->update)){
        err = stoc_mass_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(global_params.SAMPLE_METHOD == 1){
        err = stoc_halo_sample(hs_constants, rng, n_halo_out, M_out);
    }
    else if(global_params.SAMPLE_METHOD == 2){
        err = stoc_sheth_sample(hs_constants, rng, n_halo_out, M_out);
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
int build_halo_cats(gsl_rng **rng_arr, double redshift, float *dens_field, struct HaloField *halofield_large, struct HaloField *halofield_out, struct HaloSamplingConstants *hs_constants){
    int lo_dim = user_params_stoc->HII_DIM;
    int hi_dim = user_params_stoc->DIM;
    double boxlen = user_params_stoc->BOX_LEN;

    //TODO: rewrite function agruments so I don't need to unpack here
    double Mcell = hs_constants->M_cond;
    double lnMcell = hs_constants->lnM_cond;
    double Mmax_tb = hs_constants->M_max_tables;
    double lnMmax_tb = hs_constants->lnM_max_tb;
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

    //Since the conditional MF is extended press-schecter, we rescale by a factor equal to the ratio of the collapsed fractions (n_order == 1) of the UMF
    //TODO: do this ONLY if we choose to normalise (i.e flesh out all of the CMF options (rescaling, normalising, adjusting, matching))

    double ps_ratio = 1.;
    if(user_params_stoc->HMF>1 && user_params_stoc->HMF<4){
        struct parameters_gsl_MF_integrals params = {
            .redshift = hs_constants->z_out,
            .growthf = growthf,
            .HMF = 0,
        };

        if(user_params_stoc->INTEGRATION_METHOD_HALOS == 1)
            initialise_GL(NGL_INT, lnMmin, lnMcell);

        ps_ratio = IntegratedNdM(lnMmin,lnMcell,params,2,user_params_stoc->INTEGRATION_METHOD_HALOS);
        params.HMF = user_params_stoc->HMF;
        ps_ratio /= IntegratedNdM(lnMmin,lnMcell,params,2,user_params_stoc->INTEGRATION_METHOD_HALOS);
    }

#pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        //PRIVATE VARIABLES
        int x,y,z,i,j;
        int threadnum = omp_get_thread_num();

        int nh_buf=0;
        double delta;
        float prop_buf[2], prop_dummy[2];
        int crd_hi[3];
        double crd_large[3];

        //dexm overlap variables
        double halo_dist,halo_r,intersect_vol;
        double mass_defc=1;

        //buffers per cell
        float hm_buf[MAX_HALO_CELL];

        unsigned long long int count=0;
        unsigned long long int istart = threadnum * arraysize_local;
        //debug total
        double M_cell=0.;

        //we need a private version
        //TODO: its probably better to split condition and z constants
        struct HaloSamplingConstants hs_constants_priv;
        hs_constants_priv = *hs_constants;

        //assign big halos into list first (split amongst ranks)
#pragma omp for
        for (j=0;j<nhalo_in;j++){
            halofield_out->halo_masses[istart+count] = halofield_large->halo_masses[j];
            halofield_out->star_rng[istart+count] = halofield_large->star_rng[j];
            halofield_out->sfr_rng[istart+count] = halofield_large->sfr_rng[j];
            halofield_out->halo_coords[0 + 3*(istart+count)] = halofield_large->halo_coords[0 + 3*j];
            halofield_out->halo_coords[1 + 3*(istart+count)] = halofield_large->halo_coords[1 + 3*j];
            halofield_out->halo_coords[2 + 3*(istart+count)] = halofield_large->halo_coords[2 + 3*j];
            count++;
        }

#pragma omp for
        for (x=0; x<lo_dim; x++){
            for (y=0; y<lo_dim; y++){
                for (z=0; z<HII_D_PARA; z++){
                    delta = (double)dens_field[HII_R_INDEX(x,y,z)] * growthf;
                    stoc_set_consts_cond(&hs_constants_priv,delta);
                    //Subtract mass from cells near big halos
                    mass_defc = 1;
                    for (j=0;j<nhalo_in;j++){
                        //convert to lowres coordinates, no precision lost since its a double
                        crd_large[0] = halofield_large->halo_coords[0 + 3*j] * hi_dim / lo_dim;
                        crd_large[1] = halofield_large->halo_coords[1 + 3*j] * hi_dim / lo_dim;
                        crd_large[2] = halofield_large->halo_coords[2 + 3*j] * hi_dim / lo_dim;
                        //mass subtraction from cell, PRETENDING THEY ARE SPHERES OF RADIUS L_FACTOR in cell widths
                        halo_r = MtoR(halofield_large->halo_masses[j]) / boxlen * lo_dim; //units of HII_DIM cell width
                        halo_dist = sqrt((crd_large[0] - x)*(crd_large[0] - x) +
                                            (crd_large[1] - y)*(crd_large[1] - y) +
                                            (crd_large[2] - z)*(crd_large[2] - z)); //distance between sphere centres

                        //Cell is entirely outside of halo
                        if(halo_dist - L_FACTOR > halo_r){
                            continue;
                        }
                        //Cell is entirely within halo
                        else if(halo_dist + L_FACTOR < halo_r){
                            mass_defc = 0;
                            break;
                        }
                        //partially inside halo, pretend cells are spheres to do the calculation much faster without too much error
                        else{
                            intersect_vol = halo_dist*halo_dist + 2*halo_dist*L_FACTOR - 3*L_FACTOR*L_FACTOR;
                            intersect_vol += 2*halo_dist*halo_r + 6*halo_r*L_FACTOR - 3*halo_r*halo_r;
                            intersect_vol *= PI*(halo_r + L_FACTOR - halo_dist) / 12*halo_dist; //volume in cell_width^3

                            mass_defc -= intersect_vol; //since cell volume == 1, M*mass_defc should adjust the mass correctly

                            //NOTE: it is possible for a single HII_DIM cells to be partially in two halos on the DIM grid
                            //      So we don't break here
                        }
                    }
                    if(x+y+z == 0){
                        print_hs_consts(&hs_constants_priv);
                        LOG_ULTRA_DEBUG("Cell 0: delta %.2f -> (N,M) (%.2f,%.2e) overlap defc %.2f ps_ratio %.2f",
                                        delta,hs_constants_priv.expected_N,hs_constants_priv.expected_M,mass_defc,ps_ratio);
                    }
                    //TODO: the ps_ratio part will need to be moved when other CMF scalings are finished
                    hs_constants_priv.expected_M *= mass_defc/ps_ratio;
                    hs_constants_priv.expected_N *= mass_defc/ps_ratio;

                    stoc_sample(&hs_constants_priv, rng_arr[threadnum], &nh_buf, hm_buf);
                    //output total halo number, catalogues of masses and positions
                    M_cell = 0;
                    for(i=0;i<nh_buf;i++){
                        if(hm_buf[i] < global_params.SAMPLER_MIN_MASS) continue; //save only halos some factor above minimum

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
                        if(x+y+z == 0){
                            LOG_ULTRA_DEBUG("Halo %d Mass %.2e Stellar %.2e SFR %.2e",i,hm_buf[i],prop_buf[0],prop_buf[1]);
                        }
                    }
                    if(x+y+z == 0){
                        LOG_SUPER_DEBUG("Cell 0: delta %.2f | N %d (exp. %.2e) | Total M %.2e (exp. %.2e)",
                                        delta,nh_buf,hs_constants_priv.expected_N,M_cell,hs_constants_priv.expected_M);
                    }
                }
            }
        }

        if(count >= arraysize_local){
            LOG_ERROR("Ran out of memory, with %llu halos and local size %llu",count,arraysize_local);
            Throw(ValueError);
        }

        istart_threads[threadnum] = istart;
        nhalo_threads[threadnum] = count;
    }

    //Condense the sparse array
    //TODO: figure out a way to do this in parralel without overwriting elements before they are moved
    int i=0;
    unsigned long long int count_total = 0;
    for(i=0;i<user_params_stoc->N_THREADS;i++){
        memmove(&halofield_out->halo_masses[count_total],&halofield_out->halo_masses[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->star_rng[count_total],&halofield_out->star_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->sfr_rng[count_total],&halofield_out->sfr_rng[istart_threads[i]],sizeof(float)*nhalo_threads[i]);
        memmove(&halofield_out->halo_coords[3*count_total],&halofield_out->halo_coords[3*istart_threads[i]],sizeof(int)*3*nhalo_threads[i]);
        count_total += nhalo_threads[i];
    }
    halofield_out->n_halos = count_total;

    //replace the rest with zeros for clarity
    memset(&halofield_out->halo_masses[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->halo_coords[3*count_total],0,3*(arraysize_total-count_total)*sizeof(int));
    memset(&halofield_out->star_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    memset(&halofield_out->sfr_rng[count_total],0,(arraysize_total-count_total)*sizeof(float));
    return 0;
}

//TODO: there's a lot of repeated code here and in build_halo_cats, find a way to merge
int halo_update(gsl_rng ** rng_arr, double z_in, double z_out, struct HaloField *halofield_in, struct HaloField *halofield_out, struct HaloSamplingConstants *hs_constants){
    int nhalo_in = halofield_in->n_halos;
    if(z_in >= z_out){
        LOG_ERROR("halo update must go backwards in time!!! z_in = %.1f, z_out = %.1f",z_in,z_out);
        Throw(ValueError);
    }

    //TODO: rewrite function agruments so I don't need to unpack here
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

    LOG_DEBUG("Beginning stochastic halo sampling (update) on %d halos",nhalo_in);
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
        //TODO: its probably better to split condition and z constants
        //also the naming convention should be better between structs/struct pointers
        struct HaloSamplingConstants hs_constants_priv;
        hs_constants_priv = *hs_constants;

#pragma omp for
        for(ii=0;ii<nhalo_in;ii++){
            M2 = halofield_in->halo_masses[ii];
            if(M2 < Mmin || M2 > Mmax_tb){
                LOG_ERROR("Input Mass = %.2e, something went wrong in the input catalogue",M2);
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
                if(prog_buf[jj] < global_params.SAMPLER_MIN_MASS) continue; //save only halos some factor above minimum
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
                LOG_ULTRA_DEBUG(" HMF %d delta %.3f delta_coll %.3f delta_prev %.3f adjusted %.3f",user_params_stoc->HMF,
                                                                                hs_constants_priv.delta,
                                                                                get_delta_crit(user_params_stoc->HMF,hs_constants_priv.sigma_cond,growth_out),
                                                                                get_delta_crit(user_params_stoc->HMF,hs_constants_priv.sigma_cond,growth_in),
                                                                                get_delta_crit(user_params_stoc->HMF,hs_constants_priv.sigma_cond,growth_in)*growth_out/growth_in);
                print_hs_consts(&hs_constants_priv);
                LOG_SUPER_DEBUG("First Halo: Mass %.2f | N %d (exp. %.2e) | Total M %.2e (exp. %.2e)",
                                        M2,n_prog,hs_constants_priv.expected_N,M_prog,hs_constants_priv.expected_M);
            }
        }
        if(count >= arraysize_local){
            LOG_ERROR("Ran out of memory, with %llu halos and local size %llu",count,arraysize_local);
            Throw(ValueError);
        }

        istart_threads[threadnum] = istart;
        nhalo_threads[threadnum] = count;
    }

    //Condense the sparse array
    //TODO: figure out a way to do this in parralel without overwriting elements before they are moved
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
                        , int seed, float redshift_prev, float redshift, float *dens_field, struct HaloField *halos_prev, struct HaloField *halos){
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);

    int n_halo_stoc;
    int i_start,i;

    if(redshift_prev > 0 && halos_prev->n_halos == 0){
        LOG_DEBUG("No halos to update %.2f to %.2f, continuing...",redshift_prev,redshift);
        return 0;
    }

    //set up the rng
    gsl_rng * rng_stoc[user_params->N_THREADS];
    seed_rng_threads(rng_stoc,seed);

    struct HaloSamplingConstants hs_constants;
    stoc_set_consts_z(&hs_constants,redshift,redshift_prev);

    //Fill them
    //NOTE:Halos prev in the first box corresponds to the large DexM halos
    if(redshift_prev < 0.){
        LOG_DEBUG("building first halo field at z=%.1f", redshift);
        build_halo_cats(rng_stoc,redshift,dens_field,halos_prev,halos,&hs_constants);
    }
    else{
        LOG_DEBUG("updating halo field from z=%.1f to z=%.1f | %d", redshift_prev,redshift,halos->n_halos);
        halo_update(rng_stoc,redshift_prev,redshift,halos_prev,halos,&hs_constants);
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

//testing function to print stuff out from python
/* type 0: UMF/CMF value at a list of masses
 * type 1: Integrated CMF in a single condition at multiple masses N(>M)
 * type 2: Integrated CMF in multiple conditions in the entire mass range
 * type 3: Expected CMF from a list of conditions
 * type 4: Halo catalogue and excess mass from a list of conditions (stoc_sample)
 * type 5: Halo catalogue and coordinates from a list of conditions using the grid/structs (halo_update / build_halo_cat level)
 * type 6: N(<M) interpolation table output for a list of masses in one condition
 * type 7: INVERSE N(<M) interpolation table output for a list of masses in one condition */
int my_visible_function(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options
                        , int seed, int n_mass, float *M, double condition, double z_out, double z_in, int type, double *result){
    int status;
    Try{
        //make the global structs
        Broadcast_struct_global_UF(user_params,cosmo_params);
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);

        omp_set_num_threads(user_params->N_THREADS);

        //set up the rng
        gsl_rng * rng_stoc[user_params->N_THREADS];
        seed_rng_threads(rng_stoc,seed);

        if(z_in > 0 && z_out <= z_in && type!=8){
            LOG_DEBUG("update must go back in time z_out=%.2f z_in=%.2f",z_out,z_in);
            Throw(ValueError);
        }
        //gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

        //gsl_rng_set(rseed, random_seed);
        double test;
        int err=0;
        int i,j;

        struct HaloSamplingConstants hs_const_struct;
        struct HaloSamplingConstants *hs_constants = &hs_const_struct;

        LOG_DEBUG("Setting z constants. %.3f %.3f",z_out,z_in);
        stoc_set_consts_z(hs_constants,z_out,z_in);
        // print_hs_consts(hs_constants);
        //set first condition (for some outputs)
        //TODO: rewrite function agruments so I don't need to unpack here
        double Mmax_tb = hs_constants->M_max_tables;
        double lnMmax_tb = hs_constants->lnM_max_tb;
        double Mmin = hs_constants->M_min;
        double lnMmin = hs_constants->lnM_min;
        double growth_out = hs_constants->growth_out;
        double growth_in = hs_constants->growth_in;
        //placeholders for condition vals
        double Mcond,lnMcond,delta;

        struct parameters_gsl_MF_integrals integral_params = {
                .redshift = z_out,
                .growthf = growth_out,
                .HMF = user_params->HMF,
        };

        LOG_DEBUG("TEST FUNCTION: type = %d z = (%.2f,%.2f), Mmin = %.3e, cond = %.3e, M(%d)=[%.2e,%.2e,%.2e...]",type,z_out,z_in,Mmin,condition,n_mass,M[0],M[1],M[2]);

        //Since the conditional MF is press-schecter, we rescale by a factor equal to the ratio of the collapsed fractions (n_order == 1) of the UMF
        double ps_ratio = 1.;
        struct parameters_gsl_MF_integrals params_second = integral_params;
        if(!hs_constants->update && user_params_stoc->HMF>1 && user_params_stoc->HMF<4){
            params_second.HMF = 0;
            ps_ratio = IntegratedNdM(lnMmin,lnMmax_tb,params_second,2,user_params_stoc->INTEGRATION_METHOD_HALOS);
            ps_ratio /= IntegratedNdM(lnMmin,lnMmax_tb,integral_params,2,user_params_stoc->INTEGRATION_METHOD_HALOS);
        }

        if(type==0){
            stoc_set_consts_cond(hs_constants,condition);
            print_hs_consts(hs_constants);
            //using seed to select CMF or UMF since there's no RNG here
            int cmf_flag = (seed==0) ? 1 : 0;

            //parameters for CMF
            double prefactor = RHOcrit * cosmo_params_stoc->OMm * ps_ratio;
            integral_params.delta = hs_constants->delta;
            integral_params.sigma_cond = hs_constants->sigma_cond;

            #pragma omp parallel for
            for(i=0;i<n_mass;i++){
                //conditional ps mass func * pow(M,n_order)
                if((M[i] < Mmin) || (M[i] > MMAX_TABLES) || ((cmf_flag) && (M[i] > Mcond))){
                    test = 0.;
                }
                else{
                    if(cmf_flag){
                        result[i] = c_mf_integrand(log(M[i]), (void*)&integral_params) * prefactor;
                    }
                    else{
                        result[i] = u_mf_integrand(log(M[i]), (void*)&integral_params);
                    }
                }
                LOG_ULTRA_DEBUG(" D %.1e | M1 %.1e | M2 %.1e | d %.1e | s %.1e -> %.1e",
                                growth_out,M[i],hs_constants->M_cond,hs_constants->delta,hs_constants->sigma_cond,result[i]);
            }
        }
        else if(type==1){
            //integrate CMF -> N(Ml<M<Mh) in one condition
            //TODO: make it possible to integrate UMFs
            stoc_set_consts_cond(hs_constants,condition);
            Mcond = hs_constants->M_cond;
            lnMcond = hs_constants->lnM_cond;
            delta = hs_constants->delta;

            integral_params.delta = hs_constants->delta;
            integral_params.sigma_cond = hs_constants->sigma_cond;

            double lnM_hi, lnM_lo;
            #pragma omp parallel for private(test,lnM_hi,lnM_lo) num_threads(user_params->N_THREADS)
            for(i=0;i<n_mass;i++){

                lnM_lo = log(M[i]) < lnMmin ? lnMmin : log(M[i]);
                lnM_hi = log(M[i+n_mass]) > lnMcond ? lnMcond : log(M[i+n_mass]);

                if (lnM_lo > lnMcond || lnM_hi < lnMmin){
                    result[i] = 0;
                    continue;
                }

                //WARNING: SUPER INEFFICIENT
                if(user_params_stoc->INTEGRATION_METHOD_HALOS == 1)
                    initialise_GL(NGL_INT, lnM_lo, lnM_hi);

                test = IntegratedNdM(lnM_lo,lnM_hi,integral_params,seed,user_params->INTEGRATION_METHOD_HALOS);
                result[i] = test * Mcond * ps_ratio;

                LOG_ULTRA_DEBUG("%d D %.1e | Ml %.1e | Mu %.1e | Mc %.1e| d %.1e | s %.1e ==> %.8e",
                                i,growth_out,M[i],M[i+n_mass],Mcond,delta,EvaluateSigma(lnMcond,0,NULL),result[i]);
            }
        }

        else if(type==2){
            //intregrate CMF -> N_halos in many conditions
            //TODO: make it possible to integrate UMFs
            //quick hack: seed gives type
            #pragma omp parallel private(test,Mcond,lnMcond,delta) num_threads(user_params->N_THREADS)
            {
                //we need a private version
                //TODO: its probably better to split condition and z constants
                struct HaloSamplingConstants hs_constants_priv;
                struct parameters_gsl_MF_integrals int_params_priv = integral_params;
                hs_constants_priv = *hs_constants;
                double cond,tbl_arg;
                #pragma omp for
                for(i=0;i<n_mass;i++){
                    tbl_arg = hs_constants->update ? log(M[i]) : M[i];
                    if(tbl_arg < Nhalo_table.x_min){
                        result[i] = 0.;
                        continue;
                    }
                    cond = M[i];
                    stoc_set_consts_cond(&hs_constants_priv,cond);
                    Mcond = hs_constants_priv.M_cond;
                    lnMcond = hs_constants_priv.lnM_cond;
                    delta = hs_constants_priv.delta;

                    int_params_priv.delta = hs_constants_priv.delta;
                    int_params_priv.sigma_cond = hs_constants_priv.sigma_cond;

                    //WARNING: SUPER INEFFICIENT
                    if(user_params_stoc->INTEGRATION_METHOD_HALOS == 1)
                        initialise_GL(NGL_INT, lnMmin, lnMcond);

                    test = IntegratedNdM(lnMmin,lnMcond,int_params_priv,seed,user_params->INTEGRATION_METHOD_HALOS);
                    LOG_ULTRA_DEBUG("%d D %.1e | Ml %.1e | Mc %.1e | d %.1e | s %.1e ==> %.8e",
                                    i,growth_out,Mmin,Mcond,delta,EvaluateSigma(lnMcond,0,NULL),test);
                    //conditional MF multiplied by a few factors
                    result[i] = test  * Mcond * ps_ratio;
                }
            }
        }

        //Cell CMF from one cell, given M as cell descendant halos
        //uses a constant mass binning since we use the input for descendants
        else if(type==3){
            double out_cmf[100];
            double out_bins[100];
            int n_bins = 100;
            double prefactor = RHOcrit * cosmo_params_stoc->OMm;
            double test;
            double tot_mass=0;
            double lnMbin_max = hs_constants->lnM_max_tb; //arbitrary bin maximum

            for(i=0;i<n_bins;i++){
                out_cmf[i] = 0;
                out_bins[i] = lnMmin + ((double)i/((double)n_bins-1))*(lnMbin_max - lnMmin);
            }

            #pragma omp parallel num_threads(user_params->N_THREADS) private(j,lnMcond,test) reduction(+:tot_mass)
            {
                //we need a private version
                //TODO: its probably better to split condition and z constants
                double lnM_bin;
                struct HaloSamplingConstants hs_constants_priv;
                struct parameters_gsl_MF_integrals int_params_priv = integral_params;
                double cond;
                hs_constants_priv = *hs_constants;
                #pragma omp for
                for(j=0;j<n_mass;j++){
                    tot_mass += M[j];
                    cond = M[j];
                    stoc_set_consts_cond(&hs_constants_priv,cond);
                    lnMcond = hs_constants_priv.lnM_cond;

                    int_params_priv.sigma_cond = hs_constants_priv.sigma_cond;
                    int_params_priv.delta = hs_constants_priv.delta;
                    for(i=0;i<n_bins;i++){
                        lnM_bin = out_bins[i];

                        //conditional ps mass func * pow(M,n_order)
                        if(lnM_bin < lnMmin || lnM_bin > lnMcond){
                            test = 0.;
                        }
                        else{
                            test = c_mf_integrand(lnM_bin,(void*)&int_params_priv);
                            test = test * prefactor * ps_ratio;
                        }
                        out_cmf[i] += test * M[j];
                    }
                }
            }
            result[0] = n_bins;
            for(i=0;i<n_bins;i++){
                result[1+i] = out_bins[i];
                result[1+i+n_bins] = out_cmf[i] * ps_ratio / VOLUME; //assuming here you pass all the halos in the box
            }
        }

        //halo catalogues + cell sums from multiple conditions, given M as cell descendant halos/cells
        //the result mapping is n_halo_total (1) (exp_n,exp_m,n_prog,m_prog) (n_desc) M_cat (n_prog_total)
        else if(type==4){
            int n_halo_tot=0;
            #pragma omp parallel num_threads(user_params->N_THREADS) private(i,j)
            {
                float out_hm[MAX_HALO_CELL];
                double exp_M,exp_N,M_prog;
                int n_halo,n_halo_out;
                double cond;
                //we need a private version
                //TODO: its probably better to split condition and z constants
                struct HaloSamplingConstants hs_constants_priv;
                hs_constants_priv = *hs_constants;
                #pragma omp for
                for(j=0;j<n_mass;j++){
                    cond = M[j];
                    stoc_set_consts_cond(&hs_constants_priv,cond);
                    stoc_sample(&hs_constants_priv, rng_stoc[omp_get_thread_num()], &n_halo, out_hm);

                    n_halo_out = 0;
                    M_prog = 0;
                    for(i=0;i<n_halo;i++){
                        M_prog += out_hm[i];
                        n_halo_out++;

                        //critical is bad, but this is a test function so eeeehh
                        #pragma omp critical
                        {
                            result[1+4*n_mass+(n_halo_tot++)] = out_hm[i];
                        }
                    }
                    //output descendant statistics
                    result[0*n_mass + 1 + j] = (double)hs_constants_priv.expected_N;
                    result[1*n_mass + 1 + j] = (double)hs_constants_priv.expected_M;
                    result[2*n_mass + 1 + j] = (double)n_halo_out;
                    result[3*n_mass + 1 + j] = (double)M_prog;
                }

                result[0] = (double)n_halo_tot;
            }
        }

        //halo catalogue from list of conditions (Mass for update, delta for !update)
        else if(type==5){
            struct HaloField *halos_in = malloc(sizeof(struct HaloField));
            struct HaloField *halos_out = malloc(sizeof(struct HaloField));
            float *dens_field;

            int nhalo_out;
            int bufsize;

            if(hs_constants->update){
                //NOTE: using n_mass for n_conditions
                //a single coordinate is provided for each halo
                LOG_SUPER_DEBUG("assigning input arrays w %d halos",n_mass);
                init_halo_coords(halos_in,n_mass);
                for(i=0;i<n_mass;i++){
                    // LOG_ULTRA_DEBUG("Reading %d (%d %d %d)...",i,n_mass + 3*i,n_mass + 3*i + 1,n_mass + 3*i + 2);
                    // LOG_ULTRA_DEBUG("M[%d] = %.3e",i,M[i]);
                    // LOG_ULTRA_DEBUG("coords_in[%d] = (%d,%d,%d)",i,(int)(M[n_mass + 3*i + 0]),(int)(M[n_mass + 3*i + 1]),(int)(M[n_mass + 3*i + 2]));
                    halos_in->halo_masses[i] = M[i];
                    halos_in->halo_coords[3*i+0] = (int)(M[n_mass + 3*i + 0]);
                    halos_in->halo_coords[3*i+1] = (int)(M[n_mass + 3*i + 1]);
                    halos_in->halo_coords[3*i+2] = (int)(M[n_mass + 3*i + 2]);
                }

                //Halos_out allocated inside halo_update
                bufsize = n_mass*20 > MAX_HALO_CELL ? n_mass*20 : MAX_HALO_CELL;
                init_halo_coords(halos_out,bufsize);
                halos_out->buffer_size = bufsize;
                halo_update(rng_stoc, z_in, z_out, halos_in, halos_out, hs_constants);
            }
            else{
                //NOTE: halomass_in is linear delta at z = redshift_out
                LOG_SUPER_DEBUG("assigning input arrays w %d (%d) cells",n_mass,HII_TOT_NUM_PIXELS);
                if(n_mass != HII_TOT_NUM_PIXELS){
                    LOG_ERROR("passed wrong size grid num %d HII_DIM^3 %d",n_mass,HII_TOT_NUM_PIXELS);
                    Throw(ValueError);
                }
                dens_field = calloc(n_mass,sizeof(float));
                for(i=0;i<n_mass;i++){
                    dens_field[i] = M[i] / growth_out; //theres a redundant *D(z) / D(z) here but its better than redoing the real functions
                }
                //no large halos
                init_halo_coords(halos_in,0);
                bufsize = n_mass*1e3 > MAX_HALO_CELL ? n_mass*1e3 : MAX_HALO_CELL;
                init_halo_coords(halos_out,bufsize);
                halos_out->buffer_size = bufsize;
                build_halo_cats(rng_stoc, z_out, dens_field, halos_in, halos_out, hs_constants);
                free(dens_field);
            }

            free_halo_field(halos_in);
            free(halos_in);

            nhalo_out = halos_out->n_halos;
            if(nhalo_out > 3){
                LOG_DEBUG("sampling done, %d halos, %.2e %.2e %.2e",nhalo_out,halos_out->halo_masses[0],
                            halos_out->halo_masses[1],halos_out->halo_masses[2]);
            }

            result[0] = nhalo_out + 0.0;
            for(i=0;i<nhalo_out;i++){
                result[1+i] = (double)(halos_out->halo_masses[i]);
                result[nhalo_out+1+3*i] = (double)(halos_out->halo_coords[3*i]);
                result[nhalo_out+2+3*i] = (double)(halos_out->halo_coords[3*i+1]);
                result[nhalo_out+3+3*i] = (double)(halos_out->halo_coords[3*i+2]);
            }
            free_halo_field(halos_out);
            free(halos_out);
        }

        //return dNdM table result for M at a bunch of masses
        else if(type==6){
            double x_in,mass,tbl_arg;
            for(i=0;i<n_mass;i++){
                x_in = M[i];
                tbl_arg = hs_constants->update ? log(x_in) : x_in;
                if(i==0) print_hs_consts(hs_constants);
                if(tbl_arg < Nhalo_table.x_min){
                    result[i] = 0.;
                    result[i+n_mass] = 0.;
                    continue;
                }

                //this does the integrals
                stoc_set_consts_cond(hs_constants,x_in);
                result[i] = hs_constants->expected_N;
                result[i+n_mass] = hs_constants->expected_M;

                LOG_ULTRA_DEBUG("x_in %.6e (%.6e) N %.6e M = %.6e",M[i],result[i],result[i+n_mass]);
            }
        }

        //return dNdM INVERSE table result for M at a bunch of probabilities
        else if(type==7){
            double y_in,x_in;
            stoc_set_consts_cond(hs_constants,condition);
            print_hs_consts(hs_constants);
            x_in = hs_constants->cond_val;
            #pragma omp parallel for private(test,y_in)
            for(i=0;i<n_mass;i++){
                y_in = M[i];
                test = 0.;
                LOG_ULTRA_DEBUG("dNdM inverse table: cond %.6e/%.6e, p = %.6e",condition,x_in,y_in);
                //limits
                if(hs_constants->update){
                    if(y_in < 0.) test = hs_constants->lnM_min;
                    else if(y_in > 1.) test = hs_constants->lnM_cond;
                }
                else{
                    if(y_in >= 0) test = hs_constants->lnM_min;
                    else if(y_in <= global_params.MIN_LOGPROB) test = hs_constants->lnM_cond;
                }
                if(test==0){
                    test = EvaluateRGTable2D(x_in,y_in,&Nhalo_inv_table);
                }
                result[i] = exp(test);
                LOG_ULTRA_DEBUG("lnM = %.6e",test);
            }
        }
        else if(type==8){
            double R = z_out;
            double mfp = z_in;
            LOG_DEBUG("Starting mfp filter");
            test_mfp_filter(user_params,cosmo_params,astro_params,flag_options,M,R,mfp,result);
        }
        else if(type==9){
            double delta_in;
            double F_buf,N_buf,S_buf;
            double Mlim_Fstar = Mass_limit_bisection(hs_constants->M_min, hs_constants->M_max_tables,
                                                        astro_params->ALPHA_STAR, astro_params->F_STAR10);
            double Mlim_Fesc = Mass_limit_bisection(hs_constants->M_min, hs_constants->M_max_tables,
                                                        astro_params->ALPHA_ESC, astro_params->F_ESC10);
            if(hs_constants->update){
                LOG_ERROR("no update for type==9 (FgtrM test)");
                Throw(ValueError);
            }
            for(i=0;i<n_mass;i++){
                delta_in = M[i];
                stoc_set_consts_cond(hs_constants,delta_in);
                F_buf = FgtrM_bias_fast(hs_constants->growth_out,delta_in,hs_constants->sigma_min,
                                        hs_constants->sigma_cond);

                N_buf = Nion_ConditionalM(hs_constants->growth_out, hs_constants->lnM_min, hs_constants->lnM_cond, hs_constants->sigma_cond,
                                         delta_in, astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->ALPHA_ESC,
                                         astro_params->F_STAR10, astro_params->F_ESC10,Mlim_Fstar, Mlim_Fesc, user_params_stoc->INTEGRATION_METHOD_ATOMIC);

                S_buf = Nion_ConditionalM(hs_constants->growth_out, hs_constants->lnM_min, hs_constants->lnM_cond, hs_constants->sigma_cond,
                                         delta_in, astro_params->M_TURN, astro_params->ALPHA_STAR, 0.,
                                         astro_params->F_STAR10, 1., Mlim_Fstar, 0., user_params_stoc->INTEGRATION_METHOD_ATOMIC);

                result[i] = F_buf;
                result[i + n_mass] = hs_constants->expected_M / hs_constants->M_cond;
                result[i + 2*n_mass] = N_buf;
                result[i + 3*n_mass] = S_buf;
            }
        }
        else{
            LOG_ERROR("Unknown output type");
            Throw(ValueError);
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
