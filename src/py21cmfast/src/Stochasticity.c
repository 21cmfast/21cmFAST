/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//BIG TODO: sort out single/double precision all the way through
//STYLE: make the function names consistent, re-order functions
//so it makes sense and make a map of the call trees to look for modularisation

//max number of attempts for mass tolerance before failure
#define MAX_ITERATIONS 1e4
#define MAX_ITER_N 1 //for stoc_halo_sample (select N halos) how many tries for one N
#define MASS_TOL 0.4 //mass tolerance for sampling
#define MMAX_TABLES 1e14

//max halos in memory for test functions
//buffer size (per cell of arbitrary size) in the sampling function
//this is big enough for a ~20Mpc mean density cell
//TODO: both of these should be set depending on size, resolution & min mass
//HOWEVER I'd probably have to move the cell arrays to the heap in that case
#define MAX_HALO_CELL (int)1e5
//Max halo in entire box
//100 Mpc^3 box should have ~80,000,000 halos at M_min=1e7, z=6 so this should cover up to ~250 Mpc^3
#define MAXHALO_FACTOR 2 //safety factor in halo arrays (accounts for imbalance and oversampling)
#define MAX_DELTAC_FRAC (float)0.995 //max delta/deltac for interpolation tables / integrals

//do not save halos below this number * minimum sampled halo mass (given by below and M_TURN)
#define HALO_SAMPLE_FACTOR 2
//minimum halo mass to sample below M_TURN
#define HALO_MTURN_FACTOR 16

//NOTE: increasing interptable dimensions has a HUGE impact on performance
#define N_MASS_INTERP (int)500 // number of log-spaced mass bins in interpolation tables
#define N_DELTA_INTERP (int)500 // number of log-spaced overdensity bins in interpolation tables
#define N_PROB_INTERP (int)500 // number of log-spaced probability bins in interpolation tables
#define MIN_LOGPROB -12 //minimum log probability value in interptables, -12 --> 6e-6 (almost always after the turnover)

struct AstroParams *astro_params_stoc;
struct CosmoParams *cosmo_params_stoc;
struct UserParams *user_params_stoc;
struct FlagOptions *flag_options_stoc;

struct parameters_gsl_MF_con_int_{
    double redshift;
    double growthf;
    double delta;
    double n_order;
    double sigma_max;
    double M_max;
    int HMF;
};

//Modularisation that should be put in ps.c for the evaluation of sigma
double EvaluateSigma(double lnM, double *dsigmadm){
    //using log units to make the fast option faster and the slow option slower
    double sigma;
    float MassBinLow;
    int MassBin;
    double dsigma_val;

    //all this stuff is defined in ps.c and initialised with InitialiseSigmaInterpTable
    //NOTE: The interpolation tables are `float` in ps.c
    if(user_params_ps->USE_INTERPOLATION_TABLES) {
        MassBin = (int)floor( (lnM - MinMass )*inv_mass_bin_width );
        MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        sigma = Sigma_InterpTable[MassBin] + ( lnM - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
        
        dsigma_val = dSigmadm_InterpTable[MassBin] + ( lnM - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
        *dsigmadm = -pow(10.,dsigma_val);
    }
    else {
        sigma = sigma_z0(exp(lnM));
        *dsigmadm = dsigmasqdm_z0(exp(lnM));
    }

    return sigma;
}

/*
 copied from ps.c but with interpolation tables and growthf
 Expects the same delta as dNdM_conditional i.e linear delta at z=0
 TODO: put this and EvaluateSigma in ps.c
 */
double EvaluateFgtrM(double growthf, double lnM, double del_bias, double sig_bias){
    double del, sig, sigsmallR;
    double dummy;

    //LOG_ULTRA_DEBUG("FgtrM: z=%.2f M=%.3e d=%.3e s=%.3e",z,exp(lnM),del_bias,sig_bias);

    sigsmallR = EvaluateSigma(lnM,&dummy);

    //LOG_ULTRA_DEBUG("FgtrM: SigmaM %.3e",sigsmallR);

    del = (Deltac - del_bias)/growthf;

    if(del < 0){
        LOG_ERROR("error in FgtrM: condition sigma %.3e delta %.3e arg sigma %.3e delta %.3e",sig_bias,del_bias,sigsmallR,Deltac);
        Throw(ValueError);
    }
    //sometimes condition mass is close enough to minimum mass such that the sigmas are the same to float precision
    //In this case we just throw away the halo, since it is very unlikely to have progenitors
    if(sigsmallR <= sig_bias){
        return 0.;
    }

    sig = sqrt(sigsmallR*sigsmallR - sig_bias*sig_bias);

    return splined_erfc(del / (sqrt(2)*sig));
}

void Broadcast_struct_global_STOC(struct UserParams *user_params, struct CosmoParams *cosmo_params,struct AstroParams *astro_params, struct FlagOption *flag_options){
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
    int num_int = INT_MAX/16;
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

//CONDITIONAL MASS FUNCTION COPIED FROM PS.C, CHANGED TO DOUBLE PRECISION
double dNdM_conditional_double(double growthf, double M1, double M2, double delta1, double delta2, double sigma2){

    double sigma1, dsigmadm,dsigma_val;
    double MassBinLow;
    int MassBin;

    sigma1 = EvaluateSigma(M1,&dsigmadm); //WARNING: THE SIGMA TABLE IS STILL SINGLE PRECISION

    M1 = exp(M1);
    M2 = exp(M2);

    sigma1 = sigma1*sigma1;
    sigma2 = sigma2*sigma2;

    dsigmadm = dsigmadm/(2.0*sigma1); // This is actually sigma1^{2} as calculated above, however, it should just be sigma1. It cancels with the same factor below. Why I have decided to write it like that I don't know!

    if((sigma1 > sigma2)) {

        return -(( delta1 - delta2 )/growthf)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*growthf*growthf*( sigma1 - sigma2 ) ) ) )/(pow( sigma1 - sigma2, 1.5));
    }
    else if(sigma1==sigma2) {

        return -(( delta1 - delta2 )/growthf)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*growthf*growthf*( 1.e-6 ) ) ) )/(pow( 1.e-6, 1.5));

    }
    else {
        return 0.;
    }
}

//n_order is here because the variance calc can use these functions too
//remove the variable (n==0) if we remove that calculation
double MnMassfunction(double M, void *param_struct){
    struct parameters_gsl_MF_con_int_ params = *(struct parameters_gsl_MF_con_int_ *)param_struct;
    double mf;
    double growthf = params.growthf;
    double delta = params.delta;
    double n_order = params.n_order;
    double sigma2 = params.sigma_max; //M2 and sigma2 are degenerate, remove one
    double M_filter = params.M_max;
    double z = params.redshift;
    //HMF = user_params.HMF unless we want the conditional in which case its -1
    int HMF = params.HMF;

    double M_exp = exp(M);

    //M1 is the mass of interest, M2 doesn't seem to be used (input as max mass),
    // delta1 is critical, delta2 is current, sigma is sigma(Mmax,z=0)
    //WE WANT DNDLOGM HERE, SO WE ADJUST ACCORDINGLY

    //dNdlnM = dfcoll/dM * M / M * constants
    //All unconditional functions are dNdM, conditional is actually dfcoll dM == dNdlogm * constants
    if(HMF==0) {
        mf = dNdM(growthf, M_exp) * M_exp;
    }
    else if(HMF==1) {
        mf = dNdM_st(growthf, M_exp) * M_exp;
    }
    else if(HMF==2) {
        mf = dNdM_WatsonFOF(growthf, M_exp) * M_exp;
    }
    else if(HMF==3) {
        mf = dNdM_WatsonFOF_z(z, growthf, M_exp) * M_exp;
    }
    else if(HMF==-1) {
        mf = dNdM_conditional_double(growthf,M,M_filter,Deltac,delta,sigma2);
    }
    else {
        return -1;
    }
    //norder for expectation values of M^n
    return pow(M_exp,n_order) * mf;
}

//copied mostly from the Nion functions
//I might be missing something like this that already exists somewhere in the code
//TODO: rename since its now an integral of M * dfcoll/dm = dNdM / [(RHOcrit * (1+delta) / sqrt(2.*PI) * cosmo_params_stoc->OMm)]
double IntegratedNdM(double growthf, double M1, double M2, double M_filter, double delta, double n_order, int HMF){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = FRACT_FLOAT_ERR*64; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    
    double dummy;
    double sigma = EvaluateSigma(M_filter,&dummy);

    struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
        .growthf = growthf,
        .delta = delta,
        .n_order = n_order,
        .sigma_max = sigma,
        .M_max = M_filter,
        .HMF = HMF,
    };
    int status;

    F.function = &MnMassfunction;
    F.params = &parameters_gsl_MF_con;
    lower_limit = M1;
    upper_limit = M2;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%.3e upper_limit=%.3e (%.3e) rel_tol=%.3e result=%.3e error=%.3e",lower_limit,upper_limit,exp(upper_limit),rel_tol,result,error);
        LOG_ERROR("data: growthf=%.3e M2=%.3e delta=%.3e sigma2=%.3e HMF=%.3d order=%.3e",growthf,exp(M_filter),delta,sigma,HMF,n_order);
        //LOG_ERROR("data: growthf=%e M2=%e delta=%e,sigma2=%e",parameters_gsl_MF_con.growthf,parameters_gsl_MF_con.M_max,parameters_gsl_MF_con.delta,parameters_gsl_MF_con.sigma_max);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return result;
}

gsl_spline2d *Nhalo_spline;
gsl_interp_accel *Nhalo_cond_acc;
#pragma omp threadprivate(Nhalo_cond_acc)
gsl_interp_accel *Nhalo_min_acc;
#pragma omp threadprivate(Nhalo_min_acc)

gsl_spline2d *Nhalo_inv_spline;
gsl_interp_accel *Nhalo_inv_cond_acc;
#pragma omp threadprivate(Nhalo_inv_cond_acc)
gsl_interp_accel *Nhalo_inv_prob_acc;
#pragma omp threadprivate(Nhalo_inv_prob_acc)

double EvaluatedNdMSpline(double x_in, double y_in){
    return gsl_spline2d_eval(Nhalo_spline,x_in,y_in,Nhalo_cond_acc,Nhalo_min_acc);
}

//This table is N(>M | M_in), the CDF of dNdM_conditional
//NOTE: Assumes you give it ymin as the minimum mass TODO: add another argument for Mmin
void initialise_dNdM_tables(double xmin, double xmax, double ymin, double ymax, double growth1, double param, bool update){
    int nx,ny,np;
    int gsl_status;
    double delta,lnM_cond;
    
    LOG_DEBUG("Initialising dNdM Table from [[%.2e,%.2e],[%.2e,%.2e]]",xmin,xmax,ymin,ymax);        
    LOG_DEBUG("D_out %.2e P %.2e up %d",growth1,param,update);
    //halo catalogue update, x=M_filter, delta = Deltac/D1*D2, param2 = growth2
    if(update){
        nx = N_MASS_INTERP;
        delta = Deltac*growth1/param;
        if(delta < -1 || delta > Deltac){
            LOG_ERROR("Invalid delta %.3f",delta);
            Throw(ValueError);
        }
    }
    //generation from grid, x=delta, param_2 = log(M_filt)
    else{
        nx = N_DELTA_INTERP;
        lnM_cond = param;
        if(xmin < -1 || xmax > Deltac){
            LOG_ERROR("Invalid delta [%.3f,%.3f]",xmin,xmax);
            Throw(ValueError);
        }
    }
    //Check for invalid delta

    //y(min mass) is always the same
    ny = N_MASS_INTERP;
    np = N_PROB_INTERP;

    double xa[nx], ya[ny], za[nx*ny];
    double pa[np], za_inv[nx*np];
    double pbuf;

    int i,j,k;
    //set up coordinate grids
    for(i=0;i<nx;i++) xa[i] = xmin + (xmax - xmin)*((double)i)/((double)nx-1);
    for(j=0;j<ny;j++) ya[j] = ymin + (ymax - ymin)*((double)j)/((double)ny-1);
    //for(k=0;k<np;k++) pa[k] = exp(pmin + (pmax - pmin)*((double)k)/((double)np-1));

    //log(1-p) distribution to get the rare halos right
    //NOTE: although the interpolation points are linear in log(1-p),
    //the table itself is linear in p
    for(k=0;k<np-1;k++){
        pbuf = MIN_LOGPROB*(double)k/((double)np-2); //max(log(1-p)) == MIN_LOGPROB
        pbuf = exp(pbuf); // 1-p
        pa[k] = 1 - pbuf;
        //LOG_ULTRA_DEBUG("p = %.8e (%.8e)",pa[k],1 - pa[k]);
    }
    pa[np-1] = 1.;
    
    Nhalo_spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, nx, ny);
    Nhalo_inv_spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, nx, np);

    #pragma omp parallel num_threads(user_params_stoc->N_THREADS) private(i,j,k) firstprivate(delta,lnM_cond)
    {
        double x,y,buf;
        double norm;
        double lnM_prev,lnM_p;
        double prob,prob_prev;

        #pragma omp for
        for(i=0;i<nx;i++){
            x = xa[i];
            //set the condition
            if(update) lnM_cond = x;
            else delta = x;

            lnM_prev = ymin;
            prob_prev = 0.;

            //setting to zero for high delta 
            //this one needs to be done before the norm is calculated
            if(delta > MAX_DELTAC_FRAC*Deltac){
                for(j=0;j<ny;j++)
                    gsl_interp2d_set(Nhalo_spline,za,i,j,0.);
                    
                for(k=0;k<np;k++)
                    gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,k,0.);

                continue;
            }

            norm = IntegratedNdM(growth1,ymin,ymax,lnM_cond,delta,0,-1);
            
            //if the condition has no halos set the dndm table
            //the inverse table will be unaffected since p=0
            if(norm==0){
                for(j=0;j<ny;j++)
                    gsl_interp2d_set(Nhalo_spline,za,i,j,0.);
                    
                for(k=0;k<np;k++)
                    gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,k,0.);
                continue;
            }

            gsl_interp2d_set(Nhalo_spline,za,i,0,0.); //set P(<Mmin) == 0.
            gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,0,ymin); //set P(<Mmin) == 0.
            
            
            for(j=1;j<ny;j++){
                y = ya[j];
                if(lnM_cond <= y){
                    //setting to one guarantees samples at lower mass
                    //This fixes upper mass limits for the conditions
                    buf = norm;
                }
                else{
                    buf = IntegratedNdM(growth1, ymin, y, lnM_cond, delta, 0, -1); //Number density between ymin and y
                }
                //LOG_ULTRA_DEBUG("Int || x: %.2e (%d) y: %.2e (%d) ==> %.8e / %.8e",x,i,exp(y),j,buf,buf/norm);
                gsl_interp2d_set(Nhalo_spline,za,i,j,buf);
                
                prob = buf / norm; //get log-probability
                //catch some norm errors
                if(prob != prob){
                    Throw(ValueError);
                }
                for(k=1;k<np-1;k++){
                    //since we go ascending in y, prob_prev < prob
                    if((prob < pa[k]) || (prob_prev > pa[k])) continue;

                    //probabilities are between the target, interpolate to get target
                    //NOTE: linear interpolation in (lnM,probability)
                    lnM_p = (pa[k]-prob_prev)*(y - lnM_prev)/(prob-prob_prev) + lnM_prev;
                    gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,k,lnM_p);
                    //LOG_ULTRA_DEBUG("Found c: %.2e p: %.2e (c %d, m %d, p %d) z: %.5e",update ? exp(x) : x,pa[k],i,j,k,exp(lnM_p));
                }
                prob_prev = prob;
                lnM_prev = y;
            }
            
            //for halo updates, we want the max to be exactly the condition
            //for the grid, we set the rarest halo to be at MIN_LOGPROB since
            //interpolating in underdense cells up to the entire cell mass
            //vastly overestimates high mass halos.
            //NOTE: at this point, lnM_p is the last Mass interpolated
            //which will be the highest upper integral limit (log(p) = 1 - MIN_LOGPROB)
            if(update)
                gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,np-1,lnM_cond);
            else
                gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,np-1,lnM_p);

            // LOG_ULTRA_DEBUG("P limits (%.2e,%.2e) = (%.2e,%.2e)",pa[0],pa[np-1],
            //                 exp(gsl_interp2d_get(Nhalo_inv_spline,za_inv,i,0)),
            //                 exp(gsl_interp2d_get(Nhalo_inv_spline,za_inv,i,np-1)));
        }

        //initialise and fill the interp table
        //The accelerators in GSL interpolators are not threadsafe, so we need one per thread.
        //Since it's not super important which thread has which accelerator, just that they
        //aren't using the same one at the same time, I think this is safe
        Nhalo_min_acc = gsl_interp_accel_alloc();
        Nhalo_cond_acc = gsl_interp_accel_alloc();

        Nhalo_inv_prob_acc = gsl_interp_accel_alloc();
        Nhalo_inv_cond_acc = gsl_interp_accel_alloc();
    }
    gsl_status = gsl_spline2d_init(Nhalo_spline, xa, ya, za, nx, ny);
    GSL_ERROR(gsl_status);
    
    gsl_status = gsl_spline2d_init(Nhalo_inv_spline, xa, pa, za_inv, nx, np);
    GSL_ERROR(gsl_status);

    LOG_DEBUG("Done.");
}

void free_dNdM_tables(){
    gsl_spline2d_free(Nhalo_spline);

    gsl_spline2d_free(Nhalo_inv_spline);

    #pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        gsl_interp_accel_free(Nhalo_cond_acc);
        gsl_interp_accel_free(Nhalo_min_acc);
        
        gsl_interp_accel_free(Nhalo_inv_cond_acc);
        gsl_interp_accel_free(Nhalo_inv_prob_acc);
    }
}

double sample_dndM_inverse(double condition, gsl_rng * rng){
    double p_in;
    // do{
    //     p_in = gsl_rng_uniform(rng);
    // }while(p_in < exp(MIN_LOGPROB));
    p_in = gsl_rng_uniform(rng);
    double res = gsl_spline2d_eval(Nhalo_inv_spline,condition,p_in,Nhalo_inv_cond_acc,Nhalo_inv_prob_acc);

    // if(res > condition){
    //     LOG_WARNING("sampled %.2e from %.2e, p = %.2e",exp(res),exp(condition),p_in);
    // }

    return res;
}

//return the expected number and mass of halos for a given condition
void get_halo_avg(double growth_out, double delta, double lnMmin, double lnMmax, float M_in, bool update, double * exp_N, double * exp_M){
    double tbl_arg,frac,sigma_max,n_exp,dummy;
    if(!update){
        tbl_arg = delta;
    }
    else{
        tbl_arg = log(M_in);
    }

    double M,N;
    
    sigma_max = EvaluateSigma(log(M_in),&dummy);
    frac = EvaluateFgtrM(growth_out,lnMmin,delta,sigma_max);
    n_exp = EvaluatedNdMSpline(tbl_arg,lnMmax);

    N = n_exp * M_in / sqrt(2.*PI);
    M = frac * M_in;

    *exp_N = N;
    *exp_M = M;
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
    z_hi = (int)((z + randbuf) / (double)(lo_dim) * (double)(hi_dim));
    crd_hi[0] = x_hi;
    crd_hi[1] = y_hi;
    crd_hi[2] = z_hi;
}
//set the minimum source mass
//TODO: include MINI_HALOS
double minimum_source_mass(double redshift,struct AstroParams *astro_params, struct FlagOptions * flag_options){
    double Mmin;
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        Mmin = astro_params->M_TURN / HALO_MTURN_FACTOR;
    }
    else {
        if(flag_options->M_MIN_in_Mass) {
            Mmin = astro_params->M_TURN / HALO_MTURN_FACTOR;
        }
        else {
            //set the minimum source mass
            if (astro_params->ION_Tvir_MIN < 9.99999e3) { // neutral IGM
                Mmin = TtoM(redshift, astro_params->ION_Tvir_MIN, 1.22);
            }
            else { // ionized IGM
                Mmin = TtoM(redshift, astro_params->ION_Tvir_MIN, 0.6);
            }
        }
    }
    return Mmin;
}

//This function adds stochastic halo properties to an existing halo
//TODO: this needs to be updated to handle MINI_HALOS and not USE_MASS_DEPENDENT_ZETA flags
int add_halo_properties(gsl_rng *rng, float halo_mass, float redshift, float * output){
    //for now, we just have stellar mass
    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;
    double sigma_sfr = astro_params_stoc->SIGMA_SFR;

    double fstar_mean, f_sample, sm_sample;
    double sfr_mean, sfr_sample;
    double dutycycle_term;

    //duty cycle, TODO: think about a way to explicitly include the binary nature consistently with the updates
    //At the moment, we simply reduce the mean
    dutycycle_term = exp(-astro_params_stoc->M_TURN/halo_mass);

    //This clipping is normally done with the mass_limit_bisection root find.
    //I can't do that here with the stochasticity, since the f_star clipping happens AFTER the sampling
    fstar_mean = f10 * pow(halo_mass/1e10,fa) * dutycycle_term;
    if(sigma_star > 0){
        //sample stellar masses from each halo mass assuming lognormal scatter
        f_sample = gsl_ran_ugaussian(rng);
        
        /* Simply adding lognormal scatter to a delta increases the mean (2* is as likely as 0.5*)
        * We multiply by exp(-sigma^2/2) so that X = exp(mu + N(0,1)*sigma) has the desired mean */
        f_sample = fmin(fstar_mean * exp(-sigma_star*sigma_star/2 + f_sample*sigma_star),1);

        sm_sample = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * f_sample; //f_star is galactic GAS/star fraction, so OMb is needed
    }
    else{
        sm_sample = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * fmin(fstar_mean,1);
    }

    sfr_mean = sm_sample / (astro_params_stoc->t_STAR * t_hubble(redshift));
    if(sigma_sfr > 0){
        sfr_sample = gsl_ran_ugaussian(rng);
        
        //Since there's no clipping on t_STAR (I think...), we can apply the lognormal to SFR directly instead of t_STAR
        sfr_sample = sfr_mean * exp(-sigma_sfr*sigma_sfr/2 + sfr_sample*sigma_sfr);
    }
    else{
        sfr_sample = sfr_mean;
    }

    //LOG_ULTRA_DEBUG("HM %.3e | SM %.3e | SFR %.3e (%.3e) | F* %.3e (%.3e) | duty %.3e",halo_mass,sm_sample,sfr_sample,sfr_mean,f_sample,fstar_mean,dutycycle_term);

    output[0] = sm_sample;
    output[1] = sfr_sample;

    return 0;
}

//props_in has form: M*, SFR, ....
int update_halo_properties(gsl_rng * rng, float redshift, float redshift_prev, float halo_mass, float halo_mass_prev, float *props_in, float *output){
    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;
    double sigma_sfr = astro_params_stoc->SIGMA_SFR;
    double corr_star = astro_params_stoc->CORR_STAR;
    double corr_sfr = astro_params_stoc->CORR_SFR;
    double interp_star, interp_sfr;

    //sample new properties (uncorrelated)
    add_halo_properties(rng, halo_mass, redshift, output);

    if(corr_star > 0){
        interp_star = exp(-(redshift - redshift_prev)/corr_star);
    }
    else{
        interp_star = 0;
    }
    if(corr_sfr > 0){
        interp_sfr = exp(-(redshift - redshift_prev)/corr_sfr);
    }
    else{
        interp_sfr = 0;
    }
    float x1,x2,mu1,mu2;

    //STELLAR MASS: get median from mean + lognormal scatter (we leave off a bunch of constants and use the mean because we only need the ratio)
    mu1 = fmin(f10 * pow(halo_mass_prev/1e10,fa),1) * halo_mass_prev;
    mu2 = fmin(f10 * pow(halo_mass/1e10,fa),1) * halo_mass;
    //The same CDF value will be given by the ratio of the means/medians, since the scatter is z and M-independent
    x1 = props_in[0];
    x2 = mu2/mu1*x1;
    //interpolate between uncorrelated and matched properties.
    output[0] = (1-interp_star)*output[0] + interp_star*x2;

    //repeat for all other properties
    //SFR: get median (TODO: if I add z-M dependent scatters I will need to re-add the constants)
    mu1 = props_in[0] / t_hubble(redshift_prev);
    mu2 = output[0] / t_hubble(redshift);
    //calculate CDF(prop_prev|conditions) at previous snapshot (lognormal)
    x1 = props_in[1];
    x2 = mu2/mu1*x1;
    //interpolate between uncorrelated and matched properties.
    output[1] = (1-interp_sfr)*output[1] + interp_sfr*x2;

    //repeat for all other properties

    return 0;
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

    //loop through the halos and assign properties
    int i;
    //TODO: update buffer when adding halo properties, make a #define or option with the number
    float buf[2];
#pragma omp parallel for private(buf)
    for(i=0;i<nhalos;i++){
        add_halo_properties(rng_stoc[omp_get_thread_num()], halos->halo_masses[i], redshift, buf);
        halos->stellar_masses[i] = buf[0];
        halos->halo_sfr[i] = buf[1];
        //if(i<30) LOG_ULTRA_DEBUG("Halo %d, sm = %.3e, sfr = %.3e",i,buf[0],buf[1]);
    }

    free_rng_threads(rng_stoc);
    return 0;
}

/* Creates a realisation of halo properties by sampling the halo mass function and 
 * conditional property PDFs, the number of halos is poisson sampled from the integrated CMF*/
int stoc_halo_sample(double growth_out, double M_min, double M_max, double delta, double exp_N, double exp_M, bool update, gsl_rng * rng, int *n_halo_out, float *M_out){
    //delta check, if delta > deltacrit, make one big halo, if <-1 make no halos
    //both of these are possible with Lagrangian linear evolution
    if(delta > Deltac){
        *n_halo_out = 1;
        M_out[0] = M_max;
        return 0;
    }
    if(delta < -1 || exp_M < M_min){
        *n_halo_out = 0;
        return 0;
    }
    double hm_sample,tbl_arg;
    double lnMmin = log(M_min);
    double lnMmax = log(M_max);
    double M_prog;
    int ii, nh;
    int n_attempts=0, n_failures=0;
    int halo_count=0;

    //setup the condition
    if(update)
        tbl_arg = lnMmax;
    else
        tbl_arg = delta;

    for(n_failures=0;n_failures<MAX_ITERATIONS;n_failures++){
        n_attempts = 0;
        nh = 0;
        while(nh==0 || M_min*nh > exp_M){
            nh = gsl_ran_poisson(rng,exp_N);
        }
        for(n_attempts=0;n_attempts<MAX_ITER_N;n_attempts++){
            M_prog = 0;
            halo_count = 0;
            for(ii=0;ii<nh;ii++){
                hm_sample = sample_dndM_inverse(tbl_arg,rng);
                hm_sample = exp(hm_sample);
                M_prog += hm_sample;
                if(hm_sample > HALO_SAMPLE_FACTOR*M_min)
                    M_out[halo_count++] = hm_sample;
            }
            //LOG_ULTRA_DEBUG("attempt %d M=%.3e [%.3e, %.3e]",n_attempts,M_prog,exp_M*(1-MASS_TOL),exp_M*(1+MASS_TOL));
            if((M_prog < exp_M*(1+MASS_TOL)) && (M_prog > exp_M*(1-MASS_TOL))){
                //using goto to break double loop
                goto found_halo_sample;
            }
        }
    }

    //technically I don't need the if statement but it might be confusing otherwise
    if(n_failures >= MAX_ITERATIONS){
        LOG_ERROR("passed max iter in sample");
        Throw(ValueError);
    }

    found_halo_sample: *n_halo_out = halo_count;
    // LOG_ULTRA_DEBUG("Got %d (exp. %.2e) halos mass %.2e (exp. %.2e) %.2f | (%d,%d) att.",
    //                 nh,exp_N,M_prog,exp_M,M_prog/exp_M - 1, n_failures, n_attempts);
    return 0;
}

/* Creates a realisation of halo properties by sampling the halo mass function and 
 * conditional property PDFs, Sampling is done until there is no more mass in the condition
 * Stochasticity is ignored below a certain mass threshold*/
int stoc_mass_sample(double growth_out, double M_min, double M_max, double delta, double exp_N, double exp_M, bool update, gsl_rng * rng, int *n_halo_out, float *M_out){
    double tbl_arg;

    //lnMmin only used for sampling, apply factor here
    double lnMmin = log(M_min);
    double lnMmax = log(M_max);

    if(exp_M <= M_min){
        *n_halo_out = 0;
        return 0;
    }
    
    int n_halo_sampled, n_failures=0;
    double M_prog=0;
    double M_sample;

    if(update)
        tbl_arg = lnMmax;
    else
        tbl_arg = delta;

    for(n_failures=0;n_failures<MAX_ITERATIONS;n_failures++){
        n_halo_sampled = 0;
        M_prog = 0;
        while(M_prog < exp_M){
            M_sample = sample_dndM_inverse(tbl_arg,rng);
            M_sample = exp(M_sample);

            M_prog += M_sample;
            if(M_sample > HALO_SAMPLE_FACTOR * M_min)
                M_out[n_halo_sampled++] = M_sample;
            // LOG_ULTRA_DEBUG("Sampled %.3e | %.3e %d",M_sample,M_prog,n_halo_sampled);
        }
        //if the sample without the last halo is closer to the available mass take it instead
        if(-(M_prog - M_sample - exp_M) < M_prog - exp_M){
            n_halo_sampled--;
            M_prog -= M_sample;
        }
        
        //LOG_ULTRA_DEBUG("attempt %d M=%.3e [%.3e, %.3e]",n_failures,M_prog,exp_M*(1-MASS_TOL),exp_M*(1+MASS_TOL));
        //enforce some level of mass conservation
        if((M_prog < exp_M*(1+MASS_TOL)) && (M_prog > exp_M*(1-MASS_TOL))){
                //using goto to break double loop
                goto found_halo_sample;
        }
    }

    //technically I don't need the if statement but it might be confusing otherwise
    if(n_failures >= MAX_ITERATIONS){
        LOG_ERROR("passed max iter in sample, last attempt M=%.3e [%.3e, %.3e] Me %.3e Mt %.3e ln %.3e",M_prog,exp_M*(1-MASS_TOL),exp_M*(1+MASS_TOL),exp_M,M_max,lnMmax);
        Throw(ValueError);
    }
    
    found_halo_sample: *n_halo_out = n_halo_sampled;
    // LOG_ULTRA_DEBUG("Got %d (exp. %.2e) halos mass %.2e (exp. %.2e) %.2f | %d att",
    //                 n_halo_sampled,exp_N,M_prog,exp_M,M_prog/exp_M - 1,n_failures);
    return 0;
}

int stoc_sample(double growth_out, double M_min, double M_max, double delta, double exp_N, double exp_M, bool update, gsl_rng * rng, int *n_halo_out, float *M_out){
    //LOG_ULTRA_DEBUG("Condition M = %.2e (%.2e), Mmin = %.2e, delta = %.2f",M_max,exp_M,M_min,delta);
    return stoc_mass_sample(growth_out, M_min, M_max, delta, exp_N, exp_M, update, rng, n_halo_out, M_out);
    //return stoc_halo_sample(growth_out, M_min, M_max, delta, exp_N, exp_M, update, rng, n_halo_out, M_out);
}

// will have to add properties here and output grids, instead of in perturbed
int build_halo_cats(gsl_rng **rng_arr, double redshift, float *dens_field, struct HaloField *halofield_large, struct HaloField *halofield_out){    
    double growthf = dicke(redshift);
    int lo_dim = user_params_stoc->HII_DIM;
    int hi_dim = user_params_stoc->DIM;
    double boxlen = user_params_stoc->BOX_LEN;
    //cell size for smoothing / CMF calculation
    double cell_size_R = boxlen / lo_dim * L_FACTOR;
    double ps_ratio = 1.;

    double Mmax = RtoM(cell_size_R);
    double lnMmax = log(Mmax);
    double lnMmax_tb = log(MMAX_TABLES); //wiggle room for tables
    double Mmin;

    Mmin = minimum_source_mass(redshift,astro_params_stoc,flag_options_stoc);
    double lnMmin = log(Mmin);

    int nhalo_in = halofield_large->n_halos;
    
    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,MMAX_TABLES);
    }
    initialise_dNdM_tables(-1, Deltac, lnMmin, lnMmax_tb, growthf, lnMmax, false);
    
    double prefactor = boxlen * boxlen * boxlen;
    double expected_nhalo = prefactor * IntegratedNdM(growthf,log(Mmin*HALO_SAMPLE_FACTOR),lnMmax_tb,lnMmax_tb,0,0,user_params_stoc->HMF);
    int localarray_size = MAXHALO_FACTOR * (int)expected_nhalo / user_params_stoc->N_THREADS; //integer division should be fine here

    LOG_DEBUG("Beginning stochastic halo sampling on %d ^3 grid",lo_dim);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e,volume = %.3e, R = %.3e D = %.3e",redshift,Mmin,Mmax,Mmax/RHOcrit/cosmo_params_stoc->OMm,cell_size_R,growthf);
    LOG_DEBUG("Expected N_halo: %.3e, array size per thread %d (~%.3e GB total)",expected_nhalo,localarray_size,6.*localarray_size*sizeof(int)*user_params_stoc->N_THREADS/1e9);

    //Since the conditional MF is extended press-schecter, we rescale by a factor equal to the ratio of the collapsed fractions (n_order == 1) of the UMF
    if(user_params_stoc->HMF!=0){
        ps_ratio = (IntegratedNdM(growthf,lnMmin,lnMmax,lnMmax,0,1,0) 
            / IntegratedNdM(growthf,lnMmin,lnMmax,lnMmax,0,1,user_params_stoc->HMF));
    }

    //shared halo count (start at the number of big halos)
    int count_total = 0;
    int istart_local[user_params_stoc->N_THREADS];
    memset(istart_local,0,sizeof(istart_local));
    //by default heap allocations are shared
    float *local_hm;
    float *local_sm;
    float *local_sfr;
    int *local_crd;

#pragma omp parallel num_threads(user_params_stoc->N_THREADS) private(local_hm,local_sm,local_sfr,local_crd)
    {
        //PRIVATE VARIABLES
        int x,y,z,i,j;
        int threadnum = omp_get_thread_num();

        int nh_buf=0;
        double delta;
        float prop_buf[2];
        double exp_M, exp_N;
        int crd_hi[3];
        double halo_dist,halo_r,intersect_vol;
        double mass_defc=1;

        //buffers per cell
        float hm_buf[MAX_HALO_CELL];

        //debug printing
        int print_counter = 0;
        
        int count=0;
        local_hm = calloc(localarray_size,sizeof(float));
        local_sm = calloc(localarray_size,sizeof(float));
        local_sfr = calloc(localarray_size,sizeof(float));
        local_crd = calloc(localarray_size*3,sizeof(int));

        //assign big halos into list first (split amongst ranks)
#pragma omp for
        for (j=0;j<nhalo_in;j++){
            local_hm[count] = halofield_large->halo_masses[j];
            local_sm[count] = halofield_large->stellar_masses[j];
            local_sfr[count] = halofield_large->halo_sfr[j];
            place_on_hires_grid(halofield_large->halo_coords[0 + 3*j], halofield_large->halo_coords[1 + 3*j],
                                halofield_large->halo_coords[2 + 3*j], crd_hi,rng_arr[threadnum]);
            local_crd[0 + 3*count] = crd_hi[0];
            local_crd[1 + 3*count] = crd_hi[1];
            local_crd[2 + 3*count] = crd_hi[2];
            count++;
        }

#pragma omp for
        for (x=0; x<lo_dim; x++){
            for (y=0; y<lo_dim; y++){
                for (z=0; z<lo_dim; z++){
                    //Subtract mass from cells near big halos
                    for (j=0;j<nhalo_in;j++){
                        //reusing the crd_hi array here
                        crd_hi[0] = halofield_large->halo_coords[0 + 3*j];
                        crd_hi[1] = halofield_large->halo_coords[1 + 3*j];
                        crd_hi[2] = halofield_large->halo_coords[2 + 3*j];
                        //mass subtraction from cell, PRETENDING THEY ARE SPHERES OF RADIUS L_FACTOR
                        halo_r = MtoR(halofield_large->halo_masses[j]) / lo_dim * boxlen; //units of cell width
                        halo_dist = sqrt((crd_hi[0] - x)*(crd_hi[0] - x) +
                                            (crd_hi[1] - y)*(crd_hi[1] - y) +
                                            (crd_hi[2] - z)*(crd_hi[2] - z)); //dist between sphere centres

                        //entirely outside of halo
                        if(halo_dist - L_FACTOR > halo_r){
                            mass_defc = 1;
                        }
                        //entirely within halo
                        else if(halo_dist + L_FACTOR < halo_r){
                            mass_defc = 0;
                            break;
                        }
                        //partially inside halo
                        else{
                            intersect_vol = halo_dist*halo_dist + 2*halo_dist*L_FACTOR - 3*L_FACTOR*L_FACTOR;
                            intersect_vol += 2*halo_dist*halo_r + 6*halo_r*L_FACTOR - 3*halo_r*halo_r;
                            intersect_vol *= PI*(halo_r + L_FACTOR - halo_dist) / 12*halo_dist; //volume in cell_width^3

                            mass_defc = 1 - intersect_vol; //since cell volume == 1, M*mass_defc should adjust the mass correctly
                            //due to the nature of DexM it is impossible to be partially in two halos
                            break;
                        }
                    }

                    delta = (double)dens_field[HII_R_INDEX(x,y,z)] * growthf;
                    // LOG_ULTRA_DEBUG("Starting sample %d (%d) with delta = %.2f cell, from %.2e to %.2e | %d"
                    //                  ,x*lo_dim*lo_dim + y*lo_dim + z,lo_dim*lo_dim*lo_dim,delta,Mmin,Mmax,count);

                    //delta check, if delta > deltacrit, make one big halo, if <-1 make no halos
                    //both of these are possible with Lagrangian linear evolution
                    if(delta <= -1) continue;
                    
                    if(delta > MAX_DELTAC_FRAC*Deltac){
                        nh_buf = 1;
                        hm_buf[0] = Mmax;
                    }
                    else{
                        get_halo_avg(growthf, delta, lnMmin, lnMmax, Mmax, false, &exp_N, &exp_M);
                        stoc_sample(growthf, Mmin, Mmax, delta, exp_N/ps_ratio*mass_defc, exp_M/ps_ratio*mass_defc, 0, rng_arr[threadnum], &nh_buf, hm_buf);
                    }
                    if(count + nh_buf > localarray_size){
                        LOG_ERROR("ran out of memory (%d halos vs %d array size)",count+nh_buf,localarray_size);
                        Throw(ValueError);
                    }
                    //output total halo number, catalogues of masses and positions
                    for(i=0;i<nh_buf;i++){
                        add_halo_properties(rng_arr[threadnum], hm_buf[i], redshift, prop_buf);

                        place_on_hires_grid(x,y,z,crd_hi,rng_arr[threadnum]);

                        //fill in arrays now, this should be quick compared to the sampling so critical shouldn't slow this down much

                        local_hm[count] = hm_buf[i];
                        local_sm[count] = prop_buf[0];
                        local_sfr[count] = prop_buf[1];
                        local_crd[0 + 3*count] = crd_hi[0];
                        local_crd[1 + 3*count] = crd_hi[1];
                        local_crd[2 + 3*count] = crd_hi[2];
                        count++;
                    }
                }
            }
        }
#pragma omp atomic update
        count_total += count;
        //this loop exectuted on all threads, we need the start index of each local array
        //i[0] == 0, i[1] == n_0, i[2] == n_0 + n_1 etc...
        for(i=user_params_stoc->N_THREADS-1;i>threadnum;i--){
#pragma omp atomic update
            istart_local[i] += count;
        }

#pragma omp barrier

//allocate the output structure
#pragma omp single
        {
            halofield_out->halo_coords = (int*)calloc(count_total*3,sizeof(int));
            halofield_out->halo_masses = (float*)calloc(count_total,sizeof(float));
            halofield_out->stellar_masses = (float*)calloc(count_total,sizeof(float));
            halofield_out->halo_sfr = (float*)calloc(count_total,sizeof(float));
        }
        //we need each thread to be done here before copying the data (total count, indexing, allocation)
#pragma omp barrier

        int istart = istart_local[threadnum];
        LOG_SUPER_DEBUG("Thread %d has %d of %d halos, concatenating (starting at %d)...",threadnum,count,count_total,istart);
                
        //copy each local array into the struct
        memcpy(halofield_out->halo_masses + istart,local_hm,count*sizeof(float));
        memcpy(halofield_out->stellar_masses + istart,local_sm,count*sizeof(float));
        memcpy(halofield_out->halo_sfr + istart,local_sfr,count*sizeof(float));
        memcpy(halofield_out->halo_coords + istart*3,local_crd,count*sizeof(int)*3);
        
        //free local arrays
        free(local_hm);
        free(local_sm);
        free(local_sfr);
        free(local_crd);
        
    }
    halofield_out->n_halos = count_total;
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
    }
    free_dNdM_tables();

    return 0;
}

//TODO: there's a lot of repeated code here and in build_halo_cats, find a way to merge
int halo_update(gsl_rng ** rng_arr, double z_in, double z_out, struct HaloField *halofield_in, struct HaloField *halofield_out){
    int nhalo_in = halofield_in->n_halos;
    if(z_in >= z_out){
        LOG_ERROR("halo update must go backwards in time!!! z_in = %.1f, z_out = %.1f",z_in,z_out);
        Throw(ValueError);
    }
    if(nhalo_in == 0){
        LOG_DEBUG("No halos to update, continuing...");

        //allocate dummy arrays so we don't get a Bus Error by freeing unallocated pointers
        halofield_out->n_halos = 0;
        halofield_out->halo_coords = (int*)calloc(0,sizeof(int));
        halofield_out->halo_masses = (float*)calloc(0,sizeof(float));
        halofield_out->stellar_masses = (float*)calloc(0,sizeof(float));
        halofield_out->halo_sfr = (float*)calloc(0,sizeof(float));
        return 0;
    }

    double growth_in = dicke(z_in);
    double growth_out = dicke(z_out);
    int lo_dim = user_params_stoc->HII_DIM;
    int hi_dim = user_params_stoc->DIM;
    double boxlen = user_params_stoc->BOX_LEN;
    //cell size for smoothing / CMF calculation
    double cell_size_R = boxlen / lo_dim * L_FACTOR;
    double ps_ratio = 1.;

    double Mmax = RtoM(cell_size_R);
    double lnMmax = log(Mmax);
    double lnMmax_tb = log(MMAX_TABLES); //wiggle room for tables

    double Mmin = minimum_source_mass(z_out,astro_params_stoc,flag_options_stoc);
    double lnMmin = log(Mmin);
    
    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,MMAX_TABLES);
    }
    initialise_dNdM_tables(lnMmin, lnMmax_tb, lnMmin, lnMmax_tb, growth_out, growth_in, true);

    double prefactor = boxlen * boxlen * boxlen;
    double expected_nhalo = prefactor * IntegratedNdM(growth_out,log(Mmin*HALO_SAMPLE_FACTOR),lnMmax_tb,lnMmax_tb,0,0,user_params_stoc->HMF);
    int localarray_size = MAXHALO_FACTOR * (int)expected_nhalo / user_params_stoc->N_THREADS; //integer division should be fine here
    double delta = Deltac * growth_out / growth_in; //crit density at z_in evolved to z_out

    LOG_DEBUG("Beginning stochastic halo sampling (update) on %d halos",nhalo_in);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e, d = %.3e",z_out,Mmin,Mmax,delta);
    LOG_DEBUG("Expected N_halo: %.3e, array size per thread %d (~%.3e GB total)",expected_nhalo,localarray_size,6.*localarray_size*sizeof(int)*user_params_stoc->N_THREADS/1e9);

    int count_total = 0;
    int istart_local[user_params_stoc->N_THREADS];
    memset(istart_local,0,sizeof(istart_local));
    float *local_hm;
    float *local_sm;
    float *local_sfr;
    int *local_crd;

#pragma omp parallel num_threads(user_params_stoc->N_THREADS) private(local_hm,local_sm,local_sfr,local_crd)
    {
        float prog_buf[MAX_HALO_CELL];
        int n_prog;
        double exp_M, exp_N;
        
        float propbuf_in[2];
        float propbuf_out[2];

        int threadnum = omp_get_thread_num();
        float M2,sm2,sfr2;
        int ii,jj;
        int count=0;

        local_hm = calloc(localarray_size,sizeof(float));
        local_sm = calloc(localarray_size,sizeof(float));
        local_sfr = calloc(localarray_size,sizeof(float));
        local_crd = calloc(3*localarray_size,sizeof(int));

#pragma omp for
        for(ii=0;ii<nhalo_in;ii++){
            M2 = halofield_in->halo_masses[ii];
            get_halo_avg(growth_out,delta,lnMmin,lnMmax,M2,true,&exp_N,&exp_M);

            if(exp_M < Mmin) continue;
            
            //find progenitor halos by sampling halo CMF                    
            //NOTE: MAXIMUM HERE (TO LIMIT PROGENITOR MASS) IS THE DESCENDANT MASS
            //The assumption is that the expected fraction of the progenitor
            stoc_sample(growth_out,Mmin,M2,delta,exp_N,exp_M,true,rng_arr[threadnum],&n_prog,prog_buf);
            
            propbuf_in[0] = halofield_in->stellar_masses[ii];
            propbuf_in[1] = halofield_in->halo_sfr[ii];

            //place progenitors in local list
            for(jj=0;jj<n_prog;jj++){
                //sometimes this happens
                if(prog_buf[jj]<Mmin)continue;

                update_halo_properties(rng_arr[threadnum], z_out, z_in, prog_buf[jj], M2, propbuf_in, propbuf_out);    

                local_hm[count] = prog_buf[jj];
                local_crd[3*count + 0] = halofield_in->halo_coords[3*ii+0];
                local_crd[3*count + 1] = halofield_in->halo_coords[3*ii+1];
                local_crd[3*count + 2] = halofield_in->halo_coords[3*ii+2];
                
                local_sm[count] = propbuf_out[0];
                local_sfr[count] = propbuf_out[1];
                count++;
            }
        }
#pragma omp atomic update
        count_total += count;
        //this loop exectuted on all threads, we need the start index of each local array
        //i[0] == 0, i[1] == n_0, i[2] == n_0 + n_1 etc...
        for(ii=user_params_stoc->N_THREADS-1;ii>threadnum;ii--){
#pragma omp atomic update
            istart_local[ii] += count;
        }

//sizes need to be set before allocation
#pragma omp barrier
//allocate the output structure
#pragma omp single
        {
            halofield_out->halo_coords = (int*)calloc(3*count_total,sizeof(int));
            halofield_out->halo_masses = (float*)calloc(count_total,sizeof(float));
            halofield_out->stellar_masses = (float*)calloc(count_total,sizeof(float));
            halofield_out->halo_sfr = (float*)calloc(count_total,sizeof(float));
        }
        LOG_SUPER_DEBUG("Thread %d has %d of %d halos, concatenating (starting at %d)...",threadnum,count,count_total,istart_local[threadnum]);

//we need each thread to be done here before copying the data
#pragma omp barrier
        
        //copy each local array into the struct
        memcpy(halofield_out->halo_masses + istart_local[threadnum],local_hm,count*sizeof(float));
        memcpy(halofield_out->stellar_masses + istart_local[threadnum],local_sm,count*sizeof(float));
        memcpy(halofield_out->halo_sfr + istart_local[threadnum],local_sfr,count*sizeof(float));
        memcpy(halofield_out->halo_coords + istart_local[threadnum]*3,local_crd,count*sizeof(int)*3);

        free(local_sfr);
        free(local_sm);
        free(local_crd);
        free(local_hm);
    }
    halofield_out->n_halos = count_total;
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
    }
    free_dNdM_tables();
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

    //set up the rng
    gsl_rng * rng_stoc[user_params->N_THREADS];
    seed_rng_threads(rng_stoc,seed);

    //fill the hmf to possibly avoid deallocation issues:
    //TODO: actually fill the HMF in the sampling functions
    init_hmf(halos);
    
    //The hmf should already be inited(unused) by the large halos
    // if(halos->first_box)
    //     init_hmf(halos_prev);

    //Fill them
    //NOTE:Halos prev in the first box corresponds to the large DexM halos
    if(halos->first_box){
        LOG_DEBUG("building first halo field at z=%.1f", redshift);
        build_halo_cats(rng_stoc,redshift,dens_field,halos_prev,halos);
    }
    else{
        LOG_DEBUG("updating halo field from z=%.1f to z=%.1f | %d", redshift_prev,redshift,halos->n_halos);
        halo_update(rng_stoc,redshift_prev,redshift,halos_prev,halos);
    }

    LOG_DEBUG("Found %d Halos", halos->n_halos);

    if(halos->n_halos > 3){
        LOG_DEBUG("First few Masses:  %11.3e %11.3e %11.3e",halos->halo_masses[0],halos->halo_masses[1],halos->halo_masses[2]);
        LOG_DEBUG("First few Stellar: %11.3e %11.3e %11.3e",halos->stellar_masses[0],halos->stellar_masses[1],halos->stellar_masses[2]);
        LOG_DEBUG("First few SFR:     %11.3e %11.3e %11.3e",halos->halo_sfr[0],halos->halo_sfr[1],halos->halo_sfr[2]);
    }

    free_rng_threads(rng_stoc);
    return 0;
}

//This, for the moment, grids the PERTURBED halo catalogue.
//TODO: make a way to output both types by making 2 wrappers to this function that pass in arrays rather than structs
int ComputeHaloBox(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params
                    , struct FlagOptions * flag_options, struct PerturbedField * perturbed_field, struct PerturbHaloField *halos
                    , struct HaloBox *grids){
    int status;
    Try{
        LOG_DEBUG("Gridding %d halos...",halos->n_halos);

        //get parameters
        Broadcast_struct_global_UF(user_params,cosmo_params);
        Broadcast_struct_global_PS(user_params,cosmo_params);
        double alpha_esc = astro_params->ALPHA_ESC;
        double norm_esc = astro_params->F_ESC10;
        double hm_avg=0,sm_avg=0,sfr_avg=0;
        double hm_expected,sm_expected,sfr_expected;

        double growth_z,dummy;
        double M_min,M_max,lnMmin,lnMmax;
        double alpha_star,norm_star,t_star,Mlim_Fstar,Mlim_Fesc;
        double volume,sigma_max;
        double prefactor_mass,prefactor_sfr,prefactor_star;

        //TODO: interpolation tables (eh this is fast anyway)
        //TODO: PS_RATIO Term

        //calculate expected average halo box, for mean halo box fixing and debugging
        if(!flag_options->HALO_STOCHASTICITY || LOG_LEVEL >= DEBUG_LEVEL){
            init_ps();
            
            M_min = minimum_source_mass(redshift,astro_params,flag_options);
            volume = pow(user_params->BOX_LEN / user_params->HII_DIM,3);
            M_max = RtoM(user_params->BOX_LEN / user_params->HII_DIM * L_FACTOR); //mass in cell of mean dens

            if(user_params->USE_INTERPOLATION_TABLES){
                initialiseSigmaMInterpTable(M_min,global_params.M_MAX_INTEGRAL); //this needs to be initialised above MMax because of Nion_General
            }

            growth_z = dicke(redshift);
            alpha_star = astro_params->ALPHA_STAR;
            norm_star = astro_params->F_STAR10;
            t_star = astro_params->t_STAR;
           
            lnMmax = log(M_max);
            lnMmin = log(M_min);
            sigma_max = EvaluateSigma(lnMmax,&dummy);

            prefactor_mass = volume * RHOcrit * cosmo_params->OMm / sqrt(2.*PI);
            prefactor_star = volume * RHOcrit * cosmo_params->OMb * norm_star * norm_esc;
            prefactor_sfr = volume * RHOcrit * cosmo_params->OMb * norm_star / t_star / t_hubble(redshift);
            
            Mlim_Fstar = Mass_limit_bisection(M_min, M_max, alpha_star, norm_star);
            Mlim_Fesc = Mass_limit_bisection(M_min, M_max, alpha_esc, norm_esc);

            hm_expected = IntegratedNdM(growth_z, lnMmin, lnMmax, lnMmax, 0, 1, user_params->HMF);
            sm_expected = Nion_General(redshift, M_min, astro_params->M_TURN, alpha_star, alpha_esc, norm_star, norm_esc, Mlim_Fstar, Mlim_Fesc);
            sfr_expected = Nion_General(redshift, M_min, astro_params->M_TURN, alpha_star, 0., norm_star, 1., Mlim_Fstar, 0.);

            hm_expected *= volume;
            sm_expected *= prefactor_star;
            sfr_expected *= prefactor_sfr;
        }

        //do the mean HMF box
        //The default 21cmFAST has a strange behaviour where the nonlinear density is used as linear,
        //the condition mass is at mean density, but the total cell mass is multiplied by delta 
        //This part mimics that behaviour
        //TODO: interpolation tables here (although the mean boxes are just a test)
        if(!flag_options->HALO_STOCHASTICITY){
            LOG_DEBUG("Mean halo boxes || Mmin = %.2e | Mmax = %.2e (s=%.2e) | z = %.2e | D = %.2e | vol = %.2e",M_min,M_max,sigma_max,redshift,growth_z,volume);
#pragma omp parallel num_threads(user_params->N_THREADS)
            {
                int i;
                double dens;
                double mass, wstar, sfr, h_count;
#pragma omp for reduction(+:hm_avg,sm_avg,sfr_avg)
                for(i=0;i<HII_TOT_NUM_PIXELS;i++){
                    dens = perturbed_field->density[i];

                    //ignore very low density
                    if(dens <= -1){
                        mass = 0.;
                        wstar = 0.;
                        sfr = 0.;
                        h_count = 0;
                    }
                    //turn into one large halo if we exceed the critical
                    //Since these are perturbed (Eulerian) grids, I use the total cell mass (1+dens)
                    else if(dens>=MAX_DELTAC_FRAC*Deltac){
                        mass = M_max * (1+dens);
                        wstar = M_max * (1+dens) * cosmo_params->OMb / cosmo_params->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) * norm_esc * pow(M_max*(1+dens)/1e10,alpha_esc);
                        sfr = M_max * (1+dens) * cosmo_params->OMb / cosmo_params->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) / t_star / t_hubble(redshift);
                        h_count = 1;
                    }
                    else{
                        //calling IntegratedNdM with star and SFR need special care for the f*/fesc clipping, and calling NionConditionalM for mass includes duty cycle
                        //neither of which I want
                        mass = IntegratedNdM(growth_z,lnMmin,lnMmax,lnMmax,dens,1,-1) * prefactor_mass * (1+dens);
                        h_count = IntegratedNdM(growth_z,lnMmin,lnMmax,lnMmax,dens,0,-1) * prefactor_mass * (1+dens);

                        wstar = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_max, Deltac, dens, astro_params->M_TURN
                                                , astro_params->ALPHA_STAR, astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10
                                                , Mlim_Fstar, Mlim_Fesc, user_params->FAST_FCOLL_TABLES) * prefactor_star * (1+dens);

                        sfr = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_max, Deltac, dens, astro_params->M_TURN
                                                , astro_params->ALPHA_STAR, 0., astro_params->F_STAR10, 1., Mlim_Fstar, 0.
                                                , user_params->FAST_FCOLL_TABLES) * prefactor_sfr * (1+dens);
                    }

                    grids->halo_mass[i] = mass;
                    grids->wstar_mass[i] = wstar;
                    grids->halo_sfr[i] = sfr;
                    grids->count[i] = (int)h_count;
                    
                    hm_avg += mass;
                    sm_avg += wstar;
                    sfr_avg += sfr;
                }
            }
            
            hm_avg /= HII_TOT_NUM_PIXELS;
            sm_avg /= HII_TOT_NUM_PIXELS;
            sfr_avg /= HII_TOT_NUM_PIXELS;

            //This is the mean adjustment that happens in the rest of the code
            int i;
            for(i=0;i<HII_TOT_NUM_PIXELS;i++){
                grids->halo_mass[i] *= hm_expected/hm_avg;
                grids->wstar_mass[i] *= sm_expected/sm_avg;
                grids->halo_sfr[i] *= sfr_expected/sfr_avg;
            }
            
            hm_avg = hm_expected;
            sm_avg = sm_expected;
            sfr_avg = sfr_expected;
        }
        else{
#pragma omp parallel num_threads(user_params->N_THREADS)
            {
                int i_halo,idx,x,y,z;
                double m,wstar,sfr,fesc;
#pragma omp for
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] = 0.0;
                    grids->wstar_mass[idx] = 0.0;
                    grids->halo_sfr[idx] = 0.0;
                    grids->count[idx] = 0;
                }

#pragma omp barrier

#pragma omp for reduction(+:hm_avg,sm_avg,sfr_avg)
                for(i_halo=0; i_halo<halos->n_halos; i_halo++){
                    x = halos->halo_coords[0+3*i_halo]; //NOTE:PerturbedHaloField is on HII_DIM, HaloField is on DIM
                    y = halos->halo_coords[1+3*i_halo];
                    z = halos->halo_coords[2+3*i_halo];

                    m = halos->halo_masses[i_halo];

                    //This clipping is normally done with the mass_limit_bisection root find. 
                    //TODO: It could save some `pow` calls if I compute a mass limit outside the loop
                    fesc = fmin(norm_esc*pow(m/1e10,alpha_esc),1);

                    wstar = halos->stellar_masses[i_halo]*fesc;
                    sfr = halos->halo_sfr[i_halo];
                    //will probably need weighted SFR, unweighted Stellar later on
                    //Lx as well when that scatter is included

    #pragma omp atomic update
                    grids->halo_mass[HII_R_INDEX(x, y, z)] += m;
    #pragma omp atomic update
                    grids->wstar_mass[HII_R_INDEX(x, y, z)] += wstar;
    #pragma omp atomic update
                    grids->halo_sfr[HII_R_INDEX(x, y, z)] += sfr;
    #pragma omp atomic update
                    grids->count[HII_R_INDEX(x, y, z)] += 1;

                    if(LOG_LEVEL >= DEBUG_LEVEL){
                        hm_avg += m;
                        sm_avg += wstar;
                        sfr_avg += sfr;
                    }
                }
            }
            if(LOG_LEVEL >= DEBUG_LEVEL){
                hm_avg /= HII_TOT_NUM_PIXELS;
                sm_avg /= HII_TOT_NUM_PIXELS;
                sfr_avg /= HII_TOT_NUM_PIXELS;
            }
        }

        if(user_params->USE_INTERPOLATION_TABLES && (!flag_options->HALO_STOCHASTICITY || LOG_LEVEL >= DEBUG_LEVEL)){
                freeSigmaMInterpTable();
        }

        LOG_DEBUG("HaloBox Cells:  %9.2e %9.2e %9.2e %9.2e", grids->halo_mass[0], grids->halo_mass[1]
            , grids->halo_mass[2] , grids->halo_mass[3]);
        LOG_DEBUG("Stellar Masses: %9.2e %9.2e %9.2e %9.2e", grids->wstar_mass[0], grids->wstar_mass[1]
            , grids->wstar_mass[2] , grids->wstar_mass[3]);
        LOG_DEBUG("Halo SFR:       %9.2e %9.2e %9.2e %9.2e", grids->halo_sfr[0], grids->halo_sfr[1]
            , grids->halo_sfr[2] , grids->halo_sfr[3]);
            
        LOG_DEBUG("Redshift %.2f: Expected averages: (%11.3e, %11.3e, %11.3e) || Box averages (%11.3e,%11.3e,%11.3e)"
                    ,redshift,hm_expected,sm_expected,sfr_expected,hm_avg,sm_avg,sfr_avg);
    }
    Catch(status){
        return(status);
    }
    LOG_DEBUG("Done.");
    return(0);
    
}

//testing function to print stuff out from python
/* type==0: UMF/CMF
 * type==1: Integrated CMF in a single condition at multiple masses N(>M)
 * type==2: Integrated CMF in multiple conditions in the entire mass range
 * type==3: Halo catalogue from single condition (testing stoc_halo/mass_sample)
 * type==4: Not implemented
 * type==5: halo catalogue given a list of conditions (testing build_halo_cats/halo_update)
 * type==6: output halo mass given probabilities from inverse CMF tables*/
int my_visible_function(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options
                        , int seed, int n_mass, float *M, bool update, double condition, double z_out, double z_in, int type, double *result){
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

        if(update && z_out >= z_in)
            Throw(ValueError);

        //gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

        //gsl_rng_set(rseed, random_seed);
        double test,dummy;
        int err=0;
        int i,j;

        double growth_out = dicke(z_out);
        double growth_in = dicke(z_in);
        double Mmin = minimum_source_mass(z_out,astro_params,flag_options);

        double Mmax, volume, R, delta;
        double ps_ratio;

        R = user_params->BOX_LEN / user_params->HII_DIM * L_FACTOR;
        volume = pow(user_params->BOX_LEN / user_params->HII_DIM,3);
        Mmax = RtoM(R);

        //Here the condition is a mass, volume is the Lagrangian volume and delta_l is set by the
        //redshift difference which represents the difference in delta_crit across redshifts
        if(update){
            delta = Deltac * growth_out / growth_in;
            if(type<2) Mmax = condition; //for CMF returns we want the max to be the condition
        }
        //Here the condition is a cell of a given density, the volume/mass is given by the grid parameters
        //below build_halo_cats(), the delta at z_out is expected, above, the IC (z=0) delta is expected.
        //we pass in the IC delta (so we can compare populations across redshifts) and adjust here
        else{
            delta = condition;
        }

        double lnMmin = log(Mmin);
        double lnMmax = log(Mmax);
        double lnMMax_tb = log(MMAX_TABLES); //wiggle room for tables

        LOG_DEBUG("TEST FUNCTION: type = %d up %d, z = (%.2f,%.2f), Mmin = %e, Mmax = %e, R = %.2e (%.2e), delta = %.2f, M(%d)=[%.2e,%.2e,%.2e...]",type,update,z_out,z_in,Mmin,Mmax,R,volume,delta,n_mass,M[0],M[1],M[2]);

        //don't do anything for delta outside range               
        if(delta > Deltac || delta < -1){
            LOG_ERROR("delta of %f is out of bounds",delta);
            Throw(ValueError);
        }

        if(type != 5){
            init_ps();
            if(user_params_stoc->USE_INTERPOLATION_TABLES){
                initialiseSigmaMInterpTable(Mmin,MMAX_TABLES);
            }
            //we use these tables only for some functions
            if(update){
                initialise_dNdM_tables(lnMmin, lnMMax_tb, lnMmin, lnMMax_tb, growth_out, growth_in, true);
            }
            else{
                initialise_dNdM_tables(-1, Deltac, lnMmin, lnMMax_tb, growth_out, lnMmax, false);
            }
        }
        //Since the conditional MF is press-schecter, we rescale by a factor equal to the ratio of the collapsed fractions (n_order == 1) of the UMF
        if(user_params->HMF!=0){
            ps_ratio = IntegratedNdM(growth_out,lnMmin,lnMmax,lnMmax,0,1,0) / IntegratedNdM(growth_out,lnMmin,lnMmax,lnMmax,0,1,user_params->HMF);
            volume = volume / ps_ratio;
        }
        else{
            ps_ratio = 1.;
        }

        if(type==0){
            //parameters for CMF
            double prefactor = RHOcrit / sqrt(2.*PI) * cosmo_params_stoc->OMm;
            double dummy;
            struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
                .redshift = z_out,
                .growthf = growth_out,
                .delta = delta,
                .n_order = 0,
                .M_max = lnMmax,
                .sigma_max = EvaluateSigma(lnMmax,&dummy),
                .HMF = -1,
            };
            
            //using seed to select CMF or UMF since there's no RNG here
            if(seed==0){
                parameters_gsl_MF_con.HMF = user_params_stoc->HMF;
                prefactor = 1.;
                ps_ratio = 1.;
            }
            #pragma omp parallel for private(test)
            for(i=0;i<n_mass;i++){
                //conditional ps mass func * pow(M,n_order)
                if((M[i] < Mmin) || M[i] > MMAX_TABLES || (seed != 0 && M[i] > Mmax)){
                    test = 0.;
                }
                else{
                    test = MnMassfunction(log(M[i]),(void*)&parameters_gsl_MF_con);
                    
                    //convert to dndlnm
                    test = test * prefactor * ps_ratio;
                }
                LOG_ULTRA_DEBUG(" D %.1e | M1 %.1e | M2 %.1e | d %.1e | s %.1e -> %.1e",
                                growth_out,M[i],Mmax,delta,EvaluateSigma(lnMmax,&dummy),test);
                result[i] = test;
            }
        }
        else if(type==1){
            //integrate CMF -> N(<M) in one condition
            //TODO: make it possible to integrate UMFs
            double lnM_in;
            #pragma omp parallel for private(test,lnM_in)
            for(i=0;i<n_mass;i++){
                LOG_ULTRA_DEBUG(" D %.1e | Ml %.1e | Mu %.1e | Mc %.1e | d %.1e | s %.1e",
                                growth_out,Mmin,M[i],Mmax,delta,EvaluateSigma(lnMmax,&dummy));
                if(M[i] < Mmin){
                    result[i] = 0.;
                    continue;
                }
                lnM_in = log(M[i]);
                if(M[i] > Mmax){
                    lnM_in = lnMmax;
                }
                test = IntegratedNdM(growth_out,lnMmin,lnM_in,lnMmax,delta,0,-1);
                //conditional MF multiplied by a few factors
                LOG_ULTRA_DEBUG("==> %.8e",test);
                result[i] = test * Mmax / sqrt(2.*PI);
            }
        }
        else if(type==2){
            //intregrate CMF -> N_halos in many conditions
            //TODO: make it possible to integrate UMFs
            //quick hack: condition gives n_order, seed ignores tables
            #pragma omp parallel for private(test,R,volume)
            for(i=0;i<n_mass;i++){
                if(update){
                    R = MtoR(M[i]);
                    //volume = 4. / 3. * PI * R * R * R;     
                    if(user_params_stoc->USE_INTERPOLATION_TABLES && seed == 0) test = EvaluatedNdMSpline(log(M[i]),lnMmax);
                    else test = IntegratedNdM(growth_out,lnMmin,log(M[i]),log(M[i]),delta,condition,-1);
                }
                else{
                    if(user_params_stoc->USE_INTERPOLATION_TABLES && seed == 0) test = EvaluatedNdMSpline(M[i]*growth_out,lnMmax);
                    else test = IntegratedNdM(growth_out,lnMmin,lnMmax,lnMmax,M[i]*growth_out,condition,-1);
                }
                //conditional MF multiplied by a few factors
                result[i] = test * M[i] / sqrt(2.*PI);
            }
        }
        
        //Cell CMF/masses from one cell, given M as cell descendant halos
        //uses a constant mass binning since we use the input for descendants
        else if(type==3){
            double out_cmf[100];
            double out_bins[100];
            int n_bins = 100;
            double prefactor = RHOcrit / sqrt(2.*PI) * cosmo_params_stoc->OMm;
            double dummy,test;
            double lnM_bin, tot_mass=0;
            double lnMbin_max = log(10)*14; //arbitrary bin maximum
            
            struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
                .redshift = z_out,
                .growthf = growth_out,
                .delta = delta,
                .n_order = 0,
                .M_max = 0,
                .sigma_max = 0,
                .HMF = -1,
            };

            for(i=0;i<n_bins;i++){
                out_cmf[i] = 0;
                out_bins[i] = lnMmin + ((double)i/((double)n_bins-1))*(lnMbin_max - lnMmin);
            }

            #pragma omp parallel num_threads(user_params->N_THREADS) private(j)
            {
                #pragma omp for
                for(j=0;j<n_mass;j++){
                    lnMmax = log(M[j]);
                    tot_mass += M[j];
                    parameters_gsl_MF_con.M_max = lnMmax;
                    parameters_gsl_MF_con.sigma_max = EvaluateSigma(lnMmax,&dummy);
                    for(i=0;i<n_bins;i++){
                        lnM_bin = out_bins[i];
                        
                        //conditional ps mass func * pow(M,n_order)
                        if(lnM_bin < lnMmin || lnM_bin > lnMmax){
                            test = 0.;
                        }
                        else{
                            test = MnMassfunction(lnM_bin,(void*)&parameters_gsl_MF_con);
                            test = test * prefactor * ps_ratio;
                        }
                        out_cmf[i] += test * M[j] / sqrt(2.*PI);
                    }
                }
            }
            result[0] = n_bins;
            for(i=0;i<n_bins;i++){
                result[1+i] = out_bins[i];
                result[1+i+n_bins] = out_cmf[i] / pow(user_params_stoc->BOX_LEN,3); //assuming here you pass all the halos in the box
            }
        }

        //halo catalogues + cell sums from multiple conditions, given M as cell descendant halos/cells
        else if(type==4){
            int n_halo_tot=0;
            int n_cond = n_mass;
            double test_M, test_N;
            
            #pragma omp parallel num_threads(user_params->N_THREADS) private(i,j)
            {
                float out_hm[MAX_HALO_CELL];
                double exp_M,exp_N,M_prog;
                int n_halo;
                double test_M_low, test_M_mid, test_M_hi;
                //if !update, the masses are ignored, and the cell will have the given delta
                #pragma omp for
                for(j=0;j<n_cond;j++){
                    M_prog = 0;
                    get_halo_avg(growth_out,delta,lnMmin,lnMmax,M[j],update,&exp_N,&exp_M);
                    stoc_sample(growth_out, Mmin, M[j], delta, exp_N, exp_M, update, rng_stoc[omp_get_thread_num()], &n_halo, out_hm);
                    for(i=0;i<n_halo;i++){
                        if(out_hm[i]>M[j]){
                            test_M_low = gsl_spline2d_eval(Nhalo_inv_spline,log(M[j]),0.999,Nhalo_inv_cond_acc,Nhalo_inv_prob_acc);
                            test_M_mid = gsl_spline2d_eval(Nhalo_inv_spline,log(M[j]),0.9999,Nhalo_inv_cond_acc,Nhalo_inv_prob_acc);
                            test_M_hi = gsl_spline2d_eval(Nhalo_inv_spline,log(M[j]),0.99999,Nhalo_inv_cond_acc,Nhalo_inv_prob_acc);
                            LOG_WARNING("Found mass %.2e > %.2e, test interp (%.2e,%.2e,%.2e)",out_hm[i],M[j],exp(test_M_low),exp(test_M_mid),exp(test_M_hi));
                        }
                        M_prog += out_hm[i];
                        
                        #pragma omp critical
                        {
                            result[1+n_cond+(n_halo_tot++)] = out_hm[i];
                        }
                    }
                    //LOG_ULTRA_DEBUG("Cell %d %d got %.2e Mass (exp %.2e) %.2f",j,n_halo,result[1+j],exp_M,result[1+j]/(exp_M) - 1);
                    result[1+j] = M_prog - exp_M; //excess mass in the cell
                }
                
                result[0] = (double)n_halo_tot;
            }
            get_halo_avg(growth_out,delta,lnMmin,lnMmax,M[0],update,&test_N,&test_M);
            LOG_DEBUG("%d --> %d Halos, first exp N %.3e M %.3e M=[%.3e,%.3e,%.3e...]",n_cond,n_halo_tot,test_N,test_M,result[1+n_cond+0],result[1+n_cond+1],result[1+n_cond+2]);
        }
        
        //halo catalogue from list of conditions (Mass for update, delta for !update)
        else if(type==5){
            struct HaloField *halos_in;
            struct HaloField *halos_out;
            float *dens_field = calloc(HII_TOT_NUM_PIXELS,sizeof(float));

            int nhalo_out;

            if(update){
                //NOTE: using n_mass for n_conditions
                //a single coordinate is provided for each halo
                LOG_ULTRA_DEBUG("assigning input arrays w %d halos",n_mass);
                for(i=0;i<n_mass;i++){
                    // LOG_ULTRA_DEBUG("Reading %d (%d %d %d)...",i,n_mass + 3*i,n_mass + 3*i + 1,n_mass + 3*i + 2);
                    // LOG_ULTRA_DEBUG("M[%d] = %.3e",i,M[i]);
                    // LOG_ULTRA_DEBUG("coords_in[%d] = (%d,%d,%d)",i,(int)(M[n_mass + 3*i + 0]),(int)(M[n_mass + 3*i + 1]),(int)(M[n_mass + 3*i + 2]));
                    halos_in->halo_masses[i] = M[i];
                    halos_in->halo_coords[3*i+0] = (int)(M[n_mass + 3*i + 0]);
                    halos_in->halo_coords[3*i+1] = (int)(M[n_mass + 3*i + 1]);
                    halos_in->halo_coords[3*i+2] = (int)(M[n_mass + 3*i + 2]);
                }
                halos_in->n_halos = n_mass;
                //LOG_ULTRA_DEBUG("Sampling...",n_mass);
                halo_update(rng_stoc, z_in, z_out, halos_in, halos_out);
            }
            else{
                //NOTE: halomass_in is linear delta at z = redshift_out
                for(i=0;i<n_mass;i++){
                    dens_field[i] = M[i] / growth_out;
                }
                //no large halos
                halos_in->n_halos=0;
                build_halo_cats(rng_stoc, z_out, dens_field, halos_in, halos_out);
            }
            
            nhalo_out = halos_out->n_halos;
            if(nhalo_out > 3){
                LOG_DEBUG("sampling done, %d halos, %.2e %.2e %.2e",nhalo_out,halos_out->halo_masses[0],
                            halos_out->halo_masses[1],halos_out->halo_masses[2]);
            }

            result[0] = nhalo_out;
            for(i=0;i<nhalo_out;i++){
                result[1+i] = halos_out->halo_masses[i];
                result[nhalo_out+1+3*i] = halos_in->halo_coords[3*i];
                result[nhalo_out+2+3*i] = halos_in->halo_coords[3*i+1];
                result[nhalo_out+3+3*i] = halos_in->halo_coords[3*i+2];
            }
            free(dens_field);
        }

        //return inverse dNdM table result for M at a bunch of masses
        else if(type==6){
            double y_in,x_in,mass;
            #pragma omp parallel for private(test,x_in,y_in)
            for(i=0;i<n_mass;i++){
                y_in = log(M[i]);
                if(y_in < lnMmin){
                    result[i] = 0;
                    continue;
                }

                x_in = update ? log(condition) : condition;
                mass = update ? condition : Mmax;
                test = EvaluatedNdMSpline(x_in,y_in);
                result[i] = test * mass / sqrt(2.*PI);
                LOG_ULTRA_DEBUG("dNdM table: x = %.6e, y = %.6e z = %.6e",x_in,y_in,test);
            }
        }

        //return dNdM table result for M at a bunch of probabilities
        else if(type==8){
            double y_in,x_in;
            #pragma omp parallel for private(test,x_in,y_in)
            for(i=0;i<n_mass;i++){
                if(M[i] < 0 || M[i] > 1){
                    LOG_ERROR("invalid probability %.2f",M[i]);
                    Throw(ValueError);
                }
                y_in = M[i];
                x_in = update ? log(condition) : condition;
                test = gsl_spline2d_eval(Nhalo_inv_spline,x_in,y_in,Nhalo_inv_cond_acc,Nhalo_inv_prob_acc);
                result[i] = exp(test);
                LOG_ULTRA_DEBUG("dNdM inverse table: %.6e x = %.6e, y = %.6e z = %.6e",condition,x_in,y_in,test);
            }
        }
        
        else{
            LOG_ERROR("Unkown output type");
            Throw(ValueError);
        }
        if(type!=5){
            if(user_params_stoc->USE_INTERPOLATION_TABLES){
                freeSigmaMInterpTable();
            }
            free_dNdM_tables();
        }
        free_rng_threads(rng_stoc);
    } //end of try

    Catch(status){
        return(status);
    }
    LOG_DEBUG("Done.");
    return(0);
}