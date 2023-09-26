/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//BIG TODO: sort out single/double precision all the way through
//TODO: the USE_INTERPOLATION_TABLES flag may need to be forced, it's in a strange position here
//  where it only affects the sigma table.
//NOTE: for the future discrete tables this does make sense
//TODO: Don't have every error be a ValueError

//max number of attempts for mass tolerance before failure
#define MAX_ITERATIONS 1e6
#define MAX_ITER_N 1000 //for stoc_halo_sample (select N halos) how many tries for one N, this should be large to enforce a near-possion p(N)
#define MMAX_TABLES 1e14

//buffer size (per cell of arbitrary size) in the sampling function
#define MAX_HALO_CELL (int)1e5
#define MAX_DELTAC_FRAC (float)0.995 //max delta/deltac for interpolation tables / integrals

//TODO:These should be static, right?
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
    int CMF;
};

//parameters for the halo mass->stars calculations
//Note: ideally I would split this into constants set per snapshot and
//  constants set per condition, however some variables (delta or Mass)
//  can be set with differing frequencies depending on the condition type
static struct HaloSamplingConstants{
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
    double M_min_save;
    double lnM_min_save;
    double M_max_tables;
    double lnM_max_tb;
    double sigma_min;
    double sigma_min_save;

    //per-condition/redshift depending on update or not
    double delta;
    double M_cond;
    double lnM_cond;
    double sigma_cond;

    //calculated per condition
    double mu_desc_star;
    double cond_val; //need to specify for the tables
    //can't do SFR since it depends on the sampled stellar
    //although there's no pow calls so it should be faster
    double expected_N;
    double expected_M;
    double expected_N_save; //expected save values for debug, remove later
    double expected_M_save;

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
    LOG_DEBUG("delta %.2f M_c %.2e (%.2e) (%.2e) cond %.2e",c->delta,c->M_cond,c->lnM_cond,c->sigma_cond,c->cond_val);
    LOG_DEBUG("mu star %.2e exp N %.2f exp M %.2e",c->mu_desc_star,c->expected_N,c->expected_M);
    return;
}

//set the minimum source mass
//TODO: include MINI_HALOS
double minimum_source_mass(double redshift, struct AstroParams *astro_params, struct FlagOptions * flag_options){
    double Mmin;
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        Mmin = astro_params->M_TURN / global_params.HALO_MTURN_FACTOR;
    }
    else {
        if(flag_options->M_MIN_in_Mass) {
            Mmin = astro_params->M_TURN / global_params.HALO_MTURN_FACTOR;
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

//Modularisation that should be put in ps.c for the evaluation of sigma
double EvaluateSigma(double lnM, int calc_ds, double *dsigmadm){
    //using log units to make the fast option faster and the slow option slower
    double sigma;
    float MassBinLow;
    int MassBin;
    double dsigma_val;

    //all this stuff is defined in ps.c and initialised with InitialiseSigmaInterpTable
    //NOTE: The interpolation tables are `float` in ps.c
    if(user_params_ps->USE_INTERPOLATION_TABLES) {
        MassBin = (int)floor( (lnM - MinMass )*inv_mass_bin_width );
        MassBinLow = MinMass + mass_bin_width*(double)MassBin;

        sigma = Sigma_InterpTable[MassBin] + ( lnM - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
        
        if(calc_ds){
            dsigma_val = dSigmadm_InterpTable[MassBin] + ( lnM - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
            *dsigmadm = -pow(10.,dsigma_val); //this may be slow but it's only used in the construction of the dNdM interp tables
        }
    }
    else {
        sigma = sigma_z0(exp(lnM));
        if(calc_ds) *dsigmadm = dsigmasqdm_z0(exp(lnM));
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

    sigsmallR = EvaluateSigma(lnM,0,&dummy);

    //LOG_ULTRA_DEBUG("FgtrM: SigmaM %.3e",sigsmallR);
    //sometimes condition mass is close enough to minimum mass such that the sigmas are the same to float precision
    //In this case we just throw away the halo, since it is very unlikely to have progenitors
    if(sigsmallR <= sig_bias){
        return 0.;
    }
    
    del = (Deltac - del_bias)/growthf;

    //deal with floating point errors
    //TODO: this is a little hacky, check the split growth factors before calling this instead
    if(del < -FRACT_FLOAT_ERR*100){
            LOG_ERROR("error in FgtrM: condition sigma %.3e delta %.3e sigma %.3e delta %.3e (%.3e)",sig_bias,del_bias,sigsmallR,Deltac,del);
            Throw(ValueError);
    }
    if(del < FRACT_FLOAT_ERR*100){
        return 1.;
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

    sigma1 = EvaluateSigma(M1,1,&dsigmadm); //WARNING: THE SIGMA TABLE IS STILL SINGLE PRECISION

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

//Sheth Tormen 2002 fit for the CMF, while the moving barrier does not allow for a simple rescaling, it has been found
//That a taylor expansion of the barrier shape around the point of interest well approximates the simulations
//TODO: Count the growth factors needed in each term, also move to ps.c
double st_taylor_factor(double sig, double sig_cond, double growthf){
    double a = SHETH_A;
    double alpha = global_params.SHETH_c;
    double beta = global_params.SHETH_b;
    double delsq = Deltac*Deltac/growthf/growthf;
    double sigsq = sig*sig;
    double sigcsq = sig_cond*sig_cond;
    double sigdiff = sigsq - sigcsq;

    double result = 0;

    int i;
    //Taylor expansion of the x^a part around (sigsq - sigcondsq)
    for(i=5;i>-1;i--){
        result = (result + (pow(sigsq/(SHETH_a*delsq), alpha - i) - pow(sigcsq/(SHETH_a*delsq), alpha - i)))*(alpha-i+1)*sigdiff/i; //the last factor makes the factorials and nth power of nth derivative
        LOG_ULTRA_DEBUG("%d term %.2e",i,result);
    }

    //constants and the + 1 factor from the barrier 0th derivative B = A*(1 + b*x^a)
    result = (result/sigdiff*beta + 1)*sqrt(SHETH_a)*Deltac;
    return result;
}

//TODO: move to ps.c
double dNdM_conditional_ST(double growthf, double M1, double M2, double delta1, double delta2, double sigma2){
    double sigma1, sig1sq, sig2sq, dsigmadm, B1, B2;
    double MassBinLow;
    int MassBin;

    sigma1 = EvaluateSigma(M1,1,&dsigmadm); //WARNING: THE SIGMA TABLE IS STILL SINGLE PRECISION

    LOG_ULTRA_DEBUG("st fit: D: %.2f M1: %.2e M2: %.2e d1: %.2f d2: %.2f s2: %.2f",growthf,M2,M2,delta1,delta2,sigma2);

    M1 = exp(M1);
    M2 = exp(M2);

    sig1sq = sigma1*sigma1;
    sig2sq = sigma2*sigma2;
    B1 = sheth_delc(delta1/growthf,sigma1);
    B2 = sheth_delc(delta2/growthf,sigma2);
    LOG_ULTRA_DEBUG("Barriers 1: %.2e | 2: %.2e",B1,B2);
    LOG_ULTRA_DEBUG("taylor expansion factor %.6e",st_taylor_factor(sigma1,sigma2,growthf));

    if((sigma1 > sigma2)) {
        return -dsigmadm*sigma1*st_taylor_factor(sigma1,sigma2,growthf)/pow(sig1sq-sig2sq,1.5)*exp(-(B1 - B2)*(B1 - B2)/(sig1sq-sig2sq));
    }
    else if(sigma1==sigma2) {
        return -dsigmadm*sigma1*st_taylor_factor(sigma1,sigma2,growthf)/pow(1e-6,1.5)*exp(-(B1 - B2)*(B1 - B2)/(1e-6));
    }
    else {
        return 0.;
    }
}

//TODO: it may be better to place the if-elses earlier, OR pass in a function pointer
//  Although, I doubt the if-elses really have a big impact compared to the integrals
double MnMassfunction(double M, void *param_struct){
    struct parameters_gsl_MF_con_int_ params = *(struct parameters_gsl_MF_con_int_ *)param_struct;
    double mf, m_factor;
    int i;
    double growthf = params.growthf;
    double delta = params.delta;
    double n_order = params.n_order;
    double sigma2 = params.sigma_max; //M2 and sigma2 are degenerate, remove one
    double M_filter = params.M_max;
    double z = params.redshift;
    //HMF = user_params.HMF unless we want the conditional in which case its -1
    int HMF = params.HMF;
    int CMF = params.CMF;

    double M_exp = exp(M);

    //M1 is the mass of interest, M2 doesn't seem to be used (input as max mass),
    // delta1 is critical, delta2 is current, sigma is sigma(Mmax,z=0)
    //WE WANT DNDLOGM HERE, SO WE ADJUST ACCORDINGLY

    //dNdlnM = dfcoll/dM * M / M * constants
    //All unconditional functions are dNdM, conditional is actually dfcoll dM == dNdlogm * constants
    if(!CMF){
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
        else {
            //TODO: proper errors
            return -1;
        }
    }
    else{
        if(HMF==0) {
            mf = dNdM_conditional_double(growthf,M,M_filter,Deltac,delta,sigma2);
        }
        // else if(HMF==1) {
        //     mf = dNdM_conditional_ST(growthf,M,M_filter,Deltac,delta,sigma2);
        // }
        else {
            //NOTE: Normalisation scaling is currently applied outside the integral, per condition
            //This will be the rescaled EPS CMF,
            //TODO: put rescaling options here (normalised EPS, rescaled EPS, local/global scalings of UMFs from Tramonte+17)
            mf = dNdM_conditional_double(growthf,M,M_filter,Deltac,delta,sigma2);
        }
    }
    //norder for expectation values of M^n
    m_factor = pow(M_exp,n_order);
    return m_factor * mf;
}

//copied mostly from the Nion functions
//I might be missing something like this that already exists somewhere in the code
//TODO: rename since its now an integral of M * dfcoll/dm = dNdM / [(RHOcrit * (1+delta) / sqrt(2.*PI) * cosmo_params_stoc->OMm)]
double IntegratedNdM(double growthf, double M1, double M2, double M_filter, double delta, double n_order, int HMF, int CMF){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = FRACT_FLOAT_ERR*64; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);
    
    double dummy;
    double sigma = EvaluateSigma(M_filter,0,&dummy);

    struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
        .growthf = growthf,
        .delta = delta,
        .n_order = n_order,
        .sigma_max = sigma,
        .M_max = M_filter,
        .HMF = HMF,
        .CMF = CMF,
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

gsl_spline *M_exp_spline;
gsl_interp_accel *M_exp_acc;
#pragma omp threadprivate(M_exp_acc)

//TODO:Move all these tables to the const struct
gsl_spline *sigma_inv_spline;
gsl_interp_accel *sigma_inv_acc;
#pragma omp threadprivate(sigma_inv_acc)

double EvaluatedNdMSpline(double x_in, double y_in){
    return gsl_spline2d_eval(Nhalo_spline,x_in,y_in,Nhalo_cond_acc,Nhalo_min_acc);
}
double EvaluateMexpSpline(double x_in){
    return gsl_spline_eval(M_exp_spline,x_in,M_exp_acc);
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
        nx = global_params.N_MASS_INTERP;
        delta = Deltac*growth1/param;
        if(delta < -1 || delta > Deltac){
            LOG_ERROR("Invalid delta %.3f",delta);
            Throw(ValueError);
        }
    }
    //generation from grid, x=delta, param_2 = log(M_filt)
    else{
        nx = global_params.N_DELTA_INTERP;
        lnM_cond = param;
        if(xmin < -1 || xmax > Deltac){
            LOG_ERROR("Invalid delta [%.3f,%.3f]",xmin,xmax);
            Throw(ValueError);
        }
    }
    //Check for invalid delta

    //y(min mass) is always the same
    ny = global_params.N_MASS_INTERP;
    np = global_params.N_PROB_INTERP;

    double xa[nx], ya[ny], za[nx*ny];
    double pa[np], za_inv[nx*np], ma[nx];
    double pbuf;
    double one_halo;

    int i,j,k;
    //set up coordinate grids
    for(i=0;i<nx;i++) xa[i] = xmin + (xmax - xmin)*((double)i)/((double)nx-1);
    for(j=0;j<ny;j++) ya[j] = ymin + (ymax - ymin)*((double)j)/((double)ny-1);
    //for(k=0;k<np;k++) pa[k] = exp(pmin + (pmax - pmin)*((double)k)/((double)np-1));

    //log(1-p) distribution to get the rare halos right
    //NOTE: although the interpolation points are linear in log(1-p),
    //the table itself is linear in p
    for(k=0;k<np-1;k++){
        pbuf = global_params.MIN_LOGPROB*(double)k/((double)np-2); //max(log(1-p)) == MIN_LOGPROB
        pbuf = exp(pbuf); // 1-p
        pa[k] = 1 - pbuf;
        //LOG_ULTRA_DEBUG("p = %.8e (%.8e)",pa[k],1 - pa[k]);
    }
    pa[np-1] = 1.;
    
    Nhalo_spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, nx, ny);
    Nhalo_inv_spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, nx, np);
    M_exp_spline = gsl_spline_alloc(gsl_interp_linear,nx);

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
            //TODO: if this line affects performance, set it to zero, since it won't affect much
            one_halo = 1. / exp(lnM_cond) * sqrt(2*PI);

            //setting to zero for high delta 
            //this one needs to be done before the norm is calculated
            if(delta > MAX_DELTAC_FRAC*Deltac){
                //In the last bin, n_halo / mass * sqrt2pi interpolates toward one halo
                for(j=0;j<ny;j++)
                    gsl_interp2d_set(Nhalo_spline,za,i,j,one_halo);
                
                //Similarly, the inverse CDF tends to a step function at lnM_cond
                for(k=0;k<np;k++)
                    gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,k,lnM_cond);

                continue;
            }

            norm = IntegratedNdM(growth1,ymin,ymax,lnM_cond,delta,0,user_params_stoc->HMF,1);
            ma[i] = IntegratedNdM(growth1,ymin,ymax,lnM_cond,delta,1,user_params_stoc->HMF,1);
            // LOG_ULTRA_DEBUG("cond x: %.2e (%d) ==> %.8e / %.8e",x,i,norm,ma[i]);
            
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
                    buf = IntegratedNdM(growth1, ymin, y, lnM_cond, delta, 0, user_params_stoc->HMF, 1); //Number density between ymin and y
                }
                //LOG_ULTRA_DEBUG("Int || x: %.2e (%d) y: %.2e (%d) ==> %.8e / %.8e",x,i,exp(y),j,buf,buf/norm);
                gsl_interp2d_set(Nhalo_spline,za,i,j,buf);
                
                prob = buf / norm; //get log-probability
                //catch some norm errors
                if(prob != prob){
                    LOG_ERROR("Normalisation error in table generation");
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
            
            
            //We set the rarest halo to be at MIN_LOGPROB since
            //interpolating in underdense cells up to the entire cell mass
            //vastly overestimates high mass halos.
            //NOTE: at this point, lnM_p is the last Mass interpolated
            //which will be the highest upper integral limit (log(p) = 1 - MIN_LOGPROB)
            gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,np-1,lnM_p);
            
            //option for setting exact max mass
            // if(update)
            //     gsl_interp2d_set(Nhalo_inv_spline,za_inv,i,np-1,lnM_max);

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

        M_exp_acc = gsl_interp_accel_alloc();
    }
    gsl_status = gsl_spline2d_init(Nhalo_spline, xa, ya, za, nx, ny);
    GSL_ERROR(gsl_status);
    
    gsl_status = gsl_spline2d_init(Nhalo_inv_spline, xa, pa, za_inv, nx, np);
    GSL_ERROR(gsl_status);
    
    gsl_status = gsl_spline_init(M_exp_spline, xa, ma, nx);
    GSL_ERROR(gsl_status);

    LOG_DEBUG("Done.");
}

void free_dNdM_tables(){
    gsl_spline2d_free(Nhalo_spline);
    gsl_spline2d_free(Nhalo_inv_spline);
    gsl_spline_free(M_exp_spline);

    #pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        gsl_interp_accel_free(Nhalo_cond_acc);
        gsl_interp_accel_free(Nhalo_min_acc);
        
        gsl_interp_accel_free(Nhalo_inv_cond_acc);
        gsl_interp_accel_free(Nhalo_inv_prob_acc);

        gsl_interp_accel_free(M_exp_acc);
    }
}

void initialise_siginv_spline(){
    //use the sigma table to make an inverse
    //gsl spline expects strictly increasing X array, since this is the opposite we need to reverse it
    double xa[NMass], ya[NMass];
    int i;
    sigma_inv_spline = gsl_spline_alloc(gsl_interp_linear,NMass);
    for(i=0;i<NMass;i++){
        xa[i] = Sigma_InterpTable[NMass-i-1];
        ya[i] = Mass_InterpTable[NMass-i-1];
        // LOG_ULTRA_DEBUG("%d: Sigma %.2e Mass %.2e",i,xa[i],ya[i]);
    }
    sigma_inv_acc = gsl_interp_accel_alloc();
    gsl_spline_init(sigma_inv_spline,xa,ya,NMass);
    // LOG_ULTRA_DEBUG("Done.");
}

double EvaluateSiginvSpline(double sigma){
    return gsl_spline_eval(sigma_inv_spline,sigma,sigma_inv_acc);
}

void free_siginv_spline(){
    gsl_spline_free(sigma_inv_spline);
    #pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        gsl_interp_accel_free(sigma_inv_acc);
    }
}

//TODO: Speedtest the RGI interpolation present in Spintemp etc...
//  Save the X/Y/Z from the table builder and apply the Minihalo 2D interpolation
double sample_dndM_inverse(double condition, gsl_rng * rng){
    double p_in;
    p_in = gsl_rng_uniform(rng);
    double res = gsl_spline2d_eval(Nhalo_inv_spline,condition,p_in,Nhalo_inv_cond_acc,Nhalo_inv_prob_acc);
    return res;
}

//Set the constants that are calculated once per snapshot
void stoc_set_consts_z(struct HaloSamplingConstants *const_struct, double redshift, double redshift_prev){
    LOG_DEBUG("Setting z constants z=%.2f z_prev=%.2f",redshift,redshift_prev);
    const_struct->t_h = t_hubble(redshift);
    const_struct->growth_out = dicke(redshift);
    const_struct->z_out = redshift;
    const_struct->z_in = redshift_prev;

    double dummy;
    double M_min = minimum_source_mass(redshift,astro_params_stoc,flag_options_stoc);
    const_struct->M_min = M_min;
    const_struct->M_min_save = M_min * global_params.HALO_SAMPLE_FACTOR;
    const_struct->lnM_min = log(M_min);
    const_struct->lnM_min_save = log(const_struct->M_min_save);
    const_struct->M_max_tables = global_params.M_MAX_INTEGRAL;
    const_struct->lnM_max_tb = log(const_struct->M_max_tables);


    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(const_struct->M_min / 2,const_struct->M_max_tables);
        initialise_siginv_spline();
    }
    const_struct->sigma_min = EvaluateSigma(const_struct->lnM_min,0,&dummy);
    const_struct->sigma_min_save = EvaluateSigma(const_struct->lnM_min,0,&dummy);

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

        const_struct->delta = Deltac*const_struct->growth_out/const_struct->growth_in;
        const_struct->update = 1;
        //TODO: change the table functions to accept the structure
        initialise_dNdM_tables(const_struct->lnM_min, const_struct->lnM_max_tb,const_struct->lnM_min, const_struct->lnM_max_tb,
                                const_struct->growth_out, const_struct->growth_in, true);
    }
    else {
        double M_cond = RHOcrit * cosmo_params_stoc->OMm * VOLUME / HII_TOT_NUM_PIXELS;
        const_struct->M_cond = M_cond;
        const_struct->lnM_cond = log(M_cond);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond,0,&dummy);
        const_struct->update = 0;
        initialise_dNdM_tables(-1, Deltac, const_struct->lnM_min, const_struct->lnM_max_tb, const_struct->growth_out, const_struct->lnM_cond, false);
    }
    LOG_DEBUG("Done.");
    return;
}

//set the constants which are calculated once per condition
void stoc_set_consts_cond(struct HaloSamplingConstants *const_struct, double cond_val){
    double tbl_arg,sig,m_exp,n_exp,n_exp_save,dummy,del,frac_save;

    //Here the condition is a mass, volume is the Lagrangian volume and delta_l is set by the
    //redshift difference which represents the difference in delta_crit across redshifts
    if(const_struct->update){
        const_struct->M_cond = cond_val;
        const_struct->lnM_cond = log(cond_val);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond,0,&dummy);
        //mean stellar mass of this halo mass, used for stellar z correlations
        const_struct->mu_desc_star = fmin(astro_params_stoc->F_STAR10
                                        * pow(cond_val/1e10,astro_params_stoc->ALPHA_STAR)
                                        * exp(-astro_params_stoc->M_TURN/cond_val),1) * cond_val;
        const_struct->cond_val = const_struct->lnM_cond;
    }
    //Here the condition is a cell of a given density, the volume/mass is given by the grid parameters
    else{
        const_struct->delta = cond_val;
        const_struct->cond_val = cond_val;
    }

    //the splines don't work well for cells above Deltac, but there CAN be cells above deltac, since this calculation happens
    //before the overlap, and since the smallest dexm mass is M_cell*(1.01^3) there *could* be a cell above Deltac not in a halo
    if(!const_struct->update && cond_val > Deltac){
        //these values won't actually do anything they just prevent the spline calls
        const_struct->expected_M = const_struct->M_cond;
        const_struct->expected_N = 1;
        return;
    }

    //TODO: reorganize the below code with get_halo_avg, EvaluateFgtrM and EvaluatedNdMSpline
    //Get expected N from interptable
    n_exp = EvaluatedNdMSpline(const_struct->cond_val,const_struct->lnM_max_tb); //should be the same as < lnM_cond, but that can hide some interp errors
    //TODO: remove if performance is affected by this line
    n_exp_save = EvaluatedNdMSpline(const_struct->cond_val,const_struct->lnM_max_tb) - EvaluatedNdMSpline(const_struct->cond_val,log(const_struct->M_min*global_params.HALO_SAMPLE_FACTOR));
    
    //NOTE: while the most common mass functions have simpler expressions for f(<M) (erfc based) this will be general, and shouldn't impact compute time much
    m_exp = EvaluateMexpSpline(const_struct->cond_val);
    const_struct->expected_N = n_exp * const_struct->M_cond / sqrt(2.*PI);
    const_struct->expected_N_save = n_exp_save * const_struct->M_cond / sqrt(2.*PI);
    const_struct->expected_M = m_exp * const_struct->M_cond / sqrt(2.*PI);
    // const_struct->expected_M_save = frac_save * const_struct->M_cond;

    return;
}

//return the expected number and mass of halos for a given condition
//REPLACED BY stoc_set_consts_cond()
void get_halo_avg(double growth_out, double delta, double lnMmin, double lnMmax, float M_in, bool update, double * exp_N, double * exp_M){
    double tbl_arg,frac,sigma_max,n_exp,dummy;
    if(!update){
        tbl_arg = delta;
    }
    else{
        tbl_arg = log(M_in);
    }

    double M,N;
    
    sigma_max = EvaluateSigma(log(M_in),0,&dummy);
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
    z_hi = (int)((z + randbuf) / (double)(HII_D_PARA) * (double)(D_PARA));
    crd_hi[0] = x_hi;
    crd_hi[1] = y_hi;
    crd_hi[2] = z_hi;
}

//This function adds stochastic halo properties to an existing halo
//TODO: this needs to be updated to handle MINI_HALOS and not USE_MASS_DEPENDENT_ZETA flags
int add_halo_properties(gsl_rng *rng, float halo_mass, float M_turn_var, struct HaloSamplingConstants * hs_constants, float * output){
    //for now, we just have stellar mass
    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;
    double sigma_sfr = astro_params_stoc->SIGMA_SFR;

    double fstar_mean, f_sample, sm_sample;
    double sfr_mean, sfr_sample;
    double dutycycle_term;

    //This clipping is normally done with the mass_limit_bisection root find.
    //I can't do that here with the stochasticity, since the f_star clipping happens AFTER the sampling
    fstar_mean = f10 * pow(halo_mass/1e10,fa);

    //in order to remain consistent with the minihalo treatment in default (Nion_a * exp(-M/M_a) + Nion_m * exp(-M/M_m - M_a/M))
    //we need to separate halos into molecular and atomically cooled
    //for now I assign this randomly at each redshift which may cause some inconsistencies
    //TODO: discuss a better option
    dutycycle_term = exp(-M_turn_var/halo_mass);

    //In order to include the feedback: I WILL need to go forward in time. This means a few things:
    //I need to store progenitor properties (via index I guess) somewhere to do the updates
    //I can then implement the MAR-based star formation as well if needed
    //The caching becomes a huge issue though

    if(sigma_star > 0){
        //sample stellar masses from each halo mass assuming lognormal scatter
        f_sample = gsl_ran_ugaussian(rng);
        
        /* Simply adding lognormal scatter to a delta increases the mean (2* is as likely as 0.5*)
        * We multiply by exp(-sigma^2/2) so that X = exp(mu + N(0,1)*sigma) has the desired mean */
        f_sample = fmin(fstar_mean * exp(-sigma_star*sigma_star/2 + f_sample*sigma_star) * dutycycle_term,1);

        sm_sample = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * f_sample; //f_star is galactic GAS/star fraction, so OMb is needed
    }
    else{
        //duty cycle, TODO: think about a way to explicitly include the binary nature consistently with the updates
        //At the moment, we simply reduce the mean
        sm_sample = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * fmin(fstar_mean*dutycycle_term,1);
    }

    sfr_mean = sm_sample / (astro_params_stoc->t_STAR * hs_constants->t_h);
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
int update_halo_properties(gsl_rng * rng, float halo_mass, struct HaloSamplingConstants *hs_constants, float *props_in, float *output){
    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;
    double sigma_sfr = astro_params_stoc->SIGMA_SFR;
    double interp_star, interp_sfr;

    //sample new properties (uncorrelated)
    add_halo_properties(rng, astro_params_stoc->M_TURN, halo_mass, hs_constants, output);

    //get dz correlations
    interp_star = hs_constants->corr_star;
    interp_sfr = hs_constants->corr_sfr;
    float x1,x2,mu1,mu2;

    //STELLAR MASS: get median from mean + lognormal scatter (we leave off a bunch of constants and use the mean because we only need the ratio)
    mu1 = hs_constants->mu_desc_star;
    mu2 = fmin(f10 * pow(halo_mass/1e10,fa)*exp(-astro_params_stoc->M_TURN/halo_mass),1) * halo_mass; //TODO: speed this line up, exp + pow on EVERY progenitor is slow
    //The same CDF value will be given by the ratio of the means/medians, since the scatter is z and M-independent
    x1 = props_in[0];
    x2 = mu2/mu1*x1;
    //interpolate between uncorrelated and matched properties.
    output[0] = (1-interp_star)*output[0] + interp_star*x2;

    //repeat for all other properties
    //SFR: get median (TODO: if I add z-M dependent scatters I will need to re-add the constants)
    mu1 = props_in[0] / hs_constants->t_h_prev;
    mu2 = output[0] / hs_constants->t_h;
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

    //QUICK HACK: setup a fast t_h for the catalogue since this is called from FindHaloes.c
    struct HaloSamplingConstants hs_constants;
    hs_constants.z_out = redshift;
    hs_constants.t_h = t_hubble(redshift);

    //loop through the halos and assign properties
    int i;
    float buf[2];
#pragma omp parallel for private(buf)
    for(i=0;i<nhalos;i++){
        LOG_ULTRA_DEBUG("halo %d hm %.2e crd %d %d %d",i,halos->halo_masses[i],halos->halo_coords[3*i+0],halos->halo_coords[3*i+1],halos->halo_coords[3*i+2]);
        add_halo_properties(rng_stoc[omp_get_thread_num()],astro_params_stoc->M_TURN, halos->halo_masses[i], &hs_constants, buf);
        LOG_ULTRA_DEBUG("stars %.2e sfr %.2e",buf[0],buf[1]);
        halos->stellar_masses[i] = buf[0];
        halos->halo_sfr[i] = buf[1];
    }

    free_rng_threads(rng_stoc);
    return 0;
    LOG_DEBUG("Done.");
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
        nh = 0;
        //find an N which *could* fit the mass tolerance
        while((nh*M_cond < exp_M*(1-mass_tol))||(M_min*nh > exp_M*(1+mass_tol))){
            nh = gsl_ran_poisson(rng,exp_N);
        }
        for(n_attempts=0;n_attempts<MAX_ITER_N;n_attempts++){
            M_prog = 0;
            halo_count = 0;
            for(ii=0;ii<nh;ii++){
                hm_sample = sample_dndM_inverse(tbl_arg,rng);
                hm_sample = exp(hm_sample);
                M_prog += hm_sample;
                M_out[halo_count++] = hm_sample;
            }
            //LOG_ULTRA_DEBUG("attempt %d M=%.3e [%.3e, %.3e]",n_attempts,M_prog,exp_M*(1-MASS_TOL),exp_M*(1+MASS_TOL));
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
            (*n_halo_pt)--; //increment has preference over dereference
        }
    }
    else{
        while(*M_tot_pt > exp_M){
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
    if(hs_constants->update)exp_M *= 0.94; //0.95 fudge factor for assuming that internal lagrangian volumes are independent

    int n_halo_sampled, n_failures=0;
    double M_prog=0;
    double M_sample;
    int n_removed;

    double tbl_arg = hs_constants->cond_val;

    for(n_failures=0;n_failures<MAX_ITERATIONS;n_failures++){
        n_halo_sampled = 0;
        M_prog = 0;
        while(M_prog < exp_M){
            M_sample = sample_dndM_inverse(tbl_arg,rng);
            M_sample = exp(M_sample);

            // the extra delta function at N_prog == 1
            // assuming that exactly (1-exp_M) is below the mass limit
            //      IMPLIES that we cannot sample above exp_M
            // if(M_sample > exp_M){
            //     M_sample = exp_M;
            // }

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
        if((M_prog < exp_M*(1+mass_tol)) && (M_prog > exp_M*(1-mass_tol))){
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
//I'm not sure how to properly handle below-resolution mass
//  TODO: Should I subtract the expected mass at the beginning or is that given by the final remainder?
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
    double *dummy;

    double tbl_arg = hs_constants->cond_val;
    n_halo_sampled = 0;
    // LOG_ULTRA_DEBUG("Start: M %.2e (%.2e) d %.2e",M_cond,exp_M,d_cond);
    //set initial amount (subtract unresolved)
    //TODO: check if I should even do this
    M_remaining = exp_M;
    lnM_remaining = log(M_remaining);

    double x_min;

    while(M_remaining > M_min*global_params.HALO_SAMPLE_FACTOR){
        delta_current = (Deltac - d_cond)/(M_remaining/M_cond);
        sigma_r = EvaluateSigma(lnM_remaining,0,&dummy);
        del_term = delta_current*delta_current/growthf/growthf;

        //Low x --> high sigma --> low mass, high x --> low sigma --> high mass
        //|x| required to sample the smallest halo on the tables
        x_min = sqrt(del_term/(sigma_min*sigma_min - sigma_r*sigma_r));

        //LOG_ULTRA_DEBUG("M_rem %.2e d %.2e sigma %.2e min %.2e xmin %.2f",M_remaining,delta_current,sigma_r,sigma_min,x_min);

        //we use the gaussian tail distribution to enforce our Mmin limit from the sigma tables
        x_sample = gsl_ran_ugaussian_tail(rng,x_min);
        sigma_sample = sqrt(del_term/(x_sample*x_sample) + sigma_r*sigma_r);
        M_sample = EvaluateSiginvSpline(sigma_sample);
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
    double dummy;
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
    
    sigma_res = EvaluateSigma(lnm_res, 0, &dummy);
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
        sigma_start = EvaluateSigma(lnm_start,0,&dummy);
        sigmasq_start = sigma_start*sigma_start;
        sigma_half = EvaluateSigma(lnm_half,1,&alpha_half);
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
            F = 1 - EvaluateFgtrM(growth_d,lnm_res,d_start*growth_d,sigma_start);
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
            F = 1 - EvaluateFgtrM(growth_d,lnm_res,d_start*growth_d,sigma_start);
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
    // LOG_ULTRA_DEBUG("Condition M = %.2e (%.2e), Mmin = %.2e, delta = %.2f upd %d"
    //     ,hs_constants->M_cond,hs_constants->expected_M,hs_constants->M_min,hs_constants->delta,hs_constants->update);
    //TODO: really examine the case for number/mass sampling
    //The poisson sample fails spectacularly for high delta (updates or dense cells)
    //  and excludes the correlation between number and mass (e.g many small halos or few large ones)
    //The mass sample underperforms at low exp_M/M_max by excluding stochasticity in the total collapsed fraction
    //  and excluding larger halos (e.g if exp_M is 0.1*M_max we can effectively never sample the large halos)
    //i.e there is some case for a delta cut between these two methods however I have no intuition for the exact levels
    //TODO: try something like if(exp_N > 10 && exp_M < 0.9*M_cond) stoc_halo_sample();

    int err;
    //If the expected mass is below our minimum saved mass, don't bother calculating
    if(hs_constants->delta <= -1 || hs_constants->expected_M < hs_constants->M_min*global_params.HALO_SAMPLE_FACTOR){
        *n_halo_out = 0;
        return 0;
    }
    //if delta is above critical, form one big halo
    if(hs_constants->delta > MAX_DELTAC_FRAC*Deltac){
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
    
    //exptected halos IN ENTIRE BOX
    double expected_nhalo = VOLUME * IntegratedNdM(growthf,log(Mmin*global_params.HALO_SAMPLE_FACTOR),lnMmax_tb,lnMmax_tb,0,0,user_params_stoc->HMF,0);
    unsigned long long int localarray_size = global_params.MAXHALO_FACTOR * (unsigned long long int)expected_nhalo / user_params_stoc->N_THREADS; //integer division should be fine here
    localarray_size = localarray_size < MAX_HALO_CELL ? MAX_HALO_CELL : localarray_size; //set a minimum number so the end doesn't segfault

    LOG_DEBUG("Beginning stochastic halo sampling on %d ^3 grid",lo_dim);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e,volume = %.3e, D = %.3e",redshift,Mmin,Mcell,Mcell/RHOcrit/cosmo_params_stoc->OMm,growthf);
    LOG_DEBUG("Expected N_halo: %.3e, array size per thread %llu (~%.3e GB total)",expected_nhalo,localarray_size,6.*localarray_size*sizeof(int)*user_params_stoc->N_THREADS/1e9);

    //Since the conditional MF is extended press-schecter, we rescale by a factor equal to the ratio of the collapsed fractions (n_order == 1) of the UMF
    double ps_ratio = 1.;
    if(user_params_stoc->HMF!=0){
        ps_ratio = (IntegratedNdM(growthf,lnMmin,lnMcell,lnMcell,0,1,0,0) 
            / IntegratedNdM(growthf,lnMmin,lnMcell,lnMcell,0,1,user_params_stoc->HMF,0));
    }

    //shared halo count (start at the number of big halos)
    unsigned long long int count_total = 0;
    unsigned long long int istart_local[user_params_stoc->N_THREADS];
    memset(istart_local,0,sizeof(istart_local));

#pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        //PRIVATE VARIABLES
        int x,y,z,i,j;
        int threadnum = omp_get_thread_num();

        int nh_buf=0;
        double delta;
        float prop_buf[2];
        int crd_hi[3], crd_large[3];
        double halo_dist,halo_r,intersect_vol;
        double mass_defc=1;

        //buffers per cell
        float hm_buf[MAX_HALO_CELL];
        
        //debug printing
        int print_counter = 0;
        unsigned long long int istart;
        unsigned long long int count=0;
        //debug total
        double M_cell=0.;

        float *local_hm;
        float *local_sm;
        float *local_sfr;
        int *local_crd;
        local_hm = calloc(localarray_size,sizeof(float));
        local_sm = calloc(localarray_size,sizeof(float));
        local_sfr = calloc(localarray_size,sizeof(float));
        local_crd = calloc(localarray_size*3,sizeof(int));

        //we need a private version
        //TODO: its probably better to split condition and z constants
        struct HaloSamplingConstants hs_constants_priv;
        //NOTE: this will only copy right if there are no arrays in the struct
        hs_constants_priv = *hs_constants;

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
                for (z=0; z<HII_D_PARA; z++){
                    delta = (double)dens_field[HII_R_INDEX(x,y,z)] * growthf;

                    stoc_set_consts_cond(&hs_constants_priv,delta);
                    //Subtract mass from cells near big halos
                    mass_defc = 1;
                    for (j=0;j<nhalo_in;j++){
                        //NOTE: this is actually low res!!!!!
                        crd_large[0] = halofield_large->halo_coords[0 + 3*j];
                        crd_large[1] = halofield_large->halo_coords[1 + 3*j];
                        crd_large[2] = halofield_large->halo_coords[2 + 3*j];
                        //mass subtraction from cell, PRETENDING THEY ARE SPHERES OF RADIUS L_FACTOR
                        halo_r = MtoR(halofield_large->halo_masses[j]) / lo_dim * boxlen; //units of cell width
                        halo_dist = sqrt((crd_large[0] - x)*(crd_large[0] - x) +
                                            (crd_large[1] - y)*(crd_large[1] - y) +
                                            (crd_large[2] - z)*(crd_large[2] - z)); //dist between sphere centres

                        //entirely outside of halo
                        if(halo_dist - L_FACTOR > halo_r){
                            continue;
                        }
                        //entirely within halo
                        else if(halo_dist + L_FACTOR < halo_r){
                            mass_defc = 0;
                            break;
                        }
                        //partially inside halo, pretend cells are spheres to do the calculation much faster without too much error
                        else{
                            intersect_vol = halo_dist*halo_dist + 2*halo_dist*L_FACTOR - 3*L_FACTOR*L_FACTOR;
                            intersect_vol += 2*halo_dist*halo_r + 6*halo_r*L_FACTOR - 3*halo_r*halo_r;
                            intersect_vol *= PI*(halo_r + L_FACTOR - halo_dist) / 12*halo_dist; //volume in cell_width^3

                            mass_defc = 1 - intersect_vol; //since cell volume == 1, M*mass_defc should adjust the mass correctly
                            //due to the nature of DexM it is impossible to be partially in two halos so we break here
                            break;
                        }
                    }
                    // LOG_ULTRA_DEBUG("Cell delta %.2f -> (N,M) (%.2f,%.2e) overlap defc %.2f ps_ratio %.2f",delta,hs_constants_priv.expected_N,hs_constants_priv.expected_N,mass_defc,ps_ratio);
                    //TODO: the ps_ratio part will need to be moved when other CMF scalings are finished
                    hs_constants_priv.expected_M *= mass_defc/ps_ratio;
                    hs_constants_priv.expected_N *= mass_defc/ps_ratio;
                    // LOG_ULTRA_DEBUG("Starting sample (%d,%d,%d) (%d) with delta = %.2f cell"
                                    //  ,x, y, z,lo_dim*lo_dim*lo_dim,delta);

                    stoc_sample(&hs_constants_priv, rng_arr[threadnum], &nh_buf, hm_buf);

                    if(!print_counter && threadnum==0){
                        LOG_ULTRA_DEBUG("First Cell delta=%.2f expects N=%.2f M=%.2e",delta,hs_constants_priv.expected_N,hs_constants_priv.expected_M);
                    }
                    //output total halo number, catalogues of masses and positions
                    M_cell = 0;
                    for(i=0;i<nh_buf;i++){
                        if(hm_buf[i] < Mmin*global_params.HALO_SAMPLE_FACTOR) continue; //save only halos some factor above minimum
                        add_halo_properties(rng_arr[threadnum], astro_params_stoc->M_TURN, hm_buf[i], &hs_constants_priv, prop_buf);

                        place_on_hires_grid(x,y,z,crd_hi,rng_arr[threadnum]);

                        //fill in arrays now, this should be quick compared to the sampling so critical shouldn't slow this down much

                        local_hm[count] = hm_buf[i];
                        local_sm[count] = prop_buf[0];
                        local_sfr[count] = prop_buf[1];
                        local_crd[0 + 3*count] = crd_hi[0];
                        local_crd[1 + 3*count] = crd_hi[1];
                        local_crd[2 + 3*count] = crd_hi[2];
                        count++;
                        M_cell += hm_buf[i];
                        if(count > localarray_size){
                            LOG_ERROR("ran out of memory (%llu halos vs %llu array size)",count,localarray_size);
                            Throw(ValueError);
                        }
                        if(!print_counter && threadnum==0){
                            LOG_ULTRA_DEBUG("Halo %d Mass %.2e Stellar %.2e SFR %.2e",i,hm_buf[i],prop_buf[0],prop_buf[1]);
                        }
                        //LOG_ULTRA_DEBUG("Halo %d Mass %.2e Stellar %.2e SFR %.2e",i,hm_buf[i],prop_buf[0],prop_buf[1]);
                    }
                    if(!print_counter && threadnum==0){
                        LOG_ULTRA_DEBUG("Total N %d Total M %.2e",nh_buf,M_cell);
                        print_counter = 1;
                        print_hs_consts(&hs_constants_priv);
                    }
                    // LOG_ULTRA_DEBUG("cell (%d,%d,%d) %.2e Done, %d (%.2f) halos, %.2e (%.2e) Mass",x,y,z,delta,nh_buf,hs_constants_priv.expected_N,M_cell,hs_constants_priv.expected_M);
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
            init_halo_coords(halofield_out,count_total);
        }
        //we need each thread to be done here before copying the data (total count, indexing, allocation)
#pragma omp barrier

        istart = istart_local[threadnum];
        LOG_SUPER_DEBUG("Thread %d has %llu of %llu halos, concatenating (starting at %llu)...",threadnum,count,count_total,istart);
                
        //copy each local array into the struct
        memcpy(halofield_out->halo_masses + istart,local_hm,count*sizeof(float));
        memcpy(halofield_out->stellar_masses + istart,local_sm,count*sizeof(float));
        memcpy(halofield_out->halo_sfr + istart,local_sfr,count*sizeof(float));
        memcpy(halofield_out->halo_coords + istart*3,local_crd,count*sizeof(int)*3);
        
        //free local arrays
        free(local_crd);
        free(local_sfr);
        free(local_sm);
        free(local_hm);
        
    }

    return 0;
}

//TODO: there's a lot of repeated code here and in build_halo_cats, find a way to merge
int halo_update(gsl_rng ** rng_arr, double z_in, double z_out, struct HaloField *halofield_in, struct HaloField *halofield_out, struct HaloSamplingConstants *hs_constants){
    int nhalo_in = halofield_in->n_halos;
    if(z_in >= z_out){
        LOG_ERROR("halo update must go backwards in time!!! z_in = %.1f, z_out = %.1f",z_in,z_out);
        Throw(ValueError);
    }
    if(nhalo_in == 0){
        LOG_DEBUG("No halos to update, continuing...");

        //allocate dummy arrays so we don't get a Bus Error by freeing unallocated pointers
        init_halo_coords(halofield_out,0);
        return 0;
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

    double expected_nhalo = VOLUME * IntegratedNdM(growth_out,log(Mmin*global_params.HALO_SAMPLE_FACTOR),lnMmax_tb,lnMmax_tb,0,0,user_params_stoc->HMF,0);
    unsigned long long int localarray_size = global_params.MAXHALO_FACTOR * (unsigned long long int)expected_nhalo / user_params_stoc->N_THREADS; //integer division should be fine here
    localarray_size = localarray_size < MAX_HALO_CELL ? MAX_HALO_CELL : localarray_size; //set a minimum number so the end doesn't segfault

    LOG_DEBUG("Beginning stochastic halo sampling (update) on %d halos",nhalo_in);
    LOG_DEBUG("z = %f, Mmin = %e, d = %.3e",z_out,Mmin,delta);
    LOG_DEBUG("Expected N_halo: %.3e, array size per thread %llu (~%.3e GB total)",expected_nhalo,localarray_size,6.*localarray_size*sizeof(int)*user_params_stoc->N_THREADS/1e9);

    unsigned long long int count_total = 0;
    unsigned long long int istart_local[user_params_stoc->N_THREADS];
    memset(istart_local,0,sizeof(istart_local));

    int print_counter = 0;

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
        unsigned long long int istart;

        float *local_hm;
        float *local_sm;
        float *local_sfr;
        int *local_crd;

        local_hm = calloc(localarray_size,sizeof(float));
        local_sm = calloc(localarray_size,sizeof(float));
        local_sfr = calloc(localarray_size,sizeof(float));
        local_crd = calloc(3*localarray_size,sizeof(int));
        
        //we need a private version
        //TODO: its probably better to split condition and z constants
        //also the naming convention should be better between structs/struct pointers
        struct HaloSamplingConstants hs_constants_priv;
        hs_constants_priv = *hs_constants;

        // LOG_DEBUG("HM,SM,SFR,CRD pointers (%p,%p,%p,%p)",local_hm,local_sm,local_sfr,local_crd);

#pragma omp for
        for(ii=0;ii<nhalo_in;ii++){
            M2 = halofield_in->halo_masses[ii];
            if(M2 < Mmin || M2 > Mmax_tb){
                LOG_ERROR("Input Mass = %.2e, something went wrong in the input catalogue",M2);
                Throw(ValueError);
            }
            //LOG_ULTRA_DEBUG("Setting consts for M = %.3e",M2);
            stoc_set_consts_cond(&hs_constants_priv,M2);
            
            //find progenitor halos by sampling halo CMF                    
            //NOTE: MAXIMUM HERE (TO LIMIT PROGENITOR MASS) IS THE DESCENDANT MASS
            //The assumption is that the expected fraction of the progenitor
            stoc_sample(&hs_constants_priv,rng_arr[threadnum],&n_prog,prog_buf);
            
            if(!print_counter && threadnum==0){
                LOG_ULTRA_DEBUG("First Halo Mass=%.2e expects N=%.2f M=%.2e",M2,hs_constants_priv.expected_N,hs_constants_priv.expected_M);
                print_hs_consts(&hs_constants_priv);
            }
            
            propbuf_in[0] = halofield_in->stellar_masses[ii];
            propbuf_in[1] = halofield_in->halo_sfr[ii];

            //place progenitors in local list
            M_prog = 0;
            for(jj=0;jj<n_prog;jj++){
                if(prog_buf[jj] < Mmin*global_params.HALO_SAMPLE_FACTOR) continue; //save only halos some factor above minimum

                //LOG_ULTRA_DEBUG("updating props");
                update_halo_properties(rng_arr[threadnum], prog_buf[jj], &hs_constants_priv, propbuf_in, propbuf_out);
                //LOG_ULTRA_DEBUG("Assigning");

                local_hm[count] = prog_buf[jj];
                local_crd[3*count + 0] = halofield_in->halo_coords[3*ii+0];
                local_crd[3*count + 1] = halofield_in->halo_coords[3*ii+1];
                local_crd[3*count + 2] = halofield_in->halo_coords[3*ii+2];
                
                local_sm[count] = propbuf_out[0];
                local_sfr[count] = propbuf_out[1];
                count++;
                
                if(count > localarray_size){
                    LOG_ERROR("ran out of memory (%llu halos vs %llu array size)",count+n_prog,localarray_size);
                    Throw(ValueError);
                }
                
                if(!print_counter && threadnum==0){
                    M_prog += prog_buf[jj];
                    LOG_ULTRA_DEBUG("Halo %d Mass %.2e Stellar %.2e SFR %.2e",jj,prog_buf[jj],propbuf_out[0],propbuf_out[1]);
                }
            }
            if(!print_counter && threadnum==0){
                LOG_ULTRA_DEBUG("Total N %d Total M %.2e",n_prog,M_prog);
                print_counter = 1;
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
            init_halo_coords(halofield_out,count_total);
        }
        LOG_SUPER_DEBUG("Thread %d has %llu of %llu halos, concatenating (starting at %llu)...",threadnum,count,count_total,istart_local[threadnum]);

//we need each thread to be done here before copying the data
#pragma omp barrier
        
        istart = istart_local[threadnum];
        //copy each local array into the struct
        memcpy(halofield_out->halo_masses + istart,local_hm,count*sizeof(float));
        memcpy(halofield_out->stellar_masses + istart,local_sm,count*sizeof(float));
        memcpy(halofield_out->halo_sfr + istart,local_sfr,count*sizeof(float));
        memcpy(halofield_out->halo_coords + istart*3,local_crd,count*sizeof(int)*3);
        free(local_crd);
        free(local_sfr);
        free(local_sm);
        free(local_hm);
    }

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
    
    struct HaloSamplingConstants hs_constants;
    stoc_set_consts_z(&hs_constants,redshift,redshift_prev);
    print_hs_consts(&hs_constants);

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
        LOG_DEBUG("First few Stellar: %11.3e %11.3e %11.3e",halos->stellar_masses[0],halos->stellar_masses[1],halos->stellar_masses[2]);
        LOG_DEBUG("First few SFR:     %11.3e %11.3e %11.3e",halos->halo_sfr[0],halos->halo_sfr[1],halos->halo_sfr[2]);
    }
    
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
        free_siginv_spline();
    }
    free_dNdM_tables();

    free_rng_threads(rng_stoc);
    return 0;
}

//This, for the moment, grids the PERTURBED halo catalogue.
//TODO: make a way to output both types by making 2 wrappers to this function that pass in arrays rather than structs
//NOTE: this function is quite slow to generate fixed halo boxes, however I don't mind since it's a debug case
//  If we want to make it faster just replace the integrals with the existing interpolation tables
//TODO: I should also probably completely separate the fixed and sampled grids into two functions which this calls
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
        if(flag_options->PHOTON_CONS_ALPHA){
            alpha_esc = get_alpha_fit(redshift);
        }
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
        if(flag_options->FIXED_HALO_GRIDS || LOG_LEVEL >= DEBUG_LEVEL){
            init_ps();
            
            M_min = minimum_source_mass(redshift,astro_params,flag_options);
            volume = VOLUME / HII_TOT_NUM_PIXELS;
            M_max = RtoM(user_params->BOX_LEN / user_params->HII_DIM * L_FACTOR); //mass in cell of mean dens

            if(user_params->USE_INTERPOLATION_TABLES){
                initialiseSigmaMInterpTable(M_min,global_params.M_MAX_INTEGRAL); //this needs to be initialised above MMax because of Nion_General
            }

            growth_z = dicke(redshift);
            alpha_star = astro_params->ALPHA_STAR;
            norm_star = astro_params->F_STAR10;
            t_star = astro_params->t_STAR;
           
            lnMmax = log(M_max);
            lnMmin = log(M_min*global_params.HALO_SAMPLE_FACTOR);
            sigma_max = EvaluateSigma(lnMmax,0,&dummy);

            prefactor_mass = RHOcrit * cosmo_params->OMm / sqrt(2.*PI);
            prefactor_star = RHOcrit * cosmo_params->OMb * norm_star * norm_esc;
            prefactor_sfr = RHOcrit * cosmo_params->OMb * norm_star / t_star / t_hubble(redshift);
            
            Mlim_Fstar = Mass_limit_bisection(M_min, M_max, alpha_star, norm_star);
            Mlim_Fesc = Mass_limit_bisection(M_min, M_max, alpha_esc, norm_esc);

            hm_expected = IntegratedNdM(growth_z, lnMmin, lnMmax, lnMmax, 0, 1, user_params->HMF,0);
            sm_expected = Nion_General(redshift, M_min, astro_params->M_TURN, alpha_star, alpha_esc, norm_star, norm_esc, Mlim_Fstar, Mlim_Fesc);
            sfr_expected = Nion_General(redshift, M_min, astro_params->M_TURN, alpha_star, 0., norm_star, 1., Mlim_Fstar, 0.);

            sm_expected *= prefactor_star;
            sfr_expected *= prefactor_sfr;
        }

        //do the mean HMF box
        //The default 21cmFAST has a strange behaviour where the nonlinear density is used as linear,
        //the condition mass is at mean density, but the total cell mass is multiplied by delta 
        //This part mimics that behaviour
        //TODO: interpolation tables here (although the mean boxes are just a test)
        if(flag_options->FIXED_HALO_GRIDS){
            LOG_DEBUG("Mean halo boxes || Mmin = %.2e | Mmax = %.2e (s=%.2e) | z = %.2e | D = %.2e | cellvol = %.2e",M_min,M_max,sigma_max,redshift,growth_z,volume);
#pragma omp parallel num_threads(user_params->N_THREADS)
            {
                int i;
                double dens;
                double mass, wstar, sfr, h_count;
                double wsfr;
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
                        mass = M_max * (1+dens) / volume;
                        wstar = M_max * (1+dens) * cosmo_params->OMb / cosmo_params->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) * norm_esc * pow(M_max*(1+dens)/1e10,alpha_esc) / volume;
                        sfr = M_max * (1+dens) * cosmo_params->OMb / cosmo_params->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) / t_star / t_hubble(redshift) / volume;
                        h_count = 1;
                    }
                    else{
                        //calling IntegratedNdM with star and SFR need special care for the f*/fesc clipping, and calling NionConditionalM for mass includes duty cycle
                        //neither of which I want
                        mass = IntegratedNdM(growth_z,lnMmin,lnMmax,lnMmax,dens,1,user_params->HMF,1) * prefactor_mass * (1+dens);
                        h_count = IntegratedNdM(growth_z,lnMmin,lnMmax,lnMmax,dens,0,user_params->HMF,1) * prefactor_mass * (1+dens);

                        wstar = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_max, Deltac, dens, astro_params->M_TURN
                                                , astro_params->ALPHA_STAR, alpha_esc, astro_params->F_STAR10, astro_params->F_ESC10
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
                double wsfr;
#pragma omp for
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] = 0.0;
                    grids->wstar_mass[idx] = 0.0;
                    grids->halo_sfr[idx] = 0.0;
                    grids->whalo_sfr[idx] = 0.0;
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
                    wsfr = halos->halo_sfr[i_halo]*fesc;
                    //will probably need unweighted Stellar later on
                    //Lx as well when that scatter is included

#pragma omp atomic update
                    grids->halo_mass[HII_R_INDEX(x, y, z)] += m;
#pragma omp atomic update
                    grids->wstar_mass[HII_R_INDEX(x, y, z)] += wstar;
#pragma omp atomic update
                    grids->halo_sfr[HII_R_INDEX(x, y, z)] += sfr;
#pragma omp atomic update
                    grids->whalo_sfr[HII_R_INDEX(x, y, z)] += wsfr;
                    //sometimes we get zeromass halos (e.g tinkering with magnitude cuts)
                    if(m>0){
#pragma omp atomic update
                        grids->count[HII_R_INDEX(x, y, z)] += 1;
                    }

                    if(LOG_LEVEL >= DEBUG_LEVEL){
                        hm_avg += m;
                        sm_avg += wstar;
                        sfr_avg += sfr;
                    }
                }
                
                //convert to densities
#pragma omp for
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] /= volume;
                    grids->wstar_mass[idx] /= volume;
                    grids->halo_sfr[idx] /= volume;
                    grids->whalo_sfr[idx] /= volume;
                }
            }
            if(LOG_LEVEL >= DEBUG_LEVEL){
                hm_avg /= volume*HII_TOT_NUM_PIXELS;
                sm_avg /= volume*HII_TOT_NUM_PIXELS;
                sfr_avg /= volume*HII_TOT_NUM_PIXELS;
            }
        }

        if(user_params->USE_INTERPOLATION_TABLES && (flag_options->FIXED_HALO_GRIDS|| LOG_LEVEL >= DEBUG_LEVEL)){
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
/* type==0: UMF/CMF value at a list of masses
 * type==1: Integrated CMF in a single condition at multiple masses N(>M)
 * type==2: Integrated CMF in multiple conditions in the entire mass range
 * type==3: Expected CMF from a list of conditions
 * type==4: Halo catalogue and excess mass from a list of conditions (stoc_sample)
 * type==5: Halo catalogue and coordinates from a list of conditions using the grid/structs (halo_update / build_halo_cat level)
 * type==6: N(<M) interpolation table output for a list of masses in one condition
 * type==7: INVERSE N(<M) interpolation table output for a list of masses in one condition */
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
        double test,dummy;
        int err=0;
        int i,j;

        struct HaloSamplingConstants hs_const_struct;
        struct HaloSamplingConstants *hs_constants = &hs_const_struct;
        LOG_DEBUG("Setting z constants. %.3f %.3f",z_out,z_in);
        stoc_set_consts_z(hs_constants,z_out,z_in);
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

        LOG_DEBUG("TEST FUNCTION: type = %d z = (%.2f,%.2f), Mmin = %.3e, cond = %.3e, M(%d)=[%.2e,%.2e,%.2e...]",type,z_out,z_in,Mmin,condition,n_mass,M[0],M[1],M[2]);

        //Since the conditional MF is press-schecter, we rescale by a factor equal to the ratio of the collapsed fractions (n_order == 1) of the UMF
        double ps_ratio = 1.;
        if(!hs_constants->update && user_params->HMF!=0){
            ps_ratio = IntegratedNdM(growth_out,lnMmin,lnMcond,lnMcond,0,1,0,0) / IntegratedNdM(growth_out,lnMmin,lnMcond,lnMcond,0,1,user_params->HMF,0);
        }

        if(type==0){
            stoc_set_consts_cond(hs_constants,condition);
            Mcond = hs_constants->M_cond;
            lnMcond = hs_constants->lnM_cond;
            delta = hs_constants->delta;
            //using seed to select CMF or UMF since there's no RNG here
            bool cmf_flag = seed==0;

            //parameters for CMF
            double prefactor = cmf_flag ? RHOcrit / sqrt(2.*PI) * cosmo_params_stoc->OMm : 1.;
            double dummy;
            struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
                .redshift = z_out,
                .growthf = growth_out,
                .delta = delta,
                .n_order = 0,
                .M_max = lnMcond,
                .sigma_max = hs_constants->sigma_cond,
                .HMF = user_params_stoc->HMF,
                .CMF = cmf_flag,
            };
            
            #pragma omp parallel for private(test)
            for(i=0;i<n_mass;i++){
                //conditional ps mass func * pow(M,n_order)
                if((M[i] < Mmin) || M[i] > MMAX_TABLES || (seed != 0 && M[i] > Mcond)){
                    test = 0.;
                }
                else{
                    test = MnMassfunction(log(M[i]),(void*)&parameters_gsl_MF_con);
                    
                    //convert to dndlnm
                    test = test * prefactor;
                }
                // LOG_ULTRA_DEBUG(" D %.1e | M1 %.1e | M2 %.1e | d %.1e | s %.1e -> %.1e",
                //                 growth_out,M[i],Mcond,delta,hs_constants->sigma_cond,test);
                result[i] = test;
            }
        }
        else if(type==1){
            //integrate CMF -> N(Ml<M<Mh) in one condition
            //TODO: make it possible to integrate UMFs
            //seed gives n_order
            stoc_set_consts_cond(hs_constants,condition);
            Mcond = hs_constants->M_cond;
            lnMcond = hs_constants->lnM_cond;
            delta = hs_constants->delta;

            double lnM_hi, lnM_lo;
            double dummy;
            #pragma omp parallel for private(test,lnM_hi,lnM_lo) num_threads(user_params->N_THREADS)
            for(i=0;i<n_mass;i++){
                // LOG_ULTRA_DEBUG("%d %d D %.1e | Ml %.1e | Mu %.1e | Mc %.1e | Mm %.1e | d %.1e | s %.1e",
                //                 i, i+n_mass, growth_out,M[i],M[i+n_mass],Mcond,Mmin,delta,EvaluateSigma(lnMcond,0,&dummy));

                lnM_lo = log(M[i]) < lnMmin ? lnMmin : log(M[i]);
                lnM_hi = log(M[i+n_mass]) > lnMcond ? lnMcond : log(M[i+n_mass]);

                if (lnM_lo > lnMcond || lnM_hi < lnMmin){
                    result[i] = 0;
                    continue;
                }

                test = IntegratedNdM(growth_out,lnM_lo,lnM_hi,lnMcond,delta,seed,user_params->HMF,1);
                //This is a debug case, testing that the integral of M*dNdM == FgtrM
                if(seed == 1){
                    test = EvaluateFgtrM(growth_out,lnM_lo,delta,EvaluateSigma(lnMcond,0,&dummy)) - EvaluateFgtrM(growth_out,lnM_hi,delta,EvaluateSigma(lnMcond,0,&dummy));
                    result[i+n_mass] = test * Mcond * ps_ratio;
                    // LOG_ULTRA_DEBUG("==> %.8e",result[i+n_mass]);
                }

                result[i] = test * Mcond / sqrt(2.*PI) * ps_ratio;
                // LOG_ULTRA_DEBUG("==> %.8e",result[i]);
            }
        }

        else if(type==2){
            //intregrate CMF -> N_halos in many conditions
            //TODO: make it possible to integrate UMFs
            //quick hack: seed gives n_order
            double dummy;
            #pragma omp parallel private(test,Mcond,lnMcond,delta,dummy) num_threads(user_params->N_THREADS)
            {
                //we need a private version
                //TODO: its probably better to split condition and z constants
                struct HaloSamplingConstants hs_constants_priv;
                hs_constants_priv = *hs_constants;
                double cond;
                #pragma omp for
                for(i=0;i<n_mass;i++){
                    cond = M[i];
                    stoc_set_consts_cond(&hs_constants_priv,cond);
                    Mcond = hs_constants_priv.M_cond;
                    lnMcond = hs_constants_priv.lnM_cond;
                    delta = hs_constants_priv.delta;
                    
                    // LOG_ULTRA_DEBUG("%d %d D %.1e | Ml %.1e | Mc %.1e| d %.1e | s %.1e",
                    //                 i,i+n_mass,growth_out,Mmin,Mcond,delta,EvaluateSigma(lnMcond,0,&dummy));

                    test = IntegratedNdM(growth_out,lnMmin,lnMcond,lnMcond,delta,seed,user_params->HMF,1);
                    // LOG_ULTRA_DEBUG("==> %.8e",test);
                    //conditional MF multiplied by a few factors
                    result[i] = test  * Mcond / sqrt(2.*PI) * ps_ratio;
                    //This is a debug case, testing that the integral of M*dNdM == FgtrM
                    if(seed == 1){
                        test = EvaluateFgtrM(growth_out,lnMmin,delta,hs_constants_priv.sigma_cond) * Mcond;
                        result[i+n_mass] = test * ps_ratio;
                    }
                }
            }
        }
        
        //Cell CMF from one cell, given M as cell descendant halos
        //uses a constant mass binning since we use the input for descendants
        else if(type==3){
            double out_cmf[100];
            double out_bins[100];
            int n_bins = 100;
            double prefactor = RHOcrit / sqrt(2.*PI) * cosmo_params_stoc->OMm;
            double dummy,test;
            double tot_mass=0;
            double lnMbin_max = hs_constants->lnM_max_tb; //arbitrary bin maximum
            
            struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
                .redshift = z_out,
                .growthf = growth_out,
                .delta = 0,
                .n_order = 0,
                .M_max = 0,
                .sigma_max = 0,
                .HMF = user_params_stoc->HMF,
                .CMF = 1,
            };

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
                double cond;
                hs_constants_priv = *hs_constants;
                #pragma omp for
                for(j=0;j<n_mass;j++){
                    tot_mass += M[j];
                    cond = M[j];
                    stoc_set_consts_cond(&hs_constants_priv,cond);
                    lnMcond = hs_constants_priv.lnM_cond;

                    parameters_gsl_MF_con.M_max = lnMcond;
                    parameters_gsl_MF_con.sigma_max = hs_constants_priv.sigma_cond;
                    parameters_gsl_MF_con.delta = hs_constants_priv.delta;
                    for(i=0;i<n_bins;i++){
                        lnM_bin = out_bins[i];
                        
                        //conditional ps mass func * pow(M,n_order)
                        if(lnM_bin < lnMmin || lnM_bin > lnMcond){
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
                        
                        //only save halos above the save limit, but add all halos to the totals
                        if(out_hm[i]<Mmin*global_params.HALO_SAMPLE_FACTOR){
                            continue;   
                        }

                        //critical is bad, but this is a test function so eeeehh
                        #pragma omp critical
                        {
                            result[1+4*n_mass+(n_halo_tot++)] = out_hm[i];
                        }
                        n_halo_out++;
                    }
                    //output descendant statistics
                    result[0*n_mass + 1 + j] = (double)hs_constants_priv.expected_N_save;
                    result[1*n_mass + 1 + j] = (double)hs_constants_priv.expected_M_save;
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
            init_hmf(halos_out);
            init_hmf(halos_in);

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
                halo_update(rng_stoc, z_in, z_out, halos_in, halos_out, hs_constants);
            }
            else{
                //NOTE: halomass_in is linear delta at z = redshift_out
                LOG_SUPER_DEBUG("assigning input arrays w %d (%d) halos",n_mass,HII_TOT_NUM_PIXELS);
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
            double y_in,x_in,mass;
            x_in = hs_constants->cond_val;
            mass = hs_constants->M_cond;
            stoc_set_consts_cond(hs_constants,condition);
            #pragma omp parallel for private(test,x_in,y_in)
            for(i=0;i<n_mass;i++){
                y_in = log(M[i]);
                if(y_in < lnMmin){
                    result[i] = 0;
                    continue;
                }
                test = EvaluatedNdMSpline(x_in,y_in);
                result[i] = test * mass / sqrt(2.*PI);
                LOG_ULTRA_DEBUG("dNdM table: x = %.6e, y = %.6e z = %.6e",x_in,y_in,test);
            }
        }

        //return dNdM INVERSE table result for M at a bunch of probabilities
        else if(type==7){
            double y_in,x_in;
            stoc_set_consts_cond(hs_constants,condition);
            x_in = hs_constants->cond_val;
            #pragma omp parallel for private(test,y_in)
            for(i=0;i<n_mass;i++){
                if(M[i] < 0 || M[i] > 1){
                    LOG_ERROR("invalid probability %.2f",M[i]);
                    Throw(ValueError);
                }
                y_in = M[i];
                test = gsl_spline2d_eval(Nhalo_inv_spline,x_in,y_in,Nhalo_inv_cond_acc,Nhalo_inv_prob_acc);
                result[i] = exp(test);
                LOG_ULTRA_DEBUG("dNdM inverse table: %.6e x = %.6e, y = %.6e z = %.6e",condition,x_in,y_in,test);
            }
        }
        else if(type==8){
            double R = z_out;
            double mfp = z_in;
            LOG_DEBUG("Starting mfp filter");
            test_mfp_filter(user_params,cosmo_params,astro_params,flag_options,M,R,mfp,result);
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