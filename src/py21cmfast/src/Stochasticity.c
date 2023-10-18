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

//we need to define a density minimum for the tables, since we are in lagrangian density / linear growth it's possible to go below -1
//so we explicitly set a minimum here which sets table limits and puts no halos in cells below that (Lagrangian) density
#define DELTA_MIN -1

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
    double M_max_tables;
    double lnM_max_tb;
    double sigma_min;

    //table info
    double tbl_xmin;
    double tbl_xwid;
    double tbl_ymin;
    double tbl_ywid;
    double tbl_pmin;
    double tbl_pwid;

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

//The sigma interp table is regular in mass, nt sigma so we need to loop
//TODO: make a new RGI from the sigma tables if we take the partition method seriously,
double EvaluateSigmaInverse(double sigma){
    int idx;
    for(idx=0;idx<NMass;idx++){
        if(sigma < Sigma_InterpTable[idx]) break;
    }
    if(idx == NMass){
        LOG_ERROR("sigma inverse out of bounds.");
        Throw(TableEvaluationError);
    }
    double table_val_0 = Sigma_InterpTable[idx];
    double table_val_1 = Sigma_InterpTable[idx];
    double interp_point = (sigma - table_val_0)/(table_val_1-table_val_0);

    return table_val_0*(1-interp_point) + table_val_1*(interp_point);
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
            //Filter CMF type by CMF_MODE parameter (==1 for set CMF, ==2 for resnormalised, ==3 for rescaled EPS, ==4 for Tramonte local, ==5 for Tramonte global etc)
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

//TODO:Move all these tables to the const struct
double **Nhalo_spline;
double **Nhalo_inv_spline;
double *M_exp_spline;
double *sigma_inv_spline;
//double EvaluateRGTable2D(double x, double y, double **z_arr, double x_min, double x_width, double y_min, double y_width, int n_x){
// double EvaluateRGTable1D(double x, double *y_arr, double x_min, double x_width){


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
        if(delta < DELTA_MIN || delta > Deltac){
            LOG_ERROR("Invalid delta %.3f",delta);
            Throw(ValueError);
        }
    }
    //generation from grid, x=delta, param_2 = log(M_filt)
    else{
        nx = global_params.N_DELTA_INTERP;
        lnM_cond = param;
        if(xmin < DELTA_MIN || xmax > Deltac){
            LOG_ERROR("Invalid delta [%.3f,%.3f]",xmin,xmax);
            Throw(ValueError);
        }
    }
    //Check for invalid delta

    //y(min mass) is always the same
    ny = global_params.N_MASS_INTERP;
    np = global_params.N_PROB_INTERP;

    double xa[nx], ya[ny];
    double one_halo;

    int i,j,k;
    //set up coordinate grids
    for(i=0;i<nx;i++) xa[i] = xmin + (xmax - xmin)*((double)i)/((double)nx-1);
    for(j=0;j<ny;j++) ya[j] = ymin + (ymax - ymin)*((double)j)/((double)ny-1);
    
    //allocate tables
    Nhalo_spline = (double **)calloc(nx,sizeof(double *));
    for(i=0;i<nx;i++) Nhalo_spline[i] = (double *)calloc(ny,sizeof(double));

    Nhalo_inv_spline = (double **)calloc(nx,sizeof(double *));
    for(i=0;i<nx;i++) Nhalo_inv_spline[i] = (double *)calloc(np,sizeof(double));
    M_exp_spline = (double *)calloc(nx,sizeof(double));

    #pragma omp parallel num_threads(user_params_stoc->N_THREADS) private(i,j,k) firstprivate(delta,lnM_cond)
    {
        double x,y,buf;
        double norm;
        double lnM_prev,lnM_p;
        double prob;
        double p_prev,p_target;
        double k_next;

        #pragma omp for
        for(i=0;i<nx;i++){
            x = xa[i];
            //set the condition
            if(update) lnM_cond = x;
            else delta = x;

            lnM_prev = ymin;
            p_prev = 0;
            //TODO: if this line affects performance, set it to zero, since it won't affect much
            one_halo = 1. / exp(lnM_cond) * sqrt(2*PI);

            //setting to zero for high delta 
            //this one needs to be done before the norm is calculated
            if(delta > MAX_DELTAC_FRAC*Deltac){
                //In the last bin, n_halo / mass * sqrt2pi interpolates toward one halo
                for(j=0;j<ny;j++)
                    Nhalo_spline[i][j] = one_halo;
                
                //Similarly, the inverse CDF tends to a step function at lnM_cond
                for(k=0;k<np;k++)
                    Nhalo_inv_spline[i][k] = lnM_cond;

                continue;
            }

            norm = IntegratedNdM(growth1,ymin,ymax,lnM_cond,delta,0,user_params_stoc->HMF,1);
            M_exp_spline[i] = IntegratedNdM(growth1,ymin,ymax,lnM_cond,delta,1,user_params_stoc->HMF,1);
            // LOG_ULTRA_DEBUG("cond x: %.2e (%d) ==> %.8e / %.8e",x,i,norm,M_exp_spline[i]);
            
            //if the condition has no halos set the dndm table
            //the inverse table will be unaffected since p=0
            if(norm==0){
                for(j=0;j<ny;j++)
                    Nhalo_spline[i][j] = 0.;
                    
                for(k=0;k<np;k++)
                    Nhalo_inv_spline[i][k] = 0.;
                continue;
            }

            Nhalo_spline[i][0] = 0.;
            Nhalo_inv_spline[i][0] = ymin;

            //reset probability finding
            k=np-1;
            p_target = 0.;

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
                Nhalo_spline[i][j] = buf;
                
                //done with inverse table
                if(k < 0) continue;

                prob = buf / norm;
                //catch some norm errors
                if(prob != prob){
                    LOG_ERROR("Normalisation error in table generation");
                    Throw(TableGenerationError);
                }

                // LOG_ULTRA_DEBUG("Int || x: %.2e (%d) y: %.2e (%d) ==> %.8e / %.8e",update ? exp(x) : x,i,exp(y),j,prob,log1p(-prob));
                //There are time where we have gone over the probability (machine precision) limit before reaching the mass limit
                //  we set the final point to be minimum probability at the maximum mass, which crosses the true CDF in the final bin
                //  but is the best we can do without a rootfind
                //APPROXIMATION-WIP: We expect a sharp dropoff, so extrapolate the second last bin to the maximum mass, then go to the minimum
                //  The first time it goes over p==1, it does the extrapolation at the previous gradient
                //  then one more iteration will occur where it fills the final probabilities straight down
                if(prob >= 1.){
                    // if(lnM_prev == lnM_cond || k < 2){
                    prob = global_params.MIN_LOGPROB;
                    // }
                    // else{
                    //     prob = (global_params.MIN_LOGPROB/(np-1))/(Nhalo_inv_spline[i][k+1] - Nhalo_inv_spline[i][k+2])*(lnM_cond-Nhalo_inv_spline[i][k+1]);
                    //     prob += global_params.MIN_LOGPROB*(1 - (double)(k+1)/(double)(np-1));
                    //     if(i==50){
                    //         LOG_DEBUG("Second Last bin info %.2e %d: k+2 (%11.3e,%11.3e) k+1 (%11.3e,%11.3e) k (%11.3e,%11.3e)", exp(lnM_cond), k, Nhalo_inv_spline[i][k+2],
                    //                 global_params.MIN_LOGPROB*(1 - (double)(k+2)/(double)(np-1)), Nhalo_inv_spline[i][k+1],
                    //                 global_params.MIN_LOGPROB*(1 - (double)(k+2)/(double)(np-1)), lnM_cond,prob);
                    //     }
                    // }
                    y = lnM_cond;
                }
                else prob = log1p(-prob);

                if(p_prev < p_target){
                        LOG_ERROR("Target moved up?");
                        Throw(TableGenerationError);
                }
                //loop through the remaining spaces in the inverse table and fill them
                while(prob <= p_target && k >= 0){
                    //since we go ascending in y, prob > prob_prev
                    //NOTE: linear interpolation in (lnM,log(1-p))
                    lnM_p = (p_prev-p_target)*(y - lnM_prev)/(p_prev-prob) + lnM_prev;
                    Nhalo_inv_spline[i][k] = lnM_p;

                    // LOG_ULTRA_DEBUG("Found c: %.2e p: (%.2e,%.2e,%.2e) (c %d, m %d, p %d) z: %.5e",update ? exp(x) : x,p_prev,p_target,prob,i,j,k,exp(lnM_p));

                    k--;
                    p_target = global_params.MIN_LOGPROB*(1 - (double)k/(double)(np-1));
                }
                //keep the value at the previous mass bin for interpolation
                p_prev = prob;
                lnM_prev = y;
            }
        }
    }
    LOG_DEBUG("Done.");
}

void free_dNdM_tables(){
    int i;
    for(i=0;i<global_params.N_MASS_INTERP;i++) free(Nhalo_spline[i]);
    free(Nhalo_spline);

    for(i=0;i<global_params.N_MASS_INTERP;i++) free(Nhalo_inv_spline[i]);
    free(Nhalo_inv_spline);

    free(M_exp_spline);
}

//TODO: Speedtest the RGI interpolation present in Spintemp etc...
//  Save the X/Y/Z from the table builder and apply the Minihalo 2D interpolation
//NOTE: if p(x) is uniform, p(log(1-x)) follows the exponential distribution
//  But the gsl_ran_exponential function does the exact same thing but adds a mean
double sample_dndM_inverse(double condition, struct HaloSamplingConstants * hs_constants, gsl_rng * rng){
    double p_in, min_prob;
    p_in = log1p(-gsl_rng_uniform(rng));
    if(p_in < global_params.MIN_LOGPROB) p_in = global_params.MIN_LOGPROB; //we assume that M(min_logprob) ~ M_cond
    return EvaluateRGTable2D(condition,p_in,Nhalo_inv_spline,hs_constants->tbl_xmin,hs_constants->tbl_xwid,hs_constants->tbl_pmin,hs_constants->tbl_pwid);
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
    const_struct->lnM_min = log(M_min);
    const_struct->M_max_tables = global_params.M_MAX_INTEGRAL;
    const_struct->lnM_max_tb = log(const_struct->M_max_tables);


    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(const_struct->M_min / 2,const_struct->M_max_tables);
    }
    const_struct->sigma_min = EvaluateSigma(const_struct->lnM_min,0,&dummy);

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

        const_struct->tbl_xmin = const_struct->lnM_min;
        const_struct->tbl_xwid = (const_struct->lnM_max_tb - const_struct->lnM_min)/(global_params.N_MASS_INTERP - 1);
    }
    else {
        double M_cond = RHOcrit * cosmo_params_stoc->OMm * VOLUME / HII_TOT_NUM_PIXELS;
        const_struct->M_cond = M_cond;
        const_struct->lnM_cond = log(M_cond);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond,0,&dummy);
        const_struct->update = 0;
        initialise_dNdM_tables(DELTA_MIN, Deltac, const_struct->lnM_min, const_struct->lnM_max_tb, const_struct->growth_out, const_struct->lnM_cond, false);

        const_struct->tbl_xmin = DELTA_MIN;
        const_struct->tbl_xwid = (Deltac+1)/global_params.N_DELTA_INTERP;
    }

    const_struct->tbl_ymin = const_struct->lnM_min;
    const_struct->tbl_ywid = (const_struct->lnM_max_tb - const_struct->lnM_min)/(global_params.N_MASS_INTERP - 1);

    const_struct->tbl_pmin = global_params.MIN_LOGPROB; //min log(1-p)
    const_struct->tbl_pwid = (-global_params.MIN_LOGPROB)/(global_params.N_PROB_INTERP - 1);
    LOG_DEBUG("Done.");
    return;
}

//set the constants which are calculated once per condition
void stoc_set_consts_cond(struct HaloSamplingConstants *const_struct, double cond_val){
    double m_exp,n_exp,dummy;

    //Here the condition is a mass, volume is the Lagrangian volume and delta_l is set by the
    //redshift difference which represents the difference in delta_crit across redshifts
    if(const_struct->update){
        const_struct->M_cond = cond_val;
        const_struct->lnM_cond = log(cond_val);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond,0,&dummy);
        //mean stellar mass of this halo mass, used for stellar z correlations
        // const_struct->mu_desc_star = fmin(astro_params_stoc->F_STAR10
        //                                 * pow(cond_val/1e10,astro_params_stoc->ALPHA_STAR)
        //                                 * exp(-astro_params_stoc->M_TURN/cond_val),1) * cond_val;
        const_struct->cond_val = const_struct->lnM_cond;
    }
    //Here the condition is a cell of a given density, the volume/mass is given by the grid parameters
    else{
        const_struct->delta = cond_val;
        const_struct->cond_val = cond_val;
    }

    //the splines don't work well for cells above Deltac, but there CAN be cells above deltac, since this calculation happens
    //before the overlap, and since the smallest dexm mass is M_cell*(1.01^3) there *could* be a cell above Deltac not in a halo
    if(!const_struct->update){
        if(cond_val > MAX_DELTAC_FRAC*Deltac){
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
    n_exp = EvaluateRGTable2D(const_struct->cond_val,const_struct->lnM_max_tb*0.999, Nhalo_spline,
                            const_struct->tbl_xmin,const_struct->tbl_xwid,
                            const_struct->tbl_ymin,const_struct->tbl_ywid);
    // n_exp = EvaluatedNdMSpline(const_struct->cond_val,const_struct->lnM_max_tb); //should be the same as < lnM_cond, but that can hide some interp errors
    //TODO: remove if performance is affected by this line
    
    //NOTE: while the most common mass functions have simpler expressions for f(<M) (erfc based) this will be general, and shouldn't impact compute time much
    m_exp = EvaluateRGTable1D(const_struct->cond_val,M_exp_spline,const_struct->tbl_xmin,const_struct->tbl_xwid);
    const_struct->expected_N = n_exp * const_struct->M_cond / sqrt(2.*PI);
    const_struct->expected_M = m_exp * const_struct->M_cond / sqrt(2.*PI);

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

//calculates halo properties from astro parameters plus the correlated rng
//TODO: make the input and output labeled structs
//The inputs include all properties with a separate RNG
//The outputs include all halo properties PLUS all properties which cannot be recovered when mixing all the halos together
//  i.e escape fraction weighting, minihalo stuff that has separate parameters
//INPUT ARRAY || 0: Stellar mass RNG, 1: SFR RNG
//OUTPUT ARRAY || 0: Stellar Mass, 1: SFR 2: n_ion 3: f_esc weighted SFR
void set_halo_properties(float halo_mass, float M_turn_a, float M_turn_m, float t_h, double norm_esc_var, double alpha_esc_var, float * input, float * output){
    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;
    double sigma_sfr = astro_params_stoc->SIGMA_SFR;

    double f7 = astro_params_stoc->F_STAR7_MINI;
    double fa_m = astro_params_stoc->ALPHA_STAR_MINI;
    double fesc7 = astro_params_stoc->F_ESC7_MINI;

    double fstar_mean, fstar_mean_mini, sfr_mean, wsfr_mean;
    double f_sample, f_sample_mini, sm_sample, n_ion_sample, sfr_sample, wsfr_sample;
    double fesc,fesc_mini;

    //This clipping is normally done with the mass_limit_bisection root find.
    //I can't do that here with the stochasticity, since the f_star clipping happens AFTER the sampling
    //TODO: It could save some `pow` calls with F_esc if I compute a mass limit outside the loop
    //NOTE: I can only do this if f_esc remains non-stochastic, this will also be irrelevant with mean property interptables
    fesc = fmin(norm_esc_var*pow(halo_mass/1e10,alpha_esc_var),1);
    
    //A flattening of the high-mass FSTAR, HACKY VERSION FOR NOW
    //TODO: code it properly with new parameters and pivot point defined somewhere
    //NOTE: we don't want an upturn even with a negative ALPHA_STAR
    if(astro_params_stoc->ALPHA_STAR > -0.61){
        fstar_mean = f10 * exp(-M_turn_a/halo_mass) * pow(2.6e11/1e10,astro_params_stoc->ALPHA_STAR);
        fstar_mean /= pow(halo_mass/2.6e11,-astro_params_stoc->ALPHA_STAR) + pow(halo_mass/2.6e11,0.61);
    }
    else{
        fstar_mean = f10 * pow(halo_mass/1e10,fa) * exp(-M_turn_a/halo_mass);
    }

    //in order to remain consistent with the minihalo treatment in default (Nion_a * exp(-M/M_a) + Nion_m * exp(-M/M_m - M_a/M))
    //  we treat the minihalos as a shift in the mean, where each halo will have both components
    //  TODO: develop a probabilistic duty cycle model which assigns individual halos as atomically cooled, molecularly cooled, or inactive
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
    f_sample = exp(-sigma_star*sigma_star/2 + input[0]*sigma_star);
    //This clipping is normally done with the mass_limit_bisection root find. hard to do with stochastic
    //TODO: just put it in a table
    f_sample = fmin(fstar_mean * f_sample,1);
    f_sample_mini = fmin(fstar_mean_mini * f_sample,1);

    sm_sample = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * (f_sample + f_sample_mini); //f_star is galactic GAS/star fraction, so OMb is needed
    n_ion_sample = halo_mass * (cosmo_params_stoc->OMb / cosmo_params_stoc->OMm) * (f_sample*global_params.Pop2_ion*fesc + f_sample_mini*global_params.Pop3_ion*fesc_mini);

    sfr_mean = sm_sample / (astro_params_stoc->t_STAR * t_h);
    // wsfr_mean = sfr_mean * fesc;
    
    //Since there's no clipping on t_STAR, we can apply the lognormal to SFR directly instead of t_STAR
    sfr_sample = sfr_mean * exp(-sigma_sfr*sigma_sfr/2 + input[1]*sigma_sfr);
    wsfr_sample = sfr_sample * (fesc*f_sample + fesc_mini*f_sample_mini) / (f_sample + f_sample_mini); //each component has its own f_esc, this is strange but consistent with default 

    //LOG_ULTRA_DEBUG("HM %.3e | SM %.3e | SFR %.3e (%.3e) | F* %.3e (%.3e) | duty %.3e",halo_mass,sm_sample,sfr_sample,sfr_mean,f_sample,fstar_mean,dutycycle_term);

    output[0] = sm_sample;
    output[1] = sfr_sample;
    output[2] = n_ion_sample;
    output[3] = wsfr_sample;

    return 0;
}

//This function adds stochastic halo properties to an existing halo
//TODO: this needs to be updated to handle MINI_HALOS and not USE_MASS_DEPENDENT_ZETA flags
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
        LOG_ULTRA_DEBUG("halo %d hm %.2e crd %d %d %d",i,halos->halo_masses[i],halos->halo_coords[3*i+0],halos->halo_coords[3*i+1],halos->halo_coords[3*i+2]);
        set_prop_rng(rng_stoc[omp_get_thread_num()], 0, inbuf, inbuf, buf);
        LOG_ULTRA_DEBUG("stars %.2e sfr %.2e",buf[0],buf[1]);
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
        nh = 0;
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
    if(hs_constants->update)exp_M *= 1.; //~0.95 fudge factor for assuming that internal lagrangian volumes are independent

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
        M_sample = EvaluateSigmaInverse(sigma_sample);
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
    //NOTE: some of these conditions are redundant with set_consts_cond()
    if(hs_constants->delta <= DELTA_MIN || hs_constants->expected_M < hs_constants->M_min*global_params.HALO_SAMPLE_FACTOR){
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
        float prop_buf[2], prop_dummy[2];
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
            local_sm[count] = halofield_large->star_rng[j];
            local_sfr[count] = halofield_large->sfr_rng[j];
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

                    // LOG_ULTRA_DEBUG("cell (%d,%d,%d) %.2e Start",x,y,z,delta);
                    stoc_set_consts_cond(&hs_constants_priv,delta);
                    // LOG_ULTRA_DEBUG("exp %.2f halos, %.2e Mass",hs_constants_priv.expected_N,hs_constants_priv.expected_M);
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

                        set_prop_rng(rng_arr[threadnum], 0, prop_dummy, prop_dummy, prop_buf);
                        place_on_hires_grid(x,y,z,crd_hi,rng_arr[threadnum]);

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
                    // LOG_ULTRA_DEBUG("Done, %d halos, %.2e Mass",nh_buf,M_cell);
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
        memcpy(halofield_out->star_rng + istart,local_sm,count*sizeof(float));
        memcpy(halofield_out->sfr_rng + istart,local_sfr,count*sizeof(float));
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
            
            propbuf_in[0] = halofield_in->star_rng[ii];
            propbuf_in[1] = halofield_in->sfr_rng[ii];

            //place progenitors in local list
            M_prog = 0;
            for(jj=0;jj<n_prog;jj++){
                if(prog_buf[jj] < Mmin*global_params.HALO_SAMPLE_FACTOR) continue; //save only halos some factor above minimum
                //LOG_ULTRA_DEBUG("updating props");
                set_prop_rng(rng_arr[threadnum], 1, corr_arr, propbuf_in, propbuf_out);
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
        memcpy(halofield_out->star_rng + istart,local_sm,count*sizeof(float));
        memcpy(halofield_out->sfr_rng + istart,local_sfr,count*sizeof(float));
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
        LOG_DEBUG("First few Stellar: %11.3e %11.3e %11.3e",halos->star_rng[0],halos->star_rng[1],halos->star_rng[2]);
        LOG_DEBUG("First few SFR:     %11.3e %11.3e %11.3e",halos->sfr_rng[0],halos->sfr_rng[1],halos->sfr_rng[2]);
    }
    
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
    }
    free_dNdM_tables();

    free_rng_threads(rng_stoc);
    return 0;
}

//Fixed halo grids, where each property is set as the integral of the CMF on the EULERIAN cell scale
//As per default 21cmfast (strange pretending that the lagrangian density is eulerian and then *(1+delta))
//This outputs the UN-NORMALISED grids (before mean-adjustment)
//TODO: add minihalos
//TODO: use the interpolation tables (Fixed grids are currently slow but a debug case)
int set_fixed_grids(double redshift, double norm_esc, double alpha_esc, struct PerturbedField * perturbed_field, struct TsBox *previous_spin_temp, struct IonizedBox *previous_ionize_box, struct HaloBox *grids, double *averages){
    //There's quite a bit of re-calculation here but this only happens once per snapshot
    double M_min = minimum_source_mass(redshift,astro_params_stoc,flag_options_stoc)*global_params.HALO_SAMPLE_FACTOR;
    double volume = VOLUME / HII_TOT_NUM_PIXELS;
    double M_max = RtoM(user_params_stoc->BOX_LEN / user_params_stoc->HII_DIM * L_FACTOR); //mass in cell of mean dens
    double sigma_max = sigma_z0(M_max);
    double growth_z = dicke(redshift);
    double alpha_star = astro_params_stoc->ALPHA_STAR;
    double norm_star = astro_params_stoc->F_STAR10;
    double t_star = astro_params_stoc->t_STAR;

    double lnMmin = log(M_min);
    double lnMmax = log(M_max);
    
    double prefactor_mass = RHOcrit * cosmo_params_stoc->OMm / sqrt(2.*PI);
    double prefactor_nion = global_params.Pop2_ion * RHOcrit * cosmo_params_stoc->OMb * norm_star * norm_esc;
    double prefactor_sfr = RHOcrit * cosmo_params_stoc->OMb * norm_star / t_star / t_hubble(redshift);

    double Mlim_Fstar = Mass_limit_bisection(M_min, M_max, alpha_star, norm_star);
    double Mlim_Fesc = Mass_limit_bisection(M_min, M_max, alpha_esc, norm_esc);
    
    double hm_avg, nion_avg, sfr_avg, wsfr_avg;
    double Mlim_a_avg=0, Mlim_m_avg=0;
    
    double M_turn_m,M_turn_a,M_turn_r;
    M_turn_m = 0.;
    M_turn_r = 0.;
    if(flag_options_stoc->USE_MINI_HALOS)
        M_turn_a = atomic_cooling_threshold(redshift);
    else
        M_turn_a = astro_params_stoc->M_TURN;

    LOG_DEBUG("Mean halo boxes || Mmin = %.2e | Mmax = %.2e (s=%.2e) | z = %.2e | D = %.2e | cellvol = %.2e",M_min,M_max,sigma_max,redshift,growth_z,volume);
#pragma omp parallel num_threads(user_params_stoc->N_THREADS) firstprivate(M_turn_m,M_turn_a)
    {
        int i;
        double dens;
        double mass, nion, sfr, h_count;
        double wsfr;
#pragma omp for reduction(+:hm_avg,nion_avg,sfr_avg,wsfr_avg)
        for(i=0;i<HII_TOT_NUM_PIXELS;i++){
            dens = perturbed_field->density[i];
            
            //TODO: include VELOCITIES
            if(flag_options_stoc->USE_MINI_HALOS){
                M_turn_m = lyman_werner_threshold(redshift, previous_spin_temp->J_21_LW_box[i], 0.,astro_params_stoc);
                M_turn_r = reionization_feedback(redshift, previous_ionize_box->Gamma12_box[i], previous_ionize_box->z_re_box[i]);
            }
            if(M_turn_r > M_turn_a) M_turn_a = M_turn_r;
            if(M_turn_r > M_turn_m) M_turn_m = M_turn_r;

            //ignore very low density NOTE:not using DELTA_MIN since it's perturbed (Eulerian)
            if(dens <= -1){
                mass = 0.;
                nion = 0.;
                sfr = 0.;
                h_count = 0;
            }
            //turn into one large halo if we exceed the critical
            //Since these are perturbed (Eulerian) grids, I use the total cell mass (1+dens)
            else if(dens>=MAX_DELTAC_FRAC*Deltac){
                mass = M_max * (1+dens) / volume;
                nion = global_params.Pop2_ion * M_max * (1+dens) * cosmo_params_stoc->OMb / cosmo_params_stoc->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) * norm_esc * pow(M_max*(1+dens)/1e10,alpha_esc) / volume;
                sfr = M_max * (1+dens) * cosmo_params_stoc->OMb / cosmo_params_stoc->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) / t_star / t_hubble(redshift) / volume;
                h_count = 1;
            }
            else{
                //calling IntegratedNdM with star and SFR need special care for the f*/fesc clipping, and calling NionConditionalM for mass includes duty cycle
                //neither of which I want
                mass = IntegratedNdM(growth_z,lnMmin,lnMmax,lnMmax,dens,1,user_params_stoc->HMF,1);
                h_count = IntegratedNdM(growth_z,lnMmin,lnMmax,lnMmax,dens,0,user_params_stoc->HMF,1);

                nion = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_max, Deltac, dens, M_turn_a
                                        , astro_params_stoc->ALPHA_STAR, alpha_esc, astro_params_stoc->F_STAR10, norm_esc
                                        , Mlim_Fstar, Mlim_Fesc, user_params_stoc->FAST_FCOLL_TABLES);

                sfr = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_max, Deltac, dens, M_turn_a
                                        , astro_params_stoc->ALPHA_STAR, 0., astro_params_stoc->F_STAR10, 1., Mlim_Fstar, 0.
                                        , user_params_stoc->FAST_FCOLL_TABLES);

                //Same integral as Nion
                // wsfr = Nion_ConditionalM(growth_z, lnMmin, lnMmax, sigma_max, Deltac, dens, M_turn_a
                //                         , astro_params_stoc->ALPHA_STAR, alpha_esc, astro_params_stoc->F_STAR10, norm_esc, Mlim_Fstar, Mlim_Fesc
                //                         , user_params_stoc->FAST_FCOLL_TABLES);
            }
            grids->halo_mass[i] = mass * prefactor_mass * (1+dens);
            grids->n_ion[i] = nion * prefactor_nion * (1+dens);
            grids->halo_sfr[i] = sfr * prefactor_sfr * (1+dens);
            grids->whalo_sfr[i] = nion * prefactor_sfr * (1+dens) * norm_esc;
            grids->count[i] = (int)(h_count * prefactor_mass * (1+dens)); //truncated
            
            hm_avg += mass;
            nion_avg += nion;
            sfr_avg += sfr;
            wsfr_avg += wsfr;
            Mlim_a_avg += M_turn_a;
            Mlim_m_avg += M_turn_m;
        }
    }
    
    hm_avg /= HII_TOT_NUM_PIXELS;
    nion_avg /= HII_TOT_NUM_PIXELS;
    sfr_avg /= HII_TOT_NUM_PIXELS;
    wsfr_avg /= HII_TOT_NUM_PIXELS;
    Mlim_a_avg /= HII_TOT_NUM_PIXELS;
    Mlim_m_avg /= HII_TOT_NUM_PIXELS;

    averages[0] = hm_avg;
    averages[1] = nion_avg;
    averages[2] = sfr_avg;
    averages[3] = wsfr_avg;
    averages[4] = Mlim_a_avg;
    averages[5] = Mlim_m_avg;

    return 0;
}

//Expected global averages for box quantities for mean adjustment
//TODO: use the global interpolation tables (only one integral per property per snapshot so this is low priority)
//WARNING: THESE AVERAGE BOXES ARE WRONG, CHECK THEM
int get_box_averages(double redshift, double norm_esc, double alpha_esc, double M_turn_a, double M_turn_m, double *averages){
    double alpha_star = astro_params_stoc->ALPHA_STAR;
    double norm_star = astro_params_stoc->F_STAR10;
    double t_star = astro_params_stoc->t_STAR;
    //There's quite a bit of re-calculation here but this only happens once per snapshot
    double M_min = minimum_source_mass(redshift,astro_params_stoc,flag_options_stoc) * global_params.HALO_SAMPLE_FACTOR;
    double M_max = global_params.M_MAX_INTEGRAL; //mass in cell of mean dens
    double lnMmax = log(M_max);
    double lnMmin = log(M_min);
    double growth_z = dicke(redshift);

    // double prefactor_mass = RHOcrit * cosmo_params_stoc->OMm / sqrt(2.*PI);
    double prefactor_nion = global_params.Pop2_ion * RHOcrit * cosmo_params_stoc->OMb * norm_star * norm_esc;
    double prefactor_sfr = RHOcrit * cosmo_params_stoc->OMb * norm_star / t_star / t_hubble(redshift);

    double hm_expected,nion_expected,sfr_expected,wsfr_expected;

    double Mlim_Fstar = Mass_limit_bisection(M_min, M_max, alpha_star, norm_star);
    double Mlim_Fesc = Mass_limit_bisection(M_min, M_max, alpha_esc, norm_esc);

    hm_expected = IntegratedNdM(growth_z, lnMmin, lnMmax, lnMmax, 0, 1, user_params_stoc->HMF,0);
    nion_expected = Nion_General(redshift, M_min, M_turn_a, alpha_star, alpha_esc, norm_star, norm_esc, Mlim_Fstar, Mlim_Fesc);
    sfr_expected = Nion_General(redshift, M_min, M_turn_a, alpha_star, 0., norm_star, 1., Mlim_Fstar, 0.);
    // wsfr_expected = Nion_General(redshift, M_min, M_turn_a, alpha_star, alpha_esc, norm_star, norm_esc, Mlim_Fstar, Mlim_Fesc);

    // hm_expected *= prefactor_mass; //for non-CMF, the factors are already there
    wsfr_expected = nion_expected * prefactor_sfr * norm_esc; //same integral, different prefactors, different in the stochastic grids due to scatter
    nion_expected *= prefactor_nion;
    sfr_expected *= prefactor_sfr;

    averages[0] = hm_expected;
    averages[1] = nion_expected;
    averages[2] = sfr_expected;
    averages[3] = wsfr_expected;

    return 0;
}

//This, for the moment, grids the PERTURBED halo catalogue.
//TODO: make a way to output both types by making 2 wrappers to this function that pass in arrays rather than structs
//NOTE: this function is quite slow to generate fixed halo boxes, however I don't mind since it's a debug case
//  If we want to make it faster just replace the integrals with the existing interpolation tables
//TODO: I should also probably completely separate the fixed and sampled grids into two functions which this calls
int ComputeHaloBox(double redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
                    struct FlagOptions * flag_options, struct PerturbedField * perturbed_field, struct PerturbHaloField *halos,
                    struct TsBox *previous_spin_temp, struct IonizedBox *previous_ionize_box, struct HaloBox *grids){
    int status;
    Try{
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
        double hm_avg=0,nion_avg=0,sfr_avg=0,wsfr_avg=0;

        double M_min;
        double t_h = t_hubble(redshift);
        double volume = VOLUME/HII_TOT_NUM_PIXELS;
        double M_turn_a_avg = 0, M_turn_m_avg = 0;
        
        double M_turn_m,M_turn_a,M_turn_r;
        M_turn_r = 0.;
        if(flag_options_stoc->USE_MINI_HALOS){
            M_turn_a = atomic_cooling_threshold(redshift);
            M_turn_m = lyman_werner_threshold(redshift, 0., 0.,astro_params_stoc); //This only does something when there are no halos and we are printing global averages
        }
        else{
            M_turn_a = astro_params->M_TURN;
            M_turn_m = 0.;
        }

        LOG_DEBUG("atomic cooling threshold %11.3e",M_turn_a);
        
        double averages_box[6], averages_global[4];

        M_min = minimum_source_mass(redshift,astro_params,flag_options);

        //calculate expected average halo box, for mean halo box fixing and debugging
        if(flag_options->FIXED_HALO_GRIDS || LOG_LEVEL >= DEBUG_LEVEL){
            init_ps();
            if(user_params->USE_INTERPOLATION_TABLES){
                initialiseSigmaMInterpTable(M_min/2., global_params.M_MAX_INTEGRAL); //this needs to be initialised above MMax because of Nion_General
            }
        }
        //do the mean HMF box
        //The default 21cmFAST has a strange behaviour where the nonlinear density is used as linear,
        //the condition mass is at mean density, but the total cell mass is multiplied by delta 
        //This part mimics that behaviour
        //Since we need the average turnover masses before we can calculate the global means, we do the CMF integrals first
        //Then we calculate the expected UMF integrals before doing the adjustment
        if(flag_options->FIXED_HALO_GRIDS){
            set_fixed_grids(redshift, norm_esc, alpha_esc, perturbed_field, previous_spin_temp, previous_ionize_box, grids, averages_box);
            M_turn_a_avg = averages_box[4];
            M_turn_m_avg = averages_box[5];
            get_box_averages(redshift, norm_esc, alpha_esc, M_turn_a_avg, M_turn_m_avg, averages_global);
            //This is the mean adjustment that happens in the rest of the code
            int i;
            
#pragma omp parallel for num_threads(user_params->N_THREADS)
            for(i=0;i<HII_TOT_NUM_PIXELS;i++){
                grids->halo_mass[i] *= averages_global[0]/averages_box[0];
                grids->n_ion[i] *= averages_global[1]/averages_box[1];
                grids->halo_sfr[i] *= averages_global[3]/averages_box[2];
                grids->whalo_sfr[i] *= averages_global[3]/averages_box[3];
            }
            
            hm_avg = averages_global[0];
            nion_avg = averages_global[1];
            sfr_avg = averages_global[2];
            wsfr_avg = averages_global[3];
        }
        else{
#pragma omp parallel num_threads(user_params->N_THREADS) firstprivate(M_turn_a,M_turn_m,M_turn_r)
            {
                int i_halo,idx,x,y,z;
                double m,nion,sfr,wsfr,stars;
                
                float in_props[2];
                float out_props[4];
                //Check if this initialisation is necessary. aren't they already zero'd in Python?
#pragma omp for
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] = 0.0;
                    grids->n_ion[idx] = 0.0;
                    grids->halo_sfr[idx] = 0.0;
                    grids->whalo_sfr[idx] = 0.0;
                    grids->count[idx] = 0;
                }

#pragma omp barrier

#pragma omp for reduction(+:hm_avg,nion_avg,sfr_avg,wsfr_avg,M_turn_a_avg,M_turn_m_avg)
                for(i_halo=0; i_halo<halos->n_halos; i_halo++){
                    x = halos->halo_coords[0+3*i_halo]; //NOTE:PerturbedHaloField is on HII_DIM, HaloField is on DIM
                    y = halos->halo_coords[1+3*i_halo];
                    z = halos->halo_coords[2+3*i_halo];

                    if(flag_options->USE_MINI_HALOS){
                        M_turn_m = lyman_werner_threshold(redshift, previous_spin_temp->J_21_LW_box[HII_R_INDEX(x,y,z)], 0.,astro_params);
                        M_turn_r = reionization_feedback(redshift, previous_ionize_box->Gamma12_box[HII_R_INDEX(x, y, z)], previous_ionize_box->z_re_box[HII_R_INDEX(x, y, z)]);
                    }
                    
                    if(M_turn_r > M_turn_a) M_turn_a = M_turn_r;
                    if(M_turn_r > M_turn_m) M_turn_m = M_turn_r;

                    m = halos->halo_masses[i_halo];

                    //these are the halo property RNG sequences
                    in_props[0] = halos->star_rng[i_halo];
                    in_props[1] = halos->sfr_rng[i_halo];

                    set_halo_properties(m,M_turn_a,M_turn_m,t_h,norm_esc,alpha_esc,in_props,out_props);

                    stars = out_props[0]; //not needed until I feed back
                    sfr = out_props[1];
                    nion = out_props[2];
                    wsfr = out_props[3];
                    
                    // if(i_halo < 10){
                    //     LOG_DEBUG("%d: HM: %.2e SM: %.2e NI: %.2e SF: %.2e WS: %.2e",i_halo,m,stars,nion,sfr,wsfr);
                    // }

                    //feed back the calculated properties to PerturbHaloField
                    //TODO: move set_halo_properties to PertburbHaloField and move it forward in time
                    //  This will require EITHER separating mini and regular halo components OR the ternary halo model (inactive,moleculer,atomic)
                    //  OR directly storing all the grid components
                    // halos->stellar_masses[i_halo] = stars;
                    // halos->halo_sfr[i_halo] = sfr;

#pragma omp atomic update
                    grids->halo_mass[HII_R_INDEX(x, y, z)] += m;
#pragma omp atomic update
                    grids->n_ion[HII_R_INDEX(x, y, z)] += nion;
#pragma omp atomic update
                    grids->halo_sfr[HII_R_INDEX(x, y, z)] += sfr;
#pragma omp atomic update
                    grids->whalo_sfr[HII_R_INDEX(x, y, z)] += wsfr;
                    //It can be convenient to remove halos from a catalogue by setting them to zero, don't count those here
                    if(m>0){
#pragma omp atomic update
                        grids->count[HII_R_INDEX(x, y, z)] += 1;
                    }

                    if(LOG_LEVEL >= DEBUG_LEVEL){
                        hm_avg += m;
                        sfr_avg += sfr;
                        wsfr_avg += wsfr;
                        nion_avg += nion;
                        M_turn_a_avg += M_turn_a;
                        M_turn_m_avg += M_turn_m;
                    }
                }
                //mark that we've set the properties in the catalogue
                // halos->properties_set = true;
#pragma omp for
                for (idx=0; idx<HII_TOT_NUM_PIXELS; idx++) {
                    grids->halo_mass[idx] /= volume;
                    grids->n_ion[idx] /= volume;
                    grids->halo_sfr[idx] /= volume;
                    grids->whalo_sfr[idx] /= volume;
                }
            }
            if(LOG_LEVEL >= DEBUG_LEVEL){
                hm_avg /= volume*HII_TOT_NUM_PIXELS;
                sfr_avg /= volume*HII_TOT_NUM_PIXELS;
                wsfr_avg /= volume*HII_TOT_NUM_PIXELS;
                nion_avg /= volume*HII_TOT_NUM_PIXELS;
                //NOTE: There is an inconsistency here, the sampled grids use a halo-averaged turnover mass
                //  whereas the fixed grids / default 21cmfast uses a volume averaged one.
                //  Neither of these are a perfect representation due to the nonlinear way turnover mass affects N_ion
                //  However they should be consistent
                //  TODO: make another loop to calculate the average Mturn (without slowing down hopefully)
                //If there are no halos we want to use the initial global values of M_turn
                if(halos->n_halos != 0){
                    M_turn_a_avg /= halos->n_halos;
                    M_turn_m_avg /= halos->n_halos;
                }
                get_box_averages(redshift, norm_esc, alpha_esc, M_turn_a_avg, M_turn_m_avg, averages_global);
            }
        }

        if(user_params->USE_INTERPOLATION_TABLES && (flag_options->FIXED_HALO_GRIDS || LOG_LEVEL >= DEBUG_LEVEL)){
                freeSigmaMInterpTable();
        }

        LOG_DEBUG("HaloBox Cells:  %9.2e %9.2e %9.2e %9.2e", grids->halo_mass[0], grids->halo_mass[1]
            , grids->halo_mass[2] , grids->halo_mass[3]);
        LOG_DEBUG("Ion Emissivity: %9.2e %9.2e %9.2e %9.2e", grids->n_ion[0], grids->n_ion[1]
            , grids->n_ion[2] , grids->n_ion[3]);
        LOG_DEBUG("Halo SFR:       %9.2e %9.2e %9.2e %9.2e", grids->halo_sfr[0], grids->halo_sfr[1]
            , grids->halo_sfr[2] , grids->halo_sfr[3]);
            
        LOG_DEBUG("Redshift %.2f: Exp. averages: (%11.3e,%11.3e,%11.3e,%11.3e) || Box averages (%11.3e,%11.3e,%11.3e,%11.3e)"
                    ,redshift,averages_global[0],averages_global[1],averages_global[2],averages_global[3],hm_avg,nion_avg,sfr_avg,wsfr_avg);
        LOG_DEBUG("Ratio (%11.3e,%11.3e,%11.3e,%11.3e) Mta %11.3e Mtm %11.3e",averages_global[0]/hm_avg,averages_global[1]/nion_avg,
                    averages_global[2]/sfr_avg,averages_global[3]/wsfr_avg,M_turn_a_avg,M_turn_m_avg);
    
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
        double test,dummy;
        int err=0;
        int i,j;

        struct HaloSamplingConstants hs_const_struct;
        struct HaloSamplingConstants *hs_constants = &hs_const_struct;
        LOG_DEBUG("Setting z constants. %.3f %.3f",z_out,z_in);
        stoc_set_consts_z(hs_constants,z_out,z_in);
        print_hs_consts(hs_constants);
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
                if((M[i] < Mmin) || M[i] > MMAX_TABLES || (!cmf_flag && M[i] > Mcond)){
                    test = 0.;
                }
                else{
                    test = MnMassfunction(log(M[i]),(void*)&parameters_gsl_MF_con);
                    
                    //convert to dndlnm
                    test = test * prefactor;
                }
                LOG_ULTRA_DEBUG(" D %.1e | M1 %.1e | M2 %.1e | d %.1e | s %.1e -> %.1e",
                                growth_out,M[i],Mcond,delta,hs_constants->sigma_cond,test);
                result[i] = test;
            }
        }
        else if(type==1){
            //integrate CMF -> N(Ml<M<Mh) in one condition
            //TODO: make it possible to integrate UMFs
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
                        n_halo_out++;
                        
                        //only save halos above the save limit, but add all halos to the totals
                        if(out_hm[i]<Mmin*global_params.HALO_SAMPLE_FACTOR){
                            continue;
                        }

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
            print_hs_consts(hs_constants);
            #pragma omp parallel for private(test,y_in)
            for(i=0;i<n_mass;i++){
                y_in = log(M[i]);
                LOG_ULTRA_DEBUG("dNdM table: x = %.6e, y = %.6e z = %.6e",x_in,y_in);
                if(y_in < lnMmin){
                    result[i] = 0;
                    continue;
                }
                if(y_in > hs_constants->lnM_max_tb){
                    y_in = hs_constants->lnM_max_tb*0.999;
                }
                test = EvaluateRGTable2D(x_in,y_in,Nhalo_spline,hs_constants->tbl_xmin,hs_constants->tbl_xwid,hs_constants->tbl_ymin,hs_constants->tbl_ywid);
                result[i] = test * mass / sqrt(2.*PI);
                LOG_ULTRA_DEBUG("==> z = %.6e",test);
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
                if(y_in >= 0){
                    result[i] = 0.;
                    continue;
                }
                else if(y_in <= global_params.MIN_LOGPROB){
                    result[i] = hs_constants->lnM_cond;
                }
                test = EvaluateRGTable2D(x_in,y_in,Nhalo_inv_spline,hs_constants->tbl_xmin,hs_constants->tbl_xwid,hs_constants->tbl_pmin,hs_constants->tbl_pwid);
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