/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//max guesses for rejection sampling
#define MAX_ITERATIONS 1e5
//max halos in memory for test functions
#define MAX_HALO (int)1e8

struct AstroParams *astro_params_stoc;
struct CosmoParams *cosmo_params_stoc;
struct UserParams *user_params_stoc;
struct FlagOptions *flag_options_stoc;

gsl_rng * rng_stoc; //global rng
//gsl_rng * rng_stoc[user_params->N_THREADS];

struct parameters_gsl_MF_con_int_{
    double growthf;
    double delta;
    double n_order;
    double sigma_max;
    double M_max;
};

void Broadcast_struct_global_STOC(struct UserParams *user_params, struct CosmoParams *cosmo_params,struct AstroParams *astro_params, struct FlagOption *flag_options){
    cosmo_params_stoc = cosmo_params;
    user_params_stoc = user_params;
    astro_params_stoc = astro_params;
    flag_options_stoc = flag_options;
}

//n_order is here because the variance calc can use these functions too
//remove the variable (n==0) if we remove that calculation
double MnMassfunction(double M, void *param_struct){
    struct parameters_gsl_MF_con_int_ params = *(struct parameters_gsl_MF_con_int_ *)param_struct;
    double mf;
    double growthf = params.growthf;
    double delta = params.delta;
    double n_order = params.n_order;
    double sigma = params.sigma_max; //M2 and sigma2 are degenerate, remove one
    double M2 = params.M_max;

    if (M2 < M) return 0.;

    //M1 is the mass of interest, M2 doesn't seem to be used (input as max mass),
    // delta1 is critical, delta2 is current, sigma is sigma(Mmax,z=0)
    //dNdlnM = dfcoll/dM * M / M * constants
    mf = dNdM_conditional(growthf,M,M2,Deltac,delta,sigma);

    //norder for expectation values of M^n
    return exp(M * (n_order)) * mf;
}

//cdf of sub-cell structure, used for sampling
double FcollConditional(double delta1, double delta2, double M1, double M2){
    double sigma1,sigma2,arg,result;
    
    sigma1 = sigma_z0(M1);
    sigma2 = sigma_z0(sigma2);

    arg = -(delta1-delta2)*(delta1-delta2)/sqrt(2*(sigma1*sigma1 - sigma2*sigma2));

    result = erfcc(arg);

    return result;
}

//copied mostly from the Nion functions
//I might be missing something like this that already exists somewhere in the code
double IntegratedNdM(double growthf, double M1, double M2, double delta, double n_order){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.01; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    double sigma2 = sigma_z0(exp(M2));

    struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
        .growthf = growthf,
        .delta = delta,
        .n_order = n_order,
        .sigma_max = sigma2,
        .M_max = M2,
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
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: growthf=%e M2=%e delta=%e,sigma2=%e",growthf,M2,delta,sigma2);
        LOG_ERROR("data: growthf=%e M2=%e delta=%e,sigma2=%e",parameters_gsl_MF_con.growthf,parameters_gsl_MF_con.M_max,parameters_gsl_MF_con.delta,parameters_gsl_MF_con.sigma_max);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    //what is this? if the region density is greater than critical then set the result, AFTER the integration?
    //ask someone about this
    if(delta > Deltac) {
        result = 1.;
        return result;
    }
    else {
        //turn dfcoll to dNdlnm
        result = result * (RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm);
        return result;
        }
}

// Calculate the Nth moment of the lognormal distribution, integrated over the halo mass distribution
// TODO: add option for poisson process perturbing the zeroth moment for the sampling method
double EvaluatedNdMSpline(double growthf, double M1, double M2, double delta, double n_order){
    //for testing, I'm assuming fixed (mean) number of halos per cell and P(M_halo) = diracdelta(M)
    //TODO: replace with the actual calculation here integrating/interpolating over dNdM and adding noise if needed
    double buf;

    //the interpolation will have an extra dimension compared to the N_ion one for the nth moment (minus R if we build the grids)
    //
    LOG_WARNING("dNdM interpolation table not implemented, integrating...");
    return IntegratedNdM(growthf, log(M1), log(M2), delta, n_order);
}

//ERFC inverse approximation taken from https://stackoverflow.com/questions/27229371/inverse-error-function-in-c/40260471
//based on approach from M.Giles https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
//erfc-1(1-x) = erf-1(x), the approach on stackoverflow uses fmaf(xyz), which may be faster depending on hardware
float erf_inverse(float y){
    float p, x, t;
    t = y * (0.0f - y) + 1.0f;
    t = log(t);
    if (fabsf(t) > 6.125f) { // maximum ulp error = 2.35793
        p =          3.03697567e-10f; //  0x1.4deb44p-32 
        p = (p * t + 2.93243101e-8f); //  0x1.f7c9aep-26 
        p = (p * t + 1.22150334e-6f); //  0x1.47e512p-20 
        p = (p * t + 2.84108955e-5f); //  0x1.dca7dep-16 
        p = (p * t + 3.93552968e-4f); //  0x1.9cab92p-12 
        p = (p * t + 3.02698812e-3f); //  0x1.8cc0dep-9 
        p = (p * t + 4.83185798e-3f); //  0x1.3ca920p-8 
        p = (p * t - 2.64646143e-1f); // -0x1.0eff66p-2 
        p = (p * t + 8.40016484e-1f); //  0x1.ae16a4p-1 
    } else { // maximum ulp error = 2.35002
        p =          5.43877832e-9f;  //  0x1.75c000p-28 
        p = (p * t + 1.43285448e-7f); //  0x1.33b402p-23 
        p = (p * t + 1.22774793e-6f); //  0x1.499232p-20 
        p = (p * t + 1.12963626e-7f); //  0x1.e52cd2p-24 
        p = (p * t - 5.61530760e-5f); // -0x1.d70bd0p-15 
        p = (p * t - 1.47697632e-4f); // -0x1.35be90p-13 
        p = (p * t + 2.31468678e-3f); //  0x1.2f6400p-9 
        p = (p * t + 1.15392581e-2f); //  0x1.7a1e50p-7 
        p = (p * t - 2.32015476e-1f); // -0x1.db2aeep-3 
        p = (p * t + 8.86226892e-1f); //  0x1.c5bf88p-1 
    }
    x = y * p;
    return x;
}

//sigma_z0 but used in the root finder to find M from sigma
double sigma_z0_rf(double M, void *sigma){
    int MassBin;
    float MassBinLow;
    double val;

    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
        MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        val = Sigma_InterpTable[MassBin] + ( log(M) - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    }
    else{
        val = sigma_z0(exp(M));
    }
    return val - *(double*)sigma;
}

//single sample of the HMF from inverse CDF method WIP
int sample_dndM_inverse(double growthf, double delta, double Mmin, double Mmax, double ymin, double ymax, double sigma2, double *result){
    double y1, fcoll_x, sigma_M;

    y1 = gsl_rng_uniform(rng_stoc);
    //erf vs erfc doesn't matter here because 1-U(x) == U(x) for uniform (0,1)
    //also the CDF f(<M) = f(>M) - 1
    fcoll_x = (double)erf_inverse(y1);

    sigma_M = sqrt(1/2.*(Deltac-delta)*(Deltac-delta)/fcoll_x/fcoll_x/growthf/growthf - sigma2*sigma2);

    //find M from sigma(M) using gsl root finder
    //dummy params for the root finder
    //TODO: consider using the "polishing" algorithms which require dF, but are faster.

    gsl_root_fsolver_type *T = gsl_root_fsolver_bisection;
    gsl_root_fsolver *s = gsl_root_fsolver_alloc(T);
    if (!s){
        LOG_ERROR("Unable to allocate memory.");
        Throw(MemoryAllocError);
    }
    double x_lo = Mmin;
    double x_hi = Mmax;
    double r;
    int status;

    gsl_function F;
    F.function = &sigma_z0_rf;
    F.params = &sigma_M;

    gsl_root_fsolver_set(s,&F,x_lo,x_hi);

    int iter = 0;
    do
    {
      iter++;
      status = gsl_root_fsolver_iterate(s);
      r = gsl_root_fsolver_root(s);
      x_lo = gsl_root_fsolver_x_lower(s);
      x_hi = gsl_root_fsolver_x_upper(s);
      status = gsl_root_test_interval(x_lo, x_hi, 0, 0.001);
    }
    while (status == GSL_CONTINUE && iter < MAX_ITERATIONS);

    if(status != GSL_SUCCESS){
        LOG_ERROR("root finder failed to converge %d",status);
        return 1;
    }

    LOG_DEBUG("root M=%.2e (s = %.2e) iter %d | lo=%.2e hi=%.2e",exp(r),sigma_M,iter,x_lo,x_hi);
    LOG_DEBUG("min=%.2e max=%.2e fcoll_x = %.2e, sample y = %.2e, s2 = %.2e",exp(Mmin),exp(Mmax),fcoll_x,y1,sigma2);

    gsl_root_fsolver_free(s);

    if(!isfinite(r)){
        LOG_ERROR("Value for nu_tau_one_MINI is infinite or NAN");
        Throw(InfinityorNaNError);
    }

    *result = r;

    return 0;
}

//single sample of the halo mass function using rejection method
int sample_dndM_rejection(double growthf, double delta, double Mmin, double Mmax, double ymin, double ymax, double sigma, double *result){
    double x1,y1, MassFunction;
    int c=0;

    //rejection sampling for now
    //inverse CDF sampling is probably better but requires an extra root find that isn't there yet
    while(c < MAX_ITERATIONS){
        //uniform sampling of logM between given limits
        x1 = Mmin + (Mmax-Mmin)*gsl_rng_uniform(rng_stoc);
        y1 = ymin + (ymax-ymin)*gsl_rng_uniform(rng_stoc);

        //for halo abundances (dNdlogM) from dfcoll/dM, *M for dlogm and / M for dfcolldN
        MassFunction = dNdM_conditional(growthf,x1,Mmax,Deltac,delta,sigma);
        //MassFunction = MassFunction * (RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm);
#if 0
        if(c<100){
            LOG_DEBUG("%d iter sampled mass %.3e with y=%.3e under dNdlogm %.3e",c,x1,y1,MassFunction);
        }
        else{
            return 1;
        }
#endif
        if(y1<MassFunction){
            *result = x1;
            //LOG_DEBUG("%d iter sampled mass %.3e with y=%.3e under dNdlogm %.3e from box((%.3e,%.3e,%.3e,%.3e)",c,x1,MassFunction,Mmin,Mmax,ymin,ymax);
            return 0;
        }
        
        c++;
    }
    LOG_ERROR("passed max iterations for rejection sampling, box(x1,x2,y1,y2) (%.3e,%.3e,%.3e,%.3e)",Mmin,Mmax,ymin,ymax);
    return 1;
}

/* Calculates the stochasticity of halo properties by sampling the halo mass function and 
 * conditional property PDFs, summing the resulting halo properties within a cell */
int stoc_halo_sample(double z, double delta, double volume, double M1, double M2, int sampler, int outtype, int *n_halo_out, double *hm_out, double *sm_out){
    int n_halo,err;
    double nh_buf,mu_lognorm;

    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;

    double hm_sample, sm_sample, sm_mean;
    double hm_cell=0, sm_cell=0.;

    double sigma_max = sigma_z0(M2);
    double growthf = dicke(z);

    // TODO: put the integral in bins:
    // Integrate M1->M2 (need sigma arg since M2 != MMax)
    // poisson noise in each bin
    // sample M in each bin (test speed & accuracy of uniform in bin & function)
    // apply scatters of sfr etc

    //get average number of halos in cell n_order=0
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        nh_buf = EvaluatedNdMSpline(growthf, M1, M2, delta, 0);
    }
    else{
        nh_buf = IntegratedNdM(growthf, log(M1), log(M2), delta, 0);
    }
    nh_buf *= volume;

    //LOG_ERROR("nhalo = %.3e",nh_buf);

    //sample poisson for stochastic halo number
    n_halo = gsl_ran_poisson(rng_stoc,nh_buf);

    /* to save memory we calulate properties and sample within the loop
     * it may be faster to chunk it up or generate a catalogue (MANY GB)
     * before summing */

    //TODO: set up mass distribution for sampling here
    //BEWARE: assuming max(dNdM) == dNdM(Mmin)
    double ymin,ymax;
    ymin = 0;
    ymax = dNdM_conditional(growthf,log(M1),log(M2),Deltac,delta,sigma_max);
    //ymax = ymax * (RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm);

    //LOG_ERROR("n halo = %d",n_halo);

    int ii;
    for(ii=0;ii<n_halo;ii++){
        //sample from the cHMF
        /* TODO: figure out the best method for this
         * either build an x = iCDF(delta,M) table and interpolate
         * or use rejection sampling on the existing mass function table */
        if (sampler==1){
            err = sample_dndM_rejection(growthf,delta,log(M1),log(M2),ymin,ymax,sigma_max,&hm_sample);
        
        }
        else if(sampler==2){
            err = sample_dndM_inverse(growthf,delta,log(M1),log(M2),ymin,ymax,sigma_max,&hm_sample);
        }
        else err = 1;
        if(err!=0){
            return err;
        }

        hm_sample = exp(hm_sample);

        sm_mean = fmin(f10 * pow(hm_sample/1e10,fa),1) * hm_sample;

        //LOG_DEBUG("hm_sample = %.3e sm_sample %.3e",hm_sample,sm_mean);

        /* STELLAR MASS SAMPLING */
        if(sigma_star > 0){
            //sample stellar masses from each halo mass assuming lognormal scatter
            sm_sample = gsl_ran_ugaussian(rng_stoc);

            /* Simply adding lognormal scatter to a delta increases the mean (2* is as likely as 0.5*)
            * so mu (exp(mu) is the median) is set so that X = exp(u + N(0,1)*sigma) has the desired mean */
            mu_lognorm = sm_mean * exp(-sigma_star*sigma_star/2);    
            sm_sample = mu_lognorm * exp(sm_sample*sigma_star);
        }
        else{
            sm_sample = sm_mean;
        }
        if(outtype==0){
            hm_cell += hm_sample;
            sm_cell += sm_sample;
        }
        else if(outtype==1){
            hm_out[ii] = hm_sample;
            sm_out[ii] = sm_sample;
        }
        else{
            LOG_ERROR("bad output type");
            return 1;
        }
    }

    *n_halo_out = n_halo;
    if(outtype==0){
        *hm_out = hm_cell;
        *sm_out = sm_cell;
    }

    return 0;
}


/* BELOW THIS LINE VARIANCE CALCULATION*/
/* The hope here is that we can evaluate the integrals over everything apart from dNdM analytically,
 * Which we can in the case of power-law scalings and lognormal scatter 
 * this also assumes that the number of halos in each cell (or region) is large
 * I'll look into this apporach more if sampling directly is too slow 
 * WARNING: I've stopped working on this for now since it would de-correlate halo properties, i.e 
 * if a cell has halo mas above mean it doesn't make it more likely that the stellar mass is below mean 
 * I can either find a way to correlate them or only use this when we need one property (seems rare)*/
/* Each integral of variance then has three components:
 * 1: product of constants from each power law
 * 2: product of scatter terms from each (lognormal) distribution
 * 3: nth moment of the halo mass function */


/* moments of the lognormal distribution for mean/median==0
 * multiply by (mean or median)^n for the full moment
 * These make up the parts of term 2 from above */
double lognormal_moment_over_mean(double sigma, double n_order){
    return exp(n_order*(n_order-1)*sigma*sigma/2);
}

double lognormal_moment_over_median(double sigma, double n_order){
    return exp(n_order*n_order*sigma*sigma/2);
}


/* The constants in front of the integral will be made up of each previous scaling relation
 * this function will calculate the product of the required parameters including the scatter terms
 * product(A_i^n_i moment_n_i), n_i = product_j>i(m_j) from scaling relations */
double stoc_constants(int type,double n_order){
    //definitions
    double F_STAR10 = astro_params_stoc->F_STAR10;
    double ALPHA_STAR = astro_params_stoc->ALPHA_STAR;
    double SIGMA_STAR = astro_params_stoc->SIGMA_STAR;
    double F_ESC10 = astro_params_stoc->F_ESC10;
    double ALPHA_ESC = astro_params_stoc->ALPHA_ESC;
    //double SIGMA_ESC = astro_params_stoc->SIGMA_ESC;
    //double SIGMA_SFR = astro_params_stoc->SIGMA_SFR;

    //Stellar mass
    if(type==0){
        return pow(F_STAR10,n_order) * lognormal_moment_over_mean(SIGMA_STAR,n_order);
    }
    //SFR, at the moment SFR ~ Stellar mass, stochasticity from t*, so m_sfr = 1
    /*if(type==1){
        return pow(F_STAR10 / T_STAR,n_order) * lognormal_moment_over_mean(SIGMA_STAR,n_order) * lognormal_moment_over_mean(SIGMA_SFR,n_order);
    }*/
    //ionising emissivity, m_esc = alpha_esc
    /*if(type==2){
        return pow(F_STAR10,n_order*ALPHA_ESC) * pow(F_ESC10,n_order) * lognormal_moment_over_mean(SIGMA_STAR,n_order*ALPHA_ESC) * lognormal_moment_over_mean(SIGMA_ESC,n_order);
    }*/
    //TODO: and so on...

    return 0.;
}

/* more approximate method where we assume the number of halos per cell is large enough such that
 * the PDF of summed properties (eg: emissivity in a cell) will be gaussian, we find the variance
 * of the desired quantities and sample these per cell, adding to the interpolation table quantities */
double stoc_halo_variance(double z, double delta, double volume, double M1, double M2, double *out){
    double var_mstar, cell_mstar;

    /* the expectation of x^2 has an outer integral over M_h and an inner (or few)
     * over (probably lognormal) distributions, The inner integrals are 
     * calculated analytically and we can use the interpolation tables
     * for the outer */

    //The halomass independent part of second moment of a lognormal distribution
    double moment_Mstar_0;
    double moment_Mstar_1;
    double moment_Mstar_2;

    double growthf = dicke(z);

    //get the moments
    /* As we add more sources of stochasticity, we will likely also need higher order
     * moments as they will collapse down to M^n dNdM terms, assuming each has a
     * power law mean and lognormal scatter */

    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        moment_Mstar_0 = EvaluatedNdMSpline(growthf,M1,M2,delta,0);
        moment_Mstar_1 = EvaluatedNdMSpline(growthf,M1,M2,delta,1);
        moment_Mstar_2 = EvaluatedNdMSpline(growthf,M1,M2,delta,2);
    }
    else{
        moment_Mstar_0 = IntegratedNdM(growthf,M1,M2,delta,0);
        moment_Mstar_1 = IntegratedNdM(growthf,M1,M2,delta,1);
        moment_Mstar_2 = IntegratedNdM(growthf,M1,M2,delta,2);
    }

    //Poisson noise in halo number
    var_mstar = volume * (1+delta) * moment_Mstar_0 * moment_Mstar_2;
    
    //draw a sample
    double var_sample;
    var_sample = gsl_ran_gaussian(rng_stoc,sqrt(var_mstar));

    // add to the mean mass
    cell_mstar = var_sample + (moment_Mstar_0*moment_Mstar_1);
    
    return cell_mstar;
}

//testing function to print stuff out from python
/* type==0: UMF
 * type==1: CMF
 * type==2: n_halo
 * type==3: sampled HM catalogue*/
int my_visible_function(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options
                        , int seed, double M, double delta, double R, double z, int type, double *result){
    //make the global structs
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);

    rng_stoc = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng_stoc, seed);

    //gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

    //gsl_rng_set(rseed, random_seed);

    double test;
    double n_order = 0.;
    int err=0;

    double growthf = dicke(z);
    double Mmin = global_params.M_MIN_INTEGRAL;
    double Mmax = RtoM(R);
    double volume = 4. / 3. * PI * R * R * R;
    //Euler to Lagrange conversion, a certain Eulerian scale R relates to mass prop. R^3 rho (1+delta)
    Mmax *= 1+delta;

    init_ps();
    if(user_params->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,1e20);
    }
    //if we are sampling, set it based on M (TODO: an actual argument)
    int sampler;
    if(M>0){
        sampler = 1;
    }
    else{
        sampler = 2;
    }
    
    LOG_DEBUG("z = %f, growth = %f, Mmin = %e, Mmax = %e, R = %f, delta = %f, M=%e",z,growthf,Mmin,Mmax,R,delta,M);
    
    double sigma_max = sigma_z0(Mmax);

    struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
        .growthf = growthf,
        .delta = delta,
        .n_order = n_order,
        .M_max = log(Mmax),
        .sigma_max = sigma_max,
    };

    if(type==0){
        //unconditional mass func
        //no randomness here so i reuse seed
        if(seed==0) {
            test = dNdM(growthf, M);
        }
        else if(seed==1) {
            test = dNdM_st(growthf, M);
        }
        else if(seed==2) {
            test = dNdM_WatsonFOF(growthf, M);
        }
        else if(seed==3) {
            test = dNdM_WatsonFOF_z(z, growthf, M);
        }
        else{
            LOG_ERROR("bad dndm type (seed)");
            return 0;
        }
        //convert to dndlnm
        *result = test * M;
    }
    else if(type==1){
        //conditional ps mass func * pow(M,n_order)
        //(RHOcrit * (1+delta) / sqrt(2*PI) * growthf * cosmo_params_stoc->OMm)
        test = MnMassfunction(log(M),(void*)&parameters_gsl_MF_con);
        test *= (RHOcrit * (1+delta) / sqrt(2*PI) * cosmo_params_stoc->OMm);
        *result = test;
    }
    else if(type==2){
        //intregrate conditional mass func
        //re-using the seed for n-order since this isn't random
        test = IntegratedNdM(growthf,log(Mmin),log(Mmax),delta,seed);
        *result = test;
    }
    else if(type==3){
        //sample mass func and output properties (n_halo, halomass, stellarmass)
        double *out_hm = calloc(MAX_HALO,sizeof(double));
        double *out_sm = calloc(MAX_HALO,sizeof(double));
        int n_halo;

        err = stoc_halo_sample(z,delta,volume,Mmin,Mmax,sampler,1,&n_halo,out_hm,out_sm);
        //fill output array N_halo, Halomass, Stellar mass ...
        //LOG_ERROR("nhalo = %d | first hm = %.3e | first sm = %.3e",n_halo,out_hm[0],out_sm[0]);
        result[0] = (double)n_halo;
        int idx;
        for(idx=0;idx<n_halo;idx++){
            result[idx+1] = out_hm[idx];
            result[idx+n_halo+1] = out_sm[idx];
        }
        free(out_hm);
        free(out_sm);
        
    }
    else if(type==4){
        //sample mass func but only output sums in cell 
        double out_hm,out_sm;
        int n_halo;

        err = stoc_halo_sample(z,delta,volume,Mmin,Mmax,sampler,0,&n_halo,&out_hm,&out_sm);
        result[0] = (double)n_halo;
        result[1] = out_hm;
        result[2] = out_sm;
    }
    else{
        LOG_ERROR("Unkown output type");
        err = 1;
    }

    return err;
}

int build_halo_grids(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options, int seed, double redshift
                    , double *dens_field, double *nh_field, double *hm_field, double *sm_field){
    //make the global structs
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);

    rng_stoc = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng_stoc, seed);
    
    double nh_buf,hm_buf,sm_buf;
    double curr_dens;
    double cell_size = user_params->BOX_LEN / user_params->HII_DIM;

    //assuming R = cell length / 2
    double volume = 4./3. * PI * cell_size * cell_size * cell_size / 8;

    double Mmin = global_params.M_MIN_INTEGRAL;
    double Rmax = cell_size / 2;
    double Mmax_meandens = RtoM(Rmax);
    double Mmax;

    int x,y,z;

#pragma omp parallel shared(dens_field,nh_field,hm_field,sm_field,volume,Mmin,redshift) private(x,y,z,Mmax,curr_dens,nh_buf,hm_buf,sm_buf) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (x=0; x<user_params->HII_DIM; x++){
            for (y=0; y<user_params->HII_DIM; y++){
                for (z=0; z<user_params->HII_DIM; z++){
                
                curr_dens = dens_field[HII_R_INDEX(x,y,z)];

                //adjust max mass scale
                Mmax = Mmax_meandens * (1+curr_dens);

                stoc_halo_sample(redshift,curr_dens,volume,Mmin,Mmax,2,1,&nh_buf,&hm_buf,&sm_buf);

                nh_field[HII_R_INDEX(x,y,z)] = nh_buf;
                hm_field[HII_R_INDEX(x,y,z)] = hm_buf;
                sm_field[HII_R_INDEX(x,y,z)] = sm_buf;
                }
            }
        }
    }
    return 0;
}