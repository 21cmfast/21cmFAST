/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//BIG TODO: sort out single/double precision all the way through

//max guesses for rejection sampling
#define MAX_ITERATIONS 1e5

//max halos in memory for test functions
//buffer size (per cell of arbitrary size) in the sampling function
//this is big enough for a ~20Mpc mean density cell
#define MAX_HALO_CELL (int)1e8

struct AstroParams *astro_params_stoc;
struct CosmoParams *cosmo_params_stoc;
struct UserParams *user_params_stoc;
struct FlagOptions *flag_options_stoc;

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
    double sigma2 = params.sigma_max; //M2 and sigma2 are degenerate, remove one
    double M_filter = params.M_max;

    if (M_filter < M) return 0.;

    //M1 is the mass of interest, M2 doesn't seem to be used (input as max mass),
    // delta1 is critical, delta2 is current, sigma is sigma(Mmax,z=0)
    //dNdlnM = dfcoll/dM * M / M * constants
    mf = dNdM_conditional(growthf,M,M_filter,Deltac,delta,sigma2);

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
//TODO: rename since its now an integral of M * dfcoll/dm = dNdM / [(RHOcrit * (1+delta) / sqrt(2.*PI) * cosmo_params_stoc->OMm)]
double IntegratedNdM(double growthf, double M1, double M2, double M_filter, double delta, double n_order){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.01; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    double sigma = sigma_z0(exp(M_filter));

    struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
        .growthf = growthf,
        .delta = delta,
        .n_order = n_order,
        .sigma_max = sigma,
        .M_max = M_filter,
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
        LOG_ERROR("data: growthf=%e M2=%e delta=%e,sigma2=%e",growthf,M2,delta,sigma);
        LOG_ERROR("data: growthf=%e M2=%e delta=%e,sigma2=%e",parameters_gsl_MF_con.growthf,parameters_gsl_MF_con.M_max,parameters_gsl_MF_con.delta,parameters_gsl_MF_con.sigma_max);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    //constants for dfcoll -> dNdlnm
    result = result;
    return result;
}

// Calculate the Nth moment of the lognormal distribution, integrated over the halo mass distribution
// TODO: add option for poisson process perturbing the zeroth moment for the sampling method
double EvaluatedNdMSpline(double growthf, double M1, double M2, double M_filter, double delta, double n_order){
    //for testing, I'm assuming fixed (mean) number of halos per cell and P(M_halo) = diracdelta(M)
    //TODO: replace with the actual calculation here integrating/interpolating over dNdM and adding noise if needed
    double buf;

    //the interpolation will have an extra dimension compared to the N_ion one for the nth moment (minus R if we build the grids)
    //
    LOG_ULTRA_DEBUG("dNdM interpolation table not implemented, integrating...");
    return IntegratedNdM(growthf, M1, M2, M_filter, delta, n_order);
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
int sample_dndM_inverse(double growthf, double delta, double Mmin, double Mmax, double ymin, double ymax, double sigma2, gsl_rng * rng, double *result){
    double y1, fcoll_x, sigma_M;

    y1 = gsl_rng_uniform(rng);
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

    LOG_ULTRA_DEBUG("root M=%.2e (s = %.2e) iter %d | lo=%.2e hi=%.2e",exp(r),sigma_M,iter,x_lo,x_hi);
    LOG_ULTRA_DEBUG("min=%.2e max=%.2e fcoll_x = %.2e, sample y = %.2e, s2 = %.2e",exp(Mmin),exp(Mmax),fcoll_x,y1,sigma2);

    gsl_root_fsolver_free(s);

    if(!isfinite(r)){
        LOG_ERROR("Value for nu_tau_one_MINI is infinite or NAN");
        Throw(InfinityorNaNError);
    }

    *result = r;

    return 0;
}

//single sample of the halo mass function using rejection method
int sample_dndM_rejection(double growthf, double delta, double Mmin, double Mmax, double M_filter, double ymin, double ymax, double sigma, gsl_rng * rng, double *result){
    double x1,y1, MassFunction;
    int c=0;

    //rejection sampling for now
    //inverse CDF sampling is probably better but requires an extra root find that isn't there yet
    while(c < MAX_ITERATIONS){
        //uniform sampling of logM between given limits
        x1 = Mmin + (Mmax-Mmin)*gsl_rng_uniform(rng);
        y1 = ymin + (ymax-ymin)*gsl_rng_uniform(rng);

        //for halo abundances (dNdlogM) from dfcoll/dM, *M for dlogm and / M for dfcolldN
        MassFunction = dNdM_conditional(growthf,x1,M_filter,Deltac,delta,sigma);
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
int stoc_halo_sample(double z, double delta, double delta_vol, double volume, double M_min, double M_max, int nbins, int sampler, int outtype, gsl_rng * rng, int *n_halo_out, double *hm_out, double *sm_out){
    //delta check, I could also make this output 1 halo with the mass of the cell for large delta (assuming we aren't getting big haloes from FindHaloes.c)
    if(delta > Deltac || delta < -1){
        *n_halo_out = 0;
        return 1;
    }
    
    int err;
    double nh_buf,mu_lognorm;

    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;

    double hm_sample, sm_sample, sm_mean;
    double hm_cell=0, sm_cell=0.;

    double sigma_max = sigma_z0(M_max);
    double growthf = dicke(z);

    double hm_min = 1e20, hm_max = 0;

    double lnM_lo, lnM_cen, lnM_hi;
    //these are the same at the end, n_halo incremented by poisson, halo_count incremented by updating halo list
    int n_halo = 0, halo_count = 0, nh;
    int jj,ii;
    for (jj=0;jj<nbins;jj++){
        //mass range in bin
        lnM_lo = log(M_min) + ((jj+0.0)/nbins * (log(M_max) - log(M_min)));
        lnM_cen = log(M_min) + ((jj+0.5)/nbins * (log(M_max) - log(M_min)));
        lnM_hi = log(M_min) + ((jj+1.)/nbins * (log(M_max) - log(M_min)));

        //get average number of halos in cell n_order=0
        if(user_params_stoc->USE_INTERPOLATION_TABLES){
            nh_buf = EvaluatedNdMSpline(growthf, lnM_lo, lnM_hi, log(M_max), delta, 0);
        }
        else{
            nh_buf = IntegratedNdM(growthf, lnM_lo, lnM_hi, log(M_max), delta, 0);
        }
        //constants to go from integral(1/M dFcol/dlogM) dlogM to integral(dNdlogM) dlogM
        nh_buf = nh_buf * volume * (RHOcrit * (1+delta_vol) / sqrt(2.*PI) * cosmo_params_stoc->OMm);

        //LOG_ERROR("nhalo = %.3e",nh_buf);

        //sample poisson for stochastic halo number
        nh = gsl_ran_poisson(rng,nh_buf);
        n_halo += nh;

        /* to save memory we calulate properties and sample within the loop
        * it may be faster to chunk it up or generate a catalogue (MANY GB)
        * before summing */

        //TODO: set up mass distribution for sampling here
        //BEWARE: assuming max(dNdM) == dNdM(Mmin)
        double ymin,ymax;
        ymin = 0;
        ymax = dNdM_conditional(growthf,lnM_lo,log(M_max),Deltac,delta,sigma_max);
        //ymax = ymax * (RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm);

        //LOG_ERROR("n halo = %d",n_halo);
        for(ii=0;ii<nh;ii++){
            //sample from the cHMF
            /* TODO: figure out the best method for this
            * either build an x = iCDF(delta,M) table and interpolate
            * or use rejection sampling on the existing mass function table */
            if (sampler==1){
                err = sample_dndM_rejection(growthf,delta,lnM_lo,lnM_hi,log(M_max),ymin,ymax,sigma_max,rng,&hm_sample);
            }
            else if(sampler==2){
                //err = sample_dndM_inverse(growthf,delta,lnM_lo,lnM_hi,ymin,ymax,sigma_max,&hm_sample);
                LOG_ERROR("inversion sampler not finished");
                err = 1;
            }
            else err = 1;
            if(err!=0){
                return err;
            }

            hm_sample = exp(hm_sample);

            if(hm_sample > hm_max) hm_max = hm_sample;
            if(hm_sample < hm_min) hm_min = hm_sample;

            sm_mean = fmin(f10 * pow(hm_sample/1e10,fa),1) * hm_sample;

            //LOG_DEBUG("hm_sample = %.3e sm_sample %.3e",hm_sample,sm_mean);

            /* STELLAR MASS SAMPLING */
            if(sigma_star > 0){
                //sample stellar masses from each halo mass assuming lognormal scatter
                sm_sample = gsl_ran_ugaussian(rng);

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
                halo_count++;
            }
            else if(outtype==1){
                hm_out[halo_count] = hm_sample;
                sm_out[halo_count] = sm_sample;
                halo_count++;
            }
            else{
                LOG_ERROR("bad output type");
                return 1;
            }
        }
        LOG_ULTRA_DEBUG("bin %d done | log10M = [%.3f,%.3f] | M_filt = %.3e ||  nhalo = %d (p: %d m: %.3e)", jj,lnM_lo/log(10),lnM_hi/log(10),M_max,n_halo,nh,nh_buf);

    }
    *n_halo_out = n_halo;
    if(outtype==0){
        *hm_out = hm_cell;
        *sm_out = sm_cell;
    }

    LOG_ULTRA_DEBUG("sampled %d (%d) halos between masses %.3e and %.3e",n_halo,halo_count,hm_min,hm_max);
    return 0;
}

//testing function to print stuff out from python
/* type==0: UMF
 * type==1: CMF
 * type==2: n_halo
 * type==3: sampled HM catalogue*/
int my_visible_function(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options
                        , int seed, double M, bool eulerian, double delta, double R, double z, int nbins, int type, double *result){
    //make the global structs
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);

    gsl_rng * rng_stoc;
    rng_stoc = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng_stoc, seed);

    //gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

    //gsl_rng_set(rseed, random_seed);
    double test;
    int err=0;

    double growthf = dicke(z);
    double Mmin;
    double Mmax = RtoM(R);
    double volume = 4. / 3. * PI * R * R * R;
    double delta_l;
    //Euler to Lagrange conversion, a certain Eulerian scale R relates to mass prop. R^3 rho (1+delta)
    if(eulerian){
        //Mo & White 1996 TODO:put in function
        //convert Eulerian Pertubed overdensity to Lagrangian overdensity
        //the mass in the cell (and hence R) is scaled by the eulerian density, the CMF uses lagrangian
        Mmax *= 1+delta;
        //volume *= 1+delta;
        delta_l = -1.35*pow(1+delta,-2./3.) + 0.78785*pow(1+delta,-0.58661) - 1.12431*pow(1+delta,-0.5) + 1.68647;
    }
    else{
        delta = delta * growthf;
        delta_l = delta * growthf;
    }
    //don't do anything for delta outside range               
    if(delta > Deltac || delta < -1){
        return 0;
    }

    //set the minimum source mass
    //TODO: include smaller halos, apply duty cycle in HM -> SM part
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        Mmin = astro_params->M_TURN;
    }
    else {
        if(flag_options->M_MIN_in_Mass) {
            Mmin = (astro_params->M_TURN);
        }
        else {
            //set the minimum source mass
            if (astro_params->ION_Tvir_MIN < 9.99999e3) { // neutral IGM
                Mmin = TtoM(z, astro_params->ION_Tvir_MIN, 1.22);
            }
            else { // ionized IGM
                Mmin = TtoM(z, astro_params->ION_Tvir_MIN, 0.6);
            }
        }
    }

    init_ps();
    if(user_params->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,1e20);
    }
    //if we are sampling, set it based on M since this isn't used (TODO: an actual argument)
    int sampler;
    if(M>0){
        sampler = 1;
    }
    else{
        sampler = 2;
    }
    
    LOG_SUPER_DEBUG("z = %f, growth = %f, Mmin = %e, Mmax = %e, R = %f, delta = %f, M=%e",z,growthf,Mmin,Mmax,R,delta,M);

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
        struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
            .growthf = growthf,
            .delta = delta_l,
            .n_order = seed, //no rnadomness here
            .M_max = log(Mmax),
            .sigma_max = sigma_z0(Mmax),
        };
        test = MnMassfunction(log(M),(void*)&parameters_gsl_MF_con);
        test *= (RHOcrit * (1+delta) / sqrt(2.*PI) * cosmo_params_stoc->OMm);
        *result = test;
    }
    else if(type==2){
        //intregrate conditional mass func
        //re-using the seed for n-order since this isn't random
        test = IntegratedNdM(growthf,log(Mmin),log(Mmax),log(Mmax),delta_l,seed);
        test *= volume * (RHOcrit * (1+delta) / sqrt(2.*PI) * cosmo_params_stoc->OMm);
        *result = test;
    }
    else if(type==3){
        //sample mass func and output properties (n_halo, halomass, stellarmass)
        double *out_hm = calloc(MAX_HALO_CELL,sizeof(double));
        double *out_sm = calloc(MAX_HALO_CELL,sizeof(double));
        int n_halo;

        err = stoc_halo_sample(z,delta_l,delta,volume,Mmin,Mmax,nbins,sampler,1,rng_stoc,&n_halo,out_hm,out_sm);
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

        err = stoc_halo_sample(z,delta_l,delta,volume,Mmin,Mmax,nbins,sampler,0,rng_stoc,&n_halo,&out_hm,&out_sm);
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

/* TODO list: 
 * add 'forbidden' cell mask (or input masked density array) for big halos 
 * think about how to sample over mass rather than number of halos*/
int build_halo_grids(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options, int seed, double redshift
                    ,bool eulerian, float *dens_field, int *nh_field, float *hm_field, float *sm_field){
    //make the global structs
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);
    
    // setting tbe random seeds (copied from GenerateICs.c)
    gsl_rng * rng_stoc[user_params->N_THREADS];
    gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

    gsl_rng_set(rseed, seed);

    omp_set_num_threads(user_params->N_THREADS);

    unsigned int seeds[user_params->N_THREADS];

    // For multithreading, seeds for the RNGs are generated from an initial RNG (based on the input random_seed) and then shuffled (Author: Fred Davies)
    int num_int = INT_MAX/16;
    int i, thread_num;
    unsigned int *many_ints = (unsigned int *)malloc((size_t)(num_int*sizeof(unsigned int))); // Some large number of possible integers
    for (i=0; i<num_int; i++) {
        many_ints[i] = i;
    }

    gsl_ran_choose(rseed, seeds, user_params->N_THREADS, many_ints, num_int, sizeof(unsigned int)); // Populate the seeds array from the large list of integers
    gsl_ran_shuffle(rseed, seeds, user_params->N_THREADS, sizeof(unsigned int)); // Shuffle the randomly selected integers

    int checker;

    checker = 0;
    // seed the random number generators
    // TODO: this should probably be in UsefulFunctions.c
    for (thread_num = 0; thread_num < user_params->N_THREADS; thread_num++){
        switch (checker){
            case 0:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 1:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_gfsr4);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 2:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_cmrg);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 3:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_mrg);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 4:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_taus2);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
        } // end switch

        checker += 1;

        if(checker==5) {
            checker = 0;
        }
    }

    free(many_ints);
    double hm_total=0.,sm_total=0.;
    int nh_total=0;
    double curr_dens;
    //cell size for smoothing / CMF calculation
    double cell_size_R = user_params->BOX_LEN / user_params->HII_DIM * L_FACTOR;
    //cell size for volume calculation
    double cell_size_L = user_params->BOX_LEN / user_params->HII_DIM;

    double volume_meandens = cell_size_L * cell_size_L * cell_size_L; // ~~ 4/3 pi cell_size_R^3

    double Mmax_meandens = RtoM(cell_size_R);
    double Mmax;
    double volume;
    double Mmin;
    double delta_l;

    int x,y,z;
    int nbins = 1;
    int sampler = 1; //rejection sampling
    int outtype = 0; //cell outputs (not keeping catalogues in every cell)
    int error;
    int bad_cells=0;

    //set the minimum source mass
    //TODO: include smaller halos, apply duty cycle in HM -> SM part
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        Mmin = astro_params->M_TURN;
    }
    else {
        if(flag_options->M_MIN_in_Mass) {
            Mmin = (astro_params->M_TURN);
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

    init_ps();
    if(user_params->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,1e20);
    }
    double hm_buf;
    double sm_buf;
    int nh_buf;
    LOG_DEBUG("Beginning stochastic halo sampling on %d ^3 grid",user_params->HII_DIM);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e,volume = %.3e, cell length = %.3e R = %.3e",redshift,Mmin,Mmax_meandens,volume_meandens,cell_size_L,cell_size_R);

#pragma omp parallel shared(dens_field,nh_field,hm_field,sm_field,volume_meandens,Mmin,redshift,rng_stoc) private(x,y,z,Mmax,volume,curr_dens,nh_buf,hm_buf,sm_buf,error) num_threads(user_params->N_THREADS) reduction(+:bad_cells,nh_total,hm_total,sm_total)
    {
        int print_counter = 0;
#pragma omp for
        for (x=0; x<user_params->HII_DIM; x++){
            for (y=0; y<user_params->HII_DIM; y++){
                for (z=0; z<user_params->HII_DIM; z++){
                
                    curr_dens = (double)dens_field[HII_R_INDEX(x,y,z)];

                    //adjust for lagrangian/eulerian
                    //volume * delta = total mass
                    if(eulerian){
                        Mmax = Mmax_meandens * (1+curr_dens);
                        //volume = volume_meandens * (1+curr_dens);
                        volume = volume_meandens;
                        delta_l = -1.35*pow(1+curr_dens,-2./3.) + 0.78785*pow(1+curr_dens,-0.58661) - 1.12431*pow(1+curr_dens,-0.5) + 1.68647;
                    }
                    else{
                        Mmax = Mmax_meandens;
                        curr_dens = curr_dens * dicke(redshift);
                        delta_l = curr_dens;
                        volume = volume_meandens;
                    }
                    error = stoc_halo_sample(redshift,delta_l,curr_dens,volume,Mmin,Mmax,nbins,sampler,outtype,rng_stoc[omp_get_thread_num()],&nh_buf,&hm_buf,&sm_buf);

                    //output grids of summed n_halo, halo mass, stellar mass
                    if(error==0){
                        nh_field[HII_R_INDEX(x,y,z)] = nh_buf;
                        hm_field[HII_R_INDEX(x,y,z)] = (float)hm_buf;
                        sm_field[HII_R_INDEX(x,y,z)] = (float)sm_buf;
                        if(nh_buf > 0){
                            if(print_counter < 20){
                                LOG_SUPER_DEBUG("(%d,%d,%d) d = %.2f: nh = %d | hm = %.3e | sm = %.3e",x,y,z,curr_dens,nh_buf,
                                                hm_buf,sm_buf);
                            }
                            nh_total += nh_buf;
                            hm_total += hm_buf;
                            sm_total += sm_buf;
                            print_counter++;
                        }
                    }
                    else{
                        //current behaviour for delta > Deltac, delta < -1 or other errors
                        nh_field[HII_R_INDEX(x,y,z)] = 0;
                        hm_field[HII_R_INDEX(x,y,z)] = 0;
                        sm_field[HII_R_INDEX(x,y,z)] = 0;
                        bad_cells++;
                    }
                }
            }
        }
    }

    LOG_DEBUG("Finished halo sampling, %d bad cells. Totals (NH,HM,SM) = (%d,%.3e,%.3e)",bad_cells,nh_total,hm_total,sm_total);
    return 0;
}

int build_halo_cats(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options, int seed, double redshift
                    ,bool eulerian, float *dens_field, int * n_halo_out, int *halo_coords, float *halo_masses, float * stellar_masses){
    //make the global structs
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);
    
    // setting tbe random seeds (copied from GenerateICs.c)
    gsl_rng * rng_stoc[user_params->N_THREADS];
    gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

    gsl_rng_set(rseed, seed);

    omp_set_num_threads(user_params->N_THREADS);

    unsigned int seeds[user_params->N_THREADS];

    // For multithreading, seeds for the RNGs are generated from an initial RNG (based on the input random_seed) and then shuffled (Author: Fred Davies)
    int num_int = INT_MAX/16;
    int i, thread_num;
    unsigned int *many_ints = (unsigned int *)malloc((size_t)(num_int*sizeof(unsigned int))); // Some large number of possible integers
    for (i=0; i<num_int; i++) {
        many_ints[i] = i;
    }

    gsl_ran_choose(rseed, seeds, user_params->N_THREADS, many_ints, num_int, sizeof(unsigned int)); // Populate the seeds array from the large list of integers
    gsl_ran_shuffle(rseed, seeds, user_params->N_THREADS, sizeof(unsigned int)); // Shuffle the randomly selected integers

    int checker;

    checker = 0;
    // seed the random number generators
    // TODO: this should probably be in UsefulFunctions.c
    for (thread_num = 0; thread_num < user_params->N_THREADS; thread_num++){
        switch (checker){
            case 0:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 1:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_gfsr4);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 2:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_cmrg);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 3:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_mrg);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
            case 4:
                rng_stoc[thread_num] = gsl_rng_alloc(gsl_rng_taus2);
                gsl_rng_set(rng_stoc[thread_num], seeds[thread_num]);
                break;
        } // end switch

        checker += 1;

        if(checker==5) {
            checker = 0;
        }
    }

    free(many_ints);
    double hm_total=0.,sm_total=0.;
    int nh_total=0;
    double curr_dens;
    //cell size for smoothing / CMF calculation
    double cell_size_R = user_params->BOX_LEN / user_params->HII_DIM * L_FACTOR;
    //cell size for volume calculation
    double cell_size_L = user_params->BOX_LEN / user_params->HII_DIM;

    double volume_meandens = cell_size_L * cell_size_L * cell_size_L;

    double Mmax_meandens = RtoM(cell_size_R);
    double Mmax;
    double volume;
    double Mmin;
    double delta_l;

    int x,y,z;
    int nbins = 1;
    int sampler = 1; //rejection sampling
    int outtype = 1; //halo catalogues
    int error;
    int bad_cells=0;

    //set the minimum source mass
    //TODO: include smaller halos, apply duty cycle in HM -> SM part
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        Mmin = astro_params->M_TURN;
    }
    else {
        if(flag_options->M_MIN_in_Mass) {
            Mmin = (astro_params->M_TURN);
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

    init_ps();
    if(user_params->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,1e20);
    }
    LOG_DEBUG("Beginning stochastic halo sampling on %d ^3 grid",user_params->HII_DIM);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e,volume = %.3e (%.3e), cell length = %.3e R = %.3e D = %.3e",redshift,Mmin,Mmax_meandens,volume_meandens,Mmax_meandens/RHOcrit/cosmo_params->OMm,cell_size_L,cell_size_R,dicke(redshift));

    int total_n_halo = 0;
    double total_halomass = 0.;
    double total_starmass = 0.;

//BIG TODO: find a way to parallelise, (this can also apply to FindHalos.c)
//I will need to allocate buffers for the catalogues on each thread, then concatenate to the output
//the only issue is allocating a buffer of a reasonable size based on the box size
//#pragma omp parallel shared(dens_field,halo_masses,halo_coords,volume_meandens,Mmin,redshift,rng_stoc,total_n_halo) private(x,y,z,Mmax,volume,curr_dens,nh_buf,hm_buf,sm_buf,error) num_threads(user_params->N_THREADS) reduction(+:bad_cells,nh_total,hm_total,sm_total)
    {
        //buffers per cell
        double * hm_buf = calloc(MAX_HALO_CELL,sizeof(double));
        double * sm_buf = calloc(MAX_HALO_CELL,sizeof(double));
        int nh_buf;
        int print_counter = 0;
        double cell_hm = 0, cell_sm = 0;
//#pragma omp for
        for (x=0; x<user_params->HII_DIM; x++){
            for (y=0; y<user_params->HII_DIM; y++){
                for (z=0; z<user_params->HII_DIM; z++){
                
                    cell_hm = 0;
                    cell_sm = 0;
                    curr_dens = (double)dens_field[HII_R_INDEX(x,y,z)];

                    //adjust for lagrangian/eulerian
                    if(eulerian){
                        Mmax = Mmax_meandens * (1+curr_dens);
                        //volume = volume_meandens * (1+curr_dens);
                        volume = volume_meandens;
                        delta_l = -1.35*pow(1+curr_dens,-2./3.) + 0.78785*pow(1+curr_dens,-0.58661) - 1.12431*pow(1+curr_dens,-0.5) + 1.68647;
                    }
                    else{
                        Mmax = Mmax_meandens;
                        curr_dens = curr_dens * dicke(redshift);
                        delta_l = curr_dens;
                        volume = volume_meandens;
                    }

                    error = stoc_halo_sample(redshift,delta_l,curr_dens,volume,Mmin,Mmax,nbins,sampler,outtype,rng_stoc[omp_get_thread_num()],&nh_buf,hm_buf,sm_buf);

                    //output total halo number, catalogues of masses and positions
                    if(error==0){
                        for(i=0;i<nh_buf;i++){
                            halo_masses[total_n_halo] = hm_buf[i];
                            stellar_masses[total_n_halo] = sm_buf[i];
                            halo_coords[0 + 3*total_n_halo] = x;
                            halo_coords[1 + 3*total_n_halo] = y;
                            halo_coords[2 + 3*total_n_halo] = z;
                            //update totals & debug
                            total_n_halo++;
                            total_halomass += hm_buf[i];
                            total_starmass += sm_buf[i];
                            cell_hm += hm_buf[i];
                            cell_sm += sm_buf[i];
                        }
                    }
                    else{
                        //current behaviour for delta > Deltac, delta < -1 or other errors
                        //no new halos, record bad cell
                        bad_cells++;
                    }
                    if(nh_buf > 0 && print_counter < 20){
                        
                        LOG_SUPER_DEBUG("(%d,%d,%d) d = %.2f: nh = %d | hm = %.3e | sm = %.3e",x,y,z,curr_dens,nh_buf,
                                        cell_hm,cell_sm);
                        print_counter++;
                    }
                }
            }
        }    
        free(hm_buf);
        free(sm_buf);
        *n_halo_out = total_n_halo;
    }
    LOG_DEBUG("Finished halo sampling %d bad cells || nh = %d | hm = (%.3e, %.3e, %.3e) total= %.3e | sm = (%.3e, %.3e, %.e) total = %.3e",bad_cells,total_n_halo
                                    ,halo_masses[0],halo_masses[1],halo_masses[2],total_halomass
                                    ,stellar_masses[0],stellar_masses[1],stellar_masses[2],total_starmass);
    return 0;
}
