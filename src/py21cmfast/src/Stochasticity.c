/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//max guesses for rejection sampling
#define MAX_ITERATIONS 1e5
//max halos in memory for test functions
#define MAX_HALO 10000000

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

//single sample of the halo mass function
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
int stoc_halo_cell(double z, double delta, double volume, double M1, double M2,double *n_halo_out, double *hm_out, double *sm_out){
    int n_halo,err;
    double nh_buf,mu_lognorm;

    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;

    double hm_sample, sm_sample, sm_mean;
    double hm_cell=0, sm_cell=0.;

    double sigma_max = sigma_z0(M2);
    double growthf = dicke(z);

    //get average number of halos in cell n_order=0
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        nh_buf = EvaluatedNdMSpline(growthf, M1, M2, delta, 0);
    }
    else{
        nh_buf = IntegratedNdM(growthf, log(M1), log(M2), delta, 0);
    }
    nh_buf *= volume;

    //sample poisson for stochastic halo number
    if(flag_options_stoc->HALO_STOCHASTICITY){
        n_halo = gsl_ran_poisson(rng_stoc,nh_buf);
    }
    //otherwise floor (I don't think the rounding method will matter...)
    else{
        n_halo = (int)nh_buf;
    }
    
    *n_halo_out = n_halo;

    /* to save memory we calulate properties and sample within the loop
     * it may be faster to chunk it up or generate a catalogue (MANY GB)
     * before summing */

    //TODO: set up mass distribution for sampling here
    //BEWARE: assuming max(dNdM) == dNdM(Mmin)
    double ymin,ymax;
    ymin = 0;
    ymax = dNdM_conditional(growthf,log(M1),log(M2),Deltac,delta,sigma_max);
    //ymax = ymax * (RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm);

    for(int ii=0;ii<n_halo;ii++){
        //sample from the cHMF
        /* TODO: figure out the best method for this
         * either build an x = iCDF(delta,M) table and interpolate
         * or use rejection sampling on the existing mass function table */
        
        err = sample_dndM_rejection(growthf,delta,log(M1),log(M2),ymin,ymax,sigma_max,&hm_sample);
        if(err!=0){
            return 1;
        }

        hm_sample = exp(hm_sample);
        hm_cell += hm_sample;

        sm_mean = fmin(f10 * pow(hm_sample/1e10,fa),1) * hm_sample;

        /* STELLAR MASS SAMPLING */
        //sample stellar masses from each halo mass assuming lognormal scatter
        sm_sample = gsl_ran_ugaussian(rng_stoc);

        /* Simply adding lognormal scatter to a delta increases the mean (2* is as likely as 0.5*)
         * so mu (exp(mu) is the median) is set so that X = exp(u + N(0,1)*sigma) has the desired mean */
        mu_lognorm = sm_mean * exp(-sigma_star*sigma_star/2);    
        sm_sample = mu_lognorm * exp(sm_sample*sigma_star);
        
        sm_cell += sm_sample;
    }

    *hm_out = hm_cell;
    *sm_out = sm_cell;

    return 0;
}

/* As above but outputs a catalogue for testing */
int stoc_halo_cat(double z, double delta, double volume, double M1, double M2, int *n_halo_out, double *result_hm, double *result_sm){
    int n_halo,err;
    double nh_buf,mu_lognorm;
    double sm_mean,sm_sample;
    double f10 = astro_params_stoc->F_STAR10;
    double fa = astro_params_stoc->ALPHA_STAR;
    double sigma_star = astro_params_stoc->SIGMA_STAR;

    double hm_sample=0;

    double sigma_max = sigma_z0(M2);
    double growthf = dicke(z);

    //get average number of halos in cell n_order=0
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        nh_buf = EvaluatedNdMSpline(growthf, M1, M2, delta, 0);
    }
    else{
        nh_buf = IntegratedNdM(growthf, log(M1), log(M2), delta, 0);
    }
    nh_buf *= volume;

    //sample poisson for stochastic halo number
    if(flag_options_stoc->HALO_STOCHASTICITY){
        n_halo = gsl_ran_poisson(rng_stoc,nh_buf);
    }
    //otherwise floor (I don't think the rounding method will matter...)
    else{
        n_halo = (int)nh_buf;
    }

    *n_halo_out = n_halo;

    /* to save memory we calulate properties and sample within the loop
     * it may be faster to chunk it up or generate a catalogue (MANY GB)
     * before summing */

    if(n_halo > MAX_HALO/2 - 1){
        LOG_ERROR("too many halos %d >%d / 2 - 1",n_halo,MAX_HALO);
        return;
    }
    
    //bounding box for rejection sampling
    //BEWARE: assuming max(dNdM) == dNdM(Mmin)
    double ymin,ymax;
    ymin = 0;
    ymax = dNdM_conditional(growthf,log(M1),log(M2),Deltac,delta,sigma_max);
    //ymax = ymax * (RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm);
    
    //LOG_DEBUG("Mmin = %.3e | Mmax = %.3e | ymin = %.3e | ymax = %.3e | eg = %.3e",M1,M2,ymin,ymax,dNdM_conditional(growthf,log(100*M1),log(M2),Deltac,delta,sigma_max));
    //LOG_DEBUG("adjusted ymax = %.3e",ymax*(RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm));
    
    //LOG_DEBUG("Mmax = %.3e | delta = %.3e | nhalo = %d",M2,delta,n_halo);
    
    for(int ii=0;ii<n_halo;ii++){
        //sample from the HMF
        /* TODO: figure out the best method for this
         * either build an x = iCDF(delta,M) table and interpolate
         * or use rejection sampling on the existing mass function table */
        
        err = sample_dndM_rejection(growthf,delta,log(M1),log(M2),ymin,ymax,sigma_max,&hm_sample);
        if(err!=0){
            return 1;
        }

        hm_sample = exp(hm_sample);
        result_hm[ii] = hm_sample;

        /* take sampled HM and apply lognormal scatter */

        sm_mean = fmin(f10 * pow(hm_sample/1e10,fa),1) * hm_sample;

        /* STELLAR MASS SAMPLING */
        //sample stellar masses from each halo mass assuming lognormal scatter
        sm_sample = gsl_ran_ugaussian(rng_stoc);

        /* Simply adding lognormal scatter to a delta increases the mean (2* is as likely as 0.5*)
         * so mu (exp(mu) is the median) is set so that X = exp(u + N(0,1)*sigma) has the desired mean */
        mu_lognorm = sm_mean * exp(-sigma_star*sigma_star/2);
        sm_sample = mu_lognorm * exp(sm_sample*sigma_star);

        //if(ii<10){
        //    LOG_DEBUG("halo %.3e | mean %.3e | sample %.3e | mu = %.3e",hm_sample,sm_mean,sm_sample,mu_lognorm);
        //}
        
        result_sm[ii] = sm_sample;

    }
    return 0;
}

/* BELOW THIS LINE VARIANCE CALCULATION*/
/* The hope here is that we can evaluate the integrals over everything apart from dNdM analytically,
 * Which we can in the case of power-law scalings and lognormal scatter 
 * this also assumes that the number of halos in each cell (or region) is large
 * I'll look into this apporach more if sampling directly is too slow*/
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
double halo_stoc_variance(double growthf, double M1, double M2, double delta, double n_order, double volume){
    double var_mstar, cell_mstar;

    /* the expectation of x^2 has an outer integral over M_h and an inner (or few)
     * over (probably lognormal) distributions, The inner integrals are 
     * calculated analytically and we can use the interpolation tables can be used
     * for the outer */

    //The halomass independent part of second moment of a lognormal distribution
    double moment_Mstar_0;
    double moment_Mstar_1;
    double moment_Mstar_2;

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
    if(flag_options_stoc->HALO_STOCHASTICITY){
        var_mstar = volume * delta * moment_Mstar_0 * moment_Mstar_2;
    }
    //no Poisson noise
    else{
        var_mstar = volume * delta * moment_Mstar_0 * (moment_Mstar_2 - moment_Mstar_1*moment_Mstar_1);
    }

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

    /*writeUserParams(user_params_stoc);
    writeAstroParams(flag_options_stoc,astro_params_stoc);
    writeCosmoParams(cosmo_params_stoc);
    writeFlagOptions(flag_options_stoc);*/

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
    
    LOG_DEBUG("z = %f, growth = %f, Mmin = %e, Mmax = %e, R = %f, delta = %f, M=%e",z,growthf,Mmin,Mmax,R,delta,M);
    
    double sigma_max = sigma_z0(Mmax);

    struct parameters_gsl_MF_con_int_ parameters_gsl_MF_con = {
        .growthf = growthf,
        .delta = delta,
        .n_order = n_order,
        .M_max = log(Mmax),
        .sigma_max = sigma_max,
    };

    //LOG_DEBUG("test integral from %e to %e = %e",Mmin,Mmax,test);

    if(type==0){
        //unconditional PS mass func
        test = dNdM(growthf,M);
        *result = test;
    }
    else if(type==1){
        //conditional ps mass func * pow(M,n_order)
        test = MnMassfunction(log(M),(void*)&parameters_gsl_MF_con);
        test *= (RHOcrit * (1+delta) * sqrt(2/PI) / 2 * growthf * cosmo_params_stoc->OMm);
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

        err = stoc_halo_cat(z,delta,volume,Mmin,Mmax,&n_halo,out_hm,out_sm);
        //fill output array N_halo, Halomass, Stellar mass ...
        result[0] = (double)n_halo;
        int idx = 0;
        while(idx < n_halo){
            result[idx+1] = out_hm[idx];
            result[idx+n_halo+1] = out_sm[idx];
            idx++;
        }
    }
    else if(type==4){
        //sample mass func but only output sums in cell 
        double n_halo,out_hm,out_sm;
        err = stoc_halo_cell(z,delta,volume,Mmin,Mmax,&n_halo,&out_hm,&out_sm);
        result[0] = n_halo;
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
    for (x=0; x<user_params->HII_DIM; x++){
        for (y=0; y<user_params->HII_DIM; y++){
            for (z=0; z<user_params->HII_DIM; z++){
            
            curr_dens = dens_field[HII_R_INDEX(x,y,z)];

            //adjust max mass scale
            Mmax = Mmax_meandens * (1+curr_dens);

            stoc_halo_cell(z,curr_dens,volume,Mmin,Mmax,&nh_buf,&hm_buf,&sm_buf);

            nh_field[HII_R_INDEX(x,y,z)] = nh_buf;
            hm_field[HII_R_INDEX(x,y,z)] = hm_buf;
            sm_field[HII_R_INDEX(x,y,z)] = sm_buf;

            }
        }
    }
    return 0;
}