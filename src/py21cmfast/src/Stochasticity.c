/*functions which deal with stochasticity
 * i.e sampling the halo mass function and
 * other halo relations.*/

//BIG TODO: sort out single/double precision all the way through
//STYLE: make the function names consistent, re-order functions
//so it makes sense and make a map of the call trees to look for modularisation

//max guesses for rejection sampling / root finding
//it should be low because both should take <20 attempts
#define MAX_ITERATIONS 1e3

//max halos in memory for test functions
//buffer size (per cell of arbitrary size) in the sampling function
//this is big enough for a ~20Mpc mean density cell
//TODO: both of these should be set depending on size, resolution & min mass
//HOWEVER I'd probably have to move the cell arrays to the heap in that case
#define MAX_HALO_CELL (int)1e5
//Max halo in entire box
//100 Mpc^3 box should have ~80,000,000 halos at M_min=1e7, z=6 so this should cover up to ~250 Mpc^3
#define MAX_HALO (int)1e9

#define MAX_DELTAC_FRAC (float)0.995

//NOTE: increasing interptable dimensions has a HUGE impact on performance
#define N_MASS_INTERP (int)200 // number of log-spaced mass bins in interpolation tables
#define N_DELTA_INTERP (int)100 // number of log-spaced overdensity bins in interpolation tables

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

int compare_float(float *a, float *b){
    return (int)(*a - *b);
}

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
    double rel_tol = FRACT_FLOAT_ERR*128; //<- relative tolerance
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
        LOG_ERROR("(function argument): lower_limit=%.3e upper_limit=%.3e rel_tol=%.3e result=%.3e error=%.3e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: growthf=%.3e M2=%.3e delta=%.3e sigma2=%.3e HMF=%.3d order=%.3e",growthf,M_filter,delta,sigma,HMF,n_order);
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

double EvaluatedNdMSpline(double x_in, double y_in){
    return gsl_spline2d_eval(Nhalo_spline,x_in,y_in,Nhalo_cond_acc,Nhalo_min_acc);
}

//This table is N(>M | M_in), the CDF of dNdM_conditional
//NOTE: Assumes you give it ymin as the minimum mass TODO: add another argument for Mmin
void initialise_dNdM_table(double xmin, double xmax, double ymin, double ymax, double growth1, double param, bool update){
    int nx,ny;
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

    double xa[nx], ya[ny], za[nx*ny];

    int i,j;
    //set up coordinate grids
    for(i=0;i<nx;i++) xa[i] = xmin + (xmax - xmin)*(i)/(nx-1);
    for(j=0;j<ny;j++) ya[j] = ymin + (ymax - ymin)*(j)/(ny-1);
    
    Nhalo_spline = gsl_spline2d_alloc(gsl_interp2d_bilinear, nx, ny);

    #pragma omp parallel num_threads(user_params_stoc->N_THREADS) private(i,j) firstprivate(delta,lnM_cond)
    {
        double x,y,buf;
        double norm;
        #pragma omp for
        for(i=0;i<nx;i++){
            x = xa[i];
            //set the condition
            if(update) lnM_cond = x;
            else delta = x;

            if(delta > MAX_DELTAC_FRAC*Deltac){
                //setting to zero for high delta 
                for(j=0;j<ny;j++)
                    gsl_interp2d_set(Nhalo_spline,za,i,j,0.);
                continue;
            }
            norm = IntegratedNdM(growth1,ymin,ymax,lnM_cond,delta,0,-1);
            gsl_interp2d_set(Nhalo_spline,za,i,0,0.); //set P(<Mmin) == 0.
            gsl_interp2d_set(Nhalo_spline,za,i,ny-1,norm); //set P(<Mmax) == 1.
            
            for(j=1;j<ny-1;j++){
                y = ya[j];
                //LOG_ULTRA_DEBUG("%d %d integrating cond %.2e from %.2e to %.2e, (d %.2e M %.2e D %.2e, n %.2e)",i,j,x,ymin,y,delta,lnM_cond,growth1, norm);
                if(lnM_cond <= y){
                    //setting to one guarantees samples at lower mass
                    //This fixes upper mass limits for the conditions
                    buf = norm;
                }
                else{
                    buf = IntegratedNdM(growth1, ymin, y, lnM_cond, delta, 0, -1);
                }
                //LOG_ULTRA_DEBUG("==> %.8e",buf);
                gsl_interp2d_set(Nhalo_spline,za,i,j,buf);
            }
        }

        //initialise and fill the interp table
        //The accelerators in GSL interpolators are not threadsafe, so we need one per thread.
        //Since it's not super important which thread has which accelerator, just that they
        //aren't using the same one at the same time, I think this is safe
        Nhalo_min_acc = gsl_interp_accel_alloc();
        Nhalo_cond_acc = gsl_interp_accel_alloc();
    }
    gsl_status = gsl_spline2d_init(Nhalo_spline, xa, ya, za, nx, ny);
    GSL_ERROR(gsl_status);

    LOG_DEBUG("Done.");
}

void free_dNdM_table(){
    gsl_spline2d_free(Nhalo_spline);

    #pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        gsl_interp_accel_free(Nhalo_cond_acc);
        gsl_interp_accel_free(Nhalo_min_acc);
    }
}

//since threads are cell-based, each thread should get a spline
gsl_spline *combined_spline;
#pragma omp threadprivate(combined_spline)
gsl_interp_accel *combined_acc;
#pragma omp threadprivate(combined_acc)

//Cell alloc/free should be called in a parallel region
void alloc_cell_cdf(){
    combined_spline = gsl_spline_alloc(gsl_interp_linear, N_MASS_INTERP);
    combined_acc = gsl_interp_accel_alloc();
}

void free_cell_cdf(){
    gsl_spline_free(combined_spline);
    gsl_interp_accel_free(combined_acc);
}

double sample_dndM_inverse(gsl_rng * rng){
    double p_in = gsl_rng_uniform(rng);

    return gsl_spline_eval(combined_spline,p_in,combined_acc);
}

//Makes a composite halo CDF from a list of halo masses at a certain delta
//returns the total amount of mass in these halos expected to be in progenitors above Mmin
double make_cell_cmf(double growth_out, double delta, double lnMmin, double lnMmax, float * M_in, int n_desc, bool update){
    //OPTION 1:
        //for each halo in the input:
            //Evauluate CDF(M_out | M_in) for a range of M_out
                //Since I can't do an integral for each halo, I need a 2D CDF table pre-calculated for the snapshot
            //add the results to a 1D interpolation table, weighted by mass
        //Normalise to max==1
        //this approach is probably slower, and involves the construction of another table

    double cell_cdf[N_MASS_INTERP];
    double mass_ax[N_MASS_INTERP];
    double M_coll;
    double dummy;
    int i,j;
    int gsl_status;

    for(j=0;j<N_MASS_INTERP;j++){
        cell_cdf[j] = 0.;
        mass_ax[j] = lnMmin + (lnMmax - lnMmin)*(j)/(N_MASS_INTERP-1);
    }

    LOG_ULTRA_DEBUG("Making cell cmf for D %.2e d %.2e M [%.2e %.2e] n %d u %d, M %.2e",growth_out,delta,lnMmin,lnMmax,n_desc,update, M_in[0]);

    double nh_buf, frac, lnM_desc, sigma_max, tbl_arg, M_coll_d, norm;
    //reuse the same spline here but with one descendant of the whole cell
    if(!update){
        n_desc = 1;
        //this overwrites the input array, be careful not to reuse M_in afterwards
        //This is only done on the initial grid sample so it shouldn't ever matter
        M_in[0] = exp(lnMmax);
        tbl_arg = delta;
    }

    for(i=0;i<n_desc;i++){
        lnM_desc = log(M_in[i]);
        //sometimes descendants of >deltac cells go above the max by float errors
        if(lnM_desc > lnMmax)lnM_desc = lnMmax;
        sigma_max = EvaluateSigma(lnM_desc,&dummy);
        //TODO: does this need the ps_ratio?
        //we also return the total collapsed mass to pass to the sampler
        if(update) tbl_arg = lnM_desc;
        //norm = EvaluatedNdMSpline(tbl_arg,mass_ax[N_MASS_INTERP-1]);
        frac = EvaluateFgtrM(growth_out,lnMmin,delta,sigma_max);
        M_coll_d = M_in[i] * frac;
        M_coll += M_coll_d;

        for(j=1;j<N_MASS_INTERP;j++){
            nh_buf = EvaluatedNdMSpline(tbl_arg,mass_ax[j]);
            //We treat each descendant as a portion of the cell mass with a certain CMF,
            //With a mass equal to the expected fraction above the resolution mass
            //So we weight the sum by the descendant mass fraction we expect to end up
            //in halos above our minimum.
            cell_cdf[j] += nh_buf * M_coll_d; //THE NORMALISATION IS ON N(<M) NOT FCOLL, NOT N(M)
            //LOG_ULTRA_DEBUG("C %.2e M %.2e nh %.8e",tbl_arg,mass_ax[j],nh_buf);
        }
    }

    //This happens at the end of a cell's history where some small halos go to none
    //It will be ignored in the sampling
    if (M_coll < exp(lnMmin)){
        return 0.;
    }

    //normalise to 1 and deal with floating point errors which break monotonicity
    cell_cdf[0] = 0.;
    int above_one = 0;
    norm = cell_cdf[N_MASS_INTERP-1];
    for(j=0;j<N_MASS_INTERP;j++){
        cell_cdf[j] /= norm;
        //there is a plateau at 1 usually where we hit float precision errors
        //setting each value to ALMOST 1, above 1, while still retaining the monotonicity
        //effectively ignores masses above the first occurrence (which should be the condition), since p ALWAYS <=1
        //IGNORES MASSES WITH PROBABILITIES <~ 1e-5 TODO: check the effects of this / make the sigma tables double precision
        if((above_one || mass_ax[j] > lnM_desc || cell_cdf[j] > 1 - FRACT_FLOAT_ERR*32)){
            cell_cdf[j] = 1. + above_one*j*FRACT_FLOAT_ERR;
            above_one = 1;
        }
        //LOG_ULTRA_DEBUG("p %.12e M %.2e j %d n %.2e",cell_cdf[j],mass_ax[j],j,norm);
    }
    //reverse x and y for the inverse CDF
    gsl_status = gsl_spline_init(combined_spline, cell_cdf, mass_ax, N_MASS_INTERP);
    
    if(gsl_status!=0) {
        LOG_ERROR("gsl spline error occured! M_coll %.3e",M_coll);
        #pragma omp critical
        {
            for(j=0;j<N_MASS_INTERP;j++){
                LOG_ERROR("p %.12e M %.2e j %d n %.2e, Mlastnh = %.2e",cell_cdf[j],mass_ax[j],j,EvaluatedNdMSpline(tbl_arg,mass_ax[j]));
            }
        }
        GSL_ERROR(gsl_status);
    }
    //Throw(ValueError);
    //LOG_ULTRA_DEBUG("Done.");

    //OPTION 2:
        //for each halo in the input:
            //Evaluate dNdM(M_out | M_in) for a range of M_out
            //add + weight results
        //integrate the TOTAL PDF
        //normalise to max==1
        //This approach may be faster If i can figure out a way to avoid the actual integral
        //also might have issues with high-delta spikes not being added correctly

    //OPTION 3:
        //bin by mass, construct by N_in_bin * PDF_row
        //This will eventually allow mini-halo addition, be fast and memory efficient
        //but the binning by mass makes high-delta sampling strange due to the narrow spike
        //this might be okay now since we are summing PDFs in a cell
        //but smooth accretion will be difficult without fine binning

        //There also might be a way to define the dNdM table to integrate over the bins N(Mp) = INT dMp INT dMd N(Mp|Md) N(Md)

    return M_coll;
    
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
        Mmin = astro_params->M_TURN / 12;
    }
    else {
        if(flag_options->M_MIN_in_Mass) {
            Mmin = astro_params->M_TURN / 12;
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
                        , int seed, float redshift, struct PerturbHaloField *halos){
    Broadcast_struct_global_STOC(user_params,cosmo_params,astro_params,flag_options);
    //allocate space for the properties
    int nhalos = halos->n_halos;
    halos->stellar_masses = (float *) calloc(nhalos,sizeof(float));
    halos->halo_sfr = (float *) calloc(nhalos,sizeof(float));

    //set up the rng
    gsl_rng * rng_stoc[user_params->N_THREADS];
    seed_rng_threads(rng_stoc,seed); 

    LOG_DEBUG("adding stars to %d halos",nhalos);

    //loop through the halos and assign properties
    int i;
    //TODO: update buffer when adding halo properties, make a #define or option with the number
    float buf[2];
#pragma omp parallel for private(buf)
    for(i=0;i<nhalos;i++){
        add_halo_properties(rng_stoc[omp_get_thread_num()], halos->halo_masses[i], redshift, buf);
        halos->stellar_masses[i] = buf[0];
        halos->halo_sfr[i] = buf[1];
        if(i<30) LOG_ULTRA_DEBUG("Halo %d, sm = %.3e, sfr = %.3e",i,buf[0],buf[1]);
    }

    free_rng_threads(rng_stoc);
    return 0;
}

/* Creates a realisation of halo properties by sampling the halo mass function and 
 * conditional property PDFs, the number of halos is poisson sampled from the integrated CMF*/
int stoc_halo_sample(double growth_out, double growth_in, double M_min, double M_max, double delta, double ps_ratio, bool update, gsl_rng * rng, int *n_halo_out, float *M_out){
    //delta check, if delta > deltacrit, make one big halo, if <-1 make no halos
    //both of these are possible with Lagrangian linear evolution
    if(delta > Deltac){
        *n_halo_out = 1;
        M_out[0] = M_max;
        return 0;
    }
    if(delta < -1){
        *n_halo_out = 0;
        return 0;
    }
    double nh_buf,hm_sample,tbl_arg;
    double lnMmin = log(M_min);
    double lnMmax = log(M_max);
    double M_prog = 0;
    int halo_count = 0;
    int ii, nh;

    //setup the condition
    if(update){
        delta = Deltac * growth_out / growth_in;
        tbl_arg = lnMmax;
    }
    else{
        tbl_arg = delta;
    }

    LOG_ULTRA_DEBUG("Condition M = %.2e, Mmin = %.2e, delta = %.2f",M_max,M_min,delta);

    //constants to go from integral(1/M dFcol/dlogM) dlogM to integral(dNdlogM) dlogM
    //here, M_max = volume / Rhocrit / OMm
    double n_factor = M_max / ps_ratio / sqrt(2.*PI);
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        nh_buf = EvaluatedNdMSpline(tbl_arg,lnMmax); //MMax set to cell size, delta given by redshift
    }
    else{
        nh_buf = IntegratedNdM(growth_out, lnMmin, lnMmax, lnMmax, delta, 0, -1);
    }

    nh_buf = nh_buf * n_factor;

    //sample poisson for stochastic halo number
    nh = gsl_ran_poisson(rng,nh_buf);
    for(ii=0;ii<nh;ii++){
        hm_sample = sample_dndM_inverse(rng);

        hm_sample = exp(hm_sample);
        M_out[halo_count++] = hm_sample;
        M_prog += hm_sample;
    }

    *n_halo_out = halo_count;
    LOG_ULTRA_DEBUG("Got %d (%.2e) halos mass %.2e",halo_count,nh_buf,M_prog);
    return 0;
}

/* Creates a realisation of halo properties by sampling the halo mass function and 
 * conditional property PDFs, Sampling is done until there is no more mass in the condition
 * Stochasticity is ignored below a certain mass threshold*/
int stoc_mass_sample(double growth_out, double M_min, double M_max, double delta, double M_available, bool update, gsl_rng * rng, int *n_halo_out, float *M_out){
    double tbl_arg;

    //lnMmin only used for sampling, apply factor here
    double lnMmin = log(M_min);
    double lnMmax = log(M_max);

    //There are some debug cases where I make a halo population larger than the cell
    //TODO: shift M_max checks to make_cell_cmf
    // if(M_available > M_max){
    //     LOG_ERROR("M_available %.3e > M_max %.3e",M_available,M_max);
    //     Throw(ValueError);
    // }

    if(M_available <= M_min){
        *n_halo_out = 0;
        return 0;
    }

    LOG_ULTRA_DEBUG("Condition M = %.2e (%.2e), Mmin = %.2e, delta = %.2f",M_max,M_available,M_min,delta);
    
    int n_halo_sampled=0;
    double M_prog=0;
    double M_sample;

    while(M_prog < M_available){
        M_sample = sample_dndM_inverse(rng);
        M_sample = exp(M_sample);

        M_out[n_halo_sampled++] = M_sample;
        M_prog += M_sample;
        //LOG_ULTRA_DEBUG("Sampled %.3e | %.3e",M_sample,M_prog);
    }
    
    //int random_idx;
    //bool running = true;
    /*while(running || fabs(M_prog - M_available) > M_min){
        if(M_prog - M_available > 0){
            do {
                random_idx = gsl_rng_uniform_int(rng,n_halo_sampled);
            } while(M_out[random_idx] == 0);
            M_prog -= M_out[random_idx];
            //LOG_ULTRA_DEBUG("removed %.3e | %.3e",M_out[random_idx],M_prog);
            M_out[random_idx] = 0;
            //running = false;
        }
        else{
            M_sample = sample_dndM_inverse(rng);
            M_sample = exp(M_sample);

            M_out[n_halo_sampled++] = M_sample;
            M_prog += M_sample;
            //LOG_ULTRA_DEBUG("Sampled %.3e | %.3e",M_sample,M_prog);
        }
    }*/
    //cap last halo
    M_out[n_halo_sampled - 1] -= M_prog - M_available;
    if(M_out[n_halo_sampled - 1] < M_min){
        //This is so the other functions don't read the halo for property addition etc
        n_halo_sampled--;
    }
    
    LOG_ULTRA_DEBUG("Got %d halos mass %.2e (exp. %.2e) %.2f",n_halo_sampled,M_prog,M_available,M_prog/M_available - 1);
    *n_halo_out = n_halo_sampled;
    return 0;
}

int stoc_sample(double growth_out, double M_min, double M_max, double delta, double M_available, bool update, gsl_rng * rng, int *n_halo_out, float *M_out){
    return stoc_mass_sample(growth_out, M_min, M_max, delta, M_available, update, rng, n_halo_out, M_out);
    //return stoc_halo_sample(growth_out, growth_in, M_min, M_max, delta, ps_ratio, update, rng, n_halo_out, M_out);
}

// will have to add properties here and output grids, instead of in perturbed
int build_halo_cats(gsl_rng **rng_arr, double redshift, float *dens_field, int *n_halo_out, int *halo_coords, float *halo_masses, float *stellar_masses, float *halo_sfr){    
    double growthf = dicke(redshift);
    int lo_dim = user_params_stoc->HII_DIM;
    int hi_dim = user_params_stoc->DIM;
    double boxlen = user_params_stoc->BOX_LEN;
    //cell size for smoothing / CMF calculation
    double cell_size_R = boxlen / lo_dim * L_FACTOR;
    double ps_ratio = 1.;

    double Mmax = RtoM(cell_size_R);
    double lnMmax = log(Mmax);
    double Mmin;

    Mmin = minimum_source_mass(redshift,astro_params_stoc,flag_options_stoc);
    double lnMmin = log(Mmin);

    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,Mmax);
    }
    initialise_dNdM_table(-1, Deltac, lnMmin, lnMmax, growthf, lnMmax, false);


    //Since the conditional MF is extended press-schecter, we rescale by a factor equal to the ratio of the collapsed fractions (n_order == 1) of the UMF
    if(user_params_stoc->HMF!=0){
        ps_ratio = (IntegratedNdM(growthf,lnMmin,lnMmax,lnMmax,0,1,0) 
            / IntegratedNdM(growthf,lnMmin,lnMmax,lnMmax,0,1,user_params_stoc->HMF));
    }

    LOG_DEBUG("Beginning stochastic halo sampling on %d ^3 grid",lo_dim);
    LOG_DEBUG("z = %f, Mmin = %e, Mmax = %e,volume = %.3e, R = %.3e D = %.3e",redshift,Mmin,Mmax,Mmax/RHOcrit/cosmo_params_stoc->OMm,cell_size_R,growthf);
    
    //shared halo count
    int counter = 0;

#pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        //PRIVATE VARIABLES
        int x,y,z,i;
        int threadnum = omp_get_thread_num();
        //buffers per cell
        float hm_buf[MAX_HALO_CELL];
        int nh_buf=0;
        double delta;
        double M_coll;
        float prop_buf[2];
        float dummy_M[1];
        int crd_hi[3];

        //debug printing
        int print_counter = 0;

        //allocation of cell cdf in parallel
        alloc_cell_cdf();

#pragma omp for
        for (x=0; x<lo_dim; x++){
            for (y=0; y<lo_dim; y++){
                for (z=0; z<lo_dim; z++){
                    //cell_hm = 0;
                    delta = (double)dens_field[HII_R_INDEX(x,y,z)] * growthf;
                    LOG_ULTRA_DEBUG("Starting sample %d (%d) with delta = %.2f cell, from %.2e to %.2e | %d"
                                    ,x*lo_dim*lo_dim + y*lo_dim + z,lo_dim*lo_dim*lo_dim,delta,Mmin,Mmax,counter);

                    //delta check, if delta > deltacrit, make one big halo, if <-1 make no halos
                    //both of these are possible with Lagrangian linear evolution
                    if(delta <= -1) continue;
                    
                    if(delta > MAX_DELTAC_FRAC*Deltac){
                        nh_buf = 1;
                        hm_buf[0] = Mmax;
                    }
                    else{
                        M_coll = make_cell_cmf(growthf, delta, lnMmin, lnMmax, dummy_M, 1, false);
                        if(M_coll < Mmin) continue;
                        stoc_sample(growthf, Mmin, Mmax, delta, M_coll/ps_ratio, 0, rng_arr[threadnum], &nh_buf, hm_buf);
                    }
                    //output total halo number, catalogues of masses and positions
                    for(i=0;i<nh_buf;i++){
                        //the correction in stoc_mass_sample can lead to ONE halo under the minimum which is not counted
                        if(hm_buf[i]<Mmin)continue;

                        add_halo_properties(rng_arr[threadnum], hm_buf[i], redshift, prop_buf);

                        place_on_hires_grid(x,y,z,crd_hi,rng_arr[threadnum]);

                        //fill in arrays now, this should be quick compared to the sampling so critical shouldn't slow this down much
                        #pragma omp critical
                        {
                            halo_masses[counter] = hm_buf[i];
                            stellar_masses[counter] = prop_buf[0];
                            halo_sfr[counter] = prop_buf[1];
                            halo_coords[0 + 3*counter] = crd_hi[0];
                            halo_coords[1 + 3*counter] = crd_hi[1];
                            halo_coords[2 + 3*counter] = crd_hi[2];
                            counter++;
                        }
                    }
                }
            }
        }
        free_cell_cdf();
    }
    *n_halo_out = counter;
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
    }
    free_dNdM_table();

    return 0;
}
//updates halo masses (going increasing redshift, z_prev > z)
//TODO: I don't think I need ps_ratio here since it's done in the first sample, but double check
int halo_update(gsl_rng ** rng_arr, double z_in, double z_out, int nhalo_in, int *coords_in, float *halo_in, float *stellar_in, float *sfr_in,
                 int *nhalo_out, int *coords_out, float *halo_out, float *stellar_out, float *sfr_out){

    if(z_in >= z_out){
        LOG_ERROR("halo update must go backwards in time!!! z_in = %.1f, z_out = %.1f",z_in,z_out);
        Throw(ValueError);
    }
    if(nhalo_in == 0){
        LOG_DEBUG("No halos to update, continuing...");
        *nhalo_out = 0;
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

    double Mmin = minimum_source_mass(z_out,astro_params_stoc,flag_options_stoc);
    double lnMmin = log(Mmin);

    double delta = Deltac * growth_out / growth_in; //crit density at z_in evolved to z_out

    init_ps();
    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        initialiseSigmaMInterpTable(Mmin,Mmax);
    }
    initialise_dNdM_table(lnMmin, lnMmax, lnMmin, lnMmax, growth_out, growth_in, true);

    LOG_DEBUG("Updating halo cat: z_in = %f, z_out = %f (d = %f), n_in = %d  Mmin = %e",z_in,z_out,delta,nhalo_in,Mmin);

    int count = 0;
    int nohalo_count=0;

#pragma omp parallel num_threads(user_params_stoc->N_THREADS)
    {
        //allocate halo buffer, one halo splitting into >1000 in one step would be crazy I think
        float prog_buf[MAX_HALO_CELL];
        float halo_cell[MAX_HALO_CELL];
        float sm_cell[MAX_HALO_CELL];
        float sfr_cell[MAX_HALO_CELL];
        int n_prog, n_desc;
        double M_coll;
        double M_cell_desc;
        
        float propbuf_in[2];
        float propbuf_out[2];

        int crd_hi[3];
        int threadnum = omp_get_thread_num();
        int desc_idx;
        //float M2,sm2,sfr2;
        int ii,jj,x,y,z;

        alloc_cell_cdf();

#pragma omp for
        for (x=0; x<lo_dim; x++){
            for (y=0; y<lo_dim; y++){
                for (z=0; z<lo_dim; z++){
                    n_desc = 0;
                    M_cell_desc = 0;
                    
                    LOG_ULTRA_DEBUG("Cell %d of %d",x*lo_dim*lo_dim + y*lo_dim + z,lo_dim*lo_dim*lo_dim);
                    for(ii=0;ii<nhalo_in;ii++){
                        //get halo information for the cell
                        if((int)((double)coords_in[3*ii+0] * (double)lo_dim / (double)hi_dim) == x &&
                            (int)((double)coords_in[3*ii+1] * (double)lo_dim / (double)hi_dim) == y &&
                            (int)((double)coords_in[3*ii+2] * (double)lo_dim / (double)hi_dim) == z){
                            sm_cell[n_desc] = stellar_in[ii];
                            sfr_cell[n_desc] = sfr_in[ii];
                            halo_cell[n_desc++] = halo_in[ii];
                            M_cell_desc += halo_in[ii];
                            //LOG_ULTRA_DEBUG("Halo %d [%.3e %.3e %.3e]",n_desc,halo_in[ii],stellar_in[ii],sfr_in[ii]);
                        }
                    }

                    if(n_desc==0){
                        nohalo_count++;
                        continue;
                    }

                    //NOTE:MAXIMUM HERE (TO BUILD TABLES) IS THE CELL MASS
                    M_coll = make_cell_cmf(growth_out,delta,lnMmin,lnMmax,halo_cell,n_desc,true);

                    //don't bother with the rest if we have no mass
                    if(M_coll < Mmin) continue;

                    //sort descendant list
                    qsort(halo_cell,n_desc,sizeof(float),compare_float);
                    
                    //find progenitor halos by sampling halo CMF                    
                    //NOTE: MAXIMUM HERE (TO LIMIT PROGENITOR MASS) IS THE DESCENDANT MASS
                    //The assumption is that the expected fraction of the progenitor
                    stoc_sample(growth_out,Mmin,M_cell_desc,delta,M_coll,true,rng_arr[threadnum],&n_prog,prog_buf);

                    //sort progenitor list
                    qsort(prog_buf,n_prog,sizeof(float),compare_float);

                    //place progenitors in local list
                    for(jj=0;jj<n_prog;jj++){
                        //the correction in stoc_mass_sample can lead to ONE halo under the minimum which is not counted
                        if(prog_buf[jj]<Mmin)continue;
                        propbuf_in[0] = sm_cell[jj];
                        propbuf_in[1] = sfr_cell[jj];

                        //on the rare occasion that n_prog > n_desc, we wrap around the desc list
                        desc_idx = jj % n_desc;
                        update_halo_properties(rng_arr[threadnum], z_out, z_in, prog_buf[jj], halo_cell[desc_idx], propbuf_in, propbuf_out);

                        //TODO: AFTER MATCHING, REMOVE THIS AND PLACE WITH PROGENITOR
                        place_on_hires_grid(x,y,z,crd_hi,rng_arr[threadnum]);
                        
                        #pragma omp critical
                        {
                            halo_out[count] = prog_buf[jj];

                            //the halos should already be on the hires grid
                            coords_out[3*count + 0] = crd_hi[0];
                            coords_out[3*count + 1] = crd_hi[1];
                            coords_out[3*count + 2] = crd_hi[2];
                            
                            stellar_out[count] = propbuf_out[0];
                            sfr_out[count] = propbuf_out[1];
                            count++;
                        }
                    }
                }
            }
        }
        free_cell_cdf();
    }

    if(user_params_stoc->USE_INTERPOLATION_TABLES){
        freeSigmaMInterpTable();
    }

    free_dNdM_table();

    if(nohalo_count > 0){
        LOG_DEBUG("%d Cells had no halos",nohalo_count);
    }

    *nhalo_out = count;
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

    int * halo_coords = (int*)calloc(MAX_HALO,3*sizeof(int));
    float * halo_masses = (float*)calloc(MAX_HALO,sizeof(float));
    float * stellar_masses = (float*)calloc(MAX_HALO,sizeof(float));
    float * halo_sfr = (float*)calloc(MAX_HALO,sizeof(float));

    //set up the rng
    gsl_rng * rng_stoc[user_params->N_THREADS];
    seed_rng_threads(rng_stoc,seed);

    //fill the hmf to possibly avoid deallocation issues:
    //TODO: actually fill the HMF in the sampling functions
    init_hmf(halos);
    if(halos->first_box)
        init_hmf(halos_prev);

    //Fill them
    if(halos->first_box){
        LOG_DEBUG("building first halo field at z=%.1f", redshift);
        build_halo_cats(rng_stoc,(double)redshift,dens_field,&n_halo_stoc,halo_coords,halo_masses,stellar_masses,halo_sfr);
    }
    else{
        LOG_DEBUG("updating halo field from z=%.1f to z=%.1f | %d", redshift_prev,redshift,halos->n_halos);
        halo_update(rng_stoc,redshift_prev,redshift,halos_prev->n_halos,halos_prev->halo_coords,halos_prev->halo_masses,
                    halos_prev->stellar_masses,halos_prev->halo_sfr,&n_halo_stoc,halo_coords,halo_masses,stellar_masses,halo_sfr);
    }

    LOG_DEBUG("Found %d Halos", n_halo_stoc);
    //trim buffers to the correct number of halos
    if(n_halo_stoc > 0){
        halo_coords = (int*) realloc(halo_coords,sizeof(int)*3*n_halo_stoc);
        halo_masses = (float*) realloc(halo_masses,sizeof(float)*n_halo_stoc);
        stellar_masses = (float*) realloc(stellar_masses,sizeof(float)*n_halo_stoc);
        halo_sfr = (float*) realloc(halo_sfr,sizeof(float)*n_halo_stoc);
    }
    else{
        //one dummy entry for saving
        halo_coords = (int*) realloc(halo_coords,sizeof(int)*3);
        halo_masses = (float*) realloc(halo_masses,sizeof(float));
        stellar_masses = (float*) realloc(stellar_masses,sizeof(float));
        halo_sfr = (float*) realloc(halo_sfr,sizeof(float));
    }

    //assign the pointers to the structs
    halos->n_halos = n_halo_stoc;
    halos->halo_masses = halo_masses;
    halos->halo_coords = halo_coords;
    halos->stellar_masses = stellar_masses;
    halos->halo_sfr = halo_sfr;

    LOG_DEBUG("First few Masses:  %11.3e %11.3e %11.3e",halo_masses[0],halo_masses[1],halo_masses[2]);
    LOG_DEBUG("First few Stellar: %11.3e %11.3e %11.3e",stellar_masses[0],stellar_masses[1],stellar_masses[2]);
    LOG_DEBUG("First few SFR:     %11.3e %11.3e %11.3e",halo_sfr[0],halo_sfr[1],halo_sfr[2]);

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
                double mass, wstar, sfr;
#pragma omp for reduction(+:hm_avg,sm_avg,sfr_avg)
                for(i=0;i<HII_TOT_NUM_PIXELS;i++){
                    dens = perturbed_field->density[i];

                    //ignore very low density
                    if(dens <= -1){
                        mass = 0.;
                        wstar = 0.;
                        sfr = 0.;
                    }
                    //turn into one large halo if we exceed the critical
                    //Since these are perturbed (Eulerian) grids, I use the total cell mass (1+dens)
                    else if(dens>=0.99*Deltac){                  
                        mass = M_max * (1+dens);
                        wstar = M_max * (1+dens) * cosmo_params->OMb / cosmo_params->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) * norm_esc * pow(M_max*(1+dens)/1e10,alpha_esc);
                        sfr = M_max * (1+dens) * cosmo_params->OMb / cosmo_params->OMm * norm_star * pow(M_max*(1+dens)/1e10,alpha_star) / t_star / t_hubble(redshift);
                    }
                    else{
                        //calling IntegratedNdM with star and SFR need special care for the f*/fesc clipping, and calling NionConditionalM for mass includes duty cycle
                        //neither of which I want
                        mass = IntegratedNdM(growth_z,lnMmin,lnMmax,lnMmax,dens,1,-1) * prefactor_mass * (1+dens);

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

        //gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

        //gsl_rng_set(rseed, random_seed);
        double test,dummy;
        int err=0;
        int i;

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
            if(type<3 || type == 6) Mmax = condition; //for CMF returns we want the max to be the condition
        }
        //Here the condition is a cell of a given density, the volume/mass is given by the grid parameters
        //below build_halo_cats(), the delta at z_out is expected, above, the IC (z=0) delta is expected.
        //we pass in the IC delta (so we can compare populations across redshifts) and adjust here
        else{
            delta = condition;
        }

        double lnMmin = log(Mmin);
        double lnMmax = log(Mmax);

        LOG_DEBUG("TEST FUNCTION: type = %d up %d, z = (%.2f,%.2f), Mmin = %e, Mmax = %e, R = %.2e (%.2e), delta = %.2f, M(%d)=[%.2e,%.2e,%.2e...]",type,update,z_out,z_in,Mmin,Mmax,R,volume,delta,n_mass,M[0],M[1],M[2]);

        //don't do anything for delta outside range               
        if(delta > Deltac || delta < -1){
            LOG_ERROR("delta of %f is out of bounds",delta);
            Throw(ValueError);
        }

        if(type != 5){
            init_ps();
            if(user_params_stoc->USE_INTERPOLATION_TABLES){
                initialiseSigmaMInterpTable(Mmin,Mmax);
            }
            //we use these tables only for some functions
            if(type == 2 || type == 3 || type == 4 || type == 6){
                if(update){
                    initialise_dNdM_table(lnMmin, lnMmax, lnMmin, lnMmax, growth_out, growth_in, true);
                }
                else{
                    initialise_dNdM_table(-1, Deltac, lnMmin, lnMmax, growth_out, lnMmax, false);
                }
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
                if(M[i] < Mmin || M[i] > Mmax){
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
            //intregrate CMF -> N_halos in many condidions
            //TODO: make it possible to integrate UMFs
            //quick hack: condition gives n_order, seed ignores tables
            #pragma omp parallel for private(test,R,volume)
            for(i=0;i<n_mass;i++){
                if(update){
                    R = MtoR(M[i]);
                    volume = 4. / 3. * PI * R * R * R;     
                    if(user_params_stoc->USE_INTERPOLATION_TABLES && seed == 0) test = EvaluatedNdMSpline(log(M[i]),lnMmax);
                    else test = IntegratedNdM(growth_out,lnMmin,log(M[i]),log(M[i]),delta,condition,-1);
                }
                else{
                    if(user_params_stoc->USE_INTERPOLATION_TABLES && seed == 0) test = EvaluatedNdMSpline(M[i]*growth_out,lnMmax);
                    else test = IntegratedNdM(growth_out,lnMmin,lnMmax,lnMmax,M[i]*growth_out,condition,-1);
                }
                //conditional MF multiplied by a few factors
                result[i] = test * volume * (RHOcrit / sqrt(2.*PI) * cosmo_params_stoc->OMm);
            }
        }
        
        //halo catalogues from one cell, given M as cell descendant halos
        else if(type==3){
            float out_hm[MAX_HALO_CELL];
            int n_halo;
            double M_coll;
            #pragma omp single
            {
                alloc_cell_cdf();
                M_coll = make_cell_cmf(growth_out,delta,lnMmin,lnMmax,M,n_mass,update);
                
                stoc_sample(growth_out, Mmin, Mmax, delta, M_coll/ps_ratio, update, rng_stoc[0], &n_halo, out_hm);
                for(i=0;i<n_halo;i++){
                    result[1+i] = out_hm[i];
                }
                result[0] = n_halo;
                free_cell_cdf();
            }
        }

        //halo catalogues + cell sums from multiple identical cells, given M as cell descendant halos
        else if(type==4){
            int n_halo_tot=0;
            int n_cell = 10;
            
            int j;
            #pragma omp parallel num_threads(user_params->N_THREADS) private(j)
            {
                float out_hm[MAX_HALO_CELL];
                double M_coll;
                int n_halo;
                alloc_cell_cdf();
                //if !update, the masses are ignored, and the cell will have the given delta
                #pragma omp for
                for(j=0;j<n_cell;j++){
                    M_coll = make_cell_cmf(growth_out,delta,lnMmin,lnMmax,M,n_mass,update);
                    result[1+j] = 0.;
                    stoc_sample(growth_out, Mmin, Mmax, delta, M_coll/ps_ratio, update, rng_stoc[omp_get_thread_num()], &n_halo, out_hm);
                    for(i=0;i<n_halo;i++){
                        result[1+j] += out_hm[i];
                        #pragma omp critical
                        {
                            result[1+n_cell+(n_halo_tot++)] = out_hm[i];
                        }
                    }
                    LOG_ULTRA_DEBUG("Cell %d %d got %.2e Mass (exp %.2e) %.2f",j,n_halo,result[1+j],M_coll,result[1+j]/(M_coll) - 1);
                    result[1+j] = result[1+j] - M_coll; //excess mass in the cell
                }
                
                result[0] = n_halo_tot;
                free_cell_cdf();
            }
        }
        
        //halo catalogue from list of conditions (Mass for update, delta for !update)
        else if(type==5){
            float *halomass_out = calloc(MAX_HALO,sizeof(float));
            float *stellarmass_out = calloc(MAX_HALO,sizeof(float));
            float *sfr_out = calloc(MAX_HALO,sizeof(float));
            int *halocoords_out = calloc(MAX_HALO,3*sizeof(int));
            
            int *halocoords_in = calloc(MAX_HALO,3*sizeof(int));
            float *stellarmass_in = calloc(MAX_HALO,sizeof(float));
            float *sfr_in = calloc(MAX_HALO,sizeof(float));
            float *halomass_in = calloc(MAX_HALO,sizeof(float));

            int nhalo_out;

            if(update){
                //NOTE: using n_mass for n_conditions
                //a single coordinate is provided for each halo
                LOG_ULTRA_DEBUG("assigning input arrays w %d halos",n_mass);
                for(i=0;i<n_mass;i++){
                    //LOG_ULTRA_DEBUG("Reading %d (%d %d %d)...",i,n_mass + 3*i,n_mass + 3*i + 1,n_mass + 3*i + 2);
                    //LOG_ULTRA_DEBUG("M[%d] = %.3e",n_mass + 3*i + 2,M[n_mass + 3*i + 2]);
                    //LOG_ULTRA_DEBUG("coords_in[%d] = %.3e",3*i + 2,M[3*i + 2]);
                    halomass_in[i] = M[i];
                    halocoords_in[3*i+1] = M[n_mass + 3*i + 1];
                    halocoords_in[3*i+2] = M[n_mass + 3*i + 2];
                    halocoords_in[3*i] = M[n_mass + 3*i];
                }
                //LOG_ULTRA_DEBUG("Sampling...",n_mass);
                halo_update(rng_stoc, z_in, z_out, n_mass, halocoords_in, halomass_in, stellarmass_in, sfr_in,
                                    &nhalo_out, halocoords_out, halomass_out, stellarmass_out, sfr_out);
            }
            else{
                //NOTE: halomass_in is linear delta at z = redshift_out
                for(i=0;i<n_mass;i++){
                    halomass_in[i] = M[i] / growth_out;
                }
                build_halo_cats(rng_stoc, z_out, halomass_in, &nhalo_out, halocoords_out, halomass_out, stellarmass_out, sfr_out);
            }
            LOG_DEBUG("sampling done, %d halos, %.2e %.2e %.2e",nhalo_out,halomass_out[0],halomass_out[1],halomass_out[2]);

            result[0] = (double)nhalo_out;
            for(i=0;i<nhalo_out;i++){
                result[1+i] = (double)halomass_out[i];
                result[nhalo_out+1+3*i] = (double)(halocoords_out[3*i]);
                result[nhalo_out+2+3*i] = (double)(halocoords_out[3*i+1]);
                result[nhalo_out+3+3*i] = (double)(halocoords_out[3*i+2]);
            }
            free(sfr_in);
            free(halocoords_in);
            free(stellarmass_in);

            free(halomass_out);
            free(stellarmass_out);
            free(sfr_out);
            free(halocoords_out);
        }

        //return dNdM table result for M at a bunch of probabilities
        else if(type==6){
            double y_in,x_in;
            #pragma omp parallel for private(test,x_in,y_in)
            for(i=0;i<n_mass;i++){
                y_in = log(M[i]);
                if(y_in < lnMmin)
                    y_in = lnMmin;
                if(y_in > lnMmax)
                    y_in = lnMmax;

                x_in = update ? log(condition) : condition;
                test = EvaluatedNdMSpline(x_in,y_in);
                result[i] = test * Mmax / sqrt(2.*PI);
                LOG_ULTRA_DEBUG("dNdM table: x = %.6e, y = %.6e z = %.6e",x_in,y_in,test);
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
            if(type==2 || type==3 || type==4 || type == 6)
                free_dNdM_table();
        }
        free_rng_threads(rng_stoc);
    } //end of try

    Catch(status){
        return(status);
    }
    LOG_DEBUG("Done.");
    return(0);
}