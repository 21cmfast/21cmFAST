
/* This file contains integrals and integrands of the halo mass function for the source distributions */
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "cosmology.h"

#include "hmf.h"

#define EPS2 3.0e-11 //small number limit for GL integration

//Gauss-Legendre integration constants
#define NGL_INT 100 // 100
//These arrays hold the points and weights for the Gauss-Legendre integration routine
//(JD) Since these were always malloc'd one at a time with fixed length ~100, I've changed them to fixed-length arrays
static float xi_GL[NGL_INT+1], wi_GL[NGL_INT+1];
static float GL_limit[2] = {0};

//gets the (stellar/escape) fraction at a certain mass (in units of 1/normalisation at 1e10)
double get_frac_limit(double M, double norm, double alpha, double limit, bool mini){
    double pivot = mini ? 1e7 : 1e10;
    if ((alpha > 0. && M > limit) || (alpha < 0. && M < limit))
        return 1/norm;

    //if alpha is zero, this returns 1 as expected (note strict inequalities above)
    return pow(M/pivot,alpha);
}

double nion_fraction(double M, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,false);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,false);

    return Fstar * Fesc * exp(-M_turn_lower/M);
}

double nion_fraction_mini(double M, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double M_turn_upper = params.Mturn_upper;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,true);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,true);

    return Fstar * Fesc * exp(-M_turn_lower/M - M/M_turn_upper);
}

double conditional_mf(double growthf, double lnM, double delta, double sigma, int HMF){
    //dNdlnM = dfcoll/dM * M / M * constants
    if(HMF==0) {
        return dNdM_conditional_EPS(growthf,lnM,delta,sigma);
    }
    if(HMF==1) {
        return dNdM_conditional_ST(growthf,lnM,delta,sigma);
    }
    if(HMF==4) {
        return dNdlnM_conditional_Delos(growthf,lnM,delta,sigma);
    }
    //NOTE: Normalisation scaling is currently applied outside the integral, per condition
    //This will be the rescaled EPS CMF,
    return dNdM_conditional_EPS(growthf,lnM,delta,sigma);

}

double c_mf_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double growthf = params.growthf;
    double delta = params.delta; //the condition delta
    double sigma2 = params.sigma_cond;
    int HMF = params.HMF;

    return conditional_mf(growthf,lnM,delta,sigma2,HMF);
}

double c_fcoll_integrand(double lnM, void *param_struct){
    return exp(lnM) * c_mf_integrand(lnM,param_struct);
}

double c_nion_integrand(double lnM, void *param_struct){
    return nion_fraction(exp(lnM),param_struct) * exp(lnM) * c_mf_integrand(lnM,param_struct);
}

//The reason this is separated from the above is the second exponent
double c_nion_integrand_mini(double lnM, void *param_struct){

    return nion_fraction_mini(exp(lnM),param_struct) * exp(lnM) * c_mf_integrand(lnM,param_struct);
}

double unconditional_mf(double growthf, double lnM, double z, int HMF){
    //most of the UMFs are defined with M, but we integrate over lnM
    //NOTE: HMF > 4 or < 0 gets caught earlier, so unless some strange change is made this is fine
    if(HMF==0) {
        return dNdlnM_PS(growthf, lnM);
    }
    if(HMF==1) {
        return dNdlnM_st(growthf, lnM);
    }
    if(HMF==2) {
        return dNdlnM_WatsonFOF(growthf, lnM);
    }
    if(HMF==3) {
        return dNdlnM_WatsonFOF_z(z, growthf, lnM);
    }
    if(HMF==4) {
        return dNdlnM_Delos(growthf, lnM);
    }
    else{
        LOG_ERROR("Invalid HMF %d",HMF);
        Throw(ValueError);
    }
}

double u_mf_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double mf, m_factor;
    double growthf = params.growthf;
    double z = params.redshift;
    int HMF = params.HMF;

    return unconditional_mf(growthf,lnM,z,HMF);
}

double u_fcoll_integrand(double lnM, void *param_struct){
    return exp(lnM) * u_mf_integrand(lnM,param_struct);
}

double u_nion_integrand(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn = params.Mturn;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double M = exp(lnM);

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,false);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,false);

    return M * Fstar * Fesc * exp(-M_turn/M) * u_mf_integrand(lnM,param_struct);
}

//The reason this is separated from the above is the second exponent
double u_nion_integrand_mini(double lnM, void *param_struct){
    struct parameters_gsl_MF_integrals params = *(struct parameters_gsl_MF_integrals *)param_struct;
    double M_turn_lower = params.Mturn;
    double M_turn_upper = params.Mturn_upper;
    double f_starn = params.f_star_norm;
    double a_star = params.alpha_star;
    double f_escn = params.f_esc_norm;
    double a_esc = params.alpha_esc;
    double Mlim_star = params.Mlim_star;
    double Mlim_esc = params.Mlim_esc;

    double M = exp(lnM);

    double Fstar = get_frac_limit(M,f_starn,a_star,Mlim_star,true);
    double Fesc = get_frac_limit(M,f_escn,a_esc,Mlim_esc,true);

    return M * Fstar * Fesc * exp(-M_turn_lower/M - M/M_turn_upper) * u_mf_integrand(lnM,param_struct);
}

///// INTEGRATION ROUTINES BELOW /////

//TODO: make type enum for clarity (but cffi doesn't seem to like enum in 21cmFAST.h)
//NOTE: SFR is obtained from nion with alpha_esc==0 and f_esc==1
//Currently the scheme is to use negative numbers for conditionals, and (1,2,3,4) for (number,mass,n_ion,n_ion_mini)
double (*get_integrand_function(int type))(double,void*){
    if(type==1)
        return &u_mf_integrand; //Unondtional mass function integral
    if(type==2)
        return &u_fcoll_integrand; //Unconditional collapsed fraction integral
    if(type==3)
        return &u_nion_integrand; //Unconditional N_ion integral (two power-laws and one exponential)
    if(type==4)
        return &u_nion_integrand_mini; //Unconditional N_ion minihalo integral (two power-laws and two exponentials)
    if(type==-1)
        return &c_mf_integrand; //Conditional mass function integral
    if(type==-2)
        return &c_fcoll_integrand; //Conditional collapsed fraction integral
    if(type==-3)
        return &c_nion_integrand; //Conditional N_ion integral (two power-laws and one exponential)
    if(type==-4)
        return &c_nion_integrand_mini; //Conditional N_ion minihalo integral (two power-laws and two exponentials)

    LOG_ERROR("Invalid type %d for MF integral");
    Throw(ValueError);
}

//Integral of a CMF or UMF
//In future all MF integrals will go through here, simply selecting the integrand function from a switch
double IntegratedNdM_QAG(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    // double rel_tol = FRACT_FLOAT_ERR*128; //<- relative tolerance
    double rel_tol = 1e-3; //<- relative tolerance
    int w_size = 1000;
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (w_size);

    int status;
    F.function = get_integrand_function(type);
    F.params = &params;
    lower_limit = lnM_lo;
    upper_limit = lnM_hi;

    gsl_set_error_handler_off();
    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         w_size, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%.3e (%.3e) upper_limit=%.3e (%.3e) rel_tol=%.3e result=%.3e error=%.3e",lower_limit,exp(lower_limit),upper_limit,exp(upper_limit),rel_tol,result,error);
        LOG_ERROR("data: z=%.3e growthf=%.3e  HMF=%d type=%d ",params.redshift,params.growthf,params.HMF,type);
        LOG_ERROR("sigma=%.3e delta=%.3e",params.sigma_cond,params.delta);
        LOG_ERROR("Mturn_lo=%.3e f*=%.3e a*=%.3e Mlim*=%.3e",params.Mturn,params.f_star_norm,params.alpha_star,params.Mlim_star);
        LOG_ERROR("f_escn=%.3e a_esc=%.3e Mlim_esc=%.3e",params.f_esc_norm,params.alpha_esc,params.Mlim_esc);
        LOG_ERROR("Mturn_hi %.3e",params.Mturn_upper);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return result;
}

//calculates the weightings and the positions for any Gauss-Legendre quadrature.
void gauleg(float x1, float x2, float x[], float w[], int n)
//Given the lower and upper limits of integration x1 and x2, and given n, this routine returns arrays x[1..n] and w[1..n] of length n,
//containing the abscissas and weights of the Gauss- Legendre n-point quadrature formula.
{

    int m,j,i;
    double z1,z,xm,xl,pp,p3,p2,p1;

    m=(n+1)/2;
    xm=0.5*(x2+x1);
    xl=0.5*(x2-x1);
    for (i=1;i<=m;i++) {
        //High precision is a good idea for this routine.
        //The roots are symmetric in the interval, so we only have to find half of them.
        //Loop over the desired roots.

        z=cos(3.141592654*(i-0.25)/(n+0.5));

        //Starting with the above approximation to the ith root, we enter the main loop of refinement by Newton’s method.
        do {
            p1=1.0;
            p2=0.0;
            for (j=1;j<=n;j++) {
                //Loop up the recurrence relation to get the Legendre polynomial evaluated at z.
                p3=p2;
                p2=p1;
                p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
            }
            //p1 is now the desired Legendre polynomial. We next compute pp, its derivative, by a standard relation involving also p2,
            //the polynomial of one lower order.
            pp=n*(z*p1-p2)/(z*z-1.0);
            z1=z;
            z=z1-p1/pp;
        } while (fabs(z-z1) > EPS2);
        x[i]=xm-xl*z;
        x[n+1-i]=xm+xl*z;
        w[i]=2.0*xl/((1.0-z*z)*pp*pp);
        w[n+1-i]=w[i];
    }
}

//Specific initialistion for the global arrays
void initialise_GL(float lnM_Min, float lnM_Max){
    //don't redo if you don't have to
    if(lnM_Min == GL_limit[0] && lnM_Max == GL_limit[1])
        return;

    gauleg(lnM_Min,lnM_Max,xi_GL,wi_GL,NGL_INT);
    GL_limit[0] = lnM_Min;
    GL_limit[1] = lnM_Max;
}

//actually perform the GL integration
//NOTE: that the lnM limits are not used
double IntegratedNdM_GL(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type){
    int i;
    double integral = 0;
    if((float)lnM_lo != (float)GL_limit[0] || (float)lnM_hi != (float)GL_limit[1]){
        LOG_ERROR("Integral limits [%.8e %.8e] do not match Gauss Legendre limits [%.8e %.8e]!",exp(lnM_lo),exp(lnM_hi),GL_limit[0],GL_limit[1]);
        Throw(TableGenerationError);
    }

    for(i=1; i<(NGL_INT+1); i++){
        integral += wi_GL[i]*(get_integrand_function(type))(xi_GL[i],&params);
    }

    return integral;
}

#include <gsl/gsl_sf_gamma.h>
//JBM: Integral of a power-law times exponential for EPS: \int dnu nu^beta * exp(-nu/2)/sqrt(nu) from numin to infty.
double Fcollapprox(double numin, double beta){
//nu is deltacrit^2/sigma^2, corrected by delta(R) and sigma(R)
  double gg = gsl_sf_gamma_inc(0.5+beta,0.5*numin);
  return gg*pow(2,0.5+beta)*pow(2.0*PI,-0.5);
}

//This takes into account the last approximation in Munoz+22, where erfc (beta=0) is used
//NOTE: even though nu_condition is defined in the unconditional (no sigma_cond), here it
//  represents where nu_tilde == nu_condition (effectively a final pivot point)
//NOTE: This assumes numin < nucondition, otherise it fails
double Fcollapprox_condition(double numin, double nucondition, double beta){
    return (Fcollapprox(numin,beta) - Fcollapprox(nucondition,beta)) + Fcollapprox(nucondition,0.)*pow(nucondition,beta);
}

//This routine assumes sharp cutoffs for each turnover rather than exponential, assumes a triple power-law form for sigma(M)
//  and takes advantage of the fact that Gamma_inc(x,min) = integral_min^inf (t^(x-1)exp(-t)) dt which is satisfied for the HMF when the
//  above approximations are made
//Originally written by JBM within the GL integration before it was separated here and generalised to the other integrals
double MFIntegral_Approx(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type){
    //variables used in the calculation
    double lnM_higher, lnM_lower;

    double delta,sigma_c;
    double index_base;

    if(params.HMF != 0){
        LOG_ERROR("Approximate Fcoll is currently only implemented for EPS");
        LOG_ERROR("Ensure parameter input specifically to this function has HMF==0");
        Throw(TableGenerationError);
    }
    double growthf = params.growthf;
    if(type < 0){
        //we are a conditional mf
        delta = params.delta;
        sigma_c = params.sigma_cond;
    }
    else{
        //unconditional
        delta = 0.;
        sigma_c = 0.;
    }

    double lnM_lo_limit = lnM_lo;
    double lnM_hi_limit = lnM_hi;
    //(Speed): by passing in log(M_turnover) i can avoid these 2 log calls
    double lnMturn_l = log(params.Mturn);
    double lnMturn_u = log(params.Mturn_upper);
    //(Speed): LOG(MPIVOTn) can be pre-defined via macro
    double lnMp1 = log(MPIVOT1);
    double lnMp2 = log(MPIVOT2);

    //The below limit setting is done simply so that variables which do not conern particular integrals
    //      can be left undefined, rather than explicitly set to some value (0 or 1e20)
    //Mass and number integrals set the lower cutoff to the integral limit
    if(fabs(type) >= 3 && lnMturn_l > lnM_lo_limit)
        lnM_lo_limit = lnMturn_l;
    //non-minihalo integrals set the upper cutoff to the integral limit
    if(fabs(type) == 4 && lnMturn_u < lnM_hi_limit)
        lnM_hi_limit = lnMturn_u;

    //it is possible for the lower turnover (LW crit or reion feedback)
    //   to be higher than the upper limit (atomic limit) or the condition
    if(lnM_lo_limit >= lnM_hi_limit || EvaluateSigma(lnM_lo_limit) <= sigma_c){
        return 0.;
    }

    //n_ion or MINI
    if(fabs(type) >= 3)
        index_base = params.alpha_star + params.alpha_esc;
    //fcoll
    else if(fabs(type)==2)
        index_base = 0.;
    //nhalo
    else
        index_base = -1.;

    double delta_arg = pow((Deltac - delta)/growthf, 2);
    double beta1 = index_base * AINDEX1 * 0.5; //exponent for Fcollapprox for nu>nupivot1 (large M)
    double beta2 = index_base * AINDEX2 * 0.5; //exponent for Fcollapprox for nupivot2<nu<nupivot1 (small M)
    double beta3 = index_base * AINDEX3 * 0.5; //exponent for Fcollapprox for nu<nupivot2 (smallest M)

    // There are 5 nu(M) points of interest: the two power-law pivot points, the lower and upper integral limits
    // and the condition.
    //NOTE: Since sigma(M) is approximated as a power law, not (sigma(M)^2 - sigma_cond^2), this is not a simple gamma function.
    //  note especially which nu subtracts the condition sigma and not, see Appendix B of Munoz+22 (2110.13919)
    double sigma_pivot1 = EvaluateSigma(lnMp1);
    double sigma_pivot2 = EvaluateSigma(lnMp2);
    double sigma_lo_limit = EvaluateSigma(lnM_lo_limit);
    double sigma_hi_limit = EvaluateSigma(lnM_hi_limit);

    //These nu use the CMF delta (subtracted the condition delta), but not the condition sigma
    double nu_pivot1_umf = delta_arg / (sigma_pivot1*sigma_pivot1);
    double nu_pivot2_umf = delta_arg / (sigma_pivot2*sigma_pivot2);
    double nu_condition = delta_arg / (sigma_c*sigma_c);

    double nu_pivot1 = delta_arg / (sigma_pivot1*sigma_pivot1 - sigma_c*sigma_c);
    double nu_pivot2 = delta_arg / (sigma_pivot2*sigma_pivot2 - sigma_c*sigma_c);

    //These nu subtract the condition sigma as in the CMF
    double nu_lo_limit = delta_arg / (sigma_lo_limit*sigma_lo_limit - sigma_c*sigma_c);
    double nu_hi_limit = delta_arg / (sigma_hi_limit*sigma_hi_limit - sigma_c*sigma_c);

    double fcoll = 0.;

    //NOTES: For speed the minihalos ignore the condition mass limit (assuming nu_hi_limit(tilde) < nu_condition (no tilde))
    //    and never get into the high mass power law (nu_hi_limit < nu_pivot1 (both tilde))
    //ACGs ignore the upper mass limit (no upper turnover), both assume the condition is above the highest pivot
    if(fabs(type) == 4){
      // re-written for further speedups
      if (nu_hi_limit <= nu_pivot2){ //if both are below pivot2 don't bother adding and subtracting the high contribution
        fcoll += (Fcollapprox(nu_lo_limit,beta3))*pow(nu_pivot2_umf,-beta3);
        fcoll -= (Fcollapprox(nu_hi_limit,beta3))*pow(nu_pivot2_umf,-beta3);
      }
      else {
        fcoll -= (Fcollapprox(nu_hi_limit,beta2))*pow(nu_pivot1_umf,-beta2);
        if (nu_lo_limit > nu_pivot2){
            fcoll += (Fcollapprox(nu_lo_limit,beta2))*pow(nu_pivot1_umf,-beta2);
        }
        else {
            fcoll += (Fcollapprox(nu_pivot2,beta2))*pow(nu_pivot1_umf,-beta2);
            fcoll += (Fcollapprox(nu_lo_limit,beta3)-Fcollapprox(nu_pivot2,beta3) )*pow(nu_pivot2_umf,-beta3);
        }
      }
    }
    else{
        if(nu_lo_limit >= nu_condition){ //fully in the flat part of sigma(nu), M^alpha is nu-independent.
            // This is just an erfc, remembering that the conditional nu can be higher than the unconditional nu of the condition
            return Fcollapprox(nu_lo_limit,0.);
        }

        if(nu_lo_limit >= nu_pivot1){
            //We use the condition version wherever the nu range may intersect nu_condition (i.e beta1)
            fcoll += Fcollapprox_condition(nu_lo_limit,nu_condition,beta1)*pow(nu_pivot1_umf,-beta1);
        }
        else{
            fcoll += Fcollapprox_condition(nu_pivot1,nu_condition,beta1)*pow(nu_pivot1_umf,-beta1);
            if (nu_lo_limit > nu_pivot2){
                fcoll += (Fcollapprox(nu_lo_limit,beta2)-Fcollapprox(nu_pivot1,beta2))*pow(nu_pivot1_umf,-beta2);
            }
            else {
                fcoll += (Fcollapprox(nu_pivot2,beta2)-Fcollapprox(nu_pivot1,beta2) )*pow(nu_pivot1_umf,-beta2);
                fcoll += (Fcollapprox(nu_lo_limit,beta3)-Fcollapprox(nu_pivot2,beta3) )*pow(nu_pivot2_umf,-beta3);
            }
        }
    }

    if (fcoll<=0.0){
        LOG_DEBUG("Negative fcoll? fc=%.1le\n",fcoll);
        fcoll=1e-40;
    }
    return fcoll;
}

double IntegratedNdM(double lnM_lo, double lnM_hi, struct parameters_gsl_MF_integrals params, int type, int method){
    if(method==0 || (method==1 && params.delta > global_params.CRIT_DENS_TRANSITION))
        return IntegratedNdM_QAG(lnM_lo, lnM_hi, params, type);
    if(method==1)
        return IntegratedNdM_GL(lnM_lo, lnM_hi, params, type);
    if(method==2)
        return MFIntegral_Approx(lnM_lo, lnM_hi, params, type);
}

//Some wrappers over the integration functions for specific integrals//

/*
 FUNCTION FgtrM(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 */
double FgtrM(double z, double M){
    double del, sig;

    del = Deltac/dicke(z); //regular spherical collapse delta
    sig = sigma_z0(M);

    return splined_erfc(del / (sqrt(2)*sig));
}

/*
 FUNCTION FgtrM_wsigma(z, sigma_z0(M))
 Computes the fraction of mass contained in haloes with mass > M at redshift z.
 Requires sigma_z0(M) rather than M to make certain heating integrals faster
 */
double FgtrM_wsigma(double z, double sig){
    double del;

    del = Deltac/dicke(z); //regular spherical collapse delta

    return splined_erfc(del / (sqrt(2)*sig));
}

double Nhalo_General(double z, double lnM_min, double lnM_max){
    double lower_limit, upper_limit, growthf;

    growthf = dicke(z);
    struct parameters_gsl_MF_integrals integral_params = {
                .redshift = z,
                .growthf = growthf,
                .HMF = user_params_global->HMF,
    };
    return IntegratedNdM(lnM_min, lnM_max, integral_params, 1, 0) / (cosmo_params_global->OMm*RHOcrit);
}

double Fcoll_General(double z, double lnM_min, double lnM_max){
    double lower_limit, upper_limit, growthf;

    growthf = dicke(z);
    struct parameters_gsl_MF_integrals integral_params = {
                .redshift = z,
                .growthf = growthf,
                .HMF = user_params_global->HMF,
    };
    return IntegratedNdM(lnM_min, lnM_max, integral_params, 2, 0) / (cosmo_params_global->OMm*RHOcrit);
}

double Nion_General(double z, double lnM_Min, double lnM_Max, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10,
                     double Fesc10, double Mlim_Fstar, double Mlim_Fesc){
    struct parameters_gsl_MF_integrals params = {
        .redshift = z,
        .growthf = dicke(z),
        .Mturn = MassTurnover,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar10,
        .f_esc_norm = Fesc10,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        .HMF = user_params_global->HMF,
    };
    return IntegratedNdM(lnM_Min,lnM_Max,params,3,0) / ((cosmo_params_global->OMm)*RHOcrit);
}

double Nion_General_MINI(double z, double lnM_Min, double lnM_Max, double MassTurnover, double MassTurnover_upper, double Alpha_star,
                         double Alpha_esc, double Fstar7_MINI, double Fesc7_MINI, double Mlim_Fstar, double Mlim_Fesc){
    struct parameters_gsl_MF_integrals params = {
        .redshift = z,
        .growthf = dicke(z),
        .Mturn = MassTurnover,
        .Mturn_upper = MassTurnover_upper,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar7_MINI,
        .f_esc_norm = Fesc7_MINI,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        .HMF = user_params_global->HMF,
    };
    return IntegratedNdM(lnM_Min,lnM_Max,params,4,0) / ((cosmo_params_global->OMm)*RHOcrit);
}

double Nhalo_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .HMF = user_params_global->HMF,
        .sigma_cond = sigma,
        .delta = delta,
    };

    if(delta <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 1 halo AT THE CONDITION MASS if delta is exceeded
    if(delta > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2)) //this limit is not ideal, but covers floating point errors when we set lnM2 == log(M_cond)
            return 1./M_cond;
        else
            return 0.;
    }

    return IntegratedNdM(lnM1,lnM2,params,-1, method);
}

double Mcoll_Conditional(double growthf, double lnM1, double lnM2, double M_cond, double sigma, double delta, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .HMF = user_params_global->HMF,
        .sigma_cond = sigma,
        .delta = delta,
    };

    if(delta <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 100% of mass AT THE CONDITION MASS if delta is exceeded
    if(delta > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2)) //this limit is not ideal, but covers floating point errors when we set lnM2 == log(M_cond)
            return 1.;
        else
            return 0.;
    }
    return IntegratedNdM(lnM1,lnM2,params,-2, method);
}

double Nion_ConditionalM_MINI(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
                            double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar7,
                            double Fesc7, double Mlim_Fstar, double Mlim_Fesc, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .Mturn = MassTurnover,
        .Mturn_upper = MassTurnover_upper,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar7,
        .f_esc_norm = Fesc7,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        .HMF = user_params_global->HMF,
        .sigma_cond = sigma2,
        .delta = delta2,
    };

    if(delta2 <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 1 halo at the condition mass if delta is exceeded
    //NOTE: this will almost always be zero, due to the upper turover,
    // however this replaces an integral so it won't be slow
    if(delta2 > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma2,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2)) //this limit is not ideal, but covers floating point errors when we set lnM2 == log(M_cond)
            return nion_fraction_mini(M_cond,&params); //NOTE: condition mass is used as if it were Lagrangian (no 1+delta)
        else
            return 0.;
    }

    //If we don't have a corresponding CMF, use EPS and normalise
    //NOTE: it's possible we may want to use another default
    if(params.HMF != 0 && params.HMF != 1 && params.HMF != 4)
        params.HMF = 0;

    return IntegratedNdM(lnM1,lnM2,params,-4,method);
}

double Nion_ConditionalM(double growthf, double lnM1, double lnM2, double M_cond, double sigma2, double delta2, double MassTurnover,
                        double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar,
                        double Mlim_Fesc, int method){
    struct parameters_gsl_MF_integrals params = {
        .growthf = growthf,
        .Mturn = MassTurnover,
        .alpha_star = Alpha_star,
        .alpha_esc = Alpha_esc,
        .f_star_norm = Fstar10,
        .f_esc_norm = Fesc10,
        .Mlim_star = Mlim_Fstar,
        .Mlim_esc = Mlim_Fesc,
        .HMF = user_params_global->HMF,
        .sigma_cond = sigma2,
        .delta = delta2,
    };

    if(delta2 <= -1. || lnM1 >= log(M_cond))
        return 0.;
    //return 1 halo at the condition mass if delta is exceeded
    if(delta2 > MAX_DELTAC_FRAC*get_delta_crit(params.HMF,sigma2,growthf)){
        if(M_cond*(1-FRACT_FLOAT_ERR) <= exp(lnM2))
            return nion_fraction(M_cond,&params); //NOTE: condition mass is used as if it were Lagrangian (no 1+delta)
        else
            return 0.;
    }

    //If we don't have a corresponding CMF, use EPS and normalise
    //NOTE: it's possible we may want to use another default
    if(params.HMF != 0 && params.HMF != 1 && params.HMF != 4)
        params.HMF = 0;

    return IntegratedNdM(lnM1,lnM2,params,-3, method);
}

float erfcc(float x)
{
    double t,q,ans;

    q=fabs(x);
    t=1.0/(1.0+0.5*q);
    ans=t*exp(-q*q-1.2655122+t*(1.0000237+t*(0.374092+t*(0.0967842+
                                                         t*(-0.1862881+t*(0.2788681+t*(-1.13520398+t*(1.4885159+
                                                                                                      t*(-0.82215223+t*0.17087277)))))))));
    return x >= 0.0 ? ans : 2.0-ans;
}

double splined_erfc(double x){
    if (x < 0){
        return 1.0;
    }

    //The Actual ERFC spline was removed a while ago, if we remake it it should be
    //  in interp_tables.c with the others as an RGTable1D
    return erfcc(x);
}


/*
 calculates the fraction of mass contained in haloes with mass > M at redshift z,
 in regions with a linear overdensity of del_bias, and standard deviation sig_bias
*/

//I wrote a version of FgtrM which takes the growth func instead of z for a bit of speed
double FgtrM_bias_fast(float growthf, float del_bias, float sig_small, float sig_large){
    double del, sig;
    if (sig_large > sig_small){ // biased region is smaller that halo!
        LOG_ERROR("Trying to compute FgtrM in region where M_min > M_max");
        Throw(ValueError);
    }
    //sometimes they are the same to float precision, where the M_condition ~ M_Min
    if(sig_large == sig_small){
        return 0.;
    }
    // del = Deltac/growthf - del_bias; //NOTE HERE DELTA EXTRAPOLATED TO z=0
    sig = sqrt(sig_small*sig_small - sig_large*sig_large);
    del = (Deltac - del_bias)/growthf;

    //if the density is above critical on this scale, it is collapsed
    //NOTE: should we allow del < 0??? We would need to change dfcolldz to prevent zero dfcoll
    // if(del < FRACT_FLOAT_ERR){
    //     return 1.;
    // }
    return splined_erfc(del / (sqrt(2)*sig));
}

/* Uses sigma parameters instead of Mass for scale */
double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias){
    return FgtrM_bias_fast(dicke(z),del_bias,sigsmallR,sig_bias);
}

double FgtrM_bias(double z, double M, double del_bias, double sig_bias){
    return sigmaparam_FgtrM_bias(z,EvaluateSigma(log(M)),del_bias,sig_bias);
}

//  Redshift derivative of the conditional collapsed fraction
float dfcoll_dz(float z, float sigma_min, float del_bias, float sig_bias)
{
    double dz,z1,z2;
    double fc1,fc2,ans;

    dz = 0.001;
    z1 = z + dz;
    z2 = z - dz;
    fc1 = sigmaparam_FgtrM_bias(z1, sigma_min, del_bias, sig_bias);
    fc2 = sigmaparam_FgtrM_bias(z2, sigma_min, del_bias, sig_bias);
    ans = (fc1 - fc2)/(2.0*dz);
    return ans;
}
/* compute a mass limit where the stellar baryon fraction and the escape fraction exceed unity */
//NOTE (JD): Why aren't we using 1e10 * pow(FRAC,-1/PL)? what am I missing here that makes the rootfind necessary
float Mass_limit (float logM, float PL, float FRAC) {
    return FRAC*pow(pow(10.,logM)/1e10,PL);
}
void bisection(float *x, float xlow, float xup, int *iter){
    *x=(xlow + xup)/2.;
    ++(*iter);
}

float Mass_limit_bisection(float Mmin, float Mmax, float PL, float FRAC){
    int i, iter, max_iter=200;
    float rel_tol=0.001;
    float logMlow, logMupper, x, x1;
    iter = 0;
    logMlow = log10(Mmin);
    logMupper = log10(Mmax);

    if (PL < 0.) {
        if (Mass_limit(logMlow,PL,FRAC) <= 1.) {
            return Mmin;
        }
    }
    else if (PL > 0.) {
        if (Mass_limit(logMupper,PL,FRAC) <= 1.) {
            return Mmax;
        }
    }
    else
        return 0;
    bisection(&x, logMlow, logMupper, &iter);
    do {
        if((Mass_limit(logMlow,PL,FRAC)-1.)*(Mass_limit(x,PL,FRAC)-1.) < 0.)
            logMupper = x;
        else
            logMlow = x;
        bisection(&x1, logMlow, logMupper, &iter);
        if(fabs(x1-x) < rel_tol) {
            return pow(10.,x1);
        }
        x = x1;
    }
    while(iter < max_iter);

    // Got to max_iter without finding a solution.
    LOG_ERROR("Failed to find a mass limit to regulate stellar fraction/escape fraction is between 0 and 1.");
    LOG_ERROR(" The solution does not converge or iterations are not sufficient.");
//    Throw(ParameterError);
    Throw(MassDepZetaError);

    return(0.0);
}
//set the minimum source mass for the integrals, If we have an exponential cutoff we go below the chosen mass by a factor of 50
//NOTE: previously, with USE_MINI_HALOS, the sigma table was initialised with M_MIN_INTEGRAL/50, but then all integrals perofmed
//      from M_MIN_INTEGRAL
double minimum_source_mass(double redshift, bool xray, AstroParams *astro_params, FlagOptions *flag_options){
    double Mmin,min_factor,mu_factor,t_vir_min;
    if(flag_options->USE_MASS_DEPENDENT_ZETA && !flag_options->USE_MINI_HALOS)
        min_factor = 50.; // small lower bound to cover far below the turnover
    else
        min_factor = 1.; //sharp cutoff

    // automatically false if !USE_MASS_DEPENDENT_ZETA
    if(flag_options->USE_MINI_HALOS){
        Mmin = global_params.M_MIN_INTEGRAL;
    }
    // automatically true if USE_MASS_DEPENDENT_ZETA
    else if(flag_options->M_MIN_in_Mass) {
         //NOTE: previously this divided Mturn by 50 in spin temperature, but not in the ionised box
         //     which I think is a bug with M_MIN_in_Mass, since there is a sharp cutoff
        Mmin = astro_params->M_TURN;
    }
    else {
        //if the virial temp minimum is set below ionisation we need to set mu accordingly
        t_vir_min = xray ? astro_params->X_RAY_Tvir_MIN : astro_params->ION_Tvir_MIN;
        mu_factor = t_vir_min < 9.99999e3 ? 1.22 : 0.6;
        Mmin = TtoM(redshift, t_vir_min, mu_factor);
    }

    //This is mostly unused and needs to be tested
    if(global_params.P_CUTOFF){
        Mmin = fmax(Mmin,M_J_WDM());
    }

    Mmin /= min_factor;

    return Mmin;
}
