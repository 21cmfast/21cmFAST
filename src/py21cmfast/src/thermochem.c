#include "thermochem.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "exceptions.h"
#include "logger.h"

#define MIN_DENSITY_LOW_LIMIT (9e-8)

float ComputeFullyIoinizedTemperature(float z_re, float z, float delta, float T_re) {
    // z_re: the redshift of reionization
    // z:    the current redshift
    // delta:the density contrast
    float result, delta_re;
    // just be fully ionized
    if (fabs(z - z_re) < 1e-4)
        result = 1;
    else {
        // linearly extrapolate to get density at reionization
        delta_re = delta * (1. + z) / (1. + z_re);
        if (delta_re <= -1) delta_re = -1. + MIN_DENSITY_LOW_LIMIT;
        // evolving ionized box eq. 6 of McQuinn 2015, ignored the dependency of density at
        // ionization
        if (delta <= -1) delta = -1. + MIN_DENSITY_LOW_LIMIT;
        result = pow((1. + delta) / (1. + delta_re), 1.1333);
        result *= pow((1. + z) / (1. + z_re), 3.4);
        result *= expf(pow((1. + z) / 7.1, 2.5) - pow((1. + z_re) / 7.1, 2.5));
    }
    result *= pow(T_re, 1.7);
    // 1e4 before helium reionization; double it after
    result += pow(1e4 * ((1. + z) / 4.), 1.7) * (1 + delta);
    result = pow(result, 0.5882);
    // LOG_DEBUG("z_re=%.4f, z=%.4f, delta=%e, Tk=%.f", z_re, z, delta, result);
    return result;
}

float ComputePartiallyIoinizedTemperature(float T_HI, float res_xH, float T_re) {
    if (res_xH <= 0.) return T_re;
    if (res_xH >= 1) return T_HI;

    return T_HI * res_xH + T_re * (1. - res_xH);
}

/* returns the case A hydrogen recombination coefficient (Abel et al. 1997) in cm^3 s^-1*/
double alpha_A(double T) {
    double logT, ans;
    logT = log(T / (double)1.1604505e4);
    ans = pow(E, -28.6130338 - 0.72411256 * logT - 2.02604473e-2 * pow(logT, 2) -
                     2.38086188e-3 * pow(logT, 3) - 3.21260521e-4 * pow(logT, 4) -
                     1.42150291e-5 * pow(logT, 5) + 4.98910892e-6 * pow(logT, 6) +
                     5.75561414e-7 * pow(logT, 7) - 1.85676704e-8 * pow(logT, 8) -
                     3.07113524e-9 * pow(logT, 9));
    return ans;
}

/* returns the case B hydrogen recombination coefficient (Spitzer 1978) in cm^3 s^-1*/
double alpha_B(double T) { return alphaB_10k * pow(T / 1.0e4, -0.75); }

/*
 Function NEUTRAL_FRACTION returns the hydrogen neutral fraction, chi, given:
 hydrogen density (pcm^-3)
 gas temperature (10^4 K)
 ionization rate (1e-12 s^-1)
 */
double neutral_fraction(double density, double T4, double gamma, int usecaseB) {
    double chi, b, alpha, corr_He = 1.0 / (4.0 / cosmo_params_global->Y_He - 3);

    if (usecaseB)
        alpha = alpha_B(T4 * 1e4);
    else
        alpha = alpha_A(T4 * 1e4);

    gamma *= 1e-12;

    // approximation chi << 1
    chi = (1 + corr_He) * density * alpha / gamma;
    if (chi < TINY) {
        return 0;
    }
    if (chi < 1e-5) return chi;

    //  this code, while mathematically accurate, is numerically buggy for very small x_HI, so i
    //  will use valid approximation x_HI <<1 above when x_HI < 1e-5, and this otherwise... the two
    //  converge seemlessly
    // get solutions of quadratic of chi (neutral fraction)
    b = -2 - gamma / (density * (1 + corr_He) * alpha);
    chi = (-b - sqrt(b * b - 4)) / 2.0;  // correct root
    return chi;
}

/* function HeI_ion_crosssec returns the HI ionization cross section at parameter frequency
 (taken from Verner et al (1996) */
double HeI_ion_crosssec(double nu) {
    double x, y;

    if (nu < HeI_NUIONIZATION) return 0;

    x = nu / NU_over_EV / 13.61 - 0.4434;
    y = sqrt(x * x + pow(2.136, 2));
    return 9.492e-16 * ((x - 1) * (x - 1) + 2.039 * 2.039) * pow(y, (0.5 * 3.188 - 5.5)) *
           pow(1.0 + sqrt(y / 1.469), -3.188);
}

/* function HeII_ion_crosssec returns the HeII ionization cross section at parameter frequency
 (taken from Osterbrock, pg. 14) */
double HeII_ion_crosssec(double nu) {
    double epsilon, Z = 2;

    if (nu < HeII_NUIONIZATION) return 0;

    if (nu == HeII_NUIONIZATION) nu += TINY;

    epsilon = sqrt(nu / HeII_NUIONIZATION - 1);
    return (6.3e-18) / Z / Z * pow(HeII_NUIONIZATION / nu, 4) *
           pow(E, 4 - (4 * atan(epsilon) / epsilon)) / (1 - pow(E, -2 * PI / epsilon));
}

/* function HI_ion_crosssec returns the HI ionization cross section at parameter frequency
 (taken from Osterbrock, pg. 14) */
double HI_ion_crosssec(double nu) {
    double epsilon, Z = 1;

    if (nu < NUIONIZATION) return 0;

    if (nu == NUIONIZATION) nu += TINY;

    epsilon = sqrt(nu / NUIONIZATION - 1);
    return (6.3e-18) / Z / Z * pow(NUIONIZATION / nu, 4) *
           pow(E, 4 - (4 * atan(epsilon) / epsilon)) / (1 - pow(E, -2 * PI / epsilon));
}

/* Return the thomspon scattering optical depth from zstart to zend through fully ionized IGM.
 The hydrogen reionization history is given by the zarry and xHarry parameters, in increasing
 redshift order of length len.*/
typedef struct {
    float *z, *xH;
    int len;
} tau_e_params;
double dtau_e_dz(double z, void *params) {
    float xH, xi;
    int i = 1;
    tau_e_params p = *(tau_e_params *)params;

    if ((p.len == 0) || !(p.z)) {
        return (1 + z) * (1 + z) * drdz(z);
    } else {
        // find where we are in the redshift array
        if (p.z[0] > z)  // ionization fraction is 1 prior to start of array
            return (1 + z) * (1 + z) * drdz(z);
        while ((i < p.len) && (p.z[i] < z)) {
            i++;
        }
        if (i == p.len) return 0;

        // linearly interpolate in redshift
        xH = p.xH[i - 1] + (p.xH[i] - p.xH[i - 1]) / (p.z[i] - p.z[i - 1]) * (z - p.z[i - 1]);
        xi = 1.0 - xH;
        if (xi < 0) {
            LOG_WARNING("in taue: funny business xi=%e, changing to 0.", xi);
            xi = 0;
        }
        if (xi > 1) {
            LOG_WARNING("in taue: funny business xi=%e, changing to 1", xi);
            xi = 1;
        }

        return xi * (1 + z) * (1 + z) * drdz(z);
    }
}
double tau_e(float zstart, float zend, float *zarry, float *xHarry, int len, float z_re_HeII) {
    double prehelium, posthelium, error;
    gsl_function F;
    double rel_tol = 1e-3;  //<- relative tolerance
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    tau_e_params p;

    if (zstart >= zend) {
        LOG_ERROR("in tau_e: First parameter must be smaller than the second.\n");
        Throw(ValueError);
    }

    F.function = &dtau_e_dz;
    p.z = zarry;
    p.xH = xHarry;
    p.len = len;
    F.params = &p;
    if ((len > 0) && zarry) zend = zarry[len - 1] - FRACT_FLOAT_ERR;

    int status;

    gsl_set_error_handler_off();

    if (zend > z_re_HeII) {  // && (zstart < Zreion_HeII)){
        if (zstart < z_re_HeII) {
            status = gsl_integration_qag(&F, z_re_HeII, zstart, 0, rel_tol, 1000, GSL_INTEG_GAUSS61,
                                         w, &prehelium, &error);

            if (status != 0) {
                LOG_ERROR("gsl integration error occured!");
                LOG_ERROR(
                    "(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e "
                    "error=%e",
                    z_re_HeII, zstart, rel_tol, prehelium, error);
                LOG_ERROR("data: zstart=%e zend=%e", zstart, zend);
                CATCH_GSL_ERROR(status);
            }

            status = gsl_integration_qag(&F, zend, z_re_HeII, 0, rel_tol, 1000, GSL_INTEG_GAUSS61,
                                         w, &posthelium, &error);

            if (status != 0) {
                LOG_ERROR("gsl integration error occured!");
                LOG_ERROR(
                    "(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e "
                    "error=%e",
                    zend, z_re_HeII, rel_tol, posthelium, error);
                LOG_ERROR("data: zstart=%e zend=%e", zstart, zend);
                CATCH_GSL_ERROR(status);
            }
        } else {
            prehelium = 0;
            status = gsl_integration_qag(&F, zend, zstart, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w,
                                         &posthelium, &error);

            if (status != 0) {
                LOG_ERROR("gsl integration error occured!");
                LOG_ERROR(
                    "(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e "
                    "error=%e",
                    zend, zstart, rel_tol, posthelium, error);
                CATCH_GSL_ERROR(status);
            }
        }
    } else {
        posthelium = 0;
        status = gsl_integration_qag(&F, zend, zstart, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w,
                                     &prehelium, &error);

        if (status != 0) {
            LOG_ERROR("gsl integration error occured!");
            LOG_ERROR(
                "(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",
                zend, zstart, rel_tol, prehelium, error);
            CATCH_GSL_ERROR(status);
        }
    }
    gsl_integration_workspace_free(w);

    return SIGMAT * ((N_b0 + He_No) * prehelium + N_b0 * posthelium);
}

float ComputeTau(UserParams *user_params, CosmoParams *cosmo_params, int NPoints, float *redshifts,
                 float *global_xHI, float z_re_HeII) {
    Broadcast_struct_global_noastro(user_params, cosmo_params);

    return tau_e(0, redshifts[NPoints - 1], redshifts, global_xHI, NPoints, z_re_HeII);
}

double atomic_cooling_threshold(float z) { return TtoM(z, 1e4, 0.59); }

double molecular_cooling_threshold(float z) { return TtoM(z, 600, 1.22); }

double lyman_werner_threshold(float z, float J_21_LW, float vcb, AstroParams *astro_params) {
    // correction follows Schauer+20, fit jointly to LW feedback and relative velocities. They find
    // weaker effect of LW feedback than before (Stacy+11, Greif+11, etc.) due to HII self
    // shielding.

    // this follows Visbal+15, which is taken as the optimal fit from Fialkov+12
    // which was calibrated with the simulations of Stacy+11 and Greif+11;
    double mcrit_noLW = 3.314e7 * pow(1. + z,-1.5);
    double f_LW = 1.0 + astro_params->A_LW * pow(J_21_LW, astro_params->BETA_LW);

    double f_vcb = pow(1.0 + astro_params->A_VCB * vcb / SIGMAVCB, astro_params->BETA_VCB);

    // double mcrit_LW = mcrit_noLW * (1.0 + 10. * sqrt(J_21_LW)); //Eq. (12) in Schauer+20
    // return pow(10.0, log10(mcrit_LW) + 0.416 * vcb/SIGMAVCB ); //vcb and sigmacb in km/s, from
    // Eq. (9)

    return (mcrit_noLW * f_LW * f_vcb);
}

double reionization_feedback(float z, float Gamma_halo_HII, float z_IN) {
    if (z_IN <= 1e-19) return 1e-40;
    return REION_SM13_M0 * pow(HALO_BIAS * Gamma_halo_HII, REION_SM13_A) *
           pow((1. + z) / 10, REION_SM13_B) *
           pow(1 - pow((1. + z) / (1. + z_IN), REION_SM13_C), REION_SM13_D);
}
