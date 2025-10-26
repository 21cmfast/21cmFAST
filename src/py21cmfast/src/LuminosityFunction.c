//     Luv/SFR = 1 / 1.15 x 10^-28 [M_solar yr^-1/erg s^-1 Hz^-1]
//     G. Sun and S. R. Furlanetto (2016) MNRAS, 417, 33

#include "LuminosityFunction.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "exceptions.h"
#include "hmf.h"
#include "interp_tables.h"
#include "logger.h"
#include "thermochem.h"

#define Luv_over_SFR (double)(1. / 1.15 / 1e-28)

#define delta_lnMhalo (double)(5e-6)
#define Mhalo_min (double)(1e6)
#define Mhalo_max (double)(1e16)

bool initialised_ComputeLF = false;

gsl_interp_accel *LF_spline_acc;
gsl_spline *LF_spline;

gsl_interp_accel *deriv_spline_acc;
gsl_spline *deriv_spline;

double *lnMhalo_param, *Muv_param, *Mhalo_param;
double *log10phi, *M_uv_z, *M_h_z;
double *deriv, *lnM_temp, *deriv_temp;

int initialise_ComputeLF(int nbins) {
    lnMhalo_param = calloc(nbins, sizeof(double));
    Muv_param = calloc(nbins, sizeof(double));
    Mhalo_param = calloc(nbins, sizeof(double));

    LF_spline_acc = gsl_interp_accel_alloc();
    LF_spline = gsl_spline_alloc(gsl_interp_cspline, nbins);

    init_ps();

    int status;
    Try initialiseSigmaMInterpTable(0.999 * Mhalo_min, 1.001 * Mhalo_max);
    Catch(status) {
        LOG_ERROR("\t...called from initialise_ComputeLF");
        return (status);
    }

    initialised_ComputeLF = true;
    return (0);
}

void cleanup_ComputeLF() {
    free(lnMhalo_param);
    free(Muv_param);
    free(Mhalo_param);
    gsl_spline_free(LF_spline);
    gsl_interp_accel_free(LF_spline_acc);
    freeSigmaMInterpTable();
    initialised_ComputeLF = 0;
}

int ComputeLF(int nbins, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs,
              double *M_uv_z, double *M_h_z, double *log10phi) {
    /*
        This is an API-level function and thus returns an int status.
    */
    int status;
    Try {  // This try block covers the whole function.
        // This NEEDS to be done every time, because the actual object passed in as
        // simulation_options, cosmo_params etc. can change on each call, freeing up the memory.
        initialise_ComputeLF(nbins);

        int i, i_z;
        int i_unity, i_smth, mf, nbins_smth = 7;
        double dlnMhalo, lnMhalo_i, SFRparam, Muv_1, Muv_2, dMuvdMhalo;
        double Mhalo_i, lnMhalo_min, lnMhalo_max, lnMhalo_lo, lnMhalo_hi, dlnM, growthf;
        double f_duty_upper, Mcrit_atom;
        float Fstar, Fstar_temp;
        double dndm;
        int gsl_status;

        gsl_set_error_handler_off();
        if (astro_params_global->ALPHA_STAR < -0.5)
            LOG_WARNING(
                "ALPHA_STAR is %f, which is unphysical value given the observational LFs.\n"
                "Also, when ALPHA_STAR < -.5, LFs may show a kink. It is recommended to set "
                "ALPHA_STAR > -0.5.",
                astro_params_global->ALPHA_STAR);

        mf = matter_options_global->HMF;

        lnMhalo_min = log(Mhalo_min * 0.999);
        lnMhalo_max = log(Mhalo_max * 1.001);
        dlnMhalo = (lnMhalo_max - lnMhalo_min) / (double)(nbins - 1);

        for (i_z = 0; i_z < NUM_OF_REDSHIFT_FOR_LF; i_z++) {
            growthf = dicke(z_LF[i_z]);
            Mcrit_atom = atomic_cooling_threshold(z_LF[i_z]);

            i_unity = -1;
            for (i = 0; i < nbins; i++) {
                // generate interpolation arrays
                lnMhalo_param[i] = lnMhalo_min + dlnMhalo * (double)i;
                Mhalo_i = exp(lnMhalo_param[i]);

                if (component == 1)
                    Fstar = astro_params_global->F_STAR10 *
                            pow(Mhalo_i / 1e10, astro_params_global->ALPHA_STAR);
                else
                    Fstar = astro_params_global->F_STAR7_MINI *
                            pow(Mhalo_i / 1e7, astro_params_global->ALPHA_STAR_MINI);
                if (Fstar > 1.) Fstar = 1;

                if (i_unity < 0) {  // Find the array number at which Fstar crosses unity.
                    if (astro_params_global->ALPHA_STAR > 0.) {
                        if ((1. - Fstar) < FRACT_FLOAT_ERR) i_unity = i;
                    } else if (astro_params_global->ALPHA_STAR < 0. && i < nbins - 1) {
                        if (component == 1)
                            Fstar_temp = astro_params_global->F_STAR10 *
                                         pow(exp(lnMhalo_min + dlnMhalo * (double)(i + 1)) / 1e10,
                                             astro_params_global->ALPHA_STAR);
                        else
                            Fstar_temp = astro_params_global->F_STAR7_MINI *
                                         pow(exp(lnMhalo_min + dlnMhalo * (double)(i + 1)) / 1e7,
                                             astro_params_global->ALPHA_STAR_MINI);
                        if (Fstar_temp < 1. && (1. - Fstar) < FRACT_FLOAT_ERR) i_unity = i;
                    }
                }

                // parametrization of SFR
                SFRparam = Mhalo_i * cosmo_params_global->OMb / cosmo_params_global->OMm *
                           (double)Fstar *
                           (double)(hubble(z_LF[i_z]) * physconst.s_per_yr /
                                    astro_params_global->t_STAR);  // units of M_solar/year

                Muv_param[i] = 51.63 - 2.5 * log10(SFRparam * Luv_over_SFR);  // UV magnitude
                // except if Muv value is nan or inf, but avoid error put the value as 10.
                if (isinf(Muv_param[i]) || isnan(Muv_param[i])) Muv_param[i] = 10.;

                M_uv_z[i + i_z * nbins] = Muv_param[i];
            }

            gsl_status = gsl_spline_init(LF_spline, lnMhalo_param, Muv_param, nbins);
            CATCH_GSL_ERROR(gsl_status);

            lnMhalo_lo = log(Mhalo_min);
            lnMhalo_hi = log(Mhalo_max);
            dlnM = (lnMhalo_hi - lnMhalo_lo) / (double)(nbins - 1);

            // There is a kink on LFs at which Fstar crosses unity. This kink is a numerical
            // artefact caused by the derivate of dMuvdMhalo. Most of the cases the kink doesn't
            // appear in magnitude ranges we are interested (e.g. -22 < Muv < -10). However, for
            // some extreme parameters, it appears. To avoid this kink, we use the interpolation of
            // the derivate in the range where the kink appears. 'i_unity' is the array number at
            // which the kink appears. 'i_unity-3' and 'i_unity+12' are related to the range of
            // interpolation, which is an arbitrary choice. NOTE: This method does NOT work in cases
            // with ALPHA_STAR < -0.5. But, this parameter range is unphysical given that the
            //       observational LFs favour positive ALPHA_STAR in this model.
            // i_smth = 0: calculates LFs without interpolation.
            // i_smth = 1: calculates LFs using interpolation where Fstar crosses unity.
            if (i_unity - 3 < 0)
                i_smth = 0;
            else if (i_unity + 12 > nbins - 1)
                i_smth = 0;
            else
                i_smth = 1;
            if (i_smth == 0) {
                for (i = 0; i < nbins; i++) {
                    // calculate luminosity function
                    lnMhalo_i = lnMhalo_lo + dlnM * (double)i;
                    Mhalo_param[i] = exp(lnMhalo_i);

                    M_h_z[i + i_z * nbins] = Mhalo_param[i];

                    Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
                    Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);

                    dMuvdMhalo = (Muv_2 - Muv_1) / (2. * delta_lnMhalo * exp(lnMhalo_i));

                    if (component == 1)
                        f_duty_upper = 1.;
                    else
                        f_duty_upper = exp(-(Mhalo_param[i] / Mcrit_atom));

                    log10phi[i + i_z * nbins] = log10(
                        unconditional_hmf(growthf, lnMhalo_i, z_LF[i_z], mf) / Mhalo_param[i] *
                        exp(-(M_TURNs[i_z] / Mhalo_param[i])) *
                        (cosmo_params_global->OMm * RHOcrit) * f_duty_upper / fabs(dMuvdMhalo));

                    if (isinf(log10phi[i + i_z * nbins]) || isnan(log10phi[i + i_z * nbins]) ||
                        log10phi[i + i_z * nbins] < -30.)
                        log10phi[i + i_z * nbins] = -30.;
                }
            } else {
                lnM_temp = calloc(nbins_smth, sizeof(double));
                deriv_temp = calloc(nbins_smth, sizeof(double));
                deriv = calloc(nbins, sizeof(double));

                for (i = 0; i < nbins; i++) {
                    // calculate luminosity function
                    lnMhalo_i = lnMhalo_lo + dlnM * (double)i;
                    Mhalo_param[i] = exp(lnMhalo_i);

                    M_h_z[i + i_z * nbins] = Mhalo_param[i];

                    Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
                    Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);

                    dMuvdMhalo = (Muv_2 - Muv_1) / (2. * delta_lnMhalo * exp(lnMhalo_i));
                    deriv[i] = fabs(dMuvdMhalo);
                }

                deriv_spline_acc = gsl_interp_accel_alloc();
                deriv_spline = gsl_spline_alloc(gsl_interp_cspline, nbins_smth);

                // generate interpolation arrays to smooth discontinuity of the derivative causing a
                // kink Note that the number of array elements and the range of interpolation are
                // made by arbitrary choices.
                lnM_temp[0] = lnMhalo_param[i_unity - 3];
                lnM_temp[1] = lnMhalo_param[i_unity - 2];
                lnM_temp[2] = lnMhalo_param[i_unity + 8];
                lnM_temp[3] = lnMhalo_param[i_unity + 9];
                lnM_temp[4] = lnMhalo_param[i_unity + 10];
                lnM_temp[5] = lnMhalo_param[i_unity + 11];
                lnM_temp[6] = lnMhalo_param[i_unity + 12];

                deriv_temp[0] = deriv[i_unity - 3];
                deriv_temp[1] = deriv[i_unity - 2];
                deriv_temp[2] = deriv[i_unity + 8];
                deriv_temp[3] = deriv[i_unity + 9];
                deriv_temp[4] = deriv[i_unity + 10];
                deriv_temp[5] = deriv[i_unity + 11];
                deriv_temp[6] = deriv[i_unity + 12];

                gsl_status = gsl_spline_init(deriv_spline, lnM_temp, deriv_temp, nbins_smth);
                CATCH_GSL_ERROR(gsl_status);

                for (i = 0; i < 9; i++) {
                    deriv[i_unity + i - 1] = gsl_spline_eval(
                        deriv_spline, lnMhalo_param[i_unity + i - 1], deriv_spline_acc);
                }
                for (i = 0; i < nbins; i++) {
                    if (component == 1)
                        f_duty_upper = 1.;
                    else
                        f_duty_upper = exp(-(Mhalo_param[i] / Mcrit_atom));

                    dndm = unconditional_hmf(growthf, log(Mhalo_param[i]), z_LF[i_z], mf) *
                           (cosmo_params_global->OMm * RHOcrit) / Mhalo_param[i];
                    log10phi[i + i_z * nbins] = log10(dndm * exp(-(M_TURNs[i_z] / Mhalo_param[i])) *
                                                      f_duty_upper / deriv[i]);
                    if (isinf(log10phi[i + i_z * nbins]) || isnan(log10phi[i + i_z * nbins]) ||
                        log10phi[i + i_z * nbins] < -30.)
                        log10phi[i + i_z * nbins] = -30.;
                }
            }
        }

        cleanup_ComputeLF();
    }  // End try
    Catch(status) { return status; }
    return (0);
}
