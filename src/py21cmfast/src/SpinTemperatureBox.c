// Re-write of find_HII_bubbles.c for being accessible within the MCMC
#include "SpinTemperatureBox.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "cosmology.h"
#include "exceptions.h"
#include "heating_helper_progs.h"
#include "indexing.h"
#include "logger.h"
#include "thermochem.h"

/* Maximum allowed value for the kinetic temperature. Useful to set to avoid some spurious behaviour
 when the code is run with redshift poor resolution and very high X-ray heating efficiency */
#define MAX_TK (float)5e4

// a debug flag for printing results from a single cell without passing cell number to the functions
static int debug_printed;

// construct a Ts table above Z_HEAT_MAX, this can happen if we are computing the first box or if we
// request a redshift above Z_HEAT_MAX
void init_first_Ts(TsBox *box, float *dens, float z) {
    index_huge box_ct;
    double xe, TK, cT_ad;

    xe = xion_RECFAST(z, 0);
    TK = T_RECFAST(z, 0);
    if (astro_options_global->USE_ADIABATIC_FLUCTUATIONS) {
        cT_ad = cT_approx(z);
    } else {
        cT_ad = 0.;
    }

#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        double gdens;
        float curr_xalpha;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            gdens = dens[box_ct];
            box->kinetic_temp_neutral[box_ct] = TK * (1.0 + cT_ad * gdens);
            box->xray_ionised_fraction[box_ct] = xe;
            // compute the spin temperature
            box->spin_temperature[box_ct] = get_Ts(z, gdens, TK, xe, 0, &curr_xalpha);
        }
    }
}

// These are factors which only need to be calculated once per redshift
struct spintemp_from_sfr_prefactors {
    double Trad;                 // CMB temperature
    double Trad_inv;             // inverse for acceleration (/ slower than * sometimes)
    double Ts_prefactor;         // some volume factors
    double xa_tilde_prefactor;   // lyman alpha prefactor
    double xc_inverse;           // collisional prefactor
    double dcomp_dzp_prefactor;  // compton prefactor
    double Nb_zp;                // physical critical density (baryons)
    double N_zp;                 // physical critical density
    double hubble_zp;            // H(z)
    double growth_zp;
    double dgrowth_dzp;
    double dt_dzp;
};

void set_zp_consts(double zp, struct spintemp_from_sfr_prefactors *consts) {
    // constant prefactors
    double gamma_alpha;
    consts->growth_zp = dicke(zp);
    consts->hubble_zp = hubble(zp);
    consts->dgrowth_dzp = ddicke_dz(zp);
    consts->dt_dzp = dtdz(zp);

    // Required quantities for calculating the IGM spin temperature
    // Note: These used to be determined in evolveInt (and other functions). But I moved them all
    // here, into a single location.
    consts->Trad = physconst.T_cmb * (1.0 + zp);
    consts->Trad_inv = 1.0 / consts->Trad;
    consts->Ts_prefactor =
        pow(1.0e-7 * (1.342881e-7 / consts->hubble_zp) * No * pow(1 + zp, 3), 1. / 3.);

    // division of C/10. is converstion of electric charge from esu to coulomb
    gamma_alpha = physconst.f_alpha *
                  pow(physconst.nu_Ly_alpha * physconst.e_charge / (physconst.c_cms / 10.), 2.);
    // division by 1000. to convert gram to kg and division by 100. to convert cm to m
    gamma_alpha /=
        6. * (physconst.m_e / 1000.) * pow(physconst.c_cms / 100., 3.) * physconst.vac_perm;

    // 1e-8 converts angstrom to cm.
    consts->xa_tilde_prefactor =
        8. * M_PI * pow(physconst.lambda_Ly_alpha * 1.e-8, 2.) * gamma_alpha * physconst.T_21;
    consts->xa_tilde_prefactor /= 9. * physconst.A10 * consts->Trad;
    // consts->xa_tilde_prefactor = 1.66e11/(1.0+zp);

    consts->xc_inverse = pow(1.0 + zp, 3.0) * physconst.T_21 / (consts->Trad * physconst.A10);

    consts->dcomp_dzp_prefactor = (-1.51e-4) / (consts->hubble_zp / Ho) /
                                  (cosmo_params_global->hlittle) * pow(consts->Trad, 4.0) /
                                  (1.0 + zp);

    // used for lya_X and sinks NOTE: the 2 density factors are from
    // source & absorber since its downscattered x-ray
    consts->Nb_zp = N_b0 * (1 + zp) * (1 + zp) * (1 + zp);
    consts->N_zp = No * (1 + zp) * (1 + zp) * (1 + zp);  // used for CMB

    LOG_DEBUG("Set zp consts Tr %.2e Ts %.2e xa %.2e xc %.2e cm %.2e", consts->Trad,
              consts->Ts_prefactor, consts->xa_tilde_prefactor, consts->xc_inverse,
              consts->dcomp_dzp_prefactor);
    LOG_DEBUG("Nb %.2e D %.2e H %.2e dD %.2e dt %.2e", consts->Nb_zp, consts->growth_zp,
              consts->hubble_zp, consts->dgrowth_dzp, consts->dt_dzp);
}

// All the cell-dependent stuff needed to calculate Ts
struct local_rad_terms {
    double xray_ionization_rate;
    double xray_heating_rate;
    double xray_lya_flux;
    double lya_flux_continuum_injected;
    double lyw_flux;
    double lya_flux_continuum;
    double lya_flux_injected;
    double delta;
    double prev_Ts;
    double prev_Tk;
    double prev_xe;
};

// outputs from the Ts calculation, to go into new boxes
struct Ts_cell {
    double Ts;
    double x_e;
    double Tk;
    double J_21_LW;
};

// Function for calculating the Ts box outputs quickly by using pre-calculated constants
//   as much as possible
struct Ts_cell get_Ts_fast(float zp, float dzp, struct spintemp_from_sfr_prefactors *consts,
                           struct local_rad_terms *rad) {
    // Now we can solve the evolution equations  //
    struct Ts_cell output;
    double tau21, xCMB, dxion_sink_dt, dxe_dzp, dadia_dzp, dspec_dzp, dcomp_dzp, dxheat_dzp;
    double dCMBheat_dzp, eps_CMB, eps_Lya_cont, eps_Lya_inj, E_continuum, E_injected,
        Ndot_alpha_cont, Ndot_alpha_inj;
    debug_printed = 0;

    tau21 = (3 * physconst.h_p * physconst.A10 * physconst.c_cms * physconst.lambda_21 *
             physconst.lambda_21 / 32. / M_PI / physconst.k_B) *
            ((1 - rad->prev_xe) * consts->N_zp) / rad->prev_Ts / consts->hubble_zp;

    if (tau21 > 1e-8) {
        xCMB = (1. - exp(-tau21)) / tau21;
    } else {
        // When tau21 is very small, we can use the Taylor expansion of the exponential
        // to avoid numerical issues
        xCMB = 1. - tau21 / 2 * (1 - tau21 / 3 * (1 - tau21 / 4));
    }

    // Electron density
    // NOTE: Nb_zp includes helium, TODO: make sure this is right
    dxion_sink_dt = alpha_A(rad->prev_Tk) * astro_params_global->CLUMPING_FACTOR * rad->prev_xe *
                    rad->prev_xe * H_FRAC * consts->Nb_zp * (1. + rad->delta);

    dxe_dzp = consts->dt_dzp * (rad->xray_ionization_rate - dxion_sink_dt);

    // Temperature components //
    // Adiabatic heating/cooling from structure formation
    dadia_dzp = 3 / (1.0 + zp);
    if (fabs(rad->delta) > FRACT_FLOAT_ERR)
        dadia_dzp += consts->dgrowth_dzp / (consts->growth_zp * (1.0 / rad->delta + 1.0));

    dadia_dzp *= (2.0 / 3.0) * rad->prev_Tk;

    // Heating due to the changing species
    dspec_dzp = -dxe_dzp * rad->prev_Tk / (1 + rad->prev_xe);

    // Compton heating
    dcomp_dzp = consts->dcomp_dzp_prefactor * (rad->prev_xe / (1.0 + rad->prev_xe + HE_FRAC)) *
                (consts->Trad - rad->prev_Tk);

    // X-ray heating
    dxheat_dzp = 0.;
    if (astro_options_global->USE_X_RAY_HEATING) {
        dxheat_dzp = rad->xray_heating_rate * consts->dt_dzp;
    }
    // CMB heating rate
    dCMBheat_dzp = 0.;
    if (astro_options_global->USE_CMB_HEATING) {
        // Meiksin et al. 2021
        eps_CMB = (3. / 4.) * (consts->Trad / physconst.T_21) * physconst.A10 * H_FRAC *
                  (physconst.h_p * physconst.h_p / physconst.lambda_21 / physconst.lambda_21 /
                   physconst.m_p) *
                  (1. + 2. * rad->prev_Tk / physconst.T_21);
        dCMBheat_dzp = -eps_CMB * (2. / 3. / physconst.k_B / (1. + rad->prev_xe)) /
                       consts->hubble_zp / (1. + zp);
    }

    // Ly-alpha heating rate
    eps_Lya_cont = 0.;
    eps_Lya_inj = 0.;
    if (astro_options_global->USE_LYA_HEATING) {
        E_continuum =
            Energy_Lya_heating(rad->prev_Tk, rad->prev_Ts, taugp(zp, rad->delta, rad->prev_xe), 2);
        E_injected =
            Energy_Lya_heating(rad->prev_Tk, rad->prev_Ts, taugp(zp, rad->delta, rad->prev_xe), 3);
        if (isnan(E_continuum) || isinf(E_continuum)) {
            E_continuum = 0.;
        }
        if (isnan(E_injected) || isinf(E_injected)) {
            E_injected = 0.;
        }
        Ndot_alpha_cont = (4. * M_PI * physconst.nu_Ly_alpha) /
                          (consts->Nb_zp * (1. + rad->delta)) / (1. + zp) / physconst.c_cms *
                          rad->lya_flux_continuum;
        Ndot_alpha_inj = (4. * M_PI * physconst.nu_Ly_alpha) / (consts->Nb_zp * (1. + rad->delta)) /
                         (1. + zp) / physconst.c_cms * rad->lya_flux_injected;
        eps_Lya_cont =
            -Ndot_alpha_cont * E_continuum * (2. / 3. / physconst.k_B / (1. + rad->prev_xe));
        eps_Lya_inj =
            -Ndot_alpha_inj * E_injected * (2. / 3. / physconst.k_B / (1. + rad->prev_xe));
    }

    // Update the cell quantities based on the above terms
    double x_e, Tk, J_alpha_tot;
    x_e = rad->prev_xe + (dxe_dzp * dzp);  // remember dzp is negative
    // can do this late in evolution if dzp is too large
    if (x_e > 1)
        x_e = 1 - FRACT_FLOAT_ERR;
    else if (x_e < 0)
        x_e = 0;
    // NOTE: does this stop cooling if we ever go over the limit? I suppose that shouldn't happen
    // but it's strange anyway
    Tk = rad->prev_Tk;
    if (Tk < MAX_TK) {
        if (debug_printed == 0 && omp_get_thread_num() == 0)
            LOG_SUPER_DEBUG(
                "Heating Terms: T %.4e | X %.4e | c %.4e | S %.4e | A %.4e | c %.4e | lc %.4e | li "
                "%.4e | dz %.4e",
                Tk, dxheat_dzp, dcomp_dzp, dspec_dzp, dadia_dzp, dCMBheat_dzp, eps_Lya_cont,
                eps_Lya_inj, dzp);

        Tk += (dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont +
               eps_Lya_inj) *
              dzp;
        if (debug_printed == 0 && omp_get_thread_num() == 0) LOG_SUPER_DEBUG("--> T %.4e", Tk);
    }
    // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
    if (Tk < 0) Tk = consts->Trad;

    output.x_e = x_e;
    output.Tk = Tk;
    output.J_21_LW = astro_options_global->USE_MINI_HALOS ? rad->lyw_flux : 0.;

    if (astro_options_global->USE_LYA_HEATING) {
        J_alpha_tot = rad->lya_flux_continuum + rad->lya_flux_injected + rad->xray_lya_flux;
    } else {
        J_alpha_tot = rad->lya_flux_continuum_injected + rad->xray_lya_flux;
    }

    // JD: I'm leaving these as comments in case I'm wrong, but there's NO WAY a compiler doesn't
    // know the fastest way to invert a number
    //  T_inv = expf((-1.)*logf(Tk));
    //  T_inv_sq = expf((-2.)*logf(Tk));
    double T_inv, T_inv_sq;
    double xc_fast, xi_power, xa_tilde_fast_arg, xa_tilde_fast = 0.;
    double TS_fast, TSold_fast;
    T_inv = 1 / Tk;
    T_inv_sq = T_inv * T_inv;

    xc_fast = (1.0 + rad->delta) * consts->xc_inverse *
              ((1.0 - x_e) * No * kappa_10(Tk, 0) + x_e * N_b0 * kappa_10_elec(Tk, 0) +
               x_e * No * kappa_10_pH(Tk, 0));

    xi_power = consts->Ts_prefactor * cbrt((1.0 + rad->delta) * (1.0 - x_e) * T_inv_sq);

    xa_tilde_fast_arg = consts->xa_tilde_prefactor * J_alpha_tot *
                        pow(1.0 + 2.98394 * xi_power + 1.53583 * xi_power * xi_power +
                                3.85289 * xi_power * xi_power * xi_power,
                            -1.);

    if (J_alpha_tot > 1.0e-20) {  // Must use WF effect
        TS_fast = consts->Trad;
        TSold_fast = 0.0;
        while (fabs(TS_fast - TSold_fast) / TS_fast > 1.0e-3) {
            TSold_fast = TS_fast;

            xa_tilde_fast =
                (1.0 - 0.0631789 * T_inv + 0.115995 * T_inv_sq -
                 0.401403 * T_inv * pow(TS_fast, -1.) + 0.336463 * T_inv_sq * pow(TS_fast, -1.)) *
                xa_tilde_fast_arg;

            TS_fast = (xCMB + xa_tilde_fast + xc_fast) *
                      pow(xCMB * consts->Trad_inv +
                              xa_tilde_fast * (T_inv + 0.405535 * T_inv * pow(TS_fast, -1.) -
                                               0.405535 * T_inv_sq) +
                              xc_fast * T_inv,
                          -1.);
        }
    } else {  // Collisions only
        TS_fast = (xCMB + xc_fast) / (xCMB * consts->Trad_inv + xc_fast * T_inv);
    }
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
    if (debug_printed == 0 && omp_get_thread_num() == 0) {
        LOG_SUPER_DEBUG("Spin terms xc %.5e xa %.5e xC %.5e Ti %.5e T2 %.5e --> T %.4e", xc_fast,
                        xa_tilde_fast, xCMB, T_inv, T_inv_sq, TS_fast);
        debug_printed = 1;
    }
#endif
    // It can very rarely result in a negative spin temperature. If negative, it is a very small
    // number. Take the absolute value, the optical depth can deal with very large numbers, so ok to
    // be small
    TS_fast = fabs(TS_fast);
    output.Ts = TS_fast;

    return output;
}

int ComputeTsBox(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                 RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                 InitialConditions *ini_boxes, TsBox *this_spin_temp) {
    int status;
    Try {  // This Try{} wraps the whole function.
        LOG_DEBUG("Spintemp input values:");
        LOG_DEBUG("redshift=%f, prev_redshift=%f", redshift, prev_redshift);
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeCosmoParams(cosmo_params_global);
        writeAstroParams(astro_params_global);
        writeAstroOptions(astro_options_global);
#endif
        omp_set_num_threads(simulation_options_global->N_THREADS);

        if (redshift >= simulation_options_global->Z_HEAT_MAX) {
            LOG_DEBUG("redshift greater than Z_HEAT_MAX");
            init_first_Ts(this_spin_temp, perturbed_field->density, redshift);
            return (0);
        }

        this_spin_temp->Q_HI = radiation_fields->Q_HI;

        // set the constants calculated once per snapshot
        struct spintemp_from_sfr_prefactors zp_consts;
        set_zp_consts(redshift, &zp_consts);

        index_huge box_ct;
        double dzp;
        double J_alpha_ave, xheat_ave, xion_ave, Ts_ave, Tk_ave, x_e_ave;
        J_alpha_ave = xheat_ave = xion_ave = Ts_ave = Tk_ave = x_e_ave = 0;
        double J_LW_ave = 0., lya_flux_continuum_ave = 0, lya_flux_injected_ave = 0;

        dzp = redshift - prev_redshift;

#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
        {
            double curr_delta;
            struct Ts_cell ts_cell;
            struct local_rad_terms local_rad;
#pragma omp for reduction(+ : J_alpha_ave, xheat_ave, xion_ave, Ts_ave, Tk_ave, x_e_ave, \
                              lya_flux_continuum_ave, lya_flux_injected_ave)
            for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
                curr_delta = perturbed_field->density[box_ct];
                // NOTE: this corrected for aliasing before, but sometimes there are still some
                // delta==-1 cells
                //   which breaks the adiabatic part
                if (curr_delta <= -1) {
                    curr_delta = -1 + FRACT_FLOAT_ERR;
                }

                // set the local radiation fields for this cell
                if (astro_options_global->USE_X_RAY_HEATING) {
                    local_rad.xray_heating_rate = radiation_fields->xray_heating_rate[box_ct];
                }
                local_rad.xray_ionization_rate = radiation_fields->xray_ionization_rate[box_ct];
                local_rad.xray_lya_flux = radiation_fields->xray_lya_flux[box_ct];
                local_rad.delta = curr_delta;
                if (astro_options_global->USE_MINI_HALOS) {
                    local_rad.lyw_flux = radiation_fields->lyw_flux[box_ct];
                }
                if (astro_options_global->USE_LYA_HEATING) {
                    local_rad.lya_flux_continuum = radiation_fields->lya_flux_continuum[box_ct];
                    local_rad.lya_flux_injected = radiation_fields->lya_flux_injected[box_ct];
                } else {
                    local_rad.lya_flux_continuum_injected =
                        radiation_fields->lya_flux_continuum_injected[box_ct];
                }
                local_rad.prev_Ts = previous_spin_temp->spin_temperature[box_ct];
                local_rad.prev_Tk = previous_spin_temp->kinetic_temp_neutral[box_ct];
                local_rad.prev_xe = previous_spin_temp->xray_ionised_fraction[box_ct];

                // compute the spin temperature and other thermal fields for this cell
                ts_cell = get_Ts_fast(redshift, dzp, &zp_consts, &local_rad);
                this_spin_temp->spin_temperature[box_ct] = ts_cell.Ts;
                this_spin_temp->kinetic_temp_neutral[box_ct] = ts_cell.Tk;
                this_spin_temp->xray_ionised_fraction[box_ct] = ts_cell.x_e;
                if (astro_options_global->USE_MINI_HALOS) {
                    this_spin_temp->J_21_LW[box_ct] = ts_cell.J_21_LW;
                }

                // Single cell debug
                if (box_ct == 0) {
                    if (astro_options_global->USE_X_RAY_HEATING) {
                        LOG_SUPER_DEBUG(
                            "Cell0: delta: %.3e | xray_heating_rate: %.3e | xray_ionization_rate: "
                            "%.3e "
                            "| xray_lya_flux: %.3e",
                            curr_delta, local_rad.xray_heating_rate, local_rad.xray_ionization_rate,
                            local_rad.xray_lya_flux);
                    } else {
                        LOG_SUPER_DEBUG(
                            "Cell0: delta: %.3e | xray_ionization_rate: %.3e "
                            "| xray_lya_flux: %.3e",
                            curr_delta, local_rad.xray_ionization_rate, local_rad.xray_lya_flux);
                    }
                    if (astro_options_global->USE_LYA_HEATING) {
                        LOG_SUPER_DEBUG("lya_flux_injected %.3e | lya_flux_continuum %.3e",
                                        local_rad.lya_flux_injected, local_rad.lya_flux_continuum);
                    } else {
                        LOG_SUPER_DEBUG("Cell0: lya_flux_continuum_injected: %.3e",
                                        local_rad.lya_flux_continuum_injected);
                    }
                    if (astro_options_global->USE_MINI_HALOS) {
                        LOG_SUPER_DEBUG("lyw_flux %.3e", local_rad.lyw_flux);
                    }
                    LOG_SUPER_DEBUG("Ts %.5e Tk %.5e x_e %.5e J_21_LW %.5e", ts_cell.Ts, ts_cell.Tk,
                                    ts_cell.x_e, ts_cell.J_21_LW);
                }

#if LOG_LEVEL >= DEBUG_LEVEL
                xion_ave += local_rad.xray_ionization_rate;
                if (astro_options_global->USE_X_RAY_HEATING) {
                    xheat_ave += local_rad.xray_heating_rate;
                }
                if (astro_options_global->USE_MINI_HALOS) {
                    J_LW_ave += ts_cell.J_21_LW;
                }
                if (astro_options_global->USE_LYA_HEATING) {
                    lya_flux_injected_ave += local_rad.lya_flux_injected;
                    lya_flux_continuum_ave += local_rad.lya_flux_continuum;
                    J_alpha_ave += local_rad.xray_lya_flux + local_rad.lya_flux_continuum +
                                   local_rad.lya_flux_injected;
                } else {
                    J_alpha_ave += local_rad.xray_lya_flux + local_rad.lya_flux_continuum_injected;
                }
                x_e_ave += ts_cell.x_e;
                Tk_ave += ts_cell.Tk;
                Ts_ave += ts_cell.Ts;

#endif
            }
        }

#if LOG_LEVEL >= DEBUG_LEVEL
        x_e_ave /= (double)HII_TOT_NUM_PIXELS;
        Ts_ave /= (double)HII_TOT_NUM_PIXELS;
        Tk_ave /= (double)HII_TOT_NUM_PIXELS;
        LOG_DEBUG("AVERAGES zp = %.2e Ts = %.2e x_e = %.2e Tk %.2e", redshift, Ts_ave, x_e_ave,
                  Tk_ave);

        xion_ave /= (double)HII_TOT_NUM_PIXELS;
        J_alpha_ave /= (double)HII_TOT_NUM_PIXELS;
        LOG_DEBUG("J_alpha = %.2e  xion = %.2e", J_alpha_ave, xion_ave);
        if (astro_options_global->USE_X_RAY_HEATING) {
            xheat_ave /= (double)HII_TOT_NUM_PIXELS;
            LOG_DEBUG("xheat = %.2e", xheat_ave);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            J_LW_ave /= (double)HII_TOT_NUM_PIXELS;
            LOG_DEBUG("J_LW %.2e", J_LW_ave / 1e21);
        }
        if (astro_options_global->USE_LYA_HEATING) {
            lya_flux_continuum_ave /= (double)HII_TOT_NUM_PIXELS;
            lya_flux_injected_ave /= (double)HII_TOT_NUM_PIXELS;
            LOG_DEBUG("lya_flux_continuum %.2e lya_flux_injected %.2e", lya_flux_continuum_ave,
                      lya_flux_injected_ave);
        }
#endif

        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            if (isfinite(this_spin_temp->spin_temperature[box_ct]) == 0) {
                if (astro_options_global->USE_LYA_HEATING) {
                    if (astro_options_global->USE_X_RAY_HEATING) {
                        LOG_ERROR(
                            "Estimated spin temperature is either infinite of NaN!"
                            "idx %llu delta %.3e xray_heating_rate %.3e xray_ionization_rate %.3e "
                            "xray_lya_flux %.3e lya_flux_continuum %.3e lya_flux_injected %.3e",
                            box_ct, perturbed_field->density[box_ct],
                            radiation_fields->xray_heating_rate[box_ct],
                            radiation_fields->xray_ionization_rate[box_ct],
                            radiation_fields->xray_lya_flux[box_ct],
                            radiation_fields->lya_flux_continuum[box_ct],
                            radiation_fields->lya_flux_injected[box_ct]);
                    } else {
                        LOG_ERROR(
                            "Estimated spin temperature is either infinite of NaN!"
                            "idx %llu delta %.3e xray_ionization_rate %.3e xray_lya_flux %.3e "
                            "lya_flux_continuum %.3e lya_flux_injected %.3e",
                            box_ct, perturbed_field->density[box_ct],
                            radiation_fields->xray_ionization_rate[box_ct],
                            radiation_fields->xray_lya_flux[box_ct],
                            radiation_fields->lya_flux_continuum[box_ct],
                            radiation_fields->lya_flux_injected[box_ct]);
                    }
                } else {
                    if (astro_options_global->USE_X_RAY_HEATING) {
                        LOG_ERROR(
                            "Estimated spin temperature is either infinite of NaN!"
                            "idx %llu delta %.3e xray_heating_rate %.3e xray_ionization_rate %.3e "
                            "xray_lya_flux %.3e lya_flux_continuum_injected %.3e",
                            box_ct, perturbed_field->density[box_ct],
                            radiation_fields->xray_heating_rate[box_ct],
                            radiation_fields->xray_ionization_rate[box_ct],
                            radiation_fields->xray_lya_flux[box_ct],
                            radiation_fields->lya_flux_continuum_injected[box_ct]);
                    } else {
                        LOG_ERROR(
                            "Estimated spin temperature is either infinite of NaN!"
                            "idx %llu delta %.3e xray_ionization_rate %.3e xray_lya_flux %.3e "
                            "lya_flux_continuum_injected %.3e",
                            box_ct, perturbed_field->density[box_ct],
                            radiation_fields->xray_ionization_rate[box_ct],
                            radiation_fields->xray_lya_flux[box_ct],
                            radiation_fields->lya_flux_continuum_injected[box_ct]);
                    }
                }
                Throw(InfinityorNaNError);
            }
        }
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}
