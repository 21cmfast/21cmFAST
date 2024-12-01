#ifndef _SCALING_H
#define _SCALING_H

//Parameters for the scaling relations
//  These are just the values which come from the InputStruct objects and don't change within the snapshot
//  using this reduces the use of the global parameter structs and allows fewer exp/log unit changes
struct ScalingConstants{
    double redshift;
    bool fix_mean;
    bool scaling_median;

    double fstar_10;
    double alpha_star;
    double sigma_star;

    double alpha_upper;
    double pivot_upper;
    double upper_pivot_ratio;

    double fstar_7;
    double alpha_star_mini;

    double t_h;
    double t_star;
    double sigma_sfr_lim;
    double sigma_sfr_idx;

    double l_x;
    double l_x_mini;
    double sigma_xray;

    double fesc_10;
    double alpha_esc;
    double fesc_7;

    double vcb_norel;
    double mturn_a_nofb;
    double mturn_m_nofb;

    double Mlim_Fstar;
    double Mlim_Fesc;
    double Mlim_Fstar_mini;
    double Mlim_Fesc_mini;
};

double nion_fraction(double lnM, double M_turn, double ln_n_star, double a_star, double ln_n_esc,
                    double a_esc, double ln_l_star, double ln_l_esc);

double nion_fraction_mini(double lnM, double M_turn_lo, double M_turn_hi, double ln_n_star,
                     double a_star, double ln_n_esc, double a_esc, double ln_l_star, double ln_l_esc);

#endif
