#ifndef _HEATHELPER_H
#define _HEATHELPER_H

// * initialization routine * //
int init_heat();

/* destruction/deallocation routine */
void destruct_heat();

// * returns the spectral emissity * //
double spectral_emissivity(double nu_norm, int flag, int Population);

// * Ionization fraction from RECFAST. * //
double xion_RECFAST(float z, int flag);

// * IGM temperature from RECFAST; includes Compton heating and adiabatic expansion only. * //
double T_RECFAST(float z, int flag);

// approximation for the adiabatic index at z=6-50 from 2302.08506
float cT_approx(float z);

// * returns the spin temperature * //
float get_Ts(float z, float delta, float TK, float xe, float Jalpha, float * curr_xalpha);

//* Returns recycling fraction (=fraction of photons converted into Lyalpha for Ly-n resonance * //
double frecycle(int n);

// * Returns frequency of Lyman-n, in units of Lyman-alpha * //
double nu_n(int n);

double kappa_10_pH(double T, int flag);
double kappa_10_elec(double T, int flag);
double kappa_10(double TK, int flag);

double xcoll(double z, double TK, double delta, double xe);
double xcoll_HI(double z, double TK, double delta, double xe);
double xcoll_elec(double z, double TK, double delta, double xe);
double xcoll_prot(double z, double TK, double delta, double xe);

double xalpha_tilde(double z, double Jalpha, double TK, double TS, double delta, double xe);
double Tc_eff(double TK, double TS);
double Salpha_tilde(double TK, double TS, double tauGP);
double taugp(double z, double delta, double xe);

double species_weighted_x_ray_cross_section(double nu, double x_e);

// * Returns the maximum redshift at which a Lyn transition contributes to Lya flux at z * //
float zmax(float z, int n);

//Lyman-Alpha heating functions
int find_nearest_point(double min, double max, int n, double value);
int find_xyz_pos(int xpos, int ypos, int zpos, int len_yarr, int len_zarr);
double interpolate_heating_efficiencies(double tk, double ts, double taugp, double *arrE);
double Energy_Lya_heating(double Tk, double Ts, double tau_gp, int flag);

#endif