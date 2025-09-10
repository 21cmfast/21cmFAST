#ifndef _THERMOCHEM_H
#define _THERMOCHEM_H

#include "InputParameters.h"

float ComputeTau(int Npoints, float* redshifts, float* global_xHI, float z_re_HeII);
double molecular_cooling_threshold(float z);
double atomic_cooling_threshold(float z);
double lyman_werner_threshold(float z, float J_21_LW, float vcb);
double reionization_feedback(float z, float Gamma_halo_HII, float z_IN);
float ComputeFullyIoinizedTemperature(float z_re, float z, float delta, float T_re);
float ComputePartiallyIoinizedTemperature(float T_HI, float res_xH, float T_re);

/* returns the case A hydrogen recombination coefficient (Abel et al. 1997) in cm^3 s^-1*/
double alpha_A(double T);
/* returns the case B hydrogen recombination coefficient (Spitzer 1978) in cm^3 s^-1*/
double alpha_B(double T);

double HeI_ion_crosssec(double nu);
double HeII_ion_crosssec(double nu);
double HI_ion_crosssec(double nu);
double neutral_fraction(double density, double T4, double gamma, int usecaseB);

#endif
