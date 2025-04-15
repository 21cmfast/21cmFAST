/* Function prototypes for the xray/lya tables as a function of energy and ionisation fraction*/
#ifndef _ELEC_INT_H
#define _ELEC_INT_H

#define x_int_NXHII 14
#define x_int_NENERGY 258

void initialize_interp_arrays();

// Primary functions to compute heating fractions and number of Lya photons or ionization produced,
// Note that En is the energy of the *primary* photon, so the energy in the initial ionization is
// included in all these.
// All energies are in eV.
// xHII_call is the desired ionized fraction.
float interp_fheat(float En, float xHII_call);
float interp_n_Lya(float En, float xHII_call);
float interp_nion_HI(float En, float xHII_call);
float interp_nion_HeI(float En, float xHII_call);
float interp_nion_HeII(float En, float xHII_call);

int locate_energy_index(float En);
int locate_xHII_index(float xHII_call);

// this array is currently used in SpinTemperatureBox.c and passed into heating_helper_progs.c
// functions
// TODO: remove it and make it static in elec_interp.c
extern float x_int_XHII[x_int_NXHII];

#endif
