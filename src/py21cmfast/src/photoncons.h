#ifndef _PHOTONCONS_H
#define _PHOTONCONS_H

#include <stdbool.h>
#include "InputParameters.h"

//This is directly accessed in the wrapper currently
//TODO: remove this global declaration and make an internal checking function
extern bool photon_cons_allocated;

int InitialisePhotonCons(UserParams *user_params, CosmoParams *cosmo_params,
                         AstroParams *astro_params, FlagOptions *flag_options);

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
int ComputeZstart_PhotonCons(double *zstart);

float adjust_redshifts_for_photoncons(
    AstroParams *astro_params, FlagOptions *flag_options, float *redshift,
    float *stored_redshift, float *absolute_delta_z
);

void determine_deltaz_for_photoncons();
void FreePhotonConsMemory();

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons);

//alpha photoncons functions
void set_alphacons_params(double norm, double slope);

#endif
