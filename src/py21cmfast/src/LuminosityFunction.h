#ifndef _LUMFUNCTION_H
#define _LUMFUNCTION_H

#include "InputParameters.h"

int ComputeLF(int nbins, UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params,
               FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi);

#endif
