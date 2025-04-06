#ifndef _LUMFUNCTION_H
#define _LUMFUNCTION_H

#include "InputParameters.h"

int ComputeLF(int nbins, MatterParams *matter_params, MatterFlags *matter_flags,
              CosmoParams *cosmo_params, AstroParams *astro_params, AstroFlags *astro_flags,
              int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs,
              double *M_uv_z, double *M_h_z, double *log10phi);

#endif
