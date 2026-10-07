#ifndef _LUMFUNCTION_H
#define _LUMFUNCTION_H

#include "InputParameters.h"

int ComputeLF(int nbins, int component, int NUM_OF_REDSHIFT_FOR_LF, double *z_LF,
              double *M_TURNs_ACG, double *M_TURNs_MCG, double *M_uv_z, double *M_h_z,
              double *log10phi);

#endif
