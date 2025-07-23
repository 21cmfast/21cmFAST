#ifndef _DEVICECONSTANTS_CUH
#define _DEVICECONSTANTS_CUH

#include "InputParameters.h"

#ifndef _HALOFIELD_CU // double check whether this is necessary

extern __constant__ UserParams d_user_params;
extern __constant__ CosmoParams d_cosmo_params;
extern __constant__ AstroParams d_astro_params;
extern __constant__ double d_test_params;

#endif

#endif
