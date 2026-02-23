#ifndef _DEVICECONSTANTS_CUH
#define _DEVICECONSTANTS_CUH

#include "InputParameters.h"

#ifndef _HALOFIELD_CU // double check whether this is necessary

extern __constant__ SimulationOptions d_simulation_options;
extern __constant__ MatterOptions d_matter_options;
extern __constant__ CosmoParams d_cosmo_params;
extern __constant__ AstroParams d_astro_params;
extern __constant__ double d_test_params;

// Physical constants for GPU kernels
extern __constant__ double d_delta_c_sph;
extern __constant__ double d_delta_c_delos;

#endif

#endif
