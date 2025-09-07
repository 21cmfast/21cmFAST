#ifndef _HALOFIELD_CUH
#define _HALOFIELD_CUH
#include "InputParameters.h"
#include "interpolation_types.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void updateGlobalParams(SimulationOptions *h_simulation_options, CosmoParams *h_cosmo_params, AstroParams *h_astro_params);
#ifdef __cplusplus
}
#endif

#endif
