#ifndef _HALOFIELD_CUH
#define _HALOFIELD_CUH
#include "InputParameters.h"
#include "interpolation_types.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void updateGlobalParams(UserParams *h_user_params, CosmoParams *h_cosmo_params, AstroParams *h_astro_params);
#ifdef __cplusplus
}
#endif

#endif