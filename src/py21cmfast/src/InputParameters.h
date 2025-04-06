#ifndef _PARAMSTRUCTURES_H
#define _PARAMSTRUCTURES_H

#include <stdbool.h>
// since ffi.cdef() cannot include directives, we store the types and globals in another file
//   Since it is unguarded, make sure to ONLY include this file from here
#include "_inputparams_wrapper.h"

void Broadcast_struct_global_all(MatterParams *matter_params, MatterFlags *matter_flags,
                                 CosmoParams *cosmo_params, AstroParams *astro_params,
                                 AstroFlags *astro_flags);
void Broadcast_struct_global_noastro(MatterParams *matter_params, MatterFlags *matter_flags,
                                     CosmoParams *cosmo_params);

#endif
