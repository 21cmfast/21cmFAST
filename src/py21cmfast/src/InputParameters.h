#ifndef _PARAMSTRUCTURES_H
#define _PARAMSTRUCTURES_H

#include <stdbool.h>
// since ffi.cdef() cannot include directives, we store the types and globals in
// another file
//   Since it is unguarded, make sure to ONLY include this file from here
#include "_inputparams_wrapper.h"

void Broadcast_struct_global_all(UserParams *user_params,
                                 CosmoParams *cosmo_params,
                                 AstroParams *astro_params,
                                 FlagOptions *flag_options);
void Broadcast_struct_global_noastro(UserParams *user_params,
                                     CosmoParams *cosmo_params);

#endif
