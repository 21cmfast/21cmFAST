#ifndef _PARAMSTRUCTURES_H
#define _PARAMSTRUCTURES_H

#include <stdbool.h>
// Since it is unguarded, make sure to ONLY include this file from here
#include "_inputparams_wrapper.h"

#ifdef __cplusplus
extern "C"
{
#endif
void set_external_table_path(GlobalParams *params, const char *value);
char* get_external_table_path(GlobalParams *params);
void set_wisdoms_path(GlobalParams *params, const char *value);
char* get_wisdoms_path(GlobalParams *params);
void Broadcast_struct_global_all(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options);
void Broadcast_struct_global_noastro(UserParams *user_params, CosmoParams *cosmo_params);
#ifdef __cplusplus
}
#endif

#endif // _PARAMSTRUCTURES_H
