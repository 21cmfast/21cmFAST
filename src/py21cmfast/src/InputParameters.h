#ifndef _PARAMSTRUCTURES_H
#define _PARAMSTRUCTURES_H

#include <stdbool.h>
// since ffi.cdef() cannot include directives, we store the types and globals in another file
//   Since it is unguarded, make sure to ONLY include this file from here
#include "_inputparams_wrapper.h"

/* Enum-style option values from src/py21cmfast/wrapper/inputs.py */
#define HMF_PS 0
#define HMF_ST 1
#define HMF_WATSON 2
#define HMF_WATSON_Z 3
#define HMF_DELOS 4

#define POWER_SPECTRUM_EH 0
#define POWER_SPECTRUM_BBKS 1
#define POWER_SPECTRUM_EFSTATHIOU 2
#define POWER_SPECTRUM_PEEBLES 3
#define POWER_SPECTRUM_WHITE 4
#define POWER_SPECTRUM_CLASS 5

#define INTERPOLATION_NO 0
#define INTERPOLATION_SIGMA 1
#define INTERPOLATION_HMF 2

#define SAMPLE_MASS_LIMITED 0
#define SAMPLE_NUMBER_LIMITED 1
#define SAMPLE_PARTITION 2
#define SAMPLE_BINARY_SPLIT 3

#define FILTER_TOPHAT 0
#define FILTER_SHARP_K 1
#define FILTER_GAUSSIAN 2

#define PERTURB_ALGORITHM_LINEAR 0
#define PERTURB_ALGORITHM_ZELDOVICH 1
#define PERTURB_ALGORITHM_2LPT 2

#define SOURCE_MODEL_CONST_ION_EFF 0
#define SOURCE_MODEL_E_INTEGRAL 1
#define SOURCE_MODEL_L_INTEGRAL 2
#define SOURCE_MODEL_DEXM_ESF 3
#define SOURCE_MODEL_CHMF_SAMPLER 4

#define PHOTON_CONS_NONE 0
#define PHOTON_CONS_Z 1
#define PHOTON_CONS_ALPHA 2
#define PHOTON_CONS_F 3

#define INTEGRATION_METHOD_GSL_QAG 0
#define INTEGRATION_METHOD_GAUSS_LEGENDRE 1
#define INTEGRATION_METHOD_GAMMA_APPROX 2

static inline bool source_model_is_mass_dependent(int source_model) {
    return source_model > SOURCE_MODEL_CONST_ION_EFF;
}

static inline bool source_model_uses_lagrangian_grids(int source_model) {
    return source_model > SOURCE_MODEL_E_INTEGRAL;
}

static inline bool source_model_uses_eulerian_grids(int source_model) {
    return source_model < SOURCE_MODEL_L_INTEGRAL;
}

static inline bool source_model_uses_sampled_halos(int source_model) {
    return source_model > SOURCE_MODEL_L_INTEGRAL;
}

static inline bool uses_interpolation_tables(int interpolation_mode) {
    return interpolation_mode > INTERPOLATION_NO;
}

static inline bool uses_hmf_interpolation(int interpolation_mode) {
    return interpolation_mode > INTERPOLATION_SIGMA;
}

void Broadcast_struct_global_all(SimulationOptions *simulation_options,
                                 MatterOptions *matter_options, CosmoParams *cosmo_params,
                                 AstroParams *astro_params, AstroOptions *astro_options,
                                 CosmoTables *cosmo_tables);
void Broadcast_struct_global_noastro(SimulationOptions *simulation_options,
                                     MatterOptions *matter_options, CosmoParams *cosmo_params);

#endif
