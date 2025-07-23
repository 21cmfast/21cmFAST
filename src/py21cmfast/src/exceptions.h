#ifndef _EXCEPTIONS_H
#define _EXCEPTIONS_H

#include "cexcept.h"

#ifdef __cplusplus
extern "C" {
#endif
define_exception_type(int);

// NOTE: declaration here, definition in debugging.c
extern struct exception_context the_exception_context[1];

// Our own error codes
#define SUCCESS 0
#define IOError 1
#define GSLError 2
#define ValueError 3
#define PhotonConsError 4
#define TableGenerationError 5
#define TableEvaluationError 6
#define InfinityorNaNError 7
#define MassDepZetaError 8
#define MemoryAllocError 9
#define CUDAError 10

#define CATCH_GSL_ERROR(status)                                                           \
    if (status > 0) {                                                                     \
        LOG_ERROR("GSL Error Encountered (Code = %d): %s", status, gsl_strerror(status)); \
        Throw(GSLError);                                                                  \
    }
#define CATCH_CUDA_ERROR(err)                                             \
    if (err != cudaSuccess) {                                             \
        LOG_ERROR("CUDA Error Encountered: %s", cudaGetErrorString(err)); \
        Throw(CUDAError);                                                 \
    }

#ifdef __cplusplus
}
#endif
#endif
