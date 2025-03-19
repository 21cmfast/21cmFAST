// Header file for all the output structure type definitions.
//  These are not included in the same header files as their functions
//  to avoid circular dependencies. e.g HaloBox requires Spintemp which requires
//  HaloBox
#ifndef _OUTPUSTRUCTS_H
#define _OUTPUSTRUCTS_H

#include "InputParameters.h"

// since ffi.cdef() cannot include directives, we store the types and globals in
// another file
//   Since it is unguarded, make sure to ONLY include this file from here
#include "_outputstructs_wrapper.h"

#endif
