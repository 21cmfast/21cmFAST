/*
    This is the header file for the wrappable version of 21cmFAST, or 21cmMC.
    It contains all the header files of the backend. This is not imported by
    any module in the backend, but is provided to the cffi set_source() call,
    and makes sure all headers are accessible to the compiler.
*/

//These are the includes which are needed for the wrapper,
//  which have all the typedefs and globals the wrapper explicitly uses

#include "InputParameters.h"
#include "OutputStructs.h"
#include "photoncons.h"
#include "_functionprototypes_wrapper.h"
