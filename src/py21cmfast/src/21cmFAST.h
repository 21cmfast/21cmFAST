/*
    This is the header file for the wrappable version of 21cmFAST, or 21cmMC.
    It contains function signatures, struct definitions and globals to which the Python wrapper code
    requires access. This is not included in any C file but is called by cffi to expose the functions within to the wrapper
*/

//TODO: I don't need *every* function/declaration in these header files,
//  some of these lines should be replaced by *only* the functions I want
//  exposed to the wrapper
#include "BrightnessTemperatureBox.h"
#include "cosmology.h"
#include "debugging.h"
#include "filtering.h"
#include "HaloBox.h"
#include "HaloField.h"
#include "InitialConditions.h"
#include "IonisationBox.h"
#include "InputParameters.h"
#include "interp_tables.h"
#include "LuminosityFunction.h"
#include "OutputStructs.h"
#include "PerturbField.h"
#include "PerturbHaloField.h"
#include "photoncons.h"
#include "SpinTemperatureBox.h"
#include "Stochasticity.h"
#include "thermochem.h"

//CFFI needs to be able to access a free function to make the __del__ method for OutputStruct fields
//  This will expose the standard free() function to the wrapper so it can be used
//TODO: place this in a cdef within build_cffi.py instead
void free(void *ptr);
