#ifndef _DEBUGGING_H
#define _DEBUGGING_H

#include <complex.h>
#include <fftw3.h>

#include "InputParameters.h"
#include "OutputStructs.h"

// Grid layout types for debug output
// Indicates how size_z parameter is interpreted:
typedef enum {
    STANDARD_LAYOUT,     // size_z is the actual dimension (no padding)
    FFTW_REAL_LAYOUT,    // size_z is the padded real-space dimension: 2*(unpadded_z/2+1)
    FFTW_COMPLEX_LAYOUT  // size_z is the padded complex dimension: unpadded_z/2+1
} GridLayout;

// Input debugging
void writeAstroOptions(AstroOptions *p);
void writeSimulationOptions(SimulationOptions *p);
void writeMatterOptions(MatterOptions *p);
void writeCosmoParams(CosmoParams *p);
void writeAstroParams(AstroParams *p);

// output debugging
void debugSummarizeIC(InitialConditions *x, int HII_DIM, int DIM, float NCF);
void debugSummarizePerturbedField(PerturbedField *x, int HII_DIM, float NCF);
void debugSummarizeBox(float *box, int size_x, int size_y, int size_z, GridLayout layout,
                       char *indent);
void debugSummarizeBoxDouble(double *box, int size_x, int size_y, int size_z, GridLayout layout,
                             char *indent);
void debugSummarizeBoxComplex(fftwf_complex *box, int size_x, int size_y, int size_z,
                              GridLayout layout, char *indent);

// error debugging
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double *result);
void FunctionThatThrows();

#endif
