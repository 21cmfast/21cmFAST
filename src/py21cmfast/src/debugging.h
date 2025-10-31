#ifndef _DEBUGGING_H
#define _DEBUGGING_H

#include <complex.h>
#include <fftw3.h>

#include "InputParameters.h"
#include "OutputStructs.h"

// Input debugging
void writeAstroOptions(AstroOptions *p);
void writeSimulationOptions(SimulationOptions *p);
void writeMatterOptions(MatterOptions *p);
void writeCosmoParams(CosmoParams *p);
void writeAstroParams(AstroParams *p);

// output debugging
void debugSummarizeIC(InitialConditions *x, int HII_DIM, int DIM, float NCF);
void debugSummarizePerturbedField(PerturbedField *x, int HII_DIM, float NCF);
void debugSummarizeBox(float *box, int size_x, int size_y, int size_z, char *indent);
void debugSummarizeBoxDouble(double *box, int size_x, int size_y, int size_z, char *indent);
void debugSummarizeBoxComplex(fftwf_complex *box, int size_x, int size_y, int size_z, char *indent);

// error debugging
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double *result);
void FunctionThatThrows();

#endif
