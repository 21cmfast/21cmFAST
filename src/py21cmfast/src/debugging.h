#ifndef _DEBUGGING_H
#define _DEBUGGING_H

#include "InputParameters.h"

//Input debugging
void writeFlagOptions(FlagOptions *p);
void writeUserParams(UserParams *p);
void writeCosmoParams(CosmoParams *p);
void writeAstroParams(FlagOptions *fo, AstroParams *p);

//output debugging
void debugSummarizeIC(InitialConditions *x, int HII_DIM, int DIM, float NCF);
void debugSummarizePerturbField(PerturbedField *x, int HII_DIM, float NCF);
void debugSummarizeBox(float *box, int size, float ncf, char *indent);

//error debugging
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double* result);
void FunctionThatThrows();

#endif
