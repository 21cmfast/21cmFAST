#ifndef _INTERPOLATION_H
#define _INTERPOLATION_H

#include <stdbool.h>

#include "interpolation_types.h"

#ifdef __cplusplus
extern "C" {
#endif
void allocate_RGTable1D(int n_bin, RGTable1D *ptr);
void allocate_RGTable1D_f(int n_bin, RGTable1D_f *ptr);
void allocate_RGTable2D(int n_x, int n_y, RGTable2D *ptr);
void allocate_RGTable2D_f(int n_x, int n_y, RGTable2D_f *ptr);

void free_RGTable1D(RGTable1D *ptr);
void free_RGTable1D_f(RGTable1D_f *ptr);
void free_RGTable2D(RGTable2D *ptr);
void free_RGTable2D_f(RGTable2D_f *ptr);

double EvaluateRGTable1D(double x, RGTable1D *table);
double EvaluateRGTable2D(double x, double y, RGTable2D *table);
double EvaluateRGTable1D_f(double x, RGTable1D_f *table);
double EvaluateRGTable2D_f(double x, double y, RGTable2D_f *table);
#ifdef __cplusplus
}
#endif

bool RGTable2D_out_of_bounds(RGTable2D *table, double x_val, double y_val);
bool RGTable2Df_out_of_bounds(RGTable2D_f *table, double x_val, double y_val);
bool RGTable1D_out_of_bounds(RGTable1D *table, double x_val);
bool RGTable1Df_out_of_bounds(RGTable1D_f *table, double x_val);

#endif
