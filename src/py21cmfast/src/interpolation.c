// New file to store functions that deal with the interpolation tables, since they are used very
// often.
//   We use regular grid tables since they are faster to evaluate (we always know which bin we are
//   in) So I'm making a general function for the 1D and 2D cases

#include "interpolation.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "logger.h"

void allocate_RGTable1D(int n_bin, RGTable1D *ptr) {
    ptr->n_bin = n_bin;
    ptr->y_arr = calloc(n_bin, sizeof(double));
    ptr->allocated = true;
}

void allocate_RGTable1D_f(int n_bin, RGTable1D_f *ptr) {
    ptr->n_bin = n_bin;
    ptr->y_arr = calloc(n_bin, sizeof(float));
    ptr->allocated = true;
}

void free_RGTable1D(RGTable1D *ptr) {
    if (ptr->allocated) {
        free(ptr->y_arr);
        ptr->allocated = false;
    }
}

void free_RGTable1D_f(RGTable1D_f *ptr) {
    if (ptr->allocated) {
        free(ptr->y_arr);
        ptr->allocated = false;
    }
}

void allocate_RGTable2D(int n_x, int n_y, RGTable2D *ptr) {
    int i;
    ptr->nx_bin = n_x;
    ptr->ny_bin = n_y;

    ptr->z_arr = calloc(n_x, sizeof(double *));
    for (i = 0; i < n_x; i++) {
        ptr->z_arr[i] = calloc(n_y, sizeof(double));
    }
    ptr->allocated = true;
}

void allocate_RGTable2D_f(int n_x, int n_y, RGTable2D_f *ptr) {
    int i;
    ptr->nx_bin = n_x;
    ptr->ny_bin = n_y;

    ptr->z_arr = calloc(n_x, sizeof(float *));
    for (i = 0; i < n_x; i++) {
        ptr->z_arr[i] = calloc(n_y, sizeof(float));
    }
    ptr->allocated = true;
}

void free_RGTable2D_f(RGTable2D_f *ptr) {
    int i;
    if (ptr->allocated) {
        for (i = 0; i < ptr->nx_bin; i++) free(ptr->z_arr[i]);
        free(ptr->z_arr);
        ptr->allocated = false;
    }
}

void free_RGTable2D(RGTable2D *ptr) {
    int i;
    if (ptr->allocated) {
        for (i = 0; i < ptr->nx_bin; i++) free(ptr->z_arr[i]);
        free(ptr->z_arr);
        ptr->allocated = false;
    }
}

double EvaluateRGTable1D(double x, RGTable1D *table) {
    double x_min = table->x_min;
    double x_width = table->x_width;
    int idx = (int)floor((x - x_min) / x_width);
    double table_val = x_min + x_width * (double)idx;
    double interp_point = (x - table_val) / x_width;

    // a + f(a-b) is one fewer operation but less precise
    double result = table->y_arr[idx] * (1 - interp_point) + table->y_arr[idx + 1] * (interp_point);

    return result;
}

double EvaluateRGTable2D(double x, double y, RGTable2D *table) {
    double x_min = table->x_min;
    double x_width = table->x_width;
    double y_min = table->y_min;
    double y_width = table->y_width;
    int x_idx = (int)floor((x - x_min) / x_width);
    int y_idx = (int)floor((y - y_min) / y_width);

    double x_table = x_min + x_width * (double)x_idx;
    double y_table = y_min + y_width * (double)y_idx;

    double interp_point_x = (x - x_table) / x_width;
    double interp_point_y = (y - y_table) / y_width;

    double left_edge, right_edge, result;

    left_edge = table->z_arr[x_idx][y_idx] * (1 - interp_point_y) +
                table->z_arr[x_idx][y_idx + 1] * (interp_point_y);
    right_edge = table->z_arr[x_idx + 1][y_idx] * (1 - interp_point_y) +
                 table->z_arr[x_idx + 1][y_idx + 1] * (interp_point_y);

    result = left_edge * (1 - interp_point_x) + right_edge * (interp_point_x);

    return result;
}

// some tables are floats but I still need to return doubles
double EvaluateRGTable1D_f(double x, RGTable1D_f *table) {
    double x_min = table->x_min;
    double x_width = table->x_width;
    int idx = (int)floor((x - x_min) / x_width);
    double table_val = x_min + x_width * (float)idx;
    double interp_point = (x - table_val) / x_width;

    return table->y_arr[idx] * (1 - interp_point) + table->y_arr[idx + 1] * (interp_point);
}

double EvaluateRGTable2D_f(double x, double y, RGTable2D_f *table) {
    double x_min = table->x_min;
    double x_width = table->x_width;
    double y_min = table->y_min;
    double y_width = table->y_width;
    int x_idx = (int)floor((x - x_min) / x_width);
    int y_idx = (int)floor((y - y_min) / y_width);

    double x_table = x_min + x_width * (double)x_idx;
    double y_table = y_min + y_width * (double)y_idx;

    double interp_point_x = (x - x_table) / x_width;
    double interp_point_y = (y - y_table) / y_width;

    double left_edge, right_edge, result;

    left_edge = table->z_arr[x_idx][y_idx] * (1 - interp_point_y) +
                table->z_arr[x_idx][y_idx + 1] * (interp_point_y);
    right_edge = table->z_arr[x_idx + 1][y_idx] * (1 - interp_point_y) +
                 table->z_arr[x_idx + 1][y_idx + 1] * (interp_point_y);

    result = left_edge * (1 - interp_point_x) + right_edge * (interp_point_x);

    return result;
}

bool RGTable2D_out_of_bounds(RGTable2D *table, double x_val, double y_val) {
    if (x_val < table->x_min || x_val > table->x_min + table->nx_bin * table->x_width ||
        y_val < table->y_min || y_val > table->y_min + table->ny_bin * table->y_width)
        return true;

    return false;
}

bool RGTable2Df_out_of_bounds(RGTable2D_f *table, double x_val, double y_val) {
    if (x_val < table->x_min || x_val > table->x_min + table->nx_bin * table->x_width ||
        y_val < table->y_min || y_val > table->y_min + table->ny_bin * table->y_width)
        return true;

    return false;
}

bool RGTable1D_out_of_bounds(RGTable1D *table, double x_val) {
    if (x_val < table->x_min || x_val > table->x_min + table->n_bin * table->x_width) return true;

    return false;
}

bool RGTable1Df_out_of_bounds(RGTable1D_f *table, double x_val) {
    if (x_val < table->x_min || x_val > table->x_min + table->n_bin * table->x_width) return true;

    return false;
}
