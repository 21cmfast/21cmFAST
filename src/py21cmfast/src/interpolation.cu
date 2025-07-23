#include <cuda_runtime.h>

#include "interpolation.cuh"

__device__ double EvaluateRGTable1D(double x, RGTable1D *table)
{
    double x_min = table->x_min;
    double x_width = table->x_width;
    int idx = (int)floor((x - x_min) / x_width);
    double table_val = x_min + x_width * (double)idx;
    double interp_point = (x - table_val) / x_width;

    // a + f(a-b) is one fewer operation but less precise
    double result = table->y_arr[idx] * (1 - interp_point) + table->y_arr[idx + 1] * (interp_point);

    return result;
}

__device__ double EvaluateRGTable2D(double x, double y, RGTable2D *table)
{
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

    left_edge = table->z_arr[x_idx][y_idx] * (1 - interp_point_y) + table->z_arr[x_idx][y_idx + 1] * (interp_point_y);
    right_edge = table->z_arr[x_idx + 1][y_idx] * (1 - interp_point_y) + table->z_arr[x_idx + 1][y_idx + 1] * (interp_point_y);

    result = left_edge * (1 - interp_point_x) + right_edge * (interp_point_x);

    return result;
}
