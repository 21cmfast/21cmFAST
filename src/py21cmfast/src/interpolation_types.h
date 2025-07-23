#ifndef _INTERPOLATION_TYPES_H
#define _INTERPOLATION_TYPES_H

typedef struct RGTable1D {
    int n_bin;
    double x_min;
    double x_width;

    double *y_arr;
    bool allocated;
} RGTable1D;

typedef struct RGTable2D {
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    double **z_arr;
    double *flatten_data;

    double saved_ll, saved_ul;  // for future acceleration
    bool allocated;
} RGTable2D;

typedef struct RGTable1D_f {
    int n_bin;
    double x_min;
    double x_width;

    float *y_arr;
    bool allocated;
} RGTable1D_f;

typedef struct RGTable2D_f {
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    float **z_arr;

    double saved_ll, saved_ul;  // for future acceleration
    bool allocated;
} RGTable2D_f;

#endif
