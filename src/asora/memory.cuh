#pragma once

// Allocate grid memory
void device_init(const int &, const int &);

// Deallocate grid memory
void device_close();

// Copy density grid to device memory
void density_to_device(double*,const int &);

// Copy radiation tables to device memory
void photo_table_to_device(double*,double*,const int &);

// Copy source positions & fluxes to device memory
void source_data_to_device(int*, double*, const int &);

// Pointers to device memory
extern double * cdh_dev;
extern double * n_dev;
extern double * x_dev;
extern double * phi_dev;
extern double * photo_thin_table_dev;
extern double * photo_thick_table_dev;
extern int * src_pos_dev;
extern double * src_flux_dev;

// Number of sources done in parallel ("source batch size")
extern int NUM_SRC_PAR;