
#include "HaloBox.h"
#include "OutputStructs.h"
#include "scaling_relations.h"

void move_grid_masses(double redshift, float *dens_pointer, int dens_dim[3], float *vel_pointers[3],
                      float *vel_pointers_2LPT[3], int vel_dim[3], double *resampled_box,
                      int out_dim[3]);

void move_grid_galprops(double redshift, float *dens_pointer, int dens_dim[3],
                        float *vel_pointers[3], float *vel_pointers_2LPT[3], int vel_dim[3],
                        HaloBox *boxes, int out_dim[3], float *mturn_a_grid, float *mturn_m_grid,
                        ScalingConstants *consts, IntegralCondition *integral_cond);
