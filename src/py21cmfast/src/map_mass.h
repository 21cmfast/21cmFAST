
#include "HaloBox.h"
#include "OutputStructs.h"
#include "scaling_relations.h"

#ifdef __cplusplus
extern "C" {
#endif

void move_grid_masses(double redshift, float *dens_pointer, int dens_dim[3], float *vel_pointers[3],
                      float *vel_pointers_2LPT[3], int vel_dim[3], double *resampled_box,
                      int out_dim[3]);

void move_grid_galprops(double redshift, float *dens_pointer, int dens_dim[3],
                        float *vel_pointers[3], float *vel_pointers_2LPT[3], int vel_dim[3],
                        HaloBox *boxes, int out_dim[3], float *mturn_a_grid, float *mturn_m_grid,
                        ScalingConstants *consts, IntegralCondition *integral_cond);

double *MapMass_gpu(InitialConditions *boxes, double *resampled_box, int dimension,
                    float f_pixel_factor, float init_growth_factor);

#ifdef __cplusplus
}
#endif
