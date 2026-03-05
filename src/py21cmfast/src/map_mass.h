
#include "HaloBox.h"
#include "HaloCatalog.h"
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

void move_halo_galprops(double redshift, HaloCatalog *halos, float *vel_pointers[3],
                        float *vel_pointers_2LPT[3], int vel_dim[3], float *mturn_a_grid,
                        float *mturn_m_grid, HaloBox *boxes, int out_dim[3],
                        ScalingConstants *consts);

double cic_read_float_wrapper(float *box, double pos[3], int box_dim[3]);

double *MapMass_gpu(InitialConditions *boxes, double *resampled_box, int dimension,
                    double f_pixel_factor, double init_growth_factor, double velocity_scale,
                    double velocity_scale_z, double velocity_scale_2LPT,
                    double velocity_scale_2LPT_z);

#ifdef __cplusplus
}
#endif
