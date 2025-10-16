/*
    indexing.c -- Macros for indexing the cubic arrays, and their Fourier Transforms.

    How this works:

        The box size and dimensionalities (eg. D and HII_D) expressed here rely on
        ``simulation_options_global`` being initialized. This is initialized by
        ``Broadcast_struct_global_[all/noastro]`` in ``InputParameters.c``. Thus, that function
        must be called any time the user/cosmo params change. This is usually handled by the
   frontend

    A note on the Fourier Transform number of pixels and indexing:

        Details on what the "padding" is can be found here:
        http://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format

        In a nutshell, for the in-place transforms we do, there are symmetries in the
        real-complex FT that mean the last axis needs to have 2*(N/2 + 1) entries,
        with the division rounded down. For an even N, this means 2 extra entries on
        the last axis, and for odd N, it means 1 extra entry. While these are required
        to do the FFT, the extra spaces don't actually get used, and the indexing
        macros (eg. R_FFT_INDEX) skip these extra bits to index the truly used array.
*/
#include <gsl/gsl_rng.h>
#include <math.h>

#include "InputParameters.h"

// -------------------------------------------------------------------------------------
// Convenience Constants
// -------------------------------------------------------------------------------------
#define VOLUME                                                                 \
    (simulation_options_global->BOX_LEN * simulation_options_global->BOX_LEN * \
     simulation_options_global->NON_CUBIC_FACTOR *                             \
     simulation_options_global->BOX_LEN)  // in Mpc^3
#define BOXLEN_PARA \
    (simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->BOX_LEN)  // in Mpc

// -------------------------------------------------------------------------------------
// Convenience Macros for hi-resolution boxes
// -------------------------------------------------------------------------------------
#define D_PARA                                                         \
    (unsigned long long)(simulation_options_global->NON_CUBIC_FACTOR * \
                         simulation_options_global->DIM)  // the long long dimension
#define TOT_NUM_PIXELS                                                                     \
    ((unsigned long long)simulation_options_global->DIM * simulation_options_global->DIM * \
     D_PARA)  // no padding

// Fourier-Transform numbers
#define TOT_FFT_NUM_PIXELS                                                                        \
    ((unsigned long long)simulation_options_global->DIM * simulation_options_global->DIM * 2llu * \
     (D_PARA / 2 + 1llu))
#define KSPACE_NUM_PIXELS                                                                  \
    ((unsigned long long)simulation_options_global->DIM * simulation_options_global->DIM * \
     (D_PARA / 2 + 1llu))

// -------------------------------------------------------------------------------------
// Convenience Macros for low-resolution boxes
// -------------------------------------------------------------------------------------
#define HII_D_PARA                                                     \
    (unsigned long long)(simulation_options_global->NON_CUBIC_FACTOR * \
                         simulation_options_global->HII_DIM)
#define HII_TOT_NUM_PIXELS                                                        \
    (unsigned long long)((unsigned long long)simulation_options_global->HII_DIM * \
                         simulation_options_global->HII_DIM * HII_D_PARA)

// Fourier-Transform numbers
#define HII_TOT_FFT_NUM_PIXELS                                                                     \
    ((unsigned long long)simulation_options_global->HII_DIM * simulation_options_global->HII_DIM * \
     2llu * (HII_D_PARA / 2 + 1llu))
#define HII_KSPACE_NUM_PIXELS                                                                      \
    ((unsigned long long)simulation_options_global->HII_DIM * simulation_options_global->HII_DIM * \
     (HII_D_PARA / 2 + 1llu))

void wrap_position(double pos[3], double size[3]);
void wrap_coord(int idx[3], int size[3]);
void random_point_in_sphere(double centre[3], double radius, gsl_rng *rng, double *point);
void random_point_in_cell(int idx[3], double cell_len, gsl_rng *rng, double *point);

// Indexing a 3D array stored in a 1D array
inline unsigned long long grid_index_general(int x, int y, int z, int dim[3]) {
    return (z) + (dim[2] + 0llu) * (y + (dim[1] + 0llu) * x);
}

// Indexing a 3D array stored in a 1D array, where the 3D array is the real-space
// representation of a Fourier Transform (with padding on the last axis)
inline unsigned long long grid_index_fftw_r(int x, int y, int z, int dim[3]) {
    return (z) + 2llu * (dim[2] / 2 + 1llu) * (y + (dim[1] + 0llu) * x);
}

// Indexing a 3D array stored in a 1D array, where the 3D array is the complex-space
// representation of a Fourier Transform (with padding on the last axis)
inline unsigned long long grid_index_fftw_c(int x, int y, int z, int dim[3]) {
    return (z) + (dim[2] / 2 + 1llu) * (y + (dim[1] + 0llu) * x);
}

// Convert a position on [0,BOX_LEN] to an index for a particular cell size.
// NOTE: assumes the cell at idx == 0 is *centred* at (0,0,0), and so adds 0.5
inline void pos_to_index(double pos[3], double cell_size_inv, int idx[3]) {
    idx[0] = (int)(pos[0] * cell_size_inv + 0.5);
    idx[1] = (int)(pos[1] * cell_size_inv + 0.5);
    idx[2] = (int)(pos[2] * cell_size_inv + 0.5);
}

// Convert an index on one grid to the corresponding index on a grid of differing resolution
// 0.5 is added on the **OUTPUT** resolution becase dim_ratio is not necessarily an integer
inline void resample_index(int idx_in[3], double dim_ratio, int idx_out[3]) {
    idx_out[0] = (int)(idx_in[0] * dim_ratio + 0.5);
    idx_out[1] = (int)(idx_in[1] * dim_ratio + 0.5);
    idx_out[2] = (int)(idx_in[2] * dim_ratio + 0.5);
}

inline double index_to_k(int idx, double len, int dim) {
    // Convert an index to a k-mode, assuming box of length len and dim pixels
    double buf = (idx <= dim / 2) ? idx : (idx - dim);
    return buf * 2. * M_PI / len;
}
