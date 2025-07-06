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
#define DELTA_K (TWOPI / simulation_options_global->BOX_LEN)
#define DELTA_K_PARA \
    (TWOPI / (simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->BOX_LEN))

// -------------------------------------------------------------------------------------
// Convenience Macros for hi-resolution boxes
// -------------------------------------------------------------------------------------
#define D (long long)simulation_options_global->DIM  // the long long dimension
#define D_PARA                                                \
    (long long)(simulation_options_global->NON_CUBIC_FACTOR * \
                simulation_options_global->DIM)  // the long long dimension
#define MIDDLE (simulation_options_global->DIM / 2)
#define MIDDLE_PARA \
    (simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->DIM / 2)
#define MID ((long long)MIDDLE)
#define MID_PARA ((long long)MIDDLE_PARA)
#define TOT_NUM_PIXELS ((unsigned long long)(D * D * D_PARA))  // no padding

// Fourier-Transform numbers
#define TOT_FFT_NUM_PIXELS ((unsigned long long)(D * D * 2llu * (MID_PARA + 1llu)))
#define KSPACE_NUM_PIXELS ((unsigned long long)(D * D * (MID_PARA + 1llu)))

// INDEXING MACROS
// for 3D complex array
#define C_INDEX(x, y, z) ((unsigned long long)((z) + (MID_PARA + 1llu) * ((y) + D * (x))))
// for 3D real array with the FFT padding
#define R_FFT_INDEX(x, y, z) \
    ((unsigned long long)((z) + 2llu * (MID_PARA + 1llu) * ((y) + D * (x))))
// for 3D real array with no padding
#define R_INDEX(x, y, z) ((unsigned long long)((z) + D_PARA * ((y) + D * (x))))

// -------------------------------------------------------------------------------------
// Convenience Macros for low-resolution boxes
// -------------------------------------------------------------------------------------
#define HII_D (long long)(simulation_options_global->HII_DIM)
#define HII_D_PARA \
    (long long)(simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->HII_DIM)
#define HII_MIDDLE (simulation_options_global->HII_DIM / 2)
#define HII_MIDDLE_PARA \
    (simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->HII_DIM / 2)
#define HII_MID ((long long)HII_MIDDLE)
#define HII_MID_PARA ((long long)HII_MIDDLE_PARA)
#define HII_TOT_NUM_PIXELS (unsigned long long)(HII_D * HII_D * HII_D_PARA)

// Fourier-Transform numbers
#define HII_TOT_FFT_NUM_PIXELS ((unsigned long long)(HII_D * HII_D * 2llu * (HII_MID_PARA + 1llu)))
#define HII_KSPACE_NUM_PIXELS ((unsigned long long)(HII_D * HII_D * (HII_MID_PARA + 1llu)))

// INDEXING MACROS
// for 3D complex array
#define HII_C_INDEX(x, y, z) \
    ((unsigned long long)((z) + (HII_MID_PARA + 1llu) * ((y) + HII_D * (x))))
// for 3D real array with the FFT padding
#define HII_R_FFT_INDEX(x, y, z) \
    ((unsigned long long)((z) + 2llu * (HII_MID_PARA + 1llu) * ((y) + HII_D * (x))))
// for 3D real array with no padding
#define HII_R_INDEX(x, y, z) ((unsigned long long)((z) + HII_D_PARA * ((y) + HII_D * (x))))
