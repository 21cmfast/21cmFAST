#include <cuda_runtime.h>

#ifndef _HMF_CUH
#define _HMF_CUH

// define macros
#ifndef JENKINS_a
#define JENKINS_a (0.73) // Jenkins+01, SMT has 0.707
#endif

#ifndef JENKINS_b
#define JENKINS_b (0.34) // Jenkins+01 fit from Barkana+01, SMT has 0.5
#endif

#ifndef JENKINS_c
#define JENKINS_c (0.81) // Jenkins+01 from from Barkana+01, SMT has 0.6
#endif

// #ifdef __CUDA_ARCH__
__device__ double sheth_delc_fixed(double del, double sig);
__device__ double get_delta_crit(int HMF, double sigma, double growthf);
// #endif

#endif
