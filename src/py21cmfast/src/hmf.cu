#include <cuda_runtime.h>
#include <math.h>

#include "Constants.h"
#include "hmf.cuh"

__device__ double sheth_delc_fixed(double del, double sig)
{
    return sqrt(JENKINS_a) * del * (1. + JENKINS_b * pow(sig * sig / (JENKINS_a * del * del), JENKINS_c));
}

// Get the relevant excursion set barrier density given the user-specified HMF
__device__ double get_delta_crit(int HMF, double sigma, double growthf)
{
    if (HMF == 4)
        return DELTAC_DELOS;
    if (HMF == 1)
        return sheth_delc_fixed(Deltac / growthf, sigma) * growthf;

    return Deltac;
}
