#include "rng.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

// TODO: seed_rng_threads is slow, and isn't great to use every snapshot in Stochasticity.c or
// IonisationBox.c
//       There are two remedies:
//       - make it faster (by reducing the number of generated ints, I don't know why it's done that
//       way)
//       - have an rng struct which tracks its allocation and is only seeded once
//   The second option creates difficulties i.e the same seed initialised at different points
//   (for example by stopping/starting a run) can have different results
//   seed_rng threads generates random seeds for each thread by choosing N_THREADS ints
//   from an array of INT_MAX/16 (100 million?) ints. This array is shuffled, then each int is
//   used as a seed with a looping list of 5 different gsl initialisers.
//   In order to keep consistency with master I'm leaving it as is for now, but I really want to
//   understand why it was done this way.
void seed_rng_threads(gsl_rng *rng_arr[], unsigned long long int seed) {
    // An RNG for generating seeds for multithreading
    gsl_rng *rseed = gsl_rng_alloc(gsl_rng_mt19937);

    gsl_rng_set(rseed, seed);

    unsigned int seeds[matter_params_global->N_THREADS];

    // For multithreading, seeds for the RNGs are generated from an initial RNG (based on the input
    // random_seed) and then shuffled (Author: Fred Davies)
    int num_int = INT_MAX / 16;
    int i, thread_num;
    // Some large number of possible integers
    unsigned int *many_ints = (unsigned int *)malloc((size_t)(num_int * sizeof(unsigned int)));
    for (i = 0; i < num_int; i++) {
        many_ints[i] = i;
    }

    // Populate the seeds array from the large list of integers
    gsl_ran_choose(rseed, seeds, matter_params_global->N_THREADS, many_ints, num_int,
                   sizeof(unsigned int));
    // Shuffle the randomly selected integers
    gsl_ran_shuffle(rseed, seeds, matter_params_global->N_THREADS, sizeof(unsigned int));

    int checker = 0;
    // seed the random number generators
    for (thread_num = 0; thread_num < matter_params_global->N_THREADS; thread_num++) {
        switch (checker) {
            case 0:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 1:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_gfsr4);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 2:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_cmrg);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 3:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_mrg);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
            case 4:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_taus2);
                gsl_rng_set(rng_arr[thread_num], seeds[thread_num]);
                break;
        }  // end switch

        checker += 1;

        if (checker == 5) {
            checker = 0;
        }
    }

    free(many_ints);
    gsl_rng_free(rseed);
}

// Returns true if all elements of an array are unique, O(N^2) but N is small
bool array_unique(unsigned int array[], int array_size) {
    int idx_0, idx_1;
    for (idx_0 = 0; idx_0 < array_size; idx_0++) {
        for (idx_1 = 0; idx_1 < idx_0; idx_1++) {
            if (array[idx_0] == array[idx_1]) return false;
        }
    }
    return true;
}

// Samples a number of integers, repeating the samples if any repeat
void sample_n_unique_integers(unsigned int n, unsigned long long int seed, unsigned int array[]) {
    // An RNG for generating seeds for multithreading
    gsl_rng *rseed = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rseed, seed);

    int idx;
    do {
        for (idx = 0; idx < n; idx++) {
            array[idx] = gsl_rng_get(rseed);
        }
    } while (!array_unique(array, n));

    gsl_rng_free(rseed);
}

// I'm putting a faster version of the RNG seeding here to use when we need one seeding per
// snapshot.
//   This is exactly the same as above except instead of generating a huge list of ints to select
//   from, it generates seeds using gsl_rng_uniform_int(), which means there *can* be a repetition.
//   This should be very rare, and even when it occurs, it will usually use a different initialiser,
//   as well as be applied to different cells/halos, so I can't forsee any issues.
// Just in case I'm not using it for the initial conditions, which is okay since the slower version
// is only used once
void seed_rng_threads_fast(gsl_rng *rng_arr[], unsigned long long int seed) {
    unsigned int seed_thread;
    unsigned int st_arr[1000];  // surely nobody uses > 1000 threads
    int thread_num, checker;

    checker = 0;
    // seed the random number generators
    sample_n_unique_integers(matter_params_global->N_THREADS, seed, st_arr);

    for (thread_num = 0; thread_num < matter_params_global->N_THREADS; thread_num++) {
        seed_thread = st_arr[thread_num];
        switch (checker) {
            case 0:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
                gsl_rng_set(rng_arr[thread_num], seed_thread);
                break;
            case 1:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_gfsr4);
                gsl_rng_set(rng_arr[thread_num], seed_thread);
                break;
            case 2:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_cmrg);
                gsl_rng_set(rng_arr[thread_num], seed_thread);
                break;
            case 3:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_mrg);
                gsl_rng_set(rng_arr[thread_num], seed_thread);
                break;
            case 4:
                rng_arr[thread_num] = gsl_rng_alloc(gsl_rng_taus2);
                gsl_rng_set(rng_arr[thread_num], seed_thread);
                break;
        }  // end switch
        checker += 1;
        if (checker == 5) {
            checker = 0;
        }
    }
}

void free_rng_threads(gsl_rng *rng_arr[]) {
    int ii;
    for (ii = 0; ii < matter_params_global->N_THREADS; ii++) {
        gsl_rng_free(rng_arr[ii]);
    }
}
