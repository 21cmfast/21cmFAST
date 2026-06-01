
#ifndef _INITCONDITIONS_H
#define _INITCONDITIONS_H

#include <gsl/gsl_rng.h>

#include "InputParameters.h"
#include "OutputStructs.h"
#include "indexing.h"

int ComputeInitialConditions(random_huge random_seed, InitialConditions *boxes);

#endif
