// This file holds useful functions for indexing and wrapping coordinates in 3D space
#include "indexing.h"

#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "InputParameters.h"
#include "logger.h"

// Minutia: Testing showed that the while loops were faster than a modulus
// for uniformly distributed positions on a grid size greater than 8
void wrap_position(double pos[3], double size[3]) {
    // wrap the coordinates to the box size
    while (pos[0] >= size[0]) {
        pos[0] -= size[0];
    }
    while (pos[0] < 0) {
        pos[0] += size[0];
    }
    while (pos[1] >= size[1]) {
        pos[1] -= size[1];
    }
    while (pos[1] < 0) {
        pos[1] += size[1];
    }
    while (pos[2] >= size[2]) {
        pos[2] -= size[2];
    }
    while (pos[2] < 0) {
        pos[2] += size[2];
    }
    return;
}

void wrap_coord(int idx[3], int size[3]) {
    // Integer version of the wrap_position function
    while (idx[0] >= size[0]) {
        idx[0] -= size[0];
    }
    while (idx[0] < 0) {
        idx[0] += size[0];
    }
    while (idx[1] >= size[1]) {
        idx[1] -= size[1];
    }
    while (idx[1] < 0) {
        idx[1] += size[1];
    }
    while (idx[2] >= size[2]) {
        idx[2] -= size[2];
    }
    while (idx[2] < 0) {
        idx[2] += size[2];
    }
    return;
}

void random_point_in_sphere(double centre[3], double radius, gsl_rng *rng, double *point) {
    // generate a random point in a sphere of given radius and centre
    double x1, y1, z1, d1, r1;
    x1 = 2 * gsl_rng_uniform(rng) - 1;
    y1 = 2 * gsl_rng_uniform(rng) - 1;
    z1 = 2 * gsl_rng_uniform(rng) - 1;
    d1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    r1 = gsl_rng_uniform(rng);
    point[0] = centre[0] + (radius * r1 * x1 / d1);
    point[1] = centre[1] + (radius * r1 * y1 / d1);
    point[2] = centre[2] + (radius * r1 * z1 / d1);
}

void random_point_in_cell(int idx[3], double cell_len, gsl_rng *rng, double *point) {
    // generate a random point in a cell given the coordinates and length the cell
    // NOTE: assumes cell at idx == 0 is *centred* at (0,0,0)
    double randbuf;
    randbuf = gsl_rng_uniform(rng) - 0.5;
    point[0] = (idx[0] + randbuf) * cell_len;

    randbuf = gsl_rng_uniform(rng) - 0.5;
    point[1] = (idx[1] + randbuf) * cell_len;

    randbuf = gsl_rng_uniform(rng) - 0.5;
    point[2] = (idx[2] + randbuf) * cell_len;
    return;
}
