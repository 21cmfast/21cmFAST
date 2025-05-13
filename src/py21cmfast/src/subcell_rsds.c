#include "subcell_rsds.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "indexing.h"

double compute_rsds(float *box_in, float *los_displacement, int I1, int J1, int K1, int n_subcells, int n_threads, float *box_out) {

    double ave;
    int i, ii, j, k;
    float d1_low, d1_high, d2_low, d2_high;
    float subcell_displacement;
    float RSD_pos_new, RSD_pos_new_boundary_low, RSD_pos_new_boundary_high;
    float fraction_within, fraction_outside, cell_distance;
    
    omp_set_num_threads(n_threads);
    
    float **delta_T_RSD_LOS = (float **)calloc(n_threads, sizeof(float *));
    for (i = 0; i < n_threads; i++) {
        delta_T_RSD_LOS[i] = (float *)calloc(K1, sizeof(float));
    }
    
    float *x_pos = calloc(n_subcells, sizeof(float));
    
    // normalised units of cell length. 0 equals beginning of cell, 1 equals end of cell
    // These are the normalized sub-cell central positions (x_pos) between 0 and 1
    for (ii = 0; ii < n_subcells; ii++) {
        x_pos[ii] = ((float)ii + 1. / 2.) / (float)(n_subcells);
    }
    //printf("C code:\n");
    #pragma omp parallel shared(I1, J1, K1,                                                        \
            delta_T_RSD_LOS, box_in, box_out, x_pos,los_displacement, n_subcells)               \ 
            private(i, j, k, ii, d1_low, d2_low, d1_high, d2_high,                              \
                    subcell_displacement, RSD_pos_new,                                          \
                    RSD_pos_new_boundary_low, RSD_pos_new_boundary_high,                        \
                    cell_distance, fraction_outside, fraction_within)                           \
        num_threads(n_threads)
        {
    #pragma omp for reduction(+ : ave)
            for (i = 0; i < I1; i++) {
                for (j = 0; j < J1; j++) {
                    // Generate the optical-depth for the specific line-of-sight with R.S.D
                    for (k = 0; k < K1; k++) {
                        delta_T_RSD_LOS[omp_get_thread_num()][k] = 0.0;
                    }
                    for (k = 0; k < K1; k++) {
                        if (k == 0) {
                            d1_low = los_displacement[HII_R_INDEX(i, j, K1 - 1)];
                            d2_low = los_displacement[HII_R_INDEX(i, j, k)];
                        } else {
                            d1_low = los_displacement[HII_R_INDEX(i, j, k - 1)];
                            d2_low = los_displacement[HII_R_INDEX(i, j, k)];
                        }
                        /*if (i == 0 & j ==0 & k < K1) {
                            printf("\tFor cell %d:\n",k);
                        }*/
                        // Displacements (converted from velocity) for the original cell centres
                        // straddling half of the sub-cells (cell after)
                        if (k == (K1 - 1)) {
                            d1_high = los_displacement[HII_R_INDEX(i, j, k)];
                            d2_high = los_displacement[HII_R_INDEX(i, j, 0)];
                        } else {
                            d1_high = los_displacement[HII_R_INDEX(i, j, k)];
                            d2_high = los_displacement[HII_R_INDEX(i, j, k + 1)];
                        }
                        
                        for (ii = 0; ii < n_subcells; ii++) {
                            // linearly interpolate the displacements to determine the corresponding
                            // displacements of the sub-cells Checking of 0.5 is for determining if
                            // we are left or right of the mid-point of the original cell (for the
                            // linear interpolation of the displacement) to use the appropriate cell
    
                            if (x_pos[ii] <= 0.5) {
                                subcell_displacement = d1_low + (x_pos[ii] + 0.5) * (d2_low - d1_low);
                            } else {
                                subcell_displacement = d1_high + (x_pos[ii] - 0.5) * (d2_high - d1_high);
                            }
                            /*if (i==0 && j==0 && k < K1){
                                printf("\t\tFor ii=%d, subcell_displacement=%f\n",ii,subcell_displacement);
                            }*/
                            // The new centre of the sub-cell post R.S.D displacement.
                            // Normalised to units of cell width for determining it's displacement
                            /*
                            Note to convert the velocity v, to a displacement in redshift space, convert
                            from s -> r + (1+z)*v/H(z). To convert the velocity within the array v to km/s, it
                            is a*dD/dt*delta. Where the scale factor a comes from the continuity equation
                            The array v as defined in 21cmFAST is (ik/k^2)*dD/dt*delta, as it is defined as a
                            comoving quantity (scale factor is implicit). However, the conversion between real
                            and redshift space also picks up a scale factor, therefore the scale factors drop
                            out and therefore the displacement of the sub-cells is purely determined from the
                            array, v and the Hubble factor: v/H.
                            */
                            RSD_pos_new = x_pos[ii] + subcell_displacement;
                            // The sub-cell boundaries of the sub-cell, for determining the
                            // fractional contribution of the sub-cell to neighbouring cells when
                            // the sub-cell straddles two cell positions
                            RSD_pos_new_boundary_low =
                                RSD_pos_new - 1. / 2. / (float)(n_subcells);
                            RSD_pos_new_boundary_high =
                                RSD_pos_new + 1. / 2. / (float)(n_subcells);
    
                            if (RSD_pos_new_boundary_low >= 0.0 &&
                                RSD_pos_new_boundary_high < 1.0) {
                                // sub-cell has remained in the original cell (just add it back to
                                // the original cell)
    
                                delta_T_RSD_LOS[omp_get_thread_num()][k] +=
                                    box_in[HII_R_INDEX(i, j, k)] /
                                    ((float)n_subcells);
                            } else if (RSD_pos_new_boundary_low < 0.0 &&
                                        RSD_pos_new_boundary_high < 0.0) {
                                // sub-cell has moved completely into a new cell (toward the
                                // observer)
    
                                // determine how far the sub-cell has moved in units of original
                                // cell boundary
                                cell_distance = ceil(fabs(RSD_pos_new_boundary_low)) - 1.;
    
                                // Determine the location of the sub-cell relative to the original
                                // cell binning
                                if (fabs(RSD_pos_new_boundary_high) > cell_distance) {
                                    // sub-cell is entirely contained within the new cell (just add
                                    // it to the new cell)
    
                                    // check if the new cell position is at the edge of the box. If
                                    // so, periodic boundary conditions
                                    if (k < ((int)cell_distance + 1)) {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k - ((int)cell_distance + 1) +
                                                        K1] +=
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    } else {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k - ((int)cell_distance + 1)] +=
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    }
                                } else {
                                    // sub-cell is partially contained within the cell
    
                                    // Determine the fraction of the sub-cell which is in either of
                                    // the two original cells
                                    fraction_outside =
                                        (fabs(RSD_pos_new_boundary_low) - cell_distance) * (float)(n_subcells);
                                    fraction_within = 1. - fraction_outside;
    
                                    // Check if the first part of the sub-cell is at the box edge
                                    if (k < (((int)cell_distance))) {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k - ((int)cell_distance) + K1] +=
                                            fraction_within *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    } else {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k - ((int)cell_distance)] +=
                                            fraction_within *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    }
                                    // Check if the second part of the sub-cell is at the box edge
                                    if (k < (((int)cell_distance + 1))) {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k - ((int)cell_distance + 1) +
                                                        K1] +=
                                            fraction_outside *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    } else {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k - ((int)cell_distance + 1)] +=
                                            fraction_outside *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    }
                                }
                            } else if (RSD_pos_new_boundary_low < 0.0 &&
                                        (RSD_pos_new_boundary_high > 0.0 &&
                                        RSD_pos_new_boundary_high < 1.0)) {
                                // sub-cell has moved partially into a new cell (toward the
                                // observer)
    
                                // Determine the fraction of the sub-cell which is in either of the
                                // two original cells
                                fraction_within =
                                    RSD_pos_new_boundary_high * (float)(n_subcells);
                                fraction_outside = 1. - fraction_within;
    
                                // Check the periodic boundaries conditions and move the fraction of
                                // each sub-cell to the appropriate new cell
                                if (k == 0) {
                                    delta_T_RSD_LOS[omp_get_thread_num()][K1 - 1] +=
                                        fraction_outside *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                    delta_T_RSD_LOS[omp_get_thread_num()][k] +=
                                        fraction_within *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                } else {
                                    delta_T_RSD_LOS[omp_get_thread_num()][k - 1] +=
                                        fraction_outside *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                    delta_T_RSD_LOS[omp_get_thread_num()][k] +=
                                        fraction_within *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                }
                            } else if ((RSD_pos_new_boundary_low >= 0.0 &&
                                        RSD_pos_new_boundary_low < 1.0) &&
                                        (RSD_pos_new_boundary_high >= 1.0)) {
                                // sub-cell has moved partially into a new cell (away from the
                                // observer)
    
                                // Determine the fraction of the sub-cell which is in either of the
                                // two original cells
                                fraction_outside =
                                    (RSD_pos_new_boundary_high - 1.) * (float)(n_subcells);
                                fraction_within = 1. - fraction_outside;
    
                                // Check the periodic boundaries conditions and move the fraction of
                                // each sub-cell to the appropriate new cell
                                if (k == (K1 - 1)) {
                                    delta_T_RSD_LOS[omp_get_thread_num()][k] +=
                                        fraction_within *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                    delta_T_RSD_LOS[omp_get_thread_num()][0] +=
                                        fraction_outside *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                } else {
                                    delta_T_RSD_LOS[omp_get_thread_num()][k] +=
                                        fraction_within *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                    delta_T_RSD_LOS[omp_get_thread_num()][k + 1] +=
                                        fraction_outside *
                                        box_in[HII_R_INDEX(i, j, k)] /
                                        ((float)n_subcells);
                                }
                            } else {
                                // sub-cell has moved completely into a new cell (away from the
                                // observer)
    
                                // determine how far the sub-cell has moved in units of original
                                // cell boundary
                                cell_distance = floor(fabs(RSD_pos_new_boundary_high));
    
                                if (RSD_pos_new_boundary_low >= cell_distance) {
                                    // sub-cell is entirely contained within the new cell (just add
                                    // it to the new cell)
    
                                    // check if the new cell position is at the edge of the box. If
                                    // so, periodic boundary conditions
                                    if (k > (K1 - 1 - (int)cell_distance)) {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k + (int)cell_distance - K1] +=
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    } else {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k + (int)cell_distance] +=
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    }
                                } else {
                                    // sub-cell is partially contained within the cell
    
                                    // Determine the fraction of the sub-cell which is in either of
                                    // the two original cells
                                    fraction_outside =
                                        (RSD_pos_new_boundary_high - cell_distance) * (float)(n_subcells);
                                    fraction_within = 1. - fraction_outside;
    
                                    // Check if the first part of the sub-cell is at the box edge
                                    if (k > (K1 - 1 - ((int)cell_distance - 1))) {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k + (int)cell_distance - 1 - K1] +=
                                            fraction_within *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    } else {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k + (int)cell_distance - 1] +=
                                            fraction_within *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    }
                                    // Check if the second part of the sub-cell is at the box edge
                                    if (k > (K1 - 1 - ((int)cell_distance))) {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k + (int)cell_distance - K1] +=
                                            fraction_outside *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    } else {
                                        delta_T_RSD_LOS[omp_get_thread_num()]
                                                        [k + (int)cell_distance] +=
                                            fraction_outside *
                                            box_in[HII_R_INDEX(i, j, k)] /
                                            ((float)n_subcells);
                                    }
                                }
                            }
                        }
                    }
    
                    for (k = 0; k < K1; k++) {
                        box_out[HII_R_INDEX(i, j, k)] =
                            delta_T_RSD_LOS[omp_get_thread_num()][k];
    
                        ave += delta_T_RSD_LOS[omp_get_thread_num()][k];
                    }
                }
            }
        }
        free(x_pos);
        for (i = 0; i < n_threads; i++) {
            free(delta_T_RSD_LOS[i]);
        }
        free(delta_T_RSD_LOS);
    
        ave /= (float)HII_TOT_NUM_PIXELS;
}

double apply_subcell_rsds(IonizedBox *ionized_box, BrightnessTemp *box, float redshift,
                          TsBox *spin_temp, float T_rad, float *v, float H) {
    int i, ii, j, k;
    double ave;
    ave = 0.;

    omp_set_num_threads(simulation_options_global->N_THREADS);

    float cellsize = simulation_options_global->BOX_LEN / (float)simulation_options_global->HII_DIM;

    float *los_displacement = (float *)calloc(HII_TOT_NUM_PIXELS, sizeof(float));
    #pragma omp parallel shared(los_displacement,v,H,cellsize) private(i, j, k) \ 
    num_threads(simulation_options_global -> N_THREADS)
    {
    #pragma omp for 
        for (i = 0; i < simulation_options_global->HII_DIM; i++) {
            for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                for (k = 0; k < HII_D_PARA; k++) {
                    los_displacement[HII_R_INDEX(i, j, k)] = v[HII_R_FFT_INDEX(i, j, k)] / H / cellsize;
                }
            }
        }
    }

    ave = compute_rsds(box->brightness_temp,
                      los_displacement,
                      simulation_options_global->HII_DIM, 
                      simulation_options_global->HII_DIM, 
                      HII_D_PARA, 
                      astro_params_global->N_RSD_STEPS, 
                      simulation_options_global->N_THREADS, 
                      box->brightness_temp);
    return (ave);
}


