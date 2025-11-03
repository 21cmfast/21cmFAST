#ifndef _HALOBOX_H
#define _HALOBOX_H

#include "HaloCatalog.h"
#include "InitialConditions.h"
#include "InputParameters.h"
#include "IonisationBox.h"
#include "OutputStructs.h"
#include "PerturbedHaloCatalog.h"
#include "SpinTemperatureBox.h"
#include "scaling_relations.h"

// struct holding each halo property we currently need.
// This is only used for both averages over the box/catalogues
//   as well as an individual halo's properties
typedef struct HaloProperties {
    double count;  // from integral
    double halo_mass;
    double stellar_mass;
    double halo_sfr;
    double stellar_mass_mini;
    double sfr_mini;
    double fescweighted_sfr;
    double n_ion;
    double halo_xray;
    double metallicity;
    double m_turn_acg;
    double m_turn_mcg;
    double m_turn_reion;
} HaloProperties;

// TODO: apply this constant struct to the EvaluateX functions in interp_tables.c,
//  the integral_wrappers.c functions, and other places where the tables are called
//  (probably not hmf.c)
typedef struct IntegralCondition {
    double redshift;
    double growth_factor;
    double M_min;
    double lnM_min;
    double M_max;
    double lnM_max;
    double M_cell;
    double lnM_cell;
    double sigma_cell;
} IntegralCondition;

void set_integral_constants(IntegralCondition *consts, double redshift, double M_min, double M_max,
                            double M_cell);

int ComputeHaloBox(double redshift, InitialConditions *ini_boxes, HaloCatalog *halos,
                   TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, HaloBox *grids);

void get_cell_integrals(double dens, double l10_mturn_a, double l10_mturn_m,
                        ScalingConstants *consts, IntegralCondition *int_consts,
                        HaloProperties *properties);
void set_halo_properties(double halo_mass, double M_turn_a, double M_turn_m,
                         ScalingConstants *consts, double *input_rng, HaloProperties *output);

int convert_halo_props(double redshift, InitialConditions *ics, TsBox *prev_ts,
                       IonizedBox *prev_ion, HaloCatalog *halo_catalog,
                       PerturbedHaloCatalog *halo_catalog_out);
#endif
