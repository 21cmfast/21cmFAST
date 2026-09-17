#ifndef _EMISSIVITYFIELDS_H
#define _EMISSIVITYFIELDS_H

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
    double halo_mass;
    double stellar_mass_acg;
    double stellar_mass_mcg;
    double sfr_acg;
    double sfr_mcg;
    double fesc_weighted_sfr;
    double n_ion;
    double xray_luminosity;
    double metallicity;
} HaloProperties;

// Similar struct to the above, but contains mostly density-fields, evaluated from
// integral over the lower regime of the halo mass function
typedef struct IntegralProperties {
    double halo_number;
    double halo_mass_density;
    double stellar_mass_density_acg;
    double stellar_mass_density_mcg;
    double sfrd_acg;
    double sfrd_mcg;
    double fesc_weighted_sfrd;
    double n_ion;
    double xray_emissivity;
} IntegralProperties;

int ComputeEmissivityFields(double redshift, InitialConditions *ini_boxes,
                            PerturbedField *perturbed_field, HaloCatalog *halos,
                            TsBox *previous_spin_temp, IonizedBox *previous_ionize_box,
                            EmissivityFields *grids);

void get_cell_integrals(double dens, double M_min, double M_max, double l10_mturn_acg,
                        double l10_mturn_mcg, ScalingConstants *consts,
                        IntegralProperties *properties);
void set_halo_properties(double halo_mass, double M_turn_acg, double M_turn_mcg,
                         ScalingConstants *consts, double *input_rng, HaloProperties *output);

int convert_halo_props(double redshift, InitialConditions *ics, TsBox *prev_ts,
                       IonizedBox *prev_ion, HaloCatalog *halo_catalog,
                       PerturbedHaloCatalog *halo_catalog_out);
#endif
