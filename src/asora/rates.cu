#include "rates.cuh"

// ========================================================================
// Define macros. Could be passed as parameters but are kept as
// compile-time constants for now
// ========================================================================
#define TAU_PHOTO_LIMIT 1.0e-7                      // Limit to consider a cell "optically thin/thick"
#define S_STAR_REF 1e48                             // Reference ionizing flux (strength of source is given in this unit)

// ========================================================================
// Compute photoionization rate from in/out column density by looking up
// values of the integral ∫L_v*e^(-τ_v)/hv in precalculated tables. These
// tables are assumed to have been copied to device memory in advance using
// photo_table_to_device()
// ========================================================================
__device__ double photoion_rates_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig,
    const double* photo_thin_table,const double* photo_thick_table,const double & minlogtau,const double & dlogtau,const int& NumTau)
{
    // Compute optical depth and ionization rate depending on whether the cell is optically thick or thin
    double tau_in = coldens_in * sig;
    double tau_out = coldens_out * sig;

    double prefact = strength / Vfact;

    // PH (08.10.23) I'm confused about the way the rates are calculated differently for thin/thick
    // cells. The following is taken verbatim from radiation_photoionrates.F90 lines 276 - 303
    // but without true understanding... Names are slightly different to simpify notatio
    double phi_photo_in = prefact * photo_lookuptable(photo_thick_table,tau_in,minlogtau,dlogtau,NumTau);

    if (abs(tau_out-tau_in) > TAU_PHOTO_LIMIT )
    {
        double phi_photo_out = prefact * photo_lookuptable(photo_thick_table,tau_out,minlogtau,dlogtau,NumTau);
        return phi_photo_in - phi_photo_out;
    }
    else
    {
        return prefact * (tau_out-tau_in) * photo_lookuptable(photo_thin_table,tau_out,minlogtau,dlogtau,NumTau);
    }
    // double phi_photo_out = prefact * photo_lookuptable(photo_thick_table,tau_out,minlogtau,dlogtau,NumTau);
    // return phi_photo_in - phi_photo_out;
}

// ========================================================================
// Grey-opacity test case photoionization rate, computed from analytical
// expression rather than using tables. To use this version, compile
// with the -DGREY_NOTABLES flag
// ========================================================================
__device__ double photoion_rates_test_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig)
{
    // Compute optical depth and ionization rate depending on whether the cell is optically thick or thin
    double tau_in = coldens_in * sig;
    double tau_out = coldens_out * sig;

    

    // If cell is optically thick
    if (fabs(tau_out - tau_in) > TAU_PHOTO_LIMIT)
        // return strength * INV4PI / (Vfact * nHI) * (exp(-tau_in) - exp(-tau_out));
        return (strength*S_STAR_REF / (Vfact)) * (exp(-tau_in) - exp(-tau_out));
    // If cell is optically thin
    else
        // return strength * INV4PI * sig * (tau_out - tau_in) / (Vfact) * exp(-tau_in);
        return (strength*S_STAR_REF / (Vfact)) * (tau_out - tau_in) * exp(-tau_in);
}

// ========================================================================
// Utility function to look up the integral value corresponding to an
// optical depth τ by doing linear interpolation.
// ========================================================================
__device__ double photo_lookuptable(const double* table,const double & tau,const double & minlogtau,const double & dlogtau,const int & NumTau)
{
    double logtau;
    double real_i, residual;
    int i0, i1;
    // Find table index and do linear interpolation
    // Recall that tau(0) = 0 and tau(1:NumTau) ~ logspace(minlogtau,maxlogtau) (so in reality the table has size NumTau+1)
    logtau = log10(max(1.0e-20,tau));
    real_i = min(float(NumTau),max(0.0,1.0+(logtau-minlogtau)/dlogtau));
    i0 = int( real_i );
    i1 = min(NumTau, i0+1);
    residual = real_i - double(i0);
    return table[i0] + residual*(table[i1] - table[i0]);
}