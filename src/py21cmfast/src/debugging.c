/*
    The following functions are simply for testing the exception framework
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"
#include "InputParameters.h"
#include "OutputStructs.h"

#include "debugging.h"

//definition of the global exception context
struct exception_context the_exception_context[1];

void FunctionThatThrows(){
    Throw(PhotonConsError);
}

int SomethingThatCatches(bool sub_func){
    // A simple function that catches a thrown error.
    int status;
    Try{
        if(sub_func) FunctionThatThrows();
        else Throw(PhotonConsError);
    }
    Catch(status){
        return status;
    }
    return 0;
}

int FunctionThatCatches(bool sub_func, bool pass, double *result){
    int status;
    if(!pass){
        Try{
            if(sub_func) FunctionThatThrows();
            else Throw(PhotonConsError);
        }
        Catch(status){
            LOG_DEBUG("Caught the problem with status %d.", status);
            return status;
        }
    }
    *result = 5.0;
    return 0;
}
void writeUserParams(UserParams *p){
    LOG_INFO("\n        UserParams:\n"
                "       HII_DIM=%4d, DIM=%4d, BOX_LEN=%8.3f, NON_CUBIC_FACTOR=%8.3f,\n"
                "       HMF=%2d, POWER_SPECTRUM=%2d, USE_RELATIVE_VELOCITIES=%1d, N_THREADS=%2d,\n"
                "       PERTURB_ON_HIGH_RES=%1d, NO_RNG=%1d, USE_FFTW_WISDOM=%1d, USE_INTERPOLATION_TABLES=%1d,\n"
                "       INTEGRATION_METHOD_ATOMIC=%2d, INTEGRATION_METHOD_MINI=%2d, USE_2LPT=%1d, MINIMIZE_MEMORY=%1d,\n"
                "       KEEP_3D_VELOCITIES=%1d, SAMPLER_MIN_MASS=%10.3e, SAMPLER_BUFFER_FACTOR=%8.3f, MAXHALO_FACTOR=%8.3f,\n"
                "       N_COND_INTERP=%4d, N_PROB_INTERP=%4d, MIN_LOGPROB=%8.3f, SAMPLE_METHOD=%2d, AVG_BELOW_SAMPLER=%1d,\n"
                "       HALOMASS_CORRECTION=%8.3f",
             p->HII_DIM, p->DIM, p->BOX_LEN, p->NON_CUBIC_FACTOR, p->HMF, p->POWER_SPECTRUM, p->USE_RELATIVE_VELOCITIES,
             p->N_THREADS, p->PERTURB_ON_HIGH_RES, p->NO_RNG, p->USE_FFTW_WISDOM, p->USE_INTERPOLATION_TABLES,
             p->INTEGRATION_METHOD_ATOMIC,p->INTEGRATION_METHOD_MINI,p->USE_2LPT,p->MINIMIZE_MEMORY,p->KEEP_3D_VELOCITIES,
             p->SAMPLER_MIN_MASS,p->SAMPLER_BUFFER_FACTOR,p->MAXHALO_FACTOR,p->N_COND_INTERP,p->N_PROB_INTERP,p->MIN_LOGPROB,
             p->SAMPLE_METHOD,p->AVG_BELOW_SAMPLER,p->HALOMASS_CORRECTION);
}

void writeCosmoParams(CosmoParams *p){
    LOG_INFO("\n        CosmoParams:\n"
            "       SIGMA_8=%8.3f, hlittle=%8.3f, OMm=%8.3f, OMl=%8.3f, OMb=%8.3f, POWER_INDEX=%8.3f",
             p->SIGMA_8, p->hlittle, p->OMm, p->OMl, p->OMb, p->POWER_INDEX);
}

void writeAstroParams(FlagOptions *fo, AstroParams *p){
    if(fo->USE_MASS_DEPENDENT_ZETA) {
        LOG_INFO("\n        AstroParams:\n"
        "       M_TURN=%10.3e, R_BUBBLE_MAX=%8.3f, N_RSD_STEPS=%5d\n"
        "       F_STAR10=%8.3f, ALPHA_STAR=%8.3f, F_ESC10=%8.3f, ALPHA_ESC=%8.3f,\n"
        "       t_STAR=%8.3f, L_X=%10.3e, NU_X_THRESH=%8.3f, X_RAY_SPEC_INDEX=%8.3f,\n"
        "       UPPER_STELLAR_TURNOVER_MASS=%10.3e, UPPER_STELLAR_TURNOVER_INDEX=%8.3e",
             p->M_TURN, p->R_BUBBLE_MAX, p->N_RSD_STEPS, p->F_STAR10, p->ALPHA_STAR, p->F_ESC10,
             p->ALPHA_ESC, p->t_STAR, p->L_X, p->NU_X_THRESH, p->X_RAY_SPEC_INDEX,
             p->UPPER_STELLAR_TURNOVER_MASS,p->UPPER_STELLAR_TURNOVER_INDEX);
        if(fo->USE_HALO_FIELD){
            LOG_INFO("\n        HaloField AstroParams:\n"
            "      SIGMA_STAR=%8.3f, CORR_STAR=%8.3f, \n"
            "       SIGMA_SFR_LIM=%8.3f (SIGMA_SFR_INDEX=%8.3f), CORR_SFR=%8.3f\n"
            "       SIGMA_LX=%8.3f, CORR_LX=%8.3f",
            p->SIGMA_STAR, p->CORR_STAR, p->SIGMA_SFR_LIM, p->SIGMA_SFR_INDEX, p->CORR_SFR,
            p->SIGMA_LX, p->CORR_LX);
        }
        if(fo->USE_MINI_HALOS){
            LOG_INFO("\n        MiniHalo AstroParams:\n"
            "       ALPHA_STAR_MINI=%8.3f, F_ESC7_MINI=%8.3f, L_X_MINI=%10.3e, F_STAR7_MINI=%8.3f,\n"
            "       F_H2_SHIELD=%8.3f, A_LW=%8.3f, BETA_LW=%8.3f, A_VCB=%8.3f, BETA_VCB=%8.3f",
                p->ALPHA_STAR_MINI,p->F_ESC7_MINI, p->L_X_MINI, p->F_STAR7_MINI,
                p->F_H2_SHIELD, p->A_LW, p->BETA_LW, p->A_VCB, p->BETA_VCB);

        }
    }
    else {
        LOG_INFO("\n        AstroParams:\n"
        "       HII_EFF_FACTOR=%10.3e, ION_Tvir_MIN=%10.3e, X_RAY_Tvir_MIN=%10.3e,\n"
        "       R_BUBBLE_MAX=%8.3f, L_X=%10.3e, NU_X_THRESH=%8.3f, X_RAY_SPEC_INDEX=%8.3f,\n"
        "       F_STAR10=%8.3f, t_STAR=%8.3f, N_RSD_STEPS=%5d]",
             p->HII_EFF_FACTOR, p->ION_Tvir_MIN, p->X_RAY_Tvir_MIN,
             p->R_BUBBLE_MAX, p->L_X, p->NU_X_THRESH, p->X_RAY_SPEC_INDEX, p->F_STAR10, p->t_STAR, p->N_RSD_STEPS);
    }
}

void writeFlagOptions(FlagOptions *p){
    LOG_INFO("\n        FlagOptions:\n"
            "       USE_HALO_FIELD=%1d, USE_MINI_HALOS=%1d, USE_MASS_DEPENDENT_ZETA=%1d, SUBCELL_RSD=%1d,\n"
            "       INHOMO_RECO=%1d, USE_TS_FLUCT=%1d, M_MIN_in_Mass=%1d, PHOTON_CONS=%1d,\n"
            "       HALO_STOCHASTICITY=%1d, FIXED_HALO_GRIDS=%1d, USE_EXP_FILTER=%1d\n"
            "       USE_CMB_HEATING=%1d, USE_LYA_HEATING=%1d, APPLY_RSDS=%1d, FIX_VCB_AVG=%1d\n"
            "       CELL_RECOMB=%1d, PHOTON_CONS_TYPE=%2d, USE_UPPER_STELLAR_TURNOVER=%1d",
           p->USE_HALO_FIELD, p->USE_MINI_HALOS, p->USE_MASS_DEPENDENT_ZETA, p->SUBCELL_RSD,
           p->INHOMO_RECO, p->USE_TS_FLUCT, p->M_MIN_in_Mass, p->PHOTON_CONS_TYPE, p->HALO_STOCHASTICITY,
           p->FIXED_HALO_GRIDS, p->USE_EXP_FILTER,p->USE_CMB_HEATING,p->USE_LYA_HEATING,p->APPLY_RSDS,
           p->FIX_VCB_AVG,p->CELL_RECOMB,p->PHOTON_CONS_TYPE,p->USE_UPPER_STELLAR_TURNOVER);
}

void debugSummarizeBox(float *box, int size, float ncf, char *indent){
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL

        float corners[8];

        int i,j,k, counter;
        int s = size-1;
        int s_ncf = size*ncf-1;

        counter = 0;
        for(i=0;i<size;i=i+s){
            for(j=0;j<size;j=j+s){
                for(k=0;k<(int)(size*ncf);k=k+s_ncf){
                    corners[counter] =  box[k + (long long unsigned int)(size*ncf)*(j + size*i)];
                    counter++;
                }
            }
        }

        LOG_SUPER_DEBUG("%sCorners: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e",
            indent,
            corners[0], corners[1], corners[2], corners[3],
            corners[4], corners[5], corners[6], corners[7]
        );

        float sum, mean, mn, mx;
        sum=0;
        mn=box[0];
        mx=box[0];

        for (i=0; i<size*size*((int)(size*ncf)); i++){
            sum+=box[i];
            mn=fminf(mn, box[i]);
            mx = fmaxf(mx, box[i]);
        }
        mean=sum/(size*size*((int)(size*ncf)));

        LOG_SUPER_DEBUG("%sSum/Mean/Min/Max: %.4e, %.4e, %.4e, %.4e", indent, sum, mean, mn, mx);
#endif
}

void debugSummarizeBoxDouble(double *box, int size, float ncf, char *indent){
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL

        double corners[8];

        int i,j,k, counter;
        int s = size-1;
        int s_ncf = size*ncf-1;

        counter = 0;
        for(i=0;i<size;i=i+s){
            for(j=0;j<size;j=j+s){
                for(k=0;k<(int)(size*ncf);k=k+s_ncf){
                    corners[counter] =  box[k + (long long unsigned int)(size*ncf)*(j + size*i)];
                    counter++;
                }
            }
        }

        LOG_SUPER_DEBUG("%sCorners: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e",
            indent,
            corners[0], corners[1], corners[2], corners[3],
            corners[4], corners[5], corners[6], corners[7]
        );

        double sum, mean, mn, mx;
        sum=0;
        mn=box[0];
        mx=box[0];

        for (i=0; i<size*size*((int)(size*ncf)); i++){
            sum+=box[i];
            mn=fmin(mn, box[i]);
            mx = fmax(mx, box[i]);
        }
        mean=sum/(size*size*((int)(size*ncf)));

        LOG_SUPER_DEBUG("%sSum/Mean/Min/Max: %.4e, %.4e, %.4e, %.4e", indent, sum, mean, mn, mx);
#endif
}

void debugSummarizeIC(InitialConditions *x, int HII_DIM, int DIM, float NCF){
    LOG_SUPER_DEBUG("Summary of InitialConditions:");
    LOG_SUPER_DEBUG("  lowres_density: ");
    debugSummarizeBox(x->lowres_density, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  hires_density: ");
    debugSummarizeBox(x->hires_density, DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  lowres_vx: ");
    debugSummarizeBox(x->lowres_vx, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  lowres_vy: ");
    debugSummarizeBox(x->lowres_vy, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  lowres_vz: ");
    debugSummarizeBox(x->lowres_vz, HII_DIM, NCF, "    ");
}

void debugSummarizePerturbField(PerturbedField *x, int HII_DIM, float NCF){
    LOG_SUPER_DEBUG("Summary of PerturbedField:");
    LOG_SUPER_DEBUG("  density: ");
    debugSummarizeBox(x->density, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  velocity_x: ");
    debugSummarizeBox(x->velocity_x, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  velocity_y: ");
    debugSummarizeBox(x->velocity_y, HII_DIM, NCF, "    ");
    LOG_SUPER_DEBUG("  velocity_z: ");
    debugSummarizeBox(x->velocity_z, HII_DIM, NCF, "    ");

}
