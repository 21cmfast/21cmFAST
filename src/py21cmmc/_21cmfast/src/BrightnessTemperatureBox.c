
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

void ComputeBrightnessTemp(float redshift, int saturated_limit, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                           struct AstroParams *astro_params, struct FlagOptions *flag_options,
                           struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box) {

    // Makes the parameter structs visible to a variety of functions/macros
    if(StructInit==0) {
        Broadcast_struct_global_PS(user_params,cosmo_params);
        Broadcast_struct_global_UF(user_params,cosmo_params);
        
        StructInit = 1;
    }
    
    printf("EFF_FACTOR_PL_INDEX = %e HII_EFF_FACTOR = %e R_BUBBLE_MAX = %e ION_Tvir_MIN = %e L_X = %e\n",astro_params->EFF_FACTOR_PL_INDEX,astro_params->HII_EFF_FACTOR,
           astro_params->R_BUBBLE_MAX,astro_params->ION_Tvir_MIN,astro_params->L_X);
    printf("NU_X_THRESH = %e X_RAY_SPEC_INDEX = %e X_RAY_Tvir_MIN = %e\n",astro_params->NU_X_THRESH,astro_params->X_RAY_SPEC_INDEX,astro_params->X_RAY_Tvir_MIN);
    printf("F_STAR = %e t_STAR = %e N_RSD_STEPS = %d\n",astro_params->F_STAR,astro_params->t_STAR,astro_params->N_RSD_STEPS);
   
    printf("NU_X_BAND_MAX = %e NU_X_MAX = %e\n",global_params.NU_X_BAND_MAX,global_params.NU_X_MAX);
    printf("Z_HEAT_MAX = %e ZPRIME_STEP_FACTOR = %e\n",global_params.Z_HEAT_MAX,global_params.ZPRIME_STEP_FACTOR);
    
    printf("INCLUDE_ZETA_PL = %s SUBCELL_RSD = %s INHOMO_RECO = %s\n",flag_options->INCLUDE_ZETA_PL ?"true" : "false",flag_options->SUBCELL_RSD ?"true" : "false",flag_options->INHOMO_RECO ?"true" : "false");
    
}

