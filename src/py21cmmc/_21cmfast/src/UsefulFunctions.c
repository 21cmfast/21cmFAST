// ----------------------------------------------------------------------------------------- //

// Taken from INIT_PARAMS.H

// ----------------------------------------------------------------------------------------- //



#define MIDDLE (user_params_ufunc->DIM/2)
#define D (unsigned long long)user_params_ufunc->DIM // the unsigned long long dimension
#define MID ((unsigned long long)MIDDLE)
#define VOLUME (user_params_ufunc->BOX_LEN*user_params_ufunc->BOX_LEN*user_params_ufunc->BOX_LEN) // in Mpc^3
#define DELTA_K (TWOPI/user_params_ufunc->BOX_LEN)
#define TOT_NUM_PIXELS ((unsigned long long)(D*D*D)) // no padding
#define TOT_FFT_NUM_PIXELS ((unsigned long long)(D*D*2llu*(MID+1llu)))
#define KSPACE_NUM_PIXELS ((unsigned long long)(D*D*(MID+1llu)))

// Define some useful macros

// for 3D complex array
#define C_INDEX(x,y,z)((unsigned long long)((z)+(MID+1llu)*((y)+D*(x))))

// for 3D real array with the FFT padding
#define R_FFT_INDEX(x,y,z)((unsigned long long)((z)+2llu*(MID+1llu)*((y)+D*(x))))

// for 3D real array with no padding
#define R_INDEX(x,y,z)((unsigned long long)((z)+D*((y)+D*(x))))


// ----------------------------------------------------------------------------------------- //

// Taken from ANAL_PARAMS.H

// ----------------------------------------------------------------------------------------- //



#define HII_D (unsigned long long) (user_params_ufunc->HII_DIM)
#define HII_MIDDLE (user_params_ufunc->HII_DIM/2)
#define HII_MID ((unsigned long long)HII_MIDDLE)

#define HII_TOT_NUM_PIXELS (unsigned long long)(HII_D*HII_D*HII_D)
#define HII_TOT_FFT_NUM_PIXELS ((unsigned long long)(HII_D*HII_D*2llu*(HII_MID+1llu)))
#define HII_KSPACE_NUM_PIXELS ((unsigned long long)(HII_D*HII_D*(HII_MID+1llu)))

// INDEXING MACROS //
// for 3D complex array
#define HII_C_INDEX(x,y,z)((unsigned long long)((z)+(HII_MID+1llu)*((y)+HII_D*(x))))
// for 3D real array with the FFT padding
#define HII_R_FFT_INDEX(x,y,z)((unsigned long long)((z)+2llu*(HII_MID+1llu)*((y)+HII_D*(x))))
// for 3D real array with no padding
#define HII_R_INDEX(x,y,z)((unsigned long long)((z)+HII_D*((y)+HII_D*(x))))



// ----------------------------------------------------------------------------------------- //

// Taken from COSMOLOGY.H

// ----------------------------------------------------------------------------------------- //



#define Ho  (double) (cosmo_params_ufunc->hlittle*3.2407e-18) // s^-1 at z=0
#define RHOcrit (double) ( (3.0*Ho*Ho / (8.0*PI*G)) * (CMperMPC*CMperMPC*CMperMPC)/Msun) // Msun Mpc^-3 ---- at z=0
#define RHOcrit_cgs (double) (3.0*Ho*Ho / (8.0*PI*G)) // g pcm^-3 ---- at z=0
#define No  (double) (RHOcrit_cgs*cosmo_params_ufunc->OMb*(1-Y_He)/m_p)  //  current hydrogen number density estimate  (#/cm^3)  ~1.92e-7
#define He_No (double) (RHOcrit_cgs*cosmo_params_ufunc->OMb*Y_He/(4.0*m_p)) //  current helium number density estimate
#define N_b0 (double) (No+He_No) // present-day baryon num density, H + He
#define f_H (double) (No/(No+He_No))  // hydrogen number fraction
#define f_He (double) (He_No/(No+He_No))  // helium number fraction

struct CosmoParams *cosmo_params_ufunc;
struct UserParams *user_params_ufunc;

void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params){
 
    cosmo_params_ufunc = cosmo_params;
    user_params_ufunc = user_params;
}



void filter_box(fftwf_complex *box, int RES, int filter_type, float R){
    int n_x, n_z, n_y, dimension,midpoint;
    float k_x, k_y, k_z, k_mag, kR;
    
//    printf("before = %e\n",box[C_INDEX(50, 50, 50)]);
    
    switch(RES) {
        case 0:
            dimension = user_params_ufunc->DIM;
            midpoint = MIDDLE;
            break;
        case 1:
            dimension = user_params_ufunc->HII_DIM;
            midpoint = HII_MIDDLE;
            break;
    }
    
    // loop through k-box
    for (n_x=dimension; n_x--;){
        if (n_x>midpoint) {k_x =(n_x-dimension) * DELTA_K;}
        else {k_x = n_x * DELTA_K;}
        
        for (n_y=dimension; n_y--;){
            if (n_y>midpoint) {k_y =(n_y-dimension) * DELTA_K;}
            else {k_y = n_y * DELTA_K;}
            
            for (n_z=(midpoint+1); n_z--;){
                k_z = n_z * DELTA_K;
                
                k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                
                kR = k_mag*R; // real space top-hat
                
//                if(n_x == 50 && n_y == 50 && n_z == 50) {
//                    printf("n_x = %d n_y = %d n_z = %d k_x = %e k_y = %e k_z = %e k_mag = %e kR = %e\n",n_x,n_y,n_z,k_x,k_y,k_z,k_mag,kR);
//                }
                
                if (filter_type == 0){ // real space top-hat
                    if (kR > 1e-4){
//                        box[HII_C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR);
//                        printf("n_x = %d n_y = %d n_z = %d HIRES_box = %e\n",n_x,n_y,n_z,box[C_INDEX(n_x, n_y, n_z)]);
                        box[C_INDEX(n_x, n_y, n_z)] *= 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR);
//                        printf("k_x = %e k_y = %e k_z = %e k_mag = %e kR = %e arg = %e HIRES_box = %e\n",k_x,k_y,k_z,k_mag,kR,3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR),box[C_INDEX(n_x, n_y, n_z)]);
                    }
                }
                else if (filter_type == 1){ // k-space top hat
                    kR *= 0.413566994; // equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
                    if (kR > 1){
//                        box[HII_C_INDEX(n_x, n_y, n_z)] = 0;
                        box[C_INDEX(n_x, n_y, n_z)] = 0;
                    }
                }
                else if (filter_type == 2){ // gaussian
                    kR *= 0.643; // equates integrated volume to the real space top-hat
//                    box[HII_C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR*kR/2.0);
                    box[C_INDEX(n_x, n_y, n_z)] *= pow(E, -kR*kR/2.0);
                }
                else{
                    if ( (n_x==0) && (n_y==0) && (n_z==0) )
                        fprintf(stderr, "HII_filter.c: Warning, filter type %i is undefined\nBox is unfiltered\n", filter_type);
                }
            }
        }
    } // end looping through k box
    
//    printf("after = %e\n",box[C_INDEX(50, 50, 50)]);
    
    return;
}


/* R in Mpc, M in Msun */
double MtoR(double M){
    
    // set R according to M<->R conversion defined by the filter type in ../Parameter_files/COSMOLOGY.H
    if (global_params.FILTER == 0) //top hat M = (4/3) PI <rho> R^3
        return pow(3*M/(4*PI*cosmo_params_ufunc->OMm*RHOcrit), 1.0/3.0);
    else if (global_params.FILTER == 1) //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow( M/(pow(2*PI, 1.5) * cosmo_params_ufunc->OMm * RHOcrit), 1.0/3.0 );
    else // filter not defined
        fprintf(stderr, "No such filter = %i.\nResults are bogus.\n", global_params.FILTER);
    return -1;
}