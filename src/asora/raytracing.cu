#include "raytracing.cuh"
#include "memory.cuh"
#include "rates.cuh"
#include <exception>
#include <string>
#include <iostream>

// ========================================================================
// Define macros. Could be passed as parameters but are kept as
// compile-time constants for now
// ========================================================================
#define FOURPI 12.566370614359172463991853874177    // 4π
#define INV4PI 0.079577471545947672804111050482     // 1/4π
#define SQRT3 1.73205080757                         // Square root of 3
#define MAX_COLDENSH 2e30                           // Column density limit (rates are set to zero above this)
#define CUDA_BLOCK_SIZE 256                         // Size of blocks used to treat sources

// ========================================================================
// Utility Device Functions
// ========================================================================

// Fortran-type modulo function (C modulo is signed)
inline int modulo(const int & a,const int & b) { return (a%b+b)%b; }
inline __device__ int modulo_gpu(const int & a,const int & b) { return (a%b+b)%b; }

// Sign function on the device
inline __device__ int sign_gpu(const double & x) { if (x>=0) return 1; else return -1;}

// Flat-array index from 3D (i,j,k) indices
inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N) { return N*N*i + N*j + k;}

// Weight function for C2Ray interpolation function (see cinterp_gpu below)
__device__ inline double weightf_gpu(const double & cd, const double & sig) { return 1.0/max(0.6,cd*sig);}

// Mapping from cartesian coordinates of a cell to reduced cache memory space (here N = 2qmax + 1 in general)
__device__ inline int cart2cache(const int & i,const int & j,const int & k,const int & N) { return N*N*int(k<0) + N*i + j; }

// Mapping from linear 1D indices to the cartesian coords of a q-shell in asora
__device__ void linthrd2cart(const int & s,const int & q,int& i,int& j)
{
    if (s == 0)
    {
        i = q;
        j = 0;
    }
    else
    {
        int b = (s - 1) / (2*q);
        int a = (s - 1) % (2*q);

        if (a + 2*b > 2*q)
        {
            a = a + 1;
            b = b - 1 - q;
        }
        i = a + b - q;
        j = b;
    }
}

// When using a GPU with compute capability < 6.0, we must manually define the atomicAdd function for doubles
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val) {
unsigned long long int* address_as_ull = (unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;
if (val==0.0)
    return __longlong_as_double(old);
do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
} while (assumed != old);
return __longlong_as_double(old);
}
#endif

// ========================================================================
// Raytrace all sources and add up ionization rates
// ========================================================================
void do_all_sources_gpu(
    const double & R,
    double* coldensh_out,
    const double & sig,
    const double & dr,
    double* ndens,
    double* xh_av,
    double* phi_ion,
    const int & NumSrc,
    const int & m1,
    const double & minlogtau,
    const double & dlogtau,
    const int & NumTau)
    {   
        
        // Byte-size of grid data
        int meshsize = m1*m1*m1*sizeof(double);

        //std::cout << "R: " << R << std::endl;
        // Determine how large the octahedron should be, based on the raytracing radius. Currently, this is set s.t. the radius equals the distance from the source to the middle of the faces of the octahedron. To raytrace the whole box, the octahedron bust be 1.5*N in size
        int max_q = std::ceil(SQRT3 * min(R,SQRT3*m1/2.0));
        //std::cout << "max_q: " << max_q << std::endl;

        // CUDA Grid size: since 1 block = 1 source, this sets the number of sources treated in parallel
        dim3 gs(NUM_SRC_PAR);

        // CUDA Block size: more of a tuning parameter (see above), in practice anything ~128 is fine
        dim3 bs(CUDA_BLOCK_SIZE);

        // Here we fill the ionization rate array with zero before raytracing all sources. The LOCALRATES flag is for debugging purposes and will be removed later on
        cudaMemset(phi_dev,0,meshsize);

        // Copy current ionization fraction to the device
        // cudaMemcpy(n_dev,ndens,meshsize,cudaMemcpyHostToDevice);  < --- !! density array is not modified, asora assumes that it has been copied to the device before
        cudaMemcpy(x_dev,xh_av,meshsize,cudaMemcpyHostToDevice);

        // Since the grid is periodic, we limit the maximum size of the raytraced region to a cube as large as the mesh around the source.
        // See line 93 of evolve_source in C2Ray, this size will depend on if the mesh is even or odd. Basically the idea is that you never touch a cell which is outside a cube of length ~N centered on the source
        int last_r = m1/2 - 1 + modulo(m1,2);
        int last_l = -m1/2;

        // Loop over batches of sources
        for (int ns = 0; ns < NumSrc; ns += NUM_SRC_PAR)
        {   
            // Raytrace the current batch of sources in parallel
            evolve0D_gpu<<<gs,bs>>>(R,max_q,ns,NumSrc,NUM_SRC_PAR,src_pos_dev,src_flux_dev,cdh_dev, sig,dr,n_dev,x_dev,phi_dev,m1,photo_thin_table_dev,photo_thick_table_dev,
 minlogtau,dlogtau,NumTau,last_l,last_r);

            // Check for errors
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess) {
                throw std::runtime_error("Error Launching Kernel: " + std::string(cudaGetErrorName(error)) + " - " + std::string(cudaGetErrorString(error)));
            }

            // Sync device to be sure (is this required ??)
            cudaDeviceSynchronize();
        }

        // Copy the accumulated ionization fraction back to the host
        cudaError_t error = cudaMemcpy(phi_ion,phi_dev,meshsize,cudaMemcpyDeviceToHost);
        cudaError_t error2 = cudaMemcpy(coldensh_out,cdh_dev,meshsize,cudaMemcpyDeviceToHost);
    }


// ========================================================================
// Raytracing kernel, adapted from C2Ray. Calculates in/out column density
// to the current cell and finds the photoionization rate
// ========================================================================
__global__ void evolve0D_gpu(
    const double Rmax_LLS,
    const int q_max,    // Is now the size of max q
    const int ns_start,
    const int NumSrc,
    const int num_src_par,
    int* src_pos,
    double* src_flux,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1,
    const double* photo_thin_table,
    const double* photo_thick_table,
    const double minlogtau,
    const double dlogtau,
    const int NumTau,
    const int last_l,
    const int last_r
)
{   
    /* The raytracing kernel proceeds as follows:
        1. Select the source based on the block number (within the batch = the grid)
        2. Loop over the asora q-cells around the source, up to q_max (loop "A")
        3. Inside each shell, threads independently do all cells, possibly requiring multiple iterations
        if the block size is smaller than the number of cells in the shell (loop "B")
        4. After each shell, the threads are synchronized to ensure that causality is respected
    */

    // Source number = Start of batch + block number (each block does one source)
    int ns = ns_start + blockIdx.x;

    // Offset pointer to the outgoing column density array used for interpolation (each block needs its own copy of the array)
    int cdh_offset = blockIdx.x * m1 * m1 * m1;

    // Ensure the source index is valid
    if (ns < NumSrc)
    {   
        // (A) Loop over ASORA q-shells
        for (int q = 0 ; q <= q_max ; q++)
        {   
            // We figure out the number of cells in the shell and determine how many passes the block needs to take to treat all of them
            int num_cells = 4*q*q + 2;
            int Npass = num_cells / blockDim.x + 1;

            //The threads have 1D indices 0,...,blocksize-1. We map these 1D indices to the 3D positions of the cells inside the shell via the mapping described in the paper. Since in general there are more cells than threads, there is an additional loop here (B) so that all cells are treated. 
            int s_end;
            if (q == 0) {s_end = 1;}
            else {s_end = 4*q*q + 2;}
            int s_end_top = 2*q*(q+1) + 1;
            
            // (B) Loop over cells in the shell
            for (int ipass = 0 ; ipass < Npass ; ipass++)
            {
                // "s" is the index in the 1D-range [0,...,4q^2 + 1] that gets mapped to the cells in the shell
                int s = ipass * blockDim.x +  threadIdx.x;
                int i,j,k;
                int sgn;

                // Ensure the thread maps to a valid cell
                if (s < s_end)
                {
                    // Determine if cell is in top or bottom part of the shell (the mapping is slightly different due to the part that is on the same z-plane as the source)
                    if (s < s_end_top)
                    {
                        sgn = 1;
                        linthrd2cart(s,q,i,j);
                    }
                    else
                    {
                        sgn = -1;
                        linthrd2cart(s-s_end_top,q-1,i,j);
                    }
                    k = sgn*q - sgn*(abs(i) + abs(j));

                    // Only do cell if it is within the (shifted under periodicity) grid, i.e. at most ~N cells away from the source
                    if ((i >= last_l) && (i <= last_r) && (j >= last_l) && (j <= last_r) && (k >= last_l) && (k <= last_r))
                    {
                        // Get source properties
                        int i0 = src_pos[3*ns + 0];
                        int j0 = src_pos[3*ns + 1];
                        int k0 = src_pos[3*ns + 2];
                        double strength = src_flux[ns];

                        // Center to source
                        i += i0;
                        j += j0;
                        k += k0;

                        int pos[3];
                        double path;
                        double coldensh_in;     // Column density to the cell
                        double nHI_p;           // Local density of neutral hydrogen in the cell
                        double xh_av_p;         // Local ionization fraction of cell

                        double xs, ys, zs;
                        double dist2;
                        double vol_ph;

                        // When not in periodic mode, only treat cell if its in the grid
                        #if !defined(PERIODIC)
                        if (in_box_gpu(i,j,k,m1))
                        #endif
                        {   
                            // Map to periodic grid
                            pos[0] = modulo_gpu(i,m1);
                            pos[1] = modulo_gpu(j,m1);
                            pos[2] = modulo_gpu(k,m1);

                            // Get local ionization fraction & Hydrogen density
                            xh_av_p = xh_av[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
                            nHI_p = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);

                            // PH (29.9.23): There used to be a check here if the coldensh_out of the current cell was zero to "determine if it hasn't been done before". I think this isn't necessary anymore in this version and eliminates the need to set the array to zero between source batches, which for large batches is a SIGNIFICANT bottleneck.

                            // If its the source cell, just find path (no incoming column density)
                            if (i == i0 &&
                                j == j0 &&
                                k == k0)
                            {
                                coldensh_in = 0.0;
                                path = 0.5*dr;
                                vol_ph = dr*dr*dr;
                                dist2 = 0.0;
                            }

                            // If its another cell, do interpolation to find incoming column density
                            else
                            {
                                cinterp_gpu(i, j, k, i0, j0, k0, coldensh_in, path, coldensh_out + cdh_offset, sig, m1);
                                path *= dr;
                                // Find the distance to the source
                                xs = dr*(i-i0);
                                ys = dr*(j-j0);
                                zs = dr*(k-k0);
                                dist2=xs*xs+ys*ys+zs*zs;
                                // vol_ph = dist2 * path;
                                vol_ph = dist2 * path * FOURPI;
                            }

                            // Compute outgoing column density and add to array for subsequent interpolations
                            double cdho = coldensh_in + nHI_p * path;
                            
                            // Compute photoionization rates from column density. WARNING: for now this is limited to the grey-opacity test case source
                            if ((coldensh_in <= MAX_COLDENSH) && (dist2/(dr*dr) <= Rmax_LLS*Rmax_LLS))
                            {
                                #if defined(GREY_NOTABLES)
                                double phi = photoion_rates_test_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig);
                                #else
                                double phi = photoion_rates_gpu(strength, coldensh_in, cdho, vol_ph, sig, photo_thin_table, photo_thick_table, minlogtau, dlogtau, NumTau);
                                #endif
                                // Divide the photo-ionization rates by the appropriate neutral density (part of the photon-conserving rate prescription)
                                phi /= nHI_p;

                                // Add the computed ionization rate and the column density to the array ATOMICALLY since multiple blocks could be writing to the same cell at the same time!
                                atomicAdd(phi_ion + mem_offst_gpu(pos[0],pos[1],pos[2],m1),phi);
                                atomicAdd(coldensh_out + mem_offst_gpu(pos[0],pos[1],pos[2],m1),cdho);
                            }                          
                        }
                    }
                }
            }
            // IMPORTANT: Sync threads after each shell so that the next only begins when all outgoing column densities of the current shell are available
            __syncthreads();
        }
    }
}


// ========================================================================
// Short-characteristics interpolation function
// ========================================================================
__device__ void cinterp_gpu(
    const int i,
    const int j,
    const int k,
    const int i0,
    const int j0,
    const int k0,
    double & cdensi,
    double & path,
    double* coldensh_out,
    const double sigma_HI_at_ion_freq,
    const int & m1)
{
    int idel,jdel,kdel;
    int idela,jdela,kdela;
    int im,jm,km;
    unsigned int ip,imp,jp,jmp,kp,kmp;
    int sgni,sgnj,sgnk;
    double alam,xc,yc,zc,dx,dy,dz,s1,s2,s3,s4;
    double c1,c2,c3,c4;
    double w1,w2,w3,w4;
    double di,dj,dk;

    // calculate the distance between the source point (i0,j0,k0) and 
    // the destination point (i,j,k)
    idel=i-i0;
    jdel=j-j0;
    kdel=k-k0;
    idela=abs(idel);
    jdela=abs(jdel);
    kdela=abs(kdel);
    
    // Find coordinates of points closer to source
    sgni=sign_gpu(idel);
    sgnj=sign_gpu(jdel);
    sgnk=sign_gpu(kdel);
    im=i-sgni;
    jm=j-sgnj;
    km=k-sgnk;
    di=double(idel);
    dj=double(jdel);
    dk=double(kdel);

    // Z plane (bottom and top face) crossing
    // we find the central (c) point (xc,xy) where the ray crosses 
    // the z-plane below or above the destination (d) point, find the 
    // column density there through interpolation, and add the contribution
    // of the neutral material between the c-point and the destination
    // point.
    if (kdela >= jdela && kdela >= idela) {
        // alam is the parameter which expresses distance along the line s to d
        // add 0.5 to get to the interface of the d cell.
        alam=(double(km-k0)+sgnk*0.5)/dk;
            
        xc=alam*di+double(i0); // x of crossing point on z-plane 
        yc=alam*dj+double(j0); // y of crossing point on z-plane
        
        dx=2.0*abs(xc-(double(im)+0.5*sgni)); // distances from c-point to
        dy=2.0*abs(yc-(double(jm)+0.5*sgnj)); // the corners.
        
        s1=(1.-dx)*(1.-dy);    // interpolation weights of
        s2=(1.-dy)*dx;         // corner points to c-point
        s3=(1.-dx)*dy;
        s4=dx*dy;

        ip  = modulo_gpu(i  ,m1);
        imp = modulo_gpu(im ,m1);
        jp  = modulo_gpu(j  ,m1);
        jmp = modulo_gpu(jm ,m1);
        kmp = modulo_gpu(km ,m1);
        
        c1=     coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        c2=     coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
        c3=     coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
        c4=     coldensh_out[mem_offst_gpu(ip,jp,kmp,m1)];

        // extra weights for better fit to analytical solution
        w1=   s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2=   s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3=   s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4=   s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
        
        // column density at the crossing point
        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        // Take care of diagonals
        if (kdela == 1 && (idela == 1||jdela == 1))
        {
            if (idela == 1 && jdela == 1)
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }

        // Path length from c through d to other side cell.
        path=sqrt((di*di+dj*dj)/(dk*dk)+1.0);
    }
    else if (jdela >= idela && jdela >= kdela)
    {
        alam=(double(jm-j0)+sgnj*0.5)/dj;
        zc=alam*dk+double(k0);
        xc=alam*di+double(i0);
        dz=2.0*abs(zc-(double(km)+0.5*sgnk));
        dx=2.0*abs(xc-(double(im)+0.5*sgni));
        s1=(1.-dx)*(1.-dz);
        s2=(1.-dz)*dx;
        s3=(1.-dx)*dz;
        s4=dx*dz;

        ip  = modulo_gpu(i,m1);
        imp = modulo_gpu(im,m1);
        jmp = modulo_gpu(jm,m1);
        kp  = modulo_gpu(k,m1);
        kmp = modulo_gpu(km,m1);

        c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        c2=  coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
        c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
        c4=  coldensh_out[mem_offst_gpu(ip,jmp,kp,m1)];

        // extra weights for better fit to analytical solution
        w1=s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2=s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3=s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4=s4*weightf_gpu(c4,sigma_HI_at_ion_freq);

        cdensi=   (c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        // Take care of diagonals
        if (jdela == 1 && (idela == 1||kdela == 1))
        {
            if (idela == 1 && kdela == 1)
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }
        path=sqrt((di*di+dk*dk)/(dj*dj)+1.0);
    }
    else
    {
        alam=(double(im-i0)+sgni*0.5)/di;
        zc=alam*dk+double(k0);
        yc=alam*dj+double(j0);
        dz=2.0*abs(zc-(double(km)+0.5*sgnk));
        dy=2.0*abs(yc-(double(jm)+0.5*sgnj));
        s1=(1.-dz)*(1.-dy);
        s2=(1.-dz)*dy;
        s3=(1.-dy)*dz;
        s4=dy*dz;

        imp=modulo_gpu(im ,m1);
        jp= modulo_gpu(j  ,m1);
        jmp=modulo_gpu(jm ,m1);
        kp= modulo_gpu(k  ,m1);
        kmp=modulo_gpu(km ,m1);

        c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        c2=  coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
        c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
        c4=  coldensh_out[mem_offst_gpu(imp,jp,kp,m1)];

        // extra weights for better fit to analytical solution
        w1   =s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2   =s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3   =s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4   =s4*weightf_gpu(c4,sigma_HI_at_ion_freq);

        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        if ( idela == 1  &&  ( jdela == 1 || kdela == 1 ) )
        {
            if ( jdela == 1  &&  kdela == 1 )
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }
        path=sqrt(1.0+(dj*dj+dk*dk)/(di*di));
    }
}









/*

// ==========================================================================================================
// OLD OR EXPERIMENTAL CODE. KEPT AS REFERENCE BUT UNUSED
// ==========================================================================================================

// ========================================================================
// Modified version of cinterp using a different coordinate system, for future development
// ========================================================================
__device__ void cinterp_gpu2(
    const int i,
    const int j,
    const int k,
    const int i0,
    const int j0,
    const int k0,
    const int qmax,
    double & cdensi,
    double & path,
    double* coldensh_out,
    const double sigma_HI_at_ion_freq)
{
    int is,js,ks;
    int it,jt,kt;
    int idela,jdela,kdela;
    int im,jm,km;
    unsigned int ip,imp,jp,jmp,kp,kmp;
    int sgni,sgnj,sgnk;
    double alam,xc,yc,zc,dx,dy,dz,s1,s2,s3,s4;
    double c1,c2,c3,c4;
    double w1,w2,w3,w4;
    double di,dj,dk;

    // Transform coordinates: origin at source location (i0,j0,k0) "s" coordinate system
    // range [-qmax,qmax] (in reality [-q,q])
    is = i-i0;
    js = j-j0;
    ks = k-k0;

    // Get absolute distance to source cell [0,q]
    idela=abs(is);
    jdela=abs(js);
    kdela=abs(ks);
    
    // Find on which side of source the point is
    sgni=sign_gpu(is);
    sgnj=sign_gpu(js);
    sgnk=sign_gpu(ks);

    // Transform coordinates: origin at lower left corner of cache array. "tilde" or "t" coordinate system
    // range [0,2*qmax]
    it = is + qmax;
    jt = js + qmax;
    kt = ks + qmax;

    // Coordinates of "closer" cells used for the interpolation
    im = it - sgni;
    jm = jt - sgnj;
    km = kt - sgnk;

    // Distance to source (double)
    di = double(is);
    dj = double(js);
    dk = double(ks);

    // Z plane (bottom and top face) crossing
    // we find the central (c) point (xc,xy) where the ray crosses 
    // the z-plane below or above the destination (d) point, find the 
    // column density there through interpolation, and add the contribution
    // of the neutral material between the c-point and the destination
    // point.
    if (kdela >= jdela && kdela >= idela) {
        // alam is the parameter which expresses distance along the line s to d
        // add 0.5 to get to the interface of the d cell.
        alam=(double(km-k0)+sgnk*0.5)/dk;
            
        xc=alam*di+double(i0); // x of crossing point on z-plane 
        yc=alam*dj+double(j0); // y of crossing point on z-plane
        
        dx=2.0*abs(xc-(double(im)+0.5*sgni)); // distances from c-point to
        dy=2.0*abs(yc-(double(jm)+0.5*sgnj)); // the corners.
        
        s1=(1.-dx)*(1.-dy);    // interpolation weights of
        s2=(1.-dy)*dx;         // corner points to c-point
        s3=(1.-dx)*dy;
        s4=dx*dy;

        ip  = i;
        imp = im;
        jp  = j;
        jmp = jm;
        kmp = km;
        
        // New method
        c1 = coldensh_out[cart2cache(imp,jmp,kmp,2*qmax+1)];
        c2 = coldensh_out[cart2cache(ip, jmp,kmp,2*qmax+1)];
        c3 = coldensh_out[cart2cache(imp,jp, kmp,2*qmax+1)];
        c4 = coldensh_out[cart2cache(ip, jp, kmp,2*qmax+1)];

        // Old method
        // c1=     coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        // c2=     coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
        // c3=     coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
        // c4=     coldensh_out[mem_offst_gpu(ip,jp,kmp,m1)];

        // extra weights for better fit to analytical solution
        w1 = s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2 = s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3 = s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4 = s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
        
        // column density at the crossing point
        cdensi = (c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4 )/(w1+w2+w3+w4);

        // Take care of diagonals
        if (kdela == 1 && (idela == 1||jdela == 1))
        {
            if (idela == 1 && jdela == 1)
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }

        // Path length from c through d to other side cell.
        path=sqrt((di*di+dj*dj)/(dk*dk)+1.0);
    }
    else if (jdela >= idela && jdela >= kdela)
    {
        alam=(double(jm-j0)+sgnj*0.5)/dj;
        zc=alam*dk+double(k0);
        xc=alam*di+double(i0);
        dz=2.0*abs(zc-(double(km)+0.5*sgnk));
        dx=2.0*abs(xc-(double(im)+0.5*sgni));
        s1=(1.-dx)*(1.-dz);
        s2=(1.-dz)*dx;
        s3=(1.-dx)*dz;
        s4=dx*dz;

        ip  = i;
        imp = im;
        jmp = jm;
        kp  = k;
        kmp = km;

        // New method
        c1 = coldensh_out[cart2cache(imp,jmp, kmp,2*qmax+1)];
        c2 = coldensh_out[cart2cache(ip, jmp, kmp,2*qmax+1)];
        c3 = coldensh_out[cart2cache(imp,jmp, kp ,2*qmax+1)];
        c4 = coldensh_out[cart2cache(ip, jmp, kp ,2*qmax+1)];

        // Old method
        // c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        // c2=  coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
        // c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
        // c4=  coldensh_out[mem_offst_gpu(ip,jmp,kp,m1)];

        // extra weights for better fit to analytical solution
        w1=s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2=s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3=s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4=s4*weightf_gpu(c4,sigma_HI_at_ion_freq);

        cdensi=   (c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        // Take care of diagonals
        if (jdela == 1 && (idela == 1||kdela == 1))
        {
            if (idela == 1 && kdela == 1)
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }
        path=sqrt((di*di+dk*dk)/(dj*dj)+1.0);
    }
    else
    {
        alam=(double(im-i0)+sgni*0.5)/di;
        zc=alam*dk+double(k0);
        yc=alam*dj+double(j0);
        dz=2.0*abs(zc-(double(km)+0.5*sgnk));
        dy=2.0*abs(yc-(double(jm)+0.5*sgnj));
        s1=(1.-dz)*(1.-dy);
        s2=(1.-dz)*dy;
        s3=(1.-dy)*dz;
        s4=dy*dz;

        imp= im;
        jp=  j;
        jmp= jm;
        kp=  k;
        kmp= km;
        
        // New method
        c1 = coldensh_out[cart2cache(imp,jmp,kmp,2*qmax+1)];
        c2 = coldensh_out[cart2cache(imp,jp, kmp,2*qmax+1)];
        c3 = coldensh_out[cart2cache(imp,jmp,kp ,2*qmax+1)];
        c4 = coldensh_out[cart2cache(imp,jp, kp ,2*qmax+1)];

        // Old method
        // c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
        // c2=  coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
        // c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
        // c4=  coldensh_out[mem_offst_gpu(imp,jp,kp,m1)];

        // extra weights for better fit to analytical solution
        w1   =s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
        w2   =s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
        w3   =s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
        w4   =s4*weightf_gpu(c4,sigma_HI_at_ion_freq);

        cdensi   =(c1   *w1   +c2   *w2   +c3   *w3   +c4   *w4   )/(w1+w2+w3+w4);

        if ( idela == 1  &&  ( jdela == 1 || kdela == 1 ) )
        {
            if ( jdela == 1  &&  kdela == 1 )
            {
                cdensi = 1.73205080757*cdensi;
            }
            else
            {
                cdensi = 1.41421356237*cdensi;
            }
        }
        path=sqrt(1.0+(dj*dj+dk*dk)/(di*di));
    }
}

// WIP: compute rates in a separate step ? This would require having a path and coldensh_in grid (replace cdh_out by path)
__global__ void do_rates(
    const int rad,
    const int i0,
    const int j0,
    const int k0,
    const double strength,
    double* coldensh_in,
    double* path,
    double* ndens,
    double* phi_ion,
    const double sig,
    const double dr,
    const int m1
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (abs(i) + abs(j) + abs(k) <= rad)
    {
        double vol_ph;
        int pos[3];
        pos[0] = modulo_gpu(i,m1);
        pos[1] = modulo_gpu(j,m1);
        pos[2] = modulo_gpu(k,m1);
        double cdh_in = coldensh_in[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
        double nHI = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
        double cdh_out;
        double phi;
        if (i == i0 && j == j0 && k == k0)
        {
            vol_ph = dr*dr*dr / (4*M_PI);
        }
        else
        {
            double xs = dr*(i-i0);
            double ys = dr*(j-j0);
            double zs = dr*(k-k0);
            double dist2=xs*xs+ys*ys+zs*zs;
            vol_ph = dist2 * path[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
        }

        cdh_out = cdh_in + path[mem_offst_gpu(pos[0],pos[1],pos[2],m1)]*nHI;
        phi = photoion_rates_test_gpu(strength,cdh_in,cdh_out,vol_ph,sig);
        phi_ion[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi / nHI;

    } 
}


// ========================================================================
// Raytracing kernel, adapted from C2Ray. Calculates in/out column density
// to the current cell and finds the photoionization rate
// ========================================================================
__global__ void evolve0D_gpu_old(
    const int q,
    const int ns,
    int* src_pos,
    double* src_flux,
    double* coldensh_out,
    const double sig,
    const double dr,
    const double* ndens,
    const double* xh_av,
    double* phi_ion,
    const int m1,
    const double* photo_table,
    const double minlogtau,
    const double dlogtau,
    const int NumTau,
    const int last_l,
    const int last_r
)
{
    // x and y coordinates are cartesian
    int i = - q + blockIdx.x * blockDim.x + threadIdx.x;
    int j = - q + blockIdx.y * blockDim.y + threadIdx.y;

    int sgn, mq;

    // Determine whether we are in the upper or lower pyramid of the octahedron
    if (blockIdx.z == 0)
        {sgn = 1; mq = q;}
    else
        {sgn = -1; mq = q-1;}

    int k = sgn*q - sgn*(abs(i) + abs(j));

    // We only treat the cell if it respects two conditions:
    // 1. It must be part of the shell (see figure in appendix A of the paper)
    // 2. It must be within the maximum box size arount the source (see last_l and last_r def above) <- This is also important to avoid race conditions at the border
    // TODO: in the future, it may be an interesting optimization to limit the number of threads launched in the first place,
    // rather than doing this "brute force" approach where about half of the threads don't pass this "if" check and immediately return
    if (abs(i) + abs(j) <= mq && (i >= last_l) && (i <= last_r) && (j >= last_l) && (j <= last_r) && (k >= last_l) && (k <= last_r))
    {
        // Get source properties
        int i0 = src_pos[3*ns + 0];
        int j0 = src_pos[3*ns + 1];
        int k0 = src_pos[3*ns + 2];
        double strength = src_flux[ns];

        // Center to source
        i += i0;
        j += j0;
        k += k0;

        int pos[3];
        double path;
        double coldensh_in;                                // Column density to the cell
        double nHI_p;                                      // Local density of neutral hydrogen in the cell
        double xh_av_p;                                    // Local ionization fraction of cell

        double xs, ys, zs;
        double dist2;
        double vol_ph;

        // When not in periodic mode, only treat cell if its in the grid
        #if !defined(PERIODIC)
        if (in_box_gpu(i,j,k,m1))
        #endif
        {   
            // Map to periodic grid
            pos[0] = modulo_gpu(i,m1);
            pos[1] = modulo_gpu(j,m1);
            pos[2] = modulo_gpu(k,m1);

            //printf("pos = %i %i %i \n",pos[0]-i0,pos[1]-j0,pos[2]-k0);

            // Get local ionization fraction & Hydrogen density
            xh_av_p = xh_av[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
            nHI_p = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);

            // Only treat cell if it hasn't been done before
            if (coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] == 0.0)
            {   
                // If its the source cell, just find path (no incoming column density)
                if (i == i0 &&
                    j == j0 &&
                    k == k0)
                {
                    coldensh_in = 0.0;
                    path = 0.5*dr;
                    // vol_ph = dr*dr*dr / (4*M_PI);
                    vol_ph = dr*dr*dr;
                }

                // If its another cell, do interpolation to find incoming column density
                else
                {
                    cinterp_gpu(i,j,k,i0,j0,k0,coldensh_in,path,coldensh_out,sig,m1);
                    path *= dr;
                    // Find the distance to the source
                    xs = dr*(i-i0);
                    ys = dr*(j-j0);
                    zs = dr*(k-k0);
                    dist2=xs*xs+ys*ys+zs*zs;
                    // vol_ph = dist2 * path;
                    vol_ph = dist2 * path * FOURPI;
                }

                // Add to column density array. TODO: is this really necessary ?
                coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] = coldensh_in + nHI_p * path;
                
                // Compute photoionization rates from column density. WARNING: for now this is limited to the grey-opacity test case source
                if (coldensh_in <= MAX_COLDENSH)
                {
                    #if defined(GREY_NOTABLES)
                    double phi = photoion_rates_test_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig);
                    #else
                    double phi = photoion_rates_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig,photo_table,photo_table,minlogtau,dlogtau,NumTau);
                    #endif
                    // Divide the photo-ionization rates by the appropriate neutral density
                    // (part of the photon-conserving rate prescription)
                    phi /= nHI_p;

                    phi_ion[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi;
                }
            }
        }
    }
}

*/