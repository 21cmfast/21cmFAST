#ifndef _PS_H
#define _PS_H

void init_ps();
double dicke(double z);
double sigma_z0(double M);
double dsigmasqdm_z0(double M);

double MtoR(double M);
double RtoM(double R);
double TtoM(double z, double T, double mu);

void free_ps(); /* deallocates the gsl structures from init_ps */
double power_in_k(double k); /* Returns the value of the linear power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */

double TF_CLASS(double k, int flag_int, int flag_dv); //transfer function of matter (flag_dv=0) and relative velocities (flag_dv=1) fluctuations from CLASS
double power_in_vcb(double k); /* Returns the value of the DM-b relative velocity power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */

double MtoR(double M);
double RtoM(double R);
double TtoM(double z, double T, double mu);
double dicke(double z);
double ddickedt(double z);
double ddicke_dz(double z);
double dtdz(float z);
double drdz(float z); /* comoving distance, (1+z)*C*dtdz(in cm) per unit z */

double hubble(float z);
double t_hubble(float z);
double M_J_WDM();

#endif
