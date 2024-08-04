#ifndef _PS_H
#define _PS_H

double init_ps();
double dicke(double z);
double sigma_z0(double M);
double dsigmasqdm_z0(double M);
double get_delta_crit(int HMF, double sigma, double growthf);

double MtoR(double M);
double RtoM(double R);
double TtoM(double z, double T, double mu);
double Deltac_nonlinear(float z);
void gauleg(float x1, float x2, float x[], float w[], int n);

void free_ps(); /* deallocates the gsl structures from init_ps */
double power_in_k(double k); /* Returns the value of the linear power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */
double TFmdm(double k); //Eisenstein & Hu power spectrum transfer function
void TFset_parameters();

double TF_CLASS(double k, int flag_int, int flag_dv); //transfer function of matter (flag_dv=0) and relative velocities (flag_dv=1) fluctuations from CLASS
double power_in_vcb(double k); /* Returns the value of the DM-b relative velocity power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */


double FgtrM(double z, double M);
double FgtrM_wsigma(double z, double sig);
double FgtrM_st(double z, double M);
double FgtrM_Watson(double growthf, double M);
double FgtrM_Watson_z(double z, double growthf, double M);
double Fcoll_General(double z, double lnM_min, double lnM_max);

float erfcc(float x);
double splined_erfc(double x);


double MtoR(double M);
double RtoM(double R);
double TtoM(double z, double T, double mu);
double dicke(double z);
double dtdz(float z);
double ddickedt(double z);
double omega_mz(float z);
double drdz(float z); /* comoving distance, (1+z)*C*dtdz(in cm) per unit z */

double M_J_WDM();
#endif