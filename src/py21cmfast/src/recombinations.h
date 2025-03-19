#ifndef _RECOMB_H
#define _RECOMB_H

double splined_recombination_rate(double z_eff, double gamma12_bg);
void init_MHR(); /*initializes the lookup table for the PDF density integral in
                    MHR00 model at redshift z*/

#endif
