#ifndef _RECOMB_H
#define _RECOMB_H

double splined_recombination_rate(double z_eff, double gamma12_bg);
/*initializes the lookup table for the PDF density integral in MHR00 model at redshift z*/
void init_MHR();

#endif
