// Things needed for Radio/PBH projects
// Everything is in SI unit unless specified otherwise

#include "Tables.h"
#include "Hawking_Radiation.h"

// ---- Knobs and Switches ----
// nu0 is degenerate with fR so no reason to leave this as a param
#define astro_nu0 0.15 // in GHz

// Debuging options, print out gas temp and SFRD box, I am gonna keep these
#define print_SFRD_box 1
#define Reset_Radio_Temp_HMG 0
#define debug_mode 1

// Some numerical settings for the radio BKG table
#define Integration_TimeStep 2000
#define PBH_Table_Size 100
#define RadioTab_Zmin 0
#define RadioTab_Zmax 60
#define RadioTab_Mmin 1.0E3
#define RadioTab_Mmax 1.0E15

double PBH_Radio_EMS_IGM(double z, double nu, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, double Tk, double Fcoll, double delta)
{
	// Get comoving radio emissivity from accreting PBH in IGM
	// ---- inputs ----
	// z: you guessed it, it's the redshift
	// nu: Frequency
	// Tk: Gas kinetic temp
	// Fcoll: Collapsed fraction
	// delta: Overdensity

	// prepare some numbers in SI unit
	double GSI = 6.67259e-11;	 // Newtonian Grav const
	double CSI = 2.99792458E8; // Speed of light
	double nu21 = 1.42E9;			 // 21cm frequency in Hz
	double mbh_msun = pow(10, astro_params->log10_mbh);
	double mbh = mbh_msun * 1.989E30; // pbh mass
	double h = cosmo_params->hlittle;
	double OmegaC = (cosmo_params->OMm - cosmo_params->OMb);
	double RhoDM_avg = OmegaC * pow(h, 2) * 1.879E-26;													 // DM Density today (avg)
	double RhoB_avg = cosmo_params->OMb * pow(h, 2) * 1.879E-26 * pow(1 + z, 3); // Average Baryon density
	double fbh = pow(10, astro_params->log10_fbh);
	// ---- Part I: PBH in IGM ----
	double RhoIGM = RhoB_avg * (1 + delta) * (1 - Fcoll);								// IGM gas density
	double nbh_IGM = fbh * RhoDM_avg * (1 + delta) * (1 - Fcoll) / mbh; // Comoving PBH number density
	double cs = 8.3E3 * pow(Tk / 10000, 0.5);														// IGM Sound Speed
	double vrel = 30.0 * (1.0 + z);																			// We don't really need to account for accretion from z>1000 (or do we)
	double cs2 = pow(cs, 2);
	double vrel2 = pow(vrel, 2);
	// Bolometric x-ray luminosity
	double LX = astro_params->bh_fX * astro_params->bh_Eta * astro_params->bh_lambda * 4 * PI * RhoIGM * pow(GSI * CSI * mbh, 2.0) / (pow(cs2 + vrel2, 1.5));
	double LR = astro_params->bh_fR * 1.0E33 * pow(LX / 1E37, 0.85) * pow(mbh_msun / 1.0E8, 0.12); // Radio luminosity at nu21
	double EMS = LR * pow(nu / nu21, -astro_params->bh_aR) * nbh_IGM / nu21;

	return EMS;
}

double LCDM_Tk(double z)
{
	// Interpolate to find Tk at z
	int id1, id2;
	double z1, z2, t1, t2, t;
	if (z > 1000)
	{
		t = 2.728 * (1 + z);
	}
	else
	{
		id1 = (int)floor(z);
		id2 = id1 + 1;
		z1 = (double)id1;
		z2 = z1 + 1.0;
		t1 = Adia_Tk_Table(id1);
		t2 = Adia_Tk_Table(id2);

		t = (t2 - t1) * (z - z1) / (z2 - z1) + t1;
	}

	return t;
}

double Interp_Fast(double *Tab, double xmin, double xmax, int nx, double x)
{
	// Interpolate, x axis must be linear in [xmin xmax] with size nx
	// Speed: 1E-8 seconds per call for nx=100

	double y1, y2, y, dx, x1, x2;
	int id1, id2;

	dx = (xmax - xmin) / ((double)(nx - 1));
	id1 = (int)floor((x - xmin) / dx);
	id2 = id1 + 1;
	x1 = xmin + ((double)id1) * dx;
	x2 = x1 + dx;

	if ((id1 < 0) || (id1 > nx - 2))
	{
		// Do not extrapolate
		y = 0;
	}
	else
	{
		y1 = Tab[id1];
		y2 = Tab[id2];
		y = (y2 - y1) * (x - x1) / (x2 - x1) + y1;
	}

	return y;
}

double Conditional_HMF(double growthf, double M, double Delta, double sigma2, struct CosmoParams *cosmo_params)
{
	// Return conditional HMF dn/dm, unit: 1/(Msun Mpc^3)
	// ---- inputs ----
	// growthf: growth factor
	// M: halo mass in Msun
	// Delta: Overdensity
	// sigma2: sigma2

	double OmegaMh2 = cosmo_params->OMm * pow(cosmo_params->hlittle, 2);
	double r;
	double RhoM = OmegaMh2 * 2.775E11; // Average matter density in Msun/Mpc^3
	float dNdM_old = dNdM_conditional((float)growthf, (float)log(M), 30.0, Deltac, (float)Delta, (float)sigma2);
	double dNdM_double = (double)dNdM_old;
	double RhoM_gird = RhoM * (1 + Delta);

	// This should be positive anyway, check where the - sign comes from
	return fabs(-RhoM_gird * dNdM_double / M / sqrt(2 * PI));
}

double Interp_Lh(double z, double m)
{
	// Interpolate to find radio luminosity at nu21 from halo, unit: W/Hz
	// Speed: 5E-8 seconds per call

	int zid1, zid2, mid1, mid2;
	double z1, z2, m1, m2, lnm1, lnm2, r, dz, dlnm, lnm_min, lnm_max, lnm, f1, f2, F1, F2;

	lnm = log(m);

	lnm_min = log(RadioTab_Mmin);
	lnm_max = log(RadioTab_Mmax);

	dz = (RadioTab_Zmax - RadioTab_Zmin) / ((double)RadioTab_NZ - 1);
	dlnm = (lnm_max - lnm_min) / ((double)RadioTab_NM - 1);

	zid1 = (int)floor((z - RadioTab_Zmin) / dz);
	mid1 = (int)floor((lnm - lnm_min) / dlnm);
	zid2 = zid1 + 1;
	mid2 = mid1 + 1;

	if ((zid1 < 0) || (zid1 > RadioTab_NZ - 2) || (mid1 < 0) || (mid1 > RadioTab_NM - 2))
	{
		// Do not extrapolate
		r = 0.0;
	}
	else
	{

		z1 = RadioTab_Zmin + ((double)zid1) * dz;
		z2 = RadioTab_Zmin + ((double)zid2) * dz;

		lnm1 = lnm_min + ((double)mid1) * dlnm;
		lnm2 = lnm_min + ((double)mid2) * dlnm;

		// Interpolate for lnm1
		f1 = PBH_Radio_Luminosity_Table[mid1][zid1];
		f2 = PBH_Radio_Luminosity_Table[mid1][zid2];

		F1 = (f2 - f1) * (z - z1) / (z2 - z1) + f1;

		// Interpolate for lnm2
		f1 = PBH_Radio_Luminosity_Table[mid2][zid1];
		f2 = PBH_Radio_Luminosity_Table[mid2][zid2];

		F2 = (f2 - f1) * (z - z1) / (z2 - z1) + f1;

		r = (F2 - F1) * (lnm - lnm1) / (lnm2 - lnm1) + F1;
	}

	// Convert erg to Joule unit
	return r / 1.0E7;
}

double Radio_PBH_Fid_EMS_Halo(double M1, double M2, double growthf, double z, double Delta, double sigma2, struct CosmoParams *cosmo_params, int hmf_model)
{
	// Return comoving emissivity at 21cm frequency for fiducial settings, in W/Hz/M^3
	// Using a brute force rectangular integrator for now, switch to GSL in next version
	// Set hmf_model = -1 to use conditional hmf
	// All masses are in Msun, integration is done in log space

	double lnm1, lnm2, dlnm, lnm, m, fun, hmf, result;
	int idx, TimeStep;

	if (hmf_model >= 0)
	{
		TimeStep = 5 * Integration_TimeStep;
	}
	else
	{
		TimeStep = Integration_TimeStep;
	}

	// Don't waste time on out-range masses
	lnm1 = log(fmax(M1, RadioTab_Mmin));
	lnm2 = log(fmin(M2, RadioTab_Mmax));

	dlnm = (lnm2 - lnm1) / ((double)(TimeStep - 1));
	lnm = lnm1;
	result = 0.0;

	for (idx = 0; idx < TimeStep; idx++)
	{

		// The value of your function in requested mass
		m = exp(lnm);
		fun = m * Interp_Lh(z, m);

		if (hmf_model == -1)
		{
			hmf = Conditional_HMF(growthf, m, Delta, sigma2, cosmo_params);
		}
		else if (hmf_model == 0)
		{
			// PS HMF
			hmf = dNdM(growthf, m);
		}
		else if (hmf_model == 1)
		{
			// ST HMF
			hmf = dNdM_st(growthf, m);
		}
		else if (hmf_model == 2)
		{
			// Watson HMF
			hmf = dNdM_WatsonFOF(growthf, m);
		}
		else if (hmf_model == 3)
		{
			// Watson FoF HMF
			hmf = dNdM_WatsonFOF_z(z, growthf, m);
		}
		else
		{
			hmf = 0.0;
			LOG_ERROR("Wrong choice of HMF!");
			Throw(ValueError);
		}

		result = result + fun * hmf;

		lnm = lnm + dlnm;
	}

	result = result * dlnm;

	// Converting from Mpc to m
	return result / 2.9380E67;
}

double SFRD_box_Interp(struct TsBox *previous_spin_temp, double z, int Type)
{
	// Interpolate to find quantities archived in SFRD_box
	// Initial test using Test_SFRD_box_Interp shows very good (0.3%) consistency
	// ---- inputs ----
	// z: redshift
	// Type: what do you want from the box, 1 - Phi_ACG, 2- Phi_MCG, 3 - Tk
	int ArchiveSize, zid1, zid2, bingo, idx, fid1, fid2, zmin, zmax;
	double z1, z2, f1, f2, f, Internal_Debug_Switch;

	ArchiveSize = (int)round(previous_spin_temp->SFRD_box[0]);
	bingo = 0;
	Internal_Debug_Switch = 1;

	// Do some simple sanity test
	if ((Type < 1) || (Type > 3))
	{
		LOG_ERROR("Wrong Type setting, must be in [1 2 3].\n");
		Throw(ValueError);
	}

	for (idx = 1; idx < ArchiveSize; idx++)
	{
		if ((Type == 1) || (Type == 2))
		{
			// z axis for Phi and Phi_mini
			zid1 = idx * 5;
			zid2 = zid1 + 5;
		}
		else
		{
			// z axis for Tk
			zid1 = (idx - 1) * 5 + 1;
			zid2 = zid1 + 5;
		}

		// Don't forget that z1 > z2
		z1 = previous_spin_temp->SFRD_box[zid1];
		z2 = previous_spin_temp->SFRD_box[zid2];

		if ((z2 <= z) && (z < z1))
		{
			bingo = idx;
		}
	}

	if (bingo == 0)
	{
		if (Type != 3)
		{
			zmin = previous_spin_temp->SFRD_box[ArchiveSize * 5];
			zmax = previous_spin_temp->SFRD_box[5];
			LOG_ERROR("Your redshift is not in the Archive redshift range of [%E  %E], your redshift = %E\n", zmin, zmax, z);
			Throw(ValueError);
		}
		else
		{
			// Return negative Tk as a sign that z is not in range
			return -100.0;
		}
	}
	else
	{

		if (Type == 1)
		{
			// Phi_ACG
			fid1 = (bingo - 1) * 5 + 2;
			zid1 = bingo * 5;
		}
		else if (Type == 2)
		{
			// Phi_MCG
			fid1 = (bingo - 1) * 5 + 4;
			zid1 = bingo * 5;
		}
		else
		{
			// Tk
			zid1 = (bingo - 1) * 5 + 1;
			fid1 = zid1 + 2;
		}

		fid2 = fid1 + 5;
		zid2 = zid1 + 5;
		z1 = previous_spin_temp->SFRD_box[zid1];
		z2 = previous_spin_temp->SFRD_box[zid2];
		f1 = previous_spin_temp->SFRD_box[fid1];
		f2 = previous_spin_temp->SFRD_box[fid2];

		f = (f2 - f1) * (z - z1) / (z2 - z1) + f1;

		return f;
	}
}

void Test_SFRD_box_Interp(struct TsBox *previous_spin_temp, double z1, double z2)
{
	// debug function, compare SFRD_box_Interp output with SFRD_box

	FILE *OutputFile;
	double dz, z, f1, f2;
	int nz, zid;
	nz = 100;
	dz = (z2 - z1) / (((double)nz) - 1);

	z = z1;
	remove("SFRD_Test_tmp.txt");
	OutputFile = fopen("SFRD_Test_tmp.txt", "a");
	fprintf(OutputFile, "z    Phi    Phi_mini\n");
	if (z1 > z2)
	{
		LOG_ERROR("z1 and z2 range error: z1 = %f, z2 = %f\n", z1, z2);
		Throw(ValueError);
	}

	for (zid = 1; zid < nz; zid++)
	{
		f1 = SFRD_box_Interp(previous_spin_temp, z, 1);
		f2 = SFRD_box_Interp(previous_spin_temp, z, 2);
		fprintf(OutputFile, "%f	  %E   %E\n", z, f1, f2);
		z += dz;
	}
	fclose(OutputFile);
}

double Get_Radio_Temp_HMG_Astro(struct TsBox *previous_spin_temp, struct AstroParams *astro_params, struct CosmoParams *cosmo_params, struct FlagOptions *flag_options, double zpp_max, double redshift)
{

	// Find Radio Temp from sources in redshifts [zpp_max, Z_Heat_max]
	// ---- inputs ----
	// zpp_max: maximum zpp

	double z1, z2, dz, Phi, Phi_mini, z, fun_ACG, fun_MCG, Radio_Temp, Radio_Prefix_ACG, Radio_Prefix_MCG;
	int nz, zid;
	FILE *OutputFile;

	nz = 1000;
	z2 = previous_spin_temp->SFRD_box[5] - 0.01;
	z1 = zpp_max;
	dz = (z2 - z1) / (((double)nz) - 1);

	Radio_Prefix_ACG = 113.6161 * astro_params->fR * cosmo_params->OMb * (pow(cosmo_params->hlittle, 2)) * (astro_params->F_STAR10) * pow(astro_nu0 / 1.4276, astro_params->aR) * pow(1 + redshift, 3 + astro_params->aR);
	Radio_Prefix_MCG = 113.6161 * astro_params->fR_mini * cosmo_params->OMb * (pow(cosmo_params->hlittle, 2)) * (astro_params->F_STAR7_MINI) * pow(astro_nu0 / 1.4276, astro_params->aR_mini) * pow(1 + redshift, 3 + astro_params->aR_mini);

	if ((z1 > z2) || ((!flag_options->USE_RADIO_ACG) || (!flag_options->USE_RADIO_MCG)))
	{
		return 0.0;
	}
	else
	{

		z = z1;
		Radio_Temp = 0.0;

		for (zid = 1; zid <= nz; zid++)
		{
			Phi = SFRD_box_Interp(previous_spin_temp, z, 1);
			Phi_mini = SFRD_box_Interp(previous_spin_temp, z, 2);
			fun_ACG = Radio_Prefix_ACG * Phi * pow(1 + z, astro_params->X_RAY_SPEC_INDEX - astro_params->aR) * dz;
			fun_MCG = Radio_Prefix_MCG * Phi_mini * pow(1 + z, astro_params->X_RAY_SPEC_INDEX - astro_params->aR_mini) * dz;
			z += dz;
			Radio_Temp += fun_ACG + fun_MCG;
		}
		return Radio_Temp;
	}
}

double Find_Tk(struct TsBox *previous_spin_temp, double z)
{
	// Find Tk at given redshift z
	// if z not in range, use LCDM or nearest Tk
	int ArchiveSize;
	double T, Table_zmin, Table_zmax;

	ArchiveSize = (int)round(previous_spin_temp->SFRD_box[0]);
	T = SFRD_box_Interp(previous_spin_temp, z, 3);
	Table_zmax = previous_spin_temp->SFRD_box[1];
	Table_zmin = previous_spin_temp->SFRD_box[ArchiveSize * 5];

	if (T < -1.0E-8)
	{
		// Found nothing in cache box?
		// A: z too large - use LCDM
		// B: z is below prev_z - use nearest value

		if (z > 0.99 * Table_zmax)
		{
			T = LCDM_Tk(z);
		}
		else if (z < 1.001 * Table_zmin)
		{
			T = previous_spin_temp->SFRD_box[ArchiveSize * 5 - 2];
		}
		else
		{
			LOG_ERROR("Unknown error, z not in range");
			Throw(ValueError);
		}

		return T;
	}
}

void Test_Find_Tk(struct TsBox *previous_spin_temp)
{

	int nz, id;
	double z1, z2, dz, z, T;
	FILE *OutputFile;

	z1 = 0.1;
	z2 = 100;
	nz = 1000;

	dz = (z2 - z1) / ((double)nz - 1);
	z = z1;

	remove("Test_Find_Tk_tmp.txt");
	OutputFile = fopen("Test_Find_Tk_tmp.txt", "a");
	fprintf(OutputFile, "z    Tk\n");

	for (id = 1; id <= nz; id++)
	{
		T = Find_Tk(previous_spin_temp, z);
		fprintf(OutputFile, "%f    %E\n", z, T);
		z += dz;
	}
	fclose(OutputFile);
}

double Get_Radio_Temp_HMG_PBH(struct TsBox *previous_spin_temp, double z, double zpp_max, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, int hmf_model)
{
	// Calculate PBH Radio Temp at z from sources between [zpp_max RadioTab_Zmax]
	// HMG assumed

	double dz, zp, EMS_Halo, EMS_IGM, growthf, Tk, MinM, Fcoll, nu_factor, new_nu, EMS_tot, Integrand;
	int idx;
	double mbh_msun = pow(10, astro_params->log10_mbh);
	double fbh = pow(10, astro_params->log10_fbh);

	dz = (RadioTab_Zmax - zpp_max) / ((double)PBH_Table_Size - 1);
	zp = zpp_max;

	if (hmf_model == -1)
	{
		LOG_ERROR("This module is designed only for unconditional HMF!");
		Throw(ValueError);
	}

	Integrand = 0.0;
	for (idx = 0; idx < PBH_Table_Size; idx++)
	{

		Tk = Find_Tk(previous_spin_temp, zp);
		growthf = dicke(zp);
		MinM = 1.3E3 * pow(10 / (1 + zp), 1.5) * pow(Tk, 1.5);
		nu_factor = pow((1 + zp) / (1 + z), - astro_params->bh_aR);
		new_nu = 1.42E9 * (1 + zp) / (1 + z);

		Fcoll = FgtrM_General(zp, MinM);

		// Comoving fiducial emissivity at rest-frame 21 frequency
		EMS_Halo = Radio_PBH_Fid_EMS_Halo(MinM, RadioTab_Mmax, growthf, zp, 0.0, 0.0, cosmo_params, hmf_model);
		// Revert to our model params and frequency
		EMS_Halo = fabs(astro_params->bh_fR * nu_factor * EMS_Halo * fbh * pow(mbh_msun / 10, 0.82) * pow(astro_params->bh_fX * astro_params->bh_Eta * astro_params->bh_lambda / 1E-4, 0.85));

		EMS_IGM = PBH_Radio_EMS_IGM(zp, new_nu, cosmo_params, astro_params, Tk, Fcoll, 0.0);

		// Total comoving emissivity in SI unit, at redshifted 21cm frequency, from zp
		EMS_tot = EMS_Halo + EMS_IGM;

		// 3.81E28 is [ c^3 /(8 pi kb v21^2) ]
		Integrand += pow(1 + z, 3) * 3.8101E28 * EMS_tot / (1 + zp) / hubble(zp);
		zp += dz;
	}

	return (Integrand * dz);
}

void SFRD_box_Printer(struct TsBox *previous_spin_temp)
{
	FILE *OutputFile;
	int MaxIdx, idx;
	// 6000 should be enough for debug purpose, unless the user set HII_DIM < 20
	MaxIdx = 6000;
	remove("SFRD_box_Full_Super_debug_tmp.txt");
	OutputFile = fopen("SFRD_box_Full_Super_debug_tmp.txt", "a");

	for (idx = 0; idx < MaxIdx; idx++)
	{
		fprintf(OutputFile, "%E\n", previous_spin_temp->SFRD_box[idx]);
	}
	fclose(OutputFile);
}
