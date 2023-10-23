// Things needed for Radio excess

// nu0 is degenerate with fR so no reason to leave this as a param
#define astro_nu0 0.15 // in GHz

// Print debug info array to a file, info contains: History_box, Gas Temp
#define Debug_Printer 1
#define Reset_Radio_Temp_HMG 0

double History_box_Interp(struct TsBox *previous_spin_temp, double z, int Type)
{
	// Interpolate to find quantities archived in History_box
	// Initial test using Test_History_box_Interp shows very good (0.3%) consistency
	// ---- inputs ----
	// z: redshift
	// Type: what do you want from the box, 1 - Phi_ACG, 2- Phi_MCG, 3 - Tk
	int ArchiveSize, zid1, zid2, bingo, idx, fid1, fid2, zmin, zmax;
	double z1, z2, f1, f2, f, Internal_Debug_Switch;

	ArchiveSize = (int)round(previous_spin_temp->History_box[0]);
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
		z1 = previous_spin_temp->History_box[zid1];
		z2 = previous_spin_temp->History_box[zid2];

		if ((z2 <= z) && (z < z1))
		{
			bingo = idx;
		}
	}

	if (bingo == 0) // z not in range?
	{
		if (Type != 3)
		{
			zmin = previous_spin_temp->History_box[ArchiveSize * 5];
			zmax = previous_spin_temp->History_box[5];
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
		z1 = previous_spin_temp->History_box[zid1];
		z2 = previous_spin_temp->History_box[zid2];
		f1 = previous_spin_temp->History_box[fid1];
		f2 = previous_spin_temp->History_box[fid2];

		f = (f2 - f1) * (z - z1) / (z2 - z1) + f1;

		return f;
	}
}

double Get_Radio_Temp_HMG_Astro(struct TsBox *previous_spin_temp, struct AstroParams *astro_params, struct CosmoParams *cosmo_params, struct FlagOptions *flag_options, double zpp_max, double redshift)
{

	// Find Radio Temp from sources in redshifts [zpp_max, Z_Heat_max]
	// ---- inputs ----
	// zpp_max: maximum zpp

	double z1, z2, dz, Phi, Phi_mini, z, fun_ACG, fun_MCG, Radio_Temp, Radio_Prefix_ACG, Radio_Prefix_MCG;
	int nz, zid;

	nz = 1000;
	z2 = previous_spin_temp->History_box[5] - 0.01;
	z1 = zpp_max;
	dz = (z2 - z1) / (((double)nz) - 1);

	if (flag_options->USE_RADIO_ACG)
	{
		Radio_Prefix_ACG = 113.6161 * astro_params->fR * cosmo_params->OMb * (pow(cosmo_params->hlittle, 2)) * (astro_params->F_STAR10) * pow(astro_nu0 / 1.4276, astro_params->aR) * pow(1 + redshift, 3 + astro_params->aR);
	}
	else
	{
		Radio_Prefix_ACG = 0.0;
	}

	if (flag_options->USE_RADIO_MCG)
	{
		Radio_Prefix_MCG = 113.6161 * astro_params->fR_mini * cosmo_params->OMb * (pow(cosmo_params->hlittle, 2)) * (astro_params->F_STAR7_MINI) * pow(astro_nu0 / 1.4276, astro_params->aR_mini) * pow(1 + redshift, 3 + astro_params->aR_mini);
	}
	else
	{
		Radio_Prefix_MCG = 0.0;
	}

	if (z1 > z2)
	{
		return 0.0;
	}
	else
	{

		z = z1;
		Radio_Temp = 0.0;

		for (zid = 1; zid <= nz; zid++)
		{
			Phi = History_box_Interp(previous_spin_temp, z, 1);
			Phi_mini = History_box_Interp(previous_spin_temp, z, 2);
			fun_ACG = Radio_Prefix_ACG * Phi * pow(1 + z, astro_params->X_RAY_SPEC_INDEX - astro_params->aR) * dz;
			fun_MCG = Radio_Prefix_MCG * Phi_mini * pow(1 + z, astro_params->X_RAY_SPEC_INDEX - astro_params->aR_mini) * dz;
			if (z > astro_params->Radio_Zmin)
			{
				Radio_Temp += fun_ACG + fun_MCG;
			}
			z += dz;
		}
		return Radio_Temp;
	}
}

double Get_Radio_Temp_HMG(struct TsBox *previous_spin_temp, struct AstroParams *astro_params, struct CosmoParams *cosmo_params, struct FlagOptions *flag_options, struct UserParams *user_params, double zpp_max, double redshift)
{

	double Radio_Temp_HMG;
	Radio_Temp_HMG = Get_Radio_Temp_HMG_Astro(previous_spin_temp, astro_params, cosmo_params, flag_options, zpp_max, redshift);
	if (Radio_Temp_HMG < -1.0E-8)
	{
		LOG_ERROR("Negative Radio Temp? Radio_Temp_HMG = %E\n", Radio_Temp_HMG);
		Throw(ValueError);
	}
	// If for some reason you don't want to correct Radio_Temp_HMG (e.g debug)
	if (Reset_Radio_Temp_HMG == 1)
	{
		Radio_Temp_HMG = 0.0;
	}
	return Radio_Temp_HMG;
}

void Refine_T_Radio(struct TsBox *previous_spin_temp, struct TsBox *this_spin_temp, float prev_redshift, float redshift, struct AstroParams *astro_params, struct FlagOptions *flag_options)
{
	/*
	This has a number of issues:
	1. Only applicapable to sources with same spectra shape
	*/
	int box_ct;
	float T_prev, T_now, Conversion_Factor;
	if (flag_options->USE_RADIO_MCG)
	{
		if (redshift < astro_params->Radio_Zmin)
		{
			LOG_ERROR("Current module only supports Radio ACG\n");
			Throw(ValueError);
		}
	}

	Conversion_Factor = pow((1 + redshift) / (1 + prev_redshift), 3 + astro_params->aR);

	if (redshift < astro_params->Radio_Zmin)
	{
		for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++)
		{
			this_spin_temp->Trad_box[box_ct] = Conversion_Factor * previous_spin_temp->Trad_box[box_ct];
		}
	}
}

void Print_HMF(double z, struct UserParams *user_params)
{
	double lm1, lm2, dlm, lm, hmf, m, growthf;
	int nm, idx;
	FILE *OutputFile;

	// Some settings
	nm = 1000;
	lm1 = 2.0;
	lm2 = 20.0;
	growthf = dicke(z);
	dlm = (lm2 - lm1) / ((double)nm - 1.0);
	lm = lm1;
	
	OutputFile = fopen("HMF_Table_tmp.txt", "a");
    fprintf(OutputFile, "%E  ", z);

	for (idx = 0; idx < nm; idx++)
	{
		m = pow(10.0, lm);
		if (user_params->HMF == 0)
		{
			hmf = dNdM(growthf, m);
		}
		else if (user_params->HMF == 1)
		{
			hmf = dNdM_st(growthf, m);
		}
		else if (user_params->HMF == 2)
		{
			hmf = dNdM_WatsonFOF(growthf, m);
		}
		else if (user_params->HMF == 3)
		{
			hmf = dNdM_WatsonFOF_z(z, growthf, m);
		}
		else
		{
			hmf = 0.0;
			LOG_ERROR("Wrong choice of HMF!");
			Throw(ValueError);
		}
		lm = lm + dlm;

		fprintf(OutputFile, "%E  ", hmf);
	}
	fprintf(OutputFile, "\n");
	fclose(OutputFile);
}

float Phi_2_SFRD(double Phi, double z, double H, struct AstroParams *astro_params, struct CosmoParams *cosmo_params, int Use_MINI)
{
	/*
	Convert Phi to SFRD in msun/Mpc^3/yr
	*/

    double f710, SFRD;

    if (Use_MINI)
    {
        f710 = astro_params->F_STAR7_MINI;
    }
    else
    {
        f710 = astro_params->F_STAR10;
    }
	
	SFRD = Phi * cosmo_params->OMb * RHOcrit * f710 * pow(1.0+z, astro_params->X_RAY_SPEC_INDEX + 1.0) * H * SperYR;
	
	return SFRD;

}