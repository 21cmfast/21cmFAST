// Things needed for Radio excess

// nu0 is degenerate with fR so no reason to leave this as a param
#define astro_nu0 0.15	   // in GHz
#define History_box_DIM 20 // number of quantities to be saved in History_box

// Print debug info array to a file, info contains: History_box, Gas Temp
#define Debug_Printer 1

int Find_Index(double *x_axis, double x, int nx)
{
	/*
	Find the index of closest left element, need this for interpolation
	range handle:
		if x is on the LEFT of x_axis[0] : return -1
		if x is on the RIGHT of x_axis[nx-1] : return nx
	*/
	double x1, x2, x3;
	int id1, id2, id3, Stop, s1, s2, s3, idx, count, reversed;
	id1 = 0;
	id3 = nx - 1;
	Stop = 0;
	x1 = x_axis[id1];
	x3 = x_axis[id3];
	reversed = x1 < x3 ? 0 : 1;
	if (!reversed)
	{
		if (x <= x1)
		{
			Stop = 1;
			idx = -1;
		}
		if (x >= x3)
		{
			Stop = 1;
			idx = nx - 1;
		}
	}
	else
	{
		// printf("x1 = %f, x3 = %f\n", x1, x3);
		if (x >= x1)
		{
			Stop = 1;
			idx = -1;
		}
		if (x <= x3)
		{
			Stop = 1;
			idx = nx - 1;
		}
	}

	count = 0;
	while (Stop == 0)
	{
		count = count + 1;
		id2 = (int)round((((double)(id1 + id3))) / 2.0);
		if (id3 == id1 + 1)
		{
			idx = id1;
			Stop = 1;
		}

		x1 = x_axis[id1];
		x2 = x_axis[id2];
		x3 = x_axis[id3];

		if (!reversed)
		{
			if (x < x2)
			{
				id3 = id2;
			}
			else
			{
				id1 = id2;
			}
		}
		else
		{
			if (x < x2)
			{
				id1 = id2;
			}
			else
			{
				id3 = id2;
			}
		}
		if (count > 100)
		{
			fprintf(stderr, "Error: solution not found after 100 iterations.\n");
			exit(1);
		}
	}

	// printf("Stopping, id1 = %d, id3 = %d, x1 = %f, x = %f, x3 = %f, idx = %d\n", id1, id3, x1, x, x3, idx);

	return idx;
}

double Interp_1D(double x, double *x_axis, double *y_axis, int nx, int Use_LogX, int Use_LogY, int Overflow_Handle)
{
	/* Find value of y at x
	Use_LogX : whether to use log axis for x
	Use_LogY : whether to use log axis for y
	Overflow_Handle : what to do if x is not in x_axis
					  0 : raise error and exit
					  1 : give nearest value
					  2 : give -100
	*/
	int id1, id2;
	double x1, x2, y1, y2, x_, r, Small;
	id1 = Find_Index(x_axis, x, nx);
	Small = 1e-280;

	if (id1 == -1)
	{
		if (Overflow_Handle == 1)
		{
			r = y_axis[0];
		}
		else if (Overflow_Handle == 2)
		{
			r = -100.;
		}
		else
		{
			fprintf(stderr, "Error from Interp_1D: x is not in range, axis range: [%E   %E], x = %E\n", x_axis[0], x_axis[nx - 1], x);
			exit(1);
		}
	}
	else if (id1 == nx - 1)
	{
		if (Overflow_Handle == 1)
		{
			r = y_axis[nx - 1];
		}
		else if (Overflow_Handle == 2)
		{
			r = -100.;
		}
		else
		{
			fprintf(stderr, "Error from Interp_1D: x is not in range, axis range: [%E   %E], x = %E\n", x_axis[0], x_axis[nx - 1], x);
			exit(1);
		}
	}
	else
	{
		id2 = id1 + 1;
		if (!Use_LogX)
		{
			x1 = x_axis[id1];
			x2 = x_axis[id2];
			x_ = x;
		}
		else
		{
			// Detect negative element
			x1 = x_axis[id1];
			x2 = x_axis[id2];
			if (((x1 < 0) || (x2 < 0)) || (x < 0))
			{
				fprintf(stderr, "cannot use LogX for axis or x with negative element\n");
				exit(1);
			}

			x1 = log(x1);
			x2 = log(x2);
			x_ = log(x);
		}
		y1 = y_axis[id1];
		y2 = y_axis[id2];

		if (Use_LogY)
		{
			// This is to avoid nan at log
			if ((y1 < 0) || (y2 < 0))
			{
				fprintf(stderr, "cannot use LogY for axis with negative element\n");
				exit(1);
			}

			y1 = y1 > Small ? y1 : Small;
			y2 = y2 > Small ? y2 : Small;
			y1 = log(y1);
			y2 = log(y2);
		}

		r = (y2 - y1) * (x_ - x1) / (x2 - x1) + y1;

		if (Use_LogY)
		{
			r = exp(r);
		}
		// printf("x_ = %f, x1 = %f, x2 = %f, y1 = %f, y2 = %f\n", x_, x1, x2, y1, y2);
	}

	return r;
}

double History_box_Interp(struct TsBox *previous_spin_temp, double z, int Type)
{
	/*
	Interpolate to find quantities archived in History_box
	Initial test shows very good (0.3%) consistency
	---- inputs ----
	z: redshift
	Type: what do you want from the box
		1 - Phi_II
		2 - Phi_III
		3 - Tk
		4 - mturn_II
		5 - mturn_III
	*/
	int ArchiveSize, idx, head, zid, fid;
	// Very generous with memory, nobody is gonna run lightcones with 10000 timesteps (?)
	double z_axis[10000], f_axis[10000], r;

	ArchiveSize = (int)round(previous_spin_temp->History_box[0]);
	if (ArchiveSize > 8000)
	{
		fprintf(stderr, "Error: ArchiveSize exceeds z_axis size.\n");
		Throw(ValueError);
	}

	// Fill axis
	for (idx = 0; idx < ArchiveSize; idx++)
	{
		head = idx * History_box_DIM + 1;
		if ((Type == 1) || (Type == 2))
		{
			// for Phi z_axis should be zpp
			zid = head + 4;
		}
		else
		{
			zid = head;
		}
		if (Type == 1)
		{ // Phi
			fid = head + 1;
		}
		else if (Type == 2)
		{ // Phi3
			fid = head + 3;
		}
		else if (Type == 3)
		{ // Tk
			fid = head + 2;
		}
		else if (Type == 4)
		{ // mturn
			fid = head + 5;
		}
		else if (Type == 5)
		{ // mturn_III
			fid = head + 6;
		}
		else
		{
			LOG_ERROR("Wrong Type setting, must be in [1, 6].\n");
			Throw(ValueError);
		}
		z_axis[idx] = previous_spin_temp->History_box[zid];
		f_axis[idx] = previous_spin_temp->History_box[fid];
	}
	// Use_LogY for all
	r = Interp_1D(z, z_axis, f_axis, ArchiveSize, 0, 1, 2);

	return r;
}

double Get_Radio_Temp_HMG(struct TsBox *previous_spin_temp, struct AstroParams *astro_params, struct CosmoParams *cosmo_params, struct FlagOptions *flag_options, double zpp_max, double redshift)
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
			Phi = Phi > 1e-50 ? Phi : 0.;
			Phi_mini = Phi_mini > 1e-50 ? Phi_mini : 0.;
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

	SFRD = Phi * cosmo_params->OMb * RHOcrit * f710 * pow(1.0 + z, astro_params->X_RAY_SPEC_INDEX + 1.0) * H * SperYR;

	return SFRD;
}

// ---- debug features ----
// to be removed after testing

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

double get_mturn_interp_EoS_tmp(double z)
{
	int n, idx;
	double r;
	double z_axis[100] = {
		5.39517E+00, 5.64340E+00, 5.89163E+00, 6.13987E+00, 6.38810E+00, 6.63633E+00, 6.88456E+00, 7.13280E+00, 7.38103E+00, 7.62926E+00,
		7.87749E+00, 8.12572E+00, 8.37396E+00, 8.62219E+00, 8.87042E+00, 9.11865E+00, 9.36689E+00, 9.61512E+00, 9.86335E+00, 1.01116E+01,
		1.03598E+01, 1.06080E+01, 1.08563E+01, 1.11045E+01, 1.13527E+01, 1.16010E+01, 1.18492E+01, 1.20974E+01, 1.23457E+01, 1.25939E+01,
		1.28421E+01, 1.30904E+01, 1.33386E+01, 1.35868E+01, 1.38351E+01, 1.40833E+01, 1.43315E+01, 1.45798E+01, 1.48280E+01, 1.50762E+01,
		1.53245E+01, 1.55727E+01, 1.58209E+01, 1.60692E+01, 1.63174E+01, 1.65656E+01, 1.68139E+01, 1.70621E+01, 1.73103E+01, 1.75586E+01,
		1.78068E+01, 1.80550E+01, 1.83033E+01, 1.85515E+01, 1.87997E+01, 1.90480E+01, 1.92962E+01, 1.95444E+01, 1.97926E+01, 2.00409E+01,
		2.02891E+01, 2.05373E+01, 2.07856E+01, 2.10338E+01, 2.12820E+01, 2.15303E+01, 2.17785E+01, 2.20267E+01, 2.22750E+01, 2.25232E+01,
		2.27714E+01, 2.30197E+01, 2.32679E+01, 2.35161E+01, 2.37644E+01, 2.40126E+01, 2.42608E+01, 2.45091E+01, 2.47573E+01, 2.50055E+01,
		2.52538E+01, 2.55020E+01, 2.57502E+01, 2.59985E+01, 2.62467E+01, 2.64949E+01, 2.67432E+01, 2.69914E+01, 2.72396E+01, 2.74879E+01,
		2.77361E+01, 2.79843E+01, 2.82325E+01, 2.84808E+01, 2.87290E+01, 2.89772E+01, 2.92255E+01, 2.94737E+01, 2.97219E+01, 2.99702E+01};

	double lmt_axis[100] = {
		8.46618E+00, 8.31289E+00, 8.16164E+00, 8.02782E+00, 7.91852E+00, 7.82774E+00, 7.74884E+00, 7.67793E+00, 7.61205E+00, 7.54858E+00,
		7.48743E+00, 7.42967E+00, 7.37632E+00, 7.32775E+00, 7.28281E+00, 7.24020E+00, 7.19866E+00, 7.15769E+00, 7.11745E+00, 7.07812E+00,
		7.03990E+00, 7.00295E+00, 6.96732E+00, 6.93301E+00, 6.89998E+00, 6.86825E+00, 6.83778E+00, 6.80856E+00, 6.78048E+00, 6.75343E+00,
		6.72725E+00, 6.70181E+00, 6.67699E+00, 6.65263E+00, 6.62862E+00, 6.60480E+00, 6.58107E+00, 6.55744E+00, 6.53405E+00, 6.51103E+00,
		6.48851E+00, 6.46663E+00, 6.44550E+00, 6.42528E+00, 6.40601E+00, 6.38760E+00, 6.36995E+00, 6.35297E+00, 6.33654E+00, 6.32056E+00,
		6.30493E+00, 6.28955E+00, 6.27432E+00, 6.25912E+00, 6.24387E+00, 6.22859E+00, 6.21337E+00, 6.19829E+00, 6.18344E+00, 6.16891E+00,
		6.15479E+00, 6.14116E+00, 6.12812E+00, 6.11569E+00, 6.10380E+00, 6.09240E+00, 6.08142E+00, 6.07081E+00, 6.06050E+00, 6.05042E+00,
		6.04053E+00, 6.03076E+00, 6.02112E+00, 6.01160E+00, 6.00219E+00, 5.99288E+00, 5.98368E+00, 5.97457E+00, 5.96554E+00, 5.95661E+00,
		5.94774E+00, 5.93897E+00, 5.93029E+00, 5.92171E+00, 5.91325E+00, 5.90491E+00, 5.89671E+00, 5.88864E+00, 5.88073E+00, 5.87300E+00,
		5.86546E+00, 5.85814E+00, 5.85106E+00, 5.84424E+00, 5.83770E+00, 5.83146E+00, 5.82555E+00, 5.81998E+00, 5.81477E+00, 5.80995E+00};

	r = Interp_1D(z, z_axis, lmt_axis, 100, 0, 0, 1);
	r = pow(10.0, r);

	return r;
}

void Print_Nion_MINI(double z, struct AstroParams *astro_params)
{
	double r, mturn, matom, Mlim_Fstar_MINI;
	FILE *OutputFile;
	mturn = get_mturn_interp_EoS_tmp(z);
	matom = atomic_cooling_threshold(z);
	r = Nion_General_MINI(z, global_params.M_MIN_INTEGRAL, mturn, matom, 0., 0., astro_params->F_STAR7_MINI, 1., 0., 0.);
	OutputFile = fopen("Nion_Table_tmp.txt", "a");
	fprintf(OutputFile, "%E  %E\n", z, r);
	fclose(OutputFile);
}

void Test_History_box_Interp(struct TsBox *previous_spin_temp, struct AstroParams *astro_params, struct CosmoParams *cosmo_params)
{
	int idx, nz;
	double Phi, Phi3, Tk, mt, mt3, dz, z1, z2, z, H, SFRD, SFRD3;
	FILE *OutputFile;
	remove("Test_History_box_Interp_tmp.txt");
	OutputFile = fopen("Test_History_box_Interp_tmp.txt", "w");
	fprintf(OutputFile, "     z           Phi            Phi3          Tk       mturn          mturn3          SFRD		SFRD3\n");

	nz = 1000;
	z1 = 5;
	z2 = 34;
	dz = (z2 - z1) / (((double)nz) - 1);

	for (idx = 0; idx < nz; idx++)
	{
		z = z1 + ((double)idx) * dz;
		H = hubble(z);
		Phi = History_box_Interp(previous_spin_temp, z, 1);
		Phi3 = History_box_Interp(previous_spin_temp, z, 2);
		Tk = History_box_Interp(previous_spin_temp, z, 3);
		mt = History_box_Interp(previous_spin_temp, z, 4);
		mt3 = History_box_Interp(previous_spin_temp, z, 5);
		SFRD = Phi_2_SFRD(Phi, z, H, astro_params, cosmo_params, 0);
		SFRD3 = Phi_2_SFRD(Phi3, z, H, astro_params, cosmo_params, 1);

		fprintf(OutputFile, "%f  ", z);
		fprintf(OutputFile, "%E  ", Phi);
		fprintf(OutputFile, "%E  ", Phi3);
		fprintf(OutputFile, "%E  ", Tk);
		fprintf(OutputFile, "%E  ", mt);
		fprintf(OutputFile, "%E  ", mt3);
		fprintf(OutputFile, "%E  ", SFRD);
		fprintf(OutputFile, "%E \n", SFRD3);
	}

	fclose(OutputFile);
}