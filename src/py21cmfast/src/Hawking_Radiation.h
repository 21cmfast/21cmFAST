// Funcitons for Hawking Radiation
// Caveats: Only applies for Monochromatic PBHs, ICs not finished for Kerr PBHs

double GetEff_Table(double Spin, int Channel, int zid, int mid)
{
  // Get Deposition Efficiency matrix element for certain Spin, Channel, zid and mid
	int sid, cid;
	double r;

	cid = Channel;
	if (Spin <= 0.01)
	{
		sid = 1;
	}
	else if ((0.01 < Spin) && (Spin <= 0.26))
	{
		sid = 2;
	}
	else if ((0.26 < Spin) && (Spin <= 0.51))
	{
		sid = 3;
	}
	else if ((0.51 < Spin) && (Spin <= 0.76))
	{
		sid = 4;
	}
	else if ((0.76 < Spin) && (Spin <= 0.9995))
	{
		sid = 5;
	}
	else if (0.9995 < Spin)
	{
		sid = 6;
	}
	else
	{
		// Other values not supported, return 0
		sid = 100;
	}

	if ((sid == 1) && (cid == 1))
	{
		r = GetEff_s1c1(zid, mid);
	}
	else if ((sid == 1) && (cid == 3))
	{
		r = GetEff_s1c3(zid, mid);
	}
	else if ((sid == 1) && (cid == 4))
	{
		r = GetEff_s1c4(zid, mid);
	}
	else if ((sid == 2) && (cid == 1))
	{
		r = GetEff_s2c1(zid, mid);
	}
	else if ((sid == 2) && (cid == 3))
	{
		r = GetEff_s2c3(zid, mid);
	}
	else if ((sid == 2) && (cid == 4))
	{
		r = GetEff_s2c4(zid, mid);
	}
	else if ((sid == 3) && (cid == 1))
	{
		r = GetEff_s3c1(zid, mid);
	}
	else if ((sid == 3) && (cid == 3))
	{
		r = GetEff_s3c3(zid, mid);
	}
	else if ((sid == 3) && (cid == 4))
	{
		r = GetEff_s3c4(zid, mid);
	}
	else if ((sid == 4) && (cid == 1))
	{
		r = GetEff_s4c1(zid, mid);
	}
	else if ((sid == 4) && (cid == 3))
	{
		r = GetEff_s4c3(zid, mid);
	}
	else if ((sid == 4) && (cid == 4))
	{
		r = GetEff_s4c4(zid, mid);
	}
	else if ((sid == 5) && (cid == 1))
	{
		r = GetEff_s5c1(zid, mid);
	}
	else if ((sid == 5) && (cid == 3))
	{
		r = GetEff_s5c3(zid, mid);
	}
	else if ((sid == 5) && (cid == 4))
	{
		r = GetEff_s5c4(zid, mid);
	}
	else if ((sid == 6) && (cid == 1))
	{
		r = GetEff_s6c1(zid, mid);
	}
	else if ((sid == 6) && (cid == 3))
	{
		r = GetEff_s6c3(zid, mid);
	}
	else if ((sid == 6) && (cid == 4))
	{
		r = GetEff_s6c4(zid, mid);
	}
	else
	{
		r = 0.0;
	}
	return r;
}

double GetEFF(double lm, double z, double Spin, int Channel)
{
	// Interpolate to find deposition efficiency
	// ---- Inputs ----
	// lm: log10(M0/gram)
	// z: redshift
	// Spin: Currently tabulated for following values: [0 0.25 0.5 0.75 0.999 0.9999]
	// Channel: HIon (1), LyA (3), Heat (4)

	// Now define some anchor variables
	// M0 and z+1 axis are log distributed
	double LgM_min, LgM_max, LgZp_min, LgZp_max, nz, nm, lzp, r, dlm, dlz;
	double x, y, x1, x2, y1, y2, f11, f12, f21, f22, f1, f2;

	int mid1, mid2, zid1, zid2;

	LgM_min = 13.301;
	LgM_max = 18.0;
	LgZp_min = 1.0762;
	LgZp_max = 3.4390;
	nz = 63;
	nm = 200;
	dlm = 0.023612914594653;
	dlz = 0.038109557755943;

	lzp = log10(z + 1);

	// Check range first, do not use extrapolation
	if ((lm < LgM_min) || (lm > LgM_max) || (lzp < LgZp_min) || (lzp > LgZp_max))
	{
		r = 0;
	}
	else
	{

		mid1 = (int)floor((lm - LgM_min) / dlm);
		zid1 = (int)floor((lzp - LgZp_min) / dlz);

		// mid1 and zid1 should be in range already, but just in case there is some numerical thing
		if (mid1 < 0)
		{
			mid1 = 0;
		}
		if (mid1 > 198)
		{
			mid1 = 198;
		}

		if (zid1 < 0)
		{
			zid1 = 0;
		}
		if (zid1 > 61)
		{
			zid1 = 61;
		}

		mid2 = mid1 + 1;
		zid2 = zid1 + 1;

		x = lm;
		y = lzp;
		x1 = LgM_min + dlm * (double)mid1;
		x2 = x1 + dlm;
		y1 = LgZp_min + dlz * (double)zid1;
		y2 = y1 + dlz;

		f11 = GetEff_Table(Spin, Channel, zid1, mid1);
		f12 = GetEff_Table(Spin, Channel, zid1, mid2);
		f21 = GetEff_Table(Spin, Channel, zid2, mid1);
		f22 = GetEff_Table(Spin, Channel, zid2, mid2);

		f1 = (f12 - f11) * (x - x1) / (x2 - x1) + f11;
		f2 = (f22 - f21) * (x - x1) / (x2 - x1) + f21;

		r = (f2 - f1) * (y - y1) / (y2 - y1) + f1;
	}
	return r;
}

double dEdVdt_PBH_dep(double z, struct AstroParams *astro_params, struct CosmoParams *cosmo_params,int Channel)
{
	// HMG Deposition rate for PBH with monochromatic distribution, following Eq.3.10 of 2108.13256, in SI unit
  // ---- inputs ----
  //      Channel: deposition channel
	double c = 2.99792458E8;
	double M = 1.98E33 * astro_params->mbh;
	double OmegaC = (cosmo_params->OMm - cosmo_params->OMb);
	double h = cosmo_params->hlittle;
	double RhoDM = OmegaC * pow(h, 2) * 1.879E-26;																			// Current DM density in kg/m^3
	double dEdVdt_inj = 5.34E25 * pow(c, 2) * pow(M, -3) * astro_params->fbh * RhoDM * pow(1 + z, 3); // in SI unit
	double lm = log10(M);
	double fc = GetEFF(lm, z, astro_params->bh_spin, Channel);
	return fc * dEdVdt_inj;
}

double Set_ICs_z35(double mbh, double fbh, int Type)
{
  // Interpolate to find the increase of Xe and Tk (i.e. dXe and dTk) at z=35 in presence of Hawkign Radiation
  // Interpolation tables were computed using HyRec codes from our previous work (2108.13256), do this right in later versions (interface HyRec with 21cmFAST)
  // ---- inputs ----
  //      mbh: mbh in gram
  //      fbh: Omega_bh/Omega_dm
  //      Type: get dXe (Type = 0) or dTk (Type = 1)

	double x1, x2, y1, y2, x, y, f11, f12, f21, f22, f1, f2;
	int xid1, xid2, yid1, yid2, r;

	x = log10(mbh);
	if (fbh < 1E-30)
	{
		// Eliminate Inf error
		y = -20.0;
	}
	else
	{
		y = log10(fbh);
	}
	if ((mbh < 2.0E13) || (mbh > 1E18))
	{
		r = 0.0;
	}
	else
	{
		yid1 = (int)floor((y + 20.0) / 0.2);
		xid1 = (int)floor((x - 13.301) / 0.04698970004336);
		if (yid1 < 0)
		{
			yid1 = 0;
		}
		if (xid1 < 0)
		{
			xid1 = 0;
		}
		if (yid1 > 98)
		{
			yid1 = 98;
		}
		if (xid1 > 98)
		{
			xid1 = 98;
		}
		xid2 = xid1 + 1;
		yid2 = yid1 + 1;

		x1 = 13.301 + 0.04698970004336 * (double)xid1;
		x2 = x1 + 0.04698970004336;
		y1 = -20 + 0.2 * (double)yid1;
		y2 = y1 + 0.2;

		f11 = ICs_Query(xid1, yid1, Type);
		f12 = ICs_Query(xid2, yid1, Type);
		f21 = ICs_Query(xid1, yid2, Type);
		f22 = ICs_Query(xid2, yid2, Type);

		f1 = (f12 - f11) * (x - x1) / (x2 - x1) + f11;
		f2 = (f22 - f21) * (x - x1) / (x2 - x1) + f21;
		r = (f2 - f1) * (y - y1) / (y2 - y1) + f1;
	}
	return r;
}
