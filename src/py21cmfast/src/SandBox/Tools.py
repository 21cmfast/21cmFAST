# Get inputs and define power spectra function
import h5py
import numpy as np
import os
from powerbox.tools import get_power

import py21cmfast as p21c

Stop = 0
# Definning useful params
default = "a very long string"
F = f = N = n = False
T = t = Y = y = True

while Stop == 0:
    try:
        cmd = input()
        exec(cmd)
    except EOFError:
        Stop = 1

# ------ Set other params to defaults ------
U0 = p21c.UserParams._defaults_
C0 = p21c.CosmoParams._defaults_
A0 = p21c.AstroParams._defaults_
F0 = p21c.FlagOptions._defaults_

# ---- CosmoParams ----
if SIGMA_8 == default:
    SIGMA_8 = C0["SIGMA_8"]
if hlittle == default:
    hlittle = C0["hlittle"]
if OMm == default:
    OMm = C0["OMm"]
if POWER_INDEX == default:
    POWER_INDEX = C0["POWER_INDEX"]

# ---- UserParams ----
if BOX_LEN == default:
    BOX_LEN = U0["BOX_LEN"]
if DIM == default:
    DIM = U0["DIM"]
if HII_DIM == default:
    HII_DIM = U0["HII_DIM"]
if USE_FFTW_WISDOM == default:
    USE_FFTW_WISDOM = U0["USE_FFTW_WISDOM"]
if HMF == default:
    HMF = U0["HMF"]
if USE_RELATIVE_VELOCITIES == default:
    USE_RELATIVE_VELOCITIES = U0["USE_RELATIVE_VELOCITIES"]
if POWER_SPECTRUM == default:
    POWER_SPECTRUM = U0["POWER_SPECTRUM"]
if N_THREADS == default:
    N_THREADS = U0["N_THREADS"]
if PERTURB_ON_HIGH_RES == default:
    PERTURB_ON_HIGH_RES = U0["PERTURB_ON_HIGH_RES"]
if NO_RNG == default:
    NO_RNG = U0["NO_RNG"]
if USE_INTERPOLATION_TABLES == default:
    USE_INTERPOLATION_TABLES = U0["USE_INTERPOLATION_TABLES"]
if FAST_FCOLL_TABLES == default:
    FAST_FCOLL_TABLES = U0["FAST_FCOLL_TABLES"]
if USE_2LPT == default:
    USE_2LPT = U0["USE_2LPT"]
if MINIMIZE_MEMORY == default:
    MINIMIZE_MEMORY = U0["MINIMIZE_MEMORY"]

# ---- AstroParams ----
if HII_EFF_FACTOR == default:
    HII_EFF_FACTOR = A0["HII_EFF_FACTOR"]
if F_STAR10 == default:
    F_STAR10 = A0["F_STAR10"]
if F_STAR7_MINI == default:
    F_STAR7_MINI = A0["F_STAR7_MINI"]
if ALPHA_STAR == default:
    ALPHA_STAR = A0["ALPHA_STAR"]
if ALPHA_STAR_MINI == default:
    ALPHA_STAR_MINI = A0["ALPHA_STAR_MINI"]
if F_ESC10 == default:
    F_ESC10 = A0["F_ESC10"]
if F_ESC7_MINI == default:
    F_ESC7_MINI = A0["F_ESC7_MINI"]
if ALPHA_ESC == default:
    ALPHA_ESC = A0["ALPHA_ESC"]
if M_TURN == default:
    M_TURN = A0["M_TURN"]
if R_BUBBLE_MAX == default:
    R_BUBBLE_MAX = A0["R_BUBBLE_MAX"]
if ION_Tvir_MIN == default:
    ION_Tvir_MIN = A0["ION_Tvir_MIN"]
if L_X == default:
    L_X = A0["L_X"]
if L_X_MINI == default:
    L_X_MINI = A0["L_X_MINI"]
if NU_X_THRESH == default:
    NU_X_THRESH = A0["NU_X_THRESH"]
if X_RAY_SPEC_INDEX == default:
    X_RAY_SPEC_INDEX = A0["X_RAY_SPEC_INDEX"]
if X_RAY_Tvir_MIN == default:
    X_RAY_Tvir_MIN = A0["X_RAY_Tvir_MIN"]
if F_H2_SHIELD == default:
    F_H2_SHIELD = A0["F_H2_SHIELD"]
if t_STAR == default:
    t_STAR = A0["t_STAR"]
if N_RSD_STEPS == default:
    N_RSD_STEPS = A0["N_RSD_STEPS"]
if A_LW == default:
    A_LW = A0["A_LW"]
if BETA_LW == default:
    BETA_LW = A0["BETA_LW"]
if A_VCB == default:
    A_VCB = A0["A_VCB"]
if BETA_VCB == default:
    BETA_VCB = A0["BETA_VCB"]
# Now set these Radio/PBH Params
if fR == default:
    fR = A0["fR"]
if aR == default:
    aR = A0["aR"]
if fR_mini == default:
    fR_mini = A0["fR_mini"]
if aR_mini == default:
    aR_mini = A0["aR_mini"]
if log10_mbh == default:
    log10_mbh = A0["log10_mbh"]
if log10_fbh == default:
    log10_fbh = A0["log10_fbh"]
if bh_aR == default:
    bh_aR = A0["bh_aR"]
if bh_fX == default:
    bh_fX = A0["bh_fX"]
if bh_fR == default:
    bh_fR = A0["bh_fR"]
if bh_lambda == default:
    bh_lambda = A0["bh_lambda"]
if bh_Eta == default:
    bh_Eta = A0["bh_Eta"]
if bh_spin == default:
    bh_spin = A0["bh_spin"]
if Radio_Zmin == default:
    Radio_Zmin = A0["Radio_Zmin"]


# ---- FlagOptions ----
if USE_HALO_FIELD == default:
    USE_HALO_FIELD = F0["USE_HALO_FIELD"]
if USE_MINI_HALOS == default:
    USE_MINI_HALOS = F0["USE_MINI_HALOS"]
if USE_MASS_DEPENDENT_ZETA == default:
    USE_MASS_DEPENDENT_ZETA = F0["USE_MASS_DEPENDENT_ZETA"]
if SUBCELL_RSD == default:
    SUBCELL_RSD = F0["SUBCELL_RSD"]
if INHOMO_RECO == default:
    INHOMO_RECO = F0["INHOMO_RECO"]
if USE_TS_FLUCT == default:
    USE_TS_FLUCT = F0["USE_TS_FLUCT"]
if M_MIN_in_Mass == default:
    M_MIN_in_Mass = F0["M_MIN_in_Mass"]
if PHOTON_CONS == default:
    PHOTON_CONS = F0["PHOTON_CONS"]
if FIX_VCB_AVG == default:
    FIX_VCB_AVG = F0["FIX_VCB_AVG"]

# ------ Setting params ------
CosmoParams = p21c.CosmoParams(
    SIGMA_8=SIGMA_8, hlittle=hlittle, OMm=OMm, POWER_INDEX=POWER_INDEX
)

UserParams = p21c.UserParams(
    BOX_LEN=BOX_LEN,
    DIM=DIM,
    HII_DIM=HII_DIM,
    USE_FFTW_WISDOM=USE_FFTW_WISDOM,
    HMF=HMF,
    USE_RELATIVE_VELOCITIES=USE_RELATIVE_VELOCITIES,
    POWER_SPECTRUM=POWER_SPECTRUM,
    N_THREADS=N_THREADS,
    PERTURB_ON_HIGH_RES=PERTURB_ON_HIGH_RES,
    NO_RNG=NO_RNG,
    USE_INTERPOLATION_TABLES=USE_INTERPOLATION_TABLES,
    FAST_FCOLL_TABLES=FAST_FCOLL_TABLES,
    USE_2LPT=USE_2LPT,
    MINIMIZE_MEMORY=MINIMIZE_MEMORY,
)

AstroParams = p21c.AstroParams(
    HII_EFF_FACTOR=HII_EFF_FACTOR,
    F_STAR10=F_STAR10,
    F_STAR7_MINI=F_STAR7_MINI,
    ALPHA_STAR=ALPHA_STAR,
    ALPHA_STAR_MINI=ALPHA_STAR_MINI,
    F_ESC10=F_ESC10,
    F_ESC7_MINI=F_ESC7_MINI,
    ALPHA_ESC=ALPHA_ESC,
    M_TURN=M_TURN,
    R_BUBBLE_MAX=R_BUBBLE_MAX,
    ION_Tvir_MIN=ION_Tvir_MIN,
    L_X=L_X,
    L_X_MINI=L_X_MINI,
    NU_X_THRESH=NU_X_THRESH,
    X_RAY_SPEC_INDEX=X_RAY_SPEC_INDEX,
    X_RAY_Tvir_MIN=X_RAY_Tvir_MIN,
    F_H2_SHIELD=F_H2_SHIELD,
    t_STAR=t_STAR,
    N_RSD_STEPS=N_RSD_STEPS,
    A_LW=A_LW,
    BETA_LW=BETA_LW,
    A_VCB=A_VCB,
    BETA_VCB=BETA_VCB,
    fR=fR,
    aR=aR,
    fR_mini=fR_mini,
    aR_mini=aR_mini,
    log10_mbh=log10_mbh,
    log10_fbh=log10_fbh,
    bh_aR=bh_aR,
    bh_fX=bh_fX,
    bh_fR=bh_fR,
    bh_lambda=bh_lambda,
    bh_Eta=bh_Eta,
    bh_spin=bh_spin,
    Radio_Zmin=Radio_Zmin,
)

FlagOptions = p21c.FlagOptions(
    USE_HALO_FIELD=USE_HALO_FIELD,
    USE_MINI_HALOS=USE_MINI_HALOS,
    USE_MASS_DEPENDENT_ZETA=USE_MASS_DEPENDENT_ZETA,
    SUBCELL_RSD=SUBCELL_RSD,
    INHOMO_RECO=INHOMO_RECO,
    USE_TS_FLUCT=USE_TS_FLUCT,
    M_MIN_in_Mass=M_MIN_in_Mass,
    PHOTON_CONS=PHOTON_CONS,
    FIX_VCB_AVG=FIX_VCB_AVG,
)

# ---- Power spectra ----


def compute_power(
    box,
    length,
    SizeK,
    log_bins=True,
    ignore_kperp_zero=True,
    ignore_kpar_zero=False,
    ignore_k_zero=False,
):
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=np.int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=SizeK,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k
    return res


def PowerSpectra(
    LightCone,
    SizeK=50,
    SizeZ=20,
    max_k=2,
    logk=True,
    DataFile="/home/jcang/21cmFAST-data/LightConePower.h5",
):
    """
    Calculate Power Spectra for LightCone object
    This is the beta version, check again
    ---- Inputs ----
    LightCone: p21c LC object, can come from p21c.run_lightcone or existing h5
    SizeK: Number of k values you want
    SizeZ: Number of z values you want
    DataFile: File name for Power spectra
    """

    data = []
    lightcone_redshifts = LightCone.lightcone_redshifts
    MaxNz = LightCone.n_slices  # Number of slices along z axis
    BOX_LEN = LightCone.user_params.BOX_LEN
    HII_DIM = LightCone.user_params.HII_DIM
    CellSize = BOX_LEN / HII_DIM
    min_k = 1 / BOX_LEN
    # max_k=1/CellSize

    # z indexes for requested redshifts
    Interval = max(round(MaxNz / SizeZ), 1)
    IdxZs = list(
        range(
            0,
            MaxNz,
            Interval,
        )
    )

    if len(IdxZs) > SizeZ:  # This might happen due to rounding
        IdxZs = IdxZs[:-1]
    if IdxZs[-1] != MaxNz:  # Compute the last slice too
        IdxZs.append(MaxNz)

    NZ = len(IdxZs)  # Can be different from SizeZ now

    for IdxZ in range(NZ - 1):  # Must -1 or Idx2 will get index error
        Idx1 = IdxZs[IdxZ]
        Idx2 = IdxZs[IdxZ + 1]
        chunklen = (Idx2 - Idx1) * LightCone.cell_size

        Pk, k = compute_power(
            LightCone.brightness_temp[:, :, Idx1:Idx2],
            (BOX_LEN, BOX_LEN, chunklen),
            SizeK,
            log_bins=logk,
        )
        Ps = Pk * k**3 / (2 * np.pi**2)

        # it's Possible actual k size won't match SizeK
        # Create empty dataset to be filled later
        if IdxZ == 0:
            powerspectra = np.zeros((len(k), NZ, 2), dtype=np.float)
            Z_Axis = np.zeros((NZ), dtype=np.float)
        # Replacing H5 and Z_Axis with real data
        # Warning: np size convention is abit different from matlab
        powerspectra[:, IdxZ, 0] = k[:]
        powerspectra[:, IdxZ, 1] = Ps[:]
        Z_Axis[IdxZ] = lightcone_redshifts[IdxZs[IdxZ]]

        data.append({"k": k, "delta": Ps})

    # All data now acquired, saving to file
    h5f = h5py.File(DataFile, "a")
    h5f.create_dataset("PowerSpectra", data=powerspectra)
    h5f.create_dataset("PowerSpectra_Redshifts", data=Z_Axis)
    h5f.close()

    return data


def PowerSpectra_Coeval(
    Coeval,
    Field=1,
    SizeK=50,
    max_k=2,
    logk=True,
    DataFile="/home/jcang/21cmFAST-data/CoevalPower.h5",
):
    """
    Calculate Power Spectra for LightCone object
    ---- Inputs ----
    Coeval: p21c Coeval object, can come from p21c.run_coeval or existing h5
    Field: Choose field, default is brightness_temp, can also be xH or density, etc
    SizeK: Number of k values you want
    DataFile: File name for Power spectra
    """
    BOX_LEN = Coeval.user_params.BOX_LEN
    HII_DIM = Coeval.user_params.HII_DIM
    CellSize = BOX_LEN / HII_DIM
    min_k = 1 / BOX_LEN
    if Field == 1:
        DataBox = Coeval.brightness_temp

    Pk, k = compute_power(DataBox, (BOX_LEN, BOX_LEN, BOX_LEN), SizeK, log_bins=logk)
    Ps = Pk * k**3 / (2 * np.pi**2)
    h5f = h5py.File(DataFile, "a")
    h5f.create_dataset("PowerSpectra/k", data=k)
    h5f.create_dataset("PowerSpectra/Ps", data=Ps)
    h5f.close()
