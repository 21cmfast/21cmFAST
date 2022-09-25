# A HyRec style interface for 21cmFAST
from Tools import *

CoevalFile = "/home/jcang/21cmFAST-data/CV.h5"
LightConeFile = "/home/jcang/21cmFAST-data/LC.h5"
# -------- Initializng --------
os.system("echo '---- Initializing -----'; date")

if CleanUp == 1:
    os.system("echo '---- clearing cache -----'")
    os.system("rm -r /home/jcang/21cmFAST-data/*")

# -- What you want in your LC
if USE_TS_FLUCT == True:
    # LC_Quantities = ('brightness_temp','Ts_box','xH_box','dNrec_box','z_re_box','Gamma12_box','J_21_LW_box','density','Trad_box','Tk_box')
    LC_Quantities = ("brightness_temp", "Ts_box", "xH_box", "Trad_box", "Tk_box")
    GLB_Quantities = (
        "brightness_temp",
        "Ts_box",
        "xH_box",
        "dNrec_box",
        "z_re_box",
        "Gamma12_box",
        "J_21_LW_box",
        "density",
        "Trad_box",
        "Tk_box",
        "Fcoll",
    )
else:
    LC_Quantities = ("brightness_temp", "xH_box")

if RunType == 1:
    FileName = CoevalFile
    Data = p21c.run_coeval(
        redshift=redshift,
        cosmo_params=CosmoParams,
        user_params=UserParams,
        astro_params=AstroParams,
        flag_options=FlagOptions,
    )

else:
    FileName = LightConeFile
    Data = p21c.run_lightcone(
        redshift=redshift,
        max_redshift=max_redshift,
        cosmo_params=CosmoParams,
        user_params=UserParams,
        astro_params=AstroParams,
        flag_options=FlagOptions,
        lightcone_quantities=LC_Quantities,
        global_quantities=GLB_Quantities,
    )

Data.save(FileName)

# ---- Post processing ----

if RunType == 1:
    # 1 -- Get  Power Spectra
    PowerSpectra_Coeval(Data, DataFile=FileName)

else:
    # 1 -- Get power spectra
    ps = PowerSpectra(Data, DataFile=FileName)

    # 2 -- Print lightcone_redshifts
    lightcone_redshifts = Data.lightcone_redshifts
    h5f = h5py.File(FileName, "a")
    h5f.create_dataset("lightcone_redshifts", data=lightcone_redshifts)
    h5f.close()


# if CleanUp==1:
#    os.system("rm -r /home/jcang/21cmFAST-data/cache/*")

print("-------- Data saved to:--------")
print(FileName)

os.system('echo "---- All Done ----"; date')
