project('21cmFAST', 'c')

source_files = [
  'src/py21cmfast/src/BrightnessTemperatureBox.c',
  'src/py21cmfast/src/HaloBox.c',
  'src/py21cmfast/src/HaloField.c',
  'src/py21cmfast/src/InitialConditions.c',
  'src/py21cmfast/src/InputParameters.c',
  'src/py21cmfast/src/IonisationBox.c',
  'src/py21cmfast/src/LuminosityFunction.c',
  'src/py21cmfast/src/PerturbField.c',
  'src/py21cmfast/src/PerturbHaloField.c',
  'src/py21cmfast/src/SpinTemperatureBox.c',
  'src/py21cmfast/src/Stochasticity.c',
  'src/py21cmfast/src/bubble_helper_progs.c',
  'src/py21cmfast/src/cosmology.c',
  'src/py21cmfast/src/debugging.c',
  'src/py21cmfast/src/dft.c',
  'src/py21cmfast/src/elec_interp.c',
  'src/py21cmfast/src/filtering.c',
  'src/py21cmfast/src/heating_helper_progs.c',
  'src/py21cmfast/src/hmf.c',
  'src/py21cmfast/src/interp_tables.c',
  'src/py21cmfast/src/interpolation.c',
  'src/py21cmfast/src/photoncons.c',
  'src/py21cmfast/src/recombinations.c',
  'src/py21cmfast/src/subcell_rsds.c',
  'src/py21cmfast/src/thermochem.c',
]

omp = dependency('openmp')
gsl = dependency('gsl')

# fftw = dependency('fftw3f_threads')
cc = meson.get_compiler ('c')
search_paths = [ '/usr/lib', '/usr/local/lib', '/opt/homebrew/lib' ]
fftw = cc.find_library ('fftw3f', required: true, dirs: search_paths)
fftw_threads = cc.find_library ('fftw3f_threads', required: true, dirs: search_paths)

library('21cmFAST', source_files, dependencies: [omp,gsl,fftw,fftw_threads])
