# Define the source files that contribute to the 21cmFast build
source_files = [
  'BrightnessTemperatureBox.c',
  'HaloBox.c',
  'HaloField.c',
  'InitialConditions.c',
  'InputParameters.c',
  'IonisationBox.c',
  'LuminosityFunction.c',
  'PerturbField.c',
  'PerturbHaloField.c',
  'SpinTemperatureBox.c',
  'Stochasticity.c',
  'bubble_helper_progs.c',
  'cosmology.c',
  'debugging.c',
  'dft.c',
  'elec_interp.c',
  'filtering.c',
  'heating_helper_progs.c',
  'hmf.c',
  'interp_tables.c',
  'interpolation.c',
  'photoncons.c',
  'recombinations.c',
  'subcell_rsds.c',
  'thermochem.c',
  'MapMass_cpu.c',
  'rng.c',
  '_functionprototypes_wrapper.h',
  '_inputparams_wrapper.h',
  '_outputstructs_wrapper.h',
  '_wrapper.cpp',
]

# Define the 21cmFast dependencies
omp = dependency('openmp')
gsl = dependency('gsl')
nanobind = dependency('nanobind', static: true)

# If/when fftw gets added to Meson WrapDB, we'll be able to use this:
# fftw = dependency('fftw3f_threads')
# ... but until then, we need to jump through some hoops:
cc = meson.get_compiler ('c')
search_paths = [ '/usr/lib', '/usr/local/lib', '/opt/homebrew/lib' ]
fftw = cc.find_library ('fftw3f', required: true, dirs: search_paths)
fftw_threads = cc.find_library ('fftw3f_threads', required: true, dirs: search_paths)

# Define the Python extension module
deps = [omp,gsl,fftw,fftw_threads,nanobind]
py.extension_module(
  'c_21cmfast',
  source_files,
  dependencies: deps,
  install: true,
  subdir:'py21cmfast', 
)
