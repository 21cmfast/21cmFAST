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
]

omp = dependency('openmp')
gsl = dependency('gsl')
nanobind = dependency('nanobind', static: true)

# fftw = dependency('fftw3f_threads')
cc = meson.get_compiler ('c')
search_paths = [ '/usr/lib', '/usr/local/lib', '/opt/homebrew/lib' ]
fftw = cc.find_library ('fftw3f', required: true, dirs: search_paths)
fftw_threads = cc.find_library ('fftw3f_threads', required: true, dirs: search_paths)


deps = [omp,gsl,fftw,fftw_threads,nanobind]
target_dir = 'py21cmfast/c_21cmfast'
py.extension_module(
  source_files,
  dependencies: deps,
  install: true,
  subdir: target_dir
)
# library('21cmFAST', source_files, dependencies: [omp,gsl,fftw,fftw_threads])
