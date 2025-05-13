# Define the source files that contribute to the 21cmFast build
source_files = files([
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
  'integral_wrappers.c',
  'interp_tables.c',
  'interpolation.c',
  'photoncons.c',
  'recombinations.c',
  'rng.c',
  'scaling_relations.c',
  'subcell_rsds.c',
  'thermochem.c',
  'MapMass_cpu.c',
  '_wrapper.cpp',
])

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

# Define a mapping of string values to integers (enum-like structure)
log_level_map = {
  'NO_LOG': 0,
  'ERROR': 1,
  'WARNING': 2,
  'INFO': 3,
  'DEBUG': 4,
  'SUPER_DEBUG': 5,
  'ULTRA_DEBUG': 6,
}

# Get the LOG_LEVEL environment variable (default to 'WARNING' if not set)
log_level_str = get_option('log_level')

# Convert the string to an integer using the map, defaulting to 2 (warnings) if the key is invalid
log_level = log_level_map.get(log_level_str, 2)

# Print the selected log level for debugging purposes
message('Selected log level: ' + log_level.to_string())

add_project_arguments('-DLOG_LEVEL=' + log_level.to_string(), language: 'c')

# Define the Python extension module
deps = [omp,gsl,fftw,fftw_threads,nanobind]
py.extension_module(
  'c_21cmfast',
  source_files,
  dependencies: deps,
  install: true,
  subdir:'py21cmfast',
)
