# Define the C source files
c_source_files = files([
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
  'indexing.c',
  'integral_wrappers.c',
  'interp_tables.c',
  'interpolation.c',
  'photoncons.c',
  'recombinations.c',
  'rng.c',
  'scaling_relations.c',
  'thermochem.c',
  'MapMass_cpu.c',
])

# Define the CUDA source files
cuda_source_files = files([
  'HaloField.cu',
  'IonisationBox.cu',
  'SpinTemperatureBox.cu',
  'Stochasticity.cu',
  'filtering.cu',
  'hmf.cu',
  'interp_tables.cu',
  'interpolation.cu',
  'device_rng.cu',
  'MapMass_gpu.cu',
  'test_Stochasticity.cu',
  'cuda_hello_world.cu',
])

# C++ wrapper file
cpp_source_files = files(['_wrapper.cpp'])

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

# Base dependencies (always needed)
deps = [omp, gsl, nanobind, fftw, fftw_threads]

# CUDA dependency (optional)
# The root meson.build already checked for USE_CUDA environment variable and nvcc availability
cuda_dep = dependency('cuda', version: '>=10.0', required: false)

# Check if CUDA language is available (this will be true only if root meson.build successfully added it)
if cuda_dep.found()
  message('Using CUDA compilation in source build')
  cuda_compiler = meson.get_compiler('cuda', required: false)
  add_project_arguments('-DUSE_CUDA=1', language: ['c', 'cpp', 'cuda'])

  # CUDA compiler arguments
  # Try to detect GPU architecture, fall back to a reasonable default
  detect_arch_cmd = run_command('nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits', check: false)

  if detect_arch_cmd.returncode() == 0 and detect_arch_cmd.stdout().strip() != ''
    # Parse the compute capability (e.g., "7.5" -> "sm_75")
    compute_cap = detect_arch_cmd.stdout().strip().split('\n')[0]
    arch_major = compute_cap.split('.')[0]
    arch_minor = compute_cap.split('.')[1]
    detected_arch = 'sm_' + arch_major + arch_minor
    message('Detected GPU architecture: ' + detected_arch)
    cuda_arch = detected_arch
  else
    # Fall back to a widely compatible architecture
    cuda_arch = 'sm_60'  # Pascal - widely compatible
    message('Could not detect GPU architecture, using default: ' + cuda_arch)
  endif

  cuda_args = [
    '-arch=' + cuda_arch,
    '--extended-lambda',
    '--expt-relaxed-constexpr',
  ]
  add_project_arguments(cuda_args, language: 'cuda')

  # Add CUDA runtime library
  cuda_rt = cuda_compiler.find_library('cudart', required: true)
  deps = deps + [cuda_dep, cuda_rt]

  # Include CUDA source files
  all_source_files = c_source_files + cuda_source_files + cpp_source_files
else
  message('Using CPU-only compilation in source build')
  add_project_arguments('-DUSE_CUDA=0', language: ['c', 'cpp'])
  # Only include C/C++ files, no CUDA files
  all_source_files = c_source_files + cpp_source_files
endif

# Define a mapping of log level strings to integers
log_level_map = {
  'NO_LOG': 0,
  'ERROR': 1,
  'WARNING': 2,
  'INFO': 3,
  'DEBUG': 4,
  'SUPER_DEBUG': 5,
  'ULTRA_DEBUG': 6,
}

# Check for environment variable first, then fall back to meson option
env_log_level = run_command('printenv', 'LOG_LEVEL', check: false).stdout().strip()
if env_log_level != ''
  log_level_str = env_log_level
else
  log_level_str = get_option('log_level')
endif

# Convert the string to an integer using the map, defaulting to 2 (warnings) if the key is invalid
log_level = log_level_map.get(log_level_str, 2)

# Print the selected log level for debugging purposes
message('Selected log level: ' + log_level.to_string())

add_project_arguments('-DLOG_LEVEL=' + log_level.to_string(), language: 'c')

# Define the Python extension module
py.extension_module(
  'c_21cmfast',
  all_source_files,
  dependencies: deps,
  install: true,
  subdir:'py21cmfast',
)
