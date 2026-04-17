#To find the version we need to find python and run setuptools-scm, while keeping the project() call first
project('21cmFAST', ['c', 'cpp'],
   version : run_command(find_program('python'), '-c', 'from setuptools_scm import get_version; print(get_version())', check: true).stdout().strip(),
   default_options : ['cpp_std=c++17'],
)

# ==============================================================================
# Environment Variable Handling
# ==============================================================================
# PY21C_USE_CUDA takes precedence over USE_CUDA (for consistency with other PY21C_ variables)
# USE_CUDA is kept for backwards compatibility

py21c_cuda_str = run_command('printenv', 'PY21C_USE_CUDA', check: false).stdout().strip()
use_cuda_str = run_command('printenv', 'USE_CUDA', check: false).stdout().strip()

if py21c_cuda_str != ''
    # PY21C_USE_CUDA takes precedence
    if py21c_cuda_str == 'TRUE'
        env_cuda_flag = true
        message('CUDA enabled via PY21C_USE_CUDA=TRUE')
    else
        env_cuda_flag = false
        message('CUDA disabled via PY21C_USE_CUDA=' + py21c_cuda_str)
    endif
elif use_cuda_str != ''
    # Fall back to USE_CUDA
    if use_cuda_str == 'TRUE'
        env_cuda_flag = true
        message('CUDA enabled via USE_CUDA=TRUE')
    else
        env_cuda_flag = false
        message('CUDA disabled via USE_CUDA=' + use_cuda_str)
    endif
else
    # Fall back to meson option
    env_cuda_flag = get_option('use_cuda')
    if env_cuda_flag
        message('CUDA enabled via meson option')
    else
        message('CUDA disabled (default)')
    endif
endif

# Read other PY21C_* environment variables
# These are passed to src/py21cmfast/src/meson.build via variables

# PY21C_DEBUG: Override to debug mode (-O0 -g)
env_py21c_debug = run_command('printenv', 'PY21C_DEBUG', check: false).stdout().strip()
py21c_debug_flag = (env_py21c_debug == 'TRUE')

# PY21C_OPT_LEVEL: debug, release (default), fast
env_py21c_opt_level = run_command('printenv', 'PY21C_OPT_LEVEL', check: false).stdout().strip()
if env_py21c_opt_level == ''
    py21c_opt_level = 'release'
else
    py21c_opt_level = env_py21c_opt_level
endif

# PY21C_DEBUG overrides PY21C_OPT_LEVEL
if py21c_debug_flag
    if py21c_opt_level != 'debug' and py21c_opt_level != 'release'
        warning('PY21C_DEBUG=TRUE overrides PY21C_OPT_LEVEL=' + py21c_opt_level)
    endif
    py21c_opt_level = 'debug'
endif

message('Optimization level: ' + py21c_opt_level)

# PY21C_CUDA_DEBUG: CUDA-specific debug symbols (-g -G -lineinfo)
env_py21c_cuda_debug = run_command('printenv', 'PY21C_CUDA_DEBUG', check: false).stdout().strip()
py21c_cuda_debug_flag = (env_py21c_cuda_debug == 'TRUE')

if py21c_cuda_debug_flag and not env_cuda_flag
    warning('PY21C_CUDA_DEBUG=TRUE has no effect when CUDA is disabled')
endif

# PY21C_PGO_PHASE: generate, use, or empty (disabled)
env_py21c_pgo_phase = run_command('printenv', 'PY21C_PGO_PHASE', check: false).stdout().strip()
py21c_pgo_phase = env_py21c_pgo_phase

if py21c_pgo_phase != ''
    message('PGO phase: ' + py21c_pgo_phase)
endif

# PY21C_PGO_DIR: directory for PGO profile data (default: ./pgo_profiles)
env_py21c_pgo_dir = run_command('printenv', 'PY21C_PGO_DIR', check: false).stdout().strip()
if env_py21c_pgo_dir == ''
    py21c_pgo_dir = meson.current_source_dir() / 'pgo_profiles'
else
    py21c_pgo_dir = env_py21c_pgo_dir
endif

if py21c_pgo_phase != ''
    message('PGO profile directory: ' + py21c_pgo_dir)
endif

nvcc_prog = find_program('nvcc', required: false)

# Add CUDA language if both the environment variable is set AND nvcc is available
if env_cuda_flag and nvcc_prog.found()
  add_languages('cuda')
  # For CUDA, we need to pass the C++ standard as a compiler argument since we're adding the language dynamically
  add_project_arguments('--std=c++17', language: 'cuda')
  message('CUDA enabled: nvcc found at ' + nvcc_prog.full_path())
else
  if env_cuda_flag and not nvcc_prog.found()
    warning('USE_CUDA=TRUE but nvcc not found, building CPU-only version')
  endif
  message('Building CPU-only version')
endif

py = import('python').find_installation(pure: false)

subdir('src')
