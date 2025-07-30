#To find the version we need to find python and run setuptools-scm, while keeping the project() call first
project('21cmFAST', ['c', 'cpp'],
   version : run_command(find_program('python'), '-c', 'from setuptools_scm import get_version; print(get_version())', check: true).stdout().strip(),
   default_options : ['cpp_std=c++17'],
)

# Check if CUDA should be enabled after project declaration
env_cuda_str = run_command('printenv', 'USE_CUDA', check: false).stdout().strip()
if env_cuda_str == ''
    env_cuda_flag = get_option('use_cuda')
else
    if env_cuda_str == 'TRUE'
        env_cuda_flag = true
    else
        env_cuda_flag = false
    endif
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
