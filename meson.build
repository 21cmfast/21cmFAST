#To find the version we need to find python and run setuptools-scm, while keeping the project() call first
project('21cmFAST', [ 'c', 'cpp' ],
   version : run_command(find_program('python'), '-c', 'from setuptools_scm import get_version; print(get_version())', check: true).stdout().strip(),
   default_options : ['cpp_std=c++17'],
)

py = import('python').find_installation(pure: false)

subdir('src')
