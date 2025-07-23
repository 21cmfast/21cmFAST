project('21cmFAST', [ 'c', 'cpp' ],
   version : run_command('cat', 'VERSION').stdout().strip(),
   default_options : ['cpp_std=c++17'],
)

py = import('python').find_installation(pure: false)

subdir('src')
