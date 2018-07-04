from cffi import FFI
import os

ffi = FFI()
LOCATION = os.path.dirname(os.path.abspath(__file__))
CLOC = os.path.join(LOCATION, 'src', 'py21cmmc', '_21cmfast', 'src')
include_dirs = [CLOC,'/opt/local/include']

# =================================================================
# Set compilation arguments dependent on environment... a bit buggy
# =================================================================
if "DEBUG" in os.environ:
    extra_compile_args = ['-fopenmp',  '-w', '-g', '-O0']
else:
    extra_compile_args = ['-fopenmp', '-Ofast', '-w']

library_dirs = ['/opt/local/lib']
for k,v in os.environ.items():
    if "inc" in k.lower():
        include_dirs += [v]
    elif "lib" in k.lower():
        library_dirs += [v]

# =================================================================

# This is the overall C code.
ffi.set_source(
    "py21cmmc._21cmfast._21cmfast",  # Name/Location of shared library module
    '''
    #include "GenerateICs.c"
    ''',
    include_dirs = include_dirs,
    library_dirs=library_dirs,
    libraries=['m','gsl','gslcblas','fftw3f_omp', 'fftw3f'],
    extra_compile_args = extra_compile_args,
    extra_link_args=['-fopenmp']
)

# This is the Header file
with open(os.path.join(CLOC, "21CMMC.h")) as f:
    ffi.cdef(f.read())


if __name__ == "__main__":
    ffi.compile()
