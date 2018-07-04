from cffi import FFI
import os

ffi = FFI()
LOCATION = os.path.dirname(os.path.abspath(__file__))
CLOC = os.path.join(LOCATION, 'src', 'py21cmmc', '_21cmfast', 'src')
include_dirs = [CLOC]

# =================================================================
# Set compilation arguments dependent on environment... a bit buggy
# =================================================================
if "DEBUG" in os.environ:
    extra_compile_args = ['-fopenmp',  '-w', '-g', '-O0']
else:
    extra_compile_args = ['-fopenmp', '-Ofast', '-w']

if "FFTW_INC" in os.environ:
    include_dirs += [os.environ["FFTW_INC"]]

library_dirs = []
if "FFTW_DIR" in os.environ:
    library_dirs += [os.environ["FFTW_DIR"]]
if "GSL_DIR" in os.environ:
    library_dirs += [os.environ["GSL_DIR"]+'/lib']
# =================================================================

# This is the overall C code.
ffi.set_source(
    "py21cmmc._21cmfast._21cmfast",  # Name/Location of shared library module
    '''
    #include "drive_21cmMC_streamlined.c"
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
