"""Build the C code with CFFI."""

import os
import sys
import sysconfig
from cffi import FFI

# Get the compiler. We support gcc and clang.
_compiler = sysconfig.get_config_var("CC")

if "gcc" in _compiler:
    compiler = "gcc"
elif "clang" in _compiler:
    compiler = "clang"
else:
    raise ValueError(f"Compiler {_compiler} not supported for 21cmFAST")

ffi = FFI()

LOCATION = os.path.dirname(os.path.abspath(__file__))
CLOC = os.path.join(LOCATION, "src", "py21cmfast", "src")
include_dirs = [CLOC]

c_files = [
    os.path.join("src", "py21cmfast", "src", f)
    for f in os.listdir(CLOC)
    if f.endswith(".c")
]

# Compiled CUDA code
extra_objects = [
    os.path.join(CLOC, "filtering.o"),
    os.path.join(CLOC, "PerturbField.o"),
    os.path.join(CLOC, "SpinTemperatureBox.o"),
    os.path.join(CLOC, "SpinTemperatureBox_simple.o"),
    os.path.join(CLOC, "SpinTemperatureBox_ws.o"),
    os.path.join(CLOC, "IonisationBox.o"),
]
extra_link_args = ["-lcudart", "-lstdc++"]

# Set the C-code logging level.
# If DEBUG is set, we default to the highest level, but if not,
# we set it to the level just above no logging at all.
log_level = os.environ.get("LOG_LEVEL", 4 if "DEBUG" in os.environ else 1)
available_levels = [
    "NONE",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "SUPER_DEBUG",
    "ULTRA_DEBUG",
]


if isinstance(log_level, str) and log_level.upper() in available_levels:
    log_level = available_levels.index(log_level.upper())

try:
    log_level = int(log_level)
except ValueError:
    # note: for py35 support, can't use f strings.
    raise ValueError(
        "LOG_LEVEL must be specified as a positive integer, or one "
        "of {}".format(available_levels)
    )

# ==================================================
# Set compilation arguments dependent on environment
# ==================================================

extra_compile_args = ["-Wall", "--verbose", f"-DLOG_LEVEL={log_level:d}"]

if "DEBUG" in os.environ:
    extra_compile_args += ["-g", "-O0"]
else:
    extra_compile_args += ["-Ofast"]

if sys.platform == "darwin":
    extra_compile_args += ["-Xpreprocessor"]

extra_compile_args += ["-fopenmp"]

libraries = ["m", "gsl", "gslcblas", "fftw3f_omp", "fftw3f"]

# GPU fft libraries
# if True:
#     libraries += ["cufft", "cufftw"]

# stuff for gperftools
if "PROFILE" in os.environ:
    # libraries += ["profiler", "tcmalloc"] # tcmalloc causing errors
    libraries += ["profiler"]
    # we need this even if DEBUG is off
    extra_compile_args += ["-g"]

if compiler == "clang":
    libraries += ["omp"]

library_dirs = []
for k, v in os.environ.items():
    if "inc" in k.lower():
        include_dirs += [v]
    elif "lib" in k.lower():
        library_dirs += [v]

# =================================================================
# NOTES FOR DEVELOPERS:
#   The CFFI implementation works as follows:
#       - All function prototypes, global variables and type definitions *directly* used
#           in the python wrapper must be declared via ffi.cdef("""C CODE""").
#           There must be no compiler directives in this code (#include, #define, etc)
#       - All implementations of global variables and types present in the cdef() calls
#           must also be present in the second argument of set_source.
#           This is passed to the compiler.
#       - The `sources` kwarg then contains all the .c files in the library which are to be compiled

# This is the overall C code.
ffi.set_source(
    "py21cmfast.c_21cmfast",  # Name/Location of shared library module
    """
    #include "21cmFAST.h"
    """,
    sources=c_files,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_objects=extra_objects,
    extra_link_args=extra_link_args,
)

# Header files containing types, globals and function prototypes
with open(os.path.join(CLOC, "_inputparams_wrapper.h")) as f:
    ffi.cdef(f.read())
with open(os.path.join(CLOC, "_outputstructs_wrapper.h")) as f:
    ffi.cdef(f.read())
with open(os.path.join(CLOC, "_functionprototypes_wrapper.h")) as f:
    ffi.cdef(f.read())

# CFFI needs to be able to access a free function to make the __del__ method for OutputStruct fields
#  This will expose the standard free() function to the wrapper so it can be used
ffi.cdef(
    """
        void free(void *ptr);
    """
)

if __name__ == "__main__":
    ffi.compile()
