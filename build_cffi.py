"""Build the C code with CFFI."""

import os
import sys
import sysconfig
from pathlib import Path

from cffi import FFI

# Get the compiler. We support gcc and clang.
# The compiler is determnined from the environment and uses sysconfig as a fallback.
source = "environment variable 'CC'" if "CC" in os.environ else "sysconfig"
_compiler = os.environ.get("CC", sysconfig.get_config_var("CC"))
print(f"Using compiler from {source}: {_compiler}")

if "gcc" in _compiler:
    compiler = "gcc"
elif "clang" in _compiler:
    compiler = "clang"
else:
    raise ValueError(f"Compiler {_compiler} not supported for 21cmFAST")

ffi = FFI()

LOCATION = Path(__file__).resolve().parent
CLOC = LOCATION / "src" / "py21cmfast" / "src"
include_dirs = [str(CLOC)]
c_files = [str(fl.relative_to(LOCATION)) for fl in sorted(CLOC.glob("*.c"))]

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
except ValueError as e:
    # note: for py35 support, can't use f strings.
    raise ValueError(
        "LOG_LEVEL must be specified as a positive integer, or one "
        f"of {available_levels}"
    ) from e

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

# stuff for gperftools
if "PROFILE" in os.environ:
    libraries += ["profiler", "tcmalloc"]
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
)

# Header files containing types, globals and function prototypes
with (CLOC / "_inputparams_wrapper.h").open() as f:
    ffi.cdef(f.read())
with (CLOC / "_outputstructs_wrapper.h").open() as f:
    ffi.cdef(f.read())
with (CLOC / "_functionprototypes_wrapper.h").open() as f:
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
