"""Exceptions raised when running C code."""

import logging

logger = logging.getLogger(__name__)


class ParameterError(RuntimeError):
    """An exception representing a bad choice of parameters."""

    default_message = "21cmFAST does not support this combination of parameters."

    def __init__(self, msg=None):
        super().__init__(msg or self.default_message)


class FatalCError(Exception):
    """An exception representing something going wrong in C."""

    default_message = "21cmFAST is exiting."

    def __init__(self, msg=None):
        super().__init__(msg or self.default_message)


class FileIOError(FatalCError):
    """An exception when an error occurs with file I/O."""

    default_message = "Expected file could not be found! (check the LOG for more info)"


class GSLError(ParameterError):
    """An exception when a GSL routine encounters an error."""

    default_message = "A GSL routine has errored! (check the LOG for more info)"


class ArgumentValueError(FatalCError):
    """An exception when a function takes an unexpected input."""

    default_message = "An incorrect argument has been defined or passed! (check the LOG for more info)"


class PhotonConsError(ParameterError):
    """An exception when the photon non-conservation correction routine errors."""

    default_message = "An error has occured with the Photon non-conservation correction! (check the LOG for more info)"


class TableGenerationError(ParameterError):
    """An exception when an issue arises populating one of the interpolation tables."""

    default_message = """An error has occured when generating an interpolation table!
                This has likely occured due to the choice of input AstroParams (check the LOG for more info)"""


class TableEvaluationError(ParameterError):
    """An exception when an issue arises populating one of the interpolation tables."""

    default_message = """An error has occured when evaluating an interpolation table!
                This can sometimes occur due to small boxes (either small DIM/HII_DIM or BOX_LEN) (check the LOG for more info)"""


class InfinityorNaNError(ParameterError):
    """An exception when an infinity or NaN is encountered in a calculated quantity."""

    default_message = """Something has returned an infinity or a NaN! This could be due to an issue with an
                input parameter choice (check the LOG for more info)"""


class MassDepZetaError(ParameterError):
    """An exception when determining the bisection for stellar mass/escape fraction."""

    default_message = """There is an issue with the choice of parameters under MASS_DEPENDENT_ZETA. Could be an issue with
                any of the chosen F_STAR10, ALPHA_STAR, F_ESC10 or ALPHA_ESC."""


class MemoryAllocError(FatalCError):
    """An exception when unable to allocated memory."""

    default_message = """An error has occured while attempting to allocate memory! (check the LOG for more info)"""


class CUDAError(FatalCError):
    """An exception when an error occurs with CUDA."""

    default_message = """A CUDA error has occured! (check the LOG for more info)"""


SUCCESS = 0
IOERROR = 1
GSLERROR = 2
VALUEERROR = 3
PHOTONCONSERROR = 4
TABLEGENERATIONERROR = 5
TABLEEVALUATIONERROR = 6
INFINITYORNANERROR = 7
MASSDEPZETAERROR = 8
MEMORYALLOCERROR = 9
CUDAERROR = 10


def _process_exitcode(exitcode, fnc, args):
    """Determine what happens for different values of the (integer) exit code from a C function."""
    if exitcode != SUCCESS:
        logger.error(f"In function: {fnc.__name__}.  Arguments: {args}")

        if exitcode:
            try:
                raise {
                    IOERROR: FileIOError,
                    GSLERROR: GSLError,
                    VALUEERROR: ArgumentValueError,
                    PHOTONCONSERROR: PhotonConsError,
                    TABLEGENERATIONERROR: TableGenerationError,
                    TABLEEVALUATIONERROR: TableEvaluationError,
                    INFINITYORNANERROR: InfinityorNaNError,
                    MASSDEPZETAERROR: MassDepZetaError,
                    MEMORYALLOCERROR: MemoryAllocError,
                    CUDAERROR: CUDAError,
                }[exitcode]
            except KeyError as e:  # pragma: no cover
                raise FatalCError(
                    "Unknown error in C. Please report this error!"
                ) from e  # Unknown C code
