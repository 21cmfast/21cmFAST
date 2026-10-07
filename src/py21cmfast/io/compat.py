"""Compatibility with files written by older versions of 21cmFAST.

Every change to the on-disk format of 21cmFAST outputs since v4.0 is recorded in this
module, in :data:`FORMAT_HISTORY`. Each entry of the history is a
:class:`FormatRelease` -- the version of 21cmFAST that introduced the changes -- and
the list of :class:`FormatChange` objects describing what changed: renamed, added or
removed input parameters, renamed output structs, renamed or removed fields of those
output structs, and so on. This history serves two purposes:

1. When a file written by an older version of 21cmFAST is read, the changes made
   since that version are applied, in order, to the raw contents of the file, so that
   they can be turned into objects of the current version of the code. This is done
   automatically by the readers in :mod:`py21cmfast.io.h5` (and therefore by
   ``Coeval.from_file``, ``LightCone.from_file`` etc.).
2. It is a running changelog of the I/O format, which can be printed with
   :func:`describe_format_changes` (or ``21cmfast migrate --explain`` on the command
   line).

The changes operate on simple, format-agnostic representations of the contents of a
file (:class:`RawInputs` and :class:`RawOutputStruct`), so they don't need to know
anything about HDF5.

Changes are written to be *tolerant*: each one only does something if it finds the
old format in the data it is given (e.g. a renamed parameter is only renamed if the
old name is there). This means they can also safely be applied to files written by
development versions that may have only some of the changes of a given release.

Conversions are not always lossless. Wherever an older file lacks information that
the current version of the code expects (e.g. a field of an output struct that did
not exist, or whose meaning changed and can't be converted), a
:class:`CompatibilityWarning` is emitted. Files written before v4.0 are not supported
by this module at all (see the documentation for how to read them with h5py).

When adding a new change to the format of the files, add an entry to
:data:`FORMAT_HISTORY` here, under the release in which it will appear (adding a new
:class:`FormatRelease` if required), and add a test for it in
``tests/io/test_compat.py``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import attrs
import numpy as np
from packaging.version import InvalidVersion, Version

from .. import __version__
from ..wrapper.arrays import H5Backend

logger = logging.getLogger(__name__)

#: The oldest version of 21cmFAST whose files can be read.
OLDEST_SUPPORTED_VERSION = Version("4.0.0.dev0")  # including pre-releases of v4.0


class CompatibilityWarning(UserWarning):
    """Warning for when a file from an older version can't be fully converted."""


class UnsupportedVersionError(ValueError):
    """Error raised when a file was written by an unsupported version of 21cmFAST."""


class UnconvertibleError(ValueError):
    """Error raised when an object in an old file has no equivalent in this version."""


def parse_version(version: str | Version | None) -> Version:
    """Parse a 21cmFAST version string, as written into its output files.

    Parameters
    ----------
    version
        The version string. Development versions (e.g. ``4.3.dev12+g1234abc``) are
        supported.

    Raises
    ------
    UnsupportedVersionError
        If the version is None (as for files written before v4), or older than the
        oldest supported version.
    """
    if version is None:
        raise UnsupportedVersionError(
            "The file does not record the version of 21cmFAST that wrote it, so it "
            "was most likely written by a version older than v4.0, which is not "
            "supported. The arrays in such files can still be read directly using "
            "h5py."
        )
    if isinstance(version, Version):
        return version
    if isinstance(version, bytes):
        version = version.decode()

    try:
        out = Version(str(version))
    except InvalidVersion as e:
        raise UnsupportedVersionError(
            f"Could not understand the 21cmFAST version '{version}'."
        ) from e

    if out < OLDEST_SUPPORTED_VERSION:
        raise UnsupportedVersionError(
            f"Files written by 21cmFAST v{out} are not supported. The oldest version "
            f"whose files can be read is v{OLDEST_SUPPORTED_VERSION}."
        )
    return out


CURRENT_VERSION = Version(__version__)


# ======================================================================================
# Raw representations of the contents of files.
# ======================================================================================
@attrs.define
class RawInputs:
    """The contents of a set of input parameters, as read from a file.

    Parameters
    ----------
    structs
        A dictionary mapping the snake-case name of each input struct (e.g.
        ``"astro_params"``) to a dictionary of the parameters in the file.
    random_seed
        The random seed.
    node_redshifts
        The node redshifts (or None).
    cosmo_tables
        The (raw) cosmological tables stored in the file, if any.
    version
        The version of 21cmFAST that wrote the file.
    """

    structs: dict[str, dict[str, Any]]
    random_seed: int | None = None
    node_redshifts: Any = None
    cosmo_tables: dict[str, Any] | None = None
    version: Version = attrs.field(default=CURRENT_VERSION, converter=parse_version)

    def get(self, struct: str, param: str, default: Any = None) -> Any:
        """Get a parameter of a given struct, or a default if it is not there."""
        return self.structs.get(struct, {}).get(param, default)


@attrs.define(frozen=True)
class ArrayOnDisk:
    """A pointer to an array in a file (that hasn't been read into memory)."""

    backend: H5Backend
    shape: tuple[int, ...] = attrs.field(converter=tuple)
    dtype: np.dtype = attrs.field(converter=np.dtype)


#: The source of an array in a raw output struct: either a pointer to an array in a
#: file, or the array itself.
ArraySource = ArrayOnDisk | np.ndarray


@attrs.define
class RawOutputStruct:
    """The contents of an output struct, as read from a file.

    Parameters
    ----------
    kind
        The name of the kind of output struct (e.g. ``"InitialConditions"``).
    inputs
        The raw input parameters of the struct.
    arrays
        A dictionary mapping the name of each array in the file to its source.
    primitives
        A dictionary of the non-array fields of the struct.
    redshift
        The redshift of the struct, if it has one.
    version
        The version of 21cmFAST that wrote the file.
    """

    kind: str
    inputs: RawInputs
    arrays: dict[str, ArraySource] = attrs.field(factory=dict)
    primitives: dict[str, Any] = attrs.field(factory=dict)
    redshift: float | None = None
    version: Version = attrs.field(default=CURRENT_VERSION, converter=parse_version)


def _warn(msg: str):
    warnings.warn(msg, CompatibilityWarning, stacklevel=4)


# ======================================================================================
# Kinds of change to the format.
# ======================================================================================
class FormatChange:
    """A change to the format of 21cmFAST outputs.

    Subclasses implement :meth:`upgrade_inputs` and/or :meth:`upgrade_output_struct`,
    which modify raw data in-place, from the format before the change to the format
    after the change. Both should do nothing if the data is already in the new format.
    """

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        raise NotImplementedError

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Upgrade raw input parameters (in-place)."""

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Upgrade a raw output struct (in-place)."""


@attrs.define(frozen=True)
class RenameParameter(FormatChange):
    """An input parameter was renamed (and possibly moved to another struct)."""

    struct: str
    old: str
    new: str
    new_struct: str | None = None

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        new_struct = self.new_struct or self.struct
        return f"Input parameter {self.struct}.{self.old} -> {new_struct}.{self.new}"

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Rename the parameter, if its old name is present."""
        params = inputs.structs.get(self.struct, {})
        if self.old not in params:
            return
        val = params.pop(self.old)
        inputs.structs.setdefault(self.new_struct or self.struct, {}).setdefault(
            self.new, val
        )


@attrs.define(frozen=True)
class RemoveParameter(FormatChange):
    """An input parameter was removed."""

    struct: str
    name: str
    reason: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"Input parameter {self.struct}.{self.name} removed: {self.reason}"

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Remove the parameter, if present."""
        inputs.structs.get(self.struct, {}).pop(self.name, None)


@attrs.define(frozen=True)
class AddParameter(FormatChange):
    """A new input parameter was added.

    When the parameter is not present in an older file, it is set to the value that
    reproduces the behaviour of the older version, as returned by ``old_behaviour``
    (a function of the raw inputs). If ``old_behaviour`` is None, or returns None,
    the parameter is left out, so that it takes its default value.
    """

    struct: str
    name: str
    old_behaviour: Callable[[RawInputs], Any] | None = None
    explanation: str = ""

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        out = f"New input parameter {self.struct}.{self.name}"
        if self.explanation:
            out += f": {self.explanation}"
        return out

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Set the value of the parameter to reproduce the older version's behaviour."""
        params = inputs.structs.setdefault(self.struct, {})
        if self.name in params or self.old_behaviour is None:
            return
        if (val := self.old_behaviour(inputs)) is not None:
            params[self.name] = val


@attrs.define(frozen=True)
class ConvertParameters(FormatChange):
    """A more complex change to input parameters, done by a function."""

    func: Callable[[RawInputs], None]
    explanation: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return self.explanation

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Apply the conversion function."""
        self.func(inputs)


@attrs.define(frozen=True)
class RenameOutputStruct(FormatChange):
    """An output struct was renamed."""

    old: str
    new: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"Output struct {self.old} -> {self.new}"

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Rename the kind of the struct."""
        if struct.kind == self.old:
            struct.kind = self.new


@attrs.define(frozen=True)
class UnconvertibleOutputStruct(FormatChange):
    """An output struct was removed, or changed so much that it can't be converted.

    Parameters
    ----------
    name
        The (old) name of the output struct.
    reason
        Why it can't be converted.
    successor
        The name of the output struct that replaced it, if any.
    """

    name: str
    reason: str
    successor: str | None = None

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        if self.successor is None:
            return f"Output struct {self.name} removed: {self.reason}"
        return (
            f"Output struct {self.name} -> {self.successor}, but can't be "
            f"converted: {self.reason}"
        )

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Raise an error, since the struct can't be converted."""
        if struct.kind == self.name:
            raise UnconvertibleError(
                f"{self.name} (written by 21cmFAST v{struct.version}) can't be "
                f"converted to the current version of 21cmFAST: {self.reason}"
            )


def load_array(source: ArraySource) -> np.ndarray:
    """Return the data of an array in a raw output struct."""
    return source.backend.read() if isinstance(source, ArrayOnDisk) else source


@attrs.define(frozen=True)
class RenameField(FormatChange):
    """A field (array or other attribute) of an output struct was renamed.

    The ``struct`` is the name of the struct *after* any renaming of the struct in the
    same release (renamings of structs are applied first).

    If the meaning of the field also changed, ``transform`` converts the old value
    (for arrays, the data of the array) to the new one, given the old value and the
    raw struct. It may return None, in which case the field is dropped (the
    transform itself should warn if information is lost).
    """

    struct: str
    old: str
    new: str
    transform: Callable[[Any, RawOutputStruct], Any] | None = None
    explanation: str = ""

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        out = f"{self.struct}.{self.old} -> {self.struct}.{self.new}"
        if self.explanation:
            out += f": {self.explanation}"
        return out

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Rename the field, if present."""
        if struct.kind != self.struct:
            return
        for dct in (struct.arrays, struct.primitives):
            if self.old not in dct or self.new in dct:
                continue

            val = dct.pop(self.old)
            if self.transform is not None:
                if dct is struct.arrays:
                    val = load_array(val)
                val = self.transform(val, struct)
            if val is not None:
                dct[self.new] = val


@attrs.define(frozen=True)
class ConvertOutputStruct(FormatChange):
    """A more complex change to an output struct, done in-place by a function."""

    struct: str
    func: Callable[[RawOutputStruct], None]
    explanation: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"{self.struct}: {self.explanation}"

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Apply the conversion function."""
        if struct.kind == self.struct:
            self.func(struct)


@attrs.define(frozen=True)
class RemoveField(FormatChange):
    """A field of an output struct was removed or can't be converted.

    If ``warn`` is True, a :class:`CompatibilityWarning` is emitted when the field is
    found in a file, since the information in it is lost.
    """

    struct: str
    name: str
    reason: str
    warn: bool = True

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"{self.struct}.{self.name} removed: {self.reason}"

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Remove the field, if present."""
        if struct.kind != self.struct:
            return
        for dct in (struct.arrays, struct.primitives):
            if self.name in dct:
                del dct[self.name]
                if self.warn:
                    _warn(
                        f"The field '{self.name}' of {self.struct} in this file "
                        f"(written by 21cmFAST v{struct.version}) can't be used by "
                        f"this version of 21cmFAST, and is ignored: {self.reason}"
                    )


@attrs.define(frozen=True)
class AddField(FormatChange):
    """A new non-array field was added to an output struct.

    When the field is missing in an older file, it is set to the value given by
    ``value`` (a function of the raw struct). If that is None (or returns None), the
    field is left unset, so that it takes its default value.
    """

    struct: str
    name: str
    value: Callable[[RawOutputStruct], Any] | None = None
    explanation: str = ""

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        out = f"New field {self.struct}.{self.name}"
        if self.explanation:
            out += f": {self.explanation}"
        return out

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Set the value of the field, if it is not present."""
        if struct.kind != self.struct or self.name in struct.primitives:
            return
        if self.value is not None and (val := self.value(struct)) is not None:
            struct.primitives[self.name] = val


@attrs.define(frozen=True)
class FormatRelease:
    """The changes to the file format made in a release of 21cmFAST.

    Parameters
    ----------
    version
        The release in which the changes were made.
    changes
        The changes, in the order in which they should be applied.
    """

    version: Version = attrs.field(converter=Version)
    changes: tuple[FormatChange, ...] = attrs.field(converter=tuple)

    def applies_to(self, version: Version) -> bool:
        """Whether the changes of this release need to be applied to a file version.

        Files written by development versions leading up to this release (e.g.
        ``4.3.dev12``) are considered to need the changes, since they may have been
        written before some of them were made.
        """
        return version < self.version


# ======================================================================================
# Helpers for the specific changes in the history.
# ======================================================================================
_LAGRANGIAN_SOURCE_MODELS = ("L-INTEGRAL", "DEXM-ESF", "CHMF-SAMPLER")
_DISCRETE_HALO_SOURCE_MODELS = ("DEXM-ESF", "CHMF-SAMPLER")


def _source_model(inputs: RawInputs) -> str:
    return inputs.get("matter_options", "SOURCE_MODEL", "")


def _rhocrit_omb(inputs: RawInputs) -> float:
    """Compute the critical density times Omega_b (Msun/Mpc^3), as in the C code."""
    hubble = inputs.get("cosmo_params", "hlittle") * 3.2407e-18  # 1/s
    grav = 6.6743e-8  # cgs
    cm_per_mpc = 3.08567758e24
    msun = 1.989e33  # g
    rhocrit = 3 * hubble**2 / (8 * np.pi * grav) * cm_per_mpc**3 / msun
    return rhocrit * inputs.get("cosmo_params", "OMb")


def _nion_prefactor(inputs: RawInputs, mcg: bool = False) -> float | None:
    """Compute the prefactor included in the n_ion fields of IonizedBox since v4.3.

    Returns None if the prefactor can't be determined from the inputs alone (which
    is the case for f-photoncons, where the escape fraction is fit as a function of
    redshift).
    """
    if inputs.get("astro_options", "PHOTON_CONS_TYPE") == "f-photoncons":
        return None

    p = inputs.structs["astro_params"]
    if mcg:
        return 10 ** p["F_STAR7_MCG"] * 10 ** p["F_ESC7_MCG"] * p["POP3_ION"]
    if _source_model(inputs) == "CONST-ION-EFF":
        return p["HII_EFF_FACTOR"]
    return 10 ** p["F_STAR10_ACG"] * 10 ** p["F_ESC10_ACG"] * p["POP2_ION"]


def _v41_min_xe(inputs: RawInputs) -> float | None:
    # The threshold was only applied for single-cell (global) simulations, so that a
    # threshold of zero reproduces older versions. In all other cases, the default
    # is fine.
    return 0.0 if inputs.get("simulation_options", "HII_DIM") == 1 else None


def _v41_q_hi(struct: RawOutputStruct) -> None:
    if struct.inputs.get("simulation_options", "HII_DIM") == 1:
        _warn(
            f"This TsBox (written by 21cmFAST v{struct.version}) has no Q_HI, which "
            "is required by single-cell (global) simulations. It is set to 1."
        )


def _v42_recomb_model(inputs: RawInputs) -> None:
    opts = inputs.structs.setdefault("astro_options", {})
    if "INHOMO_RECO" in opts:
        inhomo = bool(opts.pop("INHOMO_RECO"))
        opts.setdefault("RECOMB_MODEL", "inhomogeneous" if inhomo else "none")


def _v42_sigma_sfr_index(inputs: RawInputs) -> None:
    # SIGMA_SFR_INDEX was given in base e, and is now in dex.
    params = inputs.structs.get("astro_params", {})
    if "SIGMA_SFR_INDEX" in params:
        params["SIGMA_SFR_INDEX"] = params["SIGMA_SFR_INDEX"] / np.log(10)


def _v43_v_cb_model(inputs: RawInputs) -> None:
    matter = inputs.structs.setdefault("matter_options", {})
    astro_opts = inputs.structs.setdefault("astro_options", {})
    astro_params = inputs.structs.setdefault("astro_params", {})

    use_rel_vel = matter.pop("USE_RELATIVE_VELOCITIES", None)
    fix_vcb = astro_opts.pop("FIX_VCB_AVG", None)
    fixed_vavg = astro_params.pop("FIXED_VAVG", None)

    if "V_CB_MODEL" in matter or use_rel_vel is None:
        return

    if use_rel_vel:
        matter["V_CB_MODEL"] = "AVG-DEBUG" if fix_vcb else "FLUCTS"
    elif fix_vcb and astro_opts.get("USE_MCGS", False):
        # The fixed average velocity was only used by mini-halos.
        matter["V_CB_MODEL"] = "AVG-DEBUG"
    else:
        matter["V_CB_MODEL"] = "NONE"

    if matter["V_CB_MODEL"] == "AVG-DEBUG" and fixed_vavg is not None:
        astro_params.setdefault("V_CB_AVG_DEBUG", fixed_vavg)


def _v43_use_metallicity(inputs: RawInputs) -> bool:
    # Before v4.3, the metallicity-dependence of the X-ray luminosity was switched on
    # by USE_UPPER_STELLAR_TURNOVER (whose default was True), and only for discrete
    # halos.
    upper = bool(inputs.get("astro_options", "USE_UPPER_STELLAR_TURNOVER", True))
    return upper and _source_model(inputs) in _DISCRETE_HALO_SOURCE_MODELS


def _v43_photoheating_feedback(inputs: RawInputs) -> bool:
    return bool(inputs.get("astro_options", "USE_MCGS", False))


def _drop_incomplete_cosmo_tables(inputs: RawInputs) -> None:
    required = {"ps_norm", "USE_SIGMA_8", "V_CB_AVG"}
    if inputs.cosmo_tables is not None and not required.issubset(inputs.cosmo_tables):
        inputs.cosmo_tables = None


def _v43_halobox_n_ion(struct: RawOutputStruct) -> None:
    # Applied to the struct while it is still called HaloBox.
    if "n_ion" in struct.arrays:
        struct.arrays["n_ion"] = load_array(struct.arrays["n_ion"]) / _rhocrit_omb(
            struct.inputs
        )
    # This was -inf (log10(0)) when there were no mini-halos, and is now 0.
    if np.isneginf(struct.primitives.get("log10_Mcrit_MCG_ave", 0.0)):
        struct.primitives["log10_Mcrit_MCG_ave"] = 0.0


def _v43_divide_by_rhocrit_omb(val, struct: RawOutputStruct):
    return val / _rhocrit_omb(struct.inputs)


def _v43_nion_unconditional(mcg: bool):
    def transform(val, struct: RawOutputStruct):
        if _source_model(struct.inputs) in _LAGRANGIAN_SOURCE_MODELS:
            # Overwritten by the mean of the (halo) source field, which is n_ion of
            # the HaloBox (for ACGs, MCGs are included in the same field).
            return val if mcg else val / _rhocrit_omb(struct.inputs)

        prefactor = _nion_prefactor(struct.inputs, mcg=mcg)
        if prefactor is None:
            _warn(
                "The mean n_ion in this IonizedBox (written by 21cmFAST "
                f"v{struct.version}) can't be converted with f-photoncons, and is "
                "unavailable."
            )
            return None
        return val * prefactor

    return transform


def _v43_nion_conditional(mcg: bool):
    def transform(val, struct: RawOutputStruct):
        if _source_model(struct.inputs) in _LAGRANGIAN_SOURCE_MODELS:
            # This is not used (or saved) for Lagrangian source models any more.
            return None

        prefactor = _nion_prefactor(struct.inputs, mcg=mcg)
        if prefactor is None:
            _warn(
                "The filtered n_ion in this IonizedBox (written by 21cmFAST "
                f"v{struct.version}) can't be converted with f-photoncons, and is "
                "unavailable."
            )
            return None
        return val * prefactor

    return transform


_NION_PREFACTOR_EXPLANATION = (
    "now includes the prefactor f_star * f_esc * N_ion (or HII_EFF_FACTOR for "
    "CONST-ION-EFF), which is applied to older files (not possible for f-photoncons)"
)

# ======================================================================================
# The history of the file format.
# ======================================================================================
#: The history of changes to the format of 21cmFAST's output files, since v4.0.
FORMAT_HISTORY: tuple[FormatRelease, ...] = (
    FormatRelease(
        "4.1.0",
        [
            AddParameter(
                "simulation_options",
                "MIN_XE_FOR_FCOLL_IN_TAUX",
                _v41_min_xe,
                "the mean x_e above which n_ion is set to zero in the X-ray optical "
                "depth (only used when HII_DIM=1). Set to zero for older files with "
                "HII_DIM=1, reproducing their behaviour.",
            ),
            AddField(
                "TsBox",
                "Q_HI",
                _v41_q_hi,
                "it is left at its default value (1) for older files. It is only "
                "used for single-cell (global) simulations.",
            ),
        ],
    ),
    FormatRelease(
        "4.2.0",
        [
            ConvertParameters(
                _v42_recomb_model,
                "astro_options.INHOMO_RECO replaced by astro_options.RECOMB_MODEL "
                "(True -> 'inhomogeneous', False -> 'none')",
            ),
            ConvertParameters(
                _v42_sigma_sfr_index,
                "astro_params.SIGMA_SFR_INDEX is now in dex rather than base e, so "
                "values in older files are divided by ln(10)",
            ),
            AddParameter(
                "astro_options",
                "LYA_MULTIPLE_SCATTERING",
                explanation="its default (False) reproduces older versions.",
            ),
            AddParameter(
                "astro_options",
                "USE_ADIABATIC_FLUCTUATIONS",
                explanation="its default (True) reproduces older versions.",
            ),
        ],
    ),
    FormatRelease(
        "4.3.0",
        [
            # ---- Input parameters ----
            RenameParameter(
                "simulation_options",
                "MIN_XE_FOR_FCOLL_IN_TAUX",
                "MIN_XE_FOR_NION_IN_TAUX",
            ),
            RenameParameter("astro_options", "USE_MINI_HALOS", "USE_MCGS"),
            RenameParameter(
                "astro_options", "INTEGRATION_METHOD_ATOMIC", "INTEGRATION_METHOD_ACGS"
            ),
            RenameParameter(
                "astro_options", "INTEGRATION_METHOD_MINI", "INTEGRATION_METHOD_MCGS"
            ),
            RenameParameter("astro_params", "F_STAR10", "F_STAR10_ACG"),
            RenameParameter("astro_params", "F_STAR7_MINI", "F_STAR7_MCG"),
            RenameParameter("astro_params", "ALPHA_STAR", "ALPHA_STAR_ACG"),
            RenameParameter("astro_params", "ALPHA_STAR_MINI", "ALPHA_STAR_MCG"),
            RenameParameter("astro_params", "F_ESC10", "F_ESC10_ACG"),
            RenameParameter("astro_params", "F_ESC7_MINI", "F_ESC7_MCG"),
            RenameParameter("astro_params", "L_X", "LX_OVER_SFR_ACG"),
            RenameParameter("astro_params", "L_X_MINI", "LX_OVER_SFR_MCG"),
            RenameParameter("astro_params", "M_TURN", "M_TURN_STELLAR_FEEDBACK"),
            ConvertParameters(
                _v43_v_cb_model,
                "matter_options.USE_RELATIVE_VELOCITIES, astro_options.FIX_VCB_AVG "
                "and astro_params.FIXED_VAVG replaced by matter_options.V_CB_MODEL "
                "and astro_params.V_CB_AVG_DEBUG",
            ),
            AddParameter(
                "astro_options",
                "USE_METALLICITY",
                _v43_use_metallicity,
                "previously, the metallicity-dependence of the X-ray luminosity of "
                "discrete halos was switched on by USE_UPPER_STELLAR_TURNOVER, so it "
                "is set to that value for older files with discrete halos (and False "
                "otherwise).",
            ),
            AddParameter(
                "astro_options",
                "USE_REIONIZATION_PHOTOHEATING_FEEDBACK",
                _v43_photoheating_feedback,
                "previously this was switched on by USE_MINI_HALOS, so it is set to "
                "that value for older files.",
            ),
            ConvertParameters(
                _drop_incomplete_cosmo_tables,
                "cosmo_tables gained V_CB_AVG (and ps_norm and USE_SIGMA_8 in v4.1). "
                "Incomplete tables in older files are discarded (they are recomputed "
                "when needed).",
            ),
            # ---- Output structs ----
            ConvertOutputStruct(
                "HaloBox",
                _v43_halobox_n_ion,
                "n_ion is now divided by rho_crit * Omega_b, and log10_Mcrit_MCG_ave "
                "is 0 rather than -inf without mini-halos",
            ),
            RenameOutputStruct("HaloBox", "EmissivityFields"),
            RenameField("EmissivityFields", "count", "halo_number"),
            RenameField("EmissivityFields", "halo_mass", "halo_mass_density"),
            RenameField("EmissivityFields", "halo_stars", "stellar_mass_density_acg"),
            RenameField(
                "EmissivityFields", "halo_stars_mini", "stellar_mass_density_mcg"
            ),
            RenameField("EmissivityFields", "halo_sfr", "sfrd_acg"),
            RenameField("EmissivityFields", "halo_sfr_mini", "sfrd_mcg"),
            RenameField("EmissivityFields", "halo_xray", "xray_emissivity"),
            RenameField("EmissivityFields", "whalo_sfr", "fesc_weighted_sfrd"),
            RenameField(
                "EmissivityFields", "log10_Mcrit_ACG_ave", "log10_mturn_acg_ave"
            ),
            RenameField(
                "EmissivityFields", "log10_Mcrit_MCG_ave", "log10_mturn_mcg_ave"
            ),
            RenameField("PerturbedHaloCatalog", "sfr", "sfr_acg"),
            RenameField("PerturbedHaloCatalog", "stellar_masses", "stellar_masses_acg"),
            RenameField(
                "PerturbedHaloCatalog",
                "ion_emissivity",
                "n_ion",
                _v43_divide_by_rhocrit_omb,
                "now divided by rho_crit * Omega_b",
            ),
            RenameField("PerturbedHaloCatalog", "xray_emissivity", "xray_luminosity"),
            RenameField("PerturbedHaloCatalog", "fesc_sfr", "fesc_weighted_sfr"),
            RenameField("PerturbedHaloCatalog", "stellar_mini", "stellar_masses_mcg"),
            RenameField("PerturbedHaloCatalog", "sfr_mini", "sfr_mcg"),
            RenameField(
                "IonizedBox",
                "mean_f_coll",
                "nion_unconditional_acg",
                _v43_nion_unconditional(mcg=False),
                _NION_PREFACTOR_EXPLANATION,
            ),
            RenameField(
                "IonizedBox",
                "mean_f_coll_MINI",
                "nion_unconditional_mcg",
                _v43_nion_unconditional(mcg=True),
                _NION_PREFACTOR_EXPLANATION,
            ),
            RenameField("IonizedBox", "log10_Mturnover_ave", "log10_mturn_ave_acg"),
            RenameField(
                "IonizedBox", "log10_Mturnover_MINI_ave", "log10_mturn_ave_mcg"
            ),
            RenameField(
                "IonizedBox",
                "unnormalised_nion",
                "nion_conditional_filtered_acg",
                _v43_nion_conditional(mcg=False),
                _NION_PREFACTOR_EXPLANATION,
            ),
            RenameField(
                "IonizedBox",
                "unnormalised_nion_mini",
                "nion_conditional_filtered_mcg",
                _v43_nion_conditional(mcg=True),
                _NION_PREFACTOR_EXPLANATION,
            ),
            UnconvertibleOutputStruct(
                "XraySourceBox",
                "it was renamed to RadiationFields, but the calculation of the "
                "radiation fields was then refactored. RadiationFields now holds "
                "the computed X-ray, Lyman-alpha and Lyman-Werner rates and fluxes, "
                "which older files don't contain, while the filtered source fields "
                "that the XraySourceBox held (as 4D arrays over all shells) are now "
                "per-shell scratch arrays of RadiationFieldsSetup. Since these are "
                "intermediate products, they are simply recomputed when needed.",
                successor="RadiationFields",
            ),
        ],
    ),
)


# ======================================================================================
# Public functions for upgrading data.
# ======================================================================================
def releases_since(version: str | Version) -> list[FormatRelease]:
    """Return the format releases whose changes apply to files of a given version."""
    version = parse_version(version)
    return [rel for rel in FORMAT_HISTORY if rel.applies_to(version)]


def needs_upgrade(version: str | Version) -> bool:
    """Whether a file written by a given version needs to be upgraded."""
    return bool(releases_since(version))


def is_legacy(version: str | Version) -> bool:
    """Whether files of a given version must be converted to the current format.

    This is the case for files written by older versions than the current one, that
    are affected by any changes in :data:`FORMAT_HISTORY`.
    """
    version = parse_version(version)
    return version < CURRENT_VERSION and needs_upgrade(version)


def upgrade_inputs(inputs: RawInputs) -> RawInputs:
    """Upgrade raw input parameters from an older version to the current format.

    The input is modified in-place, and also returned.
    """
    for release in releases_since(inputs.version):
        for change in release.changes:
            change.upgrade_inputs(inputs)
    return inputs


def upgrade_output_struct(struct: RawOutputStruct) -> RawOutputStruct:
    """Upgrade a raw output struct from an older version to the current format.

    Its inputs are upgraded as well. The input is modified in-place, and also
    returned.

    Raises
    ------
    UnconvertibleError
        If the output struct has no equivalent in the current version.
    """
    upgrade_inputs(struct.inputs)
    for release in releases_since(struct.version):
        for change in release.changes:
            change.upgrade_output_struct(struct)
    return struct


def historical_struct_names(name: str) -> set[str]:
    """Return all names an output struct has had in the history of the file format.

    Parameters
    ----------
    name
        The current name of the output struct.
    """
    names = {name}
    for release in reversed(FORMAT_HISTORY):
        for change in release.changes:
            if isinstance(change, RenameOutputStruct) and change.new in names:
                names.add(change.old)
            elif (
                isinstance(change, UnconvertibleOutputStruct)
                and change.successor in names
            ):
                names.add(change.name)
    return names


def current_struct_name(name: str) -> str:
    """Return the current name of an output struct, given a name it had in the past."""
    for release in FORMAT_HISTORY:
        for change in release.changes:
            if isinstance(change, RenameOutputStruct) and change.old == name:
                name = change.new
            elif (
                isinstance(change, UnconvertibleOutputStruct)
                and change.name == name
                and change.successor is not None
            ):
                name = change.successor
    return name


#: Output structs whose fields are never boxes (so can't be in lightcones etc.)
_NON_BOX_STRUCTS = ("HaloCatalog", "PerturbedHaloCatalog")


def upgrade_box_quantities(
    quantities: dict[str, np.ndarray], inputs: RawInputs
) -> dict[str, np.ndarray]:
    """Upgrade quantities named after fields of output structs, from an older version.

    Lightcones, and the global quantities of lightcones and global-evolution
    simulations, store quantities named after the arrays of the output structs that
    they come from (e.g. ``brightness_temp`` or ``halo_sfr``). This function renames
    (and converts) them in the same way as the arrays of the output structs.

    Parameters
    ----------
    quantities
        A dictionary of the quantities, keyed by their (old) names. It is modified
        in-place, and also returned.
    inputs
        The raw input parameters of the simulation, *already upgraded* to the current
        version with :func:`upgrade_inputs`. Its ``version`` should be the version
        that wrote the quantities.
    """
    field_changes = (RenameField, RemoveField, ConvertOutputStruct)
    for release in releases_since(inputs.version):
        for change in release.changes:
            if (
                isinstance(change, field_changes)
                and change.struct not in _NON_BOX_STRUCTS
            ):
                # The quantities are the arrays of a (fake) struct of the right kind.
                struct = RawOutputStruct(
                    kind=change.struct,
                    inputs=inputs,
                    arrays=quantities,
                    version=inputs.version,
                )
                change.upgrade_output_struct(struct)
    return quantities


def historical_field_names(struct: str, name: str) -> set[str]:
    """Return all names a field of an output struct has had in the format history.

    Parameters
    ----------
    struct
        The current name of the output struct.
    name
        The current name of the field.
    """
    structs = historical_struct_names(struct)
    names = {name}
    for release in reversed(FORMAT_HISTORY):
        for change in release.changes:
            if (
                isinstance(change, RenameField)
                and change.struct in structs
                and change.new in names
            ):
                names.add(change.old)
    return names


def current_field_name(old_name: str, structs: Iterable[str] | None = None) -> str:
    """Return the current name of a field, given an old name.

    Only renames of fields of the given structs are considered (or all structs, if
    None). If the old name was renamed differently in different structs, it is
    returned unchanged.
    """
    name = old_name
    for release in FORMAT_HISTORY:
        candidates = {
            change.new
            for change in release.changes
            if isinstance(change, RenameField)
            and change.old == name
            and (structs is None or change.struct in structs)
        }
        if len(candidates) == 1:
            name = candidates.pop()
    return name


def describe_format_changes(since: str | Version | None = None) -> str:
    """Return a human-readable changelog of the file format.

    Parameters
    ----------
    since
        Only describe the changes affecting files written by this version. By default,
        all changes since v4.0 are described.
    """
    releases = FORMAT_HISTORY if since is None else releases_since(since)
    lines = []
    for release in releases:
        lines.append(f"v{release.version}:")
        lines.extend(f"  - {change.description}" for change in release.changes)
    return "\n".join(lines)
