"""Functions for setting up and configuring inputs to driver functions."""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Sequence
from typing import Any, get_args

from .._cfg import config
from ..input_serialization import convert_inputs_to_dict
from ..io import h5
from ..io.caching import OutputCache
from ..utils import recursive_difference
from ..wrapper.cfuncs import broadcast_input_struct, construct_fftw_wisdoms
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import OutputStruct, OutputStructZ, _HashType
from ..wrapper.photoncons import _photoncons_state

logger = logging.getLogger(__name__)


def check_redshift_consistency(
    redshift: float, output_structs: list[OutputStruct], funcname: str = "unknown"
) -> None:
    """
    Check if all given :class:`OutputStruct` objects have the same redshift.

    This function iterates over a list of OutputStruct objects and verifies that all
    the redshifts are consistent with the provided redshift value. If any inconsistency
    is found, a ValueError is raised.

    Parameters
    ----------
    redshift
        The reference redshift value to compare with the redshifts of the OutputStruct
        objects.
    output_structs
        A list of :class:`OutputStruct` objects to check for redshift consistency.
    funcname (str, optional)
        The name of the function or method where this check is being performed. Used
        in the error raised. Default is "unknown".

    Raises
    ------
    ValueError
        If any :class:`OutputStruct` object in the list has a redshift different from
        the provided redshift value.
    """
    for struct in output_structs:
        if struct is not None and struct.redshift != redshift:
            raise ValueError(
                f"Incompatible redshifts with inputs and {struct.__class__.__name__} in"
                f" {funcname}: {redshift} != {struct.redshift}"
            )


def _get_incompatible_params(
    inputs1: InputParameters, inputs2: InputParameters
) -> dict[str, Any]:
    """Return a dict of parameters that differ between two InputParameters objects."""
    d1 = convert_inputs_to_dict(inputs1, only_structs=False, camel=False)
    d2 = convert_inputs_to_dict(inputs2, only_structs=False, camel=False)
    return recursive_difference(d1, d2)


def _get_incompatible_param_diffstring(
    inputs1: InputParameters, inputs2: InputParameters
) -> str:
    incompatible_params = _get_incompatible_params(inputs1, inputs2)
    rev = _get_incompatible_params(inputs2, inputs1)

    return "".join(
        (
            f"{name}:\n"
            + "\n".join(
                f"  {key}:\n    {v1:>12}\n    {rev[name][key]:>12}"
                for key, v1 in val.items()
            )
            if isinstance(val, dict)
            else f"{name}:\n  {val}\n  {rev[name]}"
        )
        for name, val in incompatible_params.items()
    )


def check_output_consistency(outputs: dict[str, OutputStruct]) -> None:
    """Ensure all OutputStruct objects have consistent InputParameters.

    This function compares each given OutputStruct with a reference element, and ensures
    that each is compatible. Recall that two :class:`InputParameters` can be compatible
    even if they differ, as long as they agree on the input components to which each
    of the OutputStructs are dependent.

    Raises
    ------
    ValueError
        If any of the OutputStructs are not compatible.
    """
    outputs = {n: output for n, output in outputs.items() if output is not None}

    if len(outputs) < 2:
        return

    o0 = next(iter(outputs.values()))
    n0 = next(iter(outputs.keys()))

    for name, output in outputs.items():
        if not output._inputs_compatible_with(o0):
            diff = _get_incompatible_param_diffstring(output.inputs, o0.inputs)
            raise ValueError(
                f"InputParameters in {name} do not match those in {n0}. Got:\n\n{diff}"
            )


def check_consistency_of_outputs_with_inputs(
    inputs: InputParameters, outputs: Sequence[OutputStruct]
):
    """Check that all structs in `outputs` are compatible with the `inputs`.

    See Also
    --------
    :func:`check_output_consistency`
        Similar function that checks consistency between several outputs.
    """
    for output in outputs:
        if not output._inputs_compatible_with(inputs):
            diff = _get_incompatible_param_diffstring(output.inputs, inputs)
            raise ValueError(
                f"InputParameters in {output.__class__.__name__} do not match those in "
                f"the provided InputParameters. Got:\n\n{diff}"
            )


class _OutputStructComputationInspect:
    """A class that does introspection on a single-field computation function.

    This class implements methods for inspecting the arguments of a single-field
    computation function (e.g. :func:`compute_initial_conditions`) and doing validation,
    cache-checking and other quality-of-life improvements.

    It is an internal toolset, not meant to be used by users directly.
    """

    def __init__(self, _func: callable):
        self._func = _func
        self._signature = inspect.signature(_func)
        self._kls = self._signature.return_annotation

        if not issubclass(self._signature.return_annotation, OutputStruct):
            raise TypeError(
                f"{_func.__name__} must return an instance of OutputStruct (and be annotated as such)."
            )

    @staticmethod
    def _get_all_output_struct_inputs(
        kwargs, recurse: bool = False
    ) -> dict[str, OutputStruct]:
        """Return all the arguments that are OutputStructs.

        If recurse is True, also add all OutputStructs that are part of iterables.
        """
        d = {k: v for k, v in kwargs.items() if isinstance(v, OutputStruct)}

        if recurse:
            for k, v in kwargs.items():
                if hasattr(v, "__len__") and isinstance(v[0], OutputStruct):
                    d |= {f"{k}_{i}": vv for i, vv in enumerate(v)}

        return d

    @staticmethod
    def _get_inputs(kwargs: dict[str, Any]) -> InputParameters:
        """Return the most detailed input parameters available.

        For a given set of parameters to a single-field function, find the "inputs"
        that should be used for instantiating the OutputStruct that the function will
        return.

        If the parameter "inputs" is given directly, just return that. If not, the
        inputs must be determined from given OutputStruct parameters that are
        dependencies of the desired OutputStruct. In this case, we can run into the
        situation that different dependent OutputStruct's have different inputs. Even
        though all must be compatible with each other, more basic OutputStructs (like
        InitialConditions) might not have the same zgrid as the PerturbedField (for
        example) and this is fine. So, here we return the inputs of the "most advanced"
        OutputStruct that is given.
        """
        inputs = kwargs.get("inputs")
        if inputs is not None:
            return inputs

        outputs = _OutputStructComputationInspect._get_all_output_struct_inputs(
            kwargs, recurse=True
        )

        minreq = _HashType(0)
        for output in outputs.values():
            if output._compat_hash.value >= minreq.value:
                inputs = output.inputs
                minreq = output._compat_hash

        if inputs is None:
            raise ValueError(
                "No parameter 'inputs' given, and no dependent OutputStruct found!"
            )

        return inputs

    @staticmethod
    def check_consistency(kwargs: dict[str, Any], outputs: dict[str, OutputStruct]):
        """Check consistency of input parameters amongst output struct inputs."""
        check_output_consistency(outputs)
        given_inputs = kwargs.get("inputs")
        if given_inputs is not None:
            check_consistency_of_outputs_with_inputs(given_inputs, outputs.values())

    def _make_wisdoms(self, use_fftw_wisdom: bool):
        construct_fftw_wisdoms(use_fftw_wisdom=use_fftw_wisdom)

    def _broadcast_inputs(self, inputs: InputParameters):
        broadcast_input_struct(inputs=inputs)

    def check_output_struct_types(self, outputs: dict[str, OutputStruct]):
        """Check given OutputStruct parameters.

        This method checks each OutputStruct given to the compuation function to ensure
        that it is of the correct type (i.e. `InitialConditions` when requested). The
        types must be specified as standard Python types (so this is essentially just
        automated run-time type-checking). It only checks OutputStructs, and allows for
        optional values.
        """
        for name, param in self._signature.parameters.items():
            val = outputs.get(name)
            tp = param.annotation
            try:
                issub = issubclass(tp, OutputStruct)
            except TypeError:
                # The parameter type is not a subclass of OutputStruct, ignore it.
                issub = False
            if issub and not isinstance(val, tp):
                raise TypeError(
                    f"{name} should be of type {param.annotation.__name__}, got {type(val)}"
                )

            if potential_types := get_args(tp):
                if type(None) in potential_types and val is None:
                    continue
                kls = tuple(
                    kls for kls in potential_types if issubclass(kls, OutputStruct)
                )
                if not kls:
                    # This is not an OutputStruct kind of parameter, ignore.
                    continue
                elif len(kls) > 1:
                    raise TypeError(
                        f"{name} parameter has a badly defined type in the signature. Please report this on Github."
                    )
                else:
                    if type(None) not in potential_types:
                        # This is supposed to be a list/sequence of OutputStruct,
                        # but this is hard to check because these values are already
                        # stripped out of the `outputs` variable that is passed here.
                        # So for now, just ignore it.
                        continue

                    kls = kls[0]

                    if not isinstance(val, kls):
                        raise TypeError(
                            f"{name} should be of type {kls.__name__}, got {type(val)}"
                        )

    def _get_current_redshift(
        self, outputs: dict[str, OutputStructZ], kwargs: dict[str, Any]
    ) -> float | None:
        """Get the current redshift of evolution from the given parameters.

        If redshift is given directly, return that. Otherwise, return a redshift from
        any given OutputStruct whose name doesn't start with "previous" or "descendant".
        We can return the first such redshift we find, because another method will check
        that all redshifts are the same.
        """
        redshift = kwargs.get("redshift")
        if redshift is None and (
            current_outputs := [
                v
                for k, v in outputs.items()
                if not k.startswith("previous_") and not k.startswith("descendant_")
            ]
        ):
            redshift = current_outputs[0].redshift

        return redshift

    def ensure_redshift_consistency(
        self, current_redshift: float, outputs: dict[str, OutputStructZ]
    ):
        """Ensure that each OutputStruct has the same redshift, if it exists.

        Checks all given OutputStruct objects that have redshifts to check if their
        redshifts are the same. It does this in three groups: previous, descendant and
        current-redshift boxes.
        """
        if not outputs:
            return

        if current_outputs := [
            v
            for k, v in outputs.items()
            if not k.startswith("previous_") and not k.startswith("descendant_")
        ]:
            check_redshift_consistency(
                current_redshift, current_outputs, funcname=self._func.__name__
            )

        inputs = next(iter(outputs.values())).inputs
        if inputs.node_redshifts is not None:
            previous_outputs = [
                v for k, v in outputs.items() if k.startswith("previous_")
            ]
            previous_z = [z for z in inputs.node_redshifts if z > current_redshift]
            if previous_outputs and previous_z:
                previous_z = previous_z[-1]
                check_redshift_consistency(
                    previous_z,
                    previous_outputs,
                    funcname=f"{self._func.__name__} (previous z)",
                )

            descendant_outputs = [
                v for k, v in outputs.items() if k.startswith("descendant_")
            ]
            descendant_z = [z for z in inputs.node_redshifts if z < current_redshift]

            if descendant_outputs and descendant_z:
                descendant_z = descendant_z[0]
                check_redshift_consistency(
                    descendant_z,
                    descendant_outputs,
                    funcname=f"{self._func.__name__} (descendant z)",
                )

    def check_backend_state(self, inputs: InputParameters):
        """Check the backend state of the computation function.

        Currently only holds the check for whether the photon conservation is
        both needed and not initialised.
        In Future, may hold more backend state checks.
        """
        # we need photon cons to be done before any non-IC box is computed
        if (
            inputs.astro_options.PHOTON_CONS_TYPE != "no-photoncons"
            and _photoncons_state.calibration_inputs != inputs
            and issubclass(self._kls, OutputStructZ)
        ):
            raise ValueError(
                "Photon conservation is needed but not initialised with the current InputParameters."
                " Call `setup_photon_cons` with your current parameters or use the high-level functions."
            )

    def _handle_read_from_cache(
        self,
        inputs: InputParameters,
        current_redshift: float | None,
        cache: OutputCache | None,
        regen: bool = True,
    ) -> OutputStruct | None:
        """Handle potential reading from cache.

        Checks the given input parameters for cache-related keywords and manages reading
        an OutputStruct from cache if possible and desired.
        """
        if cache is None or regen:
            return None

        # First check whether the boxes already exist.
        if issubclass(self._kls, OutputStructZ):
            obj = self._kls.new(inputs=inputs, redshift=current_redshift)
        else:
            obj = self._kls.new(inputs=inputs)

        path = cache.find_existing(obj)
        if path is not None:
            with contextlib.suppress(OSError):
                this = h5.read_output_struct(path, safe=config["safe_read"])
                if hasattr(this, "redshift"):
                    logger.info(
                        f"Existing {obj._name} found at z={this.redshift} and read in (seed={this.random_seed})."
                    )
                else:
                    logger.info(
                        f"Existing {obj._name} found and read in (seed={this.random_seed})."
                    )
                return this

    def _handle_write_to_cache(
        self, cache: OutputCache | None, write, obj: OutputStruct
    ):
        """Handle writing a box to cache."""
        if write and not cache:
            raise ValueError("Cannot write to cache without a cache object.")

        if write:
            cache.write(obj)


class single_field_func(_OutputStructComputationInspect):  # noqa: N801
    """A decorator for functions that compute single fields.

    This decorator is meant for internal use only.
    """

    def __call__(self, **kwargs) -> OutputStruct:
        """Call the single field function."""
        inputs = self._get_inputs(kwargs)
        outputs = self._get_all_output_struct_inputs(kwargs)
        outputs_rec = self._get_all_output_struct_inputs(kwargs, recurse=True)
        outputsz = {k: v for k, v in outputs.items() if isinstance(v, OutputStructZ)}

        # Get current redshift (could be None)
        current_redshift = self._get_current_redshift(outputsz, kwargs)

        self.check_consistency(kwargs, outputs_rec)
        self.check_output_struct_types(outputs)
        # The following checks both current and previous redshifts, if applicable
        self.ensure_redshift_consistency(current_redshift, outputsz)
        self.check_backend_state(inputs)

        cache = kwargs.pop("cache", None)
        regen = kwargs.pop("regenerate", True)
        write = kwargs.pop("write", False)

        out = self._handle_read_from_cache(inputs, current_redshift, cache, regen)

        if "inputs" in self._signature.parameters:
            # Here we set the inputs (if accepted by the function signature)
            # to the most advanced ones. This is the explicitly-passed inputs if
            # they exist, but otherwise the inputs derived from the dependency
            # that is the most advanced in the computation.
            kwargs["inputs"] = inputs

        if out is None:
            self._broadcast_inputs(inputs)
            self._make_wisdoms(inputs.matter_options.USE_FFTW_WISDOM)
            out = self._func(**kwargs)
            self._handle_write_to_cache(cache, write, out)

        return out


class high_level_func(_OutputStructComputationInspect):  # noqa: N801
    """A decorator for high-level functions like ``run_coeval``."""

    def __init__(self, _func: callable):
        self._func = _func
        self._signature = inspect.signature(_func)
        self._kls = self._signature.return_annotation

    def __call__(self, **kwargs):
        """Call the function."""
        outputs = self._get_all_output_struct_inputs(kwargs, recurse=True)
        inputs = self._get_inputs(kwargs)
        if "inputs" in self._signature.parameters:
            # Here we set the inputs (if accepted by the function signature)
            # to the most advanced ones. This is the explicitly-passed inputs if
            # they exist, but otherwise the inputs derived from the dependency
            # that is the most advanced in the computation.
            kwargs["inputs"] = inputs

        self.check_consistency(kwargs, outputs)

        yield from self._func(**kwargs)
