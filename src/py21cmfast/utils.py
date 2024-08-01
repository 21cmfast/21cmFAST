"""Utilities for interacting with 21cmFAST data structures."""

from .wrapper.outputs import InitialConditions, _OutputStructZ


def get_all_fieldnames(
    arrays_only=True, lightcone_only=False, as_dict=False
) -> dict[str, str] | set[str]:
    """Return all possible fieldnames in output structs.

    Parameters
    ----------
    arrays_only : bool, optional
        Whether to only return fields that are arrays.
    lightcone_only : bool, optional
        Whether to only return fields from classes that evolve with redshift.
    as_dict : bool, optional
        Whether to return results as a dictionary of ``quantity: class_name``.
        Otherwise returns a set of quantities.
    """
    classes = [cls(redshift=0, dummy=True) for cls in _OutputStructZ._implementations()]

    if not lightcone_only:
        classes.append(InitialConditions())

    attr = "pointer_fields" if arrays_only else "fieldnames"

    if as_dict:
        return {
            name: cls.__class__.__name__
            for cls in classes
            for name in getattr(cls, attr)
        }
    else:
        return {name for cls in classes for name in getattr(cls, attr)}


def camel_to_snake(word: str, depublicize: bool = False):
    """Convert came case to snake case."""
    word = "".join(f"_{i.lower()}" if i.isupper() else i for i in word)

    if not depublicize:
        word = word.lstrip("_")

    return word


def snake_to_camel(word: str, publicize: bool = True):
    """Convert snake case to camel case."""
    if publicize:
        word = word.lstrip("_")
    return "".join(x.capitalize() or "_" for x in word.split("_"))


def get_all_subclasses(cls):
    """Get a list of all subclasses of a given class, recursively."""
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def float_to_string_precision(x, n):
    """Prints out a standard float number at a given number of significant digits.

    Code here: https://stackoverflow.com/a/48812729
    """
    return f'{float(f"{x:.{int(n)}g}"):g}'
